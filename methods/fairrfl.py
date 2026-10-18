import copy
import torch
from torch.random import fork_rng
import numpy as np
import printer
from support_modules.batchnorm import merge_batchnorm_buffers
from support_modules.selfishness import attack_vectors
from support_modules.precision import autocast


# Receives local losses, fairness weights and recovery statistics from the FairRFL functions; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fairrfl", event, config=config, **data)


# config.train_client() passes the model and client here; Dq-FFL needs the received model's loss before local training.
def train_fairrfl_client(global_model, client, config):
    global_loss = global_model_loss(global_model, client.train_loader, config)
    updated_weights, no_of_samples = client.train(global_model, config)

    config.fairrfl_client_losses[client.client_id] = global_loss
    config.fairrfl_local_steps[client.client_id] = config.local_epochs * len(client.train_loader)


    # Client ID, received-model training loss and the local SGD steps used for L; info passing to printer.py.
    report_progress(
        config,
        "fairrfl_client_loss",
        client_id=client.client_id,
        client_loss=global_loss,
        local_steps=config.fairrfl_local_steps[client.client_id],
        training_samples=no_of_samples,
    )

    return updated_weights, no_of_samples


# train_fairrfl_client() passes the received global model here; this is F_i(w[k]) in Algorithm 3 of the paper.
@torch.no_grad()
def global_model_loss(global_model, train_loader, config):
    model = copy.deepcopy(global_model).to(config.device)
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0


    # Keep the training shuffle and augmentation randomness identical to the other methods.
    with fork_rng(devices=[]):
        for images, labels in train_loader:
            with autocast(config):
                predictions = model(images.to(config.device))
                total_loss += loss_function(predictions, labels.to(config.device)).item()
            total_samples += labels.size(0)

    if total_samples == 0 or not np.isfinite(total_loss):
        raise ValueError("FairRFL got a non-finite global-model loss on a client's training data.")
    return total_loss / total_samples


# fairrfl() sends the flagged update and median statistics here for the authors' ten-step recovery search.
def recover_update(client_update, median_update, median_norm, config=None, client_id=None):
    lower_bound = 0.0
    upper_bound = 1.0

    for step in range(10):
        beta = (lower_bound + upper_bound) / 2
        recovered_update = beta * client_update + (1 - beta) * median_update

        recovered_norm = np.linalg.norm(recovered_update)
        if config is not None:


            # Client ID, recovery step, beta, update norms and the current search bounds; info passing to printer.py.
            report_progress(
                config,
                "fairrfl_recovery",
                client_id=client_id,
                step=step + 1,
                beta=beta,
                recovered_norm=recovered_norm,
                target_norm=median_norm,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        if recovered_norm > median_norm:
            upper_bound = beta
        else:
            lower_bound = beta

    return recovered_update


# Receives local weights, client IDs and stored losses through config.aggregate() to weight, detect and recover updates.
def fairrfl(client_weights, client_sizes, global_model, clients, config):
    no_of_clients = len(client_weights)


    # Flatten model parameters for the update calculations; trained BatchNorm buffers are combined separately.
    global_weights = global_model.state_dict()
    parameter_names = []
    global_parts = []

    for parameter_name, parameter in global_model.named_parameters():
        parameter_names.append(parameter_name)
        global_parameter = parameter.detach().cpu().numpy()
        global_parts.append(global_parameter.reshape(-1))

    global_vector = np.concatenate(global_parts)


    # Use each client's previous-round loss to calculate its fairness weight, following experiments.py.
    previous_losses = []
    for client in clients:
        previous_loss = config.fairrfl_previous_losses.get(client.client_id, 1.0)
        previous_losses.append(previous_loss)

    median_previous_loss = np.median(previous_losses)
    if median_previous_loss == 0:
        positive_losses = [loss for loss in previous_losses if loss > 0]
        median_previous_loss = np.median(positive_losses) if positive_losses else 1.0
    client_vectors = []
    client_normalizers = []
    local_update_norms = []


    # q-FFL treats L(w - w_local) as one gradient; with several momentum SGD steps, L = (1 - momentum) / (lr * steps).
    lipschitz = config.fairrfl_lipschitz
    if lipschitz is None:
        steps = [config.fairrfl_local_steps[client.client_id] for client in clients]
        lipschitz = (1 - config.momentum) / (config.lr * np.mean(steps))

    for i, weights in enumerate(client_weights):
        client_id = clients[i].client_id
        client_parts = []
        for parameter_name in parameter_names:
            client_parameter = weights[parameter_name].detach().cpu()
            client_parts.append(client_parameter.numpy().reshape(-1))

        client_vector = np.concatenate(client_parts)
        if not np.isfinite(client_vector).all():
            raise ValueError(f"Non-finite FairRFL local model for client {client_id}.")
        local_update_norms.append(float(np.linalg.norm(client_vector - global_vector)))

        if config.fairrfl_q != 0:
            client_loss = config.fairrfl_client_losses[client_id]

            client_q = config.fairrfl_q
            if config.fairrfl_dynamic_q:
                client_q *= previous_losses[i] / median_previous_loss
            estimated_gradient = (client_vector - global_vector) * lipschitz
            loss_weight = np.float_power(client_loss + 0.0000000001, client_q)
            weighted_update = loss_weight * estimated_gradient
            client_normalizer = (
                client_q
                * np.float_power(client_loss + 0.0000000001, client_q - 1)
                * np.linalg.norm(estimated_gradient) ** 2
                + lipschitz * loss_weight
            )
            client_normalizers.append(client_normalizer)
            if not np.isfinite(client_normalizer) or client_normalizer <= 0 or not np.isfinite(weighted_update).all():
                raise ValueError(f"Non-finite or invalid FairRFL fairness weighting for client {client_id}; inspect losses and q.")


            # Client ID, previous/current losses, dynamic q, loss weight and normalizer; info passing to printer.py.
            report_progress(
                config,
                "fairrfl_weighting",
                client_id=client_id,
                client_loss=client_loss,
                previous_loss=previous_losses[i],
                dynamic_q=client_q,
                loss_weight=loss_weight,
                normalizer=client_normalizer,
            )

            weighted_vector = global_vector + weighted_update
            weighted_parts = []
            offset = 0
            for parameter_name in parameter_names:
                parameter = global_weights[parameter_name]
                size = parameter.numel()
                part = weighted_vector[offset:offset + size]
                part = part.astype(parameter.detach().cpu().numpy().dtype)
                weighted_parts.append(part)
                offset += size
            client_vector = np.concatenate(weighted_parts)

        if not np.isfinite(client_vector).all():
            raise ValueError(f"Non-finite FairRFL weighted model (overflow) for client {client_id}.")
        client_vectors.append(client_vector)

    if config.selfishness:
        client_vectors = attack_vectors(client_vectors, global_vector, clients, config)


    # Compare update norms with the median and recover outliers using the rule from selfishness.py.
    updates = []
    norm_values = []
    for client_vector in client_vectors:
        client_update = client_vector - global_vector
        updates.append(client_update)
        norm_values.append(np.linalg.norm(client_update))

    median_norm = np.percentile(norm_values, 50)
    median_update = np.median(updates, axis=0)
    median_deviation = np.median(np.abs(norm_values - median_norm), axis=0)
    scaled_deviation = 1.4826 * median_deviation
    detected_clients = []

    for i in range(no_of_clients):
        if scaled_deviation == 0:


            # With zero deviation, only norms above the median are outliers; this avoids dividing by zero.
            is_outlier = norm_values[i] > median_norm
        else:
            normalized_deviation = (norm_values[i] - median_norm) / scaled_deviation
            is_outlier = normalized_deviation > config.fairrfl_tau
        if is_outlier:
            detected_clients.append(clients[i].client_id)
            recovered_update = recover_update(
                updates[i],
                median_update,
                median_norm,
                config=config,
                client_id=clients[i].client_id,
            )
            client_vectors[i] = global_vector + recovered_update

    new_global_vector = np.mean(client_vectors, axis=0)


    # Apply the authors' fairness normalization after recovering any detected client updates.
    if config.fairrfl_q != 0:
        new_global_vector = (
            no_of_clients * (new_global_vector - global_vector)
            / np.sum(client_normalizers) + global_vector
        )

    new_weights = {}
    for name, value in global_weights.items():
        new_weights[name] = value.detach().cpu().clone()

    offset = 0
    for parameter_name in parameter_names:
        parameter = global_weights[parameter_name]
        size = parameter.numel()
        part = new_global_vector[offset:offset + size].copy()
        new_parameter = torch.from_numpy(part).reshape(parameter.shape)
        new_weights[parameter_name] = new_parameter.to(parameter.dtype)
        offset += size

    merge_batchnorm_buffers(new_weights, global_model, client_weights, client_sizes)

    config.fairrfl_previous_losses = config.fairrfl_client_losses.copy()
    config.fairrfl_client_losses = {}
    config.fairrfl_detected_clients = detected_clients


    # Detected client IDs, update norms, median deviation, losses and normalizers; info passing to printer.py.
    report_progress(
        config,
        "fairrfl_detection",
        detected_clients=detected_clients,
        client_ids=[client.client_id for client in clients],
        update_norms=norm_values,
        median_norm=median_norm,
        scaled_deviation=scaled_deviation,
        previous_losses=previous_losses,
        client_losses=config.fairrfl_previous_losses,
        normalizers=client_normalizers,
        lipschitz=lipschitz,
        server_update_norm=float(np.linalg.norm(new_global_vector - global_vector)),
        median_local_update_norm=float(np.median(local_update_norms)),
        small_server_update=(
            config.current_round >= 2
            and np.linalg.norm(new_global_vector - global_vector) < 0.01 * np.median(local_update_norms)
        ),
    )

    return new_weights
