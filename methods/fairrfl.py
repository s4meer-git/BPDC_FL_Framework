import copy
import torch
import numpy as np
import printer
from support_modules.batchnorm import merge_batchnorm_buffers


# Receives local losses, fairness weights and recovery statistics from the FairRFL functions; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fairrfl", event, config=config, **data)


# config.train_client() passes the model and client here to train locally and sum losses as in the authors' utils.py.
def train_fairrfl_client(global_model, client, config):


    # This client ID and the number of training samples in its loader; info passing to printer.py.
    report_progress(
        config,
        "client_started",
        client_id=client.client_id,
        training_samples=len(client.train_loader.dataset),
    )

    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(config.device)
    local_model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    client_loss = 0.0

    for epoch in range(config.local_epochs):
        no_of_batches = 0
        loss_before_epoch = client_loss


        # The client ID and local epoch that is about to start; info passing to printer.py.
        report_progress(
            config,
            "epoch_started",
            client_id=client.client_id,
            epoch=epoch + 1,
        )

        for images, labels in client.train_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)


            # The authors create the optimizer again for each batch, so its momentum state starts fresh each time.
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=config.lr,
                momentum=0.9,
                weight_decay=0.0005
            )
            optimizer.zero_grad()
            predictions = local_model.forward(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            batch_loss = loss.item()
            client_loss += batch_loss
            optimizer.step()
            no_of_batches += 1


            # Client ID, epoch, batch loss, sample count and the accumulated FairRFL loss; info passing to printer.py.
            report_progress(
                config,
                "batch_completed",
                client_id=client.client_id,
                epoch=epoch + 1,
                batch=no_of_batches,
                loss=batch_loss,
                client_loss=client_loss,
                total_loss=client_loss - loss_before_epoch,
                batch_samples=labels.size(0),
            )

        epoch_loss = client_loss - loss_before_epoch
        average_loss = epoch_loss / no_of_batches if no_of_batches else 0.0


        # Client ID, completed epoch, batch count and the average training loss; info passing to printer.py.
        report_progress(
            config,
            "epoch_completed",
            client_id=client.client_id,
            epoch=epoch + 1,
            batches=no_of_batches,
            average_loss=average_loss,
            client_loss=client_loss,
        )

    updated_weights = {}
    for name, value in local_model.state_dict().items():
        updated_weights[name] = value.detach().cpu().clone()

    config.fairrfl_client_losses[client.client_id] = client_loss
    no_of_samples = len(client.train_loader.dataset)


    # Client ID, summed local training loss and its training sample count; info passing to printer.py.
    report_progress(
        config,
        "client_completed",
        client_id=client.client_id,
        client_loss=client_loss,
        training_samples=no_of_samples,
    )

    return updated_weights, no_of_samples


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

    for i, weights in enumerate(client_weights):
        client_id = clients[i].client_id
        client_parts = []
        for parameter_name in parameter_names:
            client_parameter = weights[parameter_name].detach().cpu()
            client_parts.append(client_parameter.numpy().reshape(-1))

        client_vector = np.concatenate(client_parts)

        if config.fairrfl_q != 0:
            client_loss = config.fairrfl_client_losses[client_id]

            client_q = config.fairrfl_q * (previous_losses[i] / median_previous_loss)
            estimated_gradient = (client_vector - global_vector) / config.lr
            loss_weight = np.float_power(client_loss + 0.0000000001, client_q)
            weighted_update = loss_weight * estimated_gradient
            client_normalizer = (
                client_q
                * np.float_power(client_loss + 0.0000000001, client_q - 1)
                * np.linalg.norm(estimated_gradient) ** 2
                + (1 / config.lr) * loss_weight
            )
            client_normalizers.append(client_normalizer)


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

        client_vectors.append(client_vector)


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
    )

    return new_weights
