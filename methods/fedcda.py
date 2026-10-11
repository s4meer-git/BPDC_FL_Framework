import copy
import random
import math

import torch
import printer


# Receives selected clients, history choices and search scores from FedCDA functions; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fedcda", event, config=config, **data)


# server.py supplies the clients and round settings here to choose participants and prepare their model histories.
def prepare_fedcda_round(clients, config, round_number):
    if config.fedcda_memory_k < 1 or config.fedcda_batch_client_num < 1:
        raise ValueError("FedCDA memory and search batch sizes must be positive.")
    if config.fedcda_warmup < 0:
        raise ValueError("fedcda_warmup must be at least zero.")
    if not math.isfinite(config.fedcda_smoothness) or config.fedcda_smoothness <= 0:
        raise ValueError("fedcda_smoothness must be finite and positive.")
    if round_number == 1:
        config.fedcda_history = {}
        config.fedcda_loss_history = {}
        config.fedcda_client_losses = {}
        config.fedcda_choices = {}
        for client in clients:
            config.fedcda_history[client.client_id] = []
            config.fedcda_loss_history[client.client_id] = []
            config.fedcda_choices[client.client_id] = 0

    count = config.fedcda_clients_per_round
    if count is None:
        count = len(clients)
    if count < 1 or count > len(clients):
        raise ValueError("fedcda_clients_per_round must be between 1 and the client count.")
    indices = random.sample(range(len(clients)), count)
    config.fedcda_search_order = [clients[index].client_id for index in indices]


    # Train in the original client order; the sampled order is used only for the history search.
    selected_clients = [
        client for client in clients
        if client.client_id in config.fedcda_search_order
    ]


    # Selected client IDs, search order and the client count for this round; info passing to printer.py.
    report_progress(
        config,
        "clients_selected",
        round_number=round_number,
        client_ids=[client.client_id for client in selected_clients],
        search_order=config.fedcda_search_order,
        client_count=count,
    )

    return selected_clients


# config.train_client() sends a client here to train and measure the final local model's training loss for history search.
def train_fedcda_client(global_model, client, config):
    weights, sample_count = client.train(global_model, config)
    local_model = copy.deepcopy(global_model).to(config.device)
    local_model.load_state_dict(weights)
    local_model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for images, labels in client.train_loader:
            predictions = local_model(images.to(config.device))
            loss = loss_function(predictions, labels.to(config.device))
            if not torch.isfinite(loss).item():
                raise ValueError(f"FedCDA client {client.client_id}: non-finite final local loss.")
            total_loss += loss.item()
            total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("FedCDA requires training samples to measure local loss.")

    mean_loss = total_loss / total_samples
    config.fedcda_client_losses[client.client_id] = mean_loss


    # Client ID, final local model loss and evaluated training sample count; info passing to printer.py.
    report_progress(
        config,
        "fedcda_local_loss",
        client_id=client.client_id,
        loss=mean_loss,
        samples=total_samples,
    )
    return weights, sample_count


# The history search supplies selected models and their measured losses to evaluate the paper's Equation (5).
def history_score(history, losses, choices, parameter_names, smoothness):
    selected = [models[choices[client_id]] for client_id, models in history.items()]
    count = len(selected)
    divergence = 0.0

    for name in parameter_names:
        values = [weights[name].detach().cpu().double() for weights in selected]
        center = sum(values) / count
        for value in values:
            divergence += (value - center).square().sum().item()

    mean_loss = sum(losses[client_id][choices[client_id]] for client_id in history) / count
    score = mean_loss + smoothness * divergence / (2 * count)
    if not math.isfinite(score):
        raise ValueError("FedCDA history search produced a non-finite score.")
    return score


# fedcda() supplies one client group here; each recursive branch keeps the best score found so far.
def search_history(
    history, losses, choices, batch, depth, path, best_score,
    parameter_names, smoothness, config=None,
):
    if depth == len(batch):
        score = history_score(history, losses, path, parameter_names, smoothness)
        if config is not None:


            # Candidate histories, full objective value and whether this candidate improves the score; info passing to printer.py.
            report_progress(
                config,
                "fedcda_candidate",
                choices=path,
                score=score,
                improved=bool(score < best_score),
                batch_client_ids=batch,
            )

        if score < best_score:
            choices.update(path)
            return score
        return best_score

    client_id = batch[depth]
    for index in range(len(history[client_id])):
        path[client_id] = index
        best_score = search_history(
            history, losses, choices, batch, depth + 1, path, best_score,
            parameter_names, smoothness, config,
        )
    return best_score


# config.aggregate() passes local models here to cache them with their losses, select histories and average the chosen models.
def fedcda(client_weights, client_sizes, global_model, clients, config):
    if not client_weights or len(client_weights) != len(clients):
        raise ValueError("FedCDA requires one model for each participating client.")

    history = config.fedcda_history
    losses = config.fedcda_loss_history
    choices = config.fedcda_choices
    selected_ids = [client.client_id for client in clients]
    parameter_names = [name for name, _ in global_model.named_parameters()]

    for client, weights in zip(clients, client_weights):
        client_id = client.client_id
        history.setdefault(client_id, [])
        losses.setdefault(client_id, [])
        if len(history[client_id]) >= config.fedcda_memory_k:
            del history[client_id][0]
            del losses[client_id][0]
        history[client_id].append({
            name: value.detach().cpu().clone()
            for name, value in weights.items()
        })
        losses[client_id].append(config.fedcda_client_losses.pop(client_id))
        choices[client_id] = len(history[client_id]) - 1

    order = config.fedcda_search_order or selected_ids
    warmup = config.current_round - 1 < config.fedcda_warmup

    if warmup:
        for client_id, models in history.items():
            if models:
                choices[client_id] = len(models) - 1
    else:
        included_ids = [
            client_id for client_id, models in history.items()
            if models and client_id not in selected_ids
        ]
        for start in range(0, len(order), config.fedcda_batch_client_num):
            batch = order[start:start + config.fedcda_batch_client_num]
            included_ids.extend(batch)


            # Equation (8) includes inactive clients and groups already visited; later groups enter when their turn arrives.
            search_models = {client_id: history[client_id] for client_id in included_ids}
            search_history(
                search_models, losses, choices, batch, 0, choices.copy(), float("inf"),
                parameter_names, config.fedcda_smoothness, config,
            )

    selected_models = [models[choices[client_id]] for client_id, models in history.items() if models]
    new_weights = {}
    for name, value in global_model.state_dict().items():
        average = torch.zeros_like(value, device="cpu", dtype=torch.float64)
        for weights in selected_models:
            average += weights[name].double() / len(selected_models)
        new_weights[name] = average.to(value.dtype)


    # Selected histories, cache lengths, warmup status and the objective coefficient; info passing to printer.py.
    report_progress(
        config,
        "fedcda_aggregated",
        client_ids=selected_ids,
        cached_client_count=len(selected_models),
        history_lengths={client_id: len(models) for client_id, models in history.items()},
        choices=choices,
        warmup=warmup,
        smoothness=config.fedcda_smoothness,
    )
    return new_weights
