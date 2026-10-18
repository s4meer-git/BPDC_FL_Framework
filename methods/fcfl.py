from collections import OrderedDict
import copy

import numpy as np
import torch
from torch.random import fork_rng
import printer
from support_modules.precision import autocast


# Receives queue values, unfairness gaps, selected clients and weights from FCFL functions; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fcfl", event, config=config, **data)


# FCFL functions pass a model and a client's training loader here; the authors' inference_* helpers count correct labels.
@torch.no_grad()
def loader_accuracy(model, data_loader, config):
    model.eval()
    correct = 0
    total = 0


    # Keep the training shuffle and augmentation randomness identical to the other methods.
    with fork_rng(devices=[]):
        for images, labels in data_loader:
            with autocast(config):
                outputs = model(images.to(config.device))
            correct += (outputs.argmax(dim=1) == labels.to(config.device)).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("FCFL needs training samples to measure client accuracy.")
    return correct / total


# server.py passes the queue values here; this follows sample_by_Q_top() from the authors' FCFL_options.py.
def sample_by_queue_top(count, queue_values):
    if all(value == 0 for value in queue_values):
        return np.random.choice(range(len(queue_values)), count, replace=False).tolist()

    ranked = [index for index, _ in sorted(enumerate(queue_values), reverse=True, key=lambda item: item[-1])]
    non_zero = len(queue_values) - sum(1 for value in queue_values if value == 0)
    if non_zero < count:
        selected = ranked[:non_zero]

        # Keep the authors' set-difference order, because it decides which zero-queue clients the random draw returns.
        zero_ids = list(set(range(len(queue_values))) - set(selected))
        selected += np.random.choice(zero_ids, count - non_zero, replace=False).tolist()
        return selected
    return ranked[:count]


# server.py calls this with the current global model to update the unfairness queues and pick this round's clients.
def prepare_fcfl_round(global_model, clients, config, round_number):
    if not 0 < config.fcfl_fraction <= 1:
        raise ValueError("fcfl_fraction must be in (0, 1].")
    count = max(int(config.fcfl_fraction * len(clients)), 1)
    sizes = {client.client_id: len(client.train_loader.dataset) for client in clients}

    if round_number == 1:
        config.fcfl_queue = {client.client_id: 0.0 for client in clients}
        selected_ids = np.random.choice([client.client_id for client in clients], count, replace=False).tolist()
        weights = data_volume_weights(selected_ids, sizes)
        gaps = {}
    else:


        # uf_i = max(Acc_global - acc_i, 0), where acc_i is the global model's accuracy on client i's own data.
        gaps = {}
        for client in clients:
            client_accuracy = loader_accuracy(global_model, client.train_loader, config)
            gaps[client.client_id] = max(config.fcfl_global_accuracy - client_accuracy, 0.0)


        # Q_i(t) = max(Q_i(t-1) + alpha * uf_i - p_i(t-1), 0); clients that were not aggregated last round have p_i = 0.
        for client in clients:
            client_id = client.client_id
            previous_weight = config.fcfl_weights.get(client_id, 0.0)
            config.fcfl_queue[client_id] = max(
                config.fcfl_queue[client_id] + config.fcfl_alpha * gaps[client_id] - previous_weight, 0.0,
            )

        client_ids = [client.client_id for client in clients]
        queue_values = [config.fcfl_queue[client_id] for client_id in client_ids]
        selected_ids = [client_ids[index] for index in sample_by_queue_top(count, queue_values)]
        queue_total = sum(config.fcfl_queue[client_id] for client_id in selected_ids)

        if queue_total == 0:
            weights = data_volume_weights(selected_ids, sizes)
        else:
            weights = {client_id: config.fcfl_queue[client_id] / queue_total for client_id in selected_ids}

    config.fcfl_weights = weights
    selected_clients = [client for client in clients if client.client_id in weights]


    # Queue values, unfairness gaps, selected clients and their aggregation weights; info passing to printer.py.
    report_progress(
        config,
        "clients_selected",
        round_number=round_number,
        client_ids=[client.client_id for client in selected_clients],
        queue=config.fcfl_queue,
        unfairness=gaps,
        weights=weights,
        estimated_global_accuracy=config.fcfl_global_accuracy,
        client_count=count,
    )

    return selected_clients


# The first round and all-zero queues use data-volume weights for the clients that were actually selected.
def data_volume_weights(selected_ids, sizes):
    total = sum(sizes[client_id] for client_id in selected_ids)
    return {client_id: sizes[client_id] / total for client_id in selected_ids}


# config.train_client() sends a client here; fcfl_train() also measures the trained model on the client's training data.
def train_fcfl_client(global_model, client, config):
    weights, sample_count = client.train(global_model, config)
    local_model = copy.deepcopy(global_model).to(config.device)
    local_model.load_state_dict(weights)
    config.fcfl_train_accuracies[client.client_id] = loader_accuracy(local_model, client.train_loader, config)


    # Client ID and its trained model's accuracy on its own training data; info passing to printer.py.
    report_progress(
        config,
        "fcfl_local_accuracy",
        client_id=client.client_id,
        train_accuracy=config.fcfl_train_accuracies[client.client_id],
    )
    return weights, sample_count


# config.aggregate() passes trained models here; weight_agg() sums them with this round's queue-based weights.
def fcfl(client_weights, client_sizes, global_model, clients, config):
    if not client_weights or len(client_weights) != len(clients):
        raise ValueError("FCFL requires one model for each selected client.")
    weights = [config.fcfl_weights[client.client_id] for client in clients]

    aggregated = OrderedDict()
    for name, value in global_model.state_dict().items():


        # Copy integer buffers such as BatchNorm counters instead of treating them as floating point weights.
        if not torch.is_floating_point(value):
            aggregated[name] = client_weights[0][name].clone()
            continue
        total = torch.zeros_like(value, device="cpu", dtype=torch.float64)
        for client_state, weight in zip(client_weights, weights):
            total += client_state[name].double() * weight
        aggregated[name] = total.to(value.dtype)


    # acc_global_estimate(): the aggregation-weighted local training accuracy is next round's Acc_global.
    config.fcfl_global_accuracy = sum(
        config.fcfl_train_accuracies.pop(client.client_id) * weight
        for client, weight in zip(clients, weights)
    )


    # Aggregation weights, client IDs and the estimated global accuracy for the next queue update; info passing to printer.py.
    report_progress(
        config,
        "fcfl_aggregated",
        client_ids=[client.client_id for client in clients],
        weights=weights,
        estimated_global_accuracy=config.fcfl_global_accuracy,
    )

    return aggregated
