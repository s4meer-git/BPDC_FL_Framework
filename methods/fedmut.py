import copy
import random

import numpy as np
import torch
import printer


# Receives selected clients, mutation slots and population settings from FedMut functions; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fedmut", event, config=config, **data)


# server.py calls this with clients and round settings to assign sampled clients to mutation slots.
def prepare_fedmut_round(clients, config, round_number):
    if not 0 < config.fedmut_fraction <= 1:
        raise ValueError("fedmut_fraction must be in (0, 1].")
    if config.fedmut_bound <= 0:
        raise ValueError("fedmut_bound must be positive.")
    if not clients:
        raise ValueError("FedMut requires at least one client.")
    if round_number == 1:
        config.fedmut_population = []

    count = max(int(config.fedmut_fraction * len(clients)), 1)
    if len(config.fedmut_population) not in [0, count]:
        raise ValueError("FedMut participation count must stay fixed during a run.")
    indices = np.random.choice(len(clients), count, replace=False)
    selected_clients = []
    config.fedmut_client_slots = {}
    for slot, index in enumerate(indices):
        client = clients[index]
        selected_clients.append(client)
        config.fedmut_client_slots[client.client_id] = slot


    # Selected client IDs, their assigned mutation slots and participation count; info passing to printer.py.
    report_progress(
        config,
        "clients_selected",
        round_number=round_number,
        client_ids=[client.client_id for client in selected_clients],
        client_slots=config.fedmut_client_slots,
        client_count=count,
    )

    return selected_clients


# Start this client from its mutation slot, using the model and settings passed by config.train_client().
def train_fedmut_client(global_model, client, config):
    local_start = copy.deepcopy(global_model)
    slot = config.fedmut_client_slots[client.client_id]
    if config.fedmut_population:
        local_start.load_state_dict(config.fedmut_population[slot])


    # Client ID, assigned population slot and whether it starts from a mutated model; info passing to printer.py.
    report_progress(
        config,
        "fedmut_client_start",
        client_id=client.client_id,
        slot=slot,
        uses_population=bool(config.fedmut_population),
    )

    return client.train(local_start, config)


# fedmut() passes averaged weights and their change here to build the mutation population for the next round.
def mutation_spread(global_weights, delta, count, round_index, config):
    control_rate = config.fedmut_acc_rate * (
        1.0 - min(round_index / config.fedmut_bound, 1.0)
    )
    controls = {}
    for name in global_weights:
        values = []
        for _ in range(count // 2):
            if random.random() > 0.5:
                values.append(1.0)
                values.append(-1.0 + control_rate)
            else:
                values.append(-1.0 + control_rate)
                values.append(1.0)
        random.shuffle(values)
        controls[name] = values

    population = []
    for slot in range(count):
        weights = copy.deepcopy(global_weights)


        # The last slot stays at the global model when the population is odd.
        if not (slot == count - 1 and count % 2 == 1):
            for name in weights:
                weights[name] = weights[name] + (
                    delta[name] * controls[name][slot] * config.fedmut_radius
                )
        population.append(weights)


    # Round index, mutation controls, radius, acceleration rate and population size; info passing to printer.py.
    report_progress(
        config,
        "fedmut_mutation",
        round_index=round_index,
        population_size=count,
        radius=config.fedmut_radius,
        control_rate=control_rate,
        controls=controls,
        unmutated_slot=count - 1 if count % 2 else None,
    )

    return population


# Average the trained weights supplied by config.aggregate(), then create models for the next round.
def fedmut(client_weights, client_sizes, global_model, clients, config):
    if not client_weights:
        raise ValueError("FedMut requires at least one client model.")
    global_weights = {}
    delta = {}
    old_weights = global_model.state_dict()


    # The author code averages all state entries equally, including buffers.
    for name in old_weights:
        average = client_weights[0][name].detach().cpu().clone() * 1.0
        for index in range(1, len(client_weights)):
            average = average + client_weights[index][name].detach().cpu() * 1.0
        average = torch.div(average, len(client_weights))
        global_weights[name] = average
        delta[name] = average - old_weights[name].detach().cpu()

    config.fedmut_population = mutation_spread(
        global_weights,
        delta,
        len(client_weights),
        config.current_round - 1,
        config,
    )


    # Aggregated client count, sample counts and the next-round population size; info passing to printer.py.
    report_progress(
        config,
        "fedmut_aggregated",
        client_count=len(client_weights),
        client_sizes=client_sizes,
        population_size=len(config.fedmut_population),
    )

    return global_weights
