import printer
import torch
from support_modules.selfishness import attack_models


# Receives round numbers, participating clients and sample counts from run_one_round(); info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("participants.server", event, config=config, **data)


# main.py passes the model, clients and round settings here to train clients and aggregate their updates.
def run_one_round(global_model, clients, config, round_number):
    method = config.method.lower()

    if not clients:
        raise ValueError("A communication round requires at least one client.")
    if method not in {"fedavg", "fairrfl", "fedcda", "fedmut", "fcfl"}:
        raise ValueError("Unknown method name: " + config.method)
    config.current_round = round_number


    # The current round, selected method and number of available clients; info passing to printer.py.
    report_progress(
        config,
        "round_started",
        round_number=round_number,
        method=config.method,
        client_count=len(clients),
    )

    if method == "fedcda":
        from methods.fedcda import prepare_fedcda_round

        clients = prepare_fedcda_round(clients, config, round_number)
    elif method == "fedmut":
        from methods.fedmut import prepare_fedmut_round

        clients = prepare_fedmut_round(clients, config, round_number)
    elif method == "fcfl":
        from methods.fcfl import prepare_fcfl_round

        clients = prepare_fcfl_round(global_model, clients, config, round_number)

    all_client_weights = []
    all_client_sizes = []

    for client in clients:
        updated_weights, no_of_samples = config.train_client(global_model, client)
        all_client_weights.append(updated_weights)
        all_client_sizes.append(no_of_samples)


    # Participating client IDs and the sample counts returned by their training; info passing to printer.py.
    report_progress(
        config,
        "aggregation_started",
        client_count=len(clients),
        client_ids=[client.client_id for client in clients],
        client_sizes=all_client_sizes,
    )



    # Each attacker scales its update by its real share of this method's aggregate (FedCDA and FedMut average equally).
    if config.selfishness and method != "fairrfl":
        shares = None
        if method == "fedavg":
            total = sum(all_client_sizes)
            shares = {client.client_id: size / total for client, size in zip(clients, all_client_sizes)}
        elif method == "fcfl":
            shares = {client.client_id: config.fcfl_weights[client.client_id] for client in clients}
        all_client_weights = attack_models(all_client_weights, global_model, clients, config, shares)

    new_global_weights = config.aggregate(
        all_client_weights,
        all_client_sizes,
        global_model,
        clients
    )

    for name, value in new_global_weights.items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{method}, round {round_number}: non-finite model state in {name}.")

    global_model.load_state_dict(new_global_weights)


    # The current round and number of clients whose updates were aggregated; info passing to printer.py.
    report_progress(
        config,
        "round_completed",
        client_count=len(clients),
    )

    return global_model
