import numpy as np
import torch
import printer


# The server and FairRFL send attack settings, client IDs and update norms here; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("support_modules.selfishness", event, config=config, **data)


# main.py supplies the validated configuration; a private generator keeps training randomness unchanged.
def choose_selfish_clients(config):
    if not config.selfishness:
        return []
    count = max(1, int(config.num_of_clients * config.selfish_fraction))
    generator = np.random.default_rng(config.seed)
    return sorted(generator.choice(config.num_of_clients, count, replace=False).tolist())


# Eq. (5) of the paper with the attacker's share w_s / Gamma of the aggregate; a share of 1/n is the authors' known-count branch.
def selfish_update(local_update, global_change, previous_update, previous_share, share, strength):


    # The other clients' contribution is assumed unchanged since the last round, as in the paper.
    others = global_change - previous_share * previous_update
    target = strength * local_update + (1 - strength) * others / (1 - share)
    return (target - others) / share


# Aggregation passes outgoing parameter vectors here; FairRFL calls this after fairness weighting and before recovery.
# shares maps client IDs to their weight in the aggregate; None means the equal 1/n weights of an unweighted mean.
def attack_vectors(vectors, global_vector, clients, config, shares=None):
    if not config.selfishness:
        return vectors
    if len(clients) != config.num_of_clients:
        raise ValueError("The selfish-client experiment requires full client participation.")

    previous_global = config.selfish_previous_global
    active = config.current_round >= config.selfish_start_round
    if active and previous_global is None:
        raise ValueError("Selfish attack history is missing; start the run from round 1.")
    applied = []
    if shares is None:
        shares = {client.client_id: 1 / config.num_of_clients for client in clients}

    for index, client in enumerate(clients):
        client_id = client.client_id
        if client_id not in config.selfish_client_ids:
            continue
        local_update = vectors[index] - global_vector
        sent_update = local_update

        share = shares[client_id]


        # A client with no weight this round (for example an FCFL queue of zero) cannot steer the aggregate.
        if active and 0 < share < 1:
            previous_update = config.selfish_previous_updates.get(client_id)
            if previous_update is None:
                raise ValueError(f"Previous transmitted update is missing for selfish client {client_id}.")
            sent_update = selfish_update(
                local_update, global_vector - previous_global, previous_update,
                config.selfish_previous_shares[client_id], share, config.selfish_strength,
            )
            if not np.isfinite(sent_update).all():
                raise ValueError(f"Non-finite selfish update in round {config.current_round}, client {client_id}.")
            vectors[index] = global_vector + sent_update
            if not np.isfinite(vectors[index]).all():
                raise ValueError(f"Non-finite transmitted model for selfish client {client_id}.")
            applied.append(client_id)


            # Client ID, attack strength and honest/modified update magnitudes; info passing to printer.py.
            report_progress(
                config, "selfish_update", client_id=client_id,
                strength=config.selfish_strength,
                share=share,
                honest_update_norm=float(np.linalg.norm(local_update.astype(np.float64))),
                sent_update_norm=float(np.linalg.norm(sent_update.astype(np.float64))),
            )


        # Keep the actual outgoing update before the server applies its recovery or aggregation rule.
        config.selfish_previous_updates[client_id] = sent_update.copy()
        config.selfish_previous_shares[client_id] = share

    config.selfish_previous_global = global_vector.copy()


    # Configured and active selfish clients for this round, including the initial honest rounds; info passing to printer.py.
    report_progress(
        config, "selfish_round", selfish_client_ids=config.selfish_client_ids,
        attacked_client_ids=sorted(applied), active=active,
    )
    return vectors


# The server passes trained model states and aggregation shares here; only trainable parameters are attacked.
def attack_models(client_weights, global_model, clients, config, shares=None):
    if not config.selfishness:
        return client_weights
    names = [name for name, _ in global_model.named_parameters()]
    global_vector = np.concatenate([
        parameter.detach().cpu().numpy().reshape(-1) for parameter in global_model.parameters()
    ])
    vectors = [np.concatenate([weights[name].detach().cpu().numpy().reshape(-1) for name in names]) for weights in client_weights]
    vectors = attack_vectors(vectors, global_vector, clients, config, shares)

    for client, weights, vector in zip(clients, client_weights, vectors):
        if client.client_id not in config.selfish_client_ids or config.current_round < config.selfish_start_round:
            continue
        offset = 0
        for name in names:
            original = weights[name]
            size = original.numel()
            weights[name] = torch.from_numpy(vector[offset:offset + size].copy()).reshape(original.shape).to(device=original.device, dtype=original.dtype)
            offset += size
    return client_weights
