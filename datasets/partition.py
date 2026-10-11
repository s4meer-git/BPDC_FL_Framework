import numpy as np


# main.py supplies the labels, client count, alpha and seed used to divide sample indices among clients.
def dirichlet_partition(labels, no_of_clients, alpha, seed):
    if no_of_clients < 1:
        raise ValueError("The number of clients must be positive.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet alpha must be finite and positive.")

    rng = np.random.RandomState(seed)
    client_samples = [[] for _ in range(no_of_clients)]
    labels = np.array(labels)

    for class_label in np.unique(labels):


        # Shuffle this class's sample indices before dividing them among clients.
        item_pointers = np.flatnonzero(labels == class_label).tolist()
        rng.shuffle(item_pointers)
        dirichlet_values = rng.dirichlet([alpha] * no_of_clients)
        no_of_samples = [int(value * len(item_pointers)) for value in dirichlet_values]


        # Give rounding leftovers to the largest fractional remainders so the last client does not get them all.
        remainders = dirichlet_values * len(item_pointers) - no_of_samples
        leftover_count = len(item_pointers) - sum(no_of_samples)

        for client in np.argsort(-remainders, kind="stable")[:leftover_count]:
            no_of_samples[client] += 1

        current_index = 0

        for client, samples_to_give in enumerate(no_of_samples):
            end_index = current_index + samples_to_give
            client_samples[client].extend(item_pointers[current_index:end_index])
            current_index = end_index


    # Mix the classes within each client's final list before creating its dataset.
    for samples in client_samples:
        rng.shuffle(samples)

    return client_samples
