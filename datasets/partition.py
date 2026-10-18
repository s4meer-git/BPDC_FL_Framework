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


# main.py passes the test labels and the training partition here so each client's test set follows its own class mix.
def matching_partition(labels, train_labels, train_partitions, seed):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    train_labels = np.array(train_labels)
    client_samples = [[] for _ in train_partitions]

    for class_label in np.unique(labels):
        item_pointers = np.flatnonzero(labels == class_label).tolist()
        rng.shuffle(item_pointers)


        # Client k's share of this class in the test set equals its share of the class in the training set.
        train_counts = np.array([
            np.count_nonzero(train_labels[indices] == class_label) if indices else 0
            for indices in train_partitions
        ], dtype=float)
        if train_counts.sum() == 0:
            continue
        shares = train_counts / train_counts.sum()
        no_of_samples = [int(value * len(item_pointers)) for value in shares]


        # Give rounding leftovers to the largest fractional remainders, as dirichlet_partition() does.
        remainders = shares * len(item_pointers) - no_of_samples
        leftover_count = len(item_pointers) - sum(no_of_samples)

        for client in np.argsort(-remainders, kind="stable")[:leftover_count]:
            no_of_samples[client] += 1

        current_index = 0

        for client, samples_to_give in enumerate(no_of_samples):
            end_index = current_index + samples_to_give
            client_samples[client].extend(item_pointers[current_index:end_index])
            current_index = end_index

    for samples in client_samples:
        rng.shuffle(samples)

    return client_samples
