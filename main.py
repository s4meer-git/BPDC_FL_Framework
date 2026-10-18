from collections import Counter
import os
import torch
from torch.random import fork_rng
from torch.utils.data import DataLoader, Subset

import printer

from configuration import configuration
from datasets.loaders import load_dataset, get_dataset_labels
from datasets.partition import dirichlet_partition, matching_partition
from models.model_selector import build_model
from participants.client import Clients
from participants.server import run_one_round
from analysis.metrics import accuracy, calculate_fairness_metrics
from support_modules.seed import set_seed
from support_modules.validation import validate_config
from support_modules.selfishness import choose_selfish_clients


# Receives settings, client partitions and accuracy results from functions in main.py; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("main", event, config=config, **data)


# Every DataLoader uses these options; for Tiny ImageNet, extra processes decode JPEGs while the GPU trains.
def loader_options(config):
    workers = config.loader_workers
    if workers is None:
        workers = min(4, os.cpu_count() or 1) if config.device == "cuda" and config.dataset == "tinyimagenet" else 0
    return {"num_workers": workers, "pin_memory": config.device == "cuda"}


# Split the datasets loaded by main.py into clients using the seed and Dirichlet settings from config.
def create_clients(train_dataset, test_dataset, config):
    train_labels = get_dataset_labels(train_dataset)
    test_labels = get_dataset_labels(test_dataset)

    train_partitions = dirichlet_partition(
        train_labels,
        config.num_of_clients,
        config.dirichlet_alpha,
        config.seed
    )


    # "matched" tests each client on its own training class mix; "dirichlet" keeps the earlier independent test split.
    if config.test_partition == "matched":
        test_partitions = matching_partition(
            test_labels,
            train_labels,
            train_partitions,
            config.seed + 1
        )
    else:
        test_partitions = dirichlet_partition(
            test_labels,
            config.num_of_clients,
            config.dirichlet_alpha,
            config.seed + 1
        )

    clients = []

    for client_id, indices in enumerate(train_partitions):
        if not indices:
            raise ValueError(
                f"Client {client_id} has no training samples. "
                "Use fewer clients or increase dirichlet_alpha."
            )

    for client_id, indices in enumerate(test_partitions):
        if not indices:
            raise ValueError(f"Client {client_id} has no test samples; its accuracy cannot be used in a fairness comparison.")

    for client_id in range(config.num_of_clients):
        client_train_dataset = Subset(
            train_dataset,
            train_partitions[client_id]
        )

        client_test_dataset = Subset(
            test_dataset,
            test_partitions[client_id]
        )


        train_loader = DataLoader(
            client_train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            **loader_options(config)
        )
        test_loader = DataLoader(
            client_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **loader_options(config)
        )
        client = Clients(client_id, train_loader, test_loader)
        clients.append(client)

        training_label_counts = Counter(
            train_labels[index] for index in train_partitions[client_id]
        )
        testing_label_counts = Counter(
            test_labels[index] for index in test_partitions[client_id]
        )


        # Training/test sample counts, label counts and assigned indices for this client; info passing to printer.py.
        report_progress(
            config,
            "client_partition",
            client_id=client_id,
            training_samples=len(client_train_dataset),
            testing_samples=len(client_test_dataset),
            training_label_counts=dict(training_label_counts),
            testing_label_counts=dict(testing_label_counts),
            training_indices=train_partitions[client_id],
            testing_indices=test_partitions[client_id],
        )

    return clients


# Start an experiment with the supplied config and chosen results folder, or use the default configuration.
def main(config=None, output_directory=None, round_callback=None):
    if config is None:
        config = configuration()

    method = config.method.lower()

    if config.num_of_clients < 1:
        raise ValueError("The number of clients must be positive.")
    if method not in {"fedavg", "fairrfl", "fedcda", "fedmut", "fcfl"}:
        raise ValueError("Unknown method name: " + config.method)
    validate_config(config)
    config.reset_run_state()
    config.selfish_client_ids = choose_selfish_clients(config)


    # Training settings and the folder chosen for this comparison's method CSVs; info passing to printer.py.
    printer.start_run(config, output_directory)
    try:
        metrics = _run_experiment(config, round_callback)


        # The method CSV that contains this run's progress and final results; info passing to printer.py.
        report_progress(
            config,
            "run_completed",
            csv_file=str(printer.csv_file),
        )

        return metrics
    except (Exception, KeyboardInterrupt) as error:


        # The error or interruption and the method CSV where its progress was saved; info passing to printer.py.
        report_progress(
            config,
            "run_failed",
            error=str(error) or type(error).__name__,
            csv_file=str(printer.csv_file),
        )

        raise
    finally:
        printer.end_run()


# main() supplies the config used to load datasets, train the rounds and calculate the final scores.
def _run_experiment(config, round_callback=None):
    set_seed(config.seed)


    # Dataset, device, client count, learning rate and other training settings; info passing to printer.py.
    report_progress(
        config,
        "run_started",
    )

    train_dataset, test_dataset = load_dataset(config.dataset)


    # Total training and test sample counts from the loaded datasets; info passing to printer.py.
    report_progress(
        config,
        "dataset_loaded",
        dataset=config.dataset,
        training_samples=len(train_dataset),
        testing_samples=len(test_dataset),
    )

    clients = create_clients(train_dataset, test_dataset, config)

    global_test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_options(config)
    )



    # Reseed only for model creation so every method starts from the same initial weights for this seed.
    with fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        global_model = build_model(config.dataset)
    global_model = global_model.to(config.device)


    # The built model class, device and total number of parameters; info passing to printer.py.
    report_progress(
        config,
        "model_created",
        model_class=type(global_model).__name__,
        device=config.device,
        parameter_count=sum(p.numel() for p in global_model.parameters()),
    )

    for round_number in range(1, config.rounds + 1):
        global_model = run_one_round(
            global_model,
            clients,
            config,
            round_number
        )

        global_accuracy = accuracy(
            global_model,
            global_test_loader,
            config.device,
            config=config
        )


        # This round number, global test accuracy and evaluated sample count; info passing to printer.py.
        report_progress(
            config,
            "round_accuracy",
            round_number=round_number,
            global_accuracy=global_accuracy,
            testing_samples=len(test_dataset),
        )


        # The round and client count being evaluated for the notebook; info passing to printer.py.
        report_progress(
            config,
            "round_evaluation_started",
            round_number=round_number,
            client_count=len(clients),
        )


        # Test loaders use the CPU random generator too, so preserve it for the next round's training shuffle.
        with fork_rng(devices=[]):
            for client in clients:
                client_accuracy = accuracy(
                    global_model,
                    client.test_loader,
                    config.device,
                    config=config,
                    client_id=client.client_id,
                )


                # This client's accuracy after this round, used to calculate its round average; info passing to printer.py.
                report_progress(
                    config,
                    "round_client_accuracy",
                    round_number=round_number,
                    client_id=client.client_id,
                    client_accuracy=client_accuracy,
                    testing_samples=len(client.test_loader.dataset),
                )


        # The launcher backs up the current CSV after each fully evaluated round.
        if round_callback is not None:
            round_callback()

    client_accuracies = []


    # The number of clients whose final test accuracy will be evaluated; info passing to printer.py.
    report_progress(
        config,
        "evaluation_started",
        client_count=len(clients),
    )

    for client in clients:
        client_accuracy = accuracy(
            global_model,
            client.test_loader,
            config.device,
            config=config,
            client_id=client.client_id
        )

        client_accuracies.append(client_accuracy)


        # This client ID, its final test accuracy and its test sample count; info passing to printer.py.
        report_progress(
            config,
            "client_accuracy",
            client_id=client.client_id,
            client_accuracy=client_accuracy,
            testing_samples=len(client.test_loader.dataset),
        )

    global_accuracy = accuracy(
        global_model,
        global_test_loader,
        config.device,
        config=config
    )

    metrics = calculate_fairness_metrics(
        global_accuracy,
        client_accuracies
    )

    metrics["client_accuracies"] = client_accuracies

    if config.selfishness:
        honest = [client for client in clients if client.client_id not in config.selfish_client_ids]
        honest_scores = [client_accuracies[client.client_id] for client in honest]
        honest_sizes = [len(client.test_loader.dataset) for client in honest]
        honest_accuracy = sum(score * size for score, size in zip(honest_scores, honest_sizes)) / sum(honest_sizes)
        metrics["honest_clients"] = calculate_fairness_metrics(honest_accuracy, honest_scores)
        metrics["selfish_client_ids"] = config.selfish_client_ids


    # Final global/client accuracies, fairness metrics and their matching client IDs; info passing to printer.py.
    report_progress(
        config,
        "final_results",
        metrics=metrics,
        client_ids=[client.client_id for client in clients],
    )

    return metrics


if __name__ == "__main__":
    raise SystemExit("Open JPNotebooks/setup_remote.ipynb to configure, run and analyse an experiment.")
