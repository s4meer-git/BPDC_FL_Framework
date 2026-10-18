import numpy as np
import torch
import printer
from support_modules.precision import autocast


# Receives batch counts, correct predictions and accuracy from accuracy() before forwarding them; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("analysis.metrics", event, config=config, **data)


# Receives a model, test loader and device from main.py and counts correct predictions for global or client accuracy.
@torch.no_grad()
def accuracy(model, data_loader, device, config=None, client_id=None):
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    for batch, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if config is not None:
            with autocast(config):
                outputs = model(images)
        else:
            outputs = model(images)
        if not torch.isfinite(outputs).all().item():
            scope = "global test set" if client_id is None else f"client {client_id}"
            raise ValueError(f"Non-finite model predictions on {scope}, batch {batch}.")

        predicted_classes = outputs.argmax(dim=1)

        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)


        # Client ID, test batch, cumulative correct predictions and accuracy so far; info passing to printer.py.
        report_progress(
            config,
            "evaluation_batch",
            client_id=client_id,
            batch=batch,
            correct_predictions=correct_predictions,
            total_predictions=total_predictions,
            accuracy=correct_predictions / total_predictions,
        )

    if total_predictions == 0:
        return 0.0

    return correct_predictions / total_predictions


# main.py passes global accuracy and each client's score here to calculate the final fairness metrics.
def calculate_fairness_metrics(global_accuracy, client_accuracies):
    client_accuracies = np.array(client_accuracies, dtype=float)

    mean_client_accuracy = np.mean(client_accuracies)
    worst_client_accuracy = np.min(client_accuracies)
    client_standard_deviation = np.std(client_accuracies)

    numerator = np.sum(client_accuracies) ** 2
    denominator = len(client_accuracies) * np.sum(client_accuracies ** 2)

    if denominator == 0:
        jain_fairness_index = 0.0
    else:
        jain_fairness_index = numerator / denominator
    return {
        "global_accuracy": global_accuracy,
        "mean_client_accuracy": mean_client_accuracy,
        "worst_client_accuracy": worst_client_accuracy,
        "client_standard_deviation": client_standard_deviation,
        "jain_fairness_index": jain_fairness_index,
    }
