from collections import OrderedDict
import torch
import printer


# Receives sample counts and aggregation weights calculated by fedavg(); info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fedavg", event, config=config, **data)


# config.aggregate() sends local weights and sample counts here to calculate their weighted average.
def fedavg(client_weights, client_sizes, config=None):
    if not client_weights:
        raise ValueError("FedAvg requires at least one client model.")
    if len(client_weights) != len(client_sizes):
        raise ValueError("Each client model must have a matching sample count.")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("FedAvg requires positive client sample counts.")
    total_samples = sum(client_sizes)

    averaged_weights = OrderedDict()

    for parameter_name in client_weights[0]:
        first_parameter = client_weights[0][parameter_name]


        # Copy integer buffers such as BatchNorm counters instead of treating them as floating point weights.
        if not torch.is_floating_point(first_parameter):
            averaged_weights[parameter_name] = first_parameter.clone()
            continue

        parameter_average = torch.zeros_like(
            first_parameter,
            dtype=torch.float32
        )

        for weights, client_size in zip(client_weights, client_sizes):
            client_parameter = weights[parameter_name].float()
            client_fraction = client_size / total_samples

            parameter_average = parameter_average + (
                client_parameter * client_fraction
            )
        averaged_weights[parameter_name] = parameter_average.to(
            first_parameter.dtype
        )


    # Client sample counts, their FedAvg fractions and the total sample count; info passing to printer.py.
    report_progress(
        config,
        "fedavg_aggregated",
        client_sizes=client_sizes,
        client_fractions=[size / total_samples for size in client_sizes],
        total_samples=total_samples,
        client_count=len(client_weights),
    )

    return averaged_weights
