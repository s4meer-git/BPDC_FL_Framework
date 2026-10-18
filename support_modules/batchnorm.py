import torch
from torch.nn.modules.batchnorm import _BatchNorm


# FairRFL passes local BatchNorm buffers here alongside the newly aggregated parameters.
def merge_batchnorm_buffers(new_weights, global_model, client_weights, client_sizes):
    total_samples = sum(client_sizes)

    for module_name, module in global_model.named_modules():
        if not isinstance(module, _BatchNorm) or not module.track_running_stats:
            continue

        prefix = module_name + "." if module_name else ""

        for buffer_name in ("running_mean", "running_var"):
            name = prefix + buffer_name
            values = [weights[name].detach().cpu() for weights in client_weights]

            if any(not torch.isfinite(value).all().item() for value in values):
                raise ValueError(f"Non-finite BatchNorm buffer: {name}")
            if buffer_name == "running_var" and any((value < 0).any().item() for value in values):
                raise ValueError(f"Negative BatchNorm running variance: {name}")

            average = torch.zeros_like(values[0])
            for value, size in zip(values, client_sizes):
                average += value * (size / total_samples)
            new_weights[name] = average


        # Counters describe training steps, so keep the largest count instead of averaging integers.
        name = prefix + "num_batches_tracked"
        new_weights[name] = torch.stack([
            weights[name].detach().cpu() for weights in client_weights
        ]).max()
