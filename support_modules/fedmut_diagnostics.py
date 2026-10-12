import math

import torch
import printer


# Temporary checks receive named tensors from FedMut and local training, without changing their values.
def check_tensors(config, stage, tensors, **location):
    if config.method.lower() != "fedmut":
        return

    summaries = {}
    invalid = []

    for name, tensor in tensors.items():
        if tensor is None or tensor.numel() == 0:
            continue

        value = tensor.detach()
        minimum, maximum = torch.stack((value.amin(), value.amax())).cpu().tolist()
        finite = math.isfinite(minimum) and math.isfinite(maximum)
        summaries[name] = {"min": minimum, "max": maximum, "finite": finite}

        if not finite:
            invalid.append(name)


    # Stage, tensor ranges and round/client/batch location from the caller; info passing to printer.py.
    printer.report(
        "support_modules.fedmut_diagnostics",
        "fedmut_diagnostic",
        config=config,
        stage=stage,
        tensors=summaries,
        invalid_tensors=invalid,
        **location,
    )

    if invalid:
        details = ", ".join(f"{name} {value}" for name, value in location.items())
        raise ValueError(
            f"FedMut diagnostic: non-finite tensor at {stage}, "
            f"round {getattr(config, 'current_round', 0)}, {details}: "
            + ", ".join(invalid)
        )
