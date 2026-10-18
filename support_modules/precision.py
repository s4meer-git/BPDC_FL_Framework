import torch


# config.mixed_precision = None turns mixed precision on only for CUDA, where fp16 tensor cores make training faster.
def precision_enabled(config):
    if config.mixed_precision is None:
        return config.device == "cuda"
    return bool(config.mixed_precision)


# CUDA uses fp16; the CPU path (used only for checks) uses bfloat16, which needs no loss scaling.
def autocast(config):
    device_type = "cuda" if config.device == "cuda" else "cpu"
    dtype = torch.float16 if config.device == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=precision_enabled(config))


# One scaler per run, so its loss scale is calibrated once and then reused by every client and round.
def grad_scaler(config):
    if config.grad_scaler is None:
        config.grad_scaler = torch.amp.GradScaler("cuda", enabled=precision_enabled(config) and config.device == "cuda")
    return config.grad_scaler
