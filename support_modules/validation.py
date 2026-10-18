import math


# The launcher and main.py pass a configuration here before opening a CSV or loading a dataset.
def validate_config(config):
    models = {"mnist": "cnn_mnist", "cifar10": "cnn_cifar10", "tinyimagenet": "resnet18"}
    methods = {"fedavg", "fairrfl", "fedcda", "fedmut", "fcfl"}
    if config.dataset not in models or config.method not in methods:
        raise ValueError("Choose a supported dataset and method in setup_remote.ipynb.")
    config.model = models[config.dataset]
    numeric = ["lr", "dirichlet_alpha", "momentum"]
    if config.method == "fairrfl":
        numeric += ["fairrfl_q", "fairrfl_tau"]
    if config.method == "fedcda":
        numeric += ["fedcda_smoothness"]
    if config.method == "fedmut":
        numeric += ["fedmut_fraction", "fedmut_radius", "fedmut_acc_rate"]
        if config.fedmut_cifar_grad_norm is not None:
            numeric.append("fedmut_cifar_grad_norm")
    if config.selfishness:
        numeric += ["selfish_fraction", "selfish_strength"]
    for name in numeric:
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    for name in ("num_of_clients", "rounds", "local_epochs", "batch_size"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config.test_partition not in ("matched", "dirichlet"):
        raise ValueError('test_partition must be "matched" or "dirichlet".')
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    for name in ("lr", "dirichlet_alpha"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not math.isfinite(config.momentum) or not 0 <= config.momentum < 1:
        raise ValueError("momentum must be between 0 and 1, excluding 1.")
    for name in ("selfishness", "fairrfl_dynamic_q"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be True or False.")

    if config.method == "fairrfl":
        for name in ("fairrfl_q", "fairrfl_tau"):
            value = getattr(config, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite.")
        limit = config.fairrfl_lipschitz
        if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
            raise ValueError("fairrfl_lipschitz must be positive or None.")
    if config.method == "fcfl":
        for name in ("fcfl_alpha", "fcfl_fraction"):
            if type(getattr(config, name)) not in (int, float) or not math.isfinite(getattr(config, name)):
                raise ValueError(f"{name} must be a finite number.")
        if config.fcfl_alpha < 0 or not 0 < config.fcfl_fraction <= 1:
            raise ValueError("fcfl_alpha must be at least 0 and fcfl_fraction must be in (0, 1].")
    if config.method == "fedcda":
        if type(config.fedcda_warmup) is not int or not 0 <= config.fedcda_warmup < config.rounds:
            raise ValueError("FedCDA warmup must be less than rounds so its history search actually runs.")
        for name in ("fedcda_memory_k", "fedcda_batch_client_num"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(config.fedcda_smoothness) or config.fedcda_smoothness <= 0:
            raise ValueError("fedcda_smoothness must be positive and finite.")
        count = config.fedcda_clients_per_round
        if count is not None and (type(count) is not int or not 1 <= count <= config.num_of_clients):
            raise ValueError("fedcda_clients_per_round must be None or a valid client count.")
    if config.method == "fedmut":
        if not math.isfinite(config.fedmut_fraction) or not 0 < config.fedmut_fraction <= 1:
            raise ValueError("fedmut_fraction must be in (0, 1].")
        for name in ("fedmut_radius", "fedmut_acc_rate"):
            if not math.isfinite(getattr(config, name)) or getattr(config, name) < 0:
                raise ValueError(f"{name} must be non-negative and finite.")
        if type(config.fedmut_bound) is not int or config.fedmut_bound < 1:
            raise ValueError("fedmut_bound must be a positive integer.")
        limit = config.fedmut_cifar_grad_norm
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError("fedmut_cifar_grad_norm must be positive or None.")

    if config.selfishness:
        if config.num_of_clients < 3:
            raise ValueError("Use at least three clients for the selfish-client experiment.")
        if not math.isfinite(config.selfish_fraction) or not 0 < config.selfish_fraction < 0.5:
            raise ValueError("selfish_fraction must be above zero and below 0.5, keeping an honest majority.")
        if not math.isfinite(config.selfish_strength) or not 0 < config.selfish_strength <= 1:
            raise ValueError("selfish_strength must be in (0, 1]. Use selfishness=False for the baseline.")
        if type(config.selfish_start_round) is not int or not 2 <= config.selfish_start_round <= config.rounds:
            raise ValueError("selfish_start_round must be between 2 and rounds; use at least one honest round.")
        if config.method == "fedmut" and config.fedmut_fraction != 1:
            raise ValueError("Selfish experiments require fedmut_fraction=1 for matching attack histories.")
        if config.method == "fcfl" and config.fcfl_fraction != 1:
            raise ValueError("Selfish experiments require fcfl_fraction=1 for matching attack histories.")
        if config.method == "fedcda" and config.fedcda_clients_per_round not in (None, config.num_of_clients):
            raise ValueError("Selfish experiments require all FedCDA clients in every round.")
