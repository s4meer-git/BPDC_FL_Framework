import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
from unittest.mock import patch

import torch
import printer
from configuration import configuration
from models.model_selector import build_model
from participants.train import train_local
from support_modules.fedmut_author_reference import LocalUpdate_FedMut


# Replay messages come from this script and go through the same console printer as training.
def report(message):
    printer.report("support_modules.replay_fedmut", "fedmut_replay", message=message)


# Read tensor-only checkpoints from an archive produced by FailureSnapshot.
def read_tensors(archive, name):
    with archive.open(name) as file:
        return torch.load(file, map_location="cpu", weights_only=True)


# Hash the actual tensor bytes so a matching trace means identical values, not merely similar ranges.
def fingerprint(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        if tensor is not None:
            digest.update(name.encode())
            value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


# Both trainers receive these saved, already augmented batches with their recorded random-generator state.
class RecordedBatches:
    def __init__(self, archive, records, device):
        self.archive = archive
        self.records = records
        self.device = device
        self.epoch = 0
        self.location = None

    def __iter__(self):
        self.epoch += 1
        for record in self.records:
            if record["epoch"] != self.epoch:
                continue
            batch = read_tensors(self.archive, record["file"])
            torch.set_rng_state(batch["cpu_rng"])
            if self.device == "cuda" and batch["cuda_rng"] is not None:
                torch.cuda.set_rng_state(batch["cuda_rng"])
            self.location = {"epoch": record["epoch"], "batch": record["batch"]}
            yield batch["images"], batch["labels"]


# Run either the real framework trainer or the pinned author trainer using the same model and saved inputs.
def replay(archive, metadata, device, author):
    config = configuration()
    for name, value in metadata["settings"].items():
        setattr(config, name, value)
    config.fedmut_cifar_grad_norm = metadata["settings"].get("fedmut_cifar_grad_norm")
    config.device = device
    config.current_round = metadata["round"]
    model = build_model(config.dataset)
    model.load_state_dict(read_tensors(archive, "initial.pt"))
    model.to(device)
    loader = RecordedBatches(archive, metadata["batches"], device)
    trace = []
    active_model = [model]

    def record(stage, tensors):
        trace.append({**loader.location, "stage": stage, "hash": fingerprint(tensors)})

    def before_forward(module, inputs):
        active_model[0] = module
        record("before_forward", module.state_dict().items())

    def after_forward(module, inputs, output):
        record("predictions", [("predictions", output)])

    def layer_check(name):
        def check(module, inputs, output):
            if torch.is_tensor(output) and not torch.isfinite(output).all().item():
                record(f"invalid_layer:{name}", [(name, output)])
                raise ValueError(f"First invalid layer output: {name}, {loader.location}")
        return check

    model.register_forward_pre_hook(before_forward)
    model.register_forward_hook(after_forward)
    for name, module in model.named_modules():
        if name and not list(module.children()):
            module.register_forward_hook(layer_check(name))

    if author:
        model.register_forward_hook(lambda module, inputs, output: {"output": output})

    original_sgd = torch.optim.SGD

    class ObservedSGD(original_sgd):
        def step(self, closure=None):
            current = active_model[0]
            record("gradients", ((name, p.grad) for name, p in current.named_parameters()))
            result = super().step(closure)
            record("after_optimizer", current.state_dict().items())
            return result

    error = None
    try:
        with patch("torch.optim.SGD", ObservedSGD):
            if author:
                config.optimizer = "sgd"
                config.local_ep = config.local_epochs
                trainer = LocalUpdate_FedMut.__new__(LocalUpdate_FedMut)
                trainer.args = config
                trainer.ldr_train = loader
                trainer.loss_func = torch.nn.CrossEntropyLoss()
                trainer.verbose = False
                trainer.train(model)
            else:
                train_local(model, loader, config, metadata["client_id"])
    except (ValueError, RuntimeError) as failure:
        error = str(failure)
    return {"trace": trace, "error": error}


# The command line supplies one saved ZIP and a replay device; no training datasets are downloaded.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")

    with zipfile.ZipFile(args.snapshot) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("cublas_workspace_config"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = metadata["cublas_workspace_config"]
        torch.set_num_threads(metadata.get("cpu_threads", torch.get_num_threads()))
        torch.use_deterministic_algorithms(metadata["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = metadata["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = metadata["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = metadata["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = metadata["matmul_allow_tf32"]
        report(f"Replaying {len(metadata['batches'])} batches on {args.device}; original device: {metadata['device']}.")
        report("This compares the two local trainers on the saved client inputs; it does not rerun aggregation or mutation.")
        report("Different hardware or nondeterministic GPU kernels can change the original failure point.")
        framework = replay(archive, metadata, args.device, author=False)
        author = replay(archive, metadata, args.device, author=True)

    difference = None
    for index in range(max(len(framework["trace"]), len(author["trace"]))):
        left = framework["trace"][index] if index < len(framework["trace"]) else None
        right = author["trace"][index] if index < len(author["trace"]) else None
        if left != right:
            difference = {"index": index, "framework": left, "author": right}
            break

    result = {
        "metadata": metadata, "replay_device": args.device,
        "replay_torch_version": str(torch.__version__),
        "framework": framework, "author": author, "first_difference": difference,
    }
    output = args.snapshot.with_suffix(".replay.json")
    output.write_text(json.dumps(result, indent=2))
    report(f"Framework stop: {framework['error']}")
    report(f"Author stop: {author['error']}")
    report("Recorded traces match exactly." if difference is None else f"First trace difference: {difference}")
    report(f"Replay report: {output}")


if __name__ == "__main__":
    main()
