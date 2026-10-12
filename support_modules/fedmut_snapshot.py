import json
import os
from pathlib import Path
import tempfile
import zipfile

import torch
import printer


# train_local() supplies the current client, model and optimizer; only failed FedMut clients keep a recording.
class FailureSnapshot:
    def __init__(self, config, model, optimizer, client_id):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.client_id = client_id
        self.enabled = config.method.lower() == "fedmut" and printer.csv_file is not None
        self.batches = []


    # Open a temporary archive beside the CSV and save this client's starting weights and runtime settings.
    def __enter__(self):
        if not self.enabled:
            return self

        self.folder = printer.csv_file.parent / "fedmut_snapshots"
        self.folder.mkdir(exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix=".recording_", suffix=".zip", dir=self.folder)
        os.close(descriptor)
        self.path = Path(filename)
        self.archive = zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_STORED)
        self.save("initial.pt", self.model.state_dict())
        self.metadata = {
            "settings": dict(printer.run_settings),
            "round": self.config.current_round,
            "client_id": self.client_id,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "device": str(self.config.device),
            "model_class": type(self.model).__name__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        return self


    # Save tensors directly into the archive so complete client datasets do not accumulate in memory.
    def save(self, name, value):
        with self.archive.open(name, "w", force_zip64=True) as file:
            torch.save(value, file)


    # Each batch comes from the real training loader after augmentation, together with its pre-forward RNG state.
    def record_batch(self, images, labels, epoch, batch):
        if not self.enabled:
            return

        name = f"batch_{len(self.batches) + 1:06d}.pt"
        cuda_rng = None
        if torch.device(self.config.device).type == "cuda":
            cuda_rng = torch.cuda.get_rng_state(self.config.device)
        self.save(name, {
            "images": images.detach().cpu(),
            "labels": labels.detach().cpu(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng,
        })
        self.batches.append({"file": name, "epoch": epoch, "batch": batch})


    # Keep the archive only on failure, including the optimizer, gradients and model at the stopping point.
    def __exit__(self, error_type, error, traceback):
        if not self.enabled:
            return False

        try:
            if error is not None:
                self.save("failure.pt", {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "gradients": {name: p.grad for name, p in self.model.named_parameters()},
                })
                self.metadata.update(error=str(error), batches=self.batches)
                self.archive.writestr("metadata.json", json.dumps(self.metadata, indent=2))
            self.archive.close()

            if error is None:
                self.path.unlink()
            else:
                stem = f"round_{self.config.current_round}_client_{self.client_id}"
                version = 1
                destination = self.folder / f"{stem}_{version}.zip"
                while destination.exists():
                    version += 1
                    destination = self.folder / f"{stem}_{version}.zip"
                self.path.rename(destination)


                # Failed client, recorded batch count and the saved archive path; info passing to printer.py.
                printer.report(
                    "support_modules.fedmut_snapshot", "fedmut_snapshot", config=self.config,
                    client_id=self.client_id, batches=len(self.batches), path=str(destination),
                )
        except Exception as save_error:
            self.archive.close()
            printer.report(
                "support_modules.fedmut_snapshot", "fedmut_snapshot_error", config=self.config,
                error=str(save_error), path=str(self.path),
            )
            if error is None:
                raise
        return False
