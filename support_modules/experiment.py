import copy
import csv
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from analysis.results import comparison_settings, training_losses
from printer import settings_from_config
from support_modules.selfishness import choose_selfish_clients
from support_modules.validation import validate_config


METHODS = ("fedavg", "fairrfl", "fedcda", "fedmut", "fcfl")


# The notebook passes its settings and selections here; validate every planned run before spending training time.
# Results go to Results/<dataset>/RUN1, RUN2, ... (one RUN per seed, in SEEDS order), with one CSV per method.
def prepare_runs(config, datasets, methods, results_root, momentum_by_dataset, seeds, smoke_test=False):
    if not datasets or not methods or len(set(datasets)) != len(datasets) or len(set(methods)) != len(methods):
        raise ValueError("Choose at least one dataset and method, without duplicates.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose at least one seed, without duplicates.")
    runs = []


    # Seeds run in the outer loop so every method finishes one seed before the next seed starts.
    for run_number, seed in enumerate(seeds, start=1):
        for dataset in datasets:
            for method in methods:
                run_config = copy.deepcopy(config)
                run_config.dataset = dataset
                run_config.method = method
                run_config.seed = seed
                if dataset not in momentum_by_dataset:
                    raise ValueError(f"Choose a momentum value for {dataset}.")
                run_config.momentum = momentum_by_dataset[dataset]
                validate_config(run_config)
                run_config.reset_run_state()
                run_config.selfish_client_ids = choose_selfish_clients(run_config)


                # Selfish runs and smoke tests get their own dataset folder so they never mix with the full baseline.
                dataset_folder = dataset + ("_selfish" if run_config.selfishness else "")
                if smoke_test:
                    dataset_folder = "smoke_" + dataset_folder
                folder = Path(results_root) / dataset_folder / f"RUN{run_number}"
                runs.append((run_config, folder))
    return runs


# Existing CSVs provide their original settings even when a previous run stopped before completion.
def recorded_settings(path):
    with Path(path).open(newline="") as file:
        first = next(csv.DictReader(file), None)
    if first is None or first.get("event") != "run_started":
        raise ValueError(f"{path}: missing run settings; choose a new comparison folder.")
    return json.loads(first["data"])["settings"]


# The launcher rejects different datasets, partitions, attack scenarios or method variants before replacing any CSV.
def check_existing_results(config, folder):
    expected = settings_from_config(config)
    for path in sorted(Path(folder).glob("*.csv")):
        if path.stem not in METHODS:
            continue
        previous = recorded_settings(path)
        if previous.get("method") != path.stem:
            raise ValueError(f"{path}: the filename does not match its recorded method.")
        if comparison_settings(previous) != comparison_settings(expected):
            raise ValueError(f"{path}: shared settings differ from this run. Move or rename that folder to keep it, then run again.")
        if previous["method"] != config.method:
            continue
        defaults = {"fairrfl_dynamic_q": True}
        keys = [key for key in expected if key.startswith(config.method + "_")]
        changed = [key for key in keys if previous.get(key, defaults.get(key)) != expected[key]]
        if changed:
            raise ValueError(f"{path}: method settings changed ({', '.join(changed)}). Move or rename that folder to keep it, then run again.")


# Drive keeps the same tree as the local Results folder: <dataset>/RUN<n>, with summary.csv in <dataset>.
def backup_path(folder, backup_root):
    if backup_root is None:
        return None
    folder = Path(folder)
    depth = 2 if folder.name.startswith("RUN") else 1
    return Path(backup_root).joinpath(*folder.parts[-depth:])


# Training sends its current CSV here after each round; the plotting cells also use this to back up their PDFs.
def backup_csv(path, backup_folder):
    if backup_folder is None or not path.exists():
        return
    backup_folder = Path(backup_folder)
    if path.parent.resolve() == backup_folder.resolve():
        return
    try:
        backup_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_folder / path.name)
    except OSError as error:
        print(f"Backup failed for {path}; the local file is available: {error}")


# A run that stopped on NaN or infinity is a result (the method diverged), not an interruption to retry.
def diverged_result(path):
    if not Path(path).exists():
        return None
    data = pd.read_csv(path, usecols=["event", "data"])
    failed = data[data["event"] == "run_failed"]
    if failed.empty:
        return None
    error = json.loads(failed.iloc[-1]["data"]).get("error", "")
    return error if "non-finite" in error.lower() else None


# The runner checks existing CSVs before reusing them; incomplete accuracy or loss records require a fresh run.
def completed_result(path):
    if not path.exists():
        return False
    try:
        training_losses({path.name: pd.read_csv(path)})
    except (ValueError, KeyError, TypeError):
        return False
    return True


# The notebook starts the selected runs here; finished runs are reused and incomplete runs restart from round 1.
def run_experiments(runs, backup_root=None, rerun_completed=False):
    from main import main
    from torchvision.datasets.utils import download_and_extract_archive

    project_root = Path(__file__).resolve().parents[1]
    for config, folder in runs:
        backup_folder = backup_path(folder, backup_root)
        check_existing_results(config, folder)
        if backup_folder is not None and backup_folder.resolve() != folder.resolve() and backup_folder.exists():
            check_existing_results(config, backup_folder)

    outcomes = []
    for config, folder in runs:
        csv_path = folder / f"{config.method}.csv"
        backup_folder = backup_path(folder, backup_root)
        completed = completed_result(csv_path) if not rerun_completed else False
        if backup_folder is not None and not rerun_completed and not completed:
            saved_path = backup_folder / csv_path.name
            if saved_path.exists() and saved_path.resolve() != csv_path.resolve():
                completed = completed_result(saved_path)
                if completed or not csv_path.exists():
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved_path, csv_path)
        if completed:
            print("Already completed:", csv_path)
            backup_csv(csv_path, backup_folder)
            outcomes.append({"dataset": config.dataset, "method": config.method, "seed": config.seed, "status": "already completed", "CSV": str(csv_path)})
            continue


        # A recorded divergence is skipped on later runs instead of spending GPU time to diverge again.
        if not rerun_completed:
            divergence = diverged_result(csv_path)
            if divergence is None and backup_folder is not None:
                divergence = diverged_result(backup_folder / csv_path.name)
            if divergence is not None:
                print("Diverged earlier, skipped:", csv_path, "|", divergence)
                outcomes.append({"dataset": config.dataset, "method": config.method, "seed": config.seed, "status": "diverged", "CSV": str(csv_path)})
                continue
        if csv_path.exists() and not rerun_completed:
            print("Restarting incomplete run from round 1:", csv_path)

        if config.dataset == "tinyimagenet":
            data_root = project_root / "data"
            if not (data_root / "tiny-imagenet-200" / "train").is_dir():
                download_and_extract_archive("https://cs231n.stanford.edu/tiny-imagenet-200.zip", str(data_root))
        started = time.perf_counter()
        status = "completed"
        try:
            main(config, output_directory=folder, round_callback=lambda: backup_csv(csv_path, backup_folder))
        except ValueError as error:


            # NaN or infinity means this method diverged; record it and continue with the remaining runs.
            if "non-finite" not in str(error).lower():
                raise
            status = "diverged"
            print("Diverged:", config.method, config.dataset, "seed", config.seed, "|", error)
        finally:
            backup_csv(csv_path, backup_folder)
            print("Elapsed minutes:", round((time.perf_counter() - started) / 60, 2))
        outcomes.append({"dataset": config.dataset, "method": config.method, "seed": config.seed, "status": status, "CSV": str(csv_path)})
    return pd.DataFrame(outcomes)


# Manual analysis cells ask for the existing method CSVs in each selected comparison folder.
def comparison_csvs(runs, backup_root=None):
    folders = {}
    for config, folder in runs:
        check_existing_results(config, folder)
        if backup_root is not None:
            check_existing_results(config, backup_path(folder, backup_root))
        folders.setdefault(folder, []).append(config.method)
    for folder, methods in folders.items():
        data = {}
        for method in methods:
            path = folder / f"{method}.csv"
            if not path.exists() and backup_root is not None:
                path = backup_path(folder, backup_root) / path.name
                print("Local CSV missing; using the Drive copy:", path)
            if not path.exists():
                raise FileNotFoundError(f"No result for {method}: {path}")
            divergence = diverged_result(path)
            if divergence is not None:
                print(f"{folder.parent.name}/{folder.name}: {method} diverged and is left out of this plot | {divergence}")
                continue
            data[path.name] = pd.read_csv(path)
            print("Loaded:", path)
        yield folder, data
