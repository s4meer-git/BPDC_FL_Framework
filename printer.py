import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


LOG_DIRECTORY = Path(__file__).resolve().parent / "Results"
csv_file = None
run_settings = {}


# Client partitions can contain long index lists, so allow larger fields when reading the saved CSVs.
csv.field_size_limit(sys.maxsize)


# The launcher and main.py pass a config here to record experiment settings and selected selfish clients.
def settings_from_config(config):
    run_state = {"current_round", "fcfl_global_accuracy"}
    settings = {
        name: value
        for name, value in vars(config).items()
        if (value is None or isinstance(value, (str, bool, int, float))) and name not in run_state
    }
    settings["training_batch_size"] = config.batch_size
    settings["selfish_client_ids"] = config.selfish_client_ids
    settings["selfish_attack"] = "known_client_count_v1" if config.selfishness else "none"
    return settings


# main.py supplies the selected folder; replace this method's CSV after the launcher has checked existing settings.
def start_run(config, output_directory=None):
    global csv_file, run_settings
    folder = LOG_DIRECTORY / "comparison" if output_directory is None else Path(output_directory).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    csv_file = folder / f"{config.method.lower()}.csv"
    csv_file.write_text("", encoding="utf-8")
    run_settings = settings_from_config(config)


# main.py calls this when training ends; the method CSV stays saved and only the active run's settings are cleared.
def end_run():
    global csv_file, run_settings

    csv_file = None
    run_settings = {}


# report() sends training values here, including NumPy numbers and lists that need conversion before saving as JSON.
def clean_data(value):
    if isinstance(value, dict):
        return {str(key): clean_data(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [clean_data(item) for item in value]

    if hasattr(value, "tolist"):
        return clean_data(value.tolist())

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    return value


# log_csv() supplies the method's filename and one row; write the header when its CSV is new or empty.
def write_csv(path, row):
    new_file = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if new_file:
            writer.writeheader()

        writer.writerow(row)


# report() sends all progress and final results here so each method keeps everything in its own CSV.
def log_csv(source, event, data):
    if csv_file is None:
        return

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "event": event,
    }

    for name in ("method", "round_number", "client_id", "epoch", "batch"):
        row[name] = data.get(name, "")

    row["data"] = json.dumps(data)
    write_csv(csv_file, row)


# report() passes the event and its values here to print settings, client progress, training losses and final results.
def print_progress(event, data):
    if event == "run_started":
        settings = data["settings"]

        print("Device:", settings["device"])
        print("Dataset:", settings["dataset"])
        print("Method:", settings["method"])
        print("Clients:", settings["num_of_clients"])
        print("Rounds:", settings["rounds"])
        print("Training batch size:", settings["training_batch_size"])
        print("Selfish clients:", settings.get("selfish_client_ids", []))
        if settings.get("selfishness", False):
            print("Attack strength:", settings["selfish_strength"], "starting round:", settings["selfish_start_round"])

        print("Local epochs:", settings["local_epochs"])

        if settings["method"].lower() == "fedmut" and settings["dataset"].lower() == "cifar10":
            print("FedMut CIFAR-10 gradient norm limit:", settings.get("fedmut_cifar_grad_norm"))

    elif event == "dataset_loaded":
        print("Training samples:", data["training_samples"])
        print("Testing samples:", data["testing_samples"])

    elif event == "client_partition":
        print(
            "Client", data["client_id"],
            "training samples:", data["training_samples"],
            "testing samples:", data["testing_samples"],
        )

    elif event == "round_started":
        print(
            "\nCommunication round", data["round_number"],
            "method:", data["method"],
            flush=True,
        )

    elif event == "clients_selected":
        print("Selected clients:", data["client_ids"])

    elif event == "client_started":
        print("Training client", data["client_id"], flush=True)

    elif event == "epoch_started":
        print("Local epoch", data["epoch"], "started", flush=True)

    elif event == "batch_completed" and data["batch"] % 100 == 0:
        print(
            "Local epoch", data["epoch"],
            "completed batches:", data["batch"],
            flush=True,
        )

    elif event == "epoch_completed":
        print(
            "Local epoch", data["epoch"],
            "completed batches:", data["batches"],
            "average loss:", round(data["average_loss"], 4),
            flush=True,
        )

    elif event == "aggregation_started":
        print("Aggregating updates from", data["client_count"], "clients", flush=True)

    elif event == "fairrfl_detection":
        print("FairRFL detected clients:", data["detected_clients"])
        print("FairRFL server update norm:", round(data["server_update_norm"], 6))
        if data.get("small_server_update", False):
            print("FairRFL's global update is small relative to local updates; inspect this short run before extending it.")

    elif event == "fcfl_aggregated":
        print("FCFL weights:", [round(weight, 4) for weight in data["weights"]])

    elif event == "selfish_round":
        print("Clients applying the selfish attack:", data["attacked_client_ids"])

    elif event == "round_accuracy":
        print(
            "Global accuracy after round", data["round_number"], ":",
            round(data["global_accuracy"] * 100, 2), "%",
        )

    elif event == "evaluation_started":
        print("Evaluating final client accuracies")

    elif event == "round_evaluation_started":
        print("Evaluating client accuracies for round", data["round_number"])

    elif event == "final_results":
        metrics = data["metrics"]

        print("\nFinal results")

        print(
            "Global accuracy:",
            round(metrics["global_accuracy"] * 100, 2), "%",
        )

        print(
            "Mean client accuracy:",
            round(metrics["mean_client_accuracy"] * 100, 2), "%",
        )

        print(
            "Worst client accuracy:",
            round(metrics["worst_client_accuracy"] * 100, 2), "%",
        )

        print(
            "Client standard deviation:",
            round(metrics["client_standard_deviation"] * 100, 2),
            "percentage points",
        )

        print("Jain fairness index:", round(metrics["jain_fairness_index"], 4))
        if "honest_clients" in metrics:
            honest = metrics["honest_clients"]
            print("Honest clients mean accuracy:", round(honest["mean_client_accuracy"] * 100, 2), "%")
            print("Honest clients standard deviation:", round(honest["client_standard_deviation"] * 100, 2), "percentage points")
            print("Honest clients worst accuracy:", round(honest["worst_client_accuracy"] * 100, 2), "%")

    elif event == "run_completed":
        print("CSV:", data["csv_file"])

    elif event == "run_failed":
        print("Run failed:", data["error"])
        print("CSV:", data["csv_file"])


# main.py, participants, analysis and method files send their event names, config and progress values here.
def report(source, event, config=None, **data):
    if config is not None:
        data.setdefault("method", config.method)
        data.setdefault("round_number", getattr(config, "current_round", 0))

    if event == "model_created":
        run_settings["model_class"] = data["model_class"]

    if event in {"run_started", "final_results"}:
        data["settings"] = run_settings

    log_csv(source, event, clean_data(data))
    print_progress(event, data)
