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


# main.py calls this to let the console user choose an existing comparison folder or create a named one.
def choose_directory():
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    folders = sorted(
        folder for folder in LOG_DIRECTORY.iterdir()
        if folder.is_dir()
        and not folder.name.startswith((".", "result_"))
        and folder.name not in {"archive", "failed"}
    )

    print("\nChoose a folder for the method CSVs:")

    for number, folder in enumerate(folders, start=1):
        print(f"{number}. {folder.name}")

    while True:
        choice = input("Folder number, name or full path [comparison]: ").strip() or "comparison"

        if choice.isdigit():
            if 1 <= int(choice) <= len(folders):
                return folders[int(choice) - 1]

            print("Pick a number from the list, or type a name for a new folder.")
            continue

        folder = Path(choice).expanduser()

        if not folder.is_absolute():
            folder = LOG_DIRECTORY / folder

        folder.mkdir(parents=True, exist_ok=True)
        return folder


# main.py supplies the folder and config; clear only this method's CSV before writing its new run.
def start_run(config, output_directory=None):
    global csv_file, run_settings

    if output_directory is None:
        folder = LOG_DIRECTORY / "comparison"
    else:
        folder = Path(output_directory).expanduser()

    folder.mkdir(parents=True, exist_ok=True)

    csv_file = folder / f"{config.method.lower()}.csv"
    csv_file.write_text("", encoding="utf-8")

    run_settings = {
        name: value
        for name, value in vars(config).items()
        if value is None or isinstance(value, (str, bool, int, float))
    }
    run_settings["training_batch_size"] = config.get_training_batch_size()


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

        if settings["method"].lower() == "fairmedian":
            print("FairMedian uses one minibatch gradient per client per round.")
        else:
            print("Local epochs:", settings["local_epochs"])

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

    elif event == "gradient_started":
        print("Collecting gradient from client", data["client_id"], flush=True)

    elif event == "aggregation_started":
        print("Aggregating updates from", data["client_count"], "clients", flush=True)

    elif event == "fairrfl_detection":
        print("FairRFL detected clients:", data["detected_clients"])

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

    data = clean_data(data)
    log_csv(source, event, data)
    print_progress(event, data)
