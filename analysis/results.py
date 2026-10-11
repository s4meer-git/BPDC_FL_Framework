import json
import math

import pandas as pd


# Notebook checks use this for counts and identifiers so values such as round 1.5 cannot silently become round 1.
def whole_number(value, label, minimum=0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: expected a whole number.") from None
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{label}: expected a whole number of at least {minimum}.")
    return int(number)


# The notebook passes one method CSV here to read its recorded settings and confirm that the run finished.
def read_run(data, filename):
    required = {"event", "method", "round_number", "client_id", "epoch", "batch", "data"}
    if not required.issubset(data.columns):
        raise ValueError(f"{filename}: use the method CSV containing run settings and training events.")

    events = data["event"]
    if (
        (events == "run_started").sum() != 1
        or (events == "run_completed").sum() != 1
        or (events == "run_failed").any()
        or events.iloc[0] != "run_started"
        or events.iloc[-1] != "run_completed"
    ):
        raise ValueError(f"{filename}: expected one successfully completed run.")

    settings = json.loads(data.loc[events == "run_started", "data"].iloc[0])["settings"]
    method = str(settings["method"]).strip().lower()
    if not data["method"].astype(str).str.strip().str.lower().eq(method).all():
        raise ValueError(f"{filename}: method names do not match the run settings.")

    rounds = whole_number(settings["rounds"], f"{filename}: rounds", 1)
    clients = whole_number(settings["num_of_clients"], f"{filename}: clients", 1)
    settings = dict(settings, method=method, rounds=rounds, num_of_clients=clients)
    return settings


# Cell 2 sends all loaded CSVs here; each recorded client must have exactly one accuracy in every configured round.
def client_accuracies(csv_data):
    frames = []
    seen_methods = set()

    for filename, data in csv_data.items():
        settings = read_run(data, filename)
        method = settings["method"]
        if method in seen_methods:
            raise ValueError(f"{filename}: more than one CSV contains results for {method}.")
        seen_methods.add(method)

        rows = data.loc[data["event"] == "round_client_accuracy"].copy()
        for name in ("round_number", "client_id"):
            rows[name] = [whole_number(value, f"{filename}: {name}") for value in rows[name]]

        expected = pd.MultiIndex.from_product(
            [range(1, settings["rounds"] + 1), range(settings["num_of_clients"])],
            names=["round_number", "client_id"],
        )
        actual = pd.MultiIndex.from_frame(rows[["round_number", "client_id"]])
        if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
            raise ValueError(
                f"{filename}: missing, duplicate or unexpected client accuracies; "
                "check all configured clients and rounds."
            )

        rows["client_accuracy"] = [float(json.loads(value)["client_accuracy"]) for value in rows["data"]]
        if not rows["client_accuracy"].between(0, 1).all():
            raise ValueError(f"{filename}: client accuracies must be finite fractions between 0 and 1.")
        rows["method"] = method
        frames.append(rows[["method", "round_number", "client_id", "client_accuracy"]])

    if not frames:
        raise ValueError("No method CSVs were loaded.")
    return pd.concat(frames, ignore_index=True)


# Loss checks use each client's logged partition size and the clients actually selected in each round.
def training_schedule(data, settings, filename):
    partitions = data[data["event"] == "client_partition"]
    sizes = {}
    for row in partitions.itertuples():
        client_id = whole_number(row.client_id, f"{filename}: partition client")
        if client_id in sizes:
            raise ValueError(f"{filename}: duplicate client partition.")
        sizes[client_id] = whole_number(json.loads(row.data)["training_samples"], f"{filename}: training samples", 1)
    if set(sizes) != set(range(settings["num_of_clients"])):
        raise ValueError(f"{filename}: missing or unexpected client partitions.")

    participant_count = settings["num_of_clients"]
    if settings["method"] == "fedcda":
        participant_count = settings.get("fedcda_clients_per_round") or participant_count
    elif settings["method"] == "fedmut":
        participant_count = max(int(participant_count * float(settings["fedmut_fraction"])), 1)
    participant_count = whole_number(participant_count, f"{filename}: participating clients", 1)

    schedule = {}
    for row in data[data["event"] == "aggregation_started"].itertuples():
        round_number = whole_number(row.round_number, f"{filename}: aggregation round", 1)
        values = json.loads(row.data)
        clients = [whole_number(value, f"{filename}: selected client") for value in values["client_ids"]]
        if (
            round_number in schedule
            or len(clients) != participant_count
            or len(set(clients)) != len(clients)
            or not set(clients).issubset(sizes)
        ):
            raise ValueError(f"{filename}: invalid or duplicate participation record in round {round_number}.")
        schedule[round_number] = clients

    if set(schedule) != set(range(1, settings["rounds"] + 1)):
        raise ValueError(f"{filename}: missing or unexpected training rounds.")
    return sizes, schedule


# Cell 3 sends the loaded CSVs here; check every expected batch before calculating sample-weighted training loss.
def training_losses(csv_data):
    client_accuracies(csv_data)
    loss_rows = []

    for filename, data in csv_data.items():
        settings = read_run(data, filename)
        sizes, schedule = training_schedule(data, settings, filename)
        method = settings["method"]
        batch_size = whole_number(settings["training_batch_size"], f"{filename}: batch size", 1)
        epochs = 1 if method == "fairmedian" else whole_number(settings["local_epochs"], f"{filename}: epochs", 1)
        expected_batches = {}
        expected_epochs = {}

        for round_number, clients in schedule.items():
            for client_id in clients:
                samples = sizes[client_id]
                if method == "fairmedian":
                    expected_batches[(round_number, client_id, 0, 1)] = min(samples, batch_size)
                    continue

                batches = math.ceil(samples / batch_size)
                for epoch in range(1, epochs + 1):
                    expected_epochs[(round_number, client_id, epoch)] = batches
                    for batch in range(1, batches + 1):
                        key = (round_number, client_id, epoch, batch)
                        remaining = samples - (batch - 1) * batch_size
                        expected_batches[key] = min(batch_size, remaining)

        seen = set()
        events = data[data["event"].isin(["batch_completed", "gradient_collected"])]
        for row in events.itertuples():
            round_number = whole_number(row.round_number, f"{filename}: loss round", 1)
            client_id = whole_number(row.client_id, f"{filename}: loss client")
            if method == "fairmedian":
                if row.event != "gradient_collected":
                    raise ValueError(f"{filename}: unexpected training event for FairMedian.")
                epoch, batch = 0, 1
            else:
                if row.event != "batch_completed":
                    raise ValueError(f"{filename}: unexpected gradient event for {method}.")
                epoch = whole_number(row.epoch, f"{filename}: epoch", 1)
                batch = whole_number(row.batch, f"{filename}: batch", 1)

            key = (round_number, client_id, epoch, batch)
            values = json.loads(row.data)
            loss = float(values["loss"])
            samples = whole_number(values["batch_samples"], f"{filename}: batch samples", 1)
            if key in seen or expected_batches.get(key) != samples:
                raise ValueError(f"{filename}: duplicate, unexpected or incorrectly sized batch {key}.")
            if not math.isfinite(loss) or loss < 0:
                raise ValueError(f"{filename}: invalid training loss at batch {key}.")
            if not math.isfinite(loss * samples):
                raise ValueError(f"{filename}: weighted training loss overflowed at batch {key}.")
            seen.add(key)
            loss_rows.append({
                "method": method,
                "client_id": client_id,
                "round_number": round_number,
                "weighted_loss": loss * samples,
                "samples": samples,
            })

        if seen != set(expected_batches):
            missing = len(expected_batches) - len(seen)
            raise ValueError(f"{filename}: incomplete training losses; {missing} batch records are missing.")

        completed = {}
        for row in data[data["event"] == "epoch_completed"].itertuples():
            key = tuple(
                whole_number(getattr(row, name), f"{filename}: {name}")
                for name in ("round_number", "client_id", "epoch")
            )
            if key in completed:
                raise ValueError(f"{filename}: duplicate completed epoch {key}.")
            completed[key] = whole_number(json.loads(row.data)["batches"], f"{filename}: completed batches", 1)
        if completed != expected_epochs:
            raise ValueError(f"{filename}: missing or inconsistent epoch completion records.")

    if not loss_rows:
        raise ValueError("No training losses were loaded.")
    return pd.DataFrame(loss_rows)
