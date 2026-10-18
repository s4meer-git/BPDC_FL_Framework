from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.results import client_accuracies, comparison_runs


METHOD_NAMES = {"fedavg": "FedAvg", "fairrfl": "FairRFL", "fedcda": "FedCDA", "fedmut": "FedMut", "fcfl": "FCFL"}
METHOD_COLORS = {"fedavg": "tab:blue", "fairrfl": "tab:orange", "fedcda": "tab:red", "fedmut": "tab:purple", "fcfl": "tab:green"}


# Methods whose objective is client fairness get solid lines; the baseline and generalisation methods get dotted lines.
FAIRNESS_METHODS = {"fairrfl", "fcfl"}


# The notebook supplies method CSVs; use each client's testing accuracy from the last configured round.
def final_client_scores(csv_data):
    runs = comparison_runs(csv_data)
    data = client_accuracies(csv_data)
    settings = next(iter(runs.values()))
    scores = data[data["round_number"] == settings["rounds"]].pivot(
        index="client_id", columns="method", values="client_accuracy",
    ).sort_index() * 100
    order = [method for method in METHOD_NAMES if method in scores.columns]
    order += sorted(set(scores.columns) - set(order))
    scores = scores[order]
    scores.index = scores.index + 1
    scores.index.name = "Client"
    return scores, settings


# Final client scores and the logged attacker IDs give separate summaries for all clients and honest clients.
def fairness_summary(scores, settings):
    groups = {"All clients": scores}
    if settings.get("selfishness", False):
        groups["Honest clients"] = scores.drop(index=[i + 1 for i in settings["selfish_client_ids"]])
    rows = []
    for group, values in groups.items():
        for method in values.columns:
            column = values[method]
            rows.append({
                "Method": METHOD_NAMES.get(method, method), "Group": group,
                "Mean accuracy (%)": column.mean(), "Worst accuracy (%)": column.min(),
                "Standard deviation (pp)": column.std(ddof=0),
            })
    return pd.DataFrame(rows).round(2)


# Both notebooks send final client scores here; shaded columns identify the recorded selfish clients.
def plot_client_test_accuracy(scores, settings, pdf_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    for method in scores.columns:
        ax.plot(scores.index, scores[method], marker="o", linewidth=1.3, markersize=5,
                linestyle="-" if method in FAIRNESS_METHODS else ":",
                color=METHOD_COLORS.get(method), label=METHOD_NAMES.get(method, method))
    for index, client_id in enumerate(settings.get("selfish_client_ids", [])):
        ax.axvspan(client_id + 0.7, client_id + 1.3, color="grey", alpha=0.12,
                   label="Selfish client" if index == 0 else None)
    scenario = "selfish clients" if settings.get("selfishness", False) else "no attack"
    ax.set_title(f"{settings['dataset']}: client testing accuracy after round {settings['rounds']} ({scenario})")
    ax.set_xlabel("Client")
    ax.set_ylabel("Testing accuracy (%)")
    ax.set_xticks(scores.index)
    ax.set_xlim(scores.index.min() - 0.5, scores.index.max() + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(5, len(scores.columns) + 1))
    fig.tight_layout()
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fairness_summary(scores, settings)


# The setup notebook supplies CSVs, including incomplete runs, to inspect progress without treating it as a final result.
def plot_run_diagnostics(csv_data, pdf_path):
    import json

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for filename, data in csv_data.items():
        started = data[data["event"] == "run_started"]
        if len(started) != 1:
            raise ValueError(f"{filename}: expected one run before plotting diagnostics.")
        settings = json.loads(started.iloc[0]["data"])["settings"]
        method = settings["method"]
        complete = (data["event"] == "run_completed").sum() == 1 and not (data["event"] == "run_failed").any()
        label = METHOD_NAMES.get(method, method) + ("" if complete else " (incomplete)")
        rows = data[data["event"] == "round_accuracy"]
        accuracies = [100 * json.loads(value)["global_accuracy"] for value in rows["data"]]
        axes[0].plot(rows["round_number"], accuracies, marker="o", label=label, color=METHOD_COLORS.get(method))
        if method == "fairrfl":
            rows = data[data["event"] == "fairrfl_detection"]
            values = [json.loads(value) for value in rows["data"]]
            for key, label in (("server_update_norm", "Global update"), ("median_local_update_norm", "Median local update")):
                if values and all(key in value for value in values):
                    axes[1].plot(rows["round_number"], [value[key] for value in values], marker="o", label=label)
    axes[0].set_ylabel("Global testing accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Diagnostic: observed rounds")
    axes[1].set_ylabel("Parameter update L2 norm")
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_title("Diagnostic: FairRFL update sizes")
    for ax in axes:
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# The summary cell passes seed means and sample deviations here to print Table II in the setup's method order.
def format_seed_summary(means, deviations):
    order = [method for method in METHOD_NAMES if method in means.index]
    order += sorted(set(means.index) - set(order))
    table = pd.DataFrame(index=[METHOD_NAMES.get(method, method) for method in order])
    for column in means.columns:
        digits = 4 if column == "JFI" else 2
        table[column] = [
            f"{means.loc[method, column]:.{digits}f} ± {deviations.loc[method, column]:.{digits}f}"
            for method in order
        ]
    table.index.name = "Method"
    return table
