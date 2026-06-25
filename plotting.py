"""
This file contains functions for making several kinds of plots relating to the probing and steering experiments.
The file contains the following functions:
- plot_delta_logprob: Plots the change in log probability for each layer.
- plot_rank: Plots the change in rank for each layer.
- plot_average_gap: Plots the average gap between different types of shifts for each layer.
- plot_probe_accuracies: Plots the training and testing accuracies of the linear probes for each layer.
"""

import matplotlib.pyplot as plt
import os

def plot_delta_logprob(metrics_df, title=None):
    layer_rows = metrics_df.drop(index="baseline")

    layers = [int(i.split("_")[1]) for i in layer_rows.index]
    values = layer_rows["delta_log_prob"].values

    layers, values = zip(*sorted(zip(layers, values)))

    plt.figure(figsize=(6,4))
    plt.plot(layers, values, marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel("Δ log P(true)")
    if title:
        plt.title(title)
    plt.grid(True)

    base = f"figures/{title.replace(' ', '_')}"
    n = 0

    if not os.path.exists("figures"):
        os.makedirs("figures")

    while True:
        save_path = f"{base}-{n}.png"
        if not os.path.exists(save_path):
            break
        n += 1
    plt.savefig(save_path)

def plot_rank(metrics_df, title=None):
    layer_rows = metrics_df.drop(index="baseline")

    layers = [int(i.split("_")[1]) for i in layer_rows.index]
    ranks = layer_rows["rank"].values

    layers, ranks = zip(*sorted(zip(layers, ranks)))

    plt.figure(figsize=(6,4))
    plt.plot(layers, ranks, marker="o")
    plt.gca().invert_yaxis()
    plt.xlabel("Layer")
    plt.ylabel("Rank of true (lower is better)")
    if title:
        plt.title(title)
    plt.grid(True)

    base = f"figures/{title.replace(' ', '_')}"
    n = 0
    if not os.path.exists("figures"):
        os.makedirs("figures")
    while True:
        save_path = f"{base}-{n}.png"
        if not os.path.exists(save_path):
            break
        n += 1
    plt.savefig(save_path)


def plot_average_gap(
    averaged_results,
    id,
    contrastive_id,
    alpha,
    use_logprob=True,
    base_path="figures",
    dataset_specifier_fullname="realistic"
):
    key = (id, contrastive_id)

    if key not in averaged_results:
        return

    layers = sorted(
        averaged_results[key].keys(),
        key=lambda x: int(x.split("_")[1])
    )

    prefix = "log" if use_logprob else "prob"

    gap_vals = []
    contr_vals = []
    true_vals = []

    for layer in layers:
        gap_vals.append(averaged_results[key][layer][f"{prefix}_gap"])
        contr_vals.append(averaged_results[key][layer][f"{prefix}_contr"])
        true_vals.append(averaged_results[key][layer][f"{prefix}_true"])

    save_dir = os.path.join(
        base_path,
        f"id_{id}_contr_id_{contrastive_id}"
    )
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(gap_vals, label="Full shift")
    plt.plot(contr_vals, label="Contrastive shift")
    plt.plot(true_vals, label="True shift")
    plt.axhline(0)
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Average shift")
    # plt.title(
    #     f"Steering probabilities shift (id={id} to id={contrastive_id}) on {dataset_specifier_fullname}"
    # )

    save_path = os.path.join(
        save_dir,
        f"avg_decomposition_alpha_{alpha}_{prefix}_newplot2.png"
    )

    plt.savefig(save_path)
    plt.close()


def plot_probe_accuracies(
    results,
    save_dir="figures/probe_accuracy",
    filename="linear_probe_accuracy_per_layer.png",
    dataset_specifier="realistic"
):
    """
    Plot train and test accuracy per layer from probe_all_layers results.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Sort layers numerically
    layers = sorted(results.keys())

    train_accs = [results[layer]["train_acc"] for layer in layers]
    test_accs = [results[layer]["test_acc"] for layer in layers]

    plt.figure()

    plt.plot(layers, train_accs, label="Train accuracy")
    plt.plot(layers, test_accs, label="Test accuracy")

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    # plt.title(f"Linear probe accuracy per layer ({dataset_specifier})")
    plt.legend()

    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to {save_path}")