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
import pandas as pd
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize

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

def plot_probe_accuracies_from_csv(
    results_csv_path,
    baseline_csv_path,
    save_dir="figures/probe_accuracy",
    filename="linear_probe_accuracy_per_layer.png",
    dataset_specifier="realistic"
):
    """
    Plot train and test accuracy per layer, configured for a 2-column LaTeX 
    layout with the legend placed above the plot to avoid obstructing data.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- PUBLICATION FORMATTING ---
    plt.rcParams.update({
        'font.size': 9,    # 11      
        'axes.labelsize': 9,   #11  
        'xtick.labelsize': 8,   #10
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (3.5, 2.5), 
        'figure.dpi': 300         
    })
    # ------------------------------

    # Load both CSV files
    df_results = pd.read_csv(results_csv_path)
    df_baseline = pd.read_csv(baseline_csv_path)

    # Use the DataFrame index as the layer numbers
    layers = df_results.index.tolist()

    # Extract accuracies
    train_accs = df_results["train"].tolist()
    test_accs = df_results["test"].tolist()
    
    train_baseline_accs = df_baseline["train"].tolist()
    test_baseline_accs = df_baseline["test"].tolist()

    plt.figure()

    # Plot main accuracy curves (solid lines, thicker for visibility)
    train_line = plt.plot(layers, train_accs, label="Train acc", linestyle="-", linewidth=1.5)
    test_line = plt.plot(layers, test_accs, label="Test acc", linestyle="-", linewidth=1.5)

    # Capture colors to match main lines with baseline lines
    train_color = train_line[0].get_color()
    test_color = test_line[0].get_color()

    # Plot baseline curves (dotted lines, matching colors)
    plt.plot(layers, train_baseline_accs, color=train_color, linestyle=":", linewidth=1.5, label="Train base")
    plt.plot(layers, test_baseline_accs, color=test_color, linestyle=":", linewidth=1.5, label="Test base")

    # Set axis labels and fixed y-limit
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.ylim(0.2, 1.1) # Adjusted slightly based on your image to give breathing room at the bottom
    
    # Hide the 1.1 tick 
    current_ticks = plt.gca().get_yticks()
    filtered_ticks = [tick for tick in current_ticks if tick < 1.05]
    plt.yticks(filtered_ticks)
    
    # Place legend outside, centered above the plot, in 2 columns
    plt.legend(
        loc="lower center", 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=2, 
        frameon=False, # Removing the box makes it look cleaner when placed outside
        columnspacing=1.0
    ) 

    # Tight layout (pad handles standard spacing)
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    
    # bbox_inches="tight" is CRITICAL here so the external legend isn't cropped out
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    plt.rcdefaults() 

    print(f"Saved plot to {save_path}")


def load_matrix(file_paths, metric="prob_gap"):
    layers = None
    all_rows = []

    for fp in file_paths:
        df = pd.read_csv(fp)

        if layers is None:
            layers = df["layer"].values

        all_rows.append(df[metric].values)

    return layers, np.stack(all_rows, axis=0)  # (files, layers)

def plot_grouped_heatmap(
    file_paths,
    id,
    contr_id,
    group_size=3,
    metric="prob_gap",
    save_path="figures/prob_gap_heatmap.png",
    figsize=(10, 6),
):

    layers, matrix = load_matrix(file_paths, metric)

    n_files, n_layers = matrix.shape

    small_labels = ["letters", "tokenizer", "common"]
    big_labels = ["letters", "tokenizer", "common", "correct"]

    # ---- build matrix with gaps ----
    new_matrix = []
    new_y_labels = []

    for i in range(n_files):
        new_matrix.append(matrix[i])

        # repeating within-group labels
        new_y_labels.append(small_labels[i % group_size])

        # gap between big groups
        if (i + 1) % group_size == 0 and (i + 1) < n_files:
            new_matrix.append(np.full(n_layers, np.nan))
            new_y_labels.append("")

    new_matrix = np.array(new_matrix)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)

    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.10)

    im = ax.imshow(
        new_matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest"
    )

    # ---- x axis ----
    clean_layers = [str(l).replace("layer_", "") for l in layers]

    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(clean_layers, rotation=90, fontsize=6)

    # ---- y axis (small labels only) ----
    ax.set_yticks(np.arange(len(new_y_labels)))
    ax.set_yticklabels(new_y_labels, fontsize=6)

    ax.set_xlabel("Layers")
    # ax.set_ylabel("Files")
    ax.set_title(f"Full probability shift steering from {id} to {contr_id}")

    # ---- BIG GROUP LABELS ----
    for g in range(n_files // group_size):
        center = g * (group_size + 0.7) + 1.5
        ax.text(
            -3.3,              # x-position (left of heatmap)
            center,            # y-position
            big_labels[g],
            va="center",
            ha="center",
            rotation=90,       
            fontsize=8,
            fontweight="bold"
        )

    plt.colorbar(im, ax=ax, label="probability")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

#     print(f"Saved heatmap to {save_path}")

# 

def load_matrix(file_paths, metric="probability"):
    layers = None
    all_rows = []

    for fp in file_paths:
        df = pd.read_csv(fp)

        if layers is None:
            layers = df["layer"].values

        all_rows.append(df[metric].values)

    return layers, np.stack(all_rows, axis=0)


def plot_heatmap(
    file_paths, id, contr_id,
    title="Probability per layer",
    metric="probability",
    labels=None,
    save_path="figures/probability_heatmap.png",
    figsize=(8, 5),
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    layers, matrix = load_matrix(file_paths, metric)

    if labels is None:
        labels = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]

    fig, ax = plt.subplots(figsize=figsize)

    norm = None
    if vmin is not None or vmax is not None:
        norm = Normalize(
            vmin=vmin if vmin is not None else np.min(matrix),
            vmax=vmax if vmax is not None else np.max(matrix),
        )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # x-axis: layers
    clean_layers = [str(l).replace("layer_", "") for l in layers]
    ax.set_xticks(np.arange(len(clean_layers)))
    ax.set_xticklabels(clean_layers, rotation=90)

    # y-axis: filenames (or custom labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Dataset")
    ax.set_title(f"{title} {id} to {contr_id} {metric}")

    plt.colorbar(im, ax=ax, label=metric)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to {save_path}")


def plot_single_steering_direction(
    csv_path,
    save_path,
    use_logprob=False
):
    """
    Plots a single steering direction graph from an input CSV file path,
    keeping the legend inside the plot and using compact publication fonts.
    """
    # Ensure the directory for the save path exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- PUBLICATION FORMATTING ---
    # Scaled down all font/label sizes slightly for a more compact text footprint
    plt.rcParams.update({
        'font.size': 9.5,          
        'axes.labelsize': 9.5,     
        'xtick.labelsize': 8.5,    
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,       
        'figure.figsize': (3.5, 2.5), 
        'figure.dpi': 300         
    })
    # ------------------------------

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Safely convert 'layer_X' string into integers for a precise X-axis mapping
    df['layer_num'] = df['layer'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values('layer_num')

    layer_nums = df['layer_num'].tolist()
    
    prefix = "log" if use_logprob else "prob"
    gap_vals = df[f"{prefix}_gap"].tolist()
    contr_vals = df[f"{prefix}_contr"].tolist()
    true_vals = df[f"{prefix}_true"].tolist()

    plt.figure()
    
    # Neutral zero baseline line placed behind data curves
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.8, zorder=1)

    # Plot data curves
    plt.plot(layer_nums, gap_vals, label="Full shift", linewidth=1.5, zorder=2)
    plt.plot(layer_nums, contr_vals, label=r"Prob $\uparrow$ target", linewidth=1.5, zorder=2)
    plt.plot(layer_nums, true_vals, label=r"Prob $\downarrow$ true", linewidth=1.5, zorder=2)

    # Fixed Y-axis limit (-0.1 to 0.6) and tight X-axis margins
    plt.ylim(-0.1, 0.6)
    plt.margins(x=0.02)

    # Explicit axis tick formatting
    plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    plt.xlabel("Layer")
    plt.ylabel("Average shift")
    
    # Legend placed inside the plot at the upper right whitespace area
    # framealpha adds a subtle semi-transparent background to keep text readable over zero lines
    plt.legend(
        loc="upper right", 
        frameon=True,
        framealpha=0.8,
        edgecolor="none"  # Removes the harsh black box border for a cleaner layout
    )

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    # Reset formatting defaults
    plt.rcdefaults()

    print(f"Saved plot to {save_path}")

plot_single_steering_direction("figures/old_data_resid_post_100.0/id_0_contr_id_2/avg_gap_alpha_100.0.csv",
                               "figures/old_data_resid_post_100.0/id_0_contr_id_2/avg_gap_alpha_100.0.png")

plot_single_steering_direction("figures/old_data_resid_post_100.0/id_1_contr_id_2/avg_gap_alpha_100.0.csv",
                               "figures/old_data_resid_post_100.0/id_1_contr_id_2/avg_gap_alpha_100.0.png")

plot_probe_accuracies_from_csv("accuracies/accuracies_old_data_resid_post/accuracies_old_data_resid_post.csv", 
                               "accuracies/accuracies_cont_baseline_resid_post_w_initial_embed/accuracies_cont_baseline_resid_post_w_initial_embed.csv", 
                               save_dir="figures/probe_accuracies_with_baseline/", 
                               filename="linear_probe_accuracy_cont_resid_post.png")

plot_probe_accuracies_from_csv("accuracies/accuracies_old_data_nocont_resid_post/accuracies_old_data_nocont_resid_post.csv", 
                               "accuracies/accuracies_nocont_baseline_resid_post_w_initial_embed/accuracies_nocont_baseline_resid_post_w_initial_embed.csv", 
                               save_dir="figures/probe_accuracies_with_baseline/", 
                               filename="linear_probe_accuracy_nocont_resid_post.png")


# for id in range(3):
# #     for contr_id in range(3):
# #         if id == contr_id:
# #             continue

# #         base = f"id_{id}_contr_id_{contr_id}"

#         # path1 = f"figures/letters_probe_letters_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path2 = f"figures/letters_probe_tokenizer_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path3 = f"figures/letters_probe_common_steering_100.0/{base}/avg_gap_alpha_100.0.csv"

#         # path4 = f"figures/tokenizer_probe_letters_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path5 = f"figures/tokenizer_probe_tokenizer_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path6 = f"figures/tokenizer_probe_common_steering_100.0/{base}/avg_gap_alpha_100.0.csv"

#         # path7 = f"figures/common_probe_letters_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path8 = f"figures/common_probe_tokenizer_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path9 = f"figures/common_probe_common_steering_100.0/{base}/avg_gap_alpha_100.0.csv"

#         # # path10 = f"figures/cont_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # # path11 = f"figures/cont_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # # path12 = f"figures/cont_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path10 = f"figures/onlycorrect_letters_probe_letters_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path11 = f"figures/onlycorrect_letters_probe_tokenizer_steering_100.0/{base}/avg_gap_alpha_100.0.csv"
#         # path12 = f"figures/onlycorrect_letters_probe_common_steering_100.0/{base}/avg_gap_alpha_100.0.csv"

#         path0 = f"figures/old_data_attn_out_100.0/{base}/avg_gap_alpha_100.0.csv"
#         path1 = f"figures/old_data_resid_mid_100.0/{base}/avg_gap_alpha_100.0.csv"
#         path2 = f"figures/old_data_mlp_out_100.0/{base}/avg_gap_alpha_100.0.csv"
#         path3 = f"figures/old_data_resid_post_100.0/{base}/avg_gap_alpha_100.0.csv"

#         file_paths = [path0, path1, path2, path3]
#         labels = ["attn_out", "resid_mid", "mlp_out", "resid_post"]
#         # metric = "prob_contr"
#         metric = "prob_gap"
#         save_path = f"figures/steering_heatmaps_compare_location/{base}_{metric}.png"


#         plot_heatmap(file_paths, id, contr_id, metric=metric, labels=labels, save_path=save_path, vmin=-0.05, vmax=0.3)
