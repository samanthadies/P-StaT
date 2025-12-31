"""
plot_flip_barcharts_zs.py

Compute and visualize how zero-shot predictions flip between tasks
(e.g., True to Not True vs. Not True to True) for strictly true statements.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec as grid_spec
from matplotlib.lines import Line2D


# Only 3 perturbations exist for zero-shot; 4th panel left intentionally blank.
PERTURBATION_ORDER = ["synthetic", "fictional", "fictional_true"]
LABEL_ORDER = ["(a)", "(b)", "(c)"]

COLOR_LEFT = "#c2d58a"   # True -> Not True
COLOR_RIGHT = "#6d71b7"  # Not True -> True


def dataset_pretty_name(dataset):
    """
    Return the prettier name for the dataset.

    :param dataset: dataset tag
    :return: pretty name
    """
    if dataset == "cities_loc":
        return "City Locations"
    if dataset == "med_indications":
        return "Med. Indications"
    if dataset == "defs":
        return "Word Definitions"
    return dataset


def pert_pretty_name(pert):
    """
    Return the prettier name for the perturbation.

    :param pert: pertrubation tag
    :return: prettier name
    """
    if pert == "synthetic":
        return "Synthetic"
    if pert == "fictional":
        return "Fictional"
    if pert == "fictional_true":
        return "Fictional (True)"
    return pert


def safe_load_preds(path):
    """
    Load a 1D preds.npy array; return None if missing or invalid.

    :param path: path to npy array
    :return: preds.npy array
    """
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D preds array, got shape {arr.shape}")
        return arr
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def list_models_for_dataset(root, dataset):
    """
    Discover LLMs under root that have at least baseline preds for the given dataset.

    :param root: path to results
    :param dataset: dataset tag
    :return: list of LLMs
    """

    if not root.exists():
        raise FileNotFoundError(f"Zero-shot root not found: {root}")

    models = []
    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        base_path = model_dir / dataset / "none" / "preds.npy"
        if base_path.exists():
            models.append(model_dir.name)

    return sorted(models)


def compute_flip_counts_rates(baseline_preds, perturbed_preds):
    """
    Returns (true_to_not_true_count, not_true_to_true_count, denom), where denom is total statements.

    :param baseline_preds: predictions under baseline condition
    :param perturbed_preds: predictions under perturbed condition
    :return: results
    """

    if baseline_preds.shape != perturbed_preds.shape:
        raise ValueError(
            f"Shape mismatch baseline {baseline_preds.shape} vs perturbed {perturbed_preds.shape}"
        )

    keep = np.isfinite(baseline_preds) & np.isfinite(perturbed_preds)

    base_true = (baseline_preds == 0)
    pert_true = (perturbed_preds == 0)

    t_to_nt = int(np.sum(keep & base_true & ~pert_true))
    nt_to_t = int(np.sum(keep & ~base_true & pert_true))
    denom = int(np.sum(keep))

    return t_to_nt, nt_to_t, denom


def build_stats_df_for_perturbation(root, dataset, pert, models):
    """
    For a given dataset + perturbation, compute per-LLM flip counts and rates.

    :param root: path to results
    :param dataset: dataset tag
    :param pert: perturbation tag
    :param models: LLMs
    :return: dataframe with results
    """

    rows = []
    for m in models:
        model_dir = root / m / dataset
        base_preds = safe_load_preds(model_dir / "none" / "preds.npy")
        pert_preds = safe_load_preds(model_dir / pert / "preds.npy")

        if base_preds is None or pert_preds is None:
            continue

        try:
            left_count, right_count, denom = compute_flip_counts_rates(base_preds, pert_preds)
            left_rate = left_count / denom if denom else np.nan
            right_rate = right_count / denom if denom else np.nan
        except Exception as e:
            print(f"[WARN] Skipping model={m}, dataset={dataset}, pert={pert} due to error: {e}")
            continue

        rows.append(
            {
                "model": m,
                "left_count": left_count,
                "right_count": right_count,
                "left_rate": left_rate,
                "right_rate": right_rate,
                "n_items": denom,
                "perturbation_type": pert,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("model").reset_index(drop=True)


def plot_zero_shot_flip_barchart(dataset, zero_shot_root, output_dir, perturbations=PERTURBATION_ORDER):
    """
    Plot a grid of bar charts showing label flips across tasks for a dataset.

    For each new task in `new_tasks`, plotting:
        - True to Not True flips (counts, left bars)
        - Not True to True flips (counts, right bars)
    with a shared y-axis scale based on the global maximum count.

    :param dataset: dataset name (e.g., 'cities_loc')
    :param zero_shot_root: root directory containing merged CSVs
    :param output_dir: output directory to save the PDF
    :param perturbations: perturbation list
    :return: None
    """

    models = list_models_for_dataset(zero_shot_root, dataset)
    if not models:
        raise RuntimeError(f"No models found with baseline preds for dataset={dataset} under {zero_shot_root}")

    stats_by_pert = {}
    global_max_count = 0.0

    for pert in perturbations:
        df_stats = build_stats_df_for_perturbation(zero_shot_root, dataset, pert, models)
        stats_by_pert[pert] = df_stats

        if not df_stats.empty:
            local_max = np.nanmax(df_stats[["left_count", "right_count"]].to_numpy())
            if np.isfinite(local_max):
                global_max_count = max(global_max_count, float(local_max))

    if global_max_count <= 0:
        print(f"[INFO] No nonzero flip counts for dataset={dataset}; skipping plot.")
        return

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=4,
        ncols=2,
        height_ratios=[0.07, 0.45, 0.45, 0.08],
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_legend = fig.add_subplot(gs[3, :])

    def subplot_for_index(idx: int):
        if idx == 0:
            return fig.add_subplot(gs[1, 0])
        if idx == 1:
            return fig.add_subplot(gs[1, 1])
        if idx == 2:
            return fig.add_subplot(gs[2, 0])
        return fig.add_subplot(gs[2, 1])

    # Always render 4 slots; last one is blank by design.
    for idx in range(4):
        ax = subplot_for_index(idx)

        # Slot 3 (index 3) is intentionally blank (no noise perturbation).
        if idx >= len(perturbations):
            ax.set_axis_off()

            continue

        pert = perturbations[idx]
        df_stats = stats_by_pert.get(pert, pd.DataFrame())
        ax2 = ax.twinx()

        if df_stats.empty:
            ax.set_axis_off()
            ax.text(
                0.0, 1.0,
                f"{LABEL_ORDER[idx]} {pert_pretty_name(pert)}\n(no data)",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9, fontweight="bold", color="#555555",
            )
            continue

        x = np.arange(len(df_stats))
        width = 0.35

        ax.bar(
            x - width / 2,
            df_stats["left_count"].to_numpy(),
            width=width,
            color=COLOR_LEFT,
            label="True to Not True" if idx == 0 else None,
        )
        ax.bar(
            x + width / 2,
            df_stats["right_count"].to_numpy(),
            width=width,
            color=COLOR_RIGHT,
            label="Not True to True" if idx == 0 else None,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df_stats["model"].tolist(), rotation=45, ha="right", fontsize=7)

        ax.set_ylabel("Count")
        ax.set_ylim(0, global_max_count * 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        denoms = df_stats["n_items"].dropna().unique()
        if len(denoms) == 1 and denoms[0] > 0:
            denom = float(denoms[0])
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim(ymin / denom, ymax / denom)
            ax2.set_ylabel("Proportion", rotation=270, labelpad=15)
        else:
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("Proportion of Statements (approx.)", rotation=270, labelpad=10)
        ax2.spines["top"].set_visible(False)

        panel_txt = f"{LABEL_ORDER[idx]} {pert_pretty_name(pert)}"
        ax.text(
            -0.18, 1.15,
            panel_txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold", color="#555555",
        )

    ax_title.set_axis_off()
    title = f"{dataset_pretty_name(dataset)}: Zero-shot Belief Flips by Model"
    ax_title.text(
        -0.08, 0.0,
        title,
        va="center", ha="left",
        fontsize=12, fontweight="bold", color="#333333",
    )

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor=COLOR_LEFT, markeredgecolor="none",
               markersize=7, label="True to Not True"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor=COLOR_RIGHT, markeredgecolor="none",
               markersize=7, label="Not True to True"),
    ]
    ax_legend.set_axis_off()
    ax_legend.legend(handles=legend_handles, loc="center", ncol=2, frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"zero_shot_flip_counts_grid_{dataset}.pdf"
    fp = output_dir / out_name
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {fp}")


def main():
    datasets = ["cities_loc", "med_indications", "defs"]

    zero_shot_root = Path("outputs/probes/zero_shot")
    output_dir = Path("outputs/plots")

    for dataset in datasets:
        plot_zero_shot_flip_barchart(
            dataset=dataset,
            zero_shot_root=zero_shot_root,
            output_dir=output_dir,
            perturbations=PERTURBATION_ORDER,
        )


if __name__ == "__main__":
    main()
