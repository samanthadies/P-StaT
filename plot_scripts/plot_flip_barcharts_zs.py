"""
plot_flip_barcharts_zs.py

Compute and visualize how zero-shot predictions flip between tasks
(e.g., True -> Not True vs. Not True -> True) for strictly true statements.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec as grid_spec
from matplotlib.lines import Line2D


PERTURBATION_ORDER = ["synthetic", "fictional", "fictional_true", "noise"]
LABEL_ORDER = ["(a)", "(b)", "(c)", "(d)"]

# Map perturbation name -> task id in merged CSV columns
PERT_TO_TASK: Dict[str, int] = {
    "synthetic": 1,
    "fictional": 2,
    "fictional_true": 3,
    "noise": 4,
}

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

    :param pert: perturbation tag
    :return: pretty name
    """
    if pert == "synthetic":
        return "Synthetic"
    if pert == "fictional":
        return "Fictional"
    if pert == "fictional_true":
        return "Fictional (True)"
    if pert == "noise":
        return "Noise"
    return pert


def _discover_models_from_columns(df, baseline_task=0):
    """
    Discover model names from columns like "<model>_task0_pred".

    :param df: merged DF
    :param baseline_task: baseline task id (0)
    :return: sorted list of model names
    """
    suffix = f"_task{baseline_task}_pred"
    models = []
    for c in df.columns:
        if c.endswith(suffix):
            models.append(c[: -len(suffix)])
    return sorted(set(models))


def compute_flip_counts_rates(baseline_preds, perturbed_preds, true_label=1):
    """
    Returns (true_to_not_true_count, not_true_to_true_count, denom),
    where denom is the number of valid paired items.

    IMPORTANT:
    - In merged zero-shot outputs, preds are typically 0/1 with 1 meaning "True".
      (Because they come from y_hat=[p_neg,p_pos] and threshold p_pos>=0.5.)

    :param baseline_preds: predictions under baseline condition (0/1)
    :param perturbed_preds: predictions under perturbed condition (0/1)
    :param true_label: which integer represents "True" (default 1)
    :return: (t_to_nt, nt_to_t, denom)
    """
    baseline_preds = np.asarray(baseline_preds)
    perturbed_preds = np.asarray(perturbed_preds)

    if baseline_preds.shape != perturbed_preds.shape:
        raise ValueError(
            f"Shape mismatch baseline {baseline_preds.shape} vs perturbed {perturbed_preds.shape}"
        )

    keep = np.isfinite(baseline_preds) & np.isfinite(perturbed_preds)

    base_true = (baseline_preds == true_label)
    pert_true = (perturbed_preds == true_label)

    t_to_nt = int(np.sum(keep & base_true & ~pert_true))
    nt_to_t = int(np.sum(keep & ~base_true & pert_true))
    denom = int(np.sum(keep))

    return t_to_nt, nt_to_t, denom


def build_stats_df_for_perturbation(merged_df, models, task_id, baseline_task=0, true_label=1):
    """
    For a given perturbation (task_id), compute per-LLM flip counts and rates.

    Expects columns:
        <model>_task{baseline_task}_pred
        <model>_task{task_id}_pred

    :param merged_df: merged DF
    :param models: list of model names
    :param task_id: perturbation task id (1/2/3/4)
    :param baseline_task: baseline task id (0)
    :param true_label: which integer represents "True" (default 1)
    :return: stats DF
    """
    rows = []
    for m in models:
        base_col = f"{m}_task{baseline_task}_pred"
        pert_col = f"{m}_task{task_id}_pred"

        if base_col not in merged_df.columns or pert_col not in merged_df.columns:
            continue

        base_preds = merged_df[base_col].to_numpy()
        pert_preds = merged_df[pert_col].to_numpy()

        try:
            left_count, right_count, denom = compute_flip_counts_rates(
                base_preds, pert_preds, true_label=true_label
            )
            left_rate = left_count / denom if denom else np.nan
            right_rate = right_count / denom if denom else np.nan
        except Exception as e:
            print(f"[WARN] Skipping model={m}, task={task_id} due to error: {e}")
            continue

        rows.append(
            {
                "model": m,
                "left_count": left_count,
                "right_count": right_count,
                "left_rate": left_rate,
                "right_rate": right_rate,
                "n_items": denom,
                "task_id": task_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("model").reset_index(drop=True)


def plot_zero_shot_flip_barchart(dataset, merged_root, output_dir, baseline_tag="default",
                                 perturbations=PERTURBATION_ORDER, probe_name="zero_shot", true_label=1):
    """
    Plot a grid of bar charts showing label flips across tasks for a dataset.

    Reads:
        <merged_root>/<probe_name>/<dataset>_merged_<baseline_tag>.csv

    Produces:
        <output_dir>/zero_shot_flip_counts_grid_<dataset>_<baseline_tag>.pdf

    :param dataset: dataset tag (e.g., 'cities_loc')
    :param merged_root: root directory containing merged CSVs (cfg.paths.merged_root)
    :param output_dir: output directory for plots
    :param baseline_tag: "default" or "alt_baseline"
    :param perturbations: perturbation names to plot (e.g., ["synthetic","fictional","fictional_true"])
    :param probe_name: probe name dir ("zero_shot")
    :param true_label: which integer represents "True" in preds (default 1)
    :return: None
    """
    merged_dir = Path(merged_root) / probe_name
    merged_fp = merged_dir / f"{dataset}_merged_{baseline_tag}.csv"
    if not merged_fp.exists():
        raise FileNotFoundError(f"Merged zero-shot CSV not found: {merged_fp}")

    merged_df = pd.read_csv(merged_fp)
    models = _discover_models_from_columns(merged_df, baseline_task=0)
    if not models:
        raise RuntimeError(f"No models found in merged CSV: {merged_fp}")

    stats_by_pert: Dict[str, pd.DataFrame] = {}
    global_max_count = 0.0

    for pert in perturbations:
        if pert not in PERT_TO_TASK:
            print(f"[WARN] Unknown perturbation '{pert}' (no task mapping); skipping.")
            stats_by_pert[pert] = pd.DataFrame()
            continue

        task_id = PERT_TO_TASK[pert]
        df_stats = build_stats_df_for_perturbation(
            merged_df=merged_df,
            models=models,
            task_id=task_id,
            baseline_task=0,
            true_label=true_label,
        )
        stats_by_pert[pert] = df_stats

        if not df_stats.empty:
            local_max = np.nanmax(df_stats[["left_count", "right_count"]].to_numpy())
            if np.isfinite(local_max):
                global_max_count = max(global_max_count, float(local_max))

    if global_max_count <= 0:
        print(f"[INFO] No nonzero flip counts for dataset={dataset}, baseline_tag={baseline_tag}; skipping plot.")
        return

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=4,
        ncols=2,
        height_ratios=[0.09, 0.45, 0.45, 0.08],
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

    for idx in range(4):
        ax = subplot_for_index(idx)

        if idx >= len(perturbations):
            ax.set_axis_off()
            continue

        pert = perturbations[idx]
        df_stats = stats_by_pert.get(pert, pd.DataFrame())
        ax2 = ax.twinx()

        if df_stats.empty:
            ax.set_axis_off()
            ax.text(
                0.0,
                1.0,
                f"{LABEL_ORDER[idx]} {pert_pretty_name(pert)}\n(no data)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#555555",
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
            -0.18,
            1.15,
            panel_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )

    ax_title.set_axis_off()
    title = f"{dataset_pretty_name(dataset)}: Zero-shot Belief Flips by Model"
    ax_title.text(
        -0.08,
        0.70,
        title,
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=COLOR_LEFT,
            markeredgecolor="none",
            markersize=7,
            label="True to Not True",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=COLOR_RIGHT,
            markeredgecolor="none",
            markersize=7,
            label="Not True to True",
        ),
    ]
    ax_legend.set_axis_off()
    ax_legend.legend(handles=legend_handles, loc="center", ncol=2, frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"zero_shot_flip_counts_grid_{dataset}_{baseline_tag}.pdf"
    fp = output_dir / out_name
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {fp}")


def main():
    datasets = ["cities_loc", "med_indications", "defs"]

    merged_root = Path("outputs/merged")
    output_dir = Path("outputs/plots")

    for dataset in datasets:
        for baseline_tag in ["default", "alt_baseline"]:
            plot_zero_shot_flip_barchart(
                dataset=dataset,
                merged_root=merged_root,
                output_dir=output_dir,
                baseline_tag=baseline_tag,
                perturbations=PERTURBATION_ORDER,
            )


if __name__ == "__main__":
    main()