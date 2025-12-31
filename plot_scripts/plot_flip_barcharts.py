"""
plot_flip_barcharts.py

Compute and visualize how model predictions flip between tasks
(e.g., True to Not True vs. Not True to True) for strictly true statements.

"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec as grid_spec
from matplotlib.lines import Line2D


def strict_true_mask(df):
    """
    Build a boolean mask for rows that are strictly labeled as true.
    This uses the 'correct' column, treating value 1 as true.

    :param df: pandas DataFrame with a 'correct' column
    :return: numpy boolean array mask
    """
    t = (df.get("correct", 0) == 1).to_numpy()
    return t


def discover_models(df, task=0):
    """
    Discover model names based on prediction column patterns.
    Looks for columns named like '{model}_task{task}_pred' and returns
    the set of model name prefixes.

    :param df: pandas DataFrame with prediction columns
    :param task: integer task id used in the column name pattern
    :return: sorted list of unique model name strings
    """
    pat = re.compile(rf"^(.+)_task{task}_pred$")
    models = []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            models.append(m.group(1))

    return sorted(set(models))


def get_pred_df(df, models, task):
    """
    Extract a prediction DataFrame for a given task across models.

    For each model in `models`, this expects a column named
    '{model}_task{task}_pred' to be present in df.

    :param df: pandas DataFrame with prediction columns
    :param models: list of model name strings
    :param task: integer task id
    :return: DataFrame with one column per model containing predictions
    """
    cols = {f"{m}_task{task}_pred": m for m in models}
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(
            f"Missing pred columns for task {task}: "
            f"{miss[:5]}{'...' if len(miss) > 5 else ''}"
        )
    out = pd.DataFrame({model: df[f"{model}_task{task}_pred"].to_numpy() for model in models})
    return out


def compute_flip_counts_rates_df(df, models, pred_init, pred_new, init_task, new_task):
    """
    Compute flip counts and rates for each model between two tasks.

    For strictly true rows (based on strict_true_mask), counts how many
    predictions flip:
        - True (1) -> Not True (0)
        - Not True (0) -> True (1)

    :param df: pandas DataFrame with labels
    :param models: list of model names
    :param pred_init: DataFrame of initial task predictions (columns = models)
    :param pred_new: DataFrame of new task predictions (columns = models)
    :param init_task: initial task id
    :param new_task: new task id
    :return: DataFrame with flip counts, rates, and metadata per model
    """
    mask = strict_true_mask(df)

    rows = []
    for m in models:
        p0 = pred_init[m].to_numpy()
        p1 = pred_new[m].to_numpy()
        keep = mask & np.isfinite(p0) & np.isfinite(p1)

        tp_fn = int(np.sum(keep & (p0 == 1) & (p1 == 0)))
        fn_tp = int(np.sum(keep & (p0 == 0) & (p1 == 1)))
        denom = int(np.sum(keep))

        left_count = tp_fn
        right_count = fn_tp
        left_rate = tp_fn / denom if denom else np.nan
        right_rate = fn_tp / denom if denom else np.nan

        rows.append({
            "model": m,
            "left_count": left_count,
            "right_count": right_count,
            "left_rate": left_rate,
            "right_rate": right_rate,
            "n_items": denom,
            "init_task": init_task,
            "new_task": new_task,
        })

    return pd.DataFrame(rows).sort_values("model")


def task_short_label(task_id):
    """
    Return a human-readable short label for a given task id.

    :param task_id: integer task id
    :return: short descriptive string for that task
    """
    if task_id == 1:
        return 'Synthetic'
    if task_id == 2:
        return 'Fictional'
    if task_id == 3:
        return 'Fictional (True)'
    if task_id == 4:
        return 'Noise'
    return f'Task {task_id}'


def plot_flip_barchart(
    dataset,
    merged_root,
    output_fp,
    init_task,
    new_tasks,
    probe_name,
    noise,
):
    """
    Plot a grid of bar charts showing label flips across tasks for a dataset.

    For each new task in `new_tasks`, plotting:
        - True to Not True flips (counts, left bars)
        - Not True to True flips (counts, right bars)
    with a shared y-axis scale based on the global maximum count.

    :param dataset: dataset name (e.g., 'cities_loc')
    :param merged_root: root directory containing merged CSVs
    :param output_fp: output directory to save the PDF
    :param init_task: initial task id (int)
    :param new_tasks: list of new task ids (ints)
    :param probe_name: probe name used in merged file naming
    :param noise: noise level used in merged file naming
    :return: None
    """
    merged_fp = os.path.join(
        merged_root, probe_name, f"{dataset}_merged_y=true_noise{noise}.csv"
    )
    if not os.path.isfile(merged_fp):
        raise FileNotFoundError(f"Merged CSV not found: {merged_fp}")

    df = pd.read_csv(merged_fp)

    task_ids = [init_task] + list(new_tasks)
    model_sets = [set(discover_models(df, t)) for t in task_ids]
    common_models = sorted(set.intersection(*model_sets)) if model_sets else []
    if not common_models:
        raise RuntimeError(
            f"No overlapping models with tasks {task_ids} for dataset={dataset}"
        )

    models = common_models

    pred_init = get_pred_df(df, models, init_task)
    pred_new_map = {t: get_pred_df(df, models, t) for t in new_tasks}

    stats_by_task = {}
    global_max_count = 0

    for t in new_tasks:
        stats = compute_flip_counts_rates_df(
            df=df,
            models=models,
            pred_init=pred_init,
            pred_new=pred_new_map[t],
            init_task=init_task,
            new_task=t,
        )
        stats_by_task[t] = stats
        if not stats.empty:
            local_max = np.nanmax(
                stats[["left_count", "right_count"]].to_numpy()
            )
            if np.isfinite(local_max):
                global_max_count = max(global_max_count, float(local_max))

    if global_max_count == 0:
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

    label_order = ['(a)', '(b)', '(c)', '(d)']

    color_left = '#c2d58a'   # True -> Not True
    color_right = '#6d71b7'  # Not True -> True

    def subplot_for_index(idx):
        """
        Map subplot index (0–3) to a specific GridSpec cell.

        :param idx: integer subplot index
        :return: matplotlib Axes instance
        """
        if idx == 0:
            return fig.add_subplot(gs[1, 0])
        if idx == 1:
            return fig.add_subplot(gs[1, 1])
        if idx == 2:
            return fig.add_subplot(gs[2, 0])
        return fig.add_subplot(gs[2, 1])

    for idx, t in enumerate(new_tasks):
        stats = stats_by_task[t]
        if stats.empty:
            continue

        ax = subplot_for_index(idx)
        ax2 = ax.twinx()   # right y-axis (rates)

        x = np.arange(len(stats))
        width = 0.35

        # Counts bars (left axis)
        ax.bar(
            x - width / 2,
            stats["left_count"].to_numpy(),
            width=width,
            color=color_left,
            label="True to Not True" if idx == 0 else None,
        )
        ax.bar(
            x + width / 2,
            stats["right_count"].to_numpy(),
            width=width,
            color=color_right,
            label="Not True to True" if idx == 0 else None,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            stats["model"].tolist(),
            rotation=45,
            ha="right",
            fontsize=7,
        )

        ax.set_ylabel("Count")
        ax.set_ylim(0, global_max_count * 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        denoms = stats["n_items"].dropna().unique()
        if len(denoms) == 1 and denoms[0] > 0:
            denom = float(denoms[0])
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim(ymin / denom, ymax / denom)
            ax2.set_ylabel("Proportion", rotation=270, labelpad=15)
        else:
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel(
                "Proportion of Statements (approx.)",
                rotation=270,
                labelpad=10,
            )
        ax2.spines["top"].set_visible(False)

        short_lbl = task_short_label(t)
        subplot_text = f"{label_order[idx]} {short_lbl}"
        ax.text(
            -0.18,
            1.15,
            subplot_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )

    if dataset == 'cities_loc':
        pretty_name = 'City Locations'
    elif dataset == 'med_indications':
        pretty_name = 'Med. Indications'
    elif dataset == 'defs':
        pretty_name = 'Word Definitions'
    else:
        pretty_name = dataset

    ax_title.set_axis_off()
    title = rf"{pretty_name}: Changes in Predicted Labels by Model (Given $y=True$)"
    ax_title.text(
        -0.08,
        0.0,
        title,
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker='s',
            linestyle='none',
            markerfacecolor=color_left,
            markeredgecolor='none',
            markersize=7,
            label="True to Not True",
        ),
        Line2D(
            [0], [0],
            marker='s',
            linestyle='none',
            markerfacecolor=color_right,
            markeredgecolor='none',
            markersize=7,
            label="Not True to True",
        ),
    ]
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        frameon=False,
    )

    os.makedirs(output_fp, exist_ok=True)
    out_name = f"flip_counts_grid_{dataset}_{probe_name}_noise{noise}.pdf"
    fp = os.path.join(output_fp, out_name)
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():

    datasets = ['cities_loc', 'med_indications', 'defs']
    merged_root = 'outputs/analysis_data'
    output_fp = 'outputs/plots'
    init_task = 0
    noise = 10
    new_tasks = [1, 2, 3, 4]
    probe_name = 'sAwMIL'

    for dataset in datasets:
        plot_flip_barchart(
            dataset,
            merged_root,
            output_fp,
            init_task,
            new_tasks,
            probe_name,
            noise,
        )


if __name__ == '__main__':
    main()
