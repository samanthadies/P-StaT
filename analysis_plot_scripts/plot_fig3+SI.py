"""
plot_fig3_si.py

Create Figure 3 and SI robustness plots from combined prediction tables.

The script:
  1. Loads combined probing and zero-shot prediction CSVs.
  2. Computes model-level expansion and retraction rates directly from predictions.
  3. Aggregates rates across models using bootstrapped median confidence intervals.
  4. Creates Figure 3 and SI robustness figures.
  5. Writes figures and plotting data to outputs/plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS = ["cities_loc", "med_indications", "defs"]

DATASET_LABELS = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}

METHODS_MAIN = ["zero_shot", "mean_diff", "sAwMIL"]

METHOD_LABELS = {
    "zero_shot": "Behavioral",
    "mean_diff": "Mass-Mean",
    "sAwMIL": "Representational",
}

MODELS = [
    "_gemma-2-9b", "_gemma-3-27b", "_gemma-7b",
    "_llama-3-8b-med", "_llama-3.1-8b", "_llama-3.1-8b-bio",
    "_llama-3.2-3b", "_llama-3.3-70b", "_mistral-7B-v0.3",
    "_qwen-2.5-14b", "_qwen-2.5-72b", "_qwen-2.5-7b",
    "gemma-2-9b", "gemma-3-27b", "gemma-7b",
    "llama-3-8b", "llama-3.2-3b", "mistral-7B-v0.3",
    "qwen-2.5-14b", "qwen-2.5-72b", "qwen-2.5-7b",
]

TASKS = {
    1: "Synthetic (TF)",
    2: "Synthetic (Fi)",
    3: "Fictional",
    4: "Fictional (T)",
    5: "Noise",
}

PERTURBATION_ORDER = [
    "Synthetic (TF)",
    "Synthetic (Fi)",
    "Fictional",
    "Fictional (T)",
    "Noise",
]

CONDITION_COLORS = {
    "Synthetic (TF)": "#d8b371",
    "Synthetic (Fi)": "#ebce9b",
    "Fictional": "#7d8c9d",
    "Fictional (T)": "#b8ecff",
    "Noise": "#cf859a",
}

LAYER_LABELS = {
    "tasklayerminus2": r"$\ell-2$",
    "tasklayerminus1": r"$\ell-1$",
    "tasklayerplus1": r"$\ell+1$",
    "tasklayerplus2": r"$\ell+2$",
}


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def get_probe_col_fn(layer_label: str):
    """Return function that constructs probing prediction column names."""
    return lambda model, task: f"{model}__task_{task}__layer_{layer_label}__y_hat"


def get_zero_shot_col_fn(k: int):
    """Return function that constructs zero-shot prediction column names."""
    return lambda model, task: f"{model}__task_{task}__K_{k}__y_hat"


def get_probe_path(
    input_root: Path,
    method: str,
    dataset: str,
    layer_label: str,
) -> Path:
    """Return combined probing prediction CSV path."""
    return (
        input_root
        / method
        / dataset
        / f"{method}_{dataset}_test_predictions_{layer_label}.csv"
    )


def get_zero_shot_path(
    input_root: Path,
    dataset: str,
    k: int,
) -> Path:
    """Return combined zero-shot prediction CSV path."""
    return (
        input_root
        / "zero_shot"
        / dataset
        / f"zero_shot_{dataset}_test_predictions_K-{k}.csv"
    )


def compute_model_perturbation_stats(
    df: pd.DataFrame,
    model: str,
    task: int,
    perturbation: str,
    col_fn,
    base_task: int = 0,
):
    """Compute expansion/retraction rates for one model and perturbation."""
    base_col = col_fn(model, base_task)
    pert_col = col_fn(model, task)

    if base_col not in df.columns or pert_col not in df.columns:
        return None

    base = pd.to_numeric(df[base_col], errors="coerce")
    pert = pd.to_numeric(df[pert_col], errors="coerce")
    valid = base.isin([0, 1]) & pert.isin([0, 1])

    if valid.sum() == 0:
        return None

    b = base[valid].astype(int)
    p = pert[valid].astype(int)

    baseline_true = b.eq(1)
    baseline_false = b.eq(0)

    expansions = int((baseline_false & p.eq(1)).sum())
    retractions = int((baseline_true & p.eq(0)).sum())

    n_base_true = int(baseline_true.sum())
    n_base_false = int(baseline_false.sum())

    return {
        "model": model,
        "perturbation": perturbation,
        "expansion_rate": expansions / n_base_false if n_base_false else np.nan,
        "retraction_rate": retractions / n_base_true if n_base_true else np.nan,
    }


def build_stats_for_setting(
    input_root: Path,
    methods: list[str],
    datasets: list[str],
    metric_setting: str,
    layer_label: str = "tasklayer",
    k: int = 100,
    zero_shot_base_task: int = 0,
) -> pd.DataFrame:
    """Compute model-level rates directly from combined prediction CSVs."""
    rows = []

    for method in methods:
        for dataset in datasets:
            if method in {"mean_diff", "sAwMIL"}:
                path = get_probe_path(
                    input_root=input_root,
                    method=method,
                    dataset=dataset,
                    layer_label=layer_label,
                )
                tasks = TASKS
                col_fn = get_probe_col_fn(layer_label)
                base_task = 0

            elif method == "zero_shot":
                path = get_zero_shot_path(
                    input_root=input_root,
                    dataset=dataset,
                    k=k,
                )
                tasks = TASKS
                col_fn = get_zero_shot_col_fn(k)
                base_task = zero_shot_base_task

            else:
                raise ValueError(f"Unknown method: {method}")

            if not path.exists():
                raise FileNotFoundError(f"Could not find prediction table: {path}")

            print(f"Loading: {path}")
            df = pd.read_csv(path)

            for model in MODELS:
                for task, perturbation in tasks.items():
                    stats = compute_model_perturbation_stats(
                        df=df,
                        model=model,
                        task=task,
                        perturbation=perturbation,
                        col_fn=col_fn,
                        base_task=base_task,
                    )

                    if stats is None:
                        continue

                    stats["method"] = method
                    stats["dataset"] = dataset
                    stats["metric_setting"] = metric_setting
                    rows.append(stats)

    return pd.DataFrame(rows)


def bootstrap_median_ci(
    vals,
    n_boot: int = 5000,
    ci: int = 95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrapped median and confidence interval."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    if len(vals) == 1:
        return vals[0], vals[0], vals[0]

    rng = np.random.default_rng(random_state)
    boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
    meds = np.median(boot, axis=1)

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(meds, [alpha, 100 - alpha])

    return float(np.median(vals)), float(lo), float(hi)


def summarize_median_ci(stats_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Summarize model-level rates by method, dataset, and perturbation."""
    rows = []

    for method in stats_df["method"].dropna().unique():
        for dataset in DATASETS:
            for perturbation in PERTURBATION_ORDER:
                g = stats_df[
                    stats_df["method"].eq(method)
                    & stats_df["dataset"].eq(dataset)
                    & stats_df["perturbation"].eq(perturbation)
                ]

                vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()

                if len(vals) == 0:
                    continue

                med, lo, hi = bootstrap_median_ci(vals)

                rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "perturbation": perturbation,
                        "median": med,
                        "lo": lo,
                        "hi": hi,
                        "n_models": len(vals),
                    }
                )

    return pd.DataFrame(rows)


def style_axis(ax) -> None:
    """Apply shared axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)


def add_legend(fig, y: float = 0.02) -> None:
    """Add shared perturbation legend."""
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CONDITION_COLORS[p])
        for p in PERTURBATION_ORDER
    ]

    fig.legend(
        handles,
        PERTURBATION_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=5,
        frameon=False,
        fontsize=8,
    )


def plot_bar_panel(
    ax,
    summary_df: pd.DataFrame,
    method: str,
    dataset: str,
    metric_label: str,
    panel_label: str = None,
    title: str = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Plot one median-rate bar panel."""
    ddf = summary_df[
        summary_df["method"].eq(method)
        & summary_df["dataset"].eq(dataset)
    ]

    x = np.arange(len(PERTURBATION_ORDER))
    heights = []
    yerr_low = []
    yerr_high = []

    for perturbation in PERTURBATION_ORDER:
        row = ddf[ddf["perturbation"].eq(perturbation)]

        if row.empty:
            heights.append(np.nan)
            yerr_low.append(0)
            yerr_high.append(0)
            continue

        row = row.iloc[0]
        heights.append(row["median"])
        yerr_low.append(row["median"] - row["lo"])
        yerr_high.append(row["hi"] - row["median"])

    colors = [CONDITION_COLORS[p] for p in PERTURBATION_ORDER]

    err_scale = 1.0
    yerr_low = [e * err_scale for e in yerr_low]
    yerr_high = [e * err_scale for e in yerr_high]

    ax.bar(
        x,
        heights,
        yerr=np.vstack([yerr_low, yerr_high]),
        capsize=2,
        color=colors,
        edgecolor="none",
        width=0.75,
    )

    ax.set_xticks(x)

    if show_xlabel:
        ax.set_xticklabels(PERTURBATION_ORDER, rotation=25, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel(metric_label, fontsize=7)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    style_axis(ax)

    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)

    if title:
        ax.set_title(
            title,
            fontsize=9,
            fontweight="bold",
            color="#444444",
            pad=10,
        )

    if panel_label:
        ax.text(
            0.02,
            0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#444444",
        )


def plot_main_grid(
    stats_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    out_stem: str,
    title: str,
) -> None:
    """Create main-text two-method grid."""
    summary = summarize_median_ci(stats_df, metric)
    summary.to_csv(output_dir / f"{out_stem}_summary.csv", index=False)

    print("\n" + "=" * 80)
    print(f"Figure medians: {out_stem} ({metric})")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")

    metric_label = (
        r"Retraction Rate $\rho_{\mathcal{R}}$"
        if metric == "retraction_rate"
        else r"Expansion Rate $\rho_{\mathcal{E}}$"
    )

    methods = ["zero_shot", "sAwMIL"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(7.2, 3.8),
        sharex=False,
        sharey=False,
    )

    panel_letters = list("abcdef")

    row_ymax = {}
    for method in methods:
        vals = summary[summary["method"].eq(method)]["hi"].dropna()
        ymax = vals.max() if len(vals) else 1.0
        row_ymax[method] = ymax * 1.15 if ymax > 0 else 1.0

    for i, method in enumerate(methods):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]

            plot_bar_panel(
                ax=ax,
                summary_df=summary,
                method=method,
                dataset=dataset,
                metric_label=metric_label,
                panel_label=f"({panel_letters[i * 3 + j]})",
                title=DATASET_LABELS[dataset] if i == 0 else None,
                show_xlabel=(i == len(methods) - 1),
                show_ylabel=(j == 0),
            )

            ax.set_ylim(0, row_ymax[method])

            if j > 0:
                ax.tick_params(axis="y", length=0)

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.3, 0.5)

    y_mids = [0.73, 0.40]

    for i, method in enumerate(methods):
        fig.text(
            0.035,
            y_mids[i],
            METHOD_LABELS[method],
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444444",
        )

    fig.text(
        0.01,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    add_legend(fig, y=0.05)
    fig.tight_layout(rect=[0.055, 0.09, 1, 0.95], h_pad=0.8, w_pad=0.7)

    for ext in ["png", "pdf"]:
        path = output_dir / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote: {path}")

    plt.close(fig)


def plot_mass_mean_si_figure(
    main_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create SI Mass-Mean retraction-rate figure."""
    summary = summarize_median_ci(main_stats, "retraction_rate")
    summary.to_csv(output_dir / "si_mass_mean_retraction_rates_summary.csv", index=False)

    method = "mean_diff"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=False)
    panel_letters = list("abc")

    row_vals = summary[summary["method"].eq(method)]["hi"].dropna()
    row_ymax = row_vals.max() * 1.25 if len(row_vals) and row_vals.max() > 0 else 1.0

    for j, dataset in enumerate(DATASETS):
        ax = axes[j]

        plot_bar_panel(
            ax=ax,
            summary_df=summary,
            method=method,
            dataset=dataset,
            metric_label=r"Retraction Rate $\rho_{\mathcal{R}}$",
            panel_label=f"({panel_letters[j]})",
            title=DATASET_LABELS[dataset],
            show_xlabel=True,
            show_ylabel=(j == 0),
        )

        ax.set_ylim(0, row_ymax)

        if j == 0:
            ax.yaxis.set_label_coords(-0.35, 0.5)
        else:
            ax.tick_params(axis="y", length=0)

    fig.text(
        0.01,
        0.98,
        "Mass-Mean retraction rates",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    add_legend(fig, y=0.05)
    fig.tight_layout(rect=[0, 0.18, 1, 0.90], h_pad=0.8, w_pad=0.8)

    for ext in ["png", "pdf"]:
        path = output_dir / f"si_mass_mean_retraction_rates.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote: {path}")

    plt.close(fig)


def plot_sawmil_layer_robustness_combined(
    input_root: Path,
    output_dir: Path,
) -> None:
    """Create SI layer-robustness figure for sAwMIL."""
    layer_order = [
        "tasklayerminus2",
        "tasklayerminus1",
        "tasklayerplus1",
        "tasklayerplus2",
    ]

    all_stats = []

    for layer_label in layer_order:
        stats = build_stats_for_setting(
            input_root=input_root,
            methods=["sAwMIL"],
            datasets=DATASETS,
            metric_setting=layer_label,
            layer_label=layer_label,
        )
        stats["layer_label"] = layer_label
        all_stats.append(stats)

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(output_dir / "si_sawmil_layer_robustness_model_rates.csv", index=False)

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.0), sharex=False, sharey=False)
    panel_letters = list("abcdefghijkl")

    for i, layer_label in enumerate(layer_order):
        sub = stats_df[stats_df["layer_label"].eq(layer_label)]
        summary = summarize_median_ci(sub, "retraction_rate")

        row_vals = summary["hi"].dropna()
        row_ymax = row_vals.max() * 1.25 if len(row_vals) and row_vals.max() > 0 else 1.0

        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]

            plot_bar_panel(
                ax=ax,
                summary_df=summary,
                method="sAwMIL",
                dataset=dataset,
                metric_label=r"Retraction Rate $\rho_{\mathcal{R}}$",
                panel_label=f"({panel_letters[i * 3 + j]})",
                title=DATASET_LABELS[dataset] if i == 0 else None,
                show_xlabel=(i == len(layer_order) - 1),
                show_ylabel=(j == 0),
            )

            ax.set_ylim(0, row_ymax)

            if j > 0:
                ax.tick_params(axis="y", length=0)

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.35, 0.5)

    y_mids = [0.80, 0.62, 0.44, 0.26]

    for i, layer_label in enumerate(layer_order):
        fig.text(
            0.05,
            y_mids[i],
            LAYER_LABELS[layer_label],
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444444",
        )

    fig.text(
        0.01,
        0.985,
        r"sAwMIL: Robustness of Retraction Rates to Layer",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    add_legend(fig, y=0.01)
    fig.tight_layout(rect=[0.055, 0.08, 1, 0.95], h_pad=0.9, w_pad=0.7)

    for ext in ["png", "pdf"]:
        path = output_dir / f"si_sawmil_layer_robustness_combined.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote: {path}")

    plt.close(fig)


def plot_zero_shot_robustness_combined(
    input_root: Path,
    output_dir: Path,
) -> None:
    """Create SI context-size and baseline robustness figure for zero-shot."""
    settings = [
        (100, 0, r"Baseline" + "\n" + r"$K=100$"),
        (50, 0, r"Baseline" + "\n" + r"$K=50$"),
        (50, 5, r"Alt. baseline" + "\n" + r"$K=50$"),
        (20, 0, r"Baseline" + "\n" + r"$K=20$"),
        (20, 5, r"Alt. baseline" + "\n" + r"$K=20$"),
    ]

    all_stats = []

    for k, base_task, setting_label in settings:
        stats = build_stats_for_setting(
            input_root=input_root,
            methods=["zero_shot"],
            datasets=DATASETS,
            metric_setting=f"K{k}_base{base_task}",
            k=k,
            zero_shot_base_task=base_task,
        )
        stats["setting_label"] = setting_label
        all_stats.append(stats)

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(output_dir / "si_zero_shot_robustness_model_rates.csv", index=False)

    fig, axes = plt.subplots(5, 3, figsize=(7.2, 6.8), sharex=False, sharey=False)
    panel_letters = list("abcdefghijklmno")

    for i, (_, _, setting_label) in enumerate(settings):
        sub = stats_df[stats_df["setting_label"].eq(setting_label)]
        summary = summarize_median_ci(sub, "retraction_rate")

        row_vals = summary["hi"].dropna()
        row_ymax = row_vals.max() * 1.25 if len(row_vals) and row_vals.max() > 0 else 1.0

        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]

            plot_bar_panel(
                ax=ax,
                summary_df=summary,
                method="zero_shot",
                dataset=dataset,
                metric_label=r"Retraction Rate $\rho_{\mathcal{R}}$",
                panel_label=f"({panel_letters[i * 3 + j]})",
                title=DATASET_LABELS[dataset] if i == 0 else None,
                show_xlabel=(i == len(settings) - 1),
                show_ylabel=(j == 0),
            )

            ax.set_ylim(0, row_ymax)

            if j > 0:
                ax.tick_params(axis="y", length=0)

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.3, 0.5)

    y_mids = [0.83, 0.69, 0.53, 0.38, 0.22]

    for i, (_, _, setting_label) in enumerate(settings):
        fig.text(
            0.04,
            y_mids[i],
            setting_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

    fig.text(
        0.01,
        0.985,
        r"Behavioral Belief Context Robustness",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    add_legend(fig, y=0.05)
    fig.tight_layout(rect=[0.060, 0.075, 1, 0.95], h_pad=0.9, w_pad=0.7)

    for ext in ["png", "pdf"]:
        path = output_dir / f"si_zero_shot_robustness_combined.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote: {path}")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        default=str(project_path("outputs", "analysis_data")),
        help="Root containing combined prediction CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_path("outputs", "plots")),
    )
    parser.add_argument(
        "--zero-shot-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--main-layer-label",
        default="tasklayer",
    )

    return parser.parse_args()


def main() -> None:
    """Create Figure 3 and SI figures."""
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_stats = build_stats_for_setting(
        input_root=input_root,
        methods=METHODS_MAIN,
        datasets=DATASETS,
        metric_setting="main",
        layer_label=args.main_layer_label,
        k=args.zero_shot_k,
        zero_shot_base_task=0,
    )

    main_stats.to_csv(output_dir / "fig3_main_setting_model_level_rates.csv", index=False)

    plot_main_grid(
        stats_df=main_stats,
        metric="retraction_rate",
        output_dir=output_dir,
        out_stem="fig3_retraction_rates",
        title="Retraction rates under familiar and unfamiliar perturbations",
    )

    plot_main_grid(
        stats_df=main_stats,
        metric="expansion_rate",
        output_dir=output_dir,
        out_stem="si_expansion_rates",
        title="Expansion rates under familiar and unfamiliar perturbations",
    )

    plot_mass_mean_si_figure(
        main_stats=main_stats,
        output_dir=output_dir,
    )

    plot_sawmil_layer_robustness_combined(
        input_root=input_root,
        output_dir=output_dir,
    )

    plot_zero_shot_robustness_combined(
        input_root=input_root,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()