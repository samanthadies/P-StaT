"""
plot_SI_llm_level.py

Create SI LLM-level retraction-rate plots and transition-count tables.

The script:
  1. Loads combined probing and zero-shot prediction CSVs.
  2. Computes model-level transition counts and retraction/expansion rates.
  3. Writes table-style transition summaries.
  4. Plots LLM-level retraction rates for each method.
  5. Writes outputs to outputs/plots.
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

METHODS = ["mean_diff", "sAwMIL", "zero_shot"]

METHOD_LABELS = {
    "mean_diff": "Mass-Mean",
    "sAwMIL": "sAwMIL",
    "zero_shot": "Zero-shot",
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


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def get_col_fn(method: str, layer_label: str, context_size: int):
    """Return function that constructs prediction column names."""
    if method in {"mean_diff", "sAwMIL"}:
        return lambda model, task: f"{model}__task_{task}__layer_{layer_label}__y_hat"

    if method == "zero_shot":
        return lambda model, task: f"{model}__task_{task}__K_{context_size}__y_hat"

    raise ValueError(f"Unknown method: {method}")


def get_input_path(
    input_root: Path,
    method: str,
    dataset: str,
    layer_label: str,
    context_size: int,
) -> Path:
    """Return combined prediction CSV path for one method/dataset."""
    if method in {"mean_diff", "sAwMIL"}:
        return (
            input_root
            / method
            / dataset
            / f"{method}_{dataset}_test_predictions_{layer_label}.csv"
        )

    if method == "zero_shot":
        return (
            input_root
            / "zero_shot"
            / dataset
            / f"zero_shot_{dataset}_test_predictions_K-{context_size}.csv"
        )

    raise ValueError(f"Unknown method: {method}")


def compute_model_perturbation_stats(
    df: pd.DataFrame,
    model: str,
    task: int,
    perturbation: str,
    col_fn,
):
    """Compute transition counts and rates for one model/perturbation."""
    base_col = col_fn(model, 0)
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
    pert_true = p.eq(1)
    pert_false = p.eq(0)

    true_to_true = int((baseline_true & pert_true).sum())
    not_true_to_not_true = int((baseline_false & pert_false).sum())
    expansions = int((baseline_false & pert_true).sum())
    retractions = int((baseline_true & pert_false).sum())

    n_base_true = int(baseline_true.sum())
    n_base_false = int(baseline_false.sum())

    return {
        "model": model,
        "perturbation": perturbation,
        "n_valid": int(valid.sum()),
        "baseline_true": n_base_true,
        "baseline_false": n_base_false,
        "true_to_true": true_to_true,
        "not_true_to_not_true": not_true_to_not_true,
        "expansions": expansions,
        "retractions": retractions,
        "expansion_rate": expansions / n_base_false if n_base_false else np.nan,
        "retraction_rate": retractions / n_base_true if n_base_true else np.nan,
    }


def build_all_stats(
    input_root: Path,
    layer_label: str,
    context_size: int,
) -> pd.DataFrame:
    """Build model-level transition statistics for all methods/datasets."""
    rows = []

    for method in METHODS:
        tasks = TASKS
        col_fn = get_col_fn(
            method=method,
            layer_label=layer_label,
            context_size=context_size,
        )

        for dataset in DATASETS:
            path = get_input_path(
                input_root=input_root,
                method=method,
                dataset=dataset,
                layer_label=layer_label,
                context_size=context_size,
            )

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
                    )

                    if stats is None:
                        base_col = col_fn(model, 0)
                        pert_col = col_fn(model, task)

                        if base_col not in df.columns:
                            reason = f"missing baseline column: {base_col}"
                        elif pert_col not in df.columns:
                            reason = f"missing perturbation column: {pert_col}"
                        else:
                            base = pd.to_numeric(df[base_col], errors="coerce")
                            pert = pd.to_numeric(df[pert_col], errors="coerce")
                            reason = (
                                "columns exist, but no valid paired predictions; "
                                f"base unique={sorted(base.dropna().unique())[:10]}, "
                                f"pert unique={sorted(pert.dropna().unique())[:10]}"
                            )

                        print(
                            f"[MISSING] method={method}, dataset={dataset}, "
                            f"model={model}, task={task}, perturbation={perturbation}: "
                            f"{reason}"
                        )
                        continue

                    stats["method"] = method
                    stats["dataset"] = dataset
                    rows.append(stats)

    return pd.DataFrame(rows)


def fmt_count_pct(count: int, total: int) -> str:
    """Format count and percentage."""
    pct = 100 * count / total if total else np.nan
    return f"{count} ({pct:.1f})"


def build_table3_style_tables(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write transition-count summary tables for each method."""
    for method in METHODS:
        rows = []

        for dataset in DATASETS:
            for perturbation in PERTURBATION_ORDER:
                g = stats_df[
                    stats_df["method"].eq(method)
                    & stats_df["dataset"].eq(dataset)
                    & stats_df["perturbation"].eq(perturbation)
                ]

                if g.empty:
                    continue

                true_to_true = int(g["true_to_true"].sum())
                not_true_to_not_true = int(g["not_true_to_not_true"].sum())
                expansions = int(g["expansions"].sum())
                retractions = int(g["retractions"].sum())

                total = true_to_true + not_true_to_not_true + expansions + retractions

                rows.append(
                    {
                        "Dataset": DATASET_LABELS[dataset],
                        "Perturbation": perturbation,
                        "True to True": fmt_count_pct(true_to_true, total),
                        "Not True to Not True": fmt_count_pct(
                            not_true_to_not_true,
                            total,
                        ),
                        "Epistemic Expansions E": fmt_count_pct(expansions, total),
                        "Epistemic Retractions R": fmt_count_pct(retractions, total),
                        "n_total": total,
                        "n_models": int(g["model"].nunique()),
                    }
                )

        out = pd.DataFrame(rows)
        out_path = output_dir / f"table3_style_{method}.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")


def plot_llm_level_retraction_rates(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot LLM-level retraction rates for each method."""
    for method in METHODS:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(7.2, 5.5),
            sharex=True,
        )

        width = 0.15
        offsets = np.linspace(
            -width * (len(PERTURBATION_ORDER) - 1) / 2,
            width * (len(PERTURBATION_ORDER) - 1) / 2,
            len(PERTURBATION_ORDER),
        )

        x = np.arange(len(MODELS))
        panel_labels = ["(a)", "(b)", "(c)"]

        for ax, dataset, panel_label in zip(axes, DATASETS, panel_labels):
            ddf = stats_df[
                stats_df["method"].eq(method)
                & stats_df["dataset"].eq(dataset)
            ]

            for perturbation, offset in zip(PERTURBATION_ORDER, offsets):
                vals = []

                for model in MODELS:
                    row = ddf[
                        ddf["model"].eq(model)
                        & ddf["perturbation"].eq(perturbation)
                    ]

                    vals.append(
                        np.nan
                        if row.empty
                        else float(row.iloc[0]["retraction_rate"])
                    )

                ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    label=perturbation,
                    color=CONDITION_COLORS[perturbation],
                )

            ax.set_ylabel(r"Retraction Rate $\rho_{\mathcal{R}}$", fontsize=7)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=6)

            ax.text(
                -0.05,
                1.05,
                f"{panel_label} {DATASET_LABELS[dataset]}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#444444",
            )

            _, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax * 1.10 if ymax > 0 else 1)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(MODELS, rotation=45, ha="right", fontsize=6)

        fig.text(
            0.01,
            0.97,
            f"LLM-level retraction rates: {METHOD_LABELS[method]}",
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
            fontweight="bold",
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.08),
            ncol=5,
            frameon=False,
            fontsize=7,
        )

        for ax in axes:
            ax.yaxis.set_label_coords(-0.07, 0.5)

        fig.tight_layout(rect=[0, 0.12, 1, 0.95], h_pad=0.8)

        png_path = output_dir / f"llm_level_retraction_rates_{method}.png"
        pdf_path = output_dir / f"llm_level_retraction_rates_{method}.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote: {png_path}")
        print(f"Wrote: {pdf_path}")


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
        "--layer-label",
        default="tasklayer",
    )
    parser.add_argument(
        "--zero-shot-k",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main() -> None:
    """Create SI LLM-level plots and tables."""
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_df = build_all_stats(
        input_root=input_root,
        layer_label=args.layer_label,
        context_size=args.zero_shot_k,
    )

    stats_path = output_dir / "llm_level_all_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote: {stats_path}")

    build_table3_style_tables(
        stats_df=stats_df,
        output_dir=output_dir,
    )

    plot_llm_level_retraction_rates(
        stats_df=stats_df,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()