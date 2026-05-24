"""
build_retraction_expansion_tables_probes.py

Build retraction and expansion summary tables from combined probing predictions.

The script:
  1. Loads combined probing prediction CSVs.
  2. Infers available models and task prediction columns.
  3. Computes model-level expansion and retraction counts/rates.
  4. Computes perturbation-level summary tables.
  5. Writes CSVs for downstream plotting and paper tables.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TASK_LABELS = {
    1: "synthetic",
    2: "synthetic_fic",
    3: "fictional",
    4: "fictional_true",
    5: "noise",
}


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def parse_csv_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def prediction_col_regex(layer_label: str) -> re.Pattern:
    """Return regex for model/task prediction columns."""
    return re.compile(
        rf"(.+)__task_(\d+)__layer_{re.escape(layer_label)}__y_hat"
    )


def input_prediction_path(
    analysis_root: Path,
    probe: str,
    dataset: str,
    layer_label: str,
) -> Path:
    """Return path to the combined probing prediction CSV."""
    return (
        analysis_root
        / probe
        / dataset
        / f"{probe}_{dataset}_test_predictions_{layer_label}.csv"
    )


def output_table_dir(
    analysis_root: Path,
    probe: str,
    dataset: str,
) -> Path:
    """Return output directory for derived retraction/expansion tables."""
    return analysis_root / probe / dataset / "retraction_expansion_tables"


def infer_model_task_columns(
    df: pd.DataFrame,
    layer_label: str,
) -> dict[str, dict[int, str]]:
    """Infer available model/task prediction columns."""
    col_re = prediction_col_regex(layer_label)
    model_to_tasks: dict[str, dict[int, str]] = {}

    for col in df.columns:
        match = col_re.fullmatch(col)
        if match is None:
            continue

        model = match.group(1)
        task = int(match.group(2))
        model_to_tasks.setdefault(model, {})[task] = col

    return model_to_tasks


def safe_rate(numerator: int, denominator: int) -> float | pd.NA:
    """Compute numerator / denominator, returning NA for zero denominators."""
    if denominator <= 0:
        return pd.NA

    return numerator / denominator


def ranked_labels(row: pd.Series, metric_suffix: str) -> list[str]:
    """Rank perturbations by a model-level metric."""
    values = {
        label: row[f"{label}_{metric_suffix}"]
        for label in TASK_LABELS.values()
        if pd.notna(row[f"{label}_{metric_suffix}"])
    }

    return [
        label
        for label, _ in sorted(
            values.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def build_model_summary(
    df: pd.DataFrame,
    layer_label: str,
) -> pd.DataFrame:
    """Build one retraction/expansion summary row per model."""
    model_to_tasks = infer_model_task_columns(
        df=df,
        layer_label=layer_label,
    )

    rows = []

    for model, task_cols in sorted(model_to_tasks.items()):
        if 0 not in task_cols:
            print(f"Skipping {model}: missing task 0")
            continue

        baseline = df[task_cols[0]]
        baseline_true = int((baseline == 1).sum())
        baseline_false = int((baseline == 0).sum())

        row = {
            "model": model,
            "baseline_true": baseline_true,
            "baseline_false": baseline_false,
        }

        for task, label in TASK_LABELS.items():
            if task not in task_cols:
                row[f"{label}_expansions"] = pd.NA
                row[f"{label}_expansion_rate"] = pd.NA
                row[f"{label}_retractions"] = pd.NA
                row[f"{label}_retraction_rate"] = pd.NA
                continue

            perturbed = df[task_cols[task]]

            expansions = int(((baseline == 0) & (perturbed == 1)).sum())
            retractions = int(((baseline == 1) & (perturbed == 0)).sum())

            row[f"{label}_expansions"] = expansions
            row[f"{label}_expansion_rate"] = safe_rate(
                numerator=expansions,
                denominator=baseline_false,
            )
            row[f"{label}_retractions"] = retractions
            row[f"{label}_retraction_rate"] = safe_rate(
                numerator=retractions,
                denominator=baseline_true,
            )

        row["retraction_rate_ranked"] = str(
            ranked_labels(pd.Series(row), metric_suffix="retraction_rate")
        )
        row["expansion_rate_ranked"] = str(
            ranked_labels(pd.Series(row), metric_suffix="expansion_rate")
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_perturbation(model_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize expansion and retraction rates by perturbation."""
    rows = []

    for label in TASK_LABELS.values():
        expansion_col = f"{label}_expansion_rate"
        retraction_col = f"{label}_retraction_rate"

        rows.append(
            {
                "perturbation": label,
                "mean_expansion_rate": model_summary_df[expansion_col].mean(),
                "median_expansion_rate": model_summary_df[expansion_col].median(),
                "num_models_with_expansion_rate": int(
                    model_summary_df[expansion_col].notna().sum()
                ),
                "mean_retraction_rate": model_summary_df[retraction_col].mean(),
                "median_retraction_rate": model_summary_df[retraction_col].median(),
                "num_models_with_retraction_rate": int(
                    model_summary_df[retraction_col].notna().sum()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("median_retraction_rate", ascending=False)
        .reset_index(drop=True)
    )


def count_top_perturbations(
    model_summary_df: pd.DataFrame,
    metric_suffix: str,
) -> pd.DataFrame:
    """Count how often each perturbation has the largest value for a metric."""
    top_counts = {label: 0 for label in TASK_LABELS.values()}

    for _, row in model_summary_df.iterrows():
        values = {
            label: row[f"{label}_{metric_suffix}"]
            for label in TASK_LABELS.values()
            if pd.notna(row[f"{label}_{metric_suffix}"])
        }

        if len(values) == 0:
            continue

        max_value = max(values.values())

        for label, value in values.items():
            if value == max_value:
                top_counts[label] += 1

    rows = [
        {
            "perturbation": label,
            f"num_models_ranked_first_by_{metric_suffix}": count,
        }
        for label, count in top_counts.items()
    ]

    return (
        pd.DataFrame(rows)
        .sort_values(f"num_models_ranked_first_by_{metric_suffix}", ascending=False)
        .reset_index(drop=True)
    )


def compare_synthetic_and_fictional_retractions(
    model_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare synthetic and fictional retraction rates across models."""
    comparisons = {
        "synthetic > fictional": (
            model_summary_df["synthetic_retraction_rate"]
            > model_summary_df["fictional_retraction_rate"]
        ),
        "synthetic > fictional_true": (
            model_summary_df["synthetic_retraction_rate"]
            > model_summary_df["fictional_true_retraction_rate"]
        ),
        "synthetic_fic > fictional": (
            model_summary_df["synthetic_fic_retraction_rate"]
            > model_summary_df["fictional_retraction_rate"]
        ),
        "synthetic_fic > fictional_true": (
            model_summary_df["synthetic_fic_retraction_rate"]
            > model_summary_df["fictional_true_retraction_rate"]
        ),
        "both synthetic conditions > both fictional conditions": (
            model_summary_df[
                ["synthetic_retraction_rate", "synthetic_fic_retraction_rate"]
            ].min(axis=1)
            > model_summary_df[
                ["fictional_retraction_rate", "fictional_true_retraction_rate"]
            ].max(axis=1)
        ),
        "at least one synthetic condition > both fictional conditions": (
            model_summary_df[
                ["synthetic_retraction_rate", "synthetic_fic_retraction_rate"]
            ].max(axis=1)
            > model_summary_df[
                ["fictional_retraction_rate", "fictional_true_retraction_rate"]
            ].max(axis=1)
        ),
    }

    rows = []

    for label, mask in comparisons.items():
        rows.append(
            {
                "comparison": label,
                "num_models": int(mask.sum()),
                "prop_models": float(mask.mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_average_ranks(
    model_summary_df: pd.DataFrame,
    metric_suffix: str,
) -> pd.DataFrame:
    """Compute average perturbation rank for a model-level metric."""
    rows = []

    for _, row in model_summary_df.iterrows():
        values = {
            label: row[f"{label}_{metric_suffix}"]
            for label in TASK_LABELS.values()
            if pd.notna(row[f"{label}_{metric_suffix}"])
        }

        ranked = sorted(values, key=values.get, reverse=True)

        for rank, label in enumerate(ranked, start=1):
            rows.append(
                {
                    "model": row["model"],
                    "perturbation": label,
                    "rank": rank,
                }
            )

    rank_df = pd.DataFrame(rows)

    if len(rank_df) == 0:
        return pd.DataFrame(columns=["perturbation", f"average_{metric_suffix}_rank"])

    return (
        rank_df
        .groupby("perturbation", as_index=False)["rank"]
        .mean()
        .sort_values("rank")
        .rename(columns={"rank": f"average_{metric_suffix}_rank"})
        .reset_index(drop=True)
    )


def write_tables(
    model_summary_df: pd.DataFrame,
    out_dir: Path,
    probe: str,
    dataset: str,
    layer_label: str,
) -> None:
    """Write model-level and aggregate retraction/expansion tables."""
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{probe}_{dataset}_{layer_label}"

    perturbation_summary_df = summarize_by_perturbation(model_summary_df)
    top_retraction_df = count_top_perturbations(
        model_summary_df,
        metric_suffix="retraction_rate",
    )
    top_expansion_df = count_top_perturbations(
        model_summary_df,
        metric_suffix="expansion_rate",
    )
    synthetic_fictional_df = compare_synthetic_and_fictional_retractions(
        model_summary_df
    )
    avg_retraction_rank_df = summarize_average_ranks(
        model_summary_df,
        metric_suffix="retraction_rate",
    )
    avg_expansion_rank_df = summarize_average_ranks(
        model_summary_df,
        metric_suffix="expansion_rate",
    )

    paths = {
        "model_summary": out_dir / f"{prefix}_model_retraction_expansion_summary.csv",
        "perturbation_summary": out_dir / f"{prefix}_perturbation_summary.csv",
        "top_retraction_counts": out_dir / f"{prefix}_top_retraction_counts.csv",
        "top_expansion_counts": out_dir / f"{prefix}_top_expansion_counts.csv",
        "synthetic_fictional_retraction_comparisons": (
            out_dir / f"{prefix}_synthetic_fictional_retraction_comparisons.csv"
        ),
        "average_retraction_ranks": (
            out_dir / f"{prefix}_average_retraction_ranks.csv"
        ),
        "average_expansion_ranks": (
            out_dir / f"{prefix}_average_expansion_ranks.csv"
        ),
    }

    model_summary_df.to_csv(paths["model_summary"], index=False)
    perturbation_summary_df.to_csv(paths["perturbation_summary"], index=False)
    top_retraction_df.to_csv(paths["top_retraction_counts"], index=False)
    top_expansion_df.to_csv(paths["top_expansion_counts"], index=False)
    synthetic_fictional_df.to_csv(
        paths["synthetic_fictional_retraction_comparisons"],
        index=False,
    )
    avg_retraction_rank_df.to_csv(paths["average_retraction_ranks"], index=False)
    avg_expansion_rank_df.to_csv(paths["average_expansion_ranks"], index=False)

    for name, path in paths.items():
        print(f"Saved {name}: {path}")

    print("\n=== Retraction-rate ranking by model ===")
    print(model_summary_df[["model", "retraction_rate_ranked"]].to_string(index=False))

    print("\n=== Perturbation summary ===")
    print(perturbation_summary_df.to_string(index=False))

    print("\n=== Synthetic vs. fictional retraction comparisons ===")
    print(synthetic_fictional_df.to_string(index=False))


def build_tables_for_dataset(
    analysis_root: Path,
    probe: str,
    dataset: str,
    layer_label: str,
) -> None:
    """Build and save retraction/expansion tables for one probe/dataset/layer."""
    input_path = input_prediction_path(
        analysis_root=analysis_root,
        probe=probe,
        dataset=dataset,
        layer_label=layer_label,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find combined prediction CSV: {input_path}")

    print()
    print("=" * 80)
    print(f"Building retraction/expansion tables: probe={probe}, dataset={dataset}")
    print(f"Layer label: {layer_label}")
    print("=" * 80)
    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path)

    model_summary_df = build_model_summary(
        df=df,
        layer_label=layer_label,
    )

    out_dir = output_table_dir(
        analysis_root=analysis_root,
        probe=probe,
        dataset=dataset,
    )

    write_tables(
        model_summary_df=model_summary_df,
        out_dir=out_dir,
        probe=probe,
        dataset=dataset,
        layer_label=layer_label,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe",
        required=True,
        choices=["mean_diff", "sAwMIL"],
    )
    parser.add_argument(
        "--datasets",
        default="cities_loc,med_indications,defs",
    )
    parser.add_argument(
        "--layer-label",
        default="tasklayer",
    )
    parser.add_argument(
        "--analysis-root",
        default=str(project_path("outputs", "analysis_data")),
        help="Root containing combined probing prediction tables.",
    )

    return parser.parse_args()


def main() -> None:
    """Build retraction/expansion tables for selected probing results."""
    args = parse_args()

    analysis_root = Path(args.analysis_root)
    datasets = parse_csv_list(args.datasets)

    for dataset in datasets:
        build_tables_for_dataset(
            analysis_root=analysis_root,
            probe=args.probe,
            dataset=dataset,
            layer_label=args.layer_label,
        )


if __name__ == "__main__":
    main()