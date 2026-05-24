"""
build_results_summary_zeroshot.py

Build combined zero-shot prediction tables for ground-truth true test statements.

The script:
  1. Loads the filtered true-real test set for each dataset.
  2. Loads zero-shot predictions for each model, task, and context size.
  3. Maps ABC predictions to binary true-vs-not-true y_hat values.
  4. Aligns predictions to test rows by statement text.
  5. Writes combined prediction CSVs for downstream analysis and plotting.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROBE_NAME = "zero_shot"

CONTEXT_SIZES = [20, 50, 100]
TASKS = [0, 1, 2, 3, 4, 5, 6]
DATASETS = ["cities_loc", "med_indications", "defs"]

MODELS = [
    "_gemma-2-9b",
    "_gemma-3-27b",
    "_gemma-7b",
    "_llama-3-8b-med",
    "_llama-3.1-8b",
    "_llama-3.1-8b-bio",
    "_llama-3.2-3b",
    "_llama-3.3-70b",
    "_mistral-7B-v0.3",
    "_qwen-2.5-14b",
    "_qwen-2.5-72b",
    "_qwen-2.5-7b",
    "gemma-2-9b",
    "gemma-3-27b",
    "gemma-7b",
    "llama-3-8b",
    "llama-3.2-3b",
    "mistral-7B-v0.3",
    "qwen-2.5-14b",
    "qwen-2.5-72b",
    "qwen-2.5-7b",
]


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def add_project_to_path() -> None:
    """Add the project root to sys.path so local modules can be imported."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def load_config(model: str, dataset: str, task: int):
    """Compose the Hydra config used for test-set loading."""
    config_dir = project_path("configs").resolve()

    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        return hydra.compose(
            config_name="probe_linear_mil",
            overrides=[
                f"model={model}",
                f"datapack={dataset}",
                f"datapack@datapack_test={dataset}",
                f"task={task}",
            ],
        )


def to_binary_predictions(preds: np.ndarray) -> np.ndarray:
    """Map zero-shot ABC predictions to binary true-vs-not-true labels."""
    preds = np.asarray(preds)

    if preds.ndim == 1:
        out = np.full(len(preds), np.nan, dtype=float)
        finite_mask = pd.notna(preds)

        pred_vals = preds[finite_mask].astype(int)
        valid_vals = set(np.unique(pred_vals))

        if not valid_vals.issubset({0, 1, 2}):
            raise ValueError(
                f"Unexpected zero-shot prediction values: {sorted(valid_vals)}. "
                "Expected values in {0, 1, 2}, where 0=True, 1=False, 2=Neither."
            )

        out[finite_mask] = (pred_vals == 0).astype(float)
        return out

    if preds.ndim == 2:
        pred_idx = np.argmax(preds, axis=1)
        return (pred_idx == 0).astype(float)

    raise ValueError(f"Unexpected preds shape: {preds.shape}")


def load_test_dataframe(
    dataset: str,
    model_for_cfg: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the filtered true-real test dataframe and full-test mask."""
    add_project_to_path()
    from utils import load_data

    cfg = load_config(model=model_for_cfg, dataset=dataset, task=0)

    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        dh = load_data(cfg)
        full_test_df = dh.get_test_df().reset_index(drop=False)
    finally:
        os.chdir(old_cwd)

    if "index" in full_test_df.columns and "statement_id" not in full_test_df.columns:
        full_test_df = full_test_df.rename(columns={"index": "statement_id"})

    if "statement_id" not in full_test_df.columns:
        full_test_df.insert(0, "statement_id", np.arange(len(full_test_df)))

    required_cols = [
        "correct",
        "real_object",
        "fictional_object",
        "synthetic_fic_object",
    ]
    missing_cols = [col for col in required_cols if col not in full_test_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in test dataframe for {dataset}: {missing_cols}"
        )

    true_test_mask = (
        full_test_df["correct"].astype(bool)
        & full_test_df["real_object"].astype(int).eq(1)
        & full_test_df["fictional_object"].astype(int).eq(0)
        & full_test_df["synthetic_fic_object"].astype(int).eq(0)
    ).to_numpy()

    test_df = full_test_df.loc[true_test_mask].reset_index(drop=True)

    return test_df, true_test_mask


def get_zero_shot_task_dir(
    zero_shot_root: Path,
    model: str,
    dataset: str,
    task: int,
    context_size: int,
) -> Path:
    """Return zero-shot output directory for one model/dataset/task/K setting."""
    if task == 0:
        return zero_shot_root / model / dataset / "task-0"

    return zero_shot_root / model / dataset / f"K-{context_size}" / f"task-{task}"


def load_zero_shot_predictions(
    zero_shot_root: Path,
    model: str,
    dataset: str,
    task: int,
    context_size: int,
) -> tuple[np.ndarray, list[str], Path]:
    """Load zero-shot predictions and corresponding statements."""
    exp_dir = get_zero_shot_task_dir(
        zero_shot_root=zero_shot_root,
        model=model,
        dataset=dataset,
        task=task,
        context_size=context_size,
    )

    preds_path = exp_dir / "preds.npy"
    statements_path = exp_dir / "statements.txt"

    if not preds_path.exists():
        raise FileNotFoundError(f"Could not find predictions file: {preds_path}")

    if not statements_path.exists():
        raise FileNotFoundError(f"Could not find statements file: {statements_path}")

    preds = np.load(preds_path, allow_pickle=True)
    statements = statements_path.read_text().splitlines()

    if len(preds) != len(statements):
        raise ValueError(
            f"Prediction/statement length mismatch in {exp_dir}: "
            f"len(preds)={len(preds)}, len(statements)={len(statements)}"
        )

    return preds, statements, preds_path


def align_predictions_by_statement(
    out_df: pd.DataFrame,
    preds: np.ndarray,
    statements: list[str],
    col: str,
    preds_path: Path,
) -> np.ndarray:
    """Align zero-shot predictions to output rows by exact statement text."""
    y_bin = to_binary_predictions(preds)

    pred_df = pd.DataFrame(
        {
            "statement": statements,
            col: y_bin,
        }
    )

    duplicate_mask = pred_df["statement"].duplicated(keep=False)
    if duplicate_mask.any():
        n_dupes = int(duplicate_mask.sum())

        inconsistent = (
            pred_df.loc[duplicate_mask]
            .groupby("statement")[col]
            .nunique(dropna=False)
        )
        inconsistent = inconsistent[inconsistent > 1]

        if len(inconsistent) > 0:
            print(
                f"Warning: {len(inconsistent)} duplicate statements have inconsistent "
                f"predictions in {preds_path}; keeping the first occurrence."
            )
        else:
            print(
                f"Warning: found {n_dupes} duplicate statement rows in {preds_path}; "
                "duplicates have identical predictions, keeping one."
            )

        pred_df = pred_df.drop_duplicates(subset=["statement"], keep="first")

    merged = out_df[["statement"]].merge(
        pred_df,
        on="statement",
        how="left",
        validate="many_to_one",
    )

    n_missing = int(merged[col].isna().sum())
    if n_missing > 0:
        print(
            f"Warning: {n_missing}/{len(out_df)} rows did not match statements "
            f"for {preds_path}"
        )

    return merged[col].to_numpy()


def load_prediction(
    zero_shot_root: Path,
    model: str,
    dataset: str,
    task: int,
    context_size: int,
) -> tuple[np.ndarray, list[str], Path, str]:
    """Load direct predictions."""
    preds, statements, preds_path = load_zero_shot_predictions(
        zero_shot_root=zero_shot_root,
        model=model,
        dataset=dataset,
        task=task,
        context_size=context_size,
    )
    return preds, statements, preds_path, model


def build_dataset_table(
    zero_shot_root: Path,
    dataset: str,
    context_size: int,
    models: list[str],
    tasks: list[int],
) -> pd.DataFrame:
    """Build one combined zero-shot prediction table for a dataset and K."""
    print()
    print("=" * 80)
    print(f"Building zero-shot table: dataset={dataset}, K={context_size}")
    print("=" * 80)

    test_df, _ = load_test_dataframe(
        dataset=dataset,
        model_for_cfg=models[0],
    )

    keep_cols = [
        "statement_id",
        "statement",
        "object_1",
        "object_2",
        "correct_object_2",
        "correct",
        "negation",
        "real_object",
        "fake_object",
        "fictional_object",
        "synthetic_fic_object",
        "category",
        "in_train",
        "in_test",
        "in_cal",
    ]

    keep_cols = [col for col in keep_cols if col in test_df.columns]
    out_df = test_df[keep_cols].copy()

    prediction_cols = {}

    for model in models:
        for task in tasks:
            col = f"{model}__task_{task}__K_{context_size}__y_hat"

            try:
                preds, statements, preds_path, source_model = load_prediction(
                    zero_shot_root=zero_shot_root,
                    model=model,
                    dataset=dataset,
                    task=task,
                    context_size=context_size,
                )

                aligned_preds = align_predictions_by_statement(
                    out_df=out_df,
                    preds=preds,
                    statements=statements,
                    col=col,
                    preds_path=preds_path,
                )

                prediction_cols[col] = aligned_preds

                if source_model == model:
                    print(f"Loaded {col} from {preds_path}")
                else:
                    print(f"Fallback loaded {col} from {source_model}: {preds_path}")

            except Exception as e:
                print(
                    f"Missing/failed: model={model}, dataset={dataset}, "
                    f"task={task}, K={context_size}"
                )
                print(f"  {e}")
                prediction_cols[col] = np.full(len(out_df), np.nan)

    if prediction_cols:
        pred_df = pd.DataFrame(prediction_cols)
        out_df = pd.concat([out_df.reset_index(drop=True), pred_df], axis=1)

    return out_df


def parse_csv_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_list(value: str) -> list[int]:
    """Parse a comma-separated string into a list of integers."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        default="cities_loc,med_indications,defs",
        help="Comma-separated datasets to process.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated models to process.",
    )
    parser.add_argument(
        "--tasks",
        default="0,1,2,3,4,5,6",
        help="Comma-separated task ids to process.",
    )
    parser.add_argument(
        "--context-sizes",
        default="20,50,100",
        help="Comma-separated context sizes to process.",
    )
    parser.add_argument(
        "--zero-shot-root",
        default=str(project_path("outputs", "probes", PROBE_NAME)),
        help="Root directory containing zero-shot outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_path("outputs", "analysis_data")),
        help="Root where combined zero-shot CSVs should be written.",
    )

    return parser.parse_args()


def main() -> None:
    """Build and save combined zero-shot prediction tables."""
    args = parse_args()

    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    tasks = parse_int_csv_list(args.tasks)
    context_sizes = parse_int_csv_list(args.context_sizes)

    if len(models) == 0:
        raise ValueError("At least one model is required.")

    zero_shot_root = Path(args.zero_shot_root)

    for dataset in datasets:
        dataset_output_dir = Path(args.output_root) / PROBE_NAME / dataset
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        for context_size in context_sizes:
            df = build_dataset_table(
                zero_shot_root=zero_shot_root,
                dataset=dataset,
                context_size=context_size,
                models=models,
                tasks=tasks,
            )

            out_path = (
                dataset_output_dir
                / f"{PROBE_NAME}_{dataset}_test_predictions_K-{context_size}.csv"
            )

            df.to_csv(out_path, index=False)
            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()