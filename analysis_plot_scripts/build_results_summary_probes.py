"""
build_results_summary_probes.py

Build combined probing prediction tables for ground-truth true test statements.

The script:
  1. Loads the filtered true-real test set for each dataset.
  2. Finds saved probe predictions for each model, task, and layer offset.
  3. Converts saved predictions to binary y_hat values.
  4. Writes combined prediction CSVs for downstream analysis and plotting.
"""

import argparse
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TASKS = [0, 1, 2, 3, 4, 5]
DATASETS = ["cities_loc", "med_indications", "defs"]

LAYER_OFFSETS = {
    "tasklayerminus2": -2,
    "tasklayerminus1": -1,
    "tasklayer": 0,
    "tasklayerplus1": 1,
    "tasklayerplus2": 2,
}

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
    """Compose the Hydra config used by the probing experiment."""
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


def get_trial_name(dataset: str, task: int, search: bool, noise: int) -> str:
    """Return the probe output trial name for one dataset/task."""
    trial_name = dataset

    if search:
        trial_name += "_search"

    if noise:
        trial_name += f"_noise{noise}"

    trial_name += f"_syntheticfic_task-{task}"

    return trial_name


def get_task_layer(model: str, dataset: str) -> int:
    """Read the dataset-specific task layer from the model config."""
    cfg = load_config(model=model, dataset=dataset, task=0)
    return int(cfg.model.task_layers[dataset])


def get_experiment_dir(
    probe: str,
    model: str,
    dataset: str,
    task: int,
    search: bool,
    noise: int,
) -> Path:
    """Return the directory containing saved probe outputs."""
    trial_name = get_trial_name(
        dataset=dataset,
        task=task,
        search=search,
        noise=noise,
    )

    return (
        project_path("outputs", "probes")
        / probe
        / model
        / trial_name
    )


def find_yhat_file(exp_dir: Path, layer: int) -> Path:
    """Find the saved prediction file for one layer."""
    patterns = [
        f"y_hat_{layer}.npy",
        f"yhat_{layer}.npy",
        f"y_pred_{layer}.npy",
        f"predictions_{layer}.npy",
        f"*y_hat*{layer}*.npy",
        f"*yhat*{layer}*.npy",
        f"*pred*{layer}*.npy",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(exp_dir.glob(pattern))

    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find y_hat file for layer={layer} in {exp_dir}"
        )

    if len(matches) > 1:
        print(f"Warning: multiple y_hat candidates in {exp_dir}:")
        for match in matches:
            print(f"  {match}")
        print(f"Using: {matches[0]}")

    return matches[0]


def to_binary_predictions(y_hat: np.ndarray) -> np.ndarray:
    """Convert saved model outputs to binary labels in {0, 1}."""
    y_hat = np.asarray(y_hat)

    if y_hat.ndim == 1:
        unique_vals = set(pd.Series(y_hat).dropna().unique())

        if unique_vals.issubset({0, 1}):
            return y_hat.astype(int)

        if unique_vals.issubset({-1, 1}):
            return (y_hat == 1).astype(int)

        return (y_hat > 0).astype(int)

    if y_hat.ndim == 2:
        return np.argmax(y_hat, axis=1).astype(int)

    raise ValueError(f"Unexpected y_hat shape: {y_hat.shape}")


def load_test_dataframe(dataset: str, model_for_cfg: str) -> tuple[pd.DataFrame, np.ndarray]:
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


def build_dataset_table(
    probe: str,
    dataset: str,
    layer_label: str,
    layer_offset: int,
    models: list[str],
    tasks: list[int],
    search: bool,
    noise: int,
) -> pd.DataFrame:
    """Build one combined prediction table for a dataset and layer offset."""
    print()
    print("=" * 80)
    print(f"Building table: probe={probe}, dataset={dataset}, layer_label={layer_label}")
    print("=" * 80)

    test_df, true_test_mask = load_test_dataframe(
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

    for model in models:
        try:
            task_layer = get_task_layer(model=model, dataset=dataset)
            layer = task_layer + layer_offset
        except Exception as e:
            print(f"Skipping model={model}, dataset={dataset}: no task layer ({e})")
            continue

        for task in tasks:
            col = f"{model}__task_{task}__layer_{layer_label}__y_hat"

            exp_dir = get_experiment_dir(
                probe=probe,
                model=model,
                dataset=dataset,
                task=task,
                search=search,
                noise=noise,
            )

            try:
                yhat_path = find_yhat_file(exp_dir=exp_dir, layer=layer)
                y_hat = np.load(yhat_path, allow_pickle=True)
                y_bin = to_binary_predictions(y_hat)

                if len(y_bin) != len(true_test_mask):
                    raise ValueError(
                        f"Prediction length mismatch for {yhat_path}: "
                        f"len(y_hat)={len(y_bin)}, "
                        f"len(full_test_df)={len(true_test_mask)}, "
                        f"len(filtered_test_df)={len(out_df)}"
                    )

                out_df[col] = y_bin[true_test_mask]
                print(f"Loaded {col} from {yhat_path}")

            except Exception as e:
                print(
                    f"Missing/failed: model={model}, task={task}, "
                    f"layer_label={layer_label}, actual_layer={layer}"
                )
                print(f"  {e}")
                out_df[col] = np.nan

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
        "--probe",
        required=True,
        choices=["mean_diff", "sAwMIL"],
        help="Probe name used under outputs/probes.",
    )
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
        default="0,1,2,3,4,5",
        help="Comma-separated task ids to process.",
    )
    parser.add_argument(
        "--layer-labels",
        default=",".join(LAYER_OFFSETS.keys()),
        help="Comma-separated layer labels to process.",
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether trial names include '_search'.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=10,
        help="Noise level used in trial names. Use 0 to omit noise suffix.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_path("outputs", "analysis_data")),
        help="Root where combined CSVs should be written.",
    )

    return parser.parse_args()


def main() -> None:
    """Build and save combined probing prediction tables."""
    args = parse_args()

    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    tasks = parse_int_csv_list(args.tasks)
    layer_labels = parse_csv_list(args.layer_labels)

    unknown_layer_labels = [
        label for label in layer_labels if label not in LAYER_OFFSETS
    ]
    if unknown_layer_labels:
        raise ValueError(f"Unknown layer labels: {unknown_layer_labels}")

    if len(models) == 0:
        raise ValueError("At least one model is required.")

    for dataset in datasets:
        dataset_output_dir = Path(args.output_root) / args.probe / dataset
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        for layer_label in layer_labels:
            layer_offset = LAYER_OFFSETS[layer_label]

            df = build_dataset_table(
                probe=args.probe,
                dataset=dataset,
                layer_label=layer_label,
                layer_offset=layer_offset,
                models=models,
                tasks=tasks,
                search=args.search,
                noise=args.noise,
            )

            out_path = (
                dataset_output_dir
                / f"{args.probe}_{dataset}_test_predictions_{layer_label}.csv"
            )

            df.to_csv(out_path, index=False)
            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()