"""
analyze_within_between_activations.py

Compute within-condition and between-condition activation distances for
fictional, synthetic-fictional, and synthetic statement activations.
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


CONDITION_ORDER = ["fictional", "synthetic_fic", "synthetic"]

OUTPUT_COLUMNS = [
    "dataset",
    "model",
    "layer",
    "metric",
    "max_per_class",
    "n_fictional_raw",
    "n_fictional_used",
    "within_fictional",
    "n_synthetic_fic_raw",
    "n_synthetic_fic_used",
    "within_synthetic_fic",
    "n_synthetic_raw",
    "n_synthetic_used",
    "within_synthetic",
    "mean_within",
    "between_fictional_synthetic_fic",
    "between_fictional_synthetic_fic_over_mean_within",
    "between_fictional_synthetic",
    "between_fictional_synthetic_over_mean_within",
    "between_synthetic_fic_synthetic",
    "between_synthetic_fic_synthetic_over_mean_within",
    "within_fictional_over_synthetic_fic",
    "within_fictional_over_synthetic",
    "within_synthetic_fic_over_synthetic",
]

DEFAULT_DATASET_DIRS = {
    "cities_loc": {
        "fictional": ["cities_loc_fictional"],
        "synthetic_fic": ["cities_loc_synthetic_fic"],
        "synthetic": ["cities_loc_synthetic"],
    },
    "med_indications": {
        "fictional": ["med_indications_fictional"],
        "synthetic_fic": ["med_indications_synthetic_fic"],
        "synthetic": ["med_indications_synthetic"],
    },
    "defs": {
        "fictional": ["defs_fictional"],
        "synthetic_fic": ["defs_synthetic_fic"],
        "synthetic": ["defs_synthetic"],
    },
}


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the original project root."""
    return Path(get_original_cwd()).joinpath(*parts)


def resolve_path(path: str) -> Path:
    """Resolve absolute paths directly and relative paths from the project root."""
    path = Path(path)
    return path if path.is_absolute() else project_path(str(path))


def get_task_layer(cfg: DictConfig, dataset: str) -> int:
    """Get the model-specific task layer for a dataset."""
    if not hasattr(cfg.model, "task_layers"):
        raise ValueError(f"cfg.model for {cfg.model.name} does not define task_layers.")

    if dataset not in cfg.model.task_layers:
        raise ValueError(f"No task layer for model={cfg.model.name}, dataset={dataset}.")

    return int(cfg.model.task_layers[dataset])


def activation_file(root: Path, model: str, dataset_dir: str, layer: int) -> Path:
    """Build the expected activation file path."""
    return root / model / dataset_dir / "full" / f"layer_{layer}_e_temp.npy"


def load_activation_file(path: Path) -> np.ndarray:
    """Load an activation file saved as .npy, .npz, or raw memmap with shape.npy."""
    try:
        if path.suffix == ".npz":
            with np.load(path) as z:
                key = "arr_0" if "arr_0" in z else z.files[0]
                return z[key]

        return np.load(path, mmap_mode="r")

    except Exception:
        shape_path = path.parent / "shape.npy"
        if not shape_path.exists():
            raise FileNotFoundError(
                f"Could not load {path}, and no shape.npy found at {shape_path}."
            )

        shape = tuple(int(x) for x in np.atleast_1d(np.load(shape_path)).tolist())
        return np.memmap(path, mode="r", dtype=np.float16, shape=shape)


def reduce_to_statement_embeddings(A: np.ndarray) -> np.ndarray:
    """Reduce token-level activations to one embedding per statement."""
    A = np.asarray(A)

    if A.ndim == 2:
        return A

    if A.ndim != 3:
        raise ValueError(f"Unsupported activation shape: {A.shape}")

    n, seq_len, _ = A.shape
    nonzero_rows = (A != 0).any(axis=2)

    last_token_idx = np.full(n, seq_len - 1, dtype=int)
    for i in range(n):
        rows = np.where(nonzero_rows[i])[0]
        if len(rows) > 0:
            last_token_idx[i] = rows[-1]

    return A[np.arange(n), last_token_idx, :]


def load_activation_group(
    activation_root: Path,
    model: str,
    dataset_dirs: list[str],
    layer: int,
) -> np.ndarray:
    """Load and concatenate activations for one condition."""
    arrays = []

    for dataset_dir in dataset_dirs:
        path = activation_file(
            root=activation_root,
            model=model,
            dataset_dir=dataset_dir,
            layer=layer,
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing activation file: {path}")

        A = load_activation_file(path)
        log.warning(f"Loaded {path}: shape={A.shape}, dtype={A.dtype}")

        A = reduce_to_statement_embeddings(A).astype(np.float32)
        arrays.append(A)

    return np.concatenate(arrays, axis=0)


def subsample(X: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Subsample rows without replacement."""
    if len(X) <= max_n:
        return X

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[np.sort(idx)]


def avg_within_distance(X: np.ndarray, metric: str, chunk_size: int) -> float:
    """Compute average pairwise distance within one condition."""
    n = len(X)
    if n < 2:
        return np.nan

    total = 0.0
    count = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D = cdist(X[start:end], X, metric=metric)

        for local_i, global_i in enumerate(range(start, end)):
            vals = D[local_i, global_i + 1 :]
            total += vals.sum()
            count += vals.size

    return float(total / count)


def avg_between_distance(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str,
    chunk_size: int,
) -> float:
    """Compute average pairwise distance between two conditions."""
    if len(X) == 0 or len(Y) == 0:
        return np.nan

    total = 0.0
    count = 0

    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        D = cdist(X[start:end], Y, metric=metric)
        total += D.sum()
        count += D.size

    return float(total / count)


def get_dataset_dirs(cfg: DictConfig, dataset: str) -> dict[str, list[str]]:
    """Get activation dataset directories for each condition."""
    if hasattr(cfg, "dataset_dirs") and dataset in cfg.dataset_dirs:
        return {
            condition: list(cfg.dataset_dirs[dataset][condition])
            for condition in CONDITION_ORDER
        }

    return DEFAULT_DATASET_DIRS[dataset]


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute numerator / denominator, returning NaN for invalid denominators."""
    if denominator is None or np.isnan(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def compute_dataset_distances(cfg: DictConfig, dataset: str) -> dict:
    """Compute within/between activation distances for one model and dataset."""
    model = cfg.model.name
    layer = get_task_layer(cfg, dataset)
    activation_root = resolve_path(cfg.activation_root)
    dataset_dirs = get_dataset_dirs(cfg, dataset)

    max_per_class = int(cfg.max_per_class)
    seed = int(cfg.random_seed)
    metric = str(cfg.metric)
    chunk_size = int(cfg.chunk_size)

    raw_arrays = {}
    scaled_arrays = {}
    row = {
        "dataset": dataset,
        "model": model,
        "layer": layer,
        "metric": metric,
        "max_per_class": max_per_class,
    }

    for condition_idx, condition in enumerate(CONDITION_ORDER):
        X = load_activation_group(
            activation_root=activation_root,
            model=model,
            dataset_dirs=dataset_dirs[condition],
            layer=layer,
        )

        row[f"n_{condition}_raw"] = len(X)

        X = subsample(
            X,
            max_n=max_per_class,
            seed=seed + condition_idx,
        )

        row[f"n_{condition}_used"] = len(X)
        raw_arrays[condition] = X

    X_all = np.vstack([raw_arrays[condition] for condition in CONDITION_ORDER])
    X_all = StandardScaler().fit_transform(X_all)

    start = 0
    for condition in CONDITION_ORDER:
        end = start + len(raw_arrays[condition])
        scaled_arrays[condition] = X_all[start:end]
        start = end

    for condition in CONDITION_ORDER:
        row[f"within_{condition}"] = avg_within_distance(
            scaled_arrays[condition],
            metric=metric,
            chunk_size=chunk_size,
        )

    within_values = [row[f"within_{condition}"] for condition in CONDITION_ORDER]
    row["mean_within"] = float(np.nanmean(within_values))

    between_pairs = [
        ("fictional", "synthetic_fic"),
        ("fictional", "synthetic"),
        ("synthetic_fic", "synthetic"),
    ]

    for c1, c2 in between_pairs:
        between_col = f"between_{c1}_{c2}"
        ratio_col = f"{between_col}_over_mean_within"

        row[between_col] = avg_between_distance(
            scaled_arrays[c1],
            scaled_arrays[c2],
            metric=metric,
            chunk_size=chunk_size,
        )

        row[ratio_col] = safe_ratio(row[between_col], row["mean_within"])

    row["within_fictional_over_synthetic_fic"] = safe_ratio(
        row["within_fictional"],
        row["within_synthetic_fic"],
    )
    row["within_fictional_over_synthetic"] = safe_ratio(
        row["within_fictional"],
        row["within_synthetic"],
    )
    row["within_synthetic_fic_over_synthetic"] = safe_ratio(
        row["within_synthetic_fic"],
        row["within_synthetic"],
    )

    return {col: row.get(col, np.nan) for col in OUTPUT_COLUMNS}


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="within_between",
)
def main(cfg: DictConfig) -> None:
    output_root = resolve_path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log.warning(f"Computing within/between distances for model: {cfg.model.name}")
    log.warning(f"Datasets: {cfg.datasets}")
    log.warning(f"Activation root: {resolve_path(cfg.activation_root)}")
    log.warning(f"Output root: {output_root}")

    rows = []

    for dataset in cfg.datasets:
        log.warning(f"=== Dataset: {dataset} ===")

        try:
            rows.append(compute_dataset_distances(cfg, dataset))
        except Exception as e:
            log.warning(f"Skipping model={cfg.model.name}, dataset={dataset}: {e}")

    if not rows:
        log.warning("No successful rows. Nothing to save.")
        return

    df = pd.DataFrame(rows)
    df = df[OUTPUT_COLUMNS]

    out_path = output_root / f"{cfg.model.name}_within_between.csv"
    df.to_csv(out_path, index=False)
    log.warning(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()