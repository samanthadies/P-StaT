"""
plot_activation_heatmaps.py

Compute sliced Wasserstein distances between activation clouds for
different label categories and visualize them as heatmaps, aggregated
over layers and models and per-model.

This script:
1) Loads activations via the DataHandler and writes per-layer sW1 CSVs.
2) Aggregates sW1 matrices over models for each dataset and plots them.
3) Plots per-model activation distance heatmaps across datasets.

"""

import os
import re
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

from omegaconf import DictConfig
import hydra

from utils import load_data


BASE_CATEGORIES = ["True", "False", "Synthetic", "Fictional", "Disputed", "Noise"]
PLOT_CATEGORIES = ["True", "False", "Synthetic", "Noise", "Fictional"]

MODEL_LEVEL_DIR = "outputs/plots/model_level"

_LAYER_RX = re.compile(r"layer_(r?\d+)_e(?:_temp)?\.(?:npy|npz)$")


def _layers_in_dir(dirpath):
    """
    Return the set of layer indices available under dirpath.

    Matches files named like: layer_<L>_e(_temp).npy|npz

    :param dirpath: directory path to search
    :return: set of integer layer indices
    """
    if not os.path.isdir(dirpath):
        return set()
    layers = set()
    for fname in os.listdir(dirpath):
        m = _LAYER_RX.search(fname)
        if m:
            try:
                layers.add(int(m.group(1)))
            except ValueError:
                # If for some reason layer index is not an int, skip it
                continue
    return layers


def _possible_roots(dh):
    """
    Build a list of candidate activation roots based on the DataHandler.

    This covers the primary DataHandler path plus common fallback paths.

    :param dh: DataHandler-like object (must have activations_path attribute optionally)
    :return: list of candidate root directories (deduplicated, in order)
    """
    roots: List[str] = []
    if hasattr(dh, "activations_path") and isinstance(dh.activations_path, str):
        roots.append(dh.activations_path)

    roots += [
        "outputs/activations",
    ]

    out, seen = [], set()
    for r in roots:
        if r and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def _guess_acts_dir_for_dataset(dh, dataset):
    """
    Find the existing directory that actually contains layer files for this dataset.

    The search pattern is:
        <root>/<model>/<dataset>/<activation_type>/
    and as a last resort:
        <root>/<model>/<dataset>/

    :param dh: DataHandler-like object providing model and activation_type attributes
    :param dataset: dataset name
    :return: directory path with activation files or None if not found
    """
    model = getattr(dh, "model", None)
    act_type = getattr(dh, "activation_type", "full")
    if model is None:
        return None

    for root in _possible_roots(dh):
        d = os.path.join(root, str(model), dataset, act_type)
        if os.path.isdir(d) and _layers_in_dir(d):
            return d

    for root in _possible_roots(dh):
        d = os.path.join(root, str(model), dataset)
        if os.path.isdir(d) and _layers_in_dir(d):
            return d
    return None


def discover_layers_for_dh(dh):
    """
    Discover layer IDs that exist for ALL datasets in dh.datasets.

    Returns a sorted list. If the intersection across datasets is empty,
    returns the non-empty union of layers but prints a warning.

    :param dh: DataHandler-like object with 'datasets' and activations on disk
    :return: list of integer layer ids
    """
    datasets = list(getattr(dh, "datasets", [])) or []
    if not datasets and hasattr(dh, "get_dataframe"):
        df = dh.get_dataframe()
        datasets = sorted(set(df.get("_dataset", ["unknown"])))

    per_ds_layers = []
    for ds in datasets:
        acts_dir = _guess_acts_dir_for_dataset(dh, ds)
        if acts_dir is None:
            continue
        per_ds_layers.append(_layers_in_dir(acts_dir))

    if not per_ds_layers or all(len(s) == 0 for s in per_ds_layers):
        return []

    inter = set.intersection(*[s for s in per_ds_layers if s])
    if inter:
        return sorted(inter)

    union = set().union(*per_ds_layers)
    print(
        f"[warn] No common layer intersection across datasets; "
        f"falling back to union: {sorted(union)}"
    )
    return sorted(union)


def to_bool(x):
    """
    Convert a variety of string/number encodings to boolean.

    Treats {1, 'true', 'yes', ...} as True, {0, 'false', 'no', ...} as False.
    Returns None when conversion is not possible.

    :param x: value to convert
    :return: bool or None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    try:
        return bool(int(s))
    except Exception:
        return None


def derive_label_from_row(row):
    """
    Derive the base (non-negation) label from a dataframe row.

    Labels include:
        - "true"
        - "false"
        - "synthetic"
        - "fictional"
        - "noise"

    :param row: pandas Series representing a row
    :return: lowercase base label string or None
    """
    correct = to_bool(row.get("correct"))
    real = to_bool(row.get("real_object"))
    fict = to_bool(row.get("fictional_object"))
    noise = to_bool(row.get("noise_object"))

    if (correct is True) and (real is True):
        return "true"
    if (correct is False) and (real is True):
        return "false"
    if (real is False) and (fict is False) and (noise is False):
        return "synthetic"
    if fict is True:
        return "fictional"
    if noise is True:
        return "noise"
    return None


def label_from_row(row):
    """
    Produce the final label for a row, including negation.

    The output is:
        "<Base>" or "<Base> - Negated"
    where <Base> is in BASE_CATEGORIES (capitalized).

    :param row: pandas Series representing a row
    :return: label string or None
    """
    base = row.get("_label") or derive_label_from_row(row)
    if base is None:
        return None
    is_neg = to_bool(row.get("negation")) is True
    base_cap = base.capitalize()
    return f"{base_cap} - Negated" if is_neg else base_cap


def base_label(lbl):
    """
    Strip any negation suffix and map a label to its base category.

    :param lbl: label string (e.g., "True - Negated" or "True")
    :return: base label in BASE_CATEGORIES or None
    """
    if lbl is None:
        return None
    base = str(lbl).split(" - ")[0].strip()
    return base if base in BASE_CATEGORIES else None


def reduce_full_activations_to_2d(A, pool="last_nonzero"):
    """
    Reduce 3D (N, L, H) activations to 2D (N, H) via pooling.

    Supported pooling modes:
        - "last": last token
        - "last_nonzero": last non-zero token (fallback to last if all zero)
        - "mean_nonzero": mean over non-zero tokens (fallback to last if all zero)
        - "max_nonzero": max over non-zero tokens (fallback to last if all zero)

    :param A: activations array of shape (N, L, H)
    :param pool: pooling strategy
    :return: pooled activations of shape (N, H)
    """
    if A.ndim != 3:
        raise ValueError(f"Expected 3D (N,L,H) activations; got {A.shape}")
    N, L, H = A.shape
    if pool == "last":
        return A[:, -1, :]

    nz = (A != 0).any(axis=2)  # (N, L)

    if pool == "last_nonzero":
        idx = np.full(N, L - 1, dtype=int)
        for n in range(N):
            rows = np.where(nz[n])[0]
            if rows.size:
                idx[n] = rows[-1]
        return A[np.arange(N), idx, :]

    if pool == "mean_nonzero":
        out = np.zeros((N, H), dtype=A.dtype)
        for n in range(N):
            rows = nz[n]
            out[n] = A[n, rows, :].mean(axis=0) if rows.any() else A[n, -1, :]
        return out

    if pool == "max_nonzero":
        out = np.zeros((N, H), dtype=A.dtype)
        for n in range(N):
            rows = nz[n]
            out[n] = A[n, rows, :].max(axis=0) if rows.any() else A[n, -1, :]
        return out

    raise ValueError(f"Unknown pool='{pool}'")


def build_pointclouds_by_label(
    X_scaled,
    labels,
    max_per_class=None,
):
    """
    Construct per-label point clouds from scaled activations.

    :param X_scaled: 2D array of shape (N, H) with scaled activations
    :param labels: 1D array-like of length N with label strings
    :param max_per_class: maximum number of points to keep per class (randomly subsampled)
    :return: dict mapping base label -> 2D array of points
    """
    by = {k: [] for k in BASE_CATEGORIES}
    for i, lbl in enumerate(labels):
        b = base_label(lbl)
        if b is not None:
            by[b].append(i)

    clouds: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)
    for k, idxs in by.items():
        if not idxs:
            continue
        idxs_arr = np.array(idxs, dtype=int)
        if (max_per_class is not None) and (len(idxs_arr) > max_per_class):
            idxs_arr = rng.choice(idxs_arr, size=max_per_class, replace=False)
        clouds[k] = X_scaled[idxs_arr]
    return clouds


def sliced_wasserstein_distance(A, B, n_projections=128):
    """
    Approximate Wasserstein-1 distance using random 1D projections.

    :param A: 2D array of shape (n_samples_A, n_features)
    :param B: 2D array of shape (n_samples_B, n_features)
    :param n_projections: number of random projection directions
    :return: average 1D Wasserstein distance over projections
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D (n_samples, n_features).")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same feature dimension.")

    d = A.shape[1]
    rng = np.random.default_rng(123)
    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    acc = 0.0
    for v in dirs:
        a = A @ v
        b = B @ v
        acc += wasserstein_distance(np.asarray(a), np.asarray(b))
    return acc / float(n_projections)


def compute_wasserstein_matrix(
    clouds,
    categories=BASE_CATEGORIES,
    n_projections=128,
):
    """
    Compute the pairwise sliced Wasserstein distance matrix across categories.

    :param clouds: dict mapping base label -> 2D array of points
    :param categories: ordered list of categories to include
    :param n_projections: number of projections for sW1
    :return: square DataFrame indexed and columned by categories
    """
    C = [c for c in categories]
    M = np.full((len(C), len(C)), np.nan, dtype=float)

    for i, ci in enumerate(C):
        Ai = clouds.get(ci)
        if Ai is None or len(Ai) == 0:
            continue
        for j, cj in enumerate(C):
            if j < i:
                M[i, j] = M[j, i]
                continue
            Bj = clouds.get(cj)
            if Bj is None or len(Bj) == 0:
                continue
            M[i, j] = 0.0 if i == j else sliced_wasserstein_distance(
                Ai, Bj, n_projections=n_projections
            )
            M[j, i] = M[i, j]

    return pd.DataFrame(M, index=C, columns=C)


def load_and_label_with_dataset_from_dh(dh):
    """
    Load the main dataframe from the DataHandler and attach dataset/label columns.

    Adds:
        - "_dataset" column (if missing, fills with "unknown")
        - "_label" column derived from per-row metadata

    :param dh: DataHandler-like object with get_dataframe()
    :return: labeled pandas DataFrame
    """
    df = dh.get_dataframe().copy()
    if "_dataset" not in df.columns:
        df["_dataset"] = df.get("_dataset", pd.Series(["unknown"] * len(df)))
    df["_label"] = df.apply(label_from_row, axis=1)
    df = df[df["_label"].notna()].reset_index(drop=True)
    return df


def generate_wasserstein_csvs(
    cfg,
    out_dir=MODEL_LEVEL_DIR,
    pool_3d="last_nonzero",
    n_projections=128,
    max_per_class=5000,
):
    """
    Generate per-layer sliced Wasserstein matrices and save them as CSVs.

    The naming convention is:
        <model_name>_<dataset_tag>_layer<layer>_sW1.csv

    :param cfg: Hydra configuration object (expects cfg.model.name, cfg.datapack.name, cfg.layers)
    :param out_dir: output directory for the CSVs
    :param pool_3d: pooling method for 3D activations
    :param n_projections: number of random projections for sW1
    :param max_per_class: maximum samples per class for point clouds
    :return: None
    """
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(cfg.model, str):
        model_name = str(cfg.model)
    else:
        model_name = str(cfg.model.get("name", "model"))

    try:
        dataset_tag = str(cfg.datapack.name)
    except Exception:
        dataset_tag = "dataset"

    dh = load_data(cfg)

    data_name = cfg.datapack.name
    layer_dict = cfg.layers
    layer = int(layer_dict[data_name])

    df = load_and_label_with_dataset_from_dh(dh)
    labels_full = df["_label"].values

    try:
        X_t = dh.get_activations(layer_id=layer, module="e")
    except Exception as e:
        print(f"[skip] layer {layer}: get_activations failed ({e})")
        return

    X = np.array(X_t, dtype=np.float32)
    if X.ndim == 3:
        X = reduce_full_activations_to_2d(X, pool=pool_3d)
    elif X.ndim != 2:
        print(f"[skip] layer {layer}: unsupported activation shape {X.shape}")
        return

    if X.shape[0] != len(df):
        print(
            f"[skip] layer {layer}: rows mismatch acts={X.shape[0]} vs df={len(df)}"
        )
        return

    X_scaled = StandardScaler().fit_transform(X)
    clouds = build_pointclouds_by_label(
        X_scaled, labels_full, max_per_class=max_per_class
    )
    M = compute_wasserstein_matrix(
        clouds, categories=BASE_CATEGORIES, n_projections=n_projections
    )

    csv_path = os.path.join(
        out_dir, f"{model_name}_{dataset_tag}_layer{layer}_sW1.csv"
    )
    M.to_csv(csv_path, index=True)
    print(f"[sW1] wrote {csv_path}")


def collect_dataset_mats_for_model(
    root_dir,
    dataset,
    model_name,
):
    """
    Collect all sW1 matrices for a given (model, dataset) from disk.

    Pattern:
        "<model_name>_<dataset>_layer<digits>_sW1.csv"

    :param root_dir: directory to scan
    :param dataset: dataset tag used in filenames
    :param model_name: model name prefix used in filenames
    :return: list of DataFrames (one per layer)
    """
    pat = re.compile(
        rf"^{re.escape(model_name)}_{re.escape(dataset)}_layer\d+_sW1\.csv$"
    )
    mats: List[pd.DataFrame] = []
    for p in sorted(glob.glob(os.path.join(root_dir, "*_sW1.csv"))):
        fname = os.path.basename(p)
        if pat.match(fname):
            mats.append(read_sW1_csv(p, categories=BASE_CATEGORIES))
    return mats


def plot_single_model_activation_heatmaps(
    model_name,
    datasets,
    root_dir,
    output_fp,
    noise,
):
    """
    Plot per-model activation Wasserstein distance heatmaps across datasets.

    For each dataset, this function:
        - loads sW1 matrices across layers for the given model
        - averages them
        - enforces zero diagonal
        - restricts to PLOT_CATEGORIES
        - plots a heatmap in a 1x3 grid figure

    :param model_name: model name whose CSVs to read
    :param datasets: list of dataset tags
    :param root_dir: directory containing per-layer sW1 CSVs
    :param output_fp: output directory for the combined figure
    :param noise: noise parameter used only for filename display
    :return: None
    """
    label_order = ['(a)', '(b)', '(c)']

    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=3,
        ncols=3,
        height_ratios=[0.10, 0.83, 0.07],
    )
    ax_title = fig.add_subplot(gs[0, :])

    matrices_to_plot: Dict[str, pd.DataFrame] = {}
    global_max = 0.0

    for ds_name in datasets:
        mats = collect_dataset_mats_for_model(root_dir, ds_name, model_name)
        if not mats:
            print(
                f"[single-model] No matrices found for model='{model_name}', "
                f"dataset='{ds_name}' under {root_dir}"
            )
            continue

        C_full = len(BASE_CATEGORIES)
        stack = np.full((len(mats), C_full, C_full), np.nan, dtype=float)
        for k, M in enumerate(mats):
            M = M.reindex(index=BASE_CATEGORIES, columns=BASE_CATEGORIES).astype(float)
            stack[k] = M.values

        mean_mat = np.nanmean(stack, axis=0)

        np.fill_diagonal(mean_mat, 0.0)

        M_mean = pd.DataFrame(mean_mat, index=BASE_CATEGORIES, columns=BASE_CATEGORIES)
        M_mean_plot = M_mean.loc[PLOT_CATEGORIES, PLOT_CATEGORIES]
        matrices_to_plot[ds_name] = M_mean_plot

        local_max = np.nanmax(M_mean_plot.values)
        if np.isfinite(local_max):
            global_max = max(global_max, float(local_max))

    if not matrices_to_plot:
        print(f"[single-model] No matrices at all for model='{model_name}'. Nothing to plot.")
        return

    ds_idx = 0
    last_im = None

    for ds_name in datasets:
        if ds_name not in matrices_to_plot:
            continue

        M_mean_plot = matrices_to_plot[ds_name]
        C_plot = len(PLOT_CATEGORIES)

        ax_subplot = fig.add_subplot(gs[1, ds_idx])

        last_im = ax_subplot.imshow(
            M_mean_plot.values,
            aspect="equal",
            cmap="YlGn_r",
            vmin=0.0,
            vmax=global_max,
        )

        if ds_name == 'cities_loc':
            data_name = 'City Locations'
        elif ds_name == 'med_indications':
            data_name = 'Medical Indications'
        elif ds_name == 'defs':
            data_name = 'Word Definitions'
        else:
            data_name = ds_name

        subplot_text = f'{label_order[ds_idx]} {data_name}'

        ax_subplot.set_xticks(np.arange(C_plot))
        ax_subplot.set_yticks(np.arange(C_plot))
        ax_subplot.set_xticklabels(M_mean_plot.columns, rotation=45, ha="right")
        ax_subplot.set_yticklabels(M_mean_plot.index)

        ax_subplot.text(
            -0.40,
            1.15,
            subplot_text,
            transform=ax_subplot.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='#444444',
        )

        ds_idx += 1

    ax_title.set_axis_off()
    title = f'{model_name}: Wasserstein Distance between Activations'
    ax_title.text(
        -0.1,
        0.25,
        title,
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    if last_im is not None:
        cax = fig.add_subplot(gs[2, :])
        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.set_label(
            "Average Wasserstein Distance",
            rotation=0,
            va="center",
            labelpad=10,
        )

    os.makedirs(output_fp, exist_ok=True)
    fp = os.path.join(
        output_fp,
        f"activation_heatmaps_noise{noise}_{model_name}.pdf",
    )
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_sW1_csv(path, categories=BASE_CATEGORIES):
    """
    Read a sliced-Wasserstein CSV and coerce it to a square categories x categories matrix.

    Attempts:
        1) Read with index_col=0 and check presence of category names.
        2) Read without index_col; if columns and rows match categories, use them.
        3) As a last resort, coerce by position (top-left block).

    :param path: CSV file path
    :param categories: ordered category list
    :return: DataFrame with categories as index and columns
    """
    try:
        df = pd.read_csv(path, index_col=0)
        if set(categories).issubset(df.columns) and set(categories).issubset(df.index):
            return df.reindex(index=categories, columns=categories).astype(float)
    except Exception:
        pass

    df = pd.read_csv(path)
    if set(categories).issubset(df.columns) and df.shape[0] == len(categories):
        df = df.loc[:, categories].copy()
        df.index = categories
        return df.reindex(index=categories, columns=categories).astype(float)

    out = pd.DataFrame(
        np.nan,
        index=categories,
        columns=categories,
        dtype=float,
    )
    r = min(df.shape[0], len(categories))
    c = min(df.shape[1], len(categories))
    out.iloc[:r, :c] = df.iloc[:r, :c].values
    return out


def collect_dataset_mats(root_dir, dataset):
    """
    Collect all sW1 matrices for a given dataset across models.

    Pattern:
        "..._<dataset>_layer<digits>_sW1.csv"

    :param root_dir: directory to scan
    :param dataset: dataset tag used in filenames
    :return: list of DataFrames (one per model-layer)
    """
    pat = re.compile(rf"^.+_{re.escape(dataset)}_layer\d+_sW1\.csv$")
    mats: List[pd.DataFrame] = []
    for p in sorted(glob.glob(os.path.join(root_dir, "*_sW1.csv"))):
        fname = os.path.basename(p)
        if pat.match(fname):
            mats.append(read_sW1_csv(p, categories=BASE_CATEGORIES))
    return mats


def plot_combined_activation_heatmaps(
    datasets,
    root_dir,
    output_fp,
    noise,
):
    """
    Plot activation Wasserstein distance heatmaps aggregated over models.

    For each dataset:
        - collects all model-layer sW1 matrices
        - computes the mean and count matrices
        - enforces zero diagonal on the mean
        - saves full mean & count CSVs
        - plots a PLOT_CATEGORIES x PLOT_CATEGORIES heatmap in a 1x3 grid

    :param datasets: list of dataset tags
    :param root_dir: directory containing per-layer sW1 CSVs for all models
    :param output_fp: output directory for the combined figure
    :param noise: noise parameter used in the output filename
    :return: None
    """
    label_order = ['(a)', '(b)', '(c)']

    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=3,
        ncols=3,
        height_ratios=[0.10, 0.83, 0.07],
    )
    ax_title = fig.add_subplot(gs[0, :])

    matrices_to_plot: Dict[str, pd.DataFrame] = {}
    global_max = 0.0

    for ds_name in datasets:
        mats = collect_dataset_mats(root_dir, ds_name)
        if not mats:
            print(
                f"[aggregate] No matrices found for dataset='{ds_name}' "
                f"under {root_dir}"
            )
            continue

        C_full = len(BASE_CATEGORIES)
        stack = np.full((len(mats), C_full, C_full), np.nan, dtype=float)
        for k, M in enumerate(mats):
            M = M.reindex(index=BASE_CATEGORIES, columns=BASE_CATEGORIES).astype(float)
            stack[k] = M.values

        mean_mat = np.nanmean(stack, axis=0)
        count_mat = np.sum(np.isfinite(stack), axis=0)

        np.fill_diagonal(mean_mat, 0.0)

        M_mean = pd.DataFrame(mean_mat, index=BASE_CATEGORIES, columns=BASE_CATEGORIES)
        M_counts = pd.DataFrame(count_mat, index=BASE_CATEGORIES, columns=BASE_CATEGORIES)

        out_root = 'outputs/plots'
        os.makedirs(out_root, exist_ok=True)
        mean_csv = os.path.join(out_root, f"aggregated_{ds_name}_sW1.csv")
        counts_csv = os.path.join(out_root, f"aggregated_{ds_name}_sW1_counts.csv")

        M_mean.to_csv(mean_csv, index=True)
        M_counts.to_csv(counts_csv, index=True)

        M_mean_plot = M_mean.loc[PLOT_CATEGORIES, PLOT_CATEGORIES]
        matrices_to_plot[ds_name] = M_mean_plot

        local_max = np.nanmax(M_mean_plot.values)
        if np.isfinite(local_max):
            global_max = max(global_max, float(local_max))

    if not matrices_to_plot:
        print("[aggregate] No matrices to plot at all. Skipping combined heatmap.")
        return

    ds_num = 0
    last_im = None

    for ds_name in datasets:
        if ds_name not in matrices_to_plot:
            continue

        M_mean_plot = matrices_to_plot[ds_name]
        C_plot = len(PLOT_CATEGORIES)

        ax_subplot = fig.add_subplot(gs[1, ds_num])

        last_im = ax_subplot.imshow(
            M_mean_plot.values,
            aspect="equal",
            cmap="YlGn_r",
            vmin=0.0,
            vmax=global_max,
        )

        if ds_name == 'cities_loc':
            data_name = 'City Locations'
        elif ds_name == 'med_indications':
            data_name = 'Medical Indications'
        elif ds_name == 'defs':
            data_name = 'Word Definitions'
        else:
            data_name = ds_name

        subplot_text = f'{label_order[ds_num]} {data_name}'

        ax_subplot.set_xticks(np.arange(C_plot))
        ax_subplot.set_yticks(np.arange(C_plot))
        ax_subplot.set_xticklabels(
            M_mean_plot.columns,
            rotation=45,
            ha="right",
        )
        ax_subplot.set_yticklabels(M_mean_plot.index)

        ax_subplot.text(
            -0.40,
            1.15,
            subplot_text,
            transform=ax_subplot.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='#444444',
        )

        ds_num += 1

    ax_title.set_axis_off()
    title = 'Wasserstein Distance between Activations (Averaged over LLMs)'
    ax_title.text(
        -0.1,
        0.25,
        title,
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    if last_im is not None:
        cax = fig.add_subplot(gs[2, :])
        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.set_label(
            "Average Wasserstein Distance",
            rotation=0,
            va="center",
            labelpad=10,
        )

    os.makedirs(output_fp, exist_ok=True)
    fp = os.path.join(output_fp, f"activation_heatmaps_noise{noise}.pdf")
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="probe_linear_mil")
def main(cfg: DictConfig):

    # 1) Generate the per-layer sW1 CSVs this script later plots
    generate_wasserstein_csvs(
        cfg,
        out_dir=MODEL_LEVEL_DIR,
        pool_3d="last_nonzero",
        n_projections=128,
        max_per_class=5000,
    )

    # 2) Aggregate + plot across models (uses files in MODEL_LEVEL_DIR)
    datasets = ['cities_loc', 'med_indications', 'defs']
    plot_combined_activation_heatmaps(
        datasets,
        MODEL_LEVEL_DIR,
        'outputs/plots/',
        cfg.noise,
    )

    # 3) Per-model plots
    models = [
        '_gemma-2-9b', '_gemma-7b', '_llama-3-8b-med', '_llama-3.1-8b',
        '_llama-3.1-8b-bio', '_llama-3.2-3b', '_mistral-7B-v0.3',
        '_qwen-2.5-14b', '_qwen-2.5-7b',
        'gemma-2-9b', 'gemma-7b', 'llama-3-8b', 'llama-3.2-3b',
        'mistral-7B-v0.3', 'qwen-2.5-14b', 'qwen-2.5-7b',
    ]
    for model_name in models:
        plot_single_model_activation_heatmaps(
            model_name=model_name,
            datasets=datasets,
            root_dir=MODEL_LEVEL_DIR,
            output_fp=MODEL_LEVEL_DIR,
            noise=cfg.noise,
        )


if __name__ == '__main__':
    main()