"""
collect_noise_activations.py

Generates synthetic noise dataset and corresponding activations.

This version writes noise activations using the same raw memmap convention
as collect_activations.py:
    outputs/activations/{model}/{dataset}/{activation_type}/
        layer_{layer}_e_temp.npy
        shape.npy
        mask.npy

It infers whether the source activations are float16 or float32 and writes
noise activations using the same dtype.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import hydra
from utils import load_data


def infer_memmap_dtype(path, shape):
    """
    Infer memmap dtype from file size and expected shape.

    Supports legacy float16 activation files and newer float32 activation files.

    Args:
        path: Path to raw memmap activation file.
        shape: Expected activation shape.

    Returns:
        numpy dtype.
    """
    n_values = int(np.prod(shape))
    file_size = os.path.getsize(path)

    expected_float16 = n_values * np.dtype(np.float16).itemsize
    expected_float32 = n_values * np.dtype(np.float32).itemsize

    if file_size == expected_float16:
        return np.float16

    if file_size == expected_float32:
        return np.float32

    raise ValueError(
        f"Could not infer dtype for {path}. "
        f"file_size={file_size}, "
        f"expected_float16={expected_float16}, "
        f"expected_float32={expected_float32}, "
        f"shape={shape}"
    )


def shape_as_tuple(x):
    """
    Convert a shape array loaded from disk into a Python tuple.

    Args:
        x: numpy array encoding a shape.

    Returns:
        Shape tuple.
    """
    x = np.asarray(x)
    if x.shape[0] == 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    if x.shape[0] == 2:
        return (int(x[0]), int(x[1]))
    raise ValueError(f"Unexpected shape array: {x}")


def get_reference_activation_dtype(dh, layer_id):
    """
    Infer the dtype of the real activation file used as the source distribution.

    Args:
        dh: DataHandler.
        layer_id: Layer id.

    Returns:
        numpy dtype.
    """
    dataset = dh.datasets[0]
    data_dir = (
        Path(dh.activations_path)
        / dh.model
        / dataset
        / dh.activation_type
    )

    shape_path = data_dir / "shape.npy"
    temp_path = data_dir / f"layer_{int(layer_id)}_e_temp.npy"
    npz_path = data_dir / f"layer_{int(layer_id)}_e.npz"

    if temp_path.exists() and shape_path.exists():
        shape = shape_as_tuple(np.load(shape_path))
        dtype = infer_memmap_dtype(temp_path, shape)
        print(
            f"Inferred reference dtype for model={dh.model}, dataset={dataset}, "
            f"layer={layer_id}: {np.dtype(dtype)}"
        )
        return dtype

    if npz_path.exists():
        arr = np.load(npz_path)["arr_0"]
        print(
            f"Inferred reference dtype from npz for model={dh.model}, dataset={dataset}, "
            f"layer={layer_id}: {arr.dtype}"
        )
        return arr.dtype

    raise FileNotFoundError(
        f"Could not find reference activation file for dtype inference. "
        f"Tried {temp_path} and {npz_path}"
    )


def estimate_feature_stats(acts_tensor, sample_tokens=50000):
    """
    Estimates the mean and standard deviation of the activation features.

    Args:
        acts_tensor: activations tensor of shape [N, L, H]
        sample_tokens: number of tokens to subsample for estimating stats

    Returns:
        (mean, std) arrays of shape [H]
    """
    N, L, H = acts_tensor.shape
    k = min(sample_tokens, N * L)
    idx_n = torch.randint(0, N, (k,))
    idx_l = torch.randint(0, L, (k,))
    sample = acts_tensor[idx_n, idx_l]

    # Compute stats in float32 even if source activations are float16.
    sample = sample.float()

    mean = sample.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    std = sample.std(dim=0).cpu().numpy().astype(np.float32, copy=False)

    std[std == 0] = 1e-3
    return mean, std


def length_distribution_from_mask(mask_np):
    """
    Computes a length distribution from an attention mask.

    Args:
        mask_np: numpy array mask of shape [N, L] with 0/1 entries

    Returns:
        (sample_len_fn, max_len) where sample_len_fn draws empirical lengths.
    """
    lengths = mask_np.sum(axis=1).astype(int)

    def sample_len(size=1):
        return np.random.choice(lengths, size=size, replace=True)

    return sample_len, lengths.max()


def make_noise_arrays(n_rows, hidden_size, sample_len_fn, max_len, mean, std):
    """
    Generates synthetic activation arrays and masks using the learned length
    and feature distributions.

    Args:
        n_rows: number of synthetic rows to generate
        hidden_size: dimensionality of the hidden states
        sample_len_fn: function that samples sequence lengths
        max_len: maximum sequence length to pad to
        mean: per-feature mean vector
        std: per-feature standard deviation vector

    Returns:
        (acts, mask) where acts is [N, max_len, H] and mask is [N, max_len].
    """
    acts = np.zeros((n_rows, max_len, hidden_size), dtype=np.float32)
    mask = np.zeros((n_rows, max_len), dtype=np.int32)

    for i in range(n_rows):
        L = int(sample_len_fn(size=1)[0])
        if L < 1:
            L = 1

        toks = np.random.normal(
            loc=mean,
            scale=std,
            size=(L, hidden_size),
        ).astype(np.float32, copy=False)

        acts[i, -L:, :] = toks
        mask[i, -L:] = 1

    if not np.isfinite(acts).all():
        n_nan = int(np.isnan(acts).sum())
        n_inf = int(np.isinf(acts).sum())
        raise RuntimeError(
            f"Noise activations contain non-finite values before saving: "
            f"nan={n_nan}, inf={n_inf}"
        )

    return acts, mask


def cast_for_storage(arr, storage_dtype):
    """
    Cast activation array for storage, while refusing unsafe float16 overflow.

    Args:
        arr: Activation array, usually float32.
        storage_dtype: Target dtype.

    Returns:
        Array cast to storage_dtype.
    """
    storage_dtype = np.dtype(storage_dtype)

    if storage_dtype == np.dtype(np.float16):
        max_f16 = np.finfo(np.float16).max
        max_abs = float(np.max(np.abs(arr[np.isfinite(arr)]))) if np.isfinite(arr).any() else np.inf

        if max_abs > max_f16:
            raise RuntimeError(
                f"Refusing to cast noise activations to float16 because "
                f"max_abs={max_abs} exceeds float16 max={max_f16}. "
                f"Regenerate/write this layer as float32 instead."
            )

    arr_to_save = arr.astype(storage_dtype, copy=False)

    if not np.isfinite(arr_to_save).all():
        n_nan = int(np.isnan(arr_to_save).sum())
        n_inf = int(np.isinf(arr_to_save).sum())
        raise RuntimeError(
            f"Non-finite values after casting to {storage_dtype}: "
            f"nan={n_nan}, inf={n_inf}"
        )

    return arr_to_save


def write_noise_activations(
    root,
    model,
    dataset,
    activation_type,
    acts_by_layer,
    mask,
    dtype_by_layer,
):
    """
    Saves synthetic activation arrays and a shared attention mask to disk.

    Uses raw memmap files named layer_{layer_id}_e_temp.npy, matching
    collect_activations.py.

    Args:
        root: root directory for activation outputs
        model: model name
        dataset: dataset name
        activation_type: e.g., full
        acts_by_layer: dict mapping layer_id -> activation array
        mask: numpy mask array of shape [N, max_len]
        dtype_by_layer: dict mapping layer_id -> storage dtype

    Returns:
        None.
    """
    base = Path(root) / model / dataset / activation_type
    base.mkdir(parents=True, exist_ok=True)

    np.save(base / "mask.npy", mask)

    # All generated layers should have the same shape in this script.
    first_arr = next(iter(acts_by_layer.values()))
    np.save(base / "shape.npy", np.array(first_arr.shape, dtype=np.int64))

    for layer_id, arr in acts_by_layer.items():
        storage_dtype = np.dtype(dtype_by_layer[int(layer_id)])
        arr_to_save = cast_for_storage(arr, storage_dtype)

        save_path = base / f"layer_{int(layer_id)}_e_temp.npy"

        mmap = np.memmap(
            save_path,
            dtype=storage_dtype,
            mode="w+",
            shape=arr_to_save.shape,
        )
        mmap[:] = arr_to_save[:]
        mmap.flush()

        print(
            f"Saved noise activations: layer={layer_id}, path={save_path}, "
            f"shape={arr_to_save.shape}, dtype={storage_dtype}, "
            f"bytes={save_path.stat().st_size}"
        )


def write_noise_csv(csv_path, n_rows, category):
    """
    Writes a synthetic noise CSV with a fixed schema, unless the file already exists.

    Args:
        csv_path: path where the CSV should be written
        n_rows: number of synthetic rows to create
        category: value to place in the category column

    Returns:
        Path to the CSV file.
    """
    if os.path.exists(csv_path):
        print(f"File already exists, not overwriting: {csv_path}")
        return csv_path

    cols = [
        "statement",
        "object_1",
        "object_2",
        "correct_object_2",
        "correct",
        "negation",
        "real_object",
        "fake_object",
        "fictional_object",
        "noise_object",
        "category",
    ]

    df = pd.DataFrame(index=range(n_rows))

    df["statement"] = [f"Noise statement {i}." for i in range(n_rows)]
    df["object_1"] = [f"noise_object_{i}" for i in range(n_rows)]
    df["object_2"] = [f"noise_target_{i}" for i in range(n_rows)]
    df["correct_object_2"] = ""

    df["correct"] = 0
    df["negation"] = 0
    df["real_object"] = 0
    df["fake_object"] = 0
    df["fictional_object"] = 0
    df["noise_object"] = 1

    df["category"] = category

    df = df[cols]
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_noise_dataset(
    dh,
    new_ds,
    n_rows=None,
    pct_of_train=None,
    acts_root=None,
    csv_dir=None,
    layer_ids=None,
    sample_tokens=50000,
    seed=42,
):
    """
    Generates a synthetic noise dataset aligned with the current data handler splits.

    Args:
        dh: DataHandler object with loaded data and activations
        new_ds: new dataset name stem to use for activations and CSV
        n_rows: explicit number of noise rows to generate
        pct_of_train: fraction of total rows to use if n_rows is None
        acts_root: root directory where activations will be written
        csv_dir: directory where the CSV file will be written
        layer_ids: list of layer ids to generate noise activations for
        sample_tokens: number of tokens for estimating feature stats
        seed: random seed for reproducibility

    Returns:
        Path to the generated noise CSV.
    """
    if n_rows is None:
        if pct_of_train is None:
            pct_of_train = 0.05
        train_size = dh.train_ids.shape[0]
        cal_size = dh.calibration_ids.shape[0] if dh.calibration_ids is not None else 0
        test_size = dh.test_ids.shape[0]
        total_size = train_size + test_size + cal_size
        n_rows = max(1, int(round(pct_of_train * total_size)))

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    assert layer_ids is not None and len(layer_ids) >= 1, "Need at least one layer id."
    assert dh.activation_type == "full", "Noise generation assumes 'full' activations."

    ref_layer = int(layer_ids[0])
    acts_ref = dh.get_train_acts(layer_id=ref_layer)
    mask_ref = dh.get_train_att_mask().numpy()

    hidden_size = acts_ref.shape[-1]
    mean, std = estimate_feature_stats(acts_ref, sample_tokens=sample_tokens)
    sample_len_fn, _ = length_distribution_from_mask(mask_ref)
    max_len = mask_ref.shape[1]

    acts_by_layer = {}
    dtype_by_layer = {}

    for layer_id in map(int, layer_ids):
        storage_dtype = get_reference_activation_dtype(dh, layer_id)
        dtype_by_layer[layer_id] = storage_dtype

        acts_layer, mask_layer = make_noise_arrays(
            n_rows,
            hidden_size,
            sample_len_fn,
            max_len,
            mean,
            std,
        )
        acts_by_layer[layer_id] = acts_layer

    write_noise_activations(
        acts_root,
        dh.model,
        new_ds,
        dh.activation_type,
        acts_by_layer,
        mask_layer,
        dtype_by_layer=dtype_by_layer,
    )

    csv_path = os.path.join(csv_dir, f"{new_ds}.csv")
    csv_path = write_noise_csv(csv_path, n_rows=n_rows, category=new_ds)

    return csv_path


@hydra.main(version_base=None, config_path="configs", config_name="generate_noise")
def main(cfg):
    dh = load_data(cfg)

    pct_of_train_tag = getattr(cfg, "pct_of_train_tag", 10)
    pct_of_train = pct_of_train_tag / 100.0

    noise_prefix = getattr(cfg.datapack, "noise_prefix")
    new_ds = f"{noise_prefix}_{pct_of_train_tag}"

    acts_root = getattr(cfg, "acts_root", "outputs/activations")
    csv_dir = getattr(cfg, "csv_dir", "datasets")

    data_name = cfg.datapack.name
    layer_dict = cfg.layers
    layer = int(layer_dict[data_name])

    _ = generate_noise_dataset(
        dh,
        new_ds=new_ds,
        pct_of_train=pct_of_train,
        acts_root=acts_root,
        csv_dir=csv_dir,
        layer_ids=[layer],
        seed=42,
    )


if __name__ == "__main__":
    main()