"""
collect_noise_activations.py

Generates synthetic noise dataset and corresponding activations.

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import hydra
from utils import load_data


def estimate_feature_stats(acts_tensor, sample_tokens=50000):
    """
    Estimates the mean and standard deviation of the activation features.

    :param acts_tensor: activations tensor of shape [N, L, H]
    :param sample_tokens: number of tokens to subsample for estimating stats
    :return: (mean, std) arrays of shape [H]
    """
    N, L, H = acts_tensor.shape
    k = min(sample_tokens, N * L)
    idx_n = torch.randint(0, N, (k,))
    idx_l = torch.randint(0, L, (k,))
    sample = acts_tensor[idx_n, idx_l]
    mean = sample.mean(dim=0).cpu().numpy()
    std = sample.std(dim=0).cpu().numpy()
    std[std == 0] = 1e-3
    return mean, std


def length_distribution_from_mask(mask_np):
    """
    Computes a length distribution from an attention mask.

    :param mask_np: numpy array mask of shape [N, L] with 0/1 entries
    :return: (sample_len_fn, max_len) where sample_len_fn draws lengths from the empirical distribution
    """
    lengths = mask_np.sum(axis=1).astype(int)

    def sample_len(size=1):
        return np.random.choice(lengths, size=size, replace=True)

    return sample_len, lengths.max()


def make_noise_arrays(n_rows, hidden_size, sample_len_fn, max_len, mean, std):
    """
    Generates synthetic activation arrays and masks using the learned length and feature distributions.

    :param n_rows: number of synthetic rows to generate
    :param hidden_size: dimensionality of the hidden states
    :param sample_len_fn: function that samples sequence lengths
    :param max_len: maximum sequence length to pad to
    :param mean: per-feature mean vector
    :param std: per-feature standard deviation vector
    :return: (acts, mask) where acts is [N, max_len, H] and mask is [N, max_len]
    """
    acts = np.zeros((n_rows, max_len, hidden_size), dtype=np.float32)
    mask = np.zeros((n_rows, max_len), dtype=np.int32)
    for i in range(n_rows):
        L = int(sample_len_fn(size=1)[0])
        if L < 1:
            L = 1
        toks = np.random.normal(loc=mean, scale=std, size=(L, hidden_size)).astype(
            np.float32
        )
        acts[i, -L:, :] = toks
        mask[i, -L:] = 1
    return acts, mask


def write_noise_activations(root, model, dataset, activation_type, acts_by_layer, mask):
    """
    Saves synthetic activation arrays and a shared attention mask to disk.

    :param root: root directory for activation outputs
    :param model: model name (used as a subdirectory)
    :param dataset: dataset name (used as a subdirectory)
    :param activation_type: e.g., full, last (for MIL or SIL)
    :param acts_by_layer: dict mapping layer_id -> activation array
    :param mask: numpy mask array of shape [N, max_len]
    :return: None
    """
    base = Path(root) / model / dataset / activation_type
    base.mkdir(parents=True, exist_ok=True)

    np.save(base / "mask.npy", mask)
    for layer_id, arr in acts_by_layer.items():
        arr_to_save = arr.astype(np.float16, copy=False)
        np.savez_compressed(base / f"layer_{layer_id}_e.npz", arr_0=arr_to_save)


def write_noise_csv(csv_path, n_rows, category):
    """
    Writes a synthetic noise CSV with a fixed schema, unless the file already exists.

    :param csv_path: path where the CSV should be written
    :param n_rows: number of synthetic rows to create
    :param category: value to place in the 'category' column
    :return: path to the CSV file
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


def generate_noise_dataset(dh, new_ds, n_rows=None, pct_of_train=None, acts_root=None,
                           csv_dir=None, layer_ids=None, sample_tokens=50000, seed=42):
    """
    Generates a synthetic noise dataset aligned with the current data handler splits.

    :param dh: DataHandler object with loaded data and activations
    :param new_ds: new dataset name stem to use for activations and CSV
    :param n_rows: explicit number of noise rows to generate (overrides pct_of_train)
    :param pct_of_train: fraction of total rows to use if n_rows is None
    :param acts_root: root directory where activations will be written
    :param csv_dir: directory where the CSV file will be written
    :param layer_ids: list of layer ids to generate noise activations for
    :param sample_tokens: number of tokens for estimating feature stats
    :param seed: random seed for reproducibility
    :return: path to the generated noise CSV
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
    for layer_id in map(int, layer_ids):
        acts_layer, mask_layer = make_noise_arrays(
            n_rows, hidden_size, sample_len_fn, max_len, mean, std
        )
        acts_by_layer[layer_id] = acts_layer

    write_noise_activations(
        acts_root,
        dh.model,
        new_ds,
        dh.activation_type,
        acts_by_layer,
        mask_layer,
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
