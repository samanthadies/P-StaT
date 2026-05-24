"""
data_handler.py

Handle loading, splitting, and augmenting dataset tables
and their corresponding hidden activations.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import os
import logging
from glob import glob

import polars as pl
import pandas as pd
import torch
import numpy as np

LEGAL_ACTIVATION_TYPES = ["last", "mean", "max", "full"]

log = logging.getLogger(__name__)


LABEL_COLUMNS = [
    "correct",
    "real_object",
    "fake_object",
    "negation",
    "fictional_object",
    "disputed_object",
    "noise_object",
    "synthetic_fic_object",
]

SPLIT_COLUMNS = ["in_train", "in_test", "in_cal"]


def shape_as_tuple(x):
    """
    Convert a shape array loaded from disk into a Python tuple.

    :param x: numpy array encoding a shape (length 2 or 3)
    :return: tuple representing the shape
    """
    d = x.shape[0]
    if d == 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    elif d == 2:
        return (int(x[0]), int(x[1]))
    else:
        raise Exception("Number of dimensions is too low")


def infer_memmap_dtype(path, shape):
    """
    Infer memmap dtype from file size and expected shape.

    Supports legacy float16 activation files and newer float32 activation files.

    :param path: path to raw memmap activation file
    :param shape: expected activation array shape
    :return: numpy dtype, either np.float16 or np.float32
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


def remove_padded(x):
    """
    Remove padded rows from a 3D tensor by dropping rows equal to the first row.

    :param x: tensor with padded rows at the top
    :return: tensor with padded rows removed
    """
    mask = (x - x[0]).sum(1) != 0
    return x[mask]


def stack_tensors(tensors, padding_value=0, max_length=None):
    """
    Stack a list of variable-length 3D tensors along the batch dimension with padding.

    :param tensors: list of 3D tensors [N_i, L_i, H] to be stacked
    :param padding_value: scalar value to use for padding shorter sequences
    :param max_length: optional maximum length to pad/truncate to; inferred if None
    :return: single 3D tensor [sum_i N_i, max_length, H] with padding applied
    """
    if max_length is None:
        max_length = max(tensor.size(1) for tensor in tensors)

    padded_tensors = []
    for tensor in tensors:
        pad_size = max_length - tensor.size(1)
        if pad_size > 0:
            pad_tensor = torch.full(
                (tensor.size(0), pad_size, tensor.size(2)),
                padding_value,
                dtype=tensor.dtype,
            )
            padded_tensor = torch.cat((tensor, pad_tensor), dim=1)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.vstack(padded_tensors)
    return stacked_tensor


class DataHandler:
    """
    Manage datasets, splits, and aligned activations for probing experiments.

    :param model: model name used in activation paths
    :param datasets: list of base dataset names
    :param datasets_fictional: list of fictional dataset names
    :param datasets_noise: list of noise dataset names
    :param datasets_synthetic_fic: list of synthetic_fic dataset names
    :param use_fictional: whether to augment with fictional datasets
    :param use_noise: whether to augment with noise datasets
    :param use_synthetic_fic: whether to augment with synthetic_fic datasets
    :param activation_type: type of activations ("last", "mean", "max", "full")
    :param dataset_path: root directory for CSV datasets
    :param activations_path: root directory for activation arrays
    :param with_calibration: whether to create a calibration split
    :param load_scores: optional scores identifier for loading extra columns
    :param output_path: root directory for general outputs
    :param verbose: whether to log informational messages
    """

    def __init__(
        self,
        model="llama-3-8b",
        datasets=None,
        datasets_fictional=None,
        datasets_noise=None,
        datasets_synthetic_fic=None,
        use_fictional=False,
        use_noise=False,
        use_synthetic_fic=False,
        activation_type="last",
        dataset_path="datasets/",
        activations_path="outputs/activations/",
        with_calibration=False,
        load_scores="",
        output_path="outputs/",
        verbose=True,
    ):
        if datasets is None:
            datasets = ["cities_combined", "cities_fake"]
        if datasets_fictional is None:
            datasets_fictional = []
        if datasets_noise is None:
            datasets_noise = []
        if datasets_synthetic_fic is None:
            datasets_synthetic_fic = []

        if activation_type not in LEGAL_ACTIVATION_TYPES:
            raise ValueError(
                "Activation type must be one of {}".format(LEGAL_ACTIVATION_TYPES)
            )

        self.model = model
        self.datasets = list(datasets)
        self.datasets_fictional = list(datasets_fictional)
        self.datasets_noise = list(datasets_noise)
        self.datasets_synthetic_fic = list(datasets_synthetic_fic)

        self.use_fictional = use_fictional
        self.use_noise = use_noise
        self.use_synthetic_fic = use_synthetic_fic

        self.activation_type = activation_type
        self.dataset_path = dataset_path
        self.activations_path = activations_path
        self.with_calibration = with_calibration
        self.load_scores = load_scores
        self.output_path = output_path
        self.verbose = verbose

        self.base_datasets = list(self.datasets)

        self.data = None
        self.train_ids = None
        self.test_ids = None
        self.calibration_ids = None

    def assemble(
        self,
        exclusive_split,
        test_size=0.2,
        calibration_size=0.2,
        seed=42,
        shuffle=True,
    ):
        """
        Load base datasets, create splits, and optionally augment with
        fictional, noise, and synthetic_fic datasets.

        :param exclusive_split: if True, enforce disjoint objects across train/test
        :param test_size: fraction of base rows for the test split
        :param calibration_size: fraction of train rows for the calibration split
        :param seed: random seed for all split operations
        :param shuffle: whether to shuffle split indices
        :return: None
        """
        data_frames = []
        columns = self.column_list()

        for dataset in self.base_datasets:
            df = pl.read_csv(f"{self.dataset_path}{dataset}.csv")

            for col_name in LABEL_COLUMNS + SPLIT_COLUMNS:
                if col_name in df.columns:
                    df = df.with_columns(pl.col(col_name).cast(pl.Int32()))

            if self.load_scores not in ["", None, False]:
                try:
                    scores = np.load(
                        f"outputs/probes/prompt/{self.load_scores}/{self.model}/{dataset}/scores.npy"
                    )
                    n_scores = scores.shape[-1]
                    df = df.with_columns(
                        [pl.Series(f"scores_{i}", scores[:, i]) for i in range(n_scores)]
                    )
                except Exception as e:
                    log.error(e)
                    log.error(
                        "Scores not found for model {} and dataset {}.".format(
                            self.model, dataset
                        )
                    )

            missing_columns = set(columns).difference(set(df.columns))
            for column in missing_columns:
                df = df.with_columns(pl.lit(0.0).alias(column))

            data_frames.append(df.select(sorted(df.columns)))

        data = pl.concat(data_frames, how="vertical_relaxed").to_pandas()
        self.data = data
        self.datasets = list(self.base_datasets)

        if self.verbose:
            log.warning("Base datasets assembled.")

        if exclusive_split:
            self.exclusive_data_split(
                test_size=test_size,
                calibration_size=calibration_size,
                seed=seed,
                shuffle=shuffle,
            )
        else:
            self.data_split(
                test_size=test_size,
                calibration_size=calibration_size,
                seed=seed,
                shuffle=shuffle,
            )

        if self.use_fictional and self.datasets_fictional:
            self.augment_with_split(
                extra_datasets=self.datasets_fictional,
                test_size=test_size,
                cal_size=calibration_size,
                seed=seed,
                indicator_col="fictional_object",
            )

        if self.use_noise and self.datasets_noise:
            self.augment_with_split(
                extra_datasets=self.datasets_noise,
                test_size=test_size,
                cal_size=calibration_size,
                seed=seed,
                indicator_col="noise_object",
            )

        if self.use_synthetic_fic and self.datasets_synthetic_fic:
            self.augment_with_split(
                extra_datasets=self.datasets_synthetic_fic,
                test_size=test_size,
                cal_size=calibration_size,
                seed=seed,
                indicator_col="synthetic_fic_object",
            )

    def exclusive_data_split(
        self,
        test_size=0.2,
        calibration_size=0.2,
        seed=42,
        shuffle=True,
    ):
        """
        Split base data so that objects in train do not appear in test,
        with optional calibration split.

        :param test_size: fraction of base rows for the test split
        :param calibration_size: fraction of remaining rows for the calibration split
        :param seed: random seed for the split and shuffling
        :param shuffle: whether to shuffle indices after splitting
        :return: None
        """
        df_train, df_test = self.generate_exclusive_split(
            self.data, test_size, seed
        )

        if self.with_calibration:
            df_train, df_calib = self.generate_exclusive_split(
                df_train, calibration_size, seed
            )
            self.calibration_ids = np.array(df_calib.index)
        else:
            self.calibration_ids = None

        self.train_ids = np.array(df_train.index)
        self.test_ids = np.array(df_test.index)

        if shuffle:
            np.random.seed(seed + 1)
            np.random.shuffle(self.train_ids)
            np.random.shuffle(self.test_ids)
            if self.with_calibration and self.calibration_ids is not None:
                np.random.shuffle(self.calibration_ids)

        if self.verbose:
            train_size_ratio = len(self.train_ids) / self.data.shape[0]
            test_size_ratio = len(self.test_ids) / self.data.shape[0]

            if self.with_calibration:
                calib_size_ratio = len(self.calibration_ids) / self.data.shape[0]
                log.warning(
                    "Train size: {:.2f}, Test size: {:.2f}, Calibration size: {:.2f}".format(
                        train_size_ratio,
                        test_size_ratio,
                        calib_size_ratio,
                    )
                )
            else:
                log.warning(
                    "Train size: {:.2f}, Test size: {:.2f}, Calibration size: 0.0".format(
                        train_size_ratio,
                        test_size_ratio,
                    )
                )

    def generate_exclusive_split(self, df, test_size, seed):
        """
        Create an exclusive train/test-style split where objects do not overlap
        between subsets.

        :param df: dataframe to split
        :param test_size: target fraction of rows assigned to the second subset
        :param seed: random seed for sampling and refinement
        :return: (df_train, df_test) dataframes with disjoint objects
        """
        rnd = np.random.default_rng(seed)
        train_objects = (
            df[["object_1", "object_2"]]
            .drop_duplicates()
            .sample(frac=1.0 - test_size, random_state=seed)
            .to_numpy()
            .flatten()
        )

        train_mask = df["object_1"].isin(train_objects) | df["object_2"].isin(
            train_objects
        )
        df_train = df[train_mask]
        df_test = df[~train_mask]

        while True:
            if df_test.shape[0] / df.shape[0] > test_size:
                break

            train_objects = rnd.choice(
                train_objects,
                size=int(len(train_objects) * 0.975),
                replace=False,
            )
            train_mask = df["object_1"].isin(train_objects) | df["object_2"].isin(
                train_objects
            )
            df_train = df[train_mask]
            df_test = df[~train_mask]

        return df_train, df_test

    def data_split(
        self,
        test_size=0.2,
        calibration_size=0.2,
        seed=42,
        shuffle=True,
    ):
        """
        Randomly split base data into train, test, and optional calibration sets.

        :param test_size: fraction of rows for the test split
        :param calibration_size: fraction of train rows for the calibration split
        :param seed: random seed for the split and shuffling
        :param shuffle: whether to shuffle indices
        :return: None
        """
        np.random.seed(seed)
        ids = np.arange(len(self.data))
        mask = np.random.rand(len(self.data)) < 1 - test_size

        train_ids = ids[mask]
        test_ids = ids[~mask]

        if self.with_calibration:
            cal_mask = np.random.rand(len(train_ids)) < 1 - calibration_size
            self.train_ids = train_ids[cal_mask]
            self.calibration_ids = train_ids[~cal_mask]
        else:
            self.train_ids = train_ids
            self.calibration_ids = None

        self.test_ids = test_ids

        if shuffle:
            np.random.seed(seed + 1)
            np.random.shuffle(self.train_ids)
            np.random.shuffle(self.test_ids)

            if self.with_calibration and self.calibration_ids is not None:
                np.random.shuffle(self.calibration_ids)

        if self.verbose:
            train_size_ratio = len(self.train_ids) / self.data.shape[0]
            test_size_ratio = len(self.test_ids) / self.data.shape[0]

            if self.with_calibration:
                calib_size_ratio = len(self.calibration_ids) / self.data.shape[0]
                log.warning(
                    "Train size: {:.2f}, Test size: {:.2f}, Calibration size: {:.2f}".format(
                        train_size_ratio,
                        test_size_ratio,
                        calib_size_ratio,
                    )
                )
            else:
                log.warning(
                    "Train size: {:.2f}, Test size: {:.2f}, Calibration size: 0.0".format(
                        train_size_ratio,
                        test_size_ratio,
                    )
                )

    def three_way_random_split(
        self,
        n_rows,
        test_size,
        cal_size,
        seed,
    ):
        """
        Randomly split local indices into train, test, and optional calibration sets.

        :param n_rows: number of rows to split (0..n_rows-1)
        :param test_size: global fraction of rows to allocate to test
        :param cal_size: global fraction of rows to allocate to calibration
        :param seed: random seed for shuffling
        :return: (train_local, test_local, cal_local) index arrays
        """
        rng = np.random.default_rng(seed)
        all_idx = np.arange(n_rows, dtype=int)
        rng.shuffle(all_idx)

        n_test = int(round(test_size * n_rows))
        n_cal = (
            int(round(cal_size * n_rows))
            if self.with_calibration and cal_size > 0.0
            else 0
        )

        n_test = max(0, min(n_test, n_rows))
        n_cal = max(0, min(n_cal, n_rows - n_test))
        n_train = n_rows - n_test - n_cal

        train_local = all_idx[:n_train]
        test_local = all_idx[n_train : n_train + n_test]
        cal_local = (
            all_idx[n_train + n_test : n_train + n_test + n_cal]
            if self.with_calibration and n_cal > 0
            else None
        )

        return train_local, test_local, cal_local

    def csv_to_aligned_df(self, csv_path, columns_like):
        """
        Load a CSV and align its columns and label dtypes to match an existing
        base dataframe.

        :param csv_path: path to the CSV file to load
        :param columns_like: reference dataframe whose columns should be matched
        :return: pandas dataframe aligned to the reference schema
        """
        df_pl = pl.read_csv(csv_path)

        for col_name in LABEL_COLUMNS + SPLIT_COLUMNS:
            if col_name in df_pl.columns:
                df_pl = df_pl.with_columns(pl.col(col_name).cast(pl.Int32()))

        df = df_pl.to_pandas()
        df = df.reindex(columns=columns_like.columns, fill_value=0)

        for col_name in LABEL_COLUMNS + SPLIT_COLUMNS:
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(0).astype("int32")

        return df

    def _ensure_schema_columns(self):
        """
        Ensure that self.data contains label and split columns needed for
        augmentation and downstream task construction.

        :return: None
        """
        for col_name in LABEL_COLUMNS + SPLIT_COLUMNS:
            if col_name not in self.data.columns:
                self.data[col_name] = 0
            self.data[col_name] = self.data[col_name].fillna(0).astype("int32")

    def _existing_split_is_valid(self, extra_df):
        """
        Check whether existing in_train/in_test/in_cal columns assign every row
        to exactly one split.

        :param extra_df: pandas dataframe with split columns
        :return: (is_valid, n_bad)
        """
        split_values = extra_df[SPLIT_COLUMNS].fillna(0).astype(int)
        row_split_counts = split_values.sum(axis=1)
        bad_mask = row_split_counts != 1
        n_bad = int(bad_mask.sum())
        return n_bad == 0, n_bad

    def augment_with_split(
        self,
        extra_datasets,
        test_size,
        cal_size,
        seed,
        indicator_col=None,
    ):
        """
        Load extra datasets, split them locally, append them to data,
        and extend split indices.

        If an extra dataset CSV does not already contain valid in_train,
        in_test, and in_cal columns, add/fix those split-indicator columns
        and save the updated CSV back to disk.

        If indicator_col is provided, set that indicator column to 1 for all
        rows in the extra dataset. This is used for dataset-type indicators
        such as fictional_object, noise_object, and synthetic_fic_object.

        :param extra_datasets: list of extra dataset names to load
        :param test_size: global fraction of rows for test split
        :param cal_size: global fraction of rows for calibration split
        :param seed: random seed for splitting
        :param indicator_col: optional dataset-type indicator column to set to 1
        :return: None
        """
        if not extra_datasets:
            return

        self._ensure_schema_columns()

        base_n = len(self.data)
        all_new_dfs = []
        per_dataset_splits = {}

        for ds in extra_datasets:
            csv_path = f"{self.dataset_path}{ds}.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    "Extra dataset CSV not found: {}".format(csv_path)
                )

            raw_df = pd.read_csv(csv_path)
            has_existing_split_cols = all(col in raw_df.columns for col in SPLIT_COLUMNS)
            missing_indicator_col = (
                indicator_col is not None and indicator_col not in raw_df.columns
            )

            extra_df = self.csv_to_aligned_df(csv_path, columns_like=self.data)
            n_rows = len(extra_df)

            if indicator_col is not None:
                if indicator_col not in extra_df.columns:
                    extra_df[indicator_col] = 0
                extra_df[indicator_col] = 1
                extra_df[indicator_col] = extra_df[indicator_col].astype("int32")

            use_existing_split_cols = False

            if has_existing_split_cols:
                valid_split_cols, n_bad = self._existing_split_is_valid(extra_df)

                if valid_split_cols:
                    use_existing_split_cols = True
                else:
                    log.warning(
                        f"Existing split columns in {csv_path} are invalid for "
                        f"{n_bad} rows. Regenerating split columns and saving "
                        "the corrected CSV."
                    )

            if use_existing_split_cols:
                tr_local = np.where(extra_df["in_train"].to_numpy().astype(int) == 1)[0]
                te_local = np.where(extra_df["in_test"].to_numpy().astype(int) == 1)[0]

                if self.with_calibration:
                    cal_local = np.where(
                        extra_df["in_cal"].to_numpy().astype(int) == 1
                    )[0]
                else:
                    cal_local = None

            else:
                tr_local, te_local, cal_local = self.three_way_random_split(
                    n_rows=n_rows,
                    test_size=test_size,
                    cal_size=cal_size,
                    seed=seed,
                )

                extra_df["in_train"] = 0
                extra_df["in_test"] = 0
                extra_df["in_cal"] = 0

                extra_df.loc[tr_local, "in_train"] = 1
                extra_df.loc[te_local, "in_test"] = 1

                if cal_local is not None:
                    extra_df.loc[cal_local, "in_cal"] = 1

                for col_name in SPLIT_COLUMNS:
                    extra_df[col_name] = extra_df[col_name].astype("int32")

            if (not use_existing_split_cols) or missing_indicator_col:
                extra_df.to_csv(csv_path, index=False)

                if self.verbose:
                    log.warning(
                        "Added/fixed split/indicator columns and saved updated dataset CSV: {}".format(
                            csv_path
                        )
                    )

            per_dataset_splits[ds] = (tr_local, te_local, cal_local)
            all_new_dfs.append(extra_df)

        if all_new_dfs:
            extra_concat = pd.concat(all_new_dfs, axis=0, ignore_index=True)
            self.data = pd.concat(
                [self.data, extra_concat],
                axis=0,
                ignore_index=True,
            )

        new_train_ids = []
        new_test_ids = []
        new_cal_ids = []

        row_start = base_n
        for ds, extra_df in zip(extra_datasets, all_new_dfs):
            n_rows = len(extra_df)
            tr_local, te_local, cal_local = per_dataset_splits[ds]

            new_train_ids.append(row_start + tr_local)
            new_test_ids.append(row_start + te_local)

            if cal_local is not None:
                new_cal_ids.append(row_start + cal_local)

            row_start += n_rows

        if new_train_ids:
            new_train_ids = np.concatenate(new_train_ids, axis=0)
            self.train_ids = np.concatenate(
                [self.train_ids, new_train_ids],
                axis=0,
            )

        if new_test_ids:
            new_test_ids = np.concatenate(new_test_ids, axis=0)
            self.test_ids = np.concatenate(
                [self.test_ids, new_test_ids],
                axis=0,
            )

        if self.with_calibration and new_cal_ids:
            if self.calibration_ids is None:
                self.calibration_ids = np.array([], dtype=int)

            new_cal_ids = np.concatenate(new_cal_ids, axis=0)
            self.calibration_ids = np.concatenate(
                [self.calibration_ids, new_cal_ids],
                axis=0,
            )

        self.datasets = list(self.datasets) + list(extra_datasets)

        if self.verbose:
            log.warning(
                "Augmented with extra datasets: {}. Total rows now: {}".format(
                    extra_datasets,
                    len(self.data),
                )
            )

    def get_num_layers(self):
        """
        Count the number of available layer files for the first dataset.

        :return: integer number of layers detected on disk
        """
        pattern = (
            f"{self.activations_path}/{self.model}/"
            f"{self.datasets[0]}/{self.activation_type}/*_e.npy"
        )
        return len(glob(pattern))

    def get_activations(self, layer_id, module="e"):
        """
        Load and stack activations for a given layer across all datasets.

        :param layer_id: integer index of the layer to load
        :param module: activation module suffix ("a", "m", or "e")
        :return: stacked 3D tensor of activations aligned with self.data rows
        """
        if module not in ["a", "m", "e"]:
            raise ValueError(
                "Module must be 'a' (self-attention), 'm' (mlp layer) or 'e' (encoder output)."
            )

        activations = []

        for dataset in self.datasets:
            data_dir = (
                f"{self.activations_path}/{self.model}/{dataset}/{self.activation_type}/"
            )
            temp_path = data_dir + f"layer_{layer_id}_{module}_temp.npy"
            npz_path = data_dir + f"layer_{layer_id}_{module}.npz"

            try:
                shape = shape_as_tuple(np.load(data_dir + "shape.npy"))
                dtype = infer_memmap_dtype(temp_path, shape)

                log.warning(
                    f"Loading activations: model={self.model}, dataset={dataset}, "
                    f"layer={layer_id}, dtype={np.dtype(dtype)}, shape={shape}, path={temp_path}"
                )

                acts = np.memmap(
                    temp_path,
                    shape=shape,
                    mode="r",
                    dtype=dtype,
                )

            except Exception as e:
                log.warning(
                    f"Could not load temp memmap for model={self.model}, "
                    f"dataset={dataset}, layer={layer_id}: {e}. "
                    f"Trying npz fallback: {npz_path}"
                )
                acts = self.load_npz(npz_path)

            acts_np = np.array(acts)

            n_nan = int(np.isnan(acts_np).sum())
            n_inf = int(np.isinf(acts_np).sum())

            if n_nan > 0 or n_inf > 0:
                raise ValueError(
                    f"Non-finite activations detected after loading: "
                    f"model={self.model}, dataset={dataset}, layer={layer_id}, "
                    f"nan={n_nan}, inf={n_inf}, shape={acts_np.shape}"
                )

            activations.append(torch.from_numpy(acts_np))

        if self.activation_type == "full":
            output = stack_tensors(activations)
        else:
            raise NotImplementedError("Only 'full' activation_type is supported here.")

        self.validate_activations(output)
        return output

    def validate_activations(self, activations):
        """
        Check that the number of activation rows matches the number of data rows.

        :param activations: stacked activation tensor or array
        :return: None; raises ValueError if sizes do not match
        """
        if self.activation_type == "full":
            n = len(activations)
        else:
            n = activations.shape[0]

        n_rows = self.data.shape[0]
        if n != n_rows:
            raise ValueError(
                "Number of rows in activations ({}) does not match the number of rows in the data ({}).".format(
                    n,
                    n_rows,
                )
            )

    def load_npz(self, path):
        """
        Load a compressed numpy activation array from disk.

        :param path: path to a .npz file with key 'arr_0'
        :return: loaded numpy array containing activations
        """
        return np.load(path)["arr_0"]

    def get_att_mask(self):
        """
        Load and stack the attention masks across all datasets.

        :return: stacked tensor of attention masks aligned with self.data rows
        """
        masks = []
        for dataset in self.datasets:
            mask_path = (
                f"{self.activations_path}/{self.model}/"
                f"{dataset}/{self.activation_type}/mask.npy"
            )
            masks.append(torch.from_numpy(np.load(mask_path)))

        return torch.vstack(masks)

    def get_train_att_mask(self):
        """
        Return the attention mask rows corresponding to the training split.

        :return: tensor of attention masks for training indices
        """
        ids = self.train_ids
        return self.get_att_mask()[ids]

    def get_test_att_mask(self):
        """
        Return the attention mask rows corresponding to the test split.

        :return: tensor of attention masks for test indices
        """
        ids = self.test_ids
        return self.get_att_mask()[ids]

    def get_cal_att_mask(self):
        """
        Return the attention mask rows corresponding to the calibration split.

        :return: tensor of attention masks for calibration indices
        """
        ids = self.calibration_ids
        return self.get_att_mask()[ids]

    def get_dataframe(self):
        """
        Return the full concatenated data table.

        :return: pandas dataframe with all loaded and augmented rows
        """
        return self.data

    def get_train_df(self):
        """
        Return the subset of data rows corresponding to the training split.

        :return: pandas dataframe for training indices
        """
        return self.data.iloc[self.train_ids.tolist()]

    def get_test_df(self):
        """
        Return the subset of data rows corresponding to the test split.

        :return: pandas dataframe for test indices
        """
        return self.data.iloc[self.test_ids.tolist()]

    def get_cal_df(self):
        """
        Return the subset of data rows corresponding to the calibration split.

        :return: pandas dataframe for calibration indices
        """
        if self.calibration_ids is None:
            raise ValueError("Calibration split not initialized.")

        return self.data.iloc[self.calibration_ids.tolist()]

    def column_list(self):
        """
        Collect the union of column names across all base datasets.

        :return: list of unique column names from base CSVs
        """
        columns = set()
        for dataset in self.base_datasets:
            lf = pl.scan_csv(f"{self.dataset_path}{dataset}.csv")
            try:
                columns.update(lf.collect_schema().names())
            except Exception:
                columns.update(lf.columns)

        return list(columns)

    def get_train_acts(self, layer_id: int, module: str = "e"):
        """
        Get the training activations for the given layer.

        :param layer_id: layer index
        :param module: activation module ('a', 'm', or 'e')
        """
        return self.get_activations(layer_id, module)[self.train_ids]

    def get_test_acts(self, layer_id: int, module: str = "e"):
        """
        Get the test activations for the given layer.

        :param layer_id: layer index
        :param module: activation module ('a', 'm', or 'e')
        """
        return self.get_activations(layer_id, module)[self.test_ids]

    def get_cal_acts(self, layer_id: int, module: str = "e"):
        """
        Get the calibration activations for the given layer.

        :param layer_id: layer index
        :param module: activation module ('a', 'm', or 'e')
        """
        if self.calibration_ids is None:
            raise ValueError("Calibration split not initialized.")

        return self.get_activations(layer_id, module)[self.calibration_ids]

    def train_labeled(self, layer_id: int = -1):
        """
        Return train embeddings + 'correct' labels for a given layer.

        :param layer_id: layer index
        """
        correct = self.get_train_df()["correct"].to_numpy()

        if self.activation_type == "full":
            embeddings = self.get_train_acts(layer_id=layer_id)[:, -1]
        elif self.activation_type == "last":
            embeddings = self.get_train_acts(layer_id=layer_id)
        else:
            raise NotImplementedError(
                "train_labeled is only implemented for 'full' and 'last' activations."
            )

        return {"embeddings": embeddings, "correct": correct}

    def test_labeled(self, layer_id: int = -1):
        """
        Return test embeddings + 'correct' labels for a given layer.

        :param layer_id: layer index
        """
        correct = self.get_test_df()["correct"].to_numpy()

        if self.activation_type == "full":
            embeddings = self.get_test_acts(layer_id=layer_id)[:, -1]
        elif self.activation_type == "last":
            embeddings = self.get_test_acts(layer_id=layer_id)
        else:
            raise NotImplementedError(
                "test_labeled is only implemented for 'full' and 'last' activations."
            )

        return {"embeddings": embeddings, "correct": correct}

    def cal_labeled(self, layer_id: int = -1):
        """
        Return calibration embeddings + 'correct' labels for a given layer.

        :param layer_id: layer index
        """
        if self.calibration_ids is None:
            raise ValueError("Calibration split not initialized.")

        correct = self.get_cal_df()["correct"].to_numpy()

        if self.activation_type == "full":
            embeddings = self.get_cal_acts(layer_id=layer_id)[:, -1]
        elif self.activation_type == "last":
            embeddings = self.get_cal_acts(layer_id=layer_id)
        else:
            raise NotImplementedError(
                "cal_labeled is only implemented for 'full' and 'last' activations."
            )

        return {"embeddings": embeddings, "correct": correct}

    def _drop_zeros_einsum(self, act: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mask to a single sequence of activations via einsum,
        handling some edge cases.

        :param act: token-level activations
        :param mask: attention mask
        :return: masked activations
        """
        if act.shape[0] == mask.shape[0]:
            return torch.einsum("lh,l->lh", act, mask)
        else:
            shape = mask.shape[0]
            if mask.sum() == 0:
                mask[-5:] = 1
            return torch.einsum("lh,l->lh", act[-shape:], mask)

    def _drop_zeros(self, acts: torch.Tensor, mask: torch.Tensor | None):
        """
        Use attention masks to zero out padded tokens and drop all-zero rows.

        :param acts: batch of token-level activations
        :param mask: attention mask or None
        :return: list of nonzero token bags
        """
        if mask is not None:
            bags = [
                drop_zero_rows(self._drop_zeros_einsum(a, m))
                for a, m in zip(acts, mask)
            ]
        else:
            log.warning("No mask found. Not dropping zeros without mask.")
            bags = [a for a in acts]

        for bag in bags:
            if bag.shape[0] == 0:
                log.warning("Bag is empty after zero-drop.")

        return bags

    def train_bags(self, layer_id: int = -1, drop_zeros: bool = True):
        """
        Get training bags for MIL: variable-length sequences per statement.

        :param layer_id: layer index
        :param drop_zeros: whether to remove padding/all-zero token rows
        """
        assert (
            self.activation_type == "full"
        ), "Bags can only be generated for full activations."

        correct = self.get_train_df()["correct"].to_numpy()
        acts = self.get_train_acts(layer_id=layer_id)

        try:
            mask = self.get_train_att_mask()
        except Exception:
            mask = None

        if drop_zeros:
            bags = self._drop_zeros(acts, mask)
        else:
            bags = acts

        return {
            "embeddings": bags,
            "correct": correct,
            "last_embedding": acts[:, -1],
        }

    def test_bags(self, layer_id: int = -1, drop_zeros: bool = True):
        """
        Get test bags for MIL.

        :param layer_id: layer index
        :param drop_zeros: whether to remove padding/all-zero token rows
        """
        assert (
            self.activation_type == "full"
        ), "Bags can only be generated for full activations."

        correct = self.get_test_df()["correct"].to_numpy()
        acts = self.get_test_acts(layer_id=layer_id)

        try:
            mask = self.get_test_att_mask()
        except Exception:
            mask = None

        if drop_zeros:
            bags = self._drop_zeros(acts, mask)
        else:
            bags = acts

        return {
            "embeddings": bags,
            "correct": correct,
            "last_embedding": acts[:, -1],
        }

    def cal_bags(self, layer_id: int = -1, drop_zeros: bool = True):
        """
        Get calibration bags for MIL.

        :param layer_id: layer index
        :param drop_zeros: whether to remove padding/all-zero token rows
        """
        assert (
            self.activation_type == "full"
        ), "Bags can only be generated for full activations."

        if self.calibration_ids is None:
            raise ValueError("Calibration split not initialized.")

        correct = self.get_cal_df()["correct"].to_numpy()
        acts = self.get_cal_acts(layer_id=layer_id)

        try:
            mask = self.get_cal_att_mask()
        except Exception:
            mask = None

        if drop_zeros:
            bags = self._drop_zeros(acts, mask)
        else:
            bags = acts

        return {
            "embeddings": bags,
            "correct": correct,
            "last_embedding": acts[:, -1],
        }


def drop_zero_rows(X):
    """
    Drop rows that are entirely zero from a 2D array or tensor.

    :param X: 2D array or tensor
    :return: array or tensor with zero-only rows removed
    """
    return X[(X != 0).any(axis=1)]


def unique_rows(a):
    """
    Return the unique rows of a 2D numpy array.

    :param a: 2D numpy array
    :return: 2D numpy array with duplicate rows removed
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([("", a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))