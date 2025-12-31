"""
utils.py

Utility functions for device selection, model preparation,
statement loading, and data handling in probing experiments.

"""

import logging

import torch
import numpy as np
import polars as pl
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_handler import DataHandler

log = logging.getLogger("utils")


def get_device():
    """
    Select a computation device (CUDA, MPS, or CPU) for running models.

    :return: torch.device corresponding to the chosen backend
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        log.info("Using CUDA device: {}".format(torch.cuda.get_device_name(0)))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using MPS device")
    else:
        device = torch.device("cpu")
        log.info("Using CPU device")
    return device


def prepare_hf_model(cfg, device=None):
    """
    Prepare a HuggingFace causal LM and tokenizer according to the config.

    :param cfg: Hydra config object containing model and device settings
    :param device: optional explicit device; falls back to cfg.device if None
    :return: (model, tokenizer) ready to use for inference
    """
    if device is None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(device)

    if cfg.model["dtype"] == "float16":
        dtype = torch.float16
    elif cfg.model["dtype"] == "float32":
        dtype = torch.float32
    else:
        raise ValueError("dtype must be either 'float16' or 'float32'.")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model["model"],
        token=cfg.model["token"],
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map={"": device},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model["model"],
        token=cfg.model["token"],
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_statements(dataset):
    """
    Load the 'statement' column from a CSV dataset as a flat list of strings.

    :param dataset: dataset name stem (without .csv)
    :return: list of statement strings from the dataset
    """
    return flatten(
        pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()
    )


def return_layers(cfg, dataset=None):
    """
    Return the list of layer indices to use for the current datapack.

    :param cfg: Hydra config object containing 'layers' mapping and datapack name
    :param dataset: dataset name
    :return: the selected layer index
    """
    if dataset:
        if 'cities_loc' in dataset:
            data_name = 'cities_loc'
        elif 'med_indications' in dataset:
            data_name = 'med_indications'
        elif 'defs' in dataset:
            data_name = 'defs'
            
        layer_dict = cfg.layers
        layer = int(layer_dict[data_name])
        return layer
        
    else:
        data_name = cfg.datapack.name
        layer_dict = cfg.layers
        layer = int(layer_dict[data_name])
        return layer


def flatten(xss):
    """
    Flatten a nested list or array-of-arrays into a single list.

    :param xss: iterable of iterables
    :return: flat list containing all elements
    """
    return [x for xs in xss for x in xs]


def load_data(cfg):
    """
    Construct and assemble a DataHandler object based on the config.

    :param cfg: Hydra config object containing datapack and model settings
    :return: initialized and assembled DataHandler instance
    """
    if cfg.datapack.cal_size > 0:
        with_calibration = True
    else:
        with_calibration = False

    dh = DataHandler(
        cfg.model["name"],
        cfg.datapack["datasets"],
        datasets_fictional=getattr(cfg.datapack, "datasets_fictional", []),
        datasets_noise=getattr(cfg.datapack, "datasets_noise", []),
        use_fictional=getattr(cfg.datapack, "use_fictional", False),
        use_noise=getattr(cfg.datapack, "use_noise", False),
        activation_type=cfg.agg,
        with_calibration=with_calibration,
        load_scores=cfg.datapack["load_scores"],
    )

    dh.assemble(
        test_size=cfg.datapack["test_size"],
        calibration_size=cfg.datapack["cal_size"],
        seed=cfg.datapack["random_seed"],
        exclusive_split=cfg.datapack["exclusive_split"],
    )
    return dh


def return_label(data):
    """
    Extract label-related columns from a dataframe.

    :param data: pandas DataFrame with label columns
    :return: tuple (correct, real, fake, combined, negated, fictional, disputed, noise)
    """
    correct = data["correct"].values
    real = data["real_object"].values
    fake = data["fake_object"].values
    negated = data["negation"].values

    if "noise" in data.columns:
        noise = data["noise"].values
    else:
        noise = np.zeros(len(data), dtype=int)

    if "disputed_object" in data.columns:
        disputed = data["disputed_object"].values
    else:
        disputed = np.zeros(len(data), dtype=int)

    if "fictional_object" in data.columns:
        fictional = data["fictional_object"].values
    else:
        fictional = np.zeros(len(data), dtype=int)

    combined = np.select(
        [
            (correct == 0) & (real == 1) & (fake == 0) & (fictional == 0),
            (correct == 1) & (real == 1) & (fake == 0) & (fictional == 0),
            (fake == 1) & (fictional == 0) & (real == 0),
            ((correct == 0) & (fake == 1)) | ((correct == 0) & (fictional == 1)),
            ((correct == 1) & (fake == 1)) | ((correct == 1) & (fictional == 1)),
        ],
        [0, 1, 4, 2, 3],
        default=4,
    )

    return correct, real, fake, combined, negated, fictional, disputed, noise


def drop_rows_with_tail_keep(arr, num_rows_to_keep, last_rows_to_keep=2, random_seed=42):
    """
    Randomly select rows from a 2D array while always keeping the last rows.

    :param arr: 2D numpy array
    :param num_rows_to_keep: total number of rows to keep (including tail rows)
    :param last_rows_to_keep: number of last rows to keep deterministically
    :param random_seed: random seed for reproducibility
    :return: 2D numpy array with the selected rows
    """
    last_rows = arr[-last_rows_to_keep:]
    remaining_rows = arr[:-last_rows_to_keep]

    np.random.seed(random_seed)
    indices_to_keep = np.random.choice(
        len(remaining_rows), size=num_rows_to_keep - last_rows_to_keep, replace=False
    )
    sampled_rows = remaining_rows[indices_to_keep]

    final_array = np.vstack([sampled_rows, last_rows])
    return final_array


def bootstrap_ci(metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05, random_state=None):
    """
    Calculate bootstrapped confidence intervals for a given metric.

    :param metric_func: metric function taking (y_true, y_pred) and returning a scalar
    :param y_true: array-like of shape (n_samples,), ground truth labels
    :param y_pred: array-like of shape (n_samples,), predicted labels or scores
    :param n_bootstraps: number of bootstrap samples
    :param alpha: significance level for the confidence interval
    :param random_state: random seed
    :return: tuple (metric_value, ci_lower, ci_upper)
    """
    if random_state is not None:
        np.random.seed(random_state)

    bootstrapped_scores = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        score = metric_func(y_true_bootstrap, y_pred_bootstrap)
        bootstrapped_scores.append(score)

    ci_lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

    return metric_func(y_true, y_pred), ci_lower, ci_upper


def safe_bootstrap(metric, **kwargs):
    """
    Wrapper around bootstrap_ci that catches errors and returns a safe default.

    :param metric: metric function to use
    :param kwargs: keyword arguments for bootstrap_ci
    :return: tuple (metric_value, ci_lower, ci_upper) or safe default on error
    """
    try:
        return bootstrap_ci(metric, **kwargs)
    except Exception:
        return (0, -1e-6, 1e-6)


def should_process_layer(layer_id, cfg):
    """
    Determine if a given layer should be processed based on the configured layer range.

    :param layer_id: integer layer index
    :param cfg: Hydra config with model.layers and layer_range
    :return: True if layer is within the configured range, else False
    """
    layer_range = np.quantile(
        cfg.model["layers"], cfg.layer_range, method="closest_observation"
    )
    return layer_range[0] <= layer_id <= layer_range[1]


class LayerData:
    """
    Container for per-layer data used by the probing scripts.

    Attributes are assigned directly in __init__.
    """

    def __init__(
        self,
        layer_id,
        X_tr,
        X_te,
        X_cal,
        y_train,
        y_test,
        y_cal,
        mask_tr,
        mask_te,
        mask_cal,
    ):
        """
        Initialize a LayerData container.

        :param layer_id: integer layer index
        :param X_tr: training features
        :param X_te: test features
        :param X_cal: calibration features (or None)
        :param y_train: training labels
        :param y_test: test labels
        :param y_cal: calibration labels (or None)
        :param mask_tr: boolean mask for valid training rows
        :param mask_te: boolean mask for valid test rows
        :param mask_cal: boolean mask for valid calibration rows
        """
        self.layer_id = layer_id
        self.X_tr = X_tr
        self.X_te = X_te
        self.X_cal = X_cal
        self.y_train = y_train
        self.y_test = y_test
        self.y_cal = y_cal
        self.mask_tr = mask_tr
        self.mask_te = mask_te
        self.mask_cal = mask_cal


class BagProcessor:
    """
    Helper class to preprocess bags of embeddings for MIL algorithms.

    It trims bags to a maximum size, preserves a tail of positive labels,
    optionally applies scaling, and can generate bag-level scores.
    """

    def __init__(self, max_bag_size, pos_labels_in_bag=2, scaler=None, random_seed=42):
        """
        Initialize a BagProcessor.

        :param max_bag_size: maximum number of elements per bag
        :param pos_labels_in_bag: number of positive labels assumed at the tail
        :param scaler: optional sklearn-style scaler to apply to each bag
        :param random_seed: random seed for sampling within bags
        """
        self.max_bag_size = max_bag_size
        self.pos_labels_in_bag = pos_labels_in_bag
        self.scaler = scaler
        self.random_seed = random_seed

    def __call__(self, bags, **kwds):
        """
        Allow BagProcessor to be called like a function.

        :param bags: list of numpy arrays representing bags
        :return: tuple (processed_bags, intra_bag_masks)
        """
        return self.process(bags)

    def process(self, bags):
        """
        Process a list of bags.

        :param bags: list of numpy arrays representing bags
        :return: (output_bags, intra_bag_mask), where each mask marks tail positives
        """
        output_bags = []
        intra_bag_mask = []
        for i, bag in enumerate(bags):
            _bag, _mask = self.process_single_bag(bag, random_seed=self.random_seed + i)
            output_bags.append(_bag)
            intra_bag_mask.append(_mask)
        return output_bags, intra_bag_mask

    def process_single_bag(self, bag, random_seed=None):
        """
        Process a single bag: trim, scale, and mark intra-bag positives.

        :param bag: numpy array of shape (n_tokens, dim)
        :param random_seed: random seed for sampling, overrides default if provided
        :return: (processed_bag, intra_bag_mask)
        """
        if random_seed is None:
            random_seed = self.random_seed
        n = len(bag)
        if n > self.max_bag_size:
            _bag = drop_rows_with_tail_keep(
                bag, self.max_bag_size, self.pos_labels_in_bag, random_seed=random_seed
            )
        else:
            _bag = bag
        if self.scaler is not None:
            _bag = self.scaler.transform(_bag)
        return _bag, self._intra_bag_labels(_bag)

    def _intra_bag_labels(self, bag):
        """
        Construct a simple tail-based intra-bag label mask.

        :param bag: numpy array representing a bag
        :return: numpy array mask with ones on the last pos_labels_in_bag entries
        """
        n = len(bag)
        mask = np.zeros(n)
        mask[-self.pos_labels_in_bag :] = 1
        return mask

    def predict_scores(self, bags, direction, bias=0.0):
        """
        Compute bag-level scores using a linear direction and bias.

        :param bags: list of numpy arrays representing bags
        :param direction: 1D numpy array for the linear direction
        :param bias: scalar bias term
        :return: numpy array of scores, one per bag
        """
        preds = []
        for i, bag in enumerate(bags):
            _bag = self.process_single_bag(
                bag, random_seed=self.random_seed + i
            )[0]
            pred = np.dot(_bag, direction) + bias
            pred = np.max(pred)
            preds.append(pred)
        return np.array(preds)
