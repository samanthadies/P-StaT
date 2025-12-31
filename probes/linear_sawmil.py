"""
linear_sawmil.py

sAwMIL linear probe utilities.

- sAwMIL: MIL/bag-based probe
    * Uses bags (sequences of token embeddings).
    * Hyperparameter search over C if cfg.search=True.
    * Saves metrics as metrics_{layer}.json with 'default' and 'conformal' keys.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import logging
import os
import json
import pickle
import time

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import energy_distance
from sklearn.metrics import average_precision_score as mAP, matthews_corrcoef as mcc, \
    adjusted_mutual_info_score as ami, adjusted_rand_score as ari

from utils import (
    drop_rows_with_tail_keep,
    BagProcessor,
    safe_bootstrap,
)
from probes.sawmil import sbMIL
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from probes.transforms import SparseTransform

log = logging.getLogger(__name__)


def _filter_empty_bags(
    X,
    y,
    mask,
    where="",
):
    """
    Remove bags with zero rows and update labels and mask accordingly.

    Each element of X is a bag with shape (len_i, H). This function removes
    any bag with zero length and updates the corresponding label and mask.
    Raises an error if no non-empty bags remain.

    :param X: List of bag arrays, each of shape (len_i, H)
    :param y: Array of labels, same length as X
    :param mask: Boolean mask indicating which samples are considered
    :param where: Optional string indicating context for logging
    :return: Tuple (X_filtered, y_filtered, new_mask) with only non-empty bags
    """
    X = list(X)
    mask = np.asarray(mask, dtype=bool)

    assert len(X) == len(y) == len(mask), "Lengths must match in _filter_empty_bags."

    non_empty = np.array([bag.shape[0] > 0 for bag in X], dtype=bool)

    if where:
        n_empty = int((~non_empty & mask).sum())
        if n_empty > 0:
            log.warning(
                f"[filter_empty_bags] {where}: dropping {n_empty} empty bags "
                f"(out of {len(X)} total)"
            )

    keep = mask & non_empty
    if not keep.any():
        raise RuntimeError(
            f"[filter_empty_bags] {where}: no non-empty bags left after filtering."
        )

    X_f = [bag for bag, k in zip(X, keep) if k]
    y_f = y[keep]
    new_mask = np.ones(len(X_f), dtype=bool)

    return X_f, y_f, new_mask


def log_metric_sAwMIL(
    preds,
    scores,
    y_true,
    mask,
    cfg,
):
    """
    Compute evaluation metrics for sAwMIL predictions, including:
    - Uses safe_bootstrap for confidence intervals.
    - Applies weighted metrics based on acceptance rate (non-abstentions).
    - Returns MCC, AMI, ARI, mAP, energy distance, recall, and acceptance rate.

    :param preds: Array of predicted labels (including possible -1 abstentions)
    :param scores: Array of real-valued decision scores per bag
    :param y_true: Array of ground-truth labels in {0, 1}
    :param mask: Boolean mask over samples to include in evaluation
    :param cfg: Hydra config with eval_params.n_bootstraps
    :return: Dictionary of metrics (with bootstrapped values where applicable)
    """
    is_binary = len(np.unique(y_true)) == 2
    assert is_binary, "Only binary classification is supported."

    is_ok = (len(np.unique(preds)) > 0) & (len(np.unique(preds)) < 4)
    assert is_ok, (
        "Only binary classification is supported (or binary with "
        "abstention class '-1')."
    )

    a_mask = preds != -1
    a_rate = np.sum(a_mask[mask]) / len(a_mask[mask])

    def wmcc(y_t, y_p):
        return mcc(y_t, y_p) * a_rate

    def wami(y_t, y_p):
        return ami(y_t, y_p) * a_rate

    def wari(y_t, y_p):
        return ari(y_t, y_p) * a_rate

    full_mask = np.array(mask & a_mask)

    binary_kwargs = dict(
        y_true=y_true[full_mask],
        y_pred=preds[full_mask],
        n_bootstraps=cfg.eval_params["n_bootstraps"],
    )

    mcc_val = safe_bootstrap(mcc, **binary_kwargs)
    ami_val = safe_bootstrap(ami, **binary_kwargs)
    ari_val = safe_bootstrap(ari, **binary_kwargs)
    recall_val = safe_bootstrap(
        lambda yt, yp: (yt[yt == 1] == yp[yt == 1]).mean()
        if (yt == 1).any()
        else 0.0,
        **binary_kwargs,
    )

    if np.equal(a_mask.mean(), 1):
        wmcc_val = mcc_val
        wami_val = ami_val
        wari_val = ari_val
        wrecall_val = recall_val
    else:
        wmcc_val = safe_bootstrap(wmcc, **binary_kwargs)
        wami_val = safe_bootstrap(wami, **binary_kwargs)
        wari_val = safe_bootstrap(wari, **binary_kwargs)
        wrecall_val = recall_val

    try:
        probs = scores[full_mask]
        x_min = probs.min()
        x_max = probs.max()
        probs_scaled = (probs - x_min) / (x_max - x_min)
        targets = y_true[full_mask]
        energy_val = energy_distance(
            probs_scaled[targets == 0], probs_scaled[targets == 1]
        )
    except Exception as e:
        log.warning(f"Error calculating energy distance: {e}. Setting to 1000.")
        energy_val = 1000

    try:
        mAP_val = mAP(y_true[full_mask], scores[full_mask])
    except Exception:
        try:
            mAP_val = mAP(y_true[full_mask], np.zeros_like(scores[full_mask]))
        except Exception:
            try:
                mAP_val = mAP(y_true[mask], np.zeros_like(scores[mask]))
            except Exception:
                mAP_val = 0

    metric_with_ci = {
        "mcc": mcc_val,
        "ami": ami_val,
        "ari": ari_val,
        "wmcc": wmcc_val,
        "wami": wami_val,
        "wari": wari_val,
        "map": mAP_val,
        "wmap": mAP_val * a_rate,
        "energy": energy_val,
        "wenergy": energy_val * a_rate,
        "acceptance_rate": a_rate,
        "recall": recall_val,
        "wrecall": wrecall_val,
        "n": y_true[full_mask].shape[0],
    }

    return metric_with_ci


def single_training_sAwMIL(
    X,
    y,
    dh,
    layer_id,
    mask,
    cfg,
):
    """
    Train an sAwMIL probe without hyperparameter search.
    - Filters out empty bags.
    - Normalizes instances with StandardScaler.
    - Optionally sparsifies features with SparseTransform.
    - Adjusts bag sizes with drop_rows_with_tail_keep.
    - Estimates eta and constructs intra-bag labels.
    - Fits classifier on the processed bags.

    :param X: List of bag arrays (len_i, H) for training
    :param y: Array of bag labels in {0, 1}
    :param dh: Data handler providing train_bags for sparsification
    :param layer_id: Integer layer index (used for logging and seeding)
    :param mask: Boolean mask indicating which bags to use
    :param cfg: Hydra config with probe and sparsify_data settings
    :return: Dictionary with fitted separator, scaler, transformer, and eta
    """
    mask = np.array(mask).astype(bool)
    X, y, mask = _filter_empty_bags(
        X, y, mask, where=f"single_training (layer {layer_id})"
    )
    probe_name = cfg.probe["name"]

    y = y[mask]
    y[y == 0] = -1

    np.random.seed(cfg["random_seed"])

    # Normalize bags
    if cfg.probe["normalize_data"]:
        log.warning("\tNormalizing the data...")
        X_temp = np.vstack(X)
        scaler = StandardScaler()
        scaler.fit(X_temp)
        bags = [scaler.transform(bag) for bag in X]
    else:
        raise NotImplementedError("Only normalized pipeline is implemented for sAwMIL.")

    # Sparsify features (optional)
    if cfg["sparsify_data"] > 0:
        log.warning(f"\tSparsifying the data [{cfg['sparsify_data']}]...")
        prj = SparseTransform(
            max_k=cfg["sparsify_data"],
            normalize=False,
            show_progress=False,
        )
        X_temp = scaler.transform(dh.train_bags(layer_id)["last_embedding"])
        prj.fit(pd.DataFrame(X_temp[mask]), pd.Series(y))
        bags = [prj.transform(bag) for bag in bags]
    else:
        prj = None

    # Adjust bag sizes with tail-keeping
    new_bags = []
    for i, bag in enumerate(bags):
        n = bag.shape[0]
        if n > cfg.probe["max_bag_size"]:
            new_bags.append(
                drop_rows_with_tail_keep(
                    bag,
                    cfg.probe["max_bag_size"],
                    cfg["probe"]["num_known_positives"],
                    (cfg["random_seed"] + layer_id + i),
                )
            )
        else:
            new_bags.append(bag)
    bags = new_bags

    lengths = [len(b) for b, _y in zip(bags, y) if _y == 1]
    eta = (y[y == 1] * cfg.probe["num_known_positives"]).sum() / sum(lengths)

    y[y == 0] = -1

    bags_intra_labels = []
    for bag in bags:
        if cfg.probe["assume_known_positives"]:
            intra_labels = (
                [0] * (bag.shape[0] - cfg.probe["num_known_positives"])
                + [1] * cfg.probe["num_known_positives"]
            )
        else:
            intra_labels = [1] * bag.shape[0]
        bags_intra_labels.append(intra_labels)

    log.warning(
        f"\t{probe_name} [eta={eta:.2f} and C={cfg.probe['init_params']['C']}] probe is running..."
    )
    separator = sbMIL(
        C=cfg.probe["init_params"]["C"],
        kernel=cfg.probe["init_params"]["kernel"],
        penalty=cfg.probe["init_params"]["penalty"],
        scale_C=cfg.probe["init_params"]["scale_C"],
        verbose=cfg.probe["init_params"]["verbose"],
        eta=eta,
    )

    log.warning(f"\tNumber of bags: {len(bags)}")
    separator.fit(
        bags[: cfg.probe["train_bag_limit"]],
        y[: cfg.probe["train_bag_limit"]],
        bags_intra_labels[: cfg.probe["train_bag_limit"]],
    )

    return {
        "separator": separator,
        "scaler": scaler,
        "transformer": prj,
        "eta": eta,
    }


def parameter_search_sAwMIL(
    X,
    y,
    dh,
    mask,
    cfg,
    layer_id,
):
    """
    Perform hyperparameter search over C for sAwMIL:
    - Filters out empty bags.
    - Uses K-fold cross-validation over bag indices.
    - For each candidate C, trains sAwMIL on training folds and evaluates mAP.
    - Selects C using the one-standard-error rule.
    - Writes the selected C back into cfg.probe.init_params and retrains.

    :param X: List of bag arrays (len_i, H) for training
    :param y: Array of bag labels in {0, 1}
    :param dh: Data handler used during training for sparsification (if needed)
    :param mask: Boolean mask indicating which bags to include in the search
    :param cfg: Hydra config with probe, cv_n_folds, cv_bag_limit, and random_seed
    :param layer_id: Integer layer index (used for seeding and logging)
    :return: Dictionary from single_training_sAwMIL with the best model
    """
    log.warning("Running the hyperparameter search for sAwMIL...")
    param_grid = cfg.probe["param_grid"]["C"]

    mask = np.array(mask)
    X, y, mask = _filter_empty_bags(
        X, y, mask, where=f"parameter_search (layer {layer_id})"
    )

    kf = KFold(
        n_splits=cfg.cv_n_folds,
        shuffle=True,
        random_state=cfg.random_seed,
    )
    kf.get_n_splits(X)

    scores = []
    stds = []
    n_samples = len(X)

    for i, C in enumerate(param_grid):
        log.warning(f"\tRunning the iteration with C={C}...")
        inner_scores = []

        for j, (train_index, test_index) in enumerate(kf.split(X)):
            tr_mask = np.zeros(n_samples, dtype=bool)
            te_mask = np.zeros(n_samples, dtype=bool)
            tr_mask[train_index] = True
            te_mask[test_index] = True

            tr_mask = tr_mask & mask
            te_mask = te_mask & mask

            X_train = [x for x, m in zip(X, tr_mask) if m]
            X_test = [x for x, m in zip(X, te_mask) if m]
            y_train = y[tr_mask]
            y_test = y[te_mask]

            y_train[y_train == 0] = -1
            y_test[y_test == 0] = -1

            np.random.seed(cfg["random_seed"] + j)

            # Normalize bags
            if cfg.probe["normalize_data"]:
                log.warning("\t\tNormalizing the data...")
                X_temp = np.vstack(X_train)
                scaler = StandardScaler()
                scaler.fit(X_temp)
                bags = [scaler.transform(bag) for bag in X_train]
            else:
                raise NotImplementedError(
                    "Only a pipeline with the normalization is implemented"
                )

            # Adjust bag sizes with tail-keeping
            new_bags = []
            if cfg.probe["assume_known_positives"]:
                last_rows_to_keep = cfg.probe["num_known_positives"]
            else:
                last_rows_to_keep = 0

            max_bag_size = max(cfg.probe["max_bag_size"] - 5, 10)

            for i_b, bag in enumerate(bags):
                n = bag.shape[0]
                if len(bag) == 0:
                    log.warning("Empty bag found in the training data")
                    continue
                if n > max_bag_size:
                    new_bags.append(
                        drop_rows_with_tail_keep(
                            bag,
                            max_bag_size,
                            last_rows_to_keep,
                            (cfg["random_seed"] + layer_id + i_b),
                        )
                    )
                else:
                    new_bags.append(bag)

                if len(new_bags[-1]) == 0:
                    log.warning(
                        "\tEmpty bag after the drop rows in the training data."
                    )

            bags = new_bags

            # Cap number of bags for CV
            limit = cfg.cv_bag_limit
            bags = bags[:limit]
            y_train = y_train[:limit]

            lengths = [len(b) for b, _y in zip(bags, y_train) if _y == 1]
            eta = (
                (y_train[y_train == 1] * cfg.probe["num_known_positives"]).sum()
                / sum(lengths)
            )

            y_train[y_train == 0] = -1

            bags_labels = []
            for bag in bags:
                if cfg.probe["assume_known_positives"]:
                    intra_labels = (
                        [0]
                        * (bag.shape[0] - cfg.probe["num_known_positives"])
                        + [1] * cfg.probe["num_known_positives"]
                    )
                else:
                    intra_labels = [1] * bag.shape[0]
                bags_labels.append(intra_labels)

            try:
                log.warning(
                    f"\t\tSAWMIL [eta={eta:.2f}, l={len(bags)}] probe is running..."
                )
                separator = sbMIL(
                    C=C,
                    kernel=cfg.probe["init_params"]["kernel"],
                    penalty=cfg.probe["init_params"]["penalty"],
                    scale_C=cfg.probe["init_params"]["scale_C"],
                    verbose=False,
                    eta=eta,
                )
                separator.fit(bags, y_train, bags_labels)
                direction, bias = separator.linearize(normalize=True)
                bp = BagProcessor(
                    max_bag_size=cfg.probe["max_bag_size"],
                    pos_labels_in_bag=cfg.probe["num_known_positives"],
                    scaler=scaler,
                )
                y_te = bp.predict_scores(
                    bags=X_test, direction=direction, bias=bias
                )
                inner_scores.append(mAP(y_test, y_te))
                log.warning(f"\t\tmAP for {j}th fold: {inner_scores[-1]}")
            except Exception as e:
                log.error(f"Error: {e}")
                log.warning(
                    "\t\tSkipping the [%s] layer and moving to the next one..."
                    % layer_id
                )
                inner_scores.append(0.1)

        scores.append(np.mean(inner_scores))
        stds.append(np.std(inner_scores))
        log.warning(f"\tMean mAP for {C}: {scores[-1]}")

    se = np.array(stds) / np.sqrt(cfg.cv_n_folds)
    scores = np.array(scores)

    best_index = np.argmax(scores)
    best_score = scores[best_index]
    best_C = list(param_grid)[best_index]
    best_se = se[best_index]

    selected_index = None
    for idx, score in enumerate(scores):
        if score > (best_score - best_se):
            selected_index = idx
            break

    try:
        selected_C = list(param_grid)[selected_index]
        selected_score = scores[selected_index]
    except Exception:
        selected_C = best_C
        selected_score = best_score
        log.warning("\t\tCould not find a C within 1SE. Using the best C.")

    log.warning(f"\t\tScores: {scores}")
    log.warning(f"\t\tSE: {se}")
    log.warning(
        f"\t\tBest C: {best_C} with mAP: {best_score} (se {best_se})"
    )
    log.warning(
        f"\t\tSelected C: {selected_C} with mAP: {selected_score}"
    )

    from omegaconf import OmegaConf as _OC  # local import to avoid type hints
    _OC.set_struct(cfg, False)
    cfg["probe"]["init_params"]["C"] = selected_C
    _OC.set_struct(cfg, True)

    log.warning(
        f"MODEL: Retraining with the best C: {cfg['probe']['init_params']['C']}"
    )

    return single_training_sAwMIL(X, y, dh, layer_id, mask, cfg)


def save_sAwMIL(
    non_normalized_direction,
    non_normalized_bias,
    direction,
    bias,
    scaler,
    transformer,
    conf_calibrator,
    metric_dict,
    cfg,
    layer_id,
    y_hat=None,
    y_true=None,
):
    """
    Save sAwMIL outputs and metadata for a given layer.

    This writes all outputs:
    - config.json
    - metrics_{layer}.json
    - coef_{layer}.npy and bias_{layer}.npy
    - NON_NORMALIZED_coef_{layer}.npy and NON_NORMALIZED_bias_{layer}.npy
    - optional scaler, transformer, and conformal calibrator pickles
    - optional y_hat_{layer}.npy and y_true.npy

    :param non_normalized_direction: Raw weight vector before normalization
    :param non_normalized_bias: Raw bias value before normalization
    :param direction: Normalized weight vector for this layer
    :param bias: Normalized bias value for this layer
    :param scaler: Fitted StandardScaler or None
    :param transformer: Fitted SparseTransform or None
    :param conf_calibrator: Fitted conformal predictor or None
    :param metric_dict: Dictionary with 'default' and 'conformal' metrics
    :param cfg: Hydra config with an output_dir field
    :param layer_id: Integer layer index
    :param y_hat: Optional array of decision scores on the test set
    :param y_true: Optional array of ground-truth labels on the test set
    :return: None
    """
    save_dir = cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/config.json", "w") as f:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        json.dump(resolved, f)

    with open(f"{save_dir}/metrics_{layer_id}.json", "w") as f:
        json.dump(metric_dict, f)

    np.save(f"{save_dir}/coef_{layer_id}.npy", direction)
    np.save(f"{save_dir}/bias_{layer_id}.npy", bias)

    np.save(
        f"{save_dir}/NON_NORMALIZED_coef_{layer_id}.npy",
        non_normalized_direction,
    )
    np.save(
        f"{save_dir}/NON_NORMALIZED_bias_{layer_id}.npy",
        non_normalized_bias,
    )

    if scaler is not None:
        with open(f"{save_dir}/scaler_{layer_id}.pkl", "wb") as f:
            pickle.dump(scaler, f)

    if transformer is not None:
        with open(f"{save_dir}/transformer_{layer_id}.pkl", "wb") as f:
            pickle.dump(transformer, f)

    if conf_calibrator is not None:
        with open(f"{save_dir}/calibrator_{layer_id}.pkl", "wb") as f:
            pickle.dump(conf_calibrator, f)

    if y_hat is not None:
        np.save(f"{save_dir}/y_hat_{layer_id}.npy", y_hat)

    if y_true is not None:
        np.save(f"{save_dir}/y_true.npy", y_true)


def run_sawmil_layer(
    cfg,
    dh,
    layer_data,
):
    """
    Run the sAwMIL (bag-based) probe for a single layer.

    This function coordinates:
    - sAwMIL training (with or without hyperparameter search)
    - filtering empty bags for train, test, and calibration sets
    - computing bag-level decision scores
    - fitting a conformal predictor on calibration scores
    - computing default and conformal metrics
    - saving all outputs and metrics, if configured

    :param cfg: Hydra config with probe, search, conformal_params, and save flags
    :param dh: Data handler used in training and feature preparation
    :param layer_data: LayerData instance with bags, labels, and masks
    :return: None
    """
    layer_id = layer_data.layer_id
    X_tr = layer_data.X_tr
    X_te = layer_data.X_te
    X_cal = layer_data.X_cal
    y_train = layer_data.y_train
    y_test = layer_data.y_test
    y_cal = layer_data.y_cal
    mask_tr = layer_data.mask_tr
    mask_te = layer_data.mask_te
    mask_cal = layer_data.mask_cal

    start_time = time.time()

    # Training and hyperparameter search
    if cfg.search:
        result = parameter_search_sAwMIL(
            X=X_tr,
            y=y_train,
            dh=dh,
            mask=mask_tr,
            cfg=cfg,
            layer_id=layer_id,
        )
    else:
        try:
            result = single_training_sAwMIL(
                X=X_tr,
                y=y_train,
                dh=dh,
                layer_id=layer_id,
                mask=mask_tr,
                cfg=cfg,
            )
        except Exception as e:
            log.error(f"Error: {e}")
            log.warning(
                "\tSkipping the [%s] layer and moving to the next one..." % layer_id
            )
            return

    non_norm_dir, non_norm_bias = result["separator"].linearize(normalize=False)
    direction, bias = result["separator"].linearize(normalize=True)

    # Filter out empty bags for train, test, and calibration
    X_tr_f, y_tr_f, _ = _filter_empty_bags(
        X_tr, y_train, mask_tr, where=f"train (layer {layer_id})"
    )
    X_te_f, y_te_f, mask_te_f = _filter_empty_bags(
        X_te, y_test, mask_te, where=f"test (layer {layer_id})"
    )
    if (y_cal is not None) and (mask_cal is not None) and (X_cal is not None):
        X_cal_f, y_cal_f, mask_cal_f = _filter_empty_bags(
            X_cal, y_cal, mask_cal, where=f"calibration (layer {layer_id})"
        )
    else:
        raise RuntimeError("Calibration data required for conformal sAwMIL.")

    # BagProcessor and raw scores
    bp = BagProcessor(
        max_bag_size=max(cfg.probe["max_bag_size"], 50),
        pos_labels_in_bag=cfg.probe["num_known_positives"],
        scaler=result["scaler"],
    )
    yh_te = bp.predict_scores(bags=X_te_f, direction=direction, bias=bias)
    yh_cal = bp.predict_scores(bags=X_cal_f, direction=direction, bias=bias)

    # Conformal prediction setup
    if cfg.conformal_params["nc"] == "symmetric":
        nc = symmetric_nonconformity
    else:
        raise NotImplementedError(
            f"Nonconformity function {cfg.conformal_params['nc']} is not implemented."
        )

    calibrator = InductiveConformalPredictor(
        nonconformity_func=nc,
        alpha=cfg.conformal_params["alpha"],
        tie_breaking=cfg.conformal_params["tie_breaking"],
    )
    calibrator.fit(
        y=y_cal_f[mask_cal_f],
        scores=yh_cal[mask_cal_f],
    )
    yc_te = calibrator.predict(yh_te)

    # Metric computation
    metric_dict = {}

    metric_dict["default"] = log_metric_sAwMIL(
        preds=np.array(yh_te > 0),
        scores=yh_te,
        y_true=y_te_f,
        mask=mask_te_f,
        cfg=cfg,
    )
    metric_dict["default"]["coverage"] = 1.0
    metric_dict["default"]["C"] = result["separator"].C
    metric_dict["default"]["eta"] = result["eta"]

    metric_dict["conformal"] = log_metric_sAwMIL(
        preds=yc_te,
        scores=yh_te,
        y_true=y_te_f,
        mask=mask_te_f,
        cfg=cfg,
    )
    metric_dict["conformal"]["coverage"] = calibrator.coverage(
        scores=yh_te[mask_te_f],
        y=y_te_f[mask_te_f],
    )
    metric_dict["conformal"]["acceptance_rate"] = calibrator.acceptance_rate(yh_te)

    # Save results or log metrics
    if cfg.save_results:
        save_sAwMIL(
            non_normalized_direction=non_norm_dir,
            non_normalized_bias=non_norm_bias,
            direction=direction,
            bias=bias,
            metric_dict=metric_dict,
            scaler=result["scaler"],
            transformer=result["transformer"],
            conf_calibrator=calibrator,
            cfg=cfg,
            layer_id=layer_id,
            y_hat=yh_te,
            y_true=y_te_f,
        )
    else:
        log.warning(f"Conformal metric: {metric_dict['conformal']}")
        log.warning(f"Default metric: {metric_dict['default']}")

    end_time = time.time()
    first_part = f"\t{cfg.probe.name} probe took"
    log.warning(
        f" L{layer_id} {first_part:<20} {(end_time - start_time):<4.2f} "
        f"seconds | MCC (def): {metric_dict['default']['wmcc'][0]:>5.2f} "
        f"| MCC (conf): {metric_dict['conformal']['wmcc'][0]:>5.2f}"
    )
    log.warning(
        "\t\tA-rate (def): "
        f"{metric_dict['default']['acceptance_rate']:>5.2f} "
        "| A-rate (conf): "
        f"{metric_dict['conformal']['acceptance_rate']:>5.2f}"
    )
    log.warning(
        "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
    )