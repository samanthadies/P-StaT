"""
linear_mdc.py

Mean Difference Classifier (MDC) linear probe utilities.

- mean_diff: Mean Difference Classifier on pooled activations
    * Uses pooled (last_embedding) representations.
    * Hyperparameter search over {with_covariance, fit_intercept} if cfg.search=True.
    * Saves metrics_default_{layer}.json and metrics_conformal_{layer}.json.

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
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import energy_distance
from sklearn.metrics import average_precision_score as mAP, matthews_corrcoef as mcc, \
    adjusted_mutual_info_score as ami, adjusted_rand_score as ari

from probes.mean_difference import MeanDifferenceClassifier as MDC
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from probes.transforms import SparseTransform

log = logging.getLogger(__name__)


def binary_coverage(y_true, y_pred):
    """
    Compute the coverage for binary predictions with abstentions.

    Coverage is defined as the fraction of predictions that are not abstentions
    (i.e., y_pred != -1).

    :param y_true: Ground-truth labels
    :param y_pred: Predicted labels or abstentions (-1)
    :return: Coverage in [0, 1] as a float
    """
    return float(np.mean(y_pred != -1))


def log_metric_MDC(
    probs,
    ytrue,
):
    """
    Compute evaluation metrics for the mean-difference classifier, including
    - mean average precision (mAP)
    - Matthews correlation coefficient (MCC)
    - adjusted mutual information (AMI)
    - adjusted Rand index (ARI)
    - energy distance between normalized scores for the two classes

    :param probs: Decision scores (x·w + b) as a 1D array
    :param ytrue: Ground-truth labels in {0, 1}
    :return: Dictionary of metrics and sample count
    """
    yhat = (probs > 0).astype(int)
    out = {}

    try:
        ap = mAP(ytrue, probs)
    except Exception:
        ap = 0.0
    try:
        mcc_v = mcc(ytrue, yhat)
    except Exception:
        mcc_v = 0.0
    try:
        ami_v = ami(ytrue, yhat)
    except Exception:
        ami_v = 0.0
    try:
        ari_v = ari(ytrue, yhat)
    except Exception:
        ari_v = 0.0
    try:
        if len(probs) > 1:
            p = probs
            p = (p - p.min()) / (p.max() - p.min() + 1e-12)
            e = energy_distance(p[ytrue == 0], p[ytrue == 1])
        else:
            e = 0.0
    except Exception:
        e = 0.0

    out.update(
        dict(map=ap, mcc=mcc_v, ami=ami_v, ari=ari_v, energy=e, n=int(ytrue.shape[0]))
    )
    return out


def save_MDC(
    direction_raw,
    direction_unit,
    bias,
    scaler,
    transformer,
    conf_calibrator,
    metric_default,
    metric_conformal,
    cfg,
    layer_id,
    y_hat=None,
    y_true=None,
):
    """
    Save MDC outputs and metadata for a given layer.

    This writes all outputs in the mean_diff-style format:
    - config.json
    - metrics_default_{layer}.json
    - metrics_conformal_{layer}.json
    - coef_{layer}.npy
    - NON_NORMALIZED_coef_{layer}.npy
    - bias_{layer}.npy
    - optional scaler, transformer, conformal calibrator pickles
    - optional y_hat_{layer}.npy and y_true.npy

    :param direction_raw: Raw weight vector (unnormalized) as a 1D array
    :param direction_unit: Normalized weight vector as a 1D array
    :param bias: Scalar bias term
    :param scaler: Fitted StandardScaler or None
    :param transformer: Fitted SparseTransform or None
    :param conf_calibrator: Fitted conformal predictor or None
    :param metric_default: Metrics dict for standard MDC predictions
    :param metric_conformal: Metrics dict for conformal predictions
    :param cfg: Hydra config with an output_dir field
    :param layer_id: Integer layer index
    :param y_hat: Optional array of test decision scores
    :param y_true: Optional array of test labels
    :return: None
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(f"{cfg.output_dir}/config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)

    with open(f"{cfg.output_dir}/metrics_default_{layer_id}.json", "w") as f:
        json.dump(metric_default, f)
    with open(f"{cfg.output_dir}/metrics_conformal_{layer_id}.json", "w") as f:
        json.dump(metric_conformal, f)

    np.save(f"{cfg.output_dir}/coef_{layer_id}.npy", direction_unit)
    np.save(
        f"{cfg.output_dir}/NON_NORMALIZED_coef_{layer_id}.npy", direction_raw
    )
    np.save(f"{cfg.output_dir}/bias_{layer_id}.npy", np.array(bias, dtype=float))

    if scaler is not None:
        with open(f"{cfg.output_dir}/scaler_{layer_id}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    if transformer is not None:
        with open(f"{cfg.output_dir}/transformer_{layer_id}.pkl", "wb") as f:
            pickle.dump(transformer, f)
    if conf_calibrator is not None:
        with open(f"{cfg.output_dir}/calibrator_{layer_id}.pkl", "wb") as f:
            pickle.dump(conf_calibrator, f)

    if y_hat is not None:
        np.save(f"{cfg.output_dir}/y_hat_{layer_id}.npy", y_hat)
    if y_true is not None:
        np.save(f"{cfg.output_dir}/y_true.npy", y_true)


def single_training_MDC(
    X,
    y,
    dh,
    layer_id,
    mask,
    cfg,
):
    """
    Train a mean-difference classifier (MDC) without hyperparameter search.

    This mirrors the original MDC single_training logic:
    - apply mask to X and y
    - optionally normalize data with StandardScaler
    - optionally sparsify with SparseTransform
    - fit MDC with parameters from cfg.probe.init_params

    :param X: 2D array of pooled activations (n_samples, n_features)
    :param y: 1D array of binary labels in {0, 1}
    :param dh: Data handler
    :param layer_id: Integer layer index
    :param mask: Boolean mask over samples to include in training
    :param cfg: Hydra config with probe and sparsify_data settings
    :return: Dictionary with trained separator, scaler, transformer, and weights
    """
    mask = np.array(mask).astype(bool)
    y = y[mask]
    X = X[mask]
    np.random.seed(cfg["random_seed"])

    if cfg.probe["normalize_data"]:
        log.warning("\tNormalizing the data...")
        scaler = StandardScaler().fit(X)
        Xt = scaler.transform(X)
    else:
        raise NotImplementedError(
            "Only a pipeline with the normalization is implemented"
        )

    if cfg["sparsify_data"] > 0:
        log.warning(f"\tSparsifying the data [{cfg['sparsify_data']}]...")
        prj = SparseTransform(
            max_k=cfg["sparsify_data"], normalize=False, show_progress=False
        )
        Xt = scaler.transform(Xt)
    else:
        prj = None

    separator = MDC(
        fit_intercept=cfg.probe["init_params"]["fit_intercept"],
        with_covariance=cfg.probe["init_params"]["with_covariance"],
        verbose=cfg.probe["init_params"]["verbose"],
        cov_type=cfg.probe["init_params"]["cov_type"],
    )
    separator.fit(Xt, y)

    w_raw = np.ravel(separator.coef_)
    b_raw = getattr(separator, "intercept_", 0.0)
    if isinstance(b_raw, (list, tuple, np.ndarray)):
        b = float(b_raw[0])
    else:
        b = float(b_raw)
    w_unit = w_raw / (np.linalg.norm(w_raw) + 1e-12)

    return {
        "separator": separator,
        "scaler": scaler,
        "transformer": prj,
        "w_raw": w_raw,
        "b_raw": b,
        "w_unit": w_unit,
    }


def parameter_search_MDC(
    X,
    y,
    dh,
    mask,
    cfg,
    layer_id,
):
    """
    Perform grid search over MDC hyperparameters using K-fold cross-validation.

    The search space is:
    - with_covariance values from cfg.probe.param_grid.with_covariance
    - fit_intercept values from cfg.probe.param_grid.fit_intercept

    Each parameter combination is evaluated with mean average precision (mAP)
    on the masked data. The best combination is written back into cfg.probe.init_params,
    and a final MDC model is trained on all data using single_training_MDC.

    :param X: 2D array of pooled activations (n_samples, n_features)
    :param y: 1D array of binary labels in {0, 1}
    :param dh: Data handler (unused here, kept for API compatibility)
    :param mask: Boolean mask over samples to include in the search
    :param cfg: Hydra config with probe, cv_n_folds, and random_seed
    :param layer_id: Integer layer index (for downstream bookkeeping)
    :return: Dictionary from single_training_MDC with best model and parameters
    """
    log.warning("Running the hyperparameter search for MDC...")
    grid_cov = cfg.probe.get("param_grid", {}).get(
        "with_covariance", [cfg.probe["init_params"]["with_covariance"]]
    )
    grid_int = cfg.probe.get("param_grid", {}).get(
        "fit_intercept", [cfg.probe["init_params"]["fit_intercept"]]
    )
    combos = [(wc, fi) for wc in grid_cov for fi in grid_int]

    y = y[mask]
    X = X[mask]

    kf = KFold(
        n_splits=cfg.cv_n_folds,
        shuffle=True,
        random_state=cfg.random_seed,
    )

    best_score = -np.inf
    best_params = None

    for (wc, fi) in combos:
        fold_scores = []
        for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = MDC(
                with_covariance=wc,
                fit_intercept=fi,
                verbose=cfg.probe["init_params"]["verbose"],
                cov_type=cfg.probe["init_params"]["cov_type"],
            )
            clf.fit(X_tr_s, y_tr)

            w = np.ravel(clf.coef_)
            raw_b = getattr(clf, "intercept_", 0.0)
            if isinstance(raw_b, (list, tuple, np.ndarray)):
                b = float(raw_b[0])
            else:
                b = float(raw_b)
            y_hat = X_te_s @ w + b

            try:
                score = mAP(y_te, y_hat)
            except Exception:
                score = 0.0
            fold_scores.append(score)

        mean_ap = float(np.mean(fold_scores))
        log.warning(
            f"\tParams (with_covariance={wc}, fit_intercept={fi}) -> mean AP: {mean_ap:.4f}"
        )
        if mean_ap > best_score:
            best_score = mean_ap
            best_params = (wc, fi)

    OmegaConf.set_struct(cfg, False)
    cfg.probe["init_params"]["with_covariance"] = best_params[0]
    cfg.probe["init_params"]["fit_intercept"] = best_params[1]
    OmegaConf.set_struct(cfg, True)

    log.warning(
        f"\tSelected params: with_covariance={best_params[0]}, "
        f"fit_intercept={best_params[1]} (AP={best_score:.4f})"
    )

    full_mask = np.ones_like(y, dtype=bool)
    return single_training_MDC(X, y, dh, layer_id, full_mask, cfg)


def run_mdc_layer(
    cfg,
    dh,
    layer_data,
):
    """
    Run the mean-difference (MDC) probe for a single layer.

    This function orchestrates:
    - training MDC (with or without hyperparameter search)
    - computing decision scores on test and calibration sets
    - fitting a conformal predictor on calibration scores
    - computing metrics for default and conformal predictions
    - saving metrics, weights, and auxiliary objects to disk

    :param cfg: Hydra config with probe, search, and conformal_params settings
    :param dh: Data handler used for consistency with other probe drivers
    :param layer_data: LayerData instance with pooled features and masks
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

    if cfg.search:
        result = parameter_search_MDC(
            X=X_tr, y=y_train, dh=dh, mask=mask_tr, cfg=cfg, layer_id=layer_id
        )
    else:
        result = single_training_MDC(
            X=X_tr, y=y_train, dh=dh, layer_id=layer_id, mask=mask_tr, cfg=cfg
        )

    scaler = result["scaler"]
    Xte_s = scaler.transform(X_te)
    y_test_hat = Xte_s @ result["w_raw"] + result["b_raw"]

    if X_cal is None or y_cal is None or mask_cal is None:
        raise RuntimeError("Calibration data required for conformal MDC.")

    Xcal_s = scaler.transform(X_cal)
    y_cal_hat = Xcal_s @ result["w_raw"] + result["b_raw"]

    conformity = InductiveConformalPredictor(
        nonconformity_func=symmetric_nonconformity,
        alpha=cfg.conformal_params["alpha"],
        tie_breaking=cfg.conformal_params.get("tie_breaking", True),
    )
    conformity.fit(y_cal_hat[mask_cal], y_cal[mask_cal])
    conformal_preds = conformity.predict(y_test_hat)
    conformal_mask = conformal_preds != -1

    metric_dict_default = log_metric_MDC(
        y_test_hat[mask_te], y_test[mask_te]
    )
    metric_dict_default["coverage"] = 1.0

    metric_dict_conformal = log_metric_MDC(
        y_test_hat[conformal_mask & mask_te],
        y_test[conformal_mask & mask_te],
    )
    metric_dict_conformal["coverage"] = binary_coverage(
        y_test[mask_te], conformal_preds[mask_te]
    )
    metric_dict_conformal["params"] = OmegaConf.to_container(
        cfg.probe["init_params"]
    )

    save_MDC(
        direction_raw=result["w_raw"],
        direction_unit=result["w_unit"],
        bias=result["b_raw"],
        scaler=result["scaler"],
        transformer=result["transformer"],
        conf_calibrator=conformity,
        metric_default=metric_dict_default,
        metric_conformal=metric_dict_conformal,
        cfg=cfg,
        layer_id=layer_id,
        y_hat=y_test_hat,
        y_true=y_test,
    )

    end_time = time.time()
    first_part = f"\t{cfg.probe.name} probe took"
    log.warning(
        f"{first_part:<20} {(end_time - start_time):<4.2f} seconds | "
        f"MCC (def): {metric_dict_default['mcc']:>5.2f} | "
        f"MCC (conf): {metric_dict_conformal['mcc']:>5.2f}"
    )
    log.warning(
        f"\t\tcoverage (def): {metric_dict_default['coverage']:>5.2f} | "
        f"coverage (conf): {metric_dict_conformal['coverage']:>5.2f}"
    )
    log.warning(
        "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
    )