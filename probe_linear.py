"""
probe_linear.py

Unified linear probe driver for:
- sAwMIL: MIL/bag-based probe
    * Uses bags (sequences of token embeddings).
    * Hyperparameter search over C if cfg.search=True.
    * Saves metrics as metrics_{layer}.json with 'default' and 'conformal' keys.
- mean_diff: Mean Difference Classifier on pooled activations
    * Uses pooled (last_embedding) representations.
    * Hyperparameter search over {with_covariance, fit_intercept} if cfg.search=True.
    * Saves metrics_default_{layer}.json and metrics_conformal_{layer}.json.

"""

import logging
import os
import re
from glob import glob

import hydra
from omegaconf import DictConfig, OmegaConf

from loaders.task_stability import Task
from utils import (
    load_data,
    return_label,
    should_process_layer,
    LayerData,
)

from probes.linear_sawmil import run_sawmil_layer
from probes.linear_mdc import run_mdc_layer

log = logging.getLogger(__name__)


def load_layer_data(
    dh,
    dh_test,
    task,
    cfg,
    layer_id,
    use_bags,
):
    """
    Load train/test/calibration features and labels for a given layer.

    :param dh: DataHandler for train/calibration data
    :param dh_test: DataHandler for test data
    :param task: Task instance for label mapping
    :param cfg: Hydra config
    :param layer_id: integer layer index
    :param use_bags: True for bag-based embeddings (sAwMIL), False for pooled (MDC)
    :return: LayerData instance or None on failure
    """
    try:
        if use_bags:
            X_tr = dh.train_bags(layer_id=layer_id, drop_zeros=True)["embeddings"]
        else:
            X_tr = dh.train_bags(layer_id=layer_id, drop_zeros=True)["last_embedding"]
    except Exception as e:
        log.error(
            f"Error: Could not load the train data for layer {layer_id}: {e}. Skipping..."
        )
        return None

    data_train = dh.get_train_df().reset_index(drop=True)
    _y_train, r_train, _, _, _, f_tr, d_tr, n_tr = return_label(data_train)

    try:
        if use_bags:
            X_te = dh_test.test_bags(layer_id=layer_id, drop_zeros=True)["embeddings"]
        else:
            X_te = dh_test.test_bags(layer_id=layer_id)["last_embedding"].numpy()
    except Exception as e:
        log.error(
            f"Error: Could not load the test data for layer {layer_id}: {e}. Skipping..."
        )
        return None

    data_test = dh_test.get_test_df().reset_index(drop=True)
    _y_test, r_test, _, _, _, f_te, d_te, n_te = return_label(data_test)

    X_cal = None
    _y_cal = r_cal = f_cal = d_cal = n_cal = None
    if getattr(dh, "with_calibration", False) and dh.calibration_ids is not None:
        try:
            if use_bags:
                X_cal = dh.cal_bags(layer_id=layer_id, drop_zeros=True)["embeddings"]
            else:
                X_cal = dh.cal_bags(layer_id=layer_id)["last_embedding"].numpy()
        except Exception as e:
            log.error(
                f"Error: Could not load calibration data for layer {layer_id}: {e}. Skipping..."
            )
            return None

        data_cal = dh.get_cal_df().reset_index(drop=True)
        _y_cal, r_cal, _, _, _, f_cal, d_cal, n_cal = return_label(data_cal)

    train_labels = task.return_labels(
        _y_train, r_train, fictional=f_tr, noise=n_tr
    )
    y_train, mask_tr = train_labels["targets"], train_labels["mask"]

    test_labels = task.return_labels(
        _y_test, r_test, fictional=f_te, noise=n_te
    )
    y_test, mask_te = test_labels["targets"], test_labels["mask"]

    if getattr(dh, "with_calibration", False) and X_cal is not None:
        cal_labels = task.return_labels(
            _y_cal, r_cal, fictional=f_cal, noise=n_cal
        )
        y_cal, mask_cal = cal_labels["targets"], cal_labels["mask"]
    else:
        y_cal = None
        mask_cal = None

    return LayerData(
        layer_id=layer_id,
        X_tr=X_tr,
        X_te=X_te,
        X_cal=X_cal,
        y_train=y_train,
        y_test=y_test,
        y_cal=y_cal,
        mask_tr=mask_tr,
        mask_te=mask_te,
        mask_cal=mask_cal,
    )


def validate_config(cfg):
    """
    Validate and finalize the probe configuration.

    :param cfg: Hydra configuration object
    """
    assert isinstance(
        cfg.datapack["datasets"], (list, type(cfg.datapack["datasets"]))
    ), f"Datasets must be a list. Not {type(cfg.datapack['datasets'])}"
    assert len(cfg.datapack["datasets"]) > 0, "At least one dataset must be selected."

    OmegaConf.set_struct(cfg, False)

    trial_name = cfg.trial_name
    if cfg.get("sparsify_data", 0) > 0:
        trial_name += f"_sparse-{cfg['sparsify_data']}"
    if cfg.search:
        trial_name += "_search"
    if cfg.noise:
        trial_name += f"_noise{cfg.noise}"
    trial_name += f"_task-{cfg.task}"
    cfg["trial_name"] = trial_name

    cfg["output_dir"] = os.path.join(cfg.output_dir, trial_name)
    log.warning(f"Output directory: {cfg['output_dir']}")
    OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    """
    Log which model, datasets, and probe are being trained.

    :param cfg: Hydra configuration object
    """
    if "datasets_test" in cfg.datapack and len(cfg.datapack["datasets_test"]) > 0:
        datasets_test = cfg.datapack["datasets_test"]
    else:
        datasets_test = cfg.datapack["datasets"]

    log.warning(
        f"Training {cfg.probe['name']}-based probe for {cfg.model['name']} "
        f"activations [task: {cfg.task}]"
    )
    log.warning(f"\t\tTrain datasets: {cfg.datapack['datasets']}")
    log.warning(f"\t\tTest datasets: {datasets_test}")
    log.warning(f"\t\tOutput directory: {cfg.output_dir}")


def get_target_layer(cfg):
    """
    Resolve the single layer to run, based on datapack name and a task_layers mapping.

    :param cfg: Hydra config
    :return: integer layer index
    """
    task_layers_cfg = None
    if "model" in cfg and "task_layers" in cfg.model:
        task_layers_cfg = cfg.model.task_layers

    if task_layers_cfg is None:
        raise ValueError("Config is missing a 'task_layers' mapping.")

    dp_name = cfg.datapack.get("name", None)
    if dp_name is None:
        raise ValueError("cfg.datapack.name is not set; cannot pick target layer.")

    if dp_name not in task_layers_cfg:
        raise ValueError(
            f"Could not find task_layers entry for datapack '{dp_name}'. "
        )

    layer_val = task_layers_cfg[dp_name]
    try:
        layer_id = int(layer_val)
    except Exception as e:
        raise ValueError(
            f"task_layers['{dp_name}'] must be an int, got {layer_val!r}"
        ) from e

    log.warning(f"Resolved target layer for datapack '{dp_name}': {layer_id}")
    return layer_id


def checkpointing(cfg, candidate_layers):
    """
    Given a list of candidate layers, return the subset that still need
    to be processed.

    :param cfg: Hydra config
    :param candidate_layers: list of integer layer indices
    :return: list of layer indices that still need processing
    """
    recorded_coefs = glob(f"{cfg.output_dir}/coef_*")
    completed_layers = []
    for file in recorded_coefs:
        match = re.search(r"coef_(\d+)", file)
        if match:
            completed_layers.append(int(match.group(1)))

    completed_set = set(completed_layers)
    missing_layers = [l for l in candidate_layers if l not in completed_set]

    log.warning(
        f"Checkpointing: completed layers={sorted(completed_set)}, "
        f"missing layers={missing_layers}"
    )
    return sorted(missing_layers)


@hydra.main(version_base=None, config_path="configs", config_name=None)
def main(cfg: DictConfig):

    validate_config(cfg)
    log_stats(cfg)

    dh = load_data(cfg)
    dh_test = dh
    task = Task(cfg.task)
    
    n_train = dh.train_ids.shape[0]
    n_test = dh.test_ids.shape[0]
    if dh.with_calibration and dh.calibration_ids is not None:
        n_cal = dh.calibration_ids.shape[0]
    else:
        n_cal = 0
    
    print("=== Split sizes after adding fictional + noise ===")
    print(f"Total statements in dh.data:    {dh.data.shape[0]}")
    print(f"Train set (df.train) size:      {n_train}")
    print(f"Test set size:                  {n_test}")
    print(f"Calibration set size:           {n_cal}")
    print("=================================================")

    target_layer = get_target_layer(cfg)
    candidate_layers = [target_layer]

    if cfg.start_from_checkpoint:
        layers = checkpointing(cfg, candidate_layers)
        if len(layers) == 0:
            log.warning(
                f"All candidate layers {candidate_layers} are already processed. "
                "Nothing to do."
            )
            raise SystemExit("All layers are already processed.")
        log.warning(f"Checkpointing: will process missing layers: {layers}")
    else:
        layers = candidate_layers
        log.warning(f"Running without checkpointing. Layers to process: {layers}")

    use_bags = bool(cfg.probe.get("with_bags", False))

    for layer_id in layers:
        if cfg.run_debugging and layer_id > 6 and should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id} || Debugging mode")
        elif (not cfg.run_debugging) and should_process_layer(layer_id, cfg):
            log.warning(
                f"Processing layer {layer_id} | model: {cfg.model['name']} | task: {cfg.task}"
            )
        else:
            log.warning(f"Skipping layer {layer_id} (should_process_layer=False)")
            continue

        layer_data = load_layer_data(
            dh=dh,
            dh_test=dh_test,
            task=task,
            cfg=cfg,
            layer_id=layer_id,
            use_bags=use_bags,
        )
        if layer_data is None:
            continue

        if use_bags:
            run_sawmil_layer(cfg=cfg, dh=dh, layer_data=layer_data)
        else:
            run_mdc_layer(cfg=cfg, dh=dh, layer_data=layer_data)

    log.warning(f"Finished running the {cfg.probe.name} probe.")


if __name__ == "__main__":
    main()
