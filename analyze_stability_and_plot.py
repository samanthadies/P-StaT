"""
run_full_analysis.py

End-to-end analysis and plotting pipeline.
1) For each datapack in cfg.datasets_for_plots:
   - Generates merged analysis DataFrames from probe outputs.
   - Generates per-layer sW1 CSVs for activation heatmaps.
2) Once all per-datapack artifacts exist, generates all plots by calling:
   - plot_scripts/plot_ling_rep_dists.py
   - plot_scripts/plot_activation_heatmaps.py
   - plot_scripts/plot_boundary_heatmaps.py
   - plot_scripts/plot_flip_barcharts.py
   - plot_scripts/plot_flip_barcharts_zs.py

"""

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from utils import load_data

from plot_scripts.plot_activation_heatmaps import (
    generate_wasserstein_csvs,
    plot_single_model_activation_heatmaps,
)
from plot_scripts.plot_boundary_heatmaps import plot_boundary_heatmaps
from plot_scripts.plot_flip_barcharts import plot_flip_barchart
from plot_scripts.plot_flip_barcharts_zs import plot_zero_shot_flip_barchart
from plot_scripts.plot_ling_rep_dists import make_combined_figure


def _pick_best_yhat_file(cands):
    """
    Pick the "best" y_hat_*.npy file from a list of candidates.

    Preference:
      1) Highest numeric suffix y_hat_<N>.npy if present.
      2) Otherwise, most recent file by modification time.

    :param cands: list of candidate filepaths
    :return: chosen filepath
    :raises FileNotFoundError: if no candidates are given
    """
    if not cands:
        raise FileNotFoundError("No y_hat_*.npy files found.")

    pairs = []
    for fp in cands:
        m = re.search(r"y_hat_(\d+)\.npy$", os.path.basename(fp))
        if m:
            pairs.append((int(m.group(1)), fp))

    if pairs:
        pairs.sort(key=lambda x: x[0])
        return pairs[-1][1]

    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def _scores_and_preds_from_array(arr):
    """
    Convert a raw y_hat array into (scores, preds).

    Supported shapes:
      - 1D: scores directly
      - 2D, shape (N, 1): scores = arr[:, 0]
      - 2D, shape (N, 2):
          * probabilities [neg, pos] in [0,1] with row-sums ~1
          * logits/margins [neg, pos] (scores = pos - neg)

    :param arr: numpy array loaded from y_hat_*.npy
    :return: tuple (scores, preds) as 1D numpy arrays
    :raises ValueError: on unsupported array shape
    """
    arr = np.asarray(arr)

    if arr.ndim == 1:
        scores = arr
        preds = (scores > 0).astype(int)
        return scores, preds

    if arr.ndim == 2 and arr.shape[1] == 1:
        scores = arr[:, 0]
        preds = (scores > 0).astype(int)
        return scores, preds

    if arr.ndim == 2 and arr.shape[1] == 2:
        row_sums = arr.sum(axis=1)
        in_01 = (arr >= -1e-8).all() and (arr <= 1 + 1e-8).all()
        sums_close_1 = np.allclose(row_sums, 1.0, atol=1e-3)

        if in_01 and sums_close_1:
            # Probabilities [neg, pos]
            p_pos = arr[:, 1]
            scores = p_pos
            preds = (p_pos >= 0.5).astype(int)
            return scores, preds
        else:
            # Logits/margins [neg, pos]
            margin = arr[:, 1] - arr[:, 0]
            scores = margin
            preds = (margin > 0).astype(int)
            return scores, preds

    raise ValueError(f"Unexpected y_hat array shape: {arr.shape}")


def autodiscover_models(probes_root, probe_name):
    """
    Autodiscover models for a given probe under a probes_root directory.

    Assumes layout:
        <probes_root>/<probe_name>/<model>/

    :param probes_root: root directory containing all probe outputs
    :param probe_name: probe name subdirectory under probes_root
    :return: sorted list of model names
    """
    path = os.path.join(probes_root, probe_name)
    if not os.path.isdir(path):
        return []
    return sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )


def load_model_res(model, task, cfg, probe_name, noise):
    """
    Load scores and predictions for a single (model, task) from disk.

    For non-zero-shot probes, expects:
        <probes_root>/<probe_name>/<model>/
            <dataset>_search_noise<noise>_task-<task>/y_hat_*.npy

    For zero_shot, expects:
        <probes_root>/zero_shot/<model>/<dataset>_task-<task>/
            preds.npy   (N,)   argmax over {0,1,2} for [a,b,c]
            scores.npy  (N,4)  probabilities over [a,b,c,else]

    Returns:
        preds_list: list[int] binary 0/1 where 1 means predicted "a"
        scores_list: list[float] 1D score = P("a")
    """
    probes_root = cfg.paths.probes_root
    dataset = cfg.datapack.name

    if probe_name == "zero_shot":
        run_dirname = f"{dataset}_task-{task}"
        base = os.path.join(probes_root, probe_name, model, run_dirname)

        preds_fp = os.path.join(base, "preds.npy")
        scores_fp = os.path.join(base, "scores.npy")

        if not os.path.exists(preds_fp):
            raise FileNotFoundError(f"Missing zero-shot preds: {preds_fp}")
        if not os.path.exists(scores_fp):
            raise FileNotFoundError(f"Missing zero-shot scores: {scores_fp}")

        preds_mc = np.load(preds_fp)          # (N,) in {0,1,2}
        scores_abc = np.load(scores_fp)       # (N,4) [a,b,c,else]

        preds_mc = np.asarray(preds_mc).reshape(-1)
        scores_abc = np.asarray(scores_abc)

        if scores_abc.ndim != 2 or scores_abc.shape[1] < 1:
            raise ValueError(f"Unexpected scores.npy shape for zero-shot: {scores_abc.shape}")
        if len(preds_mc) != scores_abc.shape[0]:
            raise ValueError(
                f"Length mismatch in zero-shot artifacts: "
                f"len(preds)={len(preds_mc)} vs scores.shape[0]={scores_abc.shape[0]}"
            )

        # Binary decision: "a" (index 0) means "correct"
        preds_bin = (preds_mc == 0).astype(int)   # (N,)

        # 1D score: probability assigned to "a"
        scores_1d = scores_abc[:, 0].astype(float)

        return preds_bin.tolist(), scores_1d.tolist()

    # Non-zero-shot
    run_dirname = f"{dataset}_search_noise{noise}_task-{task}"
    base = os.path.join(probes_root, probe_name, model, run_dirname)

    pattern = os.path.join(base, "y_hat_*.npy")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern}")

    fp = _pick_best_yhat_file(matches)
    yhat = np.load(fp)

    scores, preds = _scores_and_preds_from_array(yhat)
    return preds.tolist(), scores.tolist()


def get_combined_df(data_test, models, cfg, probe_name, noise, zero_shot_alt_baseline=False):
    """
    Build a combined DataFrame of predictions and scores across models and tasks.

    For each model and each task (0-4), this function adds columns:
        <model>_task<task>_pred
        <model>_task<task>_scores

    :param data_test: test set DataFrame from DataHandler
    :param models: list of model names
    :param cfg: Hydra config with datapack and paths
    :param probe_name: probe name whose outputs to read
    :param noise: noise level integer used in filenames
    :param zero_shot_alt_baseline: bool for whether to use alternative zero-shot baseline
    :return: pandas DataFrame with base columns and model outputs
    """
    df = pd.DataFrame()
    df[["statement", "correct", "negation", "real_object"]] = data_test[
        ["statement", "correct", "negation", "real_object"]
    ]

    TASKS = [0, 1, 2, 3, 4]
    for model in models:
        for task in TASKS:
            # Only for zero-shot: optionally use task=5 as the *loaded* baseline
            task_to_load = 5 if (probe_name == "zero_shot" and task == 0 and zero_shot_alt_baseline) else task

            try:
                preds, scores = load_model_res(
                    model=model,
                    task=task_to_load,
                    cfg=cfg,
                    probe_name=probe_name,
                    noise=noise,
                )

                # Always write columns using the *analysis task id* (0..4)
                df[f"{model}_task{task}_pred"] = preds
                df[f"{model}_task{task}_scores"] = scores

            except Exception as e:
                print(
                    f"[warn] Skipping model={model}, task={task} "
                    f"(loaded task={task_to_load}) (len(df)={len(df.index)}). Error: {e}"
                )
                continue

    return df


def generate_merged_analysis_dfs(cfg):
    """
    Generate merged analysis DataFrames for the configured datapack and probe.

    Writes two CSVs:
      - full merged dataset
      - y=True subset (correct == 1 and real_object == 1)

    Output layout:
        <paths.merged_root>/<probe_name>/
            <datapack.name>_merged_noise<noise>.csv
            <datapack.name>_merged_y=true_noise<noise>.csv

    :param cfg: Hydra config with datapack, probe, noise, and paths
    :return: None
    """
    warnings.filterwarnings("ignore")

    if hasattr(cfg.probe, "name"):
        probe_name = cfg.probe.name
    else:
        probe_name = str(cfg.probe)

    noise = cfg.noise

    # Load test statements for this datapack
    dh = load_data(cfg)
    data_test = dh.get_test_df().reset_index(drop=True)

    # Zero-shot outputs are computed on the "strictly true" subset, so merge on that subset.
    if probe_name == "zero_shot":
        data_test = data_test[(data_test["correct"] == 1) & (data_test["real_object"] == 1)].reset_index(drop=True)
        print(f"[info] zero_shot merge restricted to y=True subset: n={len(data_test)}")

    # Discover models
    models = autodiscover_models(cfg.paths.probes_root, probe_name)
    print(f"[info] Probe: {probe_name}")
    print(f"[info] Datapack: {cfg.datapack.name}")
    print(f"[info] Models: {', '.join(models)}")

    # Output directory for merged CSVs
    out_root = os.path.join(cfg.paths.merged_root, probe_name)
    os.makedirs(out_root, exist_ok=True)

    baseline_variants = [("default", False)]
    if probe_name == "zero_shot":
        baseline_variants.append(("alt_baseline", True))

    for baseline_tag, use_alt_baseline in baseline_variants:
        # Combined DataFrame
        df = get_combined_df(
            data_test=data_test,
            models=models,
            cfg=cfg,
            probe_name=probe_name,
            noise=noise,
            zero_shot_alt_baseline=use_alt_baseline,
        )

        if probe_name == "zero_shot":
            merged_fp = os.path.join(out_root, f"{cfg.datapack.name}_merged_{baseline_tag}.csv")
            df.to_csv(merged_fp, index=False)
            print(f"[merged] wrote {merged_fp} (zero_shot; y=True only; n={len(df.index)})")

        else:
            merged_fp = os.path.join(out_root, f"{cfg.datapack.name}_merged_noise{noise}.csv")
            df.to_csv(merged_fp, index=False)
            print(f"[merged] wrote {merged_fp}")

            df_true = df[(df["correct"] == 1) & (df["real_object"] == 1)].copy()
            merged_true_fp = os.path.join(out_root, f"{cfg.datapack.name}_merged_y=true_noise{noise}.csv")
            df_true.to_csv(merged_true_fp, index=False)
            print(f"[merged] wrote {merged_true_fp} (y=True subset; n={len(df_true.index)})")


def make_datapack_cfg(base_cfg, datapack_name):
    """
    Build a per-datapack config by merging the baseline datapack config
    with configs/datapack/<datapack_name>.yaml.

    This preserves shared fields like cal_size while allowing per-datapack
    overrides such as the dataset name.

    :param base_cfg: original Hydra config
    :param datapack_name: datapack group name (e.g., 'cities_loc')
    :return: new config object with datapack overridden
    """
    orig_cwd = get_original_cwd()
    dp_path = os.path.join(orig_cwd, "configs", "datapack", f"{datapack_name}.yaml")
    dp_cfg = OmegaConf.load(dp_path)

    # Deep copy the full base config
    cfg_dp = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    OmegaConf.set_struct(cfg_dp, False)

    # Merge new datapack config into the existing one
    merged_dp = OmegaConf.merge(cfg_dp.datapack, dp_cfg)
    cfg_dp.datapack = merged_dp

    OmegaConf.set_struct(cfg_dp, True)

    return cfg_dp


@hydra.main(version_base=None, config_path="configs", config_name="analysis_pipeline")
def main(cfg):
    """
    Run the full analysis and plotting pipeline.

    Behavior:
      - For probe == "zero_shot": run ONLY
          (1) merged analysis CSVs (both baseline tracks) and
          (6) zero-shot flip bar charts (both baseline tracks),
        then exit.
      - For all other probes: run steps (1)–(5)

    :param cfg: Hydra config loaded from analysis_pipeline.yaml
    :return: None
    """
    datasets_for_plots = list(cfg.datasets_for_plots)

    # Determine probe name early (used to branch the pipeline).
    if hasattr(cfg.probe, "name"):
        probe_name = cfg.probe.name
    else:
        probe_name = str(cfg.probe)

    # ZERO-SHOT

    if probe_name == "zero_shot":
        print("[info] Probe is zero_shot -> running merged CSVs + zero-shot flip bar charts only.")

        # 1) Per-datapack artifacts: merged CSVs ONLY
        for dp_name in datasets_for_plots:
            print(f"\n=== Processing datapack: {dp_name} ===")
            cfg_dp = make_datapack_cfg(cfg, dp_name)
            generate_merged_analysis_dfs(cfg_dp)

        print("\n[stage] All datapacks processed: merged CSVs (zero-shot)")

        # 6) Zero-shot flip bar charts – both baseline tracks
        zs_baselines = ["default", "alt_baseline"]
        for dp_name in datasets_for_plots:
            for baseline_tag in zs_baselines:
                plot_zero_shot_flip_barchart(
                    dataset=dp_name,
                    merged_root=Path(cfg.paths.merged_root),
                    output_dir=Path(cfg.paths.plots_root),
                    baseline_tag=baseline_tag,
                )
                print(f"[stage] Zero-shot flip bar chart done for {dp_name} ({baseline_tag})")

        print("\n[done] Zero-shot analysis completed.")
        return

    # FULL PIPELINE (non-zero-shot)

    # 1) Per-datapack artifacts: merged CSVs + sW1 CSVs
    for dp_name in datasets_for_plots:
        print(f"\n=== Processing datapack: {dp_name} ===")
        cfg_dp = make_datapack_cfg(cfg, dp_name)

        # 1a) Merged analysis DFs for this datapack
        generate_merged_analysis_dfs(cfg_dp)

        # 1b) Per-layer sW1 CSVs for this model+datapack
        generate_wasserstein_csvs(
            cfg_dp,
            out_dir=cfg.paths.model_level_root,
            pool_3d="last_nonzero",
            n_projections=128,
            max_per_class=5000,
        )

    print("\n[stage] All datapacks processed: merged CSVs + sW1 CSVs")

    # 2) N-gram distribution plots (doesn't depend on cfg.datapack)
    csv_paths = {
        "City\nLocations": [
            "datasets/cities_loc_true_false.csv",
            "datasets/cities_loc_synthetic.csv",
            "datasets/cities_loc_fictional.csv",
        ],
        "Medical\nIndications": [
            "datasets/med_indications_true_false.csv",
            "datasets/med_indications_synthetic.csv",
            "datasets/med_indications_fictional.csv",
        ],
        "Word\nDefinitions": [
            "datasets/defs_true_false.csv",
            "datasets/defs_synthetic.csv",
            "datasets/defs_fictional.csv",
        ],
    }

    heatmap_ds_keys = {
        "City\nLocations": "cities_loc",
        "Medical\nIndications": "med_indications",
        "Word\nDefinitions": "defs",
    }

    make_combined_figure(
        ngram_csv_paths=csv_paths,
        heatmap_ds_keys=heatmap_ds_keys,
        heatmap_root_dir=str(cfg.paths.model_level_root),  # where *_sW1.csv live
        output_dir=str(cfg.paths.plots_root),
        output_name="combined_3x2_ngrams_and_activation_heatmaps.pdf",
        n=2,
        level="char",
        text_source="object_1+2",
        xscale="linear",
        yscale="log",
        downsample=1,
    )

    print("[stage] N-gram and Avg. Activation Heatmap plot done")

    # 3) Per-model activation heatmaps
    for model_name in cfg.models_for_activation_plots:
        plot_single_model_activation_heatmaps(
            model_name=model_name,
            datasets=datasets_for_plots,
            root_dir=cfg.paths.model_level_root,
            output_fp=cfg.paths.model_level_root,
            noise=cfg.noise,
        )
    print("[stage] LLM-level activation heatmaps done")

    # 4) Decision-boundary heatmaps
    tasks = [0, 1, 2, 3, 4]
    pretty_labels = {
        0: "Original",
        1: "Synthetic",
        2: "Fictional",
        3: "Fictional (True)",
        4: "Noise",
    }

    plot_boundary_heatmaps(
        datasets=datasets_for_plots,
        tasks=tasks,
        pretty_labels=pretty_labels,
        probe=probe_name,
        probes_root=cfg.paths.probes_root,
        output_fp=cfg.paths.plots_root,
        noise=cfg.noise,
    )
    print("[stage] Decision-boundary heatmaps done")

    # 5) Flip bar charts (for probes) – one per datapack
    merged_root = cfg.paths.merged_root
    init_task = 0
    new_tasks = [1, 2, 3, 4]

    for dp_name in datasets_for_plots:
        plot_flip_barchart(
            dataset=dp_name,
            merged_root=merged_root,
            output_fp=cfg.paths.plots_root,
            init_task=init_task,
            new_tasks=new_tasks,
            probe_name=probe_name,
            noise=cfg.noise,
        )
        print(f"[stage] Flip bar chart done for {dp_name}")


    print("\nAll plots completed.")


if __name__ == "__main__":
    main()
