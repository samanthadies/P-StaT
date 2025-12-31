"""
plot_boundary_heatmaps.py

Load linear probe weights across tasks and plot heatmaps of
decision boundary changes (cosine similarity and bias difference)
for multiple datasets and models.

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec


def find_task_dir(probes_root, probe, model, dataset, task=0, noise=10):
    """
    Locate the directory for a specific (probe, model, dataset, task, noise).

    First checks a canonical search directory; if not found, falls back to
    the most recent matching directory by modification time.

    :param probes_root: root directory containing all probe outputs
    :param probe: probe name (e.g., "sAwMIL", "mean_diff")
    :param model: model name subdirectory under the probe root
    :param dataset: dataset name (e.g., "cities_loc")
    :param task: integer task id
    :param noise: noise level used in the experiment naming
    :return: path to the task directory or None if not found
    """
    base = os.path.join(probes_root, probe, model)
    pref = os.path.join(base, f"{dataset}_search_noise{noise}_task-{task}")
    if os.path.isdir(pref):
        return pref
    cands = sorted(
        glob.glob(os.path.join(base, f"{dataset}_task-{task}*")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return cands[0] if cands else None


def discover_single_layer_by_probe(task_dir, probe_name):
    """
    Infer the single layer index used for a given probe in a task directory.

    This looks for a single matching coefficient/bias file pattern depending
    on the probe name and extracts the layer id from the filename.

    :param task_dir: directory containing saved probe files
    :param probe_name: probe name ("sAwMIL" or "mean_diff")
    :return: integer layer index or None if it cannot be resolved
    """
    if probe_name == "sAwMIL":
        patt = "NON_NORMALIZED_coef_*.npy"
        prefix = "NON_NORMALIZED_coef_"
    elif probe_name == "mean_diff":
        patt = "bias_*.npy"
        prefix = "bias_"
    else:
        return None

    files = glob.glob(os.path.join(task_dir, patt))
    if len(files) == 1:
        name = os.path.basename(files[0])
        try:
            return int(name.replace(prefix, "").replace(".npy", ""))
        except Exception:
            return None
    return None


def resolve_layer_for_model(model, task_dir, probe_name="sAwMIL"):
    """
    Resolve which layer index to use for a (model, task_dir, probe_name).

    Uses discover_single_layer_by_probe and raises an error if no unique
    layer can be determined.

    :param model: model name string
    :param task_dir: directory containing probe outputs for this task
    :param probe_name: probe name ("sAwMIL" or "mean_diff")
    :return: integer layer index
    """
    d = discover_single_layer_by_probe(task_dir, probe_name)
    if d is not None:
        return d
    avail = sorted(glob.glob(os.path.join(task_dir, "*_*_*.npy")))
    raise RuntimeError(
        f"Cannot resolve layer for {model} in {task_dir}. "
        f"Found: {avail or 'NONE'}"
    )


def load_wb_for_task(probes_root, probe_name, model, dataset, task, noise):
    """
    Load the (w, b, layer) tuple for a given probe/model/dataset/task.

    Handles both sAwMIL and mean_diff naming conventions and returns
    non-normalized weights and bias when available.

    :param probes_root: root directory containing all probe outputs
    :param probe_name: probe name ("sAwMIL" or "mean_diff")
    :param model: model name
    :param dataset: dataset name
    :param task: task id (int)
    :param noise: noise level used in naming
    :return: (w, b, layer) or None if loading fails
    """
    task_dir = find_task_dir(probes_root, probe_name, model, dataset, task, noise)
    if not task_dir:
        print(f"[warn] No task-{task} dir for {model} ({dataset}).")
        return None

    try:
        layer = resolve_layer_for_model(model, task_dir, probe_name)

        if probe_name == "sAwMIL":
            coef_fp = os.path.join(task_dir, f"NON_NORMALIZED_coef_{layer}.npy")
            bias_fp = os.path.join(task_dir, f"NON_NORMALIZED_bias_{layer}.npy")
            if not (os.path.isfile(coef_fp) and os.path.isfile(bias_fp)):
                print(f"[warn] Missing sbMIL2 files at layer {layer}")
                return None
            w = np.load(coef_fp).ravel()
            b = float(np.load(bias_fp).reshape(-1)[0])
            return (w.astype(float, copy=False), b, layer)

        elif probe_name == "mean_diff":
            coef_fp = os.path.join(task_dir, f"NON_NORMALIZED_coef_{layer}.npy")
            bias_fp = os.path.join(task_dir, f"bias_{layer}.npy")
            if not (os.path.isfile(coef_fp) and os.path.isfile(bias_fp)):
                print(f"[warn] Missing mean_diff files at layer {layer}")
                return None
            w = np.load(coef_fp).ravel()
            b = float(np.load(bias_fp).reshape(-1)[0])
            return (w.astype(float, copy=False), b, layer)

        else:
            print(f"[warn] Unknown probe_name: {probe_name}")
            return None

    except Exception as e:
        print(f"[warn] Could not load coef/bias for {model} task-{task}: {e}")
        return None


def discover_models_from_probes(probes_root, probe_name, dataset):
    """
    Discover all models under a probe root that have results for a dataset.

    Scans the probe root for model subdirectories that contain a task
    directory for the given dataset.

    :param probes_root: root directory containing all probe outputs
    :param probe_name: probe name subdirectory under probes_root
    :param dataset: dataset name
    :return: sorted list of model names
    """
    probe_root = os.path.join(probes_root, probe_name)

    candidates = []
    for entry in os.listdir(probe_root):
        model_dir = os.path.join(probe_root, entry)
        if not os.path.isdir(model_dir):
            continue

        if find_task_dir(probes_root, probe_name, entry, dataset):
            candidates.append(entry)

    models = sorted(candidates)
    return models


def cosine(u, v):
    """
    Compute cosine similarity between two vectors.

    Returns NaN if either vector has zero norm or non-finite norm.

    :param u: first 1D numpy array
    :param v: second 1D numpy array
    :return: cosine similarity as float
    """
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0 or not np.isfinite(nu) or not np.isfinite(nv):
        return np.nan
    return float(np.dot(u, v) / (nu * nv))


def build_cosine_and_bias_mats(models, tasks, dataset, probes_root, probe_name='sbMIL2', noise=10):
    """
    Build cosine similarity and bias-difference matrices across tasks and models.

    For each model, loads weights and biases for the first task and all
    subsequent tasks, and computes:
        - cosine similarity between w_0 and w_t
        - bias difference b_0 - b_t

    :param models: list of model names
    :param tasks: list of task ids (ints)
    :param dataset: dataset name
    :param probes_root: root directory of probe outputs
    :param probe_name: probe name (e.g., "sAwMIL", "mean_diff")
    :param noise: noise level used in naming
    :return: (cos_mat, bias_mat, task_list, model_list)
             cos_mat: (n_tasks, n_models) cosine similarities
             bias_mat: (n_tasks, n_models) bias differences
             task_list: list of unique tasks in order
             model_list: list of models retained (with valid task 0)
    """
    task_list = list(dict.fromkeys(tasks))
    model_list = []

    cos_mat = np.full((len(task_list), len(models)), np.nan, dtype=float)
    bias_mat = np.full((len(task_list), len(models)), np.nan, dtype=float)

    for m_idx, m in enumerate(models):
        wb0 = load_wb_for_task(probes_root, probe_name, m, dataset, task_list[0], noise)
        if wb0 is None:
            continue
        w0, b0, _ = wb0

        model_list.append(m)

        cos_mat[0, m_idx] = 1.0
        bias_mat[0, m_idx] = 0.0

        for ti, t in enumerate(task_list[1:], start=1):
            wb = load_wb_for_task(probes_root, probe_name, m, dataset, t, noise)
            if wb is None:
                continue
            wt, bt, _ = wb

            cos_mat[ti, m_idx] = cosine(w0, wt)
            delta_b = b0 - bt
            bias_mat[ti, m_idx] = float(delta_b)

    keep_cols = [i for i, m in enumerate(models) if m in model_list]
    cos_mat = cos_mat[:, keep_cols]
    bias_mat = bias_mat[:, keep_cols]
    model_list = [models[i] for i in keep_cols]

    return cos_mat, bias_mat, task_list, model_list


def plot_boundary_heatmaps(datasets, tasks, pretty_labels, probe, probes_root, output_fp, noise):
    """
    Plot cosine-similarity and bias-difference heatmaps across tasks and models.

    For each dataset, this function:
        1. Discovers models with probe results.
        2. Builds cosine and bias matrices for the specified tasks.
        3. Plots two side-by-side heatmaps per dataset:
           - Cosine similarity of weight vectors vs. the original task.
           - Difference in bias terms vs. the original task.

    :param datasets: list of dataset names
    :param tasks: list of task ids (ints)
    :param pretty_labels: dict mapping task id -> human-readable label
    :param probe: probe name (e.g., "sAwMIL", "mean_diff")
    :param probes_root: root directory of probe outputs
    :param output_fp: directory to save the resulting PDF
    :param noise: noise level used in directory naming
    :return: None
    """
    label_order = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=True)
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=5,
        ncols=2,
        height_ratios=[0.08, 0.30, 0.30, 0.30, 0.02],
    )
    ax_title = fig.add_subplot(gs[0, :])

    models = discover_models_from_probes(
        probes_root=probes_root,
        probe_name=probe,
        dataset='cities_loc',
    )

    last_im_cosine = None
    last_im_bias = None

    ds_num = 0
    for ds_name in datasets:
        cos_mat, bias_mat, task_list, model_list = build_cosine_and_bias_mats(
            models=models,
            tasks=tasks,
            dataset=ds_name,
            probes_root=probes_root,
            probe_name=probe,
            noise=noise,
        )

        ylabels = [pretty_labels[t] for t in task_list]
        xlabels = model_list

        if ds_name == 'cities_loc':
            pretty_name = 'City Locations'
        elif ds_name == 'med_indications':
            pretty_name = 'Med Indications'
        elif ds_name == 'defs':
            pretty_name = 'Word Definitions'
        else:
            pretty_name = ds_name

        ax_subplot_cosine = fig.add_subplot(gs[ds_num + 1, 0])
        ax_subplot_bias = fig.add_subplot(gs[ds_num + 1, 1])

        last_im_cosine = ax_subplot_cosine.imshow(
            cos_mat,
            aspect="equal",
            cmap="YlGn_r",
            vmin=0.0,
            vmax=1.0,
        )
        last_im_bias = ax_subplot_bias.imshow(
            bias_mat,
            aspect="equal",
            cmap="PiYG",
            vmin=-2.0,
            vmax=2.0,
        )

        ax_subplot_cosine.set_xticks(np.arange(len(xlabels)))
        ax_subplot_cosine.set_yticks(np.arange(len(ylabels)))
        plt.setp(ax_subplot_cosine.get_xticklabels(), rotation=45, ha='right')
        ax_subplot_cosine.set_xticklabels(xlabels)
        ax_subplot_cosine.set_yticklabels(ylabels)

        ax_subplot_bias.set_xticks(np.arange(len(xlabels)))
        ax_subplot_bias.set_yticks(np.arange(len(ylabels)))
        plt.setp(ax_subplot_bias.get_xticklabels(), rotation=45, ha='right')
        ax_subplot_bias.set_xticklabels(xlabels)
        ax_subplot_bias.set_yticklabels(ylabels)

        ax_subplot_cosine.tick_params(axis='both', labelsize=7)
        ax_subplot_bias.tick_params(axis='both', labelsize=7)

        if ds_num == 0:
            subplot_text = 'Cosine Similarity'
            ax_subplot_cosine.text(
                0.25,
                1.4,
                subplot_text,
                transform=ax_subplot_cosine.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                fontweight='bold',
                color='#444444',
            )
            subplot_text = 'Bias Difference'
            ax_subplot_bias.text(
                0.25,
                1.4,
                subplot_text,
                transform=ax_subplot_bias.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                fontweight='bold',
                color='#444444',
            )

        subplot_cosine_text = f'{label_order[2 * ds_num]}'
        subplot_bias_text = f'{label_order[2 * ds_num + 1]}'
        ax_subplot_cosine.text(
            -0.25,
            1.3,
            subplot_cosine_text,
            transform=ax_subplot_cosine.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            fontweight='bold',
            color='#555555',
        )
        ax_subplot_bias.text(
            -0.25,
            1.3,
            subplot_bias_text,
            transform=ax_subplot_bias.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            fontweight='bold',
            color='#555555',
        )

        subplot_text = f'{pretty_name}'
        ax_subplot_cosine.text(
            -0.40,
            1.0,
            subplot_text,
            transform=ax_subplot_cosine.transAxes,
            rotation=90,
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='#444444',
        )

        ds_num += 1

    ax_title.set_axis_off()

    if probe == 'sAwMIL':
        probe_pretty = 'sAwMIL'
    elif probe == 'mean_diff':
        probe_pretty = 'Mean Difference'
    else:
        probe_pretty = probe

    title = f'Change in {probe_pretty} Decision Boundary under Perturbations'
    ax_title.text(
        -0.18,
        0.4,
        title,
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    if last_im_cosine is not None:
        cax = fig.add_subplot(gs[4, 0])
        cbar = fig.colorbar(last_im_cosine, cax=cax, orientation="horizontal")
        cbar_label = r"$\cos\!\left(\vec{w}_{\text{Original}},\, \vec{w}_{\text{Perturbed}}\right)$"
        cbar.set_label(cbar_label, rotation=0, va="center", labelpad=10)

    if last_im_bias is not None:
        cax = fig.add_subplot(gs[4, 1])
        cbar = fig.colorbar(last_im_bias, cax=cax, orientation="horizontal")
        cbar_label = r"$b_{\mathrm{Original}} - b_{\mathrm{Perturbed}}$"
        cbar.set_label(cbar_label, rotation=0, va="center", labelpad=10)

    os.makedirs(output_fp, exist_ok=True)
    fp = f'{output_fp}/boundary_heatmaps_{probe}_noise{noise}.pdf'
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():

    datasets = ['cities_loc', 'med_indications', 'defs']
    tasks = [0, 1, 2, 3, 4]
    pretty_labels = {
        0: 'Original',
        1: 'Synthetic',
        2: 'Fictional',
        3: 'Fictional (True)',
        4: 'Noise',
    }
    probe = 'sAwMIL'
    noise = 10
    probe_root = 'outputs/probes'
    output_fp = 'outputs/plots'

    plot_boundary_heatmaps(
        datasets,
        tasks,
        pretty_labels,
        probe,
        probe_root,
        output_fp,
        noise,
    )


if __name__ == '__main__':
    main()
