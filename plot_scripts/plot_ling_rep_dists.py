"""
plot_ling_rep_dists.py

Creates a plot with ngram distributions and activation heatmaps
to allow side-by-side analysis of linguistic and representational
properties.

"""

import os
import re
import glob
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.lines as mlines


COL_NAMES = ["correct", "real_object", "fake_object", "negation", "fictional_object"]
WORD_RE = re.compile(r"[A-Za-z0-9_]+")  # fallback tokenizer

PLOT_CATEGORIES = ["True", "False", "Synthetic", "Noise", "Fictional"]
MODEL_LEVEL_DIR = "outputs/plots/model_level"


def set_axis_fontsizes(ax, *, tick_fs, label_fs, x=True, y=True):
    """
    Set tick-label + axis-label font sizes consistently
    
    :param ax: axis
    :param tick_fs: tick font size
    :param label_fs: label font size
    :param x: x-axis
    :param y: y-axis
    :return: 
    """
    if x:
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.xaxis.label.set_size(label_fs)
    if y:
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.yaxis.label.set_size(label_fs)


def load_and_concat_csvs(csv_paths):
    """
    Load and combine datasets.
    
    :param csv_paths: paths to datasets
    :return: combined dataset
    """
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = standardize_columns(df)
        frames.append(df)
    if not frames:
        raise ValueError("No CSVs loaded — please provide at least one path.")
    return pd.concat(frames, axis=0, ignore_index=True)


def to_bool(x):
    """
    Convert a value to boolean.
    
    :param x: value to convert
    :return: boolean value
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


def standardize_columns(df):
    """
    Standardize columns across dataframes.
    
    :param df: dataframe
    :return: dataframe with standardized columns.
    """
    for c in COL_NAMES:
        if c not in df.columns:
            df[c] = 0
    for c in COL_NAMES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    return df


def get_label(row):
    """
    Get type of statement from a given row.
    
    :param row: row
    :return: type of statement
    """
    correct = to_bool(row.get("correct"))
    real_obj = to_bool(row.get("real_object"))
    fict_obj = to_bool(row.get("fictional_object"))

    if (correct is True) and (real_obj is True):
        return "true"
    if (correct is False) and (real_obj is True):
        return "false"
    if fict_obj is True:
        return "fictional"
    if (real_obj is False) and (fict_obj is False):
        return "synthetic"
    return None


def word_tokens(text):
    """
    Convert text to tokens (word-level).
    
    :param text: text to tokenize
    :return: tokenized text
    """
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def char_tokens(text):
    """
    Convert text to tokens (character-level).

    :param text: text to tokenize
    :return: tokenized text
    """
    return list(re.sub(r"[^A-Za-z0-9_]", "", str(text).lower()))


def ngrams(tokens, n):
    """
    Create ngrams.
    
    :param tokens: tokens
    :param n: length of n-gram
    :return: ngrams
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        for t in tokens:
            yield (t,)
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def flatten_ngrams(ngs):
    """
    Convert tuple to ngrams.
    
    :param ngs: tuple
    :return: flattened ngram
    """
    for tup in ngs:
        yield " ".join(tup)


def resolve_text_sources(df, text_source):
    """
    Identify text for ngram analysis.
    
    :param df: dataframe
    :param text_source: tag for column with text
    :return: list of columns with text
    """
    aliases_both = {"object_1+2", "objects_both", "both_objects", "objects"}
    if text_source in aliases_both:
        cols = [c for c in ("object_1", "object_2") if c in df.columns]
        if not cols:
            raise KeyError('Neither "object_1" nor "object_2" present in the dataframe.')
        return cols
    if text_source not in df.columns:
        raise KeyError(f'"{text_source}" not found in columns: {list(df.columns)}')
    return [text_source]


def compute_ngram_distributions(df, text_source="object_1+2", n=2, level="char"):
    """
    Compute the ngram distributions.
    
    :param df: dataframe
    :param text_source: which columns to grab text from
    :param n: ngram length
    :param level: word or character ngrams
    :return: ngram distributions by statement type
    """
    labels = df.apply(get_label, axis=1)
    clean_df = df.copy()
    clean_df["_label_"] = labels
    clean_df = clean_df[clean_df["_label_"].notna()]

    tokenizer = word_tokens if level == "word" else char_tokens
    base_labels = ["true", "false", "synthetic", "fictional"]
    by_label: Dict[str, Counter] = {lbl: Counter() for lbl in base_labels}

    sources = resolve_text_sources(clean_df, text_source)

    for _, row in clean_df.iterrows():
        lbl = str(row["_label_"])
        if lbl not in by_label:
            continue
        for src in sources:
            text = row.get(src, "")
            toks = tokenizer(text if isinstance(text, str) else "")
            ngs = flatten_ngrams(list(ngrams(toks, n)))
            by_label[lbl].update(ngs)

    return by_label


def make_full_frequency_table(by_label):
    """
    Generate ngram frequency table by statement type.
    
    :param by_label: ngram distributions
    :return: frequency table
    """
    vocab = set()
    for ctr in by_label.values():
        vocab.update(ctr.keys())

    rows = []
    for ng in vocab:
        row = {"ngram": ng}
        for lbl in by_label.keys():
            row[lbl] = by_label[lbl].get(ng, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("ngram")
    for lbl in by_label.keys():
        tot = sum(by_label[lbl].values()) or 1
        df[lbl] = df[lbl] / tot
    return df


def moving_average(y, window=101):
    """
    Smooth distributions with a moving average.
    
    :param y: data to smooth
    :param window: window for moving average
    :return: smoothed data
    """
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or window > len(y):
        return y
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def compute_sorted_ngram_rank_curves(csv_paths, text_source, n, level, sort_prefer="true"):
    """
    Get the statement-type ngram distribution curves.
    
    :param csv_paths: paths to statements
    :param text_source: which columns to grab text from
    :param n: length of ngrams
    :param level: word or character-level ngrams
    :param sort_prefer: which statement type to sort by
    :return: data for plot
    """
    df = load_and_concat_csvs(csv_paths)
    by_label = compute_ngram_distributions(df=df, text_source=text_source, n=n, level=level)
    full_df = make_full_frequency_table(by_label)

    sort_col = sort_prefer if sort_prefer in full_df.columns else list(full_df.columns)[0]
    df_sorted = full_df.sort_values(sort_col, ascending=False).reset_index(drop=False)
    ranks = np.arange(1, len(df_sorted) + 1)

    curves = {}
    for k in ["true", "false", "synthetic", "fictional"]:
        if k in df_sorted.columns:
            curves[k] = moving_average(df_sorted[k].values)
    return ranks, curves


def plot_ngrams_on_ax(ax, csv_paths, text_source, n, level, xscale, yscale, downsample, 
                      show_ylabel,  show_xlabel, color_map):
    """
    Plot the ngram distributions.
    
    :param ax: plotting axes
    :param csv_paths: paths to statements
    :param text_source: which columns to grab text from
    :param n: length of ngrams
    :param level: word or character-level ngrams
    :param xscale: x-axis scale (linear, log)
    :param yscale: y-axis scale (linear, log)
    :param downsample: how much to downsample
    :param show_ylabel: whether to add the y-label to the plot
    :param show_xlabel: whether to add the x-label to the plot
    :param color_map: color map
    :return: None
    """
    ranks, curves = compute_sorted_ngram_rank_curves(csv_paths, text_source, n, level)
    sl = slice(None, None, max(1, int(downsample)))

    draw_order = ["true", "false", "synthetic", "fictional"]
    for key in draw_order:
        if key not in curves:
            continue
        ax.plot(ranks[sl], curves[key][sl], label=key.capitalize(), color=color_map.get(key))

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if show_xlabel:
        ax.set_xlabel(f"{n}-gram Rank")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("log(Norm. Freq.)" if yscale == "log" else "Normalized Freq.")
    else:
        ax.set_ylabel("")
        
    set_axis_fontsizes(
        ax,
        tick_fs=6,
        label_fs=7,
        x=True,
        y=True,
    )

    ax.set_box_aspect(1.0)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_sW1_csv(path, categories=PLOT_CATEGORIES):
    """
    Get the LLM-level activation distances.

    :param path: path to model-level results
    :param categories: statement types to consider
    :return: averaged model-level results
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

    out = pd.DataFrame(np.nan, index=categories, columns=categories, dtype=float)
    r = min(df.shape[0], len(categories))
    c = min(df.shape[1], len(categories))
    out.iloc[:r, :c] = df.iloc[:r, :c].values
    return out


def collect_dataset_mats(root_dir, dataset):
    """
    Get the data for the activation heatmaps.

    :param root_dir: directory of LLM-level results
    :param dataset: dataset tag
    :return: data for combined heatmaps
    """
    pat = re.compile(rf"^.+_{re.escape(dataset)}_layer\d+_sW1\.csv$")
    mats = []
    for p in sorted(glob.glob(os.path.join(root_dir, "*_sW1.csv"))):
        fname = os.path.basename(p)
        if pat.match(fname):
            mats.append(read_sW1_csv(p, categories=PLOT_CATEGORIES))
    return mats


def compute_aggregated_heatmap_for_dataset(root_dir, ds_key):
    """
    Compute the combined activation heatmap data.

    :param root_dir: directory of LLM-level results
    :param ds_key: dataset tag
    :return: combined matrix
    """
    mats = collect_dataset_mats(root_dir, ds_key)
    if not mats:
        raise FileNotFoundError(f"No matrices found for dataset='{ds_key}' under {root_dir}")

    C_full = len(PLOT_CATEGORIES)
    stack = np.full((len(mats), C_full, C_full), np.nan, dtype=float)
    for k, M in enumerate(mats):
        M = M.reindex(index=PLOT_CATEGORIES, columns=PLOT_CATEGORIES).astype(float)
        stack[k] = M.values

    mean_mat = np.nanmean(stack, axis=0)
    np.fill_diagonal(mean_mat, 0.0)

    M_mean = pd.DataFrame(mean_mat, index=PLOT_CATEGORIES, columns=PLOT_CATEGORIES)
    return M_mean.loc[PLOT_CATEGORIES, PLOT_CATEGORIES]


def plot_heatmap_on_ax(ax, M_plot, vmax, cmap="YlGn_r", show_xlabel=False, show_ylabel=True):
    """
    Plot the activation heatmap.

    :param ax: plot axes
    :param M_plot: data for plot
    :param vmax: max color bar value
    :param cmap: color map
    :param show_xlabel: whether to show x-axis labels
    :param show_ylabel: whether to show y-axis labels
    :return: the plot
    """
    im = ax.imshow(M_plot.values, aspect="equal", cmap=cmap, vmin=0.0, vmax=vmax)

    C_plot = len(M_plot.index)
    ax.set_xticks(np.arange(C_plot))
    ax.set_yticks(np.arange(C_plot))

    if show_xlabel:
        ax.set_xticklabels(M_plot.columns, rotation=45, ha="right", fontsize=6)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")
            t.set_va("top")
        ax.tick_params(axis="x", pad=0)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_yticklabels(M_plot.index, fontsize=6)
        ax.tick_params(axis="y", pad=2)
    else:
        ax.set_yticklabels([])

    return im


def make_combined_figure(*, ngram_csv_paths, heatmap_ds_keys, heatmap_root_dir, output_dir,
                         output_name="combined_ngrams_and_heatmaps.pdf", n=2, level="char",
                         text_source="object_1+2", xscale="linear", yscale="log", downsample=1):
    """
    Plot combined ngram and activation heatmap figure.

    :param ngram_csv_paths: paths to datasets for ngram analysis
    :param heatmap_ds_keys: dataset tags for heatmaps
    :param heatmap_root_dir: path to LLM-level activation distances
    :param output_dir: output directory
    :param output_name: name of final figure
    :param n: length of ngrams
    :param level: word or character-level ngrams
    :param text_source: which columns to grab ngram text from
    :param xscale: x-axis scale (linear or log)
    :param yscale: y-axis scale (linear or log)
    :param downsample: how much to downsample by
    :return: None
    """
    # consistent label & order
    row_names = ["City\nLocations", "Medical\nIndications", "Word\nDefinitions"]
    for rn in row_names:
        if rn not in ngram_csv_paths or rn not in heatmap_ds_keys:
            raise KeyError(f"Missing config for row '{rn}' in ngram_csv_paths or heatmap_ds_keys")

    # line colors (keep your palette)
    color_map = {
        "true": "#2F5D3A",
        "false": "#C14A2A",
        "synthetic": "#C7922B",
        "fictional": "#455B73",
    }
    
    name_map = {
        "true": "True",
        "false": "False",
        "synthetic": "Synth.",
        "fictional": "Fic."
    }

    # precompute heatmaps + shared vmax
    heatmaps: Dict[str, pd.DataFrame] = {}
    vmax = 0.0
    for rn in row_names:
        ds_key = heatmap_ds_keys[rn]
        M_plot = compute_aggregated_heatmap_for_dataset(heatmap_root_dir, ds_key)
        heatmaps[rn] = M_plot
        local_max = np.nanmax(M_plot.values)
        if np.isfinite(local_max):
            vmax = max(vmax, float(local_max))
    if vmax <= 0.0:
        vmax = 1.0

    # layout
    fig = plt.figure(figsize=(3.0, 5.0))
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=7,
        ncols=4,
        height_ratios=[0.055, 0.055, 0.27, 0.27, 0.27, 0.04, 0.06],
        width_ratios=[0.16, 0.42, 0.2, 0.42],
        hspace=0.30,
        wspace=0.35,
    )

    # title row
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_axis_off()
    ax_title.text(
        -0.15, 0.5,
        "Linguistic vs. Representational Structure",
        ha="left", va="center",
        fontsize=8, fontweight="bold",
        color="#333333",
        transform=ax_title.transAxes,
    )

    # column headers
    ax_h0 = fig.add_subplot(gs[1, 0]); ax_h0.set_axis_off()
    ax_h1 = fig.add_subplot(gs[1, 1]); ax_h1.set_axis_off()
    ax_h2 = fig.add_subplot(gs[1, 3]); ax_h2.set_axis_off()

    ax_h1.text(
        -0.5, 1.0,
        f"Bigram distribution",
        ha="left", va="center",
        fontsize=7, fontweight="bold", color="#333333",
        transform=ax_h1.transAxes,
    )
    
    ax_h2.text(
        -0.5, 1.0,
        "Activation distance",
        ha="left", va="center",
        fontsize=7, fontweight="bold", color="#333333",
        transform=ax_h2.transAxes,
    )

    # dataset rows
    label_order = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    last_im = None

    for i, rn in enumerate(row_names):
        r = 2 + i  # grid row index for dataset

        # left text label axis
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(
            -0.9,
            0.5,
            f"{rn}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8,
            fontweight="bold",
            color="#444444",
            transform=ax_lbl.transAxes,
        )

        # ngram axis
        ax_ng = fig.add_subplot(gs[r, 1])
        plot_ngrams_on_ax(
            ax_ng,
            csv_paths=ngram_csv_paths[rn],
            text_source=text_source,
            n=n,
            level=level,
            xscale=xscale,
            yscale=yscale,
            downsample=downsample,
            show_ylabel=True,
            show_xlabel=(i == len(row_names) - 1),
            color_map=color_map,
        )
        
        ax_ng.text(
            -0.4, 1.3, f"{label_order[2*i]}",
            ha="left", va="top",
            fontsize=7, fontweight="bold", color="#444444",
            transform=ax_ng.transAxes,
        )

        # heatmap axis
        ax_hm = fig.add_subplot(gs[r, 3])
        last_im = plot_heatmap_on_ax(
            ax_hm,
            heatmaps[rn],
            vmax=vmax,
            cmap="YlGn_r",
            show_xlabel=(i == len(row_names) - 1),
            show_ylabel=True,
        )
        
        ax_hm.text(
            -0.4, 1.3, f"{label_order[2*i + 1]}",
            ha="left", va="top",
            fontsize=7, fontweight="bold", color="#444444",
            transform=ax_hm.transAxes,
        )

    # gs[5, :] spacer row

    # legends row: nested sub-gridspec so legend and colorbar blocks match height
    gs_leg = gs[6, :].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[0.16, 0.44, 0.40],
        wspace=0.30,
    )
    
    # blank cell under the dataset label column
    ax_leg_blank = fig.add_subplot(gs_leg[0, 0])
    ax_leg_blank.set_axis_off()
    
    # left: line legend
    ax_leg_lines = fig.add_subplot(gs_leg[0, 1])
    ax_leg_lines.set_axis_off()
    
    handles = [
        mlines.Line2D([], [], color=color_map[k], label=name_map[k])
        for k in ["true", "false", "synthetic", "fictional"]
    ]
    ax_leg_lines.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.3, 1.25),
        ncol=2,
        frameon=False,
        fontsize=6,
        handlelength=2.0,
        columnspacing=1.0,
        labelspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    
    # right: colorbar split into bar + label area so total height matches ngram legend
    gs_cbar = gs_leg[0, 2].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.3, 0.7],
        hspace=0.0,
    )
    
    ax_cbar = fig.add_subplot(gs_cbar[0, 0])
    ax_cbar_txt = fig.add_subplot(gs_cbar[1, 0])
    ax_cbar_txt.set_axis_off()
    
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=ax_cbar, orientation="horizontal")
        cbar.ax.tick_params(labelsize=5, pad=1)
        cbar.set_label("")
    
        # label centered beneath, doesn't consume bar height
        ax_cbar_txt.text(
            0.5,
            0.0,
            "Avg. Wasserstein Dist.",
            ha="center",
            va="top",
            fontsize=6,
            # color="#333333",
            transform=ax_cbar_txt.transAxes,
        )
    else:
        ax_cbar.set_axis_off()
        ax_cbar_txt.set_axis_off()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
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

    # dataset keys for the heatmap CSV filenames
    heatmap_ds_keys = {
        "City\nLocations": "cities_loc",
        "Medical\nIndications": "med_indications",
        "Word\nDefinitions": "defs",
    }

    make_combined_figure(
        ngram_csv_paths=csv_paths,
        heatmap_ds_keys=heatmap_ds_keys,
        heatmap_root_dir=MODEL_LEVEL_DIR,  # where LLM-level results live
        output_dir="/projects/radlab/Sam/belief/outputs/paper_plots",
        output_name="combined_3x2_ngrams_and_activation_heatmaps.pdf",
        n=2,
        level="char",
        text_source="object_1+2",
        xscale="linear",
        yscale="log",
        downsample=1,
    )


if __name__ == "__main__":
    main()
