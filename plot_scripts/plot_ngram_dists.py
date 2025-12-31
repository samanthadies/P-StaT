"""
plot_ngram_dists.py

Analyze and visualize character- or word-level n-gram distributions
for different labels (true, false, synthetic, fictional) across
one or more CSV datasets.

"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec as grid_spec
import matplotlib.lines as mlines


COL_NAMES = ['correct', 'real_object', 'fake_object', 'negation', 'fictional_object']
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def load_and_concat_csvs(csv_paths):
    """
    Load and vertically concatenate multiple CSV files.

    All loaded DataFrames are standardized to share required label columns.

    :param csv_paths: list of file paths to CSVs
    :return: concatenated pandas DataFrame
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
    Convert a heterogeneous value into a boolean or None.

    Handles common string encodings (e.g., 'true', 'false', '1', '0')
    and numpy NaNs.

    :param x: value to convert
    :return: True, False, or None if conversion fails
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n'}:
        return False
    try:
        return bool(int(s))
    except Exception:
        return None


def ngrams(tokens, n):
    """
    Yield n-grams from a sequence of tokens.

    :param tokens: list of tokens (strings)
    :param n: n-gram length (must be >= 1)
    :return: generator of tuples representing n-grams
    """
    if n <= 0:
        raise ValueError('n must be >= 1')
    if n == 1:
        for t in tokens:
            yield (t,)
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


def standardize_columns(df):
    """
    Ensure that label columns exist and are integer-coded.

    Missing boolean-ish columns in COL_NAMES are added and filled with zero.
    Existing ones are coerced to int32.

    :param df: input pandas DataFrame
    :return: DataFrame with standardized label columns
    """
    # add any missing boolean-ish label columns as zeros
    for c in COL_NAMES:
        if c not in df.columns:
            df[c] = 0

    for c in COL_NAMES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")

    return df


def get_label(row):
    """
    Map a row's label columns into a high-level veracity label.

    The mapping is:
        - 'true'      : correct=True and real_object=True
        - 'false'     : correct=False and real_object=True
        - 'fictional' : fictional_object=True
        - 'synthetic' : real_object=False and fictional_object=False
        - None        : all other cases

    :param row: pandas Series representing a single row
    :return: one of {'true', 'false', 'fictional', 'synthetic', None}
    """
    correct = to_bool(row.get('correct'))
    real_obj = to_bool(row.get('real_object'))
    fict_obj = to_bool(row.get('fictional_object'))

    # True/False only defined when real_object == True
    if (correct is True) and (real_obj is True):
        return 'true'
    if (correct is False) and (real_obj is True):
        return 'false'

    # Fictional
    if fict_obj is True:
        return 'fictional'

    # Synthetic
    if (real_obj is False) and (fict_obj is False):
        return 'synthetic'

    return None


def word_tokens(text):
    """
    Tokenize text into alphanumeric "word" tokens (lowercased).

    :param text: input string
    :return: list of token strings
    """
    return [m.group(0).lower() for m in WORD_RE.finditer(text or '')]


def char_tokens(text):
    """
    Tokenize text into a sequence of characters (lowercased alphanumerics + underscore).

    :param text: input string
    :return: list of single-character strings
    """
    return list(re.sub(r'[^A-Za-z0-9_]', '', str(text).lower()))


def flatten_ngrams(ngs):
    """
    Convert a sequence of n-gram tuples into space-joined strings.

    :param ngs: iterable of tuples of tokens
    :return: generator of 'token1 token2 ...' strings
    """
    for tup in ngs:
        yield ' '.join(tup)


def resolve_text_sources(df, text_source):
    """
    Resolve one or more columns to use as text sources.

    Supports aliases for "object_1+2" (e.g., 'objects', 'object_1+2')
    which map to any combination of 'object_1' and 'object_2' present.

    :param df: pandas DataFrame with text columns
    :param text_source: column name or alias
    :return: list of column names to read text from
    """
    aliases_both = {'object_1+2', 'objects_both', 'both_objects', 'objects'}
    if text_source in aliases_both:
        cols = [c for c in ('object_1', 'object_2') if c in df.columns]
        if not cols:
            raise KeyError('Neither "object_1" nor "object_2" present in the dataframe.')
        return cols

    if text_source not in df.columns:
        raise KeyError(f'"{text_source}" not found in columns: {list(df.columns)}')

    return [text_source]


def compute_ngram_distributions(df, text_source='object_1+2', n=2, level='char'):
    """
    Compute per-label n-gram frequency counts from a DataFrame.

    Each row is assigned a high-level label (true/false/synthetic/fictional),
    and n-grams are extracted from one or more text columns.

    :param df: input pandas DataFrame with label and text columns
    :param text_source: text column name or alias (e.g., 'object_1+2')
    :param n: n-gram length
    :param level: 'char' for character-level or 'word' for token-level
    :return: dict mapping label -> Counter of n-gram string -> count
    """
    # clean data
    labels = df.apply(get_label, axis=1)
    clean_df = df.copy()
    clean_df['_label_'] = labels
    clean_df = clean_df[clean_df['_label_'].notna()]

    tokenizer = word_tokens if level == 'word' else char_tokens
    base_labels = ['true', 'false', 'synthetic', 'fictional']
    by_label = {lbl: Counter() for lbl in base_labels}

    sources = resolve_text_sources(clean_df, text_source)

    # get ngrams
    for _, row in clean_df.iterrows():
        lbl = str(row['_label_'])
        if lbl not in by_label:
            continue
        for src in sources:
            text = row.get(src, '')
            toks = tokenizer(text if isinstance(text, str) else '')
            ngs = flatten_ngrams(list(ngrams(toks, n)))
            by_label[lbl].update(ngs)

    return by_label


def make_full_frequency_table(by_label):
    """
    Build a full n-gram frequency table, normalized per label.

    :param by_label: dict mapping label -> Counter of n-gram -> count
    :return: pandas DataFrame indexed by n-gram with columns per label
             containing normalized frequencies
    """
    vocab = set()
    for ctr in by_label.values():
        vocab.update(ctr.keys())

    rows = []
    for ngram in vocab:
        row = {'ngram': ngram}
        for lbl in by_label.keys():
            row[lbl] = by_label[lbl].get(ngram, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('ngram')

    for lbl in by_label.keys():
        tot = sum(by_label[lbl].values()) or 1
        df[lbl] = df[lbl] / tot

    return df


def counters_to_dataframe(by_label, top_k=40):
    """
    Construct a compact normalized frequency table for the top-k n-grams.

    The union of the top_k most frequent n-grams per label is taken,
    then per-label frequencies are normalized.

    :param by_label: dict mapping label -> Counter of n-gram -> count
    :param top_k: number of most frequent n-grams per label to keep
    :return: pandas DataFrame indexed by n-gram with normalized frequencies
    """
    vocab = set()
    for ctr in by_label.values():
        vocab.update([k for k, _ in ctr.most_common(top_k)])

    rows = []
    for ngram in vocab:
        row = {'ngram': ngram}
        for lbl in by_label.keys():
            row[lbl] = by_label[lbl].get(ngram, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('ngram')

    # normalize per label
    for lbl in by_label.keys():
        tot = sum(by_label[lbl].values()) or 1
        df[lbl] = df[lbl] / tot

    df['max_freq'] = df.max(axis=1)
    df = df.sort_values('max_freq', ascending=False).drop(columns=['max_freq'])

    return df


def get_ngram_dfs(csv_paths, text_source, n, level):
    """
    Compute both compact and full n-gram frequency tables from CSVs.

    :param csv_paths: list of CSV paths to load and concatenate
    :param text_source: text column name or alias
    :param n: n-gram length
    :param level: 'char' or 'word'
    :return: (freq_df, full_df) where
             - freq_df is a compact top-k table
             - full_df is the full normalized frequency table
    """
    df = load_and_concat_csvs(csv_paths)
    by_label = compute_ngram_distributions(df=df, text_source=text_source, n=n, level=level)
    full_df = make_full_frequency_table(by_label)
    freq_df = counters_to_dataframe(by_label)

    return freq_df, full_df


def moving_average(y, window=101):
    """
    Smooth a 1D signal with a centered moving average.

    :param y: 1D array-like of values
    :param window: window size (odd integer recommended); if None or <=1,
                   the input is returned unchanged
    :return: numpy array of smoothed values
    """
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or window > len(y):
        return y
    kernel = np.ones(int(window), dtype=float) / float(window)

    return np.convolve(y, kernel, mode="same")


def plot_ngrams(
    csv_paths,
    text_source='object_1+2',
    n=2,
    level='char',
    output_fp='outputs',
    xscale='linear',
    yscale='log',
    downsample=1,
):
    """
    Generate a multi-panel plot of smoothed n-gram frequency distributions.

    Each panel corresponds to a dataset, and each curve corresponds to
    a veracity label (true/false/synthetic/fictional).

    :param csv_paths: dict mapping dataset name -> list of CSV paths
    :param text_source: text column name or alias
    :param n: n-gram length
    :param level: 'char' or 'word'
    :param output_fp: directory where the PDF plot will be saved
    :param xscale: x-axis scale ('linear' or 'log')
    :param yscale: y-axis scale ('linear' or 'log')
    :param downsample: integer step for rank downsampling (>=1)
    :return: None
    """
    draw_order = ['true', 'false', 'synthetic', 'fictional']
    color_map = {
        'true': '#2F5D3A',
        'false': '#C14A2A',
        'synthetic': '#C7922B',
        'fictional': '#455B73',
    }

    label_order = ['(a)', '(b)', '(c)']

    fig = plt.figure(figsize=(7.2, 2.9), constrained_layout=True)
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=3,
        ncols=3,
        height_ratios=[0.08, 0.82, 0.10],
    )
    ax_title = fig.add_subplot(gs[0, :])
    ax_legend = fig.add_subplot(gs[2, :])

    ds_num = 0
    for ds_name, paths in csv_paths.items():
        freq_df, full_df = get_ngram_dfs(paths, text_source, n, level)

        # pick a primary sort column (prefer 'true', else the first available)
        if 'true' in full_df.columns:
            sort_col = 'true'
        else:
            sort_col = list(full_df.columns)[0]

        df_sorted = full_df.sort_values(sort_col, ascending=False).reset_index(drop=False)
        ranks = np.arange(1, len(df_sorted) + 1)
        sl = slice(None, None, max(1, int(downsample)))

        ax_subplot = fig.add_subplot(gs[1, ds_num])
        subplot_text = f'{label_order[ds_num]} {ds_name}'

        for key in draw_order:
            if key not in df_sorted.columns:
                continue
            y = df_sorted[key].values
            y_s = moving_average(y)

            ax_subplot.plot(
                ranks[sl],
                y_s[sl],
                label=key.capitalize(),
                color=color_map.get(key),
            )

        ax_subplot.set_xlabel(f'{n}-gram Rank')

        if ds_num == 0:
            ax_subplot.set_ylabel('log(Normalized Freq.)')
        else:
            ax_subplot.set_ylabel('')

        ax_subplot.set_xscale(xscale)
        ax_subplot.set_yscale(yscale)

        ax_subplot.spines['top'].set_visible(False)
        ax_subplot.spines['right'].set_visible(False)

        ax_subplot.text(
            -0.35,
            1.2,
            subplot_text,
            transform=ax_subplot.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='#444444',
        )

        ds_num += 1

    ax_title.set_axis_off()
    title = f'Distribution of {n}-grams (Smoothed)'
    ax_title.text(
        -0.1,
        0.25,
        title,
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    ax_legend.set_axis_off()
    handles = [
        mlines.Line2D([], [], color=color_map[k], label=k.capitalize())
        for k in draw_order
    ]
    ax_legend.legend(
        handles=handles,
        loc='center',
        ncol=len(draw_order),
        frameon=False,
        fontsize=8,
    )

    os.makedirs(output_fp, exist_ok=True)
    fp = f'{output_fp}/ngram_dists.pdf'
    fig.savefig(fp, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():

    csv_paths = {
        'City Locations': [
            'datasets/cities_loc_true_false.csv',
            'datasets/cities_loc_synthetic.csv',
            'datasets/cities_loc_fictional.csv',
        ],
        'Medical Indications': [
            'datasets/med_indications_true_false.csv',
            'datasets/med_indications_synthetic.csv',
            'datasets/med_indications_fictional.csv',
        ],
        'Word Definitions': [
            'datasets/defs_true_false.csv',
            'datasets/defs_synthetic.csv',
            'datasets/defs_fictional.csv',
        ],
    }

    plot_ngrams(
        csv_paths=csv_paths,
        text_source='object_1+2',
        n=2,
        level='char',
        output_fp='outputs/plots',
    )


if __name__ == '__main__':
    main()
