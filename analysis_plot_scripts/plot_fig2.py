"""
plot_fig2.py

Create Figure 2: familiarity, surface-form structure, and activation heterogeneity.

The script:
  1. Computes bigram rank-frequency curves from dataset CSVs.
  2. Loads token-level familiarity caches from outputs/perplexity.
  3. Loads within-group activation-distance summaries.
  4. Builds the combined Figure 2 panel.
  5. Writes the figure to outputs/plots.
"""

import argparse
import logging
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS = ["cities_loc", "med_indications", "defs"]

DATASET_LABELS = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}

DATASET_SHORT_LABELS = {
    "cities_loc": "City Locations",
    "med_indications": "Med. Indications",
    "defs": "Word Definitions",
}

DATASET_MARKERS = {
    "cities_loc": "o",
    "med_indications": "s",
    "defs": "^",
}

DATASET_OFFSETS = {
    "cities_loc": 0.22,
    "med_indications": 0.00,
    "defs": -0.22,
}

MODELS = [
    "_gemma-2-9b",
    "_gemma-3-27b",
    "_gemma-7b",
    "_llama-3-8b-med",
    "_llama-3.1-8b",
    "_llama-3.1-8b-bio",
    "_llama-3.2-3b",
    "_llama-3.3-70b",
    "_mistral-7B-v0.3",
    "_qwen-2.5-14b",
    "_qwen-2.5-72b",
    "_qwen-2.5-7b",
    "gemma-2-9b",
    "gemma-3-27b",
    "gemma-7b",
    "llama-3-8b",
    "llama-3.2-3b",
    "mistral-7B-v0.3",
    "qwen-2.5-14b",
    "qwen-2.5-72b",
    "qwen-2.5-7b",
]

BIGRAM_TYPES = [
    "true",
    "false",
    "synthetic_tf",
    "synthetic_fi",
    "fictional",
]

BIGRAM_LABELS = {
    "true": "True",
    "false": "False",
    "synthetic_tf": "Synthetic (TF)",
    "synthetic_fi": "Synthetic (Fi)",
    "fictional": "Fictional",
}

COLORS = {
    "true": "#7980ff",
    "false": "#008c00",
    "fictional_true": "#b8ecff",
    "fictional": "#7d8c9d",
    "synthetic_tf": "#d8b371",
    "synthetic_fi": "#ebce9b",
}

FAMILIARITY_TYPES = [
    "true",
    "false",
    "fictional_true",
    "fictional",
    "synthetic_tf",
    "synthetic_fi",
]

FAMILIARITY_LABELS = {
    "true": "True",
    "false": "False",
    "fictional_true": "Fictional (T)",
    "fictional": "Fictional",
    "synthetic_tf": "Synthetic (TF)",
    "synthetic_fi": "Synthetic (Fi)",
}

CACHE_STATEMENT_TYPES = [
    "true_false",
    "synthetic",
    "synthetic_fic",
    "fictional",
]

HETERO_TYPES = [
    "Synthetic (TF)",
    "Synthetic (Fi)",
    "Fictional",
]

HETERO_COLORS = {
    "Synthetic (TF)": COLORS["synthetic_tf"],
    "Synthetic (Fi)": COLORS["synthetic_fi"],
    "Fictional": COLORS["fictional"],
}

DISTANCE_COLUMNS = {
    "Synthetic (TF)": ["within_synthetic", "within_synthetic_tf"],
    "Synthetic (Fi)": [
        "within_synthetic_fic",
        "within_synthetic_fictional",
        "within_synthetic_fi",
    ],
    "Fictional": ["within_fictional"],
}

BOOLEAN_COLS = [
    "correct",
    "real_object",
    "fake_object",
    "negation",
    "fictional_object",
    "synthetic_fic_object",
]


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def style_axis(ax) -> None:
    """Apply shared axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)


def to_bool(x):
    """Convert common boolean-like values to bool."""
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


def dataset_csv_paths(dataset: str, dataset_root: Path) -> list[Path]:
    """Return dataset CSV paths needed for bigram distributions."""
    return [
        dataset_root / f"{dataset}_true_false.csv",
        dataset_root / f"{dataset}_fictional.csv",
        dataset_root / f"{dataset}_synthetic.csv",
        dataset_root / f"{dataset}_synthetic_fic.csv",
    ]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing boolean metadata columns and coerce them to integers."""
    df = df.copy()

    for col in BOOLEAN_COLS:
        if col not in df.columns:
            df[col] = 0

    for col in BOOLEAN_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")

    return df


def load_and_concat_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate dataset CSV files."""
    frames = []

    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset CSV: {path}")

        df = pd.read_csv(path)
        df = standardize_columns(df)
        df["_source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)


def get_bigram_statement_type(row: pd.Series):
    """Assign statement type for bigram analysis."""
    correct = to_bool(row.get("correct"))
    real_obj = to_bool(row.get("real_object"))
    fict_obj = to_bool(row.get("fictional_object"))
    synth_fic_obj = to_bool(row.get("synthetic_fic_object"))
    source_file = str(row.get("_source_file", "")).lower()

    if correct is True and real_obj is True and not fict_obj and not synth_fic_obj:
        return "true"

    if correct is False and real_obj is True and not fict_obj and not synth_fic_obj:
        return "false"

    if fict_obj is True:
        return "fictional"

    if synth_fic_obj is True or "synthetic_fic" in source_file:
        return "synthetic_fi"

    if real_obj is False and fict_obj is False and synth_fic_obj is False:
        return "synthetic_tf"

    return None


def resolve_text_sources(df: pd.DataFrame, text_source: str) -> list[str]:
    """Resolve text columns used to compute character bigrams."""
    if text_source in {"object_1+2", "objects_both", "both_objects", "objects"}:
        cols = [col for col in ["object_1", "object_2"] if col in df.columns]

        if not cols:
            raise KeyError('Neither "object_1" nor "object_2" is present.')

        return cols

    if text_source not in df.columns:
        raise KeyError(f'"{text_source}" not found in columns.')

    return [text_source]


def char_tokens(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric characters."""
    return list(re.sub(r"[^A-Za-z0-9_]", "", str(text).lower()))


def ngrams(tokens: list[str], n: int):
    """Yield n-grams from token list."""
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def flatten_ngrams(ngs):
    """Convert tuple n-grams to strings."""
    for tup in ngs:
        yield " ".join(tup)


def compute_ngram_distributions(
    df: pd.DataFrame,
    text_source: str = "object_1+2",
    n: int = 2,
) -> dict[str, Counter]:
    """Compute n-gram count distributions by statement type."""
    df = df.copy()
    df["_statement_type"] = df.apply(get_bigram_statement_type, axis=1)
    df = df[df["_statement_type"].notna()]

    sources = resolve_text_sources(df, text_source)
    by_type = {kind: Counter() for kind in BIGRAM_TYPES}

    for _, row in df.iterrows():
        kind = row["_statement_type"]

        if kind not in by_type:
            continue

        for src in sources:
            toks = char_tokens(row.get(src, ""))
            by_type[kind].update(flatten_ngrams(ngrams(toks, n)))

    return by_type


def make_frequency_table(by_type: dict[str, Counter]) -> pd.DataFrame:
    """Convert n-gram counts to normalized frequency table."""
    vocab = set()

    for ctr in by_type.values():
        vocab.update(ctr.keys())

    rows = []

    for ng in vocab:
        row = {"ngram": ng}
        for kind in BIGRAM_TYPES:
            row[kind] = by_type[kind].get(ng, 0)
        rows.append(row)

    freq_df = pd.DataFrame(rows).set_index("ngram")

    for kind in BIGRAM_TYPES:
        total = sum(by_type[kind].values()) or 1
        freq_df[kind] = freq_df[kind] / total

    return freq_df


def moving_average(y, window: int = 101):
    """Smooth a vector with a centered moving average."""
    y = np.asarray(y, dtype=float)

    if window is None or window <= 1 or window > len(y):
        return y

    kernel = np.ones(int(window), dtype=float) / float(window)

    return np.convolve(y, kernel, mode="same")


def compute_bigram_rank_curves(dataset: str, dataset_root: Path):
    """Compute bigram rank-frequency curves for one dataset."""
    csv_paths = dataset_csv_paths(dataset, dataset_root)
    df = load_and_concat_csvs(csv_paths)
    by_type = compute_ngram_distributions(df=df, text_source="object_1+2", n=2)
    freq_df = make_frequency_table(by_type)

    sorted_df = freq_df.sort_values("true", ascending=False).reset_index(drop=False)
    ranks = np.arange(1, len(sorted_df) + 1)
    curves = {kind: moving_average(sorted_df[kind].values) for kind in BIGRAM_TYPES}

    return ranks, curves


def plot_bigram_panel(
    ax,
    dataset: str,
    dataset_root: Path,
    panel_label: str,
    show_xlabel: bool,
) -> None:
    """Plot one bigram rank-frequency panel."""
    ranks, curves = compute_bigram_rank_curves(
        dataset=dataset,
        dataset_root=dataset_root,
    )

    for kind in BIGRAM_TYPES:
        ax.plot(
            ranks,
            curves[kind],
            label=BIGRAM_LABELS[kind],
            color=COLORS[kind],
            linewidth=1.2,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Norm. Freq.", fontsize=7)

    if show_xlabel:
        ax.set_xlabel("Bigram Rank", fontsize=7)
    else:
        ax.set_xticklabels([])

    ax.text(
        -0.5,
        1.25,
        f"{panel_label} {DATASET_SHORT_LABELS[dataset]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color="#444444",
    )

    style_axis(ax)


def load_one_cache(
    model: str,
    dataset_base: str,
    statement_type: str,
    perplexity_root: Path,
) -> pd.DataFrame | None:
    """Load one token-level familiarity cache."""
    dataset_folder = f"{dataset_base}_{statement_type}"
    folder = perplexity_root / model / dataset_folder
    token_path = folder / "token_scores.parquet"
    metadata_path = folder / "statement_metadata.csv"

    if not token_path.exists():
        log.warning(f"Missing token cache: {token_path}")
        return None

    token_df = pd.read_parquet(token_path)
    token_df["model"] = model
    token_df["dataset_base"] = dataset_base
    token_df["statement_type_raw"] = statement_type

    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)

        keep_cols = [
            "row_id",
            "statement",
            "object_1",
            "object_2",
            "correct",
            "category",
            "real_object",
            "fake_object",
            "fictional_object",
            "synthetic_fic_object",
            "in_train",
            "in_test",
            "in_cal",
        ]

        keep_cols = [col for col in keep_cols if col in meta.columns]
        meta = meta[keep_cols].drop_duplicates("row_id")
        token_df = token_df.merge(meta, on="row_id", how="left", suffixes=("", "_meta"))

    return token_df


def load_all_token_caches(
    perplexity_root: Path,
    models: list[str],
    datasets: list[str],
) -> pd.DataFrame:
    """Load all available token-level familiarity caches."""
    dfs = []

    for model in models:
        for dataset_base in datasets:
            for statement_type in CACHE_STATEMENT_TYPES:
                df = load_one_cache(
                    model=model,
                    dataset_base=dataset_base,
                    statement_type=statement_type,
                    perplexity_root=perplexity_root,
                )

                if df is not None:
                    dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No token caches found under {perplexity_root}")

    return pd.concat(dfs, ignore_index=True)


def assign_familiarity_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Assign analysis condition used for familiarity ranking."""
    df = df.copy()
    rows = []

    correct = pd.to_numeric(df.get("correct", np.nan), errors="coerce")
    is_true_false = df["statement_type_raw"].eq("true_false")
    is_synthetic = df["statement_type_raw"].eq("synthetic")
    is_synthetic_fic = df["statement_type_raw"].eq("synthetic_fic")
    is_fictional = df["statement_type_raw"].eq("fictional")

    base = df.copy()
    base["analysis_condition"] = np.nan

    base.loc[is_true_false & correct.eq(1), "analysis_condition"] = "true"
    base.loc[is_true_false & correct.eq(0), "analysis_condition"] = "false"
    base.loc[is_synthetic, "analysis_condition"] = "synthetic_tf"
    base.loc[is_synthetic_fic, "analysis_condition"] = "synthetic_fi"
    base.loc[is_fictional, "analysis_condition"] = "fictional"

    rows.append(base[base["analysis_condition"].notna()].copy())

    fictional_true = df[is_fictional & correct.eq(1)].copy()
    fictional_true["analysis_condition"] = "fictional_true"
    rows.append(fictional_true)

    return pd.concat(rows, ignore_index=True)


def compute_familiarity_rankings(
    token_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compute familiarity ranks by model/dataset/condition."""
    df = assign_familiarity_condition(token_df)

    required = ["is_object_token", "is_padding", "token_prob"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing token columns: {missing}")

    df = df[
        df["is_object_token"].astype(bool)
        & ~df["is_padding"].astype(bool)
        & df["token_prob"].notna()
    ].copy()

    statement_scores = (
        df.groupby(
            ["model", "dataset_base", "analysis_condition", "row_id"],
            as_index=False,
        )["token_prob"]
        .mean()
        .rename(columns={"token_prob": "statement_prob"})
    )

    condition_scores = (
        statement_scores
        .groupby(["model", "dataset_base", "analysis_condition"], as_index=False)
        .agg(score=("statement_prob", "mean"), n_statements=("statement_prob", "count"))
    )

    ranks = []

    for _, sub in condition_scores.groupby(["model", "dataset_base"]):
        sub = sub[sub["analysis_condition"].isin(FAMILIARITY_TYPES)].copy()

        if sub["analysis_condition"].nunique() < 2:
            continue

        sub["rank"] = sub["score"].rank(ascending=False, method="average")
        ranks.append(sub)

    if not ranks:
        raise RuntimeError("No familiarity ranks could be computed.")

    out = pd.concat(ranks, ignore_index=True)
    out.to_csv(output_dir / "fig2_combined_familiarity_rankings.csv", index=False)

    return out


def plot_familiarity_panel(ax, ranks: pd.DataFrame, panel_label: str) -> None:
    """Plot familiarity rank panel."""
    rng = np.random.default_rng(1693)

    y_positions = {
        condition: len(FAMILIARITY_TYPES) - 1 - i
        for i, condition in enumerate(FAMILIARITY_TYPES)
    }

    for condition in FAMILIARITY_TYPES:
        cond_data = ranks[ranks["analysis_condition"].eq(condition)]
        color = COLORS[condition]
        y_base = y_positions[condition]

        for dataset in DATASETS:
            dataset_data = cond_data[cond_data["dataset_base"].eq(dataset)]

            if dataset_data.empty:
                continue

            x_jitter = rng.uniform(-0.055, 0.055, size=len(dataset_data))
            y_jitter = rng.uniform(-0.035, 0.035, size=len(dataset_data))

            ax.scatter(
                dataset_data["rank"].to_numpy() + x_jitter,
                np.full(len(dataset_data), y_base + DATASET_OFFSETS[dataset]) + y_jitter,
                marker=DATASET_MARKERS[dataset],
                s=18,
                color=color,
                alpha=0.75,
                linewidths=0.30,
                edgecolors="white",
                zorder=3,
            )

    for condition in FAMILIARITY_TYPES:
        y_base = y_positions[condition]

        for dataset in DATASETS:
            dataset_data = ranks[
                ranks["analysis_condition"].eq(condition)
                & ranks["dataset_base"].eq(dataset)
            ]

            if dataset_data.empty:
                continue

            mean_rank = dataset_data["rank"].mean()
            y = y_base + DATASET_OFFSETS[dataset]

            ax.vlines(
                x=mean_rank,
                ymin=y - 0.07,
                ymax=y + 0.07,
                color="black",
                linewidth=1.3,
                zorder=5,
            )

    for y in y_positions.values():
        ax.axhline(y, color="#dddddd", linewidth=0.4, zorder=0)

    ax.set_yticks([y_positions[c] for c in FAMILIARITY_TYPES])
    ax.set_yticklabels([FAMILIARITY_LABELS[c] for c in FAMILIARITY_TYPES], fontsize=7)
    ax.set_xlim(0.6, len(FAMILIARITY_TYPES) + 0.4)
    ax.set_xticks(range(1, len(FAMILIARITY_TYPES) + 1))
    ax.set_xlabel("Rank (1 = Most Familiar)", fontsize=7)
    ax.set_ylim(-0.65, len(FAMILIARITY_TYPES) - 0.35)
    ax.grid(axis="x", alpha=0.38, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.text(
        -0.20,
        1.06,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color="#444444",
    )

    style_axis(ax)


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first available column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"Could not find any of {candidates}. Available columns: {list(df.columns)}")


def heterogeneity_file_candidates(
    heterogeneity_root: Path,
    dataset: str,
) -> list[Path]:
    """Return candidate heterogeneity CSV paths for one dataset."""
    return [
        heterogeneity_root / f"{dataset}_llm_level_within_between.csv",
        heterogeneity_root / f"{dataset}_within_between.csv",
    ]


def load_dataset_heterogeneity(
    heterogeneity_root: Path,
    dataset: str,
) -> pd.DataFrame:
    """Load heterogeneity data for one dataset."""
    for path in heterogeneity_file_candidates(heterogeneity_root, dataset):
        if path.exists():
            df = pd.read_csv(path)

            if "dataset" not in df.columns:
                df["dataset"] = dataset

            return df

    all_path = heterogeneity_root / "all_llm_level_within_between.csv"
    if all_path.exists():
        df = pd.read_csv(all_path)

        if "dataset" not in df.columns:
            raise ValueError(f"{all_path} exists but does not contain a dataset column.")

        return df[df["dataset"].eq(dataset)].copy()

    checked = "\n".join(
        str(path) for path in heterogeneity_file_candidates(heterogeneity_root, dataset)
    )
    checked += f"\n{all_path}"

    raise FileNotFoundError(
        f"Could not find heterogeneity CSV for {dataset}. Checked:\n{checked}"
    )


def load_heterogeneity_data(
    heterogeneity_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Load within-group activation distances in long format."""
    rows = []

    paths = sorted(heterogeneity_root.glob("*_within_between.csv"))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No *_within_between.csv files found under {heterogeneity_root}"
        )

    for path in paths:
        df = pd.read_csv(path)

        if "dataset" not in df.columns or "model" not in df.columns:
            raise ValueError(f"{path} must contain dataset and model columns.")

        for statement_type, candidates in DISTANCE_COLUMNS.items():
            col = resolve_column(df, candidates)
            tmp = df[["model", "dataset", col]].copy()
            tmp = tmp.rename(columns={col: "within_distance"})
            tmp["statement_type"] = statement_type
            tmp["source_file"] = path.name
            rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out["within_distance"] = pd.to_numeric(out["within_distance"], errors="coerce")
    out = out[out["dataset"].isin(DATASETS)]
    out = out.dropna(subset=["within_distance"])
    out.to_csv(output_dir / "fig2_statement_heterogeneity_long.csv", index=False)

    return out


def plot_heterogeneity_panel(ax, df: pd.DataFrame, panel_label: str) -> None:
    """Plot activation heterogeneity panel."""
    rng = np.random.default_rng(1693)

    models_in_data = [model for model in MODELS if model in set(df["model"])]
    extra_models = sorted(set(df["model"]) - set(models_in_data))
    models = models_in_data + extra_models

    y_positions = {
        model: len(models) - 1 - i
        for i, model in enumerate(models)
    }

    for statement_type in HETERO_TYPES:
        type_data = df[df["statement_type"].eq(statement_type)]
        color = HETERO_COLORS[statement_type]

        for dataset in DATASETS:
            subset = type_data[type_data["dataset"].eq(dataset)]

            if subset.empty:
                continue

            for _, row in subset.iterrows():
                model = row["model"]

                if model not in y_positions:
                    continue

                ax.scatter(
                    row["within_distance"] + rng.uniform(-0.006, 0.006),
                    y_positions[model] + DATASET_OFFSETS[dataset] + rng.uniform(-0.012, 0.012),
                    marker=DATASET_MARKERS[dataset],
                    s=14,
                    color=color,
                    alpha=0.75,
                    linewidths=0.30,
                    edgecolors="white",
                    zorder=3,
                )

    for y in y_positions.values():
        ax.axhline(y, color="#dddddd", linewidth=0.4, zorder=0)

    ax.set_yticks([y_positions[m] for m in models])
    ax.set_yticklabels(models, fontsize=6)
    ax.set_xlabel("Within-group Avg. Euclidean Distance", fontsize=7)
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.text(
        -0.20,
        1.06,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color="#444444",
    )

    style_axis(ax)


def make_figure(
    dataset_root: Path,
    perplexity_root: Path,
    heterogeneity_root: Path,
    output_dir: Path,
    output_name: str,
) -> None:
    """Create and save Figure 2."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hetero_df = load_heterogeneity_data(
        heterogeneity_root=heterogeneity_root,
        output_dir=output_dir,
    )

    token_df = load_all_token_caches(
        perplexity_root=perplexity_root,
        models=MODELS,
        datasets=DATASETS,
    )

    ranks = compute_familiarity_rankings(
        token_df=token_df,
        output_dir=output_dir,
    )

    fig = plt.figure(figsize=(7.2, 5.0))

    gs = GridSpec(
        nrows=5,
        ncols=3,
        figure=fig,
        width_ratios=[0.72, 1.35, 1.35],
        height_ratios=[0.05, 1, 1, 1, 0.5],
        wspace=0.90,
        hspace=0.70,
    )

    for j, title in enumerate(
        [
            "Bigram distributions",
            "Statement heterogeneity",
            "Statement familiarity",
        ]
    ):
        ax_title = fig.add_subplot(gs[0, j])
        ax_title.axis("off")
        ax_title.text(
            0.5,
            0.25,
            title,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#444444",
            transform=ax_title.transAxes,
        )

    for i, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[i + 1, 0])
        plot_bigram_panel(
            ax=ax,
            dataset=dataset,
            dataset_root=dataset_root,
            panel_label=f"({chr(ord('a') + i)})",
            show_xlabel=(i == len(DATASETS) - 1),
        )

    ax_hetero = fig.add_subplot(gs[1:4, 1])
    plot_heterogeneity_panel(ax_hetero, hetero_df, "(d)")

    ax_fam = fig.add_subplot(gs[1:4, 2])
    plot_familiarity_panel(ax_fam, ranks, "(e)")

    ax_leg = fig.add_subplot(gs[4, :])
    ax_leg.axis("off")

    shared_handles = [
        Line2D([], [], linestyle="-", color=COLORS["true"], linewidth=3.0, label="True"),
        Line2D([], [], linestyle="-", color=COLORS["false"], linewidth=3.0, label="False"),
        Line2D([], [], linestyle="-", color=COLORS["fictional"], linewidth=3.0, label="Fictional"),
        Line2D([], [], linestyle="-", color=COLORS["fictional_true"], linewidth=3.0, label="Fictional (T)"),
        Line2D([], [], linestyle="-", color=COLORS["synthetic_tf"], linewidth=3.0, label="Synthetic (TF)"),
        Line2D([], [], linestyle="-", color=COLORS["synthetic_fi"], linewidth=3.0, label="Synthetic (Fi)"),
        Line2D([], [], marker=DATASET_MARKERS["cities_loc"], linestyle="None", color="#555555", markerfacecolor="#555555", markersize=5, label="City Locations"),
        Line2D([], [], marker=DATASET_MARKERS["med_indications"], linestyle="None", color="#555555", markerfacecolor="#555555", markersize=5, label="Medical Indications"),
        Line2D([], [], marker=DATASET_MARKERS["defs"], linestyle="None", color="#555555", markerfacecolor="#555555", markersize=5, label="Word Definitions"),
        Line2D([], [], marker="|", linestyle="None", color="black", markersize=9, markeredgewidth=1.3, label="Avg. Rank"),
    ]

    ax_leg.legend(
        handles=shared_handles,
        loc="center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.3),
        fontsize=6.5,
        columnspacing=1.1,
        handletextpad=0.45,
        labelspacing=0.9,
    )

    fig.tight_layout(rect=[0, 0, 1, 1])

    for ext in ["png", "pdf"]:
        out_path = output_dir / f"{output_name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Wrote: {out_path}")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default=str(project_path("datasets")),
    )

    parser.add_argument(
        "--perplexity-root",
        default=str(project_path("outputs", "perplexity")),
    )

    parser.add_argument(
        "--heterogeneity-root",
        default=str(project_path("outputs", "analysis_data", "within_between")),
    )

    parser.add_argument(
        "--output-dir",
        default=str(project_path("outputs", "plots")),
    )
    parser.add_argument(
        "--output-name",
        default="fig2",
    )

    return parser.parse_args()


def main() -> None:
    """Create Figure 2."""
    args = parse_args()

    make_figure(
        dataset_root=Path(args.dataset_root),
        perplexity_root=Path(args.perplexity_root),
        heterogeneity_root=Path(args.heterogeneity_root),
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()