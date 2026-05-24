"""
cluster_retraction_activations.py

Cluster statement-level test activations and summarize whether epistemic
retractions are geometrically structured in activation space.
"""

import argparse
import copy
import itertools
import json
import os
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    umap = None

try:
    from hdbscan import HDBSCAN as ExternalHDBSCAN
except ImportError:
    ExternalHDBSCAN = None

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:
    SklearnHDBSCAN = None


PROJECT_ROOT = Path(__file__).resolve().parent
PROBE_NAME = "sAwMIL"

DEFAULT_MODELS = [
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

TASK_NAME = {
    1: "synthetic",
    2: "synthetic_fic",
    3: "fictional",
    4: "fictional_true",
    5: "noise",
}


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def add_project_to_path() -> None:
    """Add project root to sys.path so local project modules can be imported."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def safe_model_name(model: str) -> str:
    """Return a filesystem-safe model name."""
    return str(model).replace("/", "__")


def compose_probe_cfg(model: str, dataset: str, task: int = 0):
    """Compose the Hydra config used by the probe."""
    with hydra.initialize(config_path="configs", version_base=None):
        return hydra.compose(
            config_name="probe_linear_mil",
            overrides=[
                f"model={model}",
                f"datapack={dataset}",
                f"datapack@datapack_test={dataset}",
                f"task={task}",
            ],
        )


def get_prediction_csv_path(dataset: str, layer_label: str) -> Path:
    """Return the combined prediction CSV path."""
    path = (
        project_path("outputs", "analysis_data")
        / PROBE_NAME
        / dataset
        / f"{PROBE_NAME}_{dataset}_test_predictions_{layer_label}.csv"
    )

    if path.exists():
        return path

    raise FileNotFoundError(f"Could not find prediction CSV: {path}")


def load_full_and_filtered_test_df(
    dataset: str,
    model_for_cfg: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load the full test dataframe and recreate the true-real filtering."""
    add_project_to_path()
    from utils import load_data

    cfg = compose_probe_cfg(model=model_for_cfg, dataset=dataset, task=0)

    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        dh = load_data(cfg)
        full_test_df = dh.get_test_df().reset_index(drop=False)
    finally:
        os.chdir(old_cwd)

    if "index" in full_test_df.columns and "statement_id" not in full_test_df.columns:
        full_test_df = full_test_df.rename(columns={"index": "statement_id"})

    if "statement_id" not in full_test_df.columns:
        full_test_df.insert(0, "statement_id", np.arange(len(full_test_df)))

    required_cols = [
        "correct",
        "real_object",
        "fictional_object",
        "synthetic_fic_object",
    ]
    missing_cols = [col for col in required_cols if col not in full_test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required test dataframe columns: {missing_cols}")

    true_test_mask = (
        full_test_df["correct"].astype(bool)
        & full_test_df["real_object"].astype(int).eq(1)
        & full_test_df["fictional_object"].astype(int).eq(0)
        & full_test_df["synthetic_fic_object"].astype(int).eq(0)
    ).to_numpy()

    filtered_test_df = full_test_df.loc[true_test_mask].reset_index(drop=True)

    return full_test_df, true_test_mask, filtered_test_df


def load_filtered_test_activations(
    model: str,
    dataset: str,
    true_test_mask: np.ndarray,
    pooling: str,
) -> tuple[np.ndarray, int]:
    """Load full test activations at the model's task layer and filter rows."""
    add_project_to_path()
    from utils import load_data

    cfg = compose_probe_cfg(model=model, dataset=dataset, task=0)
    layer = int(cfg.model.task_layers[dataset])

    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        dh = load_data(cfg)
        bags_obj = dh.test_bags(layer_id=layer, drop_zeros=True)
    finally:
        os.chdir(old_cwd)

    if "embeddings" not in bags_obj:
        raise KeyError("Expected dh.test_bags(..., drop_zeros=True) to return 'embeddings'.")

    bags = bags_obj["embeddings"]
    if len(bags) != len(true_test_mask):
        raise ValueError(
            f"Activation/test mask length mismatch for {model}: "
            f"len(bags)={len(bags)}, len(true_test_mask)={len(true_test_mask)}"
        )

    pooled = []

    for i, (bag, keep) in enumerate(zip(bags, true_test_mask)):
        if not keep:
            continue

        arr = np.asarray(bag)

        if arr.ndim != 2:
            raise ValueError(f"Expected bag {i} to be 2D, got shape={arr.shape}")

        if arr.shape[0] == 0:
            raise ValueError(f"Bag {i} has zero tokens after drop_zeros=True")

        if pooling == "mean":
            vec = arr.mean(axis=0)
        elif pooling == "last":
            vec = arr[-1]
        elif pooling == "first":
            vec = arr[0]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        pooled.append(vec)

    X = np.vstack(pooled).astype(np.float32, copy=False)

    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"Non-finite values in pooled activations for {model}: {bad}")

    return X, layer


def prediction_col(model: str, task: int, layer_label: str) -> str:
    return f"{model}__task_{task}__layer_{layer_label}__y_hat"


def add_retraction_columns(
    df: pd.DataFrame,
    model: str,
    layer_label: str,
) -> pd.DataFrame:
    """Add binary retraction columns for one model."""
    out = df.copy()
    base_col = prediction_col(model, 0, layer_label)

    if base_col not in out.columns:
        raise KeyError(f"Missing baseline prediction column: {base_col}")

    baseline_true = out[base_col].astype(float).eq(1)

    for task, name in TASK_NAME.items():
        col = prediction_col(model, task, layer_label)
        if col not in out.columns:
            print(f"Warning: missing task column for {model}: {col}")
            out[f"retract_{name}"] = False
            continue

        out[f"retract_{name}"] = baseline_true & out[col].astype(float).eq(0)

    retract_cols = [f"retract_{name}" for name in TASK_NAME.values()]
    out["baseline_true"] = baseline_true
    out["retracted_any"] = out[retract_cols].any(axis=1)
    out["never_retracted"] = out["baseline_true"] & ~out["retracted_any"]

    def retraction_type(row: pd.Series) -> str:
        active = [name for name in TASK_NAME.values() if bool(row[f"retract_{name}"])]

        if len(active) == 0:
            if bool(row["baseline_true"]):
                return "baseline_true_never_retracted"
            return "not_predicted_true_at_baseline"

        if len(active) == 1:
            return active[0]

        return "multiple"

    out["retraction_type"] = out.apply(retraction_type, axis=1)
    out["n_retraction_settings"] = out[retract_cols].sum(axis=1).astype(int)

    return out


def prepare_model_inputs(
    model: str,
    pred_df: pd.DataFrame,
    filtered_test_df: pd.DataFrame,
    true_test_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Validate metadata alignment, add retraction labels, and load activations."""
    if len(pred_df) != len(filtered_test_df):
        raise ValueError(
            f"Prediction CSV rows ({len(pred_df)}) do not match recreated filtered "
            f"test rows ({len(filtered_test_df)})."
        )

    if "statement_id" in pred_df.columns and "statement_id" in filtered_test_df.columns:
        left = pred_df["statement_id"].astype(str).to_numpy()
        right = filtered_test_df["statement_id"].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                "statement_id order mismatch between prediction CSV and recreated test dataframe."
            )

    df = add_retraction_columns(
        df=pred_df,
        model=model,
        layer_label=args.layer_label,
    )
    df["_source_row_id"] = np.arange(len(df))

    X, actual_layer = load_filtered_test_activations(
        model=model,
        dataset=args.dataset,
        true_test_mask=true_test_mask,
        pooling=args.pooling,
    )

    if len(X) != len(df):
        raise ValueError(
            f"Activation rows ({len(X)}) do not match prediction rows ({len(df)})."
        )

    return df, X, actual_layer


def select_scope(
    df: pd.DataFrame,
    X: np.ndarray,
    scope: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter rows and activations according to the requested analysis scope."""
    if scope == "all":
        mask = df["baseline_true"].to_numpy()
    elif scope == "all_rows":
        mask = np.ones(len(df), dtype=bool)
    elif scope == "retracted":
        mask = df["retracted_any"].to_numpy()
    elif scope == "never_retracted":
        mask = df["never_retracted"].to_numpy()
    else:
        raise ValueError(f"Unknown scope: {scope}")

    return df.loc[mask].reset_index(drop=True), X[mask]


def make_cluster_features(
    X: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Standardize activations, optionally apply PCA, and optionally apply UMAP."""
    X_scaled = StandardScaler().fit_transform(X)

    info = {
        "pca_components_used": 0,
        "umap_components_used": 0,
    }

    if args.pca_components > 0 and args.pca_components < min(X_scaled.shape):
        pca = PCA(n_components=args.pca_components, random_state=args.random_state)
        X_reduced = pca.fit_transform(X_scaled)
        info["pca_components_used"] = int(args.pca_components)
        info["pca_explained_variance_ratio_sum"] = float(
            pca.explained_variance_ratio_.sum()
        )
    else:
        X_reduced = X_scaled

    if args.cluster_method != "umap_hdbscan":
        return X_reduced, info

    if umap is None:
        raise ImportError(
            "cluster-method=umap_hdbscan requires umap-learn. "
            "Install it with: pip install umap-learn"
        )

    if len(X_reduced) <= args.umap_n_neighbors:
        n_neighbors = max(2, len(X_reduced) - 1)
        print(
            f"Warning: reducing umap_n_neighbors from {args.umap_n_neighbors} "
            f"to {n_neighbors} because only {len(X_reduced)} rows are available."
        )
    else:
        n_neighbors = args.umap_n_neighbors

    reducer = umap.UMAP(
        n_components=args.umap_components,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
    )
    X_umap = reducer.fit_transform(X_reduced)

    info.update(
        {
            "umap_components_used": int(args.umap_components),
            "umap_n_neighbors_used": int(n_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "umap_metric": args.umap_metric,
        }
    )

    return X_umap, info


def fit_clusters(
    X: np.ndarray,
    method: str,
    n_clusters: int,
    random_state: int,
    min_cluster_size: int = 5,
    min_samples: int = None,
) -> np.ndarray:
    """Fit a clustering model and return integer cluster labels."""
    if method in {"kmeans", "agglomerative"} and len(X) < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} but only {len(X)} rows are available.")

    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=random_state,
        )
        return model.fit_predict(X)

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(X)

    if method == "umap_hdbscan":
        if len(X) < min_cluster_size:
            raise ValueError(
                f"min_cluster_size={min_cluster_size} but only {len(X)} rows are available."
            )

        if ExternalHDBSCAN is not None:
            model = ExternalHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            return model.fit_predict(X)

        if SklearnHDBSCAN is not None:
            model = SklearnHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
            )
            return model.fit_predict(X)

        raise ImportError(
            "cluster-method=umap_hdbscan requires either hdbscan or "
            "sklearn.cluster.HDBSCAN. Try: pip install hdbscan"
        )

    raise ValueError(f"Unknown clustering method: {method}")


def safe_frac(num, den) -> float:
    return float(num) / float(den) if den else 0.0


def summarize_clusters(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Build one row per cluster with retraction and metadata summaries."""
    rows = []
    retract_cols = [f"retract_{name}" for name in TASK_NAME.values()]

    for cluster_id, g in df.groupby("cluster", sort=True):
        row = {
            "model": model,
            "cluster": int(cluster_id),
            "n": int(len(g)),
            "baseline_true_frac": g["baseline_true"].mean(),
            "retracted_any_frac": g["retracted_any"].mean(),
            "never_retracted_frac": g["never_retracted"].mean(),
            "mean_n_retraction_settings": g["n_retraction_settings"].mean(),
        }

        for col in retract_cols:
            row[f"{col}_frac"] = g[col].mean()

        if "category" in g.columns:
            row["top_categories"] = "; ".join(
                f"{idx}:{val}"
                for idx, val in g["category"].value_counts(dropna=False).head(5).items()
            )

        if "negation" in g.columns:
            row["negation_frac"] = pd.to_numeric(
                g["negation"],
                errors="coerce",
            ).mean()

        if "object_2" in g.columns:
            row["top_object_2"] = "; ".join(
                f"{idx}:{val}"
                for idx, val in g["object_2"].value_counts(dropna=False).head(8).items()
            )

        row["retraction_type_counts"] = "; ".join(
            f"{idx}:{val}"
            for idx, val in g["retraction_type"].value_counts(dropna=False).items()
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def format_prompt_example(row: pd.Series) -> str:
    """Format one statement row for an LLM cluster-labeling prompt."""
    flags = []

    if bool(row.get("retracted_any", False)):
        flags.append(f"retracted={row.get('retraction_type', 'unknown')}")
    else:
        flags.append("not_retracted")

    if "category" in row:
        flags.append(f"category={row.get('category')}")
    if "negation" in row:
        flags.append(f"negation={row.get('negation')}")
    if "object_1" in row and pd.notna(row.get("object_1")):
        flags.append(f"object_1={row.get('object_1')}")
    if "object_2" in row and pd.notna(row.get("object_2")):
        flags.append(f"object_2={row.get('object_2')}")

    return f"- [{', '.join(flags)}] {row['statement']}"


def select_prompt_examples(
    g: pd.DataFrame,
    max_examples: int,
) -> tuple[list[str], list[str]]:
    """Select retracted and stable examples for cluster-labeling prompts."""
    retracted = g[g["retracted_any"].astype(bool)].copy()
    stable = g[g["never_retracted"].astype(bool)].copy()

    retracted = retracted.sort_values(
        by=["n_retraction_settings", "retraction_type"],
        ascending=[False, True],
    )
    stable = stable.sort_values(
        by=["n_retraction_settings", "retraction_type"],
        ascending=[False, True],
    )

    if len(retracted) > 0 and len(stable) > 0:
        n_retracted = max(1, min(len(retracted), max_examples // 2))
        n_stable = max(1, min(len(stable), max_examples - n_retracted))
    elif len(retracted) > 0:
        n_retracted = min(len(retracted), max_examples)
        n_stable = 0
    else:
        n_retracted = 0
        n_stable = min(len(stable), max_examples)

    retracted_lines = [
        format_prompt_example(row)
        for _, row in retracted.head(n_retracted).iterrows()
    ]
    stable_lines = [
        format_prompt_example(row)
        for _, row in stable.head(n_stable).iterrows()
    ]

    return retracted_lines, stable_lines


def build_nearest_stable_examples(
    assignments_df: pd.DataFrame,
    full_df: pd.DataFrame,
    full_X: np.ndarray,
    n_neighbors: int,
) -> pd.DataFrame:
    """Retrieve nearby never-retracted examples for prompt construction only."""
    required_col = "_source_row_id"
    if required_col not in assignments_df.columns or required_col not in full_df.columns:
        return pd.DataFrame()

    stable_pool = full_df[full_df["never_retracted"].astype(bool)].copy()
    baseline_pool = full_df[full_df["baseline_true"].astype(bool)].copy()

    if len(stable_pool) == 0 or len(baseline_pool) == 0:
        return pd.DataFrame()

    baseline_ids = baseline_pool[required_col].to_numpy(dtype=int)
    scaler = StandardScaler().fit(full_X[baseline_ids])
    X_scaled = scaler.transform(full_X)

    stable_ids = stable_pool[required_col].to_numpy(dtype=int)
    X_stable = X_scaled[stable_ids]

    k = min(max(1, n_neighbors), len(stable_pool))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_stable)

    neighbor_rows = []

    for cluster_id, g in assignments_df.groupby("cluster", sort=True):
        member_ids = g[required_col].to_numpy(dtype=int)
        if len(member_ids) == 0:
            continue

        centroid = X_scaled[member_ids].mean(axis=0, keepdims=True)
        distances, indices = nn.kneighbors(centroid)

        for rank, (dist, stable_pos) in enumerate(zip(distances[0], indices[0]), start=1):
            row = stable_pool.iloc[int(stable_pos)].copy()
            row["cluster"] = int(cluster_id)
            row["neighbor_rank"] = int(rank)
            row["neighbor_distance_cosine"] = float(dist)
            row["neighbor_source"] = "nearest_never_retracted_baseline_true"
            neighbor_rows.append(row)

    return pd.DataFrame(neighbor_rows).reset_index(drop=True)


def cluster_labeling_prompt(
    cluster_df: pd.DataFrame,
    model: str,
    dataset: str,
    cluster_id: int,
    max_examples: int,
    stable_neighbors_df: pd.DataFrame | None = None,
) -> str:
    """Create a contrastive LLM-ready prompt for one cluster."""
    g = cluster_df[cluster_df["cluster"].eq(cluster_id)].copy()
    counts = g["retraction_type"].value_counts(dropna=False).to_dict()

    n_retracted = int(g["retracted_any"].astype(bool).sum())
    n_stable = int(g["never_retracted"].astype(bool).sum())
    n_other = int(len(g) - n_retracted - n_stable)

    retracted_lines, stable_lines = select_prompt_examples(
        g=g,
        max_examples=max_examples,
    )

    stable_source_label = "from the same activation-space cluster"

    if (
        len(stable_lines) == 0
        and stable_neighbors_df is not None
        and len(stable_neighbors_df) > 0
    ):
        nn_g = stable_neighbors_df[stable_neighbors_df["cluster"].eq(cluster_id)].copy()
        if len(nn_g) > 0:
            nn_g = nn_g.sort_values("neighbor_rank")
            n_stable_prompt = min(len(nn_g), max(1, max_examples - len(retracted_lines)))
            stable_lines = [
                format_prompt_example(row)
                for _, row in nn_g.head(n_stable_prompt).iterrows()
            ]
            stable_source_label = (
                "retrieved after clustering as nearest never-retracted neighbors"
            )
            n_stable = int(len(nn_g))

    if len(retracted_lines) > 0:
        retracted_section = (
            "Retracted examples from this activation-space cluster:\n"
            + "\n".join(retracted_lines)
        )
    else:
        retracted_section = (
            "Retracted examples from this activation-space cluster:\n"
            "- None available in this cluster."
        )

    if len(stable_lines) > 0:
        stable_section = (
            f"\n\nNearby non-retracted examples ({stable_source_label}):\n"
            + "\n".join(stable_lines)
            + "\n"
        )
        contrast_instruction = (
            "Compare the retracted examples against the nearby non-retracted examples. "
            "Focus on what seems more common or distinctive among the retracted examples, "
            "not just what all examples in the cluster share."
        )
    else:
        stable_section = """

Nearby non-retracted examples:
- None available in this prompt. If this is a retracted-only analysis, do not claim to know what distinguishes these statements from stable neighbors.
"""
        contrast_instruction = (
            "No nearby non-retracted examples are included, so focus on shared noun/entity-level "
            "patterns among the retracted examples and state that a contrastive comparison is not possible from this prompt alone."
        )

    return f"""You are helping analyze clusters of natural-language statements from an LLM activation space.

Goal:
Identify semantic patterns that may help explain why some statements undergo epistemic retractions while nearby statements do not.

Context:
- Dataset: {dataset}
- Model: {model}
- Cluster id: {cluster_id}
- Cluster size: {len(g)}
- Number retracted: {n_retracted}
- Number nearby non-retracted: {n_stable}
- Number other baseline statuses: {n_other}
- Retraction-type counts: {json.dumps(counts, sort_keys=True)}

Important instructions:
- Do NOT focus primarily on the surface template, such as "is a synonym of", "is a type of", or "is/is not a".
- Do NOT use the dataset category name as the main explanation unless it captures a more specific noun-level pattern.
- Many statements in the dataset share the same templates, so template labels are usually not informative.
- Focus on what the nouns/entities have in common: named entities, occupations, biological terms, artifacts, abstract concepts, rare or archaic words, technical jargon, ambiguity/polysemy, concreteness, familiarity, or semantic domain.
- The cluster label should be short and snappy, ideally 2-5 words, like a topic-model label.
- {contrast_instruction}
- Prefer concise hypotheses grounded directly in the observed nouns/entities.
- Avoid overly speculative cognitive language. Phrases like "may suggest" or "could indicate" are better than strong causal claims.
- If there is no clear noun-level pattern, say so rather than forcing one.

Task:
1. Give this cluster a short noun/entity-focused label.
2. Describe the pattern shared by the cluster.
3. If stable examples are available, describe what appears to distinguish the retracted examples from the nearby non-retracted examples.
4. Hypothesize why statements involving this kind of content might be vulnerable to epistemic retractions.

{retracted_section}{stable_section}

Return JSON with keys:
  "cluster_label"
  "shared_cluster_pattern"
  "retracted_vs_stable_contrast"
  "retraction_hypothesis"
  "caveats"
"""


def write_prompts(
    assignments_df: pd.DataFrame,
    out_dir: Path,
    model: str,
    dataset: str,
    max_examples: int,
    stable_neighbors_df: pd.DataFrame | None = None,
) -> None:
    """Write one prompt text file per cluster plus a combined JSONL."""
    prompt_dir = out_dir / "llm_labeling_prompts_contrastive"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = prompt_dir / f"{model}__cluster_labeling_prompts.jsonl"

    with jsonl_path.open("w") as f_jsonl:
        for cluster_id in sorted(assignments_df["cluster"].unique()):
            prompt = cluster_labeling_prompt(
                cluster_df=assignments_df,
                model=model,
                dataset=dataset,
                cluster_id=int(cluster_id),
                max_examples=max_examples,
                stable_neighbors_df=stable_neighbors_df,
            )

            txt_path = prompt_dir / f"{model}__cluster_{int(cluster_id):02d}.txt"
            txt_path.write_text(prompt)

            f_jsonl.write(
                json.dumps(
                    {
                        "model": model,
                        "dataset": dataset,
                        "cluster": int(cluster_id),
                        "prompt": prompt,
                    }
                )
                + "\n"
            )


def evaluate_cluster_label_alignment(assignments_df: pd.DataFrame) -> dict[str, float]:
    """Compute diagnostic alignment between clusters and retraction labels."""
    labels = assignments_df["cluster"].to_numpy()
    binary = assignments_df["retracted_any"].astype(int).to_numpy()

    out = {
        "ari_retracted_any": float(adjusted_rand_score(binary, labels)),
        "nmi_retracted_any": float(normalized_mutual_info_score(binary, labels)),
    }

    type_codes = pd.Categorical(assignments_df["retraction_type"]).codes
    out["ari_retraction_type"] = float(adjusted_rand_score(type_codes, labels))
    out["nmi_retraction_type"] = float(normalized_mutual_info_score(type_codes, labels))

    return out


def infer_models_from_prediction_table(
    df: pd.DataFrame,
    layer_label: str,
) -> list[str]:
    """Infer available model names from prediction-table columns."""
    pattern = re.compile(rf"(.+)__task_0__layer_{re.escape(layer_label)}__y_hat$")

    return sorted(
        match.group(1)
        for col in df.columns
        if (match := pattern.match(col)) is not None
    )


def parse_models(
    models_arg: str,
    pred_df: pd.DataFrame,
    layer_label: str,
) -> list[str]:
    """Parse --models argument."""
    if models_arg == "all":
        return infer_models_from_prediction_table(pred_df, layer_label)

    if models_arg == "default":
        return [
            model
            for model in DEFAULT_MODELS
            if prediction_col(model, 0, layer_label) in pred_df.columns
        ]

    return [model.strip() for model in models_arg.split(",") if model.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_optional_int_grid(value: str):
    out = []

    for x in value.split(","):
        x = x.strip()
        if not x:
            continue

        if x.lower() in {"none", "null", "na"}:
            out.append(None)
        else:
            out.append(int(x))

    return out


def get_cluster_size_stats(labels: np.ndarray):
    """Summarize non-noise cluster sizes."""
    non_noise = labels[labels != -1]

    if len(non_noise) == 0:
        return {
            "min_cluster_n": None,
            "median_cluster_n": None,
            "max_cluster_n": None,
        }

    _, counts = np.unique(non_noise, return_counts=True)

    return {
        "min_cluster_n": int(counts.min()),
        "median_cluster_n": float(np.median(counts)),
        "max_cluster_n": int(counts.max()),
    }


def evaluate_clustering_quality(
    X_for_cluster: np.ndarray,
    labels: np.ndarray,
    assignments_df: pd.DataFrame,
):
    """Evaluate clustering quality and retraction-label alignment diagnostics."""
    labels = np.asarray(labels)

    n_rows = int(len(labels))
    n_noise = int((labels == -1).sum())
    noise_frac = safe_frac(n_noise, n_rows)
    n_clusters_found = int(len(set(labels)) - (1 if -1 in set(labels) else 0))

    out = {
        "n_rows": n_rows,
        "n_clusters_found": n_clusters_found,
        "n_noise": n_noise,
        "noise_frac": noise_frac,
        "valid_for_quality": False,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "selection_score": -1e9,
        **get_cluster_size_stats(labels),
    }

    non_noise_mask = labels != -1
    labels_non_noise = labels[non_noise_mask]
    X_non_noise = X_for_cluster[non_noise_mask]

    if n_clusters_found >= 2 and len(X_non_noise) > n_clusters_found:
        out["valid_for_quality"] = True
        out["silhouette"] = float(silhouette_score(X_non_noise, labels_non_noise))
        out["calinski_harabasz"] = float(
            calinski_harabasz_score(X_non_noise, labels_non_noise)
        )
        out["davies_bouldin"] = float(
            davies_bouldin_score(X_non_noise, labels_non_noise)
        )

        tiny_cluster_penalty = (
            0.10
            if out["min_cluster_n"] is not None and out["min_cluster_n"] < 3
            else 0.0
        )
        too_few_clusters_penalty = 0.05 if n_clusters_found < 3 else 0.0

        out["selection_score"] = (
            float(out["silhouette"])
            - 0.25 * noise_frac
            - tiny_cluster_penalty
            - too_few_clusters_penalty
        )

    try:
        out.update(evaluate_cluster_label_alignment(assignments_df))
    except Exception:
        pass

    return out


def make_sweep_grid(
    args: argparse.Namespace,
    n_rows: int,
) -> list[argparse.Namespace]:
    """Create one argument object per valid clustering hyperparameter setting."""
    grid_args = []

    if args.cluster_method == "umap_hdbscan":
        grid = itertools.product(
            parse_int_grid(args.sweep_umap_n_neighbors),
            parse_int_grid(args.sweep_umap_components),
            parse_float_grid(args.sweep_umap_min_dists),
            parse_int_grid(args.sweep_min_cluster_sizes),
            parse_optional_int_grid(args.sweep_min_samples),
        )

        for n_neighbors, n_components, min_dist, min_cluster_size, min_samples in grid:
            if n_rows <= 2 or n_rows < min_cluster_size:
                continue

            run_args = copy.copy(args)
            run_args.umap_n_neighbors = min(n_neighbors, max(2, n_rows - 1))
            run_args.umap_components = min(n_components, max(2, n_rows - 1))
            run_args.umap_min_dist = min_dist
            run_args.min_cluster_size = min_cluster_size
            run_args.min_samples = min_samples
            grid_args.append(run_args)

        return grid_args

    for n_clusters in parse_int_grid(args.sweep_n_clusters):
        if n_rows < n_clusters:
            continue

        run_args = copy.copy(args)
        run_args.n_clusters = n_clusters
        grid_args.append(run_args)

    return grid_args


def clustering_param_record(args: argparse.Namespace) -> dict:
    """Return clustering hyperparameters relevant to the current method."""
    record = {
        "cluster_method": args.cluster_method,
        "pca_components": args.pca_components,
        "random_state": args.random_state,
    }

    if args.cluster_method == "umap_hdbscan":
        record.update(
            {
                "umap_n_neighbors": args.umap_n_neighbors,
                "umap_components": args.umap_components,
                "umap_min_dist": args.umap_min_dist,
                "umap_metric": args.umap_metric,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
            }
        )
    else:
        record["n_clusters"] = args.n_clusters

    return record


def add_clustering_metadata(
    assignments: pd.DataFrame,
    model: str,
    actual_layer: int,
    args: argparse.Namespace,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Add run metadata and cluster labels to an assignment dataframe."""
    out = assignments.copy()
    out["model"] = model
    out["actual_layer"] = actual_layer
    out["pooling"] = args.pooling
    out["scope"] = args.scope
    out["cluster_method"] = args.cluster_method
    out["n_clusters"] = args.n_clusters
    out["min_cluster_size"] = args.min_cluster_size
    out["min_samples"] = args.min_samples
    out["umap_n_neighbors"] = getattr(args, "umap_n_neighbors", None)
    out["umap_components"] = getattr(args, "umap_components", None)
    out["umap_min_dist"] = getattr(args, "umap_min_dist", None)
    out["cluster"] = cluster_labels.astype(int)
    out["is_hdbscan_noise"] = out["cluster"].eq(-1)

    return out


def run_single_clustering(
    scoped_df: pd.DataFrame,
    scoped_X: np.ndarray,
    model: str,
    actual_layer: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run one clustering configuration."""
    X_for_cluster, reduction_info = make_cluster_features(scoped_X, args)

    cluster_labels = fit_clusters(
        X_for_cluster,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    assignments = add_clustering_metadata(
        assignments=scoped_df,
        model=model,
        actual_layer=actual_layer,
        args=args,
        cluster_labels=cluster_labels,
    )

    summary = summarize_clusters(assignments, model=model)

    quality = evaluate_clustering_quality(
        X_for_cluster=X_for_cluster,
        labels=cluster_labels,
        assignments_df=assignments,
    )

    diagnostics = {
        "model": model,
        "dataset": args.dataset,
        "layer_label": args.layer_label,
        "actual_layer": actual_layer,
        "scope": args.scope,
        "pooling": args.pooling,
        **clustering_param_record(args),
        **quality,
        **reduction_info,
    }

    return assignments, summary, diagnostics


def fixed_run_output_dir(args: argparse.Namespace, model: str) -> Path:
    """Build output directory for one fixed clustering run."""
    safe_model = safe_model_name(model)

    if args.cluster_method == "umap_hdbscan":
        suffix = f"minc{args.min_cluster_size}_mins{args.min_samples}"
    else:
        suffix = f"k{args.n_clusters}"

    return (
        Path(args.output_dir)
        / PROBE_NAME
        / safe_model
        / args.dataset
        / args.scope
        / args.cluster_method
        / suffix
    )


def sweep_output_dir(args: argparse.Namespace, model: str) -> Path:
    """Build output directory for one sweep."""
    return (
        Path(args.output_dir)
        / PROBE_NAME
        / safe_model_name(model)
        / args.dataset
        / args.scope
        / args.cluster_method
    )


def write_cluster_outputs(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: dict,
    out_dir: Path,
    model: str,
    dataset: str,
    max_examples: int,
    stable_neighbors_df: pd.DataFrame | None = None,
) -> None:
    """Write assignments, summary, diagnostics, nearest neighbors, and prompts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = safe_model_name(model)

    assignments_path = out_dir / f"{safe_model}__cluster_assignments.csv"
    summary_path = out_dir / f"{safe_model}__cluster_summary.csv"
    diagnostics_path = out_dir / f"{safe_model}__cluster_diagnostics.json"

    assignments.to_csv(assignments_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))

    if stable_neighbors_df is not None and len(stable_neighbors_df) > 0:
        stable_neighbors_df.to_csv(
            out_dir / f"{safe_model}__nearest_stable_neighbors_for_prompts.csv",
            index=False,
        )

    write_prompts(
        assignments_df=assignments,
        out_dir=out_dir,
        model=safe_model,
        dataset=dataset,
        max_examples=max_examples,
        stable_neighbors_df=stable_neighbors_df,
    )

    print(f"Wrote assignments: {assignments_path}")
    print(f"Wrote summary:     {summary_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")


def run_for_model(
    model: str,
    pred_df: pd.DataFrame,
    filtered_test_df: pd.DataFrame,
    true_test_mask: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Run one fixed clustering configuration for one model."""
    print("\n" + "=" * 90)
    print(f"Model: {model}")
    print("=" * 90)

    df, X, actual_layer = prepare_model_inputs(
        model=model,
        pred_df=pred_df,
        filtered_test_df=filtered_test_df,
        true_test_mask=true_test_mask,
        args=args,
    )

    scoped_df, scoped_X = select_scope(df, X, scope=args.scope)
    print(f"Rows in scope={args.scope}: {len(scoped_df)}")

    if len(scoped_df) == 0:
        print(f"Skipping {model}: no rows available for scope={args.scope}.")
        return

    assignments, summary, diagnostics = run_single_clustering(
        scoped_df=scoped_df,
        scoped_X=scoped_X,
        model=model,
        actual_layer=actual_layer,
        args=args,
    )

    stable_neighbors = build_nearest_stable_examples(
        assignments_df=assignments,
        full_df=df,
        full_X=X,
        n_neighbors=args.neighbor_examples_per_cluster,
    )

    out_dir = fixed_run_output_dir(args, model)

    write_cluster_outputs(
        assignments=assignments,
        summary=summary,
        diagnostics=diagnostics,
        out_dir=out_dir,
        model=model,
        dataset=args.dataset,
        max_examples=args.max_examples_per_cluster,
        stable_neighbors_df=stable_neighbors,
    )

    print("Diagnostics:", json.dumps(diagnostics, indent=2, sort_keys=True))


def run_sweep_for_model(
    model: str,
    pred_df: pd.DataFrame,
    filtered_test_df: pd.DataFrame,
    true_test_mask: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Run a clustering hyperparameter sweep for one model."""
    print("\n" + "=" * 90)
    print(f"Model: {model}")
    print("=" * 90)

    df, X, actual_layer = prepare_model_inputs(
        model=model,
        pred_df=pred_df,
        filtered_test_df=filtered_test_df,
        true_test_mask=true_test_mask,
        args=args,
    )

    scoped_df, scoped_X = select_scope(df, X, scope=args.scope)
    print(f"Rows in scope={args.scope}: {len(scoped_df)}")

    if len(scoped_df) == 0:
        print(f"Skipping {model}: no rows available for scope={args.scope}.")
        return

    grid = make_sweep_grid(args, n_rows=len(scoped_df))
    if len(grid) == 0:
        print(f"Skipping {model}: no valid sweep settings for n_rows={len(scoped_df)}.")
        return

    safe_model = safe_model_name(model)
    sweep_dir = sweep_output_dir(args, model)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    records = []
    best = None
    best_score = -1e18

    for sweep_id, run_args in enumerate(grid):
        try:
            assignments, summary, diagnostics = run_single_clustering(
                scoped_df=scoped_df,
                scoped_X=scoped_X,
                model=model,
                actual_layer=actual_layer,
                args=run_args,
            )

            diagnostics["sweep_id"] = sweep_id
            diagnostics["status"] = "ok"
            records.append(diagnostics)

            score = float(diagnostics.get("selection_score", -1e9))
            if score > best_score:
                best_score = score
                best = (sweep_id, assignments, summary, diagnostics)

        except Exception as e:
            fail_record = {
                "sweep_id": sweep_id,
                "status": "failed",
                "error": str(e),
                "model": model,
                "dataset": args.dataset,
                "scope": args.scope,
                **clustering_param_record(run_args),
            }
            records.append(fail_record)
            print(f"  Sweep setting {sweep_id} failed: {e}")

    sweep_results_path = sweep_dir / f"{safe_model}__sweep_results.csv"
    sweep_json_path = sweep_dir / f"{safe_model}__sweep_results.jsonl"

    pd.DataFrame(records).to_csv(sweep_results_path, index=False)
    with sweep_json_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    if best is None:
        print(f"No successful clustering settings for {model}. Wrote: {sweep_results_path}")
        return

    best_id, best_assignments, best_summary, best_diagnostics = best
    best_dir = sweep_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_diagnostics = dict(best_diagnostics)
    best_diagnostics["best_sweep_id"] = int(best_id)
    best_diagnostics["selection_rule"] = (
        "maximize selection_score = silhouette - 0.25*noise_frac "
        "- small penalties for tiny/too-few clusters; retraction ARI/NMI not used for selection"
    )

    stable_neighbors = build_nearest_stable_examples(
        assignments_df=best_assignments,
        full_df=df,
        full_X=X,
        n_neighbors=args.neighbor_examples_per_cluster,
    )

    best_assignments_path = best_dir / f"{safe_model}__best_cluster_assignments.csv"
    best_summary_path = best_dir / f"{safe_model}__best_cluster_summary.csv"
    best_diagnostics_path = best_dir / f"{safe_model}__best_cluster_diagnostics.json"

    best_assignments.to_csv(best_assignments_path, index=False)
    best_summary.to_csv(best_summary_path, index=False)
    best_diagnostics_path.write_text(
        json.dumps(best_diagnostics, indent=2, sort_keys=True)
    )

    if len(stable_neighbors) > 0:
        stable_neighbors.to_csv(
            best_dir / f"{safe_model}__nearest_stable_neighbors_for_prompts.csv",
            index=False,
        )

    write_prompts(
        assignments_df=best_assignments,
        out_dir=best_dir,
        model=safe_model,
        dataset=args.dataset,
        max_examples=args.max_examples_per_cluster,
        stable_neighbors_df=stable_neighbors,
    )

    print(f"Wrote sweep results: {sweep_results_path}")
    print(f"Wrote best assignments: {best_assignments_path}")
    print("Best diagnostics:", json.dumps(best_diagnostics, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cities_loc")
    parser.add_argument("--layer-label", default="tasklayer")

    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        default="_gemma-2-9b",
        help="Comma-separated list, 'default', or 'all'.",
    )

    parser.add_argument(
        "--scope",
        choices=["all", "all_rows", "retracted", "never_retracted"],
        default="all",
    )

    parser.add_argument("--pooling", choices=["mean", "last", "first"], default="mean")

    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "agglomerative", "umap_hdbscan"],
        default="kmeans",
    )

    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--umap-components", type=int, default=10)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-examples-per-cluster", type=int, default=30)
    parser.add_argument("--neighbor-examples-per-cluster", type=int, default=15)

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a hyperparameter sweep and save the best clustering.",
    )

    parser.add_argument("--sweep-n-clusters", default="3,4,5,6,8,10,12")
    parser.add_argument("--sweep-umap-n-neighbors", default="5,10,15,30")
    parser.add_argument("--sweep-umap-components", default="2,5,10")
    parser.add_argument("--sweep-umap-min-dists", default="0.0,0.1")
    parser.add_argument("--sweep-min-cluster-sizes", default="3,5,10,20")
    parser.add_argument("--sweep-min-samples", default="none,2,5,10")

    parser.add_argument(
        "--output-dir",
        default=str(project_path("outputs", "retraction_activation_clusters")),
    )

    return parser.parse_args()


def main() -> None:
    """Run clustering for selected models."""
    args = parse_args()

    pred_csv = get_prediction_csv_path(args.dataset, args.layer_label)
    print(f"Loading prediction CSV: {pred_csv}")

    pred_df = pd.read_csv(pred_csv)

    models = parse_models(
        models_arg=args.models,
        pred_df=pred_df,
        layer_label=args.layer_label,
    )

    if len(models) == 0:
        raise ValueError("No models selected or found in prediction table.")

    print(f"Selected models ({len(models)}): {models}")

    _, true_test_mask, filtered_test_df = load_full_and_filtered_test_df(
        dataset=args.dataset,
        model_for_cfg=models[0],
    )

    for model in models:
        try:
            if args.sweep:
                run_sweep_for_model(
                    model=model,
                    pred_df=pred_df,
                    filtered_test_df=filtered_test_df,
                    true_test_mask=true_test_mask,
                    args=args,
                )
            else:
                run_for_model(
                    model=model,
                    pred_df=pred_df,
                    filtered_test_df=filtered_test_df,
                    true_test_mask=true_test_mask,
                    args=args,
                )
        except Exception as e:
            print(f"FAILED model={model}: {e}")


if __name__ == "__main__":
    main()