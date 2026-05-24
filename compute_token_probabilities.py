"""
compute_token_probabilities.py

Compute and save reusable token-level next-token probabilities/log-probabilities/
surprisal values for statement datasets.

"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import get_device, prepare_hf_model

log = logging.getLogger(__name__)


STATEMENT_TYPES = {
    "true_false": "{dataset}_true_false.csv",
    "fictional": "{dataset}_fictional.csv",
    "synthetic": "{dataset}_synthetic.csv",
    "synthetic_fic": "{dataset}_synthetic_fic.csv",
}

DEFAULT_CUE = "In a fictional setting,"
OBJECT_COLS = ["object_1", "object_2"]


def project_path(*parts: str) -> Path:
    """
    Resolve a path relative to the original project root, not Hydra's run dir.

    Returns:
        Project-root-relative path.
    """
    return Path(get_original_cwd()).joinpath(*parts)


def load_hf_token(cfg: DictConfig) -> None:
    """
    Populate cfg.model.token from an environment variable or token file
    before calling prepare_hf_model(cfg).

    Priority:
      1. existing cfg.model.token if set and not placeholder
      2. HF_TOKEN environment variable
      3. cfg.hf_token_path file, if provided
    """
    token = None

    if hasattr(cfg.model, "token"):
        cfg_token = cfg.model.token
        if cfg_token not in [None, "", "INSERT_ACCESS_TOKEN", "???"]:
            token = cfg_token

    if token is None:
        token = os.environ.get("HF_TOKEN", None)

    if token is None and hasattr(cfg, "hf_token_path") and cfg.hf_token_path is not None:
        token_path = Path(cfg.hf_token_path)
        if not token_path.is_absolute():
            token_path = project_path(str(token_path))

        if token_path.exists():
            token = token_path.read_text().strip()

    if token is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.token = token
        OmegaConf.set_struct(cfg, True)
    else:
        log.warning("No HF token found in cfg.model.token, HF_TOKEN, or cfg.hf_token_path.")


def validate_config(cfg: DictConfig) -> None:
    """
    Validate config and populate device if needed.
    """
    assert isinstance(cfg.datasets, (list, type(OmegaConf.create([]))))
    assert len(cfg.datasets) > 0

    if cfg.device is None:
        OmegaConf.set_struct(cfg, False)
        cfg["device"] = str(get_device())
        OmegaConf.set_struct(cfg, True)


def find_case_insensitive_span(
    text: str,
    query: str,
    search_start: int = 0,
) -> Optional[tuple[int, int]]:
    """
    Find query inside text using case-insensitive matching.

    Returns:
        Character span as (start, end), or None if not found.
    """
    if query is None or pd.isna(query):
        return None

    query = str(query).strip()
    if query == "":
        return None

    text_lower = text.lower()
    query_lower = query.lower()

    start = text_lower.find(query_lower, search_start)
    if start == -1:
        return None

    end = start + len(query)
    return start, end


def find_object_spans(
    scoring_text: str,
    row: pd.Series,
    target_start_char: int,
) -> list[tuple[int, int, str]]:
    """
    Find object_1 and object_2 spans inside scoring_text.

    For cued statements, target_start_char skips the cue so we search only
    inside the original statement portion.

    Returns:
        A list of (start, end, object_col) spans.
    """
    spans = []

    for col in OBJECT_COLS:
        if col not in row.index:
            continue

        span = find_case_insensitive_span(
            text=scoring_text,
            query=row[col],
            search_start=target_start_char,
        )

        if span is None:
            log.warning(
                f"Could not find {col}={repr(row[col])} in "
                f"scoring_text={repr(scoring_text)}"
            )
            continue

        start, end = span
        spans.append((start, end, col))

    return spans


def token_overlaps_span(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> bool:
    """
    Return True if a token character span overlaps a target character span.
    """
    if token_end <= token_start:
        return False

    return token_start < span_end and token_end > span_start


def load_statement_df(dataset: str, statement_type: str) -> pd.DataFrame:
    """
    Load one statement file and add scoring metadata.

    Returns:
        DataFrame with scoring_text, condition, target_start_char, and object spans.
    """
    fname = STATEMENT_TYPES[statement_type].format(dataset=dataset)
    path = project_path("datasets", fname)

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path).copy()
    df["dataset"] = dataset
    df["statement_type"] = statement_type

    if "statement" not in df.columns:
        raise ValueError(f"{path} does not contain a 'statement' column")

    df["scoring_text"] = df["statement"].astype(str)
    df["target_start_char"] = 0

    if statement_type == "true_false" and "correct" in df.columns:
        df["condition"] = np.where(df["correct"] == 1, "true", "false")
    else:
        df["condition"] = statement_type

    object_span_records = []
    object_span_strings = []
    n_found_objects = []

    for _, row in df.iterrows():
        spans = find_object_spans(
            scoring_text=row["scoring_text"],
            row=row,
            target_start_char=int(row["target_start_char"]),
        )

        object_span_records.append(spans)
        object_span_strings.append(
            ";".join([f"{col}:{start}-{end}" for start, end, col in spans])
        )
        n_found_objects.append(len(spans))

    df["object_spans"] = object_span_records
    df["object_span_string"] = object_span_strings
    df["n_found_objects"] = n_found_objects

    missing_objects = df[df["n_found_objects"] < len(OBJECT_COLS)]
    if len(missing_objects) > 0:
        log.warning(
            f"{path}: {len(missing_objects)}/{len(df)} rows were missing at least "
            f"one object span."
        )

    return df


def load_dataset_statement_type_df(dataset: str, statement_type: str) -> pd.DataFrame:
    """
    Load one dataset/statement_type pair.

    Returns:
        DataFrame for exactly one statement file, with row_id added.
    """
    df = load_statement_df(dataset, statement_type)
    df.insert(0, "row_id", np.arange(len(df)))

    return df


def tokenize_texts(texts: list[str], tokenizer, cfg: DictConfig):
    """
    Tokenize plain text and return offset mappings.

    Offset mappings allow us to save token-level character spans and construct
    downstream masks for cue-excluded statement tokens and object tokens.
    """
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(cfg, "max_length", 256),
        return_offsets_mapping=True,
    )


def get_token_object_flags(
    token_start: int,
    token_end: int,
    object_spans: list[tuple[int, int, str]],
) -> dict[str, bool | str]:
    """
    Determine whether a token overlaps object_1 and/or object_2.

    Returns:
        Dictionary with object mask flags.
    """
    overlaps_object_1 = False
    overlaps_object_2 = False

    for span_start, span_end, col in object_spans:
        overlaps = token_overlaps_span(
            token_start=token_start,
            token_end=token_end,
            span_start=span_start,
            span_end=span_end,
        )

        if not overlaps:
            continue

        if col == "object_1":
            overlaps_object_1 = True
        elif col == "object_2":
            overlaps_object_2 = True

    is_object_token = overlaps_object_1 or overlaps_object_2

    if overlaps_object_1 and overlaps_object_2:
        object_col = "both"
    elif overlaps_object_1:
        object_col = "object_1"
    elif overlaps_object_2:
        object_col = "object_2"
    else:
        object_col = "none"

    return {
        "is_object_1_token": overlaps_object_1,
        "is_object_2_token": overlaps_object_2,
        "is_object_token": is_object_token,
        "object_col": object_col,
    }


@torch.no_grad()
def batch_token_surprisal(
    batch_df: pd.DataFrame,
    model,
    tokenizer,
    cfg: DictConfig,
) -> pd.DataFrame:
    """
    Compute token-level next-token quantities for one batch.

    Each output row corresponds to an input token position. For causal LMs,
    the probability for token position t is computed from logits at position t-1,
    i.e., p(token_t | tokens_<t). The first token is retained but has NaN values
    for next-token probability/logprob/surprisal.

    Returns:
        Token-level DataFrame for the batch.
    """
    texts = batch_df["scoring_text"].astype(str).tolist()

    tokenized = tokenize_texts(texts, tokenizer, cfg)

    offset_mapping = tokenized.pop("offset_mapping")
    input_ids = tokenized["input_ids"].to(cfg.device)
    attention_mask = tokenized["attention_mask"].to(cfg.device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = out.logits

    # Log-probabilities for next-token prediction.
    # shift_logprobs[:, j, :] predicts shift_labels[:, j] = input_ids[:, j + 1].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_logprobs = torch.log_softmax(shift_logits, dim=-1)

    observed_shift_logprobs = torch.gather(
        shift_logprobs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    observed_shift_probs = observed_shift_logprobs.exp()
    observed_shift_surprisals = -observed_shift_logprobs

    records = []

    offsets = offset_mapping.cpu().tolist()
    input_ids_cpu = input_ids.cpu().tolist()
    attention_cpu = attention_mask.cpu().tolist()

    observed_shift_logprobs_cpu = observed_shift_logprobs.cpu().float().numpy()
    observed_shift_probs_cpu = observed_shift_probs.cpu().float().numpy()
    observed_shift_surprisals_cpu = observed_shift_surprisals.cpu().float().numpy()

    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    for batch_idx, (_, row) in enumerate(batch_df.iterrows()):
        row_id = int(row["row_id"])
        scoring_text = str(row["scoring_text"])
        target_start_char = int(row["target_start_char"])
        object_spans = row["object_spans"]

        for token_position, token_id in enumerate(input_ids_cpu[batch_idx]):
            attn = int(attention_cpu[batch_idx][token_position])
            token_start, token_end = offsets[batch_idx][token_position]

            is_padding = attn == 0
            is_special_or_empty = token_end <= token_start
            is_special_token = (not is_padding) and token_id in special_ids

            token_text = tokenizer.decode(
                [token_id],
                clean_up_tokenization_spaces=False,
            )

            # Cue-excluded statement scoring mask.
            # For uncued examples, target_start_char=0, so this includes all real tokens.
            is_target_statement_token = (
                not is_padding
                and not is_special_or_empty
                and token_end > target_start_char
            )

            is_cue_token = (
                not is_padding
                and not is_special_or_empty
                and token_end <= target_start_char
            )

            object_flags = get_token_object_flags(
                token_start=token_start,
                token_end=token_end,
                object_spans=object_spans,
            )

            # The first token has no within-sequence previous-token prediction.
            if token_position == 0 or is_padding:
                token_logprob = np.nan
                token_prob = np.nan
                token_surprisal = np.nan
            else:
                shift_idx = token_position - 1
                token_logprob = observed_shift_logprobs_cpu[batch_idx, shift_idx]
                token_prob = observed_shift_probs_cpu[batch_idx, shift_idx]
                token_surprisal = observed_shift_surprisals_cpu[batch_idx, shift_idx]

            records.append(
                {
                    "row_id": row_id,
                    "dataset": row["dataset"],
                    "statement_type": row["statement_type"],
                    "condition": row["condition"],
                    "token_position": token_position,
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "char_start": int(token_start),
                    "char_end": int(token_end),
                    "attention_mask": attn,
                    "is_padding": bool(is_padding),
                    "is_special_or_empty": bool(is_special_or_empty),
                    "is_special_token": bool(is_special_token),
                    "is_cue_token": bool(is_cue_token),
                    "is_target_statement_token": bool(is_target_statement_token),
                    "is_object_1_token": bool(object_flags["is_object_1_token"]),
                    "is_object_2_token": bool(object_flags["is_object_2_token"]),
                    "is_object_token": bool(object_flags["is_object_token"]),
                    "object_col": object_flags["object_col"],
                    "token_logprob": token_logprob,
                    "token_prob": token_prob,
                    "token_surprisal": token_surprisal,
                    "scoring_text": scoring_text,
                }
            )

    return pd.DataFrame(records)


def summarize_statement_surprisal(token_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate token-level scores to cue-excluded statement-level scores.

    Returns:
        Statement-level summary DataFrame.
    """
    scoring_df = token_df[
        token_df["is_target_statement_token"]
        & ~token_df["is_padding"]
        & token_df["token_surprisal"].notna()
    ].copy()

    summary = (
        scoring_df
        .groupby(["dataset", "statement_type", "condition", "row_id"], as_index=False)
        .agg(
            n_tokens=("token_surprisal", "count"),
            total_nll=("token_surprisal", "sum"),
            mean_surprisal=("token_surprisal", "mean"),
        )
    )

    summary["perplexity"] = np.exp(summary["mean_surprisal"].clip(upper=50))

    return summary


def summarize_object_surprisal(token_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate token-level scores to object-level scores.

    Returns:
        Object-level summary DataFrame.
    """
    scoring_df = token_df[
        token_df["is_object_token"]
        & ~token_df["is_padding"]
        & token_df["token_surprisal"].notna()
    ].copy()

    summary = (
        scoring_df
        .groupby(["dataset", "statement_type", "condition", "row_id"], as_index=False)
        .agg(
            n_object_tokens=("token_surprisal", "count"),
            object_total_nll=("token_surprisal", "sum"),
            object_mean_surprisal=("token_surprisal", "mean"),
        )
    )

    summary["object_perplexity"] = np.exp(
        summary["object_mean_surprisal"].clip(upper=50)
    )

    return summary


def save_outputs(
    df: pd.DataFrame,
    token_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save token-level scores, statement metadata, and basic aggregate summaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    token_path = output_dir / "token_scores.parquet"
    metadata_path = output_dir / "statement_metadata.csv"
    statement_summary_path = output_dir / "statement_surprisal_summary.csv"
    object_summary_path = output_dir / "object_surprisal_summary.csv"

    token_df.to_parquet(token_path, index=False)

    metadata_cols = [
        "row_id",
        "dataset",
        "statement_type",
        "condition",
        "statement",
        "scoring_text",
        "target_start_char",
        "object_span_string",
        "n_found_objects",
    ]

    optional_cols = [
        "object_1",
        "object_2",
        "correct_object_2",
        "correct",
        "negation",
        "fictional_object",
        "real_object",
        "fake_object",
        "category",
        "in_train",
        "in_test",
        "in_cal",
        "cued_statement",
    ]

    metadata_cols = [
        col for col in metadata_cols + optional_cols
        if col in df.columns
    ]

    df[metadata_cols].to_csv(metadata_path, index=False)

    statement_summary = summarize_statement_surprisal(token_df)
    statement_summary.to_csv(statement_summary_path, index=False)

    object_summary = summarize_object_surprisal(token_df)
    object_summary.to_csv(object_summary_path, index=False)

    log.warning(f"Saved token-level scores to {token_path}")
    log.warning(f"Saved statement metadata to {metadata_path}")
    log.warning(f"Saved statement summary to {statement_summary_path}")
    log.warning(f"Saved object summary to {object_summary_path}")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="token_probabilities",
)
def main(cfg: DictConfig):
    validate_config(cfg)
    load_hf_token(cfg)

    log.warning(f"Computing token-level surprisal for model: {cfg.model.name}")
    log.warning(f"Datasets: {cfg.datasets}")
    log.warning(f"Device: {cfg.device}")
    log.warning(f"Original project root: {get_original_cwd()}")

    model, tokenizer = prepare_hf_model(cfg)

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "This script requires a fast tokenizer because it uses "
            "offset_mapping to identify cue/object/token character spans."
        )

    model.eval()
    torch.set_grad_enabled(False)

    output_root = project_path(getattr(cfg, "output_dir", "outputs/perplexity"))

    for dataset in cfg.datasets:
        log.warning(f"=== Dataset: {dataset} ===")
    
        for statement_type in STATEMENT_TYPES:
            output_dataset_name = f"{dataset}_{statement_type}"
    
            log.warning(f"--- Statement type: {statement_type} ---")
            log.warning(f"Output dataset name: {output_dataset_name}")
    
            try:
                df = load_dataset_statement_type_df(
                    dataset=dataset,
                    statement_type=statement_type,
                )
            except FileNotFoundError:
                log.warning(f"Skipping missing file for {dataset}/{statement_type}")
                continue
    
            batch_size = int(getattr(cfg, "batch_size", 8))
            token_dfs = []
    
            for start in tqdm(range(0, len(df), batch_size)):
                batch_df = df.iloc[start : start + batch_size].copy()
    
                token_dfs.append(
                    batch_token_surprisal(
                        batch_df=batch_df,
                        model=model,
                        tokenizer=tokenizer,
                        cfg=cfg,
                    )
                )
    
            token_df = pd.concat(token_dfs, ignore_index=True)
    
            output_dir = output_root / cfg.model.name / output_dataset_name
            save_outputs(
                df=df,
                token_df=token_df,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()