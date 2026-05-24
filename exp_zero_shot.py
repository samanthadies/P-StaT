"""
exp_zero_shot.py

Experiment script to collect zero-shot ABC (a/b/c) responses
for true test statements, optionally conditioned on a set of
synthetic/fictional/noise/synthetic_fic statements that the model
is told it "believes".
"""

import logging
from pathlib import Path
import hashlib

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from response.collector import MultichoiceLogitCollector
from response.prompt_templates import ABC3Prompt
from utils import get_device, prepare_hf_model

log = logging.getLogger(__name__)


ALL_BENCHMARK_DOMAINS = ["cities_loc", "med_indications", "defs"]


def validate_config(cfg):
    """
    Config validation and device setup.

    :param cfg: config
    :return: None
    """
    assert isinstance(cfg.datasets, (list, type(OmegaConf.create([])))), (
        f"datasets parameter must be a list. Not {type(cfg.datasets)}"
    )
    assert len(cfg.datasets) > 0, "At least one dataset must be selected."

    assert len(cfg.enum_list) == 3, "enum_list must have exactly 3 entries (for a/b/c)."

    legal_pert = [0, 1, 2, 3, 4, 5, 6]
    assert cfg.perturbation_type in legal_pert, (
        f"perturbation_type must be one of {legal_pert}, "
        f"got {cfg.perturbation_type}"
    )

    context_size = get_context_size(cfg)
    if context_size is not None:
        assert context_size >= 0, f"context_size must be >= 0, got {context_size}"

    if cfg.device is None:
        OmegaConf.set_struct(cfg, False)
        cfg["device"] = str(get_device())
        OmegaConf.set_struct(cfg, True)


def get_context_size(cfg):
    """
    Resolve the number of context statements to use.

    Prefer cfg.context_size, but fall back to cfg.max_context_statements
    for backward compatibility.

    :param cfg: config
    :return: integer context size or None
    """
    if hasattr(cfg, "context_size") and cfg.context_size is not None:
        return int(cfg.context_size)

    if hasattr(cfg, "max_context_statements") and cfg.max_context_statements is not None:
        return int(cfg.max_context_statements)

    return None


def log_stats(cfg):
    """
    Log info about the config.

    :param cfg: config
    :return: None
    """
    context_size = get_context_size(cfg)

    log.warning(
        f"Collecting ABC (a/b/c) prompt-based scores for: {cfg.model.name} "
        f"(device: {cfg.device})"
    )
    log.warning(f"Datasets: {cfg.datasets}")
    log.warning(f"Perturbation type: {cfg.perturbation_type}")
    log.warning(f"Context size: {context_size}")
    log.warning(f"Output dir base: {cfg.output_dir}")


def coerce_binary_series(series):
    """
    Convert common bool/string/int encodings to integer 0/1.

    :param series: pandas Series
    :return: pandas Series of ints
    """
    if series.dtype == bool:
        return series.astype(int)

    if series.dtype == object:
        normalized = series.astype(str).str.strip().str.lower()
        return normalized.map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            }
        ).fillna(series).astype(int)

    return series.astype(int)


def load_csv(path):
    """
    Load a CSV and normalize common binary columns.

    :param path: path to CSV
    :return: pandas DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    for col in ["correct", "in_train", "in_test", "in_cal"]:
        if col in df.columns:
            df[col] = coerce_binary_series(df[col])

    return df


def load_true_test_statements(dataset):
    """
    Load datasets/{dataset}_true_false.csv and subset to correct==1 & in_test==1.

    :param dataset: dataset tag
    :return: list of statements
    """
    path = Path("datasets") / f"{dataset}_true_false.csv"
    df = load_csv(path)

    mask = (df["correct"] == 1) & (df["in_test"] == 1)
    return df.loc[mask, "statement"].tolist()


def load_true_train_statements(dataset):
    """
    Load datasets/{dataset}_true_false.csv and subset to correct==1 & in_train==1.

    :param dataset: dataset tag
    :return: list of statements
    """
    path = Path("datasets") / f"{dataset}_true_false.csv"
    df = load_csv(path)

    mask = (df["in_train"] == 1) & (df["correct"] == 1)
    return df.loc[mask, "statement"].tolist()


def sample_statements(statements, k, rng_seed, dataset, perturbation_type):
    """
    Deterministically sample up to k statements.

    :param statements: list of candidate statements
    :param k: desired number of statements
    :param rng_seed: base random seed
    :param dataset: dataset name
    :param perturbation_type: perturbation type
    :return: sampled list of statements
    """
    if k is None or k <= 0 or len(statements) <= k:
        return statements

    key = f"{dataset}::{perturbation_type}::{k}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    seed = int(rng_seed) + int(h[:8], 16)
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(statements), size=k, replace=False)
    idx = np.sort(idx)
    return [statements[i] for i in idx]


def load_noise_context(dataset, rng_seed=42, total_k=100):
    """
    Load true statements from domains other than the core dataset to serve
    as the non-semantic noise condition.

    :param dataset: dataset tag
    :param rng_seed: base random seed
    :param total_k: total number of context statements to sample
    :return: list of statements
    """
    other_domains = [d for d in ALL_BENCHMARK_DOMAINS if d != dataset]
    if len(other_domains) == 0:
        raise ValueError("No other domains available for noise context.")

    h = hashlib.sha1(dataset.encode("utf-8")).hexdigest()
    seed = int(rng_seed) + int(h[:8], 16)
    rng = np.random.default_rng(seed)

    base_n = total_k // len(other_domains)
    remainder = total_k % len(other_domains)

    sampled = []
    for i, od in enumerate(other_domains):
        n_for_domain = base_n + (1 if i < remainder else 0)
        if n_for_domain == 0:
            continue

        pool = load_true_train_statements(od)
        if len(pool) == 0:
            raise ValueError(
                f"No True-train statements available for noise context in domain '{od}'."
            )

        replace = len(pool) < n_for_domain
        idx = rng.choice(len(pool), size=n_for_domain, replace=replace)
        sampled.extend([pool[j] for j in idx])

    rng.shuffle(sampled)
    return sampled


def load_context_statements(dataset, perturbation_type, rng_seed, context_size):
    """
    Depending on perturbation_type, load an additional CSV and subset appropriately.

    :param dataset: dataset tag
    :param perturbation_type: perturbation type
    :param rng_seed: base random seed
    :param context_size: number of context statements for K ablation
    :return: additional perturbation statements
    """
    if perturbation_type == 0:
        return []

    if perturbation_type == 1:
        fname = f"{dataset}_synthetic.csv"
        df = load_csv(Path("datasets") / fname)
        mask = df["in_train"] == 1
        statements = df.loc[mask, "statement"].tolist()

    elif perturbation_type == 2:
        fname = f"{dataset}_fictional.csv"
        df = load_csv(Path("datasets") / fname)
        mask = df["in_train"] == 1
        statements = df.loc[mask, "statement"].tolist()

    elif perturbation_type == 3:
        fname = f"{dataset}_fictional.csv"
        df = load_csv(Path("datasets") / fname)
        mask = (df["in_train"] == 1) & (df["correct"] == 1)
        statements = df.loc[mask, "statement"].tolist()

    elif perturbation_type == 4:
        if context_size is None:
            context_size = 100
        statements = load_noise_context(
            dataset,
            rng_seed=rng_seed,
            total_k=int(context_size),
        )

    elif perturbation_type == 5:
        fname = f"{dataset}_true_false.csv"
        df = load_csv(Path("datasets") / fname)
        mask = (df["in_train"] == 1) & (df["correct"] == 1)
        statements = df.loc[mask, "statement"].tolist()

    elif perturbation_type == 6:
        fname = f"{dataset}_synthetic_fic.csv"
        df = load_csv(Path("datasets") / fname)
        mask = df["in_train"] == 1
        statements = df.loc[mask, "statement"].tolist()

    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    statements = sample_statements(
        statements=statements,
        k=context_size,
        rng_seed=rng_seed,
        dataset=dataset,
        perturbation_type=perturbation_type,
    )

    return statements


def get_save_dir(cfg, dataset):
    """
    Construct the output directory.

    Baseline task 0 is saved as:
        outputs/probes/zero_shot/{model}/{dataset}/task-0/

    Perturbation tasks are saved as:
        outputs/probes/zero_shot/{model}/{dataset}/K-{K}/task-{task}/

    :param cfg: config
    :param dataset: dataset name
    :return: Path
    """
    task = int(cfg.perturbation_type)
    base = Path(cfg.output_dir) / dataset

    if task == 0:
        return base / "task-0"

    context_size = get_context_size(cfg)
    if context_size is None:
        raise ValueError(
            "context_size must be set for perturbation tasks 1-6 so outputs "
            "can be saved under K-{context_size}/task-{task}."
        )

    return base / f"K-{int(context_size)}" / f"task-{task}"


def tokenize(batch, tokenizer, cfg):
    """
    Dispatch to default vs instruct tokenization based on cfg.model.instruct.

    :param batch: batch of statements
    :param tokenizer: tokenizer
    :param cfg: config
    :return: sequences from tokenizer
    """
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    else:
        return default_tokenize(batch, tokenizer)


def default_tokenize(batch, tokenizer):
    """
    Standard HF tokenization for a list of strings.

    :param batch: batch of statements
    :param tokenizer: default tokenizer
    :return: sequences from tokenizer
    """
    input_seqs = tokenizer(batch, return_tensors="pt", padding="longest")
    return input_seqs


def instruct_tokenize(batch, tokenizer, cfg):
    """
    Tokenizer for chat/instruction-tuned models.

    :param batch: batch of statements
    :param tokenizer: tokenizer
    :param cfg: config
    :return: sequences from tokenizer
    """
    batch = tokenizer.apply_chat_template(
        batch,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if cfg.model["end_token"] == "auto":
        end_token = tokenizer.eos_token
    else:
        end_token = cfg.model["end_token"]

    text_batch = [
        b.strip(" ").rstrip(end_token).strip(" ") + " " for b in batch
    ]
    input_seqs = tokenizer(text_batch, return_tensors="pt", padding="longest")
    return input_seqs


@hydra.main(version_base=None, config_path="configs", config_name="zero_shot")
def main(cfg: DictConfig):
    """
    Compute zero-shot ABC (a/b/c) scores for each true test statement in each
    dataset listed in cfg.datasets, optionally conditioned on a set of
    context statements that the model is told it believes.
    """
    validate_config(cfg)
    log_stats(cfg)

    rng_seed = getattr(cfg, "random_seed", 42)
    np.random.seed(rng_seed)

    model, tokenizer = prepare_hf_model(cfg)
    torch.set_grad_enabled(False)

    if cfg.model["instruct"]:
        mode = "instruct"
    else:
        mode = "default"

    if cfg.question_type != "multichoice":
        raise ValueError("This script is designed for 'multichoice' (a/b/c) only.")

    enum_list = [str(e) for e in cfg.enum_list]

    prompt_f = ABC3Prompt(
        prompt_type=mode,
        enumeration=enum_list,
        system_role=cfg.model["system_role"],
        user_role=cfg.model["user_role"],
        assist_role=cfg.model["assist_role"],
    )
    collector = MultichoiceLogitCollector(tokenizer, prompt_f)

    context_size = get_context_size(cfg)

    for dataset in cfg.datasets:
        log.warning(f"=== Processing dataset: {dataset} ===")

        test_statements = load_true_test_statements(dataset)
        context_statements = load_context_statements(
            dataset=dataset,
            perturbation_type=cfg.perturbation_type,
            rng_seed=rng_seed,
            context_size=context_size,
        )

        log.warning(
            f"Loaded {len(test_statements)} test statements and "
            f"{len(context_statements)} context statements for dataset '{dataset}'."
        )

        if len(test_statements) == 0:
            log.warning("No test statements found for this dataset; skipping.")
            continue

        if len(context_statements) == 0:
            def make_block(t: str) -> str:
                return f"Is the following statement correct?\n{t}"
        else:
            context_block = (
                "Let's say that you believe these statements:\n"
                + "\n".join(context_statements)
            )

            def make_block(t: str) -> str:
                return f"{context_block}\nIs the following statement correct?\n{t}"

        prompts = [prompt_f(make_block(t)) for t in test_statements]

        if cfg.model["instruct"]:
            example_str = tokenizer.apply_chat_template(
                prompts[0],
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            example_str = prompts[0]

        log.warning(f"Example prompt for dataset '{dataset}':\n---\n{example_str}\n---")

        n = len(prompts)
        n_batches = int(np.ceil(n / cfg.batch_size))
        n_batches = max(n_batches, 1)

        if cfg.limit_batches > 0:
            batches = np.array_split(np.array(prompts, dtype=object), n_batches)[
                : cfg.limit_batches
            ]
        else:
            batches = np.array_split(np.array(prompts, dtype=object), n_batches)

        __log_intervals = np.linspace(0, len(batches), 5, dtype=int)

        all_scores = []
        all_preds = []

        for i, batch in enumerate(tqdm(batches, total=len(batches))):
            batch_list = batch.tolist()

            tokenized = tokenize(batch_list, tokenizer, cfg)
            input_ids = tokenized["input_ids"].to(cfg.device)
            input_att = tokenized["attention_mask"].to(cfg.device)

            if i in __log_intervals:
                status = i / len(batches)
                log.warning(
                    f"(BATCH) [{dataset}] Processed {status:.2%} of the statements"
                )

            out = model(
                input_ids,
                attention_mask=input_att,
                use_cache=False,
            ).logits
            last_logits = out[:, -1]

            probs, pred_idx = collector.collect_proba_and_argmax(last_logits)
            all_scores.append(probs)
            all_preds.append(pred_idx)

        scores = torch.cat(all_scores, dim=0)
        preds = torch.cat(all_preds, dim=0)

        save_dir = get_save_dir(cfg, dataset)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "scores.npy", scores.detach().cpu().float().numpy())
        np.save(save_dir / "preds.npy", preds.detach().cpu().numpy())

        with open(save_dir / "statements.txt", "w") as f:
            for s in test_statements:
                f.write(s.replace("\n", " ") + "\n")

        if int(cfg.perturbation_type) != 0:
            with open(save_dir / "context_statements.txt", "w") as f:
                for s in context_statements:
                    f.write(s.replace("\n", " ") + "\n")

        log.warning(
            f"(BATCH) [{dataset}] Processed 100.00% of the statements; saved to {save_dir}\n\n"
        )


if __name__ == "__main__":
    main()