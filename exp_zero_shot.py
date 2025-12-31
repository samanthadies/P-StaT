"""
zero_shot.py

Experiment script to collect zero-shot ABC (a/b/c) responses
for true test statements, optionally conditioned on a set of
synthetic/fictional statements that the model is told it
"believes".

"""

import logging
import os
from pathlib import Path

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


def validate_config(cfg):
    """
    Config validation and device setup.

    :param cfg: config
    :return: None
    """
    # datasets must be a list (Hydra ListConfig or Python list)
    assert isinstance(cfg.datasets, (list, type(OmegaConf.create([])))), (
        f"datasets parameter must be a list. Not {type(cfg.datasets)}"
    )
    assert len(cfg.datasets) > 0, "At least one dataset must be selected."

    # enum_list should be length 3 for a/b/c
    assert len(cfg.enum_list) == 3, "enum_list must have exactly 3 entries (for a/b/c)."

    legal_pert = [0, 1, 2, 3]
    assert cfg.perturbation_type in legal_pert, (
        f"perturbation_type must be one of {legal_pert}, "
        f"got {cfg.perturbation_type}"
    )

    # max_context_statements: optional but if present must be non-negative
    if hasattr(cfg, "max_context_statements") and cfg.max_context_statements is not None:
        assert cfg.max_context_statements >= 0, (
            "max_context_statements must be >= 0 "
            f"(got {cfg.max_context_statements})"
        )

    if cfg.device is None:
        OmegaConf.set_struct(cfg, False)  # Allow overriding
        cfg["device"] = str(get_device())  # auto-select device
        OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    """
    Log info about the config.

    :param cfg: config
    :return: None
    """
    log.warning(
        f"Collecting ABC (a/b/c) prompt-based scores for: {cfg.model.name} "
        f"(device: {cfg.device})"
    )
    log.warning(f"Datasets: {cfg.datasets}")
    log.warning(f"Perturbation type: {cfg.perturbation_type}")
    log.warning(f"Output dir base: {cfg.output_dir}")
    if hasattr(cfg, "max_context_statements") and cfg.max_context_statements is not None:
        log.warning(f"max_context_statements: {cfg.max_context_statements}")
    else:
        log.warning("max_context_statements: (not set, will use all context statements)")


def load_true_test_statements(dataset):
    """
    Load datasets/{dataset}_true_false.csv and subset to correct==1 & in_test==1.

    :param dataset: dataset tag
    :return: list of statements
    """
    path = Path("datasets") / f"{dataset}_true_false.csv"
    if not path.exists():
        raise FileNotFoundError(f"True/false CSV not found: {path}")
    df = pd.read_csv(path)
    mask = (df["correct"] == 1) & (df["in_test"] == 1)
    return df.loc[mask, "statement"].tolist()


def load_context_statements(dataset, perturbation_type):
    """
    Depending on perturbation_type, load an additional CSV and subset appropriately.

    :param dataset: dataset tag
    :param perturbation_type: perturbation type
    :return: additional perturbation statements
    """
    if perturbation_type == 0:
        return []

    if perturbation_type == 1:
        fname = f"{dataset}_synthetic.csv"
        df = pd.read_csv(Path("datasets") / fname)
        mask = df["in_train"] == 1

    elif perturbation_type == 2:
        fname = f"{dataset}_fictional.csv"
        df = pd.read_csv(Path("datasets") / fname)
        mask = df["in_train"] == 1

    elif perturbation_type == 3:
        fname = f"{dataset}_fictional.csv"
        df = pd.read_csv(Path("datasets") / fname)
        mask = (df["in_train"] == 1) & (df["correct"] == 1)

    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    return df.loc[mask, "statement"].tolist()


def tokenize(batch, tokenizer, cfg):
    """
    Dispatch to default vs instruct tokenization based on cfg.model.instruct

    :param batch: batch of statements
    :param tokenizer: tokenizer
    :param cfg: config
    :return: sequences from tokenizer
    """
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    else:
        return default_tokenize(batch, tokenizer, cfg)


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
def main(cfg):
    """
    Compute zero-shot ABC (a/b/c) scores for each true test statement in each
    dataset listed in cfg.datasets, optionally conditioned on a set of
    "perturbed" statements (synthetic/fictional) that the model is told it
    believes.
    """
    validate_config(cfg)
    log_stats(cfg)

    # Make RNG reproducible
    rng_seed = getattr(cfg, "random_seed", 42)
    np.random.seed(rng_seed)

    model, tokenizer = prepare_hf_model(cfg)
    torch.set_grad_enabled(False)

    # 1. Prepare prompt template & collector
    if cfg.model["instruct"]:
        mode = "instruct"
    else:
        mode = "default"

    if cfg.question_type != "multichoice":
        raise ValueError("This script is designed for 'multichoice' (a/b/c) only.")

    # use enum_list from config (e.g., ["a", "b", "c"])
    enum_list = [str(e) for e in cfg.enum_list]

    prompt_f = ABC3Prompt(
        prompt_type=mode,
        enumeration=enum_list,
        system_role=cfg.model["system_role"],
        user_role=cfg.model["user_role"],
        assist_role=cfg.model["assist_role"],
    )
    collector = MultichoiceLogitCollector(tokenizer, prompt_f)

    # 2. Loop over datasets
    for dataset in cfg.datasets:
        log.warning(f"=== Processing dataset: {dataset} ===")

        # 2a. Load statements
        test_statements = load_true_test_statements(dataset)
        context_statements = load_context_statements(dataset, cfg.perturbation_type)

        log.warning(
            f"Loaded {len(test_statements)} test statements and "
            f"{len(context_statements)} context statements for dataset '{dataset}'."
        )

        if len(test_statements) == 0:
            log.warning("No test statements found for this dataset; skipping.")
            continue

        # 2b. Optionally subsample context statements to max_context_statements
        if len(context_statements) > 0:
            max_ctx = getattr(cfg, "max_context_statements", None)
            if max_ctx is not None and max_ctx > 0 and len(context_statements) > max_ctx:
                orig_n = len(context_statements)
                # Use a dedicated RNG for clarity
                rng = np.random.default_rng(rng_seed)
                idx = rng.choice(orig_n, size=max_ctx, replace=False)
                # Sort indices so the sampled block has a stable order
                idx = np.sort(idx)
                context_statements = [context_statements[i] for i in idx]
                log.warning(
                    f"Subsampled context statements from {orig_n} to "
                    f"{len(context_statements)} using seed={rng_seed}."
                )

        # 3. Build the *full text block* that ABC3Prompt will use
        if len(context_statements) == 0:
            # Baseline case (perturbation_type == 0; "none")
            def make_block(t: str) -> str:
                return f"Is the following statement correct?\n{t}"
        else:
            # Perturbed cases; use (possibly subsampled) context statements as a block.
            context_block = (
                "Let's say that you believe these statements:\n"
                + "\n".join(context_statements)
            )

            def make_block(t: str) -> str:
                return f"{context_block}\nIs the following statement correct?\n{t}"

        # 4. Assemble prompts in order
        prompts = [prompt_f(make_block(t)) for t in test_statements]

        # Show one example prompt
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

        # 5. Batch setup
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

        # 6. Main loop: tokenize, run model, collect probs + argmax
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
            last_logits = out[:, -1]  # (batch_size, vocab_size)

            probs, pred_idx = collector.collect_proba_and_argmax(last_logits)
            all_scores.append(probs)
            all_preds.append(pred_idx)

        scores = torch.cat(all_scores, dim=0)  # (N, 4) --> [a, b, c, else]
        preds = torch.cat(all_preds, dim=0)    # (N,) --> 0/1/2

        # 7. Save outputs for this dataset
        safe_pert = cfg.perturbation_type
        save_dir = os.path.join(cfg.output_dir, dataset, safe_pert)
        os.makedirs(save_dir, exist_ok=True)

        np.save(
            os.path.join(save_dir, "scores.npy"),
            scores.cpu().float().numpy(),
        )
        np.save(
            os.path.join(save_dir, "preds.npy"),
            preds.cpu().numpy(),
        )

        with open(os.path.join(save_dir, "statements.txt"), "w") as f:
            for s in test_statements:
                f.write(s.replace("\n", " ") + "\n")

        log.warning(
            f"(BATCH) [{dataset}] Processed 100.00% of the statements; "
            f"saved to {save_dir}\n\n"
        )


if __name__ == "__main__":
    main()
