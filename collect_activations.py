"""
collect_activations.py

Generate and save hidden activations from a HuggingFace model
for one or more text datasets.
"""

import logging
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm

from utils import get_device, prepare_hf_model, load_statements, return_layers


log = logging.getLogger(__name__)

# Store activations as float32.
# This avoids np.float16 overflow, which can create inf values for large models.
ACTIVATION_DTYPE = np.float32
MEMMAP_DTYPE = "float32"


def validate_config(cfg):
    """
    Verifies the config file has all the necessary fields.

    Args:
        cfg: config file

    Returns:
        None
    """
    assert cfg.agg in [
        "last",
        "mean",
        "max",
        "full",
    ], "Aggregation type must be either 'last', 'mean', 'max', or 'full'."

    assert len(cfg.layers) > 0, "At least one layer must be selected."

    assert type(cfg.datasets) == list or type(cfg.datasets).__name__ == "ListConfig", (
        f"Datasets must be a list. Not {type(cfg.datasets)}"
    )

    assert len(cfg.datasets) > 0, "At least one dataset must be selected."

    if cfg.device is None:
        OmegaConf.set_struct(cfg, False)
        cfg["device"] = str(get_device())
        OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    """
    Prints initial debugging info.

    Args:
        cfg: config file

    Returns:
        None
    """
    log.warning(f"Collecting activations for: {cfg.model.name} (device: {cfg.device})")
    log.warning(f"Max length of the input sequences: {cfg.max_length}")
    log.warning(f"Activation storage dtype: {MEMMAP_DTYPE}")


def load_statements_from_original_cwd(dataset: str, original_cwd: Path):
    """
    Load statements.

    Args:
        dataset: Dataset name.
        original_cwd: Hydra original working directory.

    Returns:
        Loaded statements.
    """
    old_cwd = Path.cwd()
    try:
        os.chdir(original_cwd)
        statements = load_statements(dataset)
    finally:
        os.chdir(old_cwd)

    return statements


def tokenize(batch, tokenizer, cfg):
    """
    Tokenizes the statements based on the type of model.

    Args:
        batch: batch of statements
        tokenizer: HuggingFace tokenizer
        cfg: config file

    Returns:
        Tokenized sequence.
    """
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    return default_tokenize(batch, tokenizer, cfg)


def default_tokenize(batch, tokenizer, cfg):
    """
    Tokenizer for base model.

    Args:
        batch: batch of statements
        tokenizer: HuggingFace tokenizer
        cfg: config file

    Returns:
        Tokenized sequence.
    """
    if cfg.agg == "last":
        input_seqs = tokenizer(
            batch.tolist(),
            return_tensors="pt",
            padding=True,
        )
    elif cfg.agg == "full":
        input_seqs = tokenizer(
            batch.tolist(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
    else:
        raise NotImplementedError("Only 'last' and 'full' aggregations are implemented.")

    return input_seqs


def instruct_tokenize(batch, tokenizer, cfg):
    """
    Tokenizer for instruct model.

    Args:
        batch: batch of statements
        tokenizer: HuggingFace tokenizer
        cfg: config file

    Returns:
        Tokenized sequence.
    """
    message_batch = [[{"role": "user", "content": x}] for x in batch]
    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=False,
    )

    if cfg.agg == "last":
        input_seqs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
        )
    elif cfg.agg == "full":
        input_seqs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
    else:
        raise NotImplementedError("Only 'last' and 'full' aggregations are implemented.")

    return input_seqs


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        """
        Forward-hook callback used by PyTorch.

        Args:
            module: the instance the hook is attached to
            module_inputs: inputs passed to the module
            module_outputs: output returned by the module

        Returns:
            None
        """
        if isinstance(module_outputs, (tuple, list)):
            self.out = module_outputs[0]
        else:
            self.out = module_outputs


def log_tensor_stats(output: torch.Tensor, *, layer: int, dataset: str, batch_idx: int) -> None:
    """
    Log basic finite-value diagnostics for an activation tensor before saving.

    Args:
        output: Activation tensor.
        layer: Layer id.
        dataset: Dataset name.
        batch_idx: Batch index.

    Returns:
        None
    """
    finite = torch.isfinite(output)
    n_bad = int((~finite).sum().item())

    if finite.any():
        max_abs = float(output[finite].abs().max().item())
    else:
        max_abs = float("nan")

    if n_bad > 0:
        log.warning(
            f"Non-finite activation values before saving | "
            f"dataset={dataset}, layer={layer}, batch={batch_idx}, "
            f"shape={tuple(output.shape)}, dtype={output.dtype}, "
            f"nonfinite={n_bad}, max_abs_finite={max_abs}"
        )
    elif batch_idx == 0:
        log.warning(
            f"Activation stats before saving | "
            f"dataset={dataset}, layer={layer}, batch={batch_idx}, "
            f"shape={tuple(output.shape)}, dtype={output.dtype}, "
            f"nonfinite={n_bad}, max_abs_finite={max_abs}"
        )


@hydra.main(version_base="1.1", config_path="configs", config_name="activations")
def main(cfg: DictConfig):
    validate_config(cfg)
    log_stats(cfg)

    original_cwd = Path(get_original_cwd())
    log.warning(f"Hydra runtime cwd: {Path.cwd()}")
    log.warning(f"Original cwd: {original_cwd}")

    model, tokenizer = prepare_hf_model(cfg)

    # device & dtype setup
    want = str(cfg.device)
    has_cuda = torch.cuda.is_available()

    if want.startswith("cuda") and has_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(dtype).to(want)
        torch.backends.cuda.matmul.allow_tf32 = True
        log.warning(f"Moved model to {want} with dtype {dtype}.")
    else:
        log.warning(f"Staying on {cfg.device} (cuda available? {has_cuda}).")

    model.eval()
    torch.set_grad_enabled(False)

    for dataset in cfg.datasets:
        # Setup forward hooks once, one per selected layer.
        layer = return_layers(cfg, dataset)
        hooks, handles = [], []

        encoder = model.get_submodule(cfg.model["module"]).get_submodule(
            cfg.model["encoders"]
        )

        hook = Hook()
        handle = encoder[layer].register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

        try:
            statements = load_statements_from_original_cwd(
                dataset=dataset,
                original_cwd=original_cwd,
            )

            n_batches = max(1, len(statements) // int(cfg.batch_size))
            batches = np.array_split(statements, n_batches)

            log.warning(
                f"Generating activations for {dataset} with {len(statements)} "
                f"statements in {len(batches)} batches."
            )
            log.info(f"\tExample of a statement: {statements[0]}")

            # Get hidden size from a single forward pass.
            warmup_batch = np.array(statements[:1])
            warmup = tokenize(warmup_batch, tokenizer, cfg)
            warmup_ids = warmup["input_ids"].to(cfg.device)
            warmup_att = warmup["attention_mask"].to(cfg.device)

            _ = model(warmup_ids, attention_mask=warmup_att, use_cache=False)

            if hook.out is None:
                raise RuntimeError("Hook output is None after warmup forward pass.")

            hidden_size = int(hook.out.shape[-1])

            # Build a dataset-specific save directory from cfg.output_dir.
            # e.g., <project_root>/outputs/activations/<model>/<dataset>/<agg>/
            base_out = original_cwd / Path(cfg.output_dir)
            save_dir = base_out / dataset / cfg.agg
            save_dir.mkdir(parents=True, exist_ok=True)

            max_len = int(cfg.max_length)
            log.warning(f"Max length: {max_len}")
            log.warning(f"Hidden size: {hidden_size}")
            log.warning(f"Saving activations to: {save_dir}")
            log.warning(f"Saving activation memmap dtype: {MEMMAP_DTYPE}")

            acts_memmap = {}
            save_path = {}
            compress_path = {}

            save_path[layer] = save_dir / f"layer_{layer}_e_temp.npy"
            compress_path[layer] = save_dir / f"layer_{layer}_e.npz"

            if cfg.agg == "last":
                acts_memmap[layer] = np.memmap(
                    save_path[layer],
                    dtype=MEMMAP_DTYPE,
                    mode="w+",
                    shape=(len(statements), hidden_size),
                )
                np.save(
                    save_dir / "shape.npy",
                    (len(statements), hidden_size),
                )

            elif cfg.agg == "full":
                acts_memmap[layer] = np.memmap(
                    save_path[layer],
                    dtype=MEMMAP_DTYPE,
                    mode="w+",
                    shape=(len(statements), max_len, hidden_size),
                )
                np.save(
                    save_dir / "shape.npy",
                    (len(statements), max_len, hidden_size),
                )

            else:
                raise NotImplementedError(
                    "Only 'last' and 'full' aggregations are implemented."
                )

            last_row = 0
            masks = []

            for batch_idx, batch in tqdm(enumerate(batches), total=len(batches)):
                input_seqs = tokenize(batch, tokenizer, cfg)
                input_ids = input_seqs["input_ids"].to(cfg.device)
                input_att = input_seqs["attention_mask"].to(cfg.device)
                masks.append(input_att[:, -max_len:].detach())

                _ = model(input_ids, attention_mask=input_att, use_cache=False)

                output = hook.out

                if output is None:
                    raise RuntimeError("Hook output is None after model forward pass.")

                if output.dtype != torch.float32:
                    output = output.float()

                log_tensor_stats(
                    output,
                    layer=layer,
                    dataset=dataset,
                    batch_idx=batch_idx,
                )

                if cfg.agg == "last":
                    assert output.ndim == 3, f"Expected (B,T,H), got {output.shape}"

                    embeddings = (
                        output[:, -1, :]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(ACTIVATION_DTYPE, copy=False)
                    )

                    if not np.isfinite(embeddings).all():
                        raise RuntimeError(
                            f"Non-finite values detected in embeddings before write: "
                            f"dataset={dataset}, layer={layer}, batch={batch_idx}"
                        )

                    acts_memmap[layer][
                        last_row:last_row + embeddings.shape[0], :
                    ] = embeddings

                elif cfg.agg == "full":
                    assert output.ndim == 3, f"Expected (B,T,H), got {output.shape}"
                    assert output.shape[1] == max_len, (
                        f"Expected T={max_len}, got {output.shape[1]}"
                    )

                    embeddings = (
                        output.detach()
                        .cpu()
                        .numpy()
                        .astype(ACTIVATION_DTYPE, copy=False)
                    )

                    if not np.isfinite(embeddings).all():
                        raise RuntimeError(
                            f"Non-finite values detected in embeddings before write: "
                            f"dataset={dataset}, layer={layer}, batch={batch_idx}"
                        )

                    acts_memmap[layer][
                        last_row:last_row + embeddings.shape[0], :, :
                    ] = embeddings

                else:
                    raise NotImplementedError

                last_row += batch.shape[0]

            # Save attention masks.
            masks = torch.vstack(masks).cpu().numpy()
            np.save(save_dir / "mask.npy", masks)

            log.info(f"\tFlush of activations for {dataset} started...")
            acts_memmap[layer].flush()

            log.warning(
                f"{cfg.model.name} activations saved for {dataset} -> {save_dir}"
            )

        finally:
            for handle in handles:
                handle.remove()

    log.warning("Done.")


if __name__ == "__main__":
    main()