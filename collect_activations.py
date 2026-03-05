"""
collect_activations.py

Generate and save hidden activations from a HuggingFace model
for one or more text datasets.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm

from utils import get_device, prepare_hf_model, load_statements, return_layers
log = logging.getLogger(__name__)


def validate_config(cfg):
    """
    Verifies the config file has all the necessary fields.

    :param cfg: config file
    :return: None
    """
    assert cfg.agg in [
        "last", "mean", "max", 'full'], "Aggregation tupe must be either 'last', 'mean' or 'max'."
    assert len(cfg.layers) > 0, "At least one layer must be selected."
    assert type(
        cfg.datasets) == list or type(cfg.datasets).__name__ == "ListConfig", f"Datasets must be a list. Not {type(cfg.datasets)}"
    assert len(cfg.datasets) > 0, "At least one dataset must be selected."
    if cfg.device is None:
        OmegaConf.set_struct(cfg, False)  # Allow overriding
        cfg["device"] = str(get_device())
        OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    """
    Prints initial debugging info.

    :param cfg: config file
    :return: None
    """
    log.warning(f"Collecting activations for: {cfg.model.name} (device: {cfg.device})")
    log.warning(f'Max length of the input sequences: {cfg.max_length}')


def tokenize(batch, tokenizer, cfg):
    """
    Tokenizes the statements based on the type of model (instruct or base).

    :param batch: batch of statements
    :param tokenizer: HuggingFace tokenizer
    :param cfg: config file
    :return: tokenized sequence
    """
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    else:
        return default_tokenize(batch, tokenizer, cfg)


def default_tokenize(batch, tokenizer, cfg):
    """
    Tokenizer for base model.

    :param batch: batch of statements
    :param tokenizer: HuggingFace tokenizer
    :param cfg: config file
    :return: tokenized sequence
    """
    if cfg.agg == 'last':
        input_seqs = tokenizer(
            batch.tolist(), return_tensors="pt", padding=True)
    elif cfg.agg == 'full':
        input_seqs = tokenizer(
            batch.tolist(), return_tensors="pt", padding="max_length",  truncation=True, max_length=cfg.max_length)
    return input_seqs


def instruct_tokenize(batch, tokenizer, cfg):
    """
    Tokenizer for instruct model.

    :param batch: batch of statements
    :param tokenizer: HuggingFace tokenizer
    :param cfg: config file
    :return: tokenized sequence
    """
    message_batch = [[{"role": "user", "content": x}] for x in batch]
    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=False,
    )
    if cfg.agg == 'last':
        input_seqs = tokenizer(
            text_batch, return_tensors="pt", padding=True)
    elif cfg.agg == 'full':
        input_seqs = tokenizer(
            text_batch, return_tensors="pt", padding="max_length",  truncation=True, max_length=cfg.max_length)
    return input_seqs


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        """
        Forward-hook callback used by PyTorch.

        :param module: the instance the hook is attached to
        :param module_inputs: inputs passed to the module
        :param module_outputs: output returned by the module
        :return: None
        """
        if isinstance(module_outputs, (tuple, list)):
            self.out = module_outputs[0]
        else:
            # If it returns a tensor, keep it as-is
            self.out = module_outputs



@hydra.main(config_path="configs", config_name="activations")
def main(cfg: DictConfig):

    validate_config(cfg)
    log_stats(cfg)
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

        # Setup forward hooks once (one per layer)
        layer = return_layers(cfg, dataset)
        hooks, handles = [], []
        encoder = model.get_submodule(cfg.model["module"]).get_submodule(
            cfg.model["encoders"]
        )

        hook = Hook()
        handle = encoder[layer].register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

        statements = load_statements(dataset)
        n_batches = max(1, len(statements) // int(cfg.batch_size))
        batches = np.array_split(statements, n_batches)

        log.warning(
            f"Generating activations for {dataset} with {len(statements)} "
            f"statements in {len(batches)} batches."
        )
        log.info(f"\tExample of a statement: {statements[0]}")

        # Get hidden size from a single forward pass
        input_seq = tokenizer(statements[0], return_tensors="pt")
        _ = model(input_seq["input_ids"].to(cfg.device))
        hidden_size = hooks[0].out.shape[-1]

        # Build a dataset-specific save directory from cfg.output_dir
        # e.g., structure: <output_dir>/<model.name>/<dataset>/<agg>/
        base_out = Path(cfg.output_dir)
        save_dir = base_out / dataset / cfg.agg
        save_dir.mkdir(parents=True, exist_ok=True)

        MAX_LEN = cfg.max_length
        print(f"\n\nmax length: {MAX_LEN}")

        acts_memmap = {}
        save_path = {}
        compress_path = {}

        save_path[layer] = save_dir / f"layer_{layer}_e_temp.npy"
        compress_path[layer] = save_dir / f"layer_{layer}_e.npz"

        if cfg.agg == "last":
            acts_memmap[layer] = np.memmap(
                save_path[layer],
                dtype="float16",
                mode="w+",
                shape=(len(statements), hidden_size),
            )
        elif cfg.agg == "full":
            acts_memmap[layer] = np.memmap(
                save_path[layer],
                dtype="float16",
                mode="w+",
                shape=(len(statements), MAX_LEN, hidden_size),
            )
            np.save(save_dir / "shape.npy", (len(statements), MAX_LEN, hidden_size))
        else:
            raise NotImplementedError("Only 'last' and 'full' aggregations are implemented.")

        _last_row = 0
        masks = []

        for _, batch in tqdm(enumerate(batches), total=len(batches)):
            input_seqs = tokenize(batch, tokenizer, cfg)
            input_ids = input_seqs["input_ids"].to(cfg.device)
            input_att = input_seqs["attention_mask"].to(cfg.device)
            masks.append(input_att[:, -MAX_LEN:].detach())

            _ = model(input_ids, attention_mask=input_att, use_cache=False)

            output = hook.out
            if output.dtype != torch.float32:
                output = output.float()

            if cfg.agg == "last":
                # expect (B, T, H)
                assert output.ndim == 3, f"Expected (B,T,H) got {output.shape}"
                embeddings = output[:, -1, :].detach().cpu().numpy().astype(np.float16)
                acts_memmap[layer][_last_row:_last_row + embeddings.shape[0], :] = embeddings

            elif cfg.agg == "full":
                # expect (B, T, H)
                assert output.ndim == 3, f"Expected (B,T,H) got {output.shape}"
                assert output.shape[1] == MAX_LEN, f"Expected T={MAX_LEN} got {output.shape[1]}"
                embeddings = output.detach().cpu().numpy().astype(np.float16)
                acts_memmap[layer][_last_row:_last_row + embeddings.shape[0], :, :] = embeddings
            else:
                raise NotImplementedError

            _last_row += batch.shape[0]

        # Save attention masks
        masks = torch.vstack(masks).cpu().numpy()
        np.save(save_dir / "mask.npy", masks)

        log.info(f"\tCompression of activations for {dataset} started...")
        acts_memmap[layer].flush()

        log.warning(f"{cfg.model.name} activations saved for {dataset} -> {save_dir}")

    # Clean up hooks
    for h in handles:
        h.remove()

    log.warning("Done.")


if __name__ == '__main__':
    main()