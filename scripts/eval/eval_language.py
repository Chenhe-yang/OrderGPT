"""Token perplexity on fixed records in original token-position coordinates."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ordergpt.language_model import LanguageModelConfig, OrderGPTLanguageModel, filter_legacy_language_model_args
from scripts.eval.common import autocast_context, load_checkpoint, resolve_autocast_dtype, resolve_device, sha256_file, validate_order, write_json


def open_records(path: Path, width: int):
    if path.suffix == ".npy":
        records = np.load(path, mmap_mode="r", allow_pickle=False)
        if records.ndim != 2 or records.shape[1] != width or records.dtype.kind not in "iu":
            raise ValueError(f"records must be an integer [N, {width}] .npy array")
        dropped = 0
    elif path.suffix == ".bin":
        if path.stat().st_size % 2:
            raise ValueError("uint16 stream has an odd byte count")
        stream = np.memmap(path, dtype=np.uint16, mode="r")
        count, dropped = divmod(len(stream), width)
        records = stream[:count * width].reshape(count, width)
    else:
        raise ValueError("--records must be .npy fixed records or a uint16 .bin stream")
    if not len(records):
        raise ValueError("records contain no complete sequences")
    return records, dropped


def checkpoint_order(checkpoint, width, order_name):
    if "current_to_original" not in checkpoint:
        raise ValueError("checkpoint is missing its current_to_original coordinate mapping")
    mapping = torch.tensor(validate_order(checkpoint["current_to_original"], width))
    if order_name == "canonical":
        return mapping, torch.argsort(mapping)
    if order_name == "random":
        return mapping, None
    cached = checkpoint.get("cached_order")
    if cached is None:
        cached = checkpoint.get("runtime", {}).get("cached_order")
    if cached is None:
        raise ValueError("checkpoint has no saved order; select --order random/canonical explicitly")
    config = checkpoint["config"]
    if config.get("method") in {"mlp", "teacher", "fixed_curriculum"}:
        boundary = config.get("anneal_end", config.get("anneal_end_iter"))
        if boundary is None or int(checkpoint.get("iter_num", -1)) < int(boundary):
            raise ValueError("saved learned order is not frozen yet (checkpoint precedes anneal_end)")
    return mapping, torch.tensor(validate_order(cached, width))


@torch.inference_mode()
def evaluate(checkpoint, records, *, order_name="saved", device=torch.device("cpu"), dtype=None,
             batch_size=1, random_orders=1, seed=2357, max_records=0):
    model = OrderGPTLanguageModel(LanguageModelConfig(**filter_legacy_language_model_args(checkpoint["model_args"])))
    state = {}
    for key, value in checkpoint["model"].items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        state[key] = value
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    width = model.config.block_size
    mapping, fixed = checkpoint_order(checkpoint, width, order_name)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(len(records), max_records) if max_records else len(records)
    repeats = random_orders if order_name == "random" else 1
    nll_sum = 0.0
    for start in range(0, count, batch_size):
        values = np.array(records[start:min(start + batch_size, count)], dtype=np.int64)
        if values.min() < 0 or values.max() >= model.config.vocab_size:
            raise ValueError("record token outside checkpoint vocabulary")
        tokens = torch.from_numpy(values)[:, mapping].to(device)
        for _ in range(repeats):
            orders = (torch.stack([torch.randperm(width, generator=generator) for _ in range(len(values))])
                      if fixed is None else fixed.expand(len(values), -1)).to(device)
            with autocast_context(device, dtype):
                _, _, losses = model(tokens, mode=None, orders=orders, return_token_loss=True, return_logits=False)
            nll_sum += float(losses.double().sum().item()) / repeats
    n_tokens = count * width
    nll = nll_sum / n_tokens
    return {"records": count, "tokens": n_tokens, "context_length": width, "nll_sum": nll_sum,
            "nll_per_token": nll, "perplexity": math.exp(nll), "order": order_name,
            "random_orders": repeats, "seed": seed, "includes_first_token": True,
            "checkpoint_iteration": int(checkpoint.get("iter_num", -1)),
            "protocol": "fixed_records_all_targets_mean_nll_then_exp_v1"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("saved", "canonical", "random"), default="saved")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--random-orders", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.random_orders) <= 0 or args.max_records < 0:
        parser.error("batch-size/random-orders must be positive; max-records must be nonnegative")
    if args.output.exists():
        parser.error("output exists; select a new output path")
    ckpt_path, checkpoint = load_checkpoint(args.checkpoint)
    records, dropped = open_records(args.records, int(checkpoint["model_args"]["block_size"]))
    device = resolve_device(args.device)
    result = evaluate(checkpoint, records, order_name=args.order, device=device,
                      dtype=resolve_autocast_dtype(args.dtype, device), batch_size=args.batch_size,
                      random_orders=args.random_orders, seed=args.seed, max_records=args.max_records)
    result.update(checkpoint=str(ckpt_path), checkpoint_sha256=sha256_file(ckpt_path),
                  records_path=str(args.records.resolve()), records_sha256=sha256_file(args.records),
                  trailing_tokens_dropped=dropped, dtype=args.dtype, device=str(device))
    write_json(args.output, result)
    print(f"NLL/token={result['nll_per_token']:.6f}, perplexity={result['perplexity']:.6f}")


if __name__ == "__main__":
    main()
