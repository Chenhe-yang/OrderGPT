from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aligned_tabular.data import AlignedTabularDataset
from aligned_tabular.model import (
    AlignedTabularOrderGPT,
    AlignedTabularConfig,
)
from scripts.eval.common import (
    autocast_context,
    load_checkpoint,
    evaluation_orders,
    evaluation_generator,
    selected_checkpoints,
    resolve_autocast_dtype,
    resolve_device,
    resolve_repo_path,
    sha256_file,
    validate_core_checkpoint,
    write_json,
)



SPLITS = ("train", "val", "test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate any supported method checkpoint.")
    parser.add_argument("--data-dir", type=Path, help="Override the checkpoint data path.")
    parser.add_argument("--order-seed", type=int, default=20260713)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Direct Spectral Teacher checkpoint to evaluate.",
    )
    parser.add_argument(
        "--mlp-checkpoint",
        type=Path,
        help="Order MLP checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/eval/census/full_table"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("config", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=0,
        help="Positive values are for smoke tests; zero evaluates every row.",
    )
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_rows_per_split < 0 or args.progress_every < 0:
        parser.error("row and progress limits must be non-negative")
    if args.checkpoint is None and args.teacher_checkpoint is None and args.mlp_checkpoint is None:
        parser.error(
            "one of --checkpoint, --teacher-checkpoint or --mlp-checkpoint is required"
        )
    return args


def build_model(
    dataset: AlignedTabularDataset,
    config: Mapping[str, Any],
) -> AlignedTabularOrderGPT:
    return AlignedTabularOrderGPT(
        AlignedTabularConfig(
            num_variables=dataset.num_variables,
            max_states=dataset.max_states,
            cardinalities=dataset.cardinalities.tolist(),
            n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]),
            n_embd=int(config["n_embd"]),
            dropout=float(config["dropout"]),
            bias=bool(config["bias"]),
            condition_dim=0,
        )
    )


def validate_model_frame(
    checkpoint: Mapping[str, Any],
    dataset: AlignedTabularDataset,
) -> None:
    saved_cardinalities = checkpoint["model"].get("cardinalities")
    if saved_cardinalities is None or not torch.equal(
        saved_cardinalities.detach().cpu().long(),
        dataset.cardinalities.cpu().long(),
    ):
        raise ValueError("checkpoint cardinalities do not match its current-ID frame")


def split_metrics(
    *,
    nll_sum: float,
    correct: int,
    rows: int,
    width: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    cells = int(rows) * int(width)
    nll = float(nll_sum) / float(cells)
    return {
        "num_rows": int(rows),
        "num_variables": int(width),
        "num_cells": cells,
        "nll_sum_nats": float(nll_sum),
        "teacher_forced_nll_per_variable": nll,
        "teacher_forced_bits_per_variable": nll / math.log(2.0),
        "joint_nll_nats_per_datapoint": nll * int(width),
        "joint_bits_per_datapoint": nll * int(width) / math.log(2.0),
        "teacher_forced_state_accuracy": float(correct) / float(cells),
        "elapsed_seconds": float(elapsed_seconds),
    }


@torch.inference_mode()
def evaluate_method(
    method: str,
    *,
    checkpoint_input: str | Path,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    max_rows_per_split: int,
    progress_every: int,
    data_dir: str | Path | None = None,
    order_seed: int = 20260713,
) -> Dict[str, Any]:
    checkpoint_path, checkpoint = load_checkpoint(checkpoint_input)
    embedded = dict(checkpoint["config"])
    dataset = AlignedTabularDataset(
        resolve_repo_path(data_dir if data_dir is not None else str(embedded["data_dir"])),
        int(embedded["variable_permutation_seed"]),
    )
    if dataset.dataset_name.lower() != "census":
        raise ValueError(f"expected CENSUS data, got {dataset.dataset_name!r}")
    config, cached_order = validate_core_checkpoint(
        checkpoint,
        dataset_name="census",
        width=dataset.num_variables,
        expected_policy=method,
    )
    validate_model_frame(checkpoint, dataset)
    selected_dtype_name = str(config["dtype"]) if dtype_name == "config" else dtype_name
    selected_dtype = resolve_autocast_dtype(selected_dtype_name, device)
    model = build_model(dataset, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    pooled_nll = 0.0
    pooled_correct = 0
    pooled_rows = 0
    split_results: Dict[str, Any] = {}
    order_generator = evaluation_generator(device, order_seed)
    started = time.time()
    for split in SPLITS:
        split_started = time.time()
        nll_sum = 0.0
        correct = 0
        rows = 0
        for batch_id, batch in enumerate(
            dataset.iter_batches(split, int(batch_size), device=device)
        ):
            remaining = (
                int(max_rows_per_split) - rows
                if int(max_rows_per_split) > 0
                else 0
            )
            if max_rows_per_split > 0 and remaining <= 0:
                break
            if max_rows_per_split > 0 and int(batch.targets.shape[0]) > remaining:
                batch.targets = batch.targets[:remaining]
                batch.conditions = batch.conditions[:remaining]
            current_rows = int(batch.targets.shape[0])
            orders = evaluation_orders(cached_order, current_rows, dataset.num_variables, device, order_generator)
            with autocast_context(device, selected_dtype):
                output = model(batch.targets, batch.conditions, orders)
            nll_sum += float(output.step_losses.float().sum().item())
            correct += int(
                (output.logits_current.argmax(dim=-1) == batch.targets).sum().item()
            )
            rows += current_rows
            if progress_every > 0 and (batch_id + 1) % progress_every == 0:
                print(f"[{method}:{split}] rows={rows:,}", flush=True)
        if rows == 0:
            raise ValueError(f"split {split!r} selected no rows")
        metrics = split_metrics(
            nll_sum=nll_sum,
            correct=correct,
            rows=rows,
            width=dataset.num_variables,
            elapsed_seconds=time.time() - split_started,
        )
        split_results[split] = metrics
        pooled_nll += nll_sum
        pooled_correct += correct
        pooled_rows += rows
        print(
            f"[{method}:{split}] "
            f"NLL/var={metrics['teacher_forced_nll_per_variable']:.9f}",
            flush=True,
        )

    result = {
        "method": method,
        "dataset": "census",
        "variable_permutation_seed": int(dataset.variable_permutation_seed),
        "current_to_canonical": dataset.current_to_canonical.tolist(),
        "policy": method,
        "affinity": "dense_mutual_geometric" if method in {"teacher", "mlp"} else None,
        "cached_order_current_ids": list(cached_order) if cached_order is not None else None,
        "order_seed": int(order_seed),
        "data_dir": str(dataset.data_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iter_num", -1)),
        "model_seed": int(config["seed"]),
        "dtype": selected_dtype_name,
        "device": str(device),
        "splits": split_results,
        "full_table": split_metrics(
            nll_sum=pooled_nll,
            correct=pooled_correct,
            rows=pooled_rows,
            width=dataset.num_variables,
            elapsed_seconds=time.time() - started,
        ),
    }
    return result


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "method",
        "scope",
        "num_rows",
        "nll_per_variable",
        "bits_per_variable",
        "joint_nll_nats_per_datapoint",
        "joint_bits_per_datapoint",
        "state_accuracy",
        "variable_permutation_seed",
        "checkpoint_iteration",
        "checkpoint_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for scope in (*SPLITS, "full_table"):
                metrics = row["full_table"] if scope == "full_table" else row["splits"][scope]
                writer.writerow(
                    {
                        "method": row["method"],
                        "scope": scope,
                        "num_rows": metrics["num_rows"],
                        "nll_per_variable": (
                            f"{metrics['teacher_forced_nll_per_variable']:.12f}"
                        ),
                        "bits_per_variable": (
                            f"{metrics['teacher_forced_bits_per_variable']:.12f}"
                        ),
                        "joint_nll_nats_per_datapoint": (
                            f"{metrics['joint_nll_nats_per_datapoint']:.12f}"
                        ),
                        "joint_bits_per_datapoint": (
                            f"{metrics['joint_bits_per_datapoint']:.12f}"
                        ),
                        "state_accuracy": (
                            f"{metrics['teacher_forced_state_accuracy']:.12f}"
                        ),
                        "variable_permutation_seed": row[
                            "variable_permutation_seed"
                        ],
                        "checkpoint_iteration": row["checkpoint_iteration"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoints = selected_checkpoints(args)
    methods = tuple(
        method
        for method, checkpoint in checkpoints.items()
        if checkpoint is not None
    )
    output_dir = resolve_repo_path(args.output_dir)
    device = resolve_device(args.device)
    rows = []
    for method in methods:
        result_path = output_dir / f"{method}.json"
        if result_path.is_file() and not args.force:
            raise FileExistsError(
                f"{result_path} already exists; pass --force to replace it"
            )
        result = evaluate_method(
            method,
            checkpoint_input=checkpoints[method],
            data_dir=args.data_dir,
            order_seed=args.order_seed,
            device=device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            max_rows_per_split=args.max_rows_per_split,
            progress_every=args.progress_every,
        )
        write_json(result_path, result)
        rows.append(result)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    print(
        json.dumps({"output_dir": str(output_dir), "methods": list(methods)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
