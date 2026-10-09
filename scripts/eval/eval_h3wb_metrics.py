from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aligned_continuous.data import AlignedContinuousDataset
from aligned_continuous.model import (
    AlignedContinuousOrderGPT,
    AlignedContinuousConfig,
)
from h3wb.evaluation import (
    denormalize_current_id_poses,
    greedy_mean_rollout,
    official_mpjpe_metrics,
    restore_canonical_ids,
    target_normalization_in_current_ids,
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



METRIC_KEYS = (
    "teacher_forced_nll_per_joint",
    "teacher_forced_nll_per_coordinate",
    "free_running_mpjpe_whole_body_mm",
    "free_running_mpjpe_body_mm",
    "free_running_mpjpe_face_nose_aligned_mm",
    "free_running_mpjpe_hands_wrist_aligned_mm",
)


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
        "--split",
        choices=("train", "discovery", "val", "test"),
        default="test",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("config", "float32", "float16", "bfloat16"),
        default="config",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/eval/h3wb"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_batches < 0 or args.max_samples < 0:
        parser.error("--max-batches and --max-samples must be non-negative")
    if args.checkpoint is None and args.teacher_checkpoint is None and args.mlp_checkpoint is None:
        parser.error(
            "one of --checkpoint, --teacher-checkpoint or --mlp-checkpoint is required"
        )
    return args


def build_model(
    config: Mapping[str, Any],
    dataset: AlignedContinuousDataset,
) -> AlignedContinuousOrderGPT:
    return AlignedContinuousOrderGPT(
        AlignedContinuousConfig(
            num_variables=dataset.num_variables,
            value_dim=dataset.value_dim,
            condition_dim=dataset.condition_dim,
            n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]),
            n_embd=int(config["n_embd"]),
            condition_layers=int(config["condition_layers"]),
            dropout=float(config["dropout"]),
            bias=bool(config["bias"]),
            min_log_scale=float(config["min_log_scale"]),
            max_log_scale=float(config["max_log_scale"]),
        )
    )


@torch.inference_mode()
def evaluate_method(
    method: str,
    *,
    checkpoint_input: str | Path,
    split: str,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    max_batches: int,
    max_samples: int,
    data_dir: str | Path | None = None,
    order_seed: int = 20260713,
) -> Dict[str, Any]:
    checkpoint_path, checkpoint = load_checkpoint(checkpoint_input)
    embedded = dict(checkpoint["config"])
    data_dir = resolve_repo_path(data_dir if data_dir is not None else str(embedded["data_dir"]))
    dataset = AlignedContinuousDataset(
        data_dir,
        int(embedded["variable_permutation_seed"]),
    )
    if dataset.dataset_name.lower() != "h3wb":
        raise ValueError(f"expected H3WB data, got {dataset.dataset_name!r}")
    if dataset.num_variables != 133 or dataset.value_dim != 3:
        raise ValueError("H3WB evaluator requires 133 three-dimensional joints")
    if split not in dataset.split_names:
        raise ValueError(f"split {split!r} is absent from the prepared H3WB data")
    config, cached_order = validate_core_checkpoint(
        checkpoint,
        dataset_name="h3wb",
        width=dataset.num_variables,
        expected_policy=method,
    )

    selected_dtype_name = str(config["dtype"]) if dtype_name == "config" else dtype_name
    selected_dtype = resolve_autocast_dtype(selected_dtype_name, device)
    model = build_model(config, dataset)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    predictions = []
    targets = []
    nll_sum = 0.0
    sample_count = 0
    order_generator = evaluation_generator(device, order_seed)
    started = time.time()
    for batch_id, batch in enumerate(
        dataset.iter_batches(split, int(batch_size), device=device)
    ):
        if max_batches > 0 and batch_id >= max_batches:
            break
        remaining = max_samples - sample_count if max_samples > 0 else 0
        if max_samples > 0 and remaining <= 0:
            break
        if max_samples > 0 and int(batch.targets.shape[0]) > remaining:
            batch.targets = batch.targets[:remaining]
            batch.conditions = batch.conditions[:remaining]
        current_batch = int(batch.targets.shape[0])
        orders = evaluation_orders(cached_order, current_batch, dataset.num_variables, device, order_generator)
        with autocast_context(device, selected_dtype):
            teacher_forced = model(batch.targets, batch.conditions, orders)
        nll_sum += float(teacher_forced.step_losses.float().sum().item())
        generated = greedy_mean_rollout(
            model,
            batch.conditions,
            orders,
            autocast_context=lambda: autocast_context(device, selected_dtype),
        )
        predictions.append(generated.float().cpu())
        targets.append(batch.targets.float().cpu())
        sample_count += current_batch
        print(
            json.dumps(
                {"method": method, "batch": batch_id + 1, "samples": sample_count},
                sort_keys=True,
            ),
            flush=True,
        )

    if sample_count == 0:
        raise ValueError("evaluation selected no H3WB samples")
    predicted_current = torch.cat(predictions)
    target_current = torch.cat(targets)
    current_to_canonical = dataset.variable_permutation.tolist()
    mean_current, std_current = target_normalization_in_current_ids(
        dataset.metadata,
        current_to_canonical,
    )
    predicted_mm = restore_canonical_ids(
        denormalize_current_id_poses(
            predicted_current,
            mean_current,
            std_current,
        ),
        current_to_canonical,
    )
    target_mm = restore_canonical_ids(
        denormalize_current_id_poses(
            target_current,
            mean_current,
            std_current,
        ),
        current_to_canonical,
    )
    metrics: Dict[str, Any] = {
        "method": method,
        "dataset": "h3wb",
        "split": split,
        "num_samples": sample_count,
        "teacher_forced_nll_per_joint": nll_sum
        / (sample_count * dataset.num_variables),
        "teacher_forced_nll_per_coordinate": nll_sum
        / (sample_count * dataset.num_variables * dataset.value_dim),
        **{
            key: value
            for key, value in official_mpjpe_metrics(
                predicted_mm,
                target_mm,
            ).items()
            if key
            in {
                "free_running_mpjpe_whole_body_mm",
                "free_running_mpjpe_body_mm",
                "free_running_mpjpe_face_nose_aligned_mm",
                "free_running_mpjpe_hands_wrist_aligned_mm",
            }
        },
        "policy": method,
        "affinity": "dense_mutual_geometric" if method in {"teacher", "mlp"} else None,
        "variable_permutation_seed": int(config["variable_permutation_seed"]),
        "model_seed": int(config["seed"]),
        "checkpoint_iteration": int(checkpoint.get("iter_num", -1)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cached_order_current_ids": list(cached_order) if cached_order is not None else None,
        "order_seed": int(order_seed),
        "data_dir": str(dataset.data_dir),
        "dtype": selected_dtype_name,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    }
    return metrics


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "method",
        "num_samples",
        *METRIC_KEYS,
        "checkpoint_iteration",
        "checkpoint_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


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
        row = evaluate_method(
            method,
            checkpoint_input=checkpoints[method],
            data_dir=args.data_dir,
            order_seed=args.order_seed,
            split=args.split,
            device=device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            max_samples=args.max_samples,
        )
        write_json(result_path, row)
        rows.append(row)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    print(
        json.dumps({"output_dir": str(output_dir), "methods": list(methods)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
