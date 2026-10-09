from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aligned_tabular.data import AlignedTabularDataset
from aligned_tabular.query_evaluation import (
    IncrementalAlignedTabularSampler,
    build_allowed_masks,
    exact_query_cardinalities,
    progressive_sample_log_weights,
    q_error,
    summarize_q_errors,
)
from scripts.eval.common import (
    autocast_context,
    load_checkpoint,
    checkpoint_method,
    evaluation_orders,
    resolve_autocast_dtype,
    resolve_device,
    resolve_repo_path,
    sha256_file,
    validate_core_checkpoint,
    write_json,
)
from scripts.eval.eval_census_full_table_nll import (
    build_model,
    validate_model_frame,
)


DEFAULT_OUTPUT_DIR = Path("out/eval/census/progressive_sampling")
SAMPLER_PROTOCOL = "ordergpt_census_all68_progressive_sampling_v2"
REMAP_PROTOCOL = "canonical_varskip_workload_remapped_to_current_ids_v1"
QUERY_EVALUATION_SOURCE = REPO_ROOT / "aligned_tabular/query_evaluation.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="Override the checkpoint data path.")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("config", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="Use the first N canonical workload queries (default: 1000).",
    )
    parser.add_argument(
        "--budget",
        action="append",
        type=int,
        default=[],
        help="Nested particle budget; defaults to the paper budget of 10000.",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260724)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--particle-chunk-size", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if args.query_batch_size <= 0 or args.particle_chunk_size <= 0:
        parser.error("batch and chunk sizes must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_canonical_workload(
    path: str | Path,
    *,
    num_queries: int,
) -> tuple[Dict[str, Any], str]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if not isinstance(stored, dict):
        raise ValueError("canonical workload must be a JSON object")
    stored_sha = str(stored.get("workload_sha256", ""))
    unsigned = dict(stored)
    unsigned.pop("workload_sha256", None)
    if canonical_sha256(unsigned) != stored_sha:
        raise ValueError(f"canonical workload checksum mismatch: {resolved}")
    fingerprint = stored.get("dataset_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError("canonical workload has no dataset fingerprint")
    if int(fingerprint.get("variable_permutation_seed", -2)) != -1:
        raise ValueError("workload must be in canonical column coordinates")
    queries = list(stored.get("queries", []))
    if int(num_queries) > len(queries):
        raise ValueError(
            f"requested {num_queries} queries, workload contains {len(queries)}"
        )
    selected = dict(stored)
    selected["queries"] = copy.deepcopy(queries[: int(num_queries)])
    selected["num_queries"] = int(num_queries)
    if int(num_queries) == len(queries):
        selected_sha = stored_sha
    else:
        selected.pop("workload_sha256", None)
        selected_sha = canonical_sha256(selected)
        selected["workload_sha256"] = selected_sha
    return selected, selected_sha


def workload_dataset_fingerprint(
    dataset: AlignedTabularDataset,
) -> Dict[str, Any]:
    source = dataset.metadata.get("source", {})
    splits = dataset.metadata["splits"]
    return {
        "dataset": dataset.dataset_name,
        "num_rows": int(
            sum(dataset.num_samples(split) for split in ("train", "val", "test"))
        ),
        "num_variables": int(dataset.num_variables),
        "variable_permutation_seed": int(dataset.variable_permutation_seed),
        "current_to_canonical": dataset.current_to_canonical.tolist(),
        "cardinalities_current_ids": dataset.cardinalities.tolist(),
        "source_sha256": (
            source.get("sha256") if isinstance(source, Mapping) else None
        ),
        "split_sha256": {
            split: splits[split].get("sha256")
            for split in ("train", "val", "test")
        },
    }


def full_records_current_ids(dataset: AlignedTabularDataset) -> np.ndarray:
    canonical = np.concatenate(
        [
            np.asarray(dataset._records[split], dtype=np.uint8)
            for split in ("train", "val", "test")
        ],
        axis=0,
    )
    current_to_canonical = dataset.current_to_canonical.cpu().numpy()
    return np.ascontiguousarray(canonical[:, current_to_canonical])


def remap_workload_to_current_ids(
    canonical_workload: Mapping[str, Any],
    dataset: AlignedTabularDataset,
    canonical_workload_sha256: str,
) -> tuple[Dict[str, Any], str]:
    canonical_to_current = dataset.canonical_to_current.tolist()
    remapped_queries = []
    for source_query in canonical_workload["queries"]:
        query = copy.deepcopy(source_query)
        canonical_columns = [int(value) for value in source_query["columns"]]
        query["canonical_columns"] = canonical_columns
        query["columns"] = [
            int(canonical_to_current[column]) for column in canonical_columns
        ]
        for predicate in query["predicates"]:
            canonical_column = int(predicate["column"])
            predicate["canonical_column"] = canonical_column
            predicate["column"] = int(canonical_to_current[canonical_column])
        remapped_queries.append(query)
    workload = {
        "protocol": REMAP_PROTOCOL,
        "canonical_workload_sha256": canonical_workload_sha256,
        "query_seed": int(canonical_workload["query_seed"]),
        "num_queries": len(remapped_queries),
        "dataset_fingerprint": workload_dataset_fingerprint(dataset),
        "queries": remapped_queries,
    }
    remapped_sha = canonical_sha256(
        {
            "protocol": workload["protocol"],
            "canonical_workload_sha256": canonical_workload_sha256,
            "current_to_canonical": dataset.current_to_canonical.tolist(),
            "queries": remapped_queries,
        }
    )
    workload["remapped_workload_sha256"] = remapped_sha
    return workload, remapped_sha


def verify_remapped_cardinalities(
    dataset: AlignedTabularDataset,
    workload: Mapping[str, Any],
) -> Dict[str, Any]:
    records = full_records_current_ids(dataset)
    observed = exact_query_cardinalities(records, workload["queries"])
    expected = np.asarray(
        [int(query["actual_cardinality"]) for query in workload["queries"]],
        dtype=np.int64,
    )
    del records
    if not np.array_equal(observed, expected):
        mismatch = np.flatnonzero(observed != expected)
        first = int(mismatch[0])
        raise ValueError(
            "canonical-to-current remap changed query cardinality at "
            f"{first}: expected {expected[first]}, got {observed[first]}"
        )
    return {
        "verified": True,
        "num_queries": int(expected.size),
        "num_exact_matches": int((observed == expected).sum()),
        "min_actual_cardinality": int(expected.min()),
        "max_actual_cardinality": int(expected.max()),
    }


def _seed_for_query_batch(seed: int, query_start: int) -> int:
    return int((int(seed) + 1_000_003 * int(query_start)) % (2**63 - 1))


def _new_generator(device: torch.device, seed: int) -> torch.Generator:
    target = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=target).manual_seed(int(seed))


def _partial_path(output_dir: Path, method: str = "mlp") -> Path:
    return output_dir / f"{method}.partial.npz"


def _save_partial(
    path: Path,
    *,
    signature: str,
    completed: np.ndarray,
    estimates: Mapping[int, np.ndarray],
    log_selectivities: Mapping[int, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    arrays: Dict[str, Any] = {
        "signature": np.asarray(signature),
        "completed": completed.astype(np.bool_, copy=False),
    }
    for budget in sorted(estimates):
        arrays[f"estimate_{budget}"] = estimates[budget]
        arrays[f"log_selectivity_{budget}"] = log_selectivities[budget]
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _load_partial(
    path: Path,
    *,
    signature: str,
    budgets: Sequence[int],
    num_queries: int,
) -> Optional[tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]]:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["signature"].item()) != signature:
            return None
        completed = archive["completed"].astype(np.bool_, copy=True)
        if completed.shape != (int(num_queries),):
            return None
        estimates = {
            int(budget): archive[f"estimate_{int(budget)}"].astype(
                np.int64, copy=True
            )
            for budget in budgets
        }
        log_selectivities = {
            int(budget): archive[f"log_selectivity_{int(budget)}"].astype(
                np.float64, copy=True
            )
            for budget in budgets
        }
    return completed, estimates, log_selectivities


@torch.inference_mode()
def evaluate_progressive_sampling(
    *,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    dataset: AlignedTabularDataset,
    cached_order: Optional[Sequence[int]],
    workload: Mapping[str, Any],
    workload_sha256: str,
    allowed_masks_cpu: torch.Tensor,
    device: torch.device,
    dtype_name: str,
    budgets: Sequence[int],
    sampling_seed: int,
    query_batch_size: int,
    particle_chunk_size: int,
    output_dir: Path,
    force: bool,
    progress_every: int,
) -> Dict[str, Any]:
    config = dict(checkpoint["config"])
    method = checkpoint_method(checkpoint)
    selected_dtype_name = str(config["dtype"]) if dtype_name == "config" else dtype_name
    selected_dtype = resolve_autocast_dtype(selected_dtype_name, device)
    source_sha256 = {
        "evaluator": sha256_file(Path(__file__).resolve()),
        "query_evaluation": sha256_file(QUERY_EVALUATION_SOURCE),
        "common": sha256_file(Path(__file__).with_name("common.py")),
    }
    signature = canonical_sha256(
        {
            "protocol": SAMPLER_PROTOCOL,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "workload_sha256": workload_sha256,
            "budgets": list(map(int, budgets)),
            "dtype": selected_dtype_name,
            "sampling_seed": int(sampling_seed),
            "query_batch_size": int(query_batch_size),
            "particle_chunk_size": int(particle_chunk_size),
            "source_sha256": source_sha256,
        }
    )
    result_path = output_dir / f"{method}.json"
    if result_path.is_file() and not force:
        with result_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("evaluation_signature") == signature:
            print(f"compatible result reused: {result_path}", flush=True)
            return existing

    model = build_model(dataset, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    sampler = IncrementalAlignedTabularSampler(model)
    allowed_masks = allowed_masks_cpu.to(device=device, non_blocking=True)
    queries = workload["queries"]
    num_queries = len(queries)
    num_rows = int(workload["dataset_fingerprint"]["num_rows"])
    max_budget = max(map(int, budgets))
    partial = None if force else _load_partial(
        _partial_path(output_dir, method),
        signature=signature,
        budgets=budgets,
        num_queries=num_queries,
    )
    if partial is None:
        completed = np.zeros(num_queries, dtype=np.bool_)
        estimates = {
            int(budget): np.full(num_queries, -1, dtype=np.int64)
            for budget in budgets
        }
        log_selectivities = {
            int(budget): np.full(num_queries, np.nan, dtype=np.float64)
            for budget in budgets
        }
    else:
        completed, estimates, log_selectivities = partial

    started = time.time()
    processed_batches = 0
    for query_start in range(0, num_queries, int(query_batch_size)):
        query_stop = min(query_start + int(query_batch_size), num_queries)
        if bool(completed[query_start:query_stop].all()):
            continue
        if bool(completed[query_start:query_stop].any()):
            raise ValueError("partial progress must end on a query-batch boundary")
        query_count = query_stop - query_start
        batch_allowed = allowed_masks[query_start:query_stop]
        log_sums = np.full((query_count, len(budgets)), -np.inf, dtype=np.float64)
        value_generator = _new_generator(
            device,
            _seed_for_query_batch(sampling_seed, query_start),
        )
        order_generator = _new_generator(device, _seed_for_query_batch(sampling_seed + 2_000_003, query_start))
        for particle_start in range(0, max_budget, int(particle_chunk_size)):
            particle_stop = min(
                particle_start + int(particle_chunk_size),
                max_budget,
            )
            particles = particle_stop - particle_start
            paths = query_count * particles
            path_query_ids = torch.arange(
                query_count,
                dtype=torch.long,
                device=device,
            ).repeat_interleave(particles)
            orders = evaluation_orders(cached_order, paths, dataset.num_variables, device, order_generator)
            uniforms = torch.rand(
                paths,
                dataset.num_variables,
                dtype=torch.float32,
                device=device,
                generator=value_generator,
            )
            cache = sampler.new_cache(paths)

            def next_logits(
                step: int,
                previous_values: Optional[torch.Tensor],
            ) -> torch.Tensor:
                return sampler.next_logits(orders, step, previous_values, cache)

            with autocast_context(device, selected_dtype):
                sampled = progressive_sample_log_weights(
                    next_logits=next_logits,
                    orders=orders,
                    allowed_masks=batch_allowed,
                    path_query_ids=path_query_ids,
                    uniforms=uniforms,
                )
            chunk = sampled.log_weights.view(query_count, particles)
            for budget_id, budget in enumerate(budgets):
                overlap_stop = min(particle_stop, int(budget))
                if overlap_stop <= particle_start:
                    continue
                segment = torch.logsumexp(
                    chunk[:, : overlap_stop - particle_start],
                    dim=1,
                ).cpu().numpy()
                log_sums[:, budget_id] = np.logaddexp(
                    log_sums[:, budget_id],
                    segment,
                )
            del sampled, chunk, cache, uniforms, orders, path_query_ids

        for budget_id, budget in enumerate(budgets):
            log_selectivity = log_sums[:, budget_id] - math.log(float(budget))
            estimated = np.ceil(np.exp(log_selectivity) * float(num_rows)).astype(
                np.int64
            )
            estimates[int(budget)][query_start:query_stop] = estimated
            log_selectivities[int(budget)][query_start:query_stop] = log_selectivity
        completed[query_start:query_stop] = True
        _save_partial(
            _partial_path(output_dir, method),
            signature=signature,
            completed=completed,
            estimates=estimates,
            log_selectivities=log_selectivities,
        )
        processed_batches += 1
        if progress_every > 0 and processed_batches % int(progress_every) == 0:
            print(
                f"queries {query_stop:,}/{num_queries:,}; "
                f"{int(completed.sum()) / max(time.time() - started, 1e-9):.2f} q/s",
                flush=True,
            )

    actual = np.asarray(
        [int(query["actual_cardinality"]) for query in queries],
        dtype=np.int64,
    )
    budget_results: Dict[str, Any] = {}
    for budget in budgets:
        current_estimates = estimates[int(budget)]
        if bool((current_estimates < 0).any()):
            raise RuntimeError("progressive evaluation has missing estimates")
        errors = np.asarray(
            [
                q_error(int(estimate), int(truth))
                for estimate, truth in zip(current_estimates, actual)
            ],
            dtype=np.float64,
        )
        budget_results[str(int(budget))] = {
            "particle_budget": int(budget),
            "metrics": summarize_q_errors(errors),
            "estimated_cardinalities": current_estimates.tolist(),
            "log_estimated_selectivities": log_selectivities[int(budget)].tolist(),
            "q_errors": errors.tolist(),
        }
    result = {
        "method": method,
        "dataset": "census",
        "sampler_kind": "random_per_particle_all68" if cached_order is None else "vanilla_fixed_order_progressive_sampling",
        "sampler_protocol": SAMPLER_PROTOCOL,
        "evaluation_signature": signature,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iter_num", -1)),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "canonical_workload_sha256": workload["canonical_workload_sha256"],
        "remapped_workload_sha256": workload_sha256,
        "source_sha256": source_sha256,
        "variable_permutation_seed": int(dataset.variable_permutation_seed),
        "current_to_canonical": dataset.current_to_canonical.tolist(),
        "cached_order_current_ids": list(map(int, cached_order)) if cached_order is not None else None,
        "data_dir": str(dataset.data_dir),
        "nested_particle_budgets": True,
        "budgets": list(map(int, budgets)),
        "sampling_seed": int(sampling_seed),
        "num_queries": num_queries,
        "num_table_rows": num_rows,
        "dtype": selected_dtype_name,
        "device": str(device),
        "query_batch_size": int(query_batch_size),
        "particle_chunk_size": int(particle_chunk_size),
        "elapsed_seconds": time.time() - started,
        "budget_results": budget_results,
    }
    write_json(result_path, result)
    _partial_path(output_dir, method).unlink(missing_ok=True)
    del sampler, model, allowed_masks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def write_tables(
    output_dir: Path,
    result: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "method",
            "particle_budget",
            "p50_q_error",
            "p99_q_error",
            "max_q_error",
            "num_queries",
            "checkpoint_iteration",
            "num_parameters",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for budget in result["budgets"]:
            metrics = result["budget_results"][str(budget)]["metrics"]
            writer.writerow(
                {
                    "method": result["method"],
                    "particle_budget": budget,
                    "p50_q_error": f"{metrics['p50']:.12g}",
                    "p99_q_error": f"{metrics['p99']:.12g}",
                    "max_q_error": f"{metrics['max']:.12g}",
                    "num_queries": result["num_queries"],
                    "checkpoint_iteration": result["checkpoint_iteration"],
                    "num_parameters": result["num_parameters"],
                }
            )
    with (output_dir / "per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "particle_budget",
            "query_id",
            "num_filters",
            "actual_cardinality",
            "estimated_cardinality",
            "q_error",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for budget in result["budgets"]:
            budget_result = result["budget_results"][str(budget)]
            for query_id, query in enumerate(queries):
                writer.writerow(
                    {
                        "particle_budget": budget,
                        "query_id": query_id,
                        "num_filters": query["num_filters"],
                        "actual_cardinality": query["actual_cardinality"],
                        "estimated_cardinality": budget_result[
                            "estimated_cardinalities"
                        ][query_id],
                        "q_error": f"{budget_result['q_errors'][query_id]:.12g}",
                    }
                )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    budgets = tuple(sorted(set(args.budget or (10000,))))
    if any(int(budget) <= 0 for budget in budgets):
        raise ValueError("particle budgets must be positive")
    checkpoint_path, checkpoint = load_checkpoint(args.checkpoint)
    embedded = dict(checkpoint["config"])
    dataset = AlignedTabularDataset(
        resolve_repo_path(args.data_dir if args.data_dir is not None else str(embedded["data_dir"])),
        int(embedded["variable_permutation_seed"]),
    )
    if dataset.dataset_name.lower() != "census":
        raise ValueError(f"expected CENSUS data, got {dataset.dataset_name!r}")
    _, cached_order = validate_core_checkpoint(
        checkpoint,
        dataset_name="census",
        width=dataset.num_variables,
    )
    validate_model_frame(checkpoint, dataset)
    canonical_workload, canonical_sha = load_canonical_workload(
        args.workload,
        num_queries=args.num_queries,
    )
    remapped, remapped_sha = remap_workload_to_current_ids(
        canonical_workload,
        dataset,
        canonical_sha,
    )
    audit = verify_remapped_cardinalities(dataset, remapped)
    output_dir = resolve_repo_path(args.output_dir)
    write_json(
        output_dir / "frame_remap_audit.json",
        {
            "protocol": REMAP_PROTOCOL,
            "canonical_workload": str(resolve_repo_path(args.workload)),
            "canonical_workload_sha256": canonical_sha,
            "remapped_workload_sha256": remapped_sha,
            "variable_permutation_seed": int(dataset.variable_permutation_seed),
            "current_to_canonical": dataset.current_to_canonical.tolist(),
            "canonical_to_current": dataset.canonical_to_current.tolist(),
            "oracle": audit,
        },
    )
    allowed_masks = torch.from_numpy(
        build_allowed_masks(
            remapped["queries"],
            dataset.cardinalities.tolist(),
            dataset.max_states,
        )
    ).bool()
    result = evaluate_progressive_sampling(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        dataset=dataset,
        cached_order=cached_order,
        workload=remapped,
        workload_sha256=remapped_sha,
        allowed_masks_cpu=allowed_masks,
        device=resolve_device(args.device),
        dtype_name=args.dtype,
        budgets=budgets,
        sampling_seed=args.sampling_seed,
        query_batch_size=args.query_batch_size,
        particle_chunk_size=args.particle_chunk_size,
        output_dir=output_dir,
        force=args.force,
        progress_every=args.progress_every,
    )
    write_tables(output_dir, result, remapped["queries"])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "budgets": list(budgets),
                "oracle_verified": audit["verified"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
