from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _attention_stats(prefix: str, matrix: torch.Tensor) -> Dict[str, float]:
    values = matrix.detach().cpu().to(dtype=torch.float64)
    nonzero = values > 0
    return {
        f"{prefix}/input_attention_mean": float(values.mean().item()),
        f"{prefix}/input_attention_std": float(values.std().item()),
        f"{prefix}/input_attention_max": float(values.max().item()),
        f"{prefix}/input_attention_nonzero_fraction": float(
            nonzero.to(dtype=torch.float64).mean().item()
        ),
    }


def save_attention_matrix(
    out_dir: str | Path,
    iteration: int,
    matrix: torch.Tensor,
) -> str:


    directory = Path(out_dir) / "teacher_attention"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f"iter_{int(iteration):07d}_input_attention_current_ids.npy"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    array = matrix.detach().cpu().to(dtype=torch.float32).numpy()
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, destination)
    return str(destination.relative_to(Path(out_dir)))


def log_teacher_update(
    run: Any,
    record: Mapping[str, Any],
    matrix: torch.Tensor,
    *,
    phase: str,
    scheduled_probability: float,
    effective_probability: float,
    cache_frozen: bool,
    order_version: int,
) -> None:


    spectral = record.get("spectral", {})
    orientation = record.get("orientation", {})
    payload: Dict[str, Any] = {
        "iter": int(record["iteration"]),
        "teacher/method": "teacher",
        "teacher/phase": str(phase),
        "teacher/affinity": "dense_mutual_geometric",
        "teacher/scheduled_probability": float(scheduled_probability),
        "teacher/effective_probability": float(effective_probability),
        "teacher/updated": float(bool(record.get("updated", False))),
        "teacher/cache_frozen": float(cache_frozen),
        "teacher/order_version": int(order_version),
        "teacher/attention_samples": int(record.get("attention_samples", 0)),
        "teacher/orientation_extra_forward_passes": int(record.get("orientation_extra_forward_passes", 0)),
        "teacher/degenerate": float(
            bool(spectral.get("degenerate", False))
            if isinstance(spectral, Mapping)
            else False
        ),
        **_attention_stats("teacher", matrix),
    }
    for source, destination in (
        ("fiedler_eigenvalue", "fiedler_eigenvalue"),
        ("fiedler_eigengap", "fiedler_eigengap"),
        ("zero_eigenvalue_multiplicity", "zero_eigenvalue_multiplicity"),
    ):
        value = (
            _finite_number(spectral.get(source))
            if isinstance(spectral, Mapping)
            else None
        )
        if value is not None:
            payload[f"teacher/{destination}"] = value
    for source, destination in (
        ("raw_attention_score", "raw_attention_score"),
        ("reverse_attention_score", "reverse_attention_score"),
    ):
        value = (
            _finite_number(orientation.get(source))
            if isinstance(orientation, Mapping)
            else None
        )
        if value is not None:
            payload[f"teacher/{destination}"] = value
    gap = _finite_number(record.get("orientation_gap"))
    if gap is not None:
        payload["teacher/orientation_gap"] = gap
    run.log(payload)


def log_teacher_mlp_update(
    run: Any,
    record: Mapping[str, Any],
    matrix: torch.Tensor,
    *,
    phase: str,
    scheduled_probability: float,
    cache_frozen: bool,
) -> None:


    mlp = record.get("mlp", {})
    payload: Dict[str, Any] = {
        "iter": int(record["iteration"]),
        "teacher_mlp/method": "mlp",
        "teacher_mlp/phase": str(phase),
        "teacher_mlp/scheduled_probability": float(scheduled_probability),
        "teacher_mlp/supervision_active": float(
            bool(record.get("supervision_active", False))
        ),
        "teacher_mlp/supervision_valid": float(
            bool(record.get("supervision_valid", False))
        ),
        "teacher_mlp/supervision_status": str(
            record.get("supervision_status", "unknown")
        ),
        "teacher_mlp/cache_refreshed": float(bool(record.get("updated", False))),
        "teacher_mlp/cache_frozen": float(cache_frozen),
        "teacher_mlp/attention_samples": int(record.get("attention_samples", 0)),
        "teacher_mlp/orientation_extra_forward_passes": int(
            record.get("orientation_extra_forward_passes", 0)
        ),
        **_attention_stats("teacher_mlp", matrix),
    }
    if isinstance(mlp, Mapping):
        for key in (
            "parameter_count",
            "parameter_updates",
            "order_refreshes",
            "parameters_frozen",
            "learning_rate",
            "distill_loss_pre_step",
            "grad_norm_pre_clip",
            "mse",
            "mae",
            "pearson",
            "priority_pair_tau",
            "exact_position_fraction",
        ):
            value = _finite_number(mlp.get(key))
            if value is not None:
                payload[f"teacher_mlp/{key}"] = value
    run.log(payload)


__all__ = [
    "log_teacher_mlp_update",
    "log_teacher_update",
    "save_attention_matrix",
]
