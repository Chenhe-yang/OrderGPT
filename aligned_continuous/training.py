from __future__ import annotations

import ast
import json
import math
import os
import random
import runpy
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from aligned_continuous.data import AlignedContinuousDataset
from aligned_continuous.model import AlignedContinuousConfig, AlignedContinuousOrderGPT
from aligned_continuous.order_mlp import (
    AlignedContinuousOrderMLP,
    FiedlerMLPRuntime,
    build_fiedler_priority_target,
    priority_mse_diagnostics,
    priority_to_order,
)
from aligned_continuous.wandb_logging import (
    log_teacher_mlp_update,
    log_teacher_update,
    save_attention_matrix,
)
from aligned_continuous.order_teacher import (
    AttentionSampleAccumulator,
    CompletedAttentionWindow,
    FiedlerOrderResult,
    OrientationResult,
    orient_raw_reverse_from_attention,
    recover_fiedler_order,
)
from aligned_continuous.sample_teacher_schedule import SampleTeacherSchedule


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "config" / "aligned_continuous_base.py"
VALID_METHODS = {"teacher", "mlp", "random", "canonical", "fixed", "fixed_curriculum"}


ORDERGPT_PROTOCOL: Dict[str, Any] = {
    "order_mode": "strict_fiedler",
    "teacher_trigger_mode": "eligible_sample_window",
    "teacher_attention_source": "all_training_orders",
    "teacher_attention_layers": "all",
    "teacher_attention_view": "target_to_observed",
    "teacher_attention_aggregation_backend": "vectorized",
    "teacher_attention_observer_chunk_size": 0,
    "teacher_affinity_symmetrization": "mutual_geometric",
    "teacher_affinity_row_topk": 0,
    "teacher_orientation_score": "directed_attention_strict_triangular_sum",
    "teacher_orientation_tie_break": "lexicographic_current_ids",
    "teacher_isolate_rng": True,
    "teacher_require_cache_at_freeze": True,
}
ORDERGPT_MLP_PROTOCOL: Dict[str, Any] = {
    "teacher_mlp_dropout": 0.0,
    "teacher_mlp_activation": "gelu",
    "teacher_mlp_input_normalization": "zscore",
    "teacher_mlp_target": "oriented_minmax_fiedler_priority",
    "teacher_mlp_order_mode": "argsort_desc",
}


@dataclass(frozen=True)
class TrainingTaskAdapter:


    build_dataset: Callable[[Mapping[str, Any]], Any]
    build_model: Callable[[Any, Mapping[str, Any]], torch.nn.Module]
    evaluate: Callable[..., Dict[str, float]]
    validate_config: Optional[Callable[[Mapping[str, Any]], None]] = None


def _public_config_values(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    namespace = runpy.run_path(str(path))
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("_") and not callable(value)
    }


def load_config(argv: Sequence[str]) -> Dict[str, Any]:
    config_args = [arg for arg in argv if not arg.startswith("--")]
    overrides = [arg for arg in argv if arg.startswith("--")]
    if len(config_args) != 1:
        raise ValueError("expected one config path followed by optional --key=value overrides")
    config_path = Path(config_args[0])
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    defaults = _public_config_values(BASE_CONFIG)
    selected = _public_config_values(config_path)
    unknown = sorted(set(selected).difference(defaults))
    if unknown:
        raise ValueError("unknown config keys: " + ", ".join(unknown))
    config = dict(defaults)
    config.update(selected)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must have form --key=value: {override}")
        key, raw = override[2:].split("=", 1)
        if key not in config:
            raise ValueError(f"unknown config override: {key}")
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            value = raw
        expected = config[key]
        if isinstance(expected, bool):
            if not isinstance(value, bool):
                raise TypeError(f"override {key} must be bool")
        elif isinstance(expected, int) and not isinstance(expected, bool):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"override {key} must be int")
        elif isinstance(expected, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"override {key} must be numeric")
            value = float(value)
        elif not isinstance(value, type(expected)):
            raise TypeError(f"override {key} must be {type(expected).__name__}")
        config[key] = value
    config["config_path"] = str(config_path.resolve())
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config["expected_condition_mode"], str):
        raise TypeError("expected_condition_mode must be a string")
    method = str(config["method"])
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(VALID_METHODS)}"
        )
    positive_ints = (
        "n_layer", "n_head", "n_embd", "batch_size",
        "gradient_accumulation_steps", "max_iters", "eval_interval",
        "log_interval", "lr_decay_iters", "attention_window_samples",
    )
    for key in positive_ints:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["condition_layers"]) < 0:
        raise ValueError("condition_layers must be non-negative")
    if int(config["n_embd"]) % int(config["n_head"]) != 0:
        raise ValueError("n_embd must be divisible by n_head")
    for key in (
        "eval_batches",
        "test_eval_batches",
        "warmup_iters",
        "best_selection_start_iter",
    ):
        if int(config[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if float(config["learning_rate"]) <= 0.0 or float(config["min_lr"]) < 0.0:
        raise ValueError("learning rates are invalid")
    samples_per_step = int(config["batch_size"]) * int(config["gradient_accumulation_steps"])
    milestones = tuple(int(config[key]) for key in ("distill_start", "distill_end", "anneal_end"))
    if not 0 <= milestones[0] < milestones[1] < milestones[2]:
        raise ValueError("require 0 <= distill_start < distill_end < anneal_end")
    if method in {"teacher", "mlp", "fixed_curriculum"} and milestones[2] > int(config["max_iters"]):
        raise ValueError("anneal_end exceeds the training budget")
    boundaries = tuple(value * samples_per_step for value in milestones)
    for name, value in zip(("_distill_start_samples", "_distill_end_samples", "_anneal_end_samples"), boundaries):
        config[name] = value
    target_samples = int(config["attention_window_samples"])
    if method in {"teacher", "mlp"}:
        if target_samples % samples_per_step:
            raise ValueError("attention_window_samples must contain whole optimizer updates")
        if any((boundary - boundaries[0]) % target_samples for boundary in boundaries[1:]):
            raise ValueError("distill_end and anneal_end must end on exact attention windows")
    if method == "mlp":
        hidden_dims = config["mlp_hidden_dims"]
        if not isinstance(hidden_dims, (tuple, list)) or not hidden_dims:
            raise ValueError("mlp_hidden_dims must be a non-empty sequence")
        if any(int(value) <= 0 for value in hidden_dims):
            raise ValueError("mlp_hidden_dims entries must be positive")
        if float(config["mlp_learning_rate"]) <= 0.0:
            raise ValueError("mlp_learning_rate must be positive")
        if float(config["mlp_weight_decay"]) < 0.0:
            raise ValueError("mlp_weight_decay must be non-negative")
        for key in ("mlp_beta1", "mlp_beta2"):
            if not 0.0 <= float(config[key]) < 1.0:
                raise ValueError(f"{key} must be in [0, 1)")
        if float(config["mlp_grad_clip"]) < 0.0:
            raise ValueError("mlp_grad_clip must be non-negative")
        if int(config["mlp_seed_offset"]) < 0:
            raise ValueError("mlp_seed_offset must be non-negative")
    if str(config["dtype"]) not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    if str(config["wandb_mode"]) not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled")
    if str(config["eval_only_split"]) not in {"train", "discovery", "val", "test"}:
        raise ValueError("eval_only_split must name a prepared data split")


def resolve_device(requested: str) -> torch.device:
    name = str(requested).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_autocast_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if name == "float32" or device.type == "cpu":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    return nullcontext() if dtype is None else torch.autocast(device_type=device.type, dtype=dtype)


def validate_dataset_condition_mode(
    config: Mapping[str, Any], dataset: AlignedContinuousDataset
) -> str:


    expected = str(config.get("expected_condition_mode", "")).strip()
    observed_value = dataset.metadata.get("condition_mode")
    observed = "" if observed_value is None else str(observed_value).strip()
    if not observed:


        masking = dataset.metadata.get("masking")
        if isinstance(masking, Mapping) and str(masking.get("protocol", "")) == (
            "official_h3wb_i2d"
        ):
            observed = "official_i2d"
    if not expected:
        return observed
    if not observed:
        raise ValueError(
            "prepared metadata has no condition_mode, but the config requires "
            f"expected_condition_mode={expected!r}"
        )
    if observed != expected:
        raise ValueError(
            "prepared condition_mode mismatch: config requires "
            f"{expected!r}, dataset provides {observed!r}"
        )
    return observed


def random_orders(batch_size: int, width: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    scores = torch.rand(batch_size, width, generator=generator, device="cpu")
    return torch.argsort(scores, dim=1).to(device)


def validate_order(values: Sequence[int], width: int) -> Tuple[int, ...]:
    order = tuple(int(value) for value in values)
    if len(order) != width or sorted(order) != list(range(width)):
        raise ValueError(f"order must be a permutation of 0..{width - 1}")
    return order


class OrderController:
    def __init__(
        self,
        config: Mapping[str, Any],
        width: int,
        generator: torch.Generator,
        current_to_canonical: Optional[Sequence[int]] = None,
    ):
        self.config = config
        self.width = int(width)
        self.generator = generator
        self.cached_order: Optional[Tuple[int, ...]] = None
        self.teacher_history: List[Dict[str, Any]] = []


        method = str(config["method"])
        if method == "canonical":
            if current_to_canonical is None:
                raise ValueError("canonical baseline needs the declared coordinate mapping")
            self.cached_order = tuple(np.argsort(current_to_canonical).tolist())
        elif method in {"fixed", "fixed_curriculum"}:
            fixed_generator = torch.Generator().manual_seed(int(config["seed"]) + 2_000_003)
            self.cached_order = tuple(torch.randperm(self.width, generator=fixed_generator).tolist())
        self.sample_schedule = SampleTeacherSchedule.from_config(config)

    def _sample_schedule_snapshot_for_iteration(self, iteration: int):
        samples_per_step = int(self.config["batch_size"]) * int(
            self.config["gradient_accumulation_steps"]
        )
        return self.sample_schedule.snapshot(int(iteration) * samples_per_step)

    def teacher_scheduled_probability(self, iteration: int) -> float:
        if self.config["method"] in {"canonical", "fixed"}:
            return 1.0
        if self.config["method"] == "random":
            return 0.0
        return float(
            self._sample_schedule_snapshot_for_iteration(
                iteration
            ).learned_order_probability
        )

    def teacher_probability(self, iteration: int) -> float:
        probability = self.teacher_scheduled_probability(iteration)
        if self.cached_order is None:
            if probability >= 1.0:
                raise RuntimeError("final order is missing at anneal_end")
            return 0.0
        return probability

    def teacher_fixed_start_iteration(self) -> int:
        samples_per_step = int(self.config["batch_size"]) * int(
            self.config["gradient_accumulation_steps"]
        )
        return (
            int(self.config["_anneal_end_samples"])
            // samples_per_step
        )

    def teacher_final_refresh_iteration(self) -> int:


        samples_per_step = int(self.config["batch_size"]) * int(
            self.config["gradient_accumulation_steps"]
        )
        return (
            int(self.config["_anneal_end_samples"])
            // samples_per_step
            - 1
        )

    def teacher_cache_frozen(self, iteration: int) -> bool:
        snapshot = self._sample_schedule_snapshot_for_iteration(iteration)
        return not bool(snapshot.cache_updates_allowed) and int(iteration) >= (
            int(self.config["_anneal_end_samples"])
            // (
                int(self.config["batch_size"])
                * int(self.config["gradient_accumulation_steps"])
            )
        )

    def teacher_phase(self, iteration: int) -> str:
        return str(
            self._sample_schedule_snapshot_for_iteration(iteration).phase
        )

    def make(self, batch_size: int, device: torch.device, iteration: int, training: bool = True) -> torch.Tensor:
        probability = self.teacher_probability(iteration)
        if self.cached_order is None or probability <= 0.0:
            return random_orders(batch_size, self.width, device, self.generator)
        fixed = (
            torch.tensor(self.cached_order, dtype=torch.long, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        )
        if not training or probability >= 1.0:
            return fixed
        random_batch = random_orders(
            batch_size, self.width, device, self.generator
        )
        use_teacher = (
            torch.rand(batch_size, generator=self.generator) < probability
        )
        return torch.where(
            use_teacher.to(device).view(-1, 1), fixed, random_batch
        )


def learning_rate(iteration: int, config: Mapping[str, Any]) -> float:
    warmup = int(config["warmup_iters"])
    if warmup > 0 and iteration < warmup:
        return float(config["learning_rate"]) * float(iteration + 1) / float(warmup)
    if iteration >= int(config["lr_decay_iters"]):
        return float(config["min_lr"])
    ratio = (iteration - warmup) / max(1, int(config["lr_decay_iters"]) - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * max(0.0, ratio)))
    return float(config["min_lr"]) + coefficient * (float(config["learning_rate"]) - float(config["min_lr"]))


@torch.no_grad()
def evaluate(
    model: AlignedContinuousOrderGPT,
    dataset: AlignedContinuousDataset,
    split: str,
    config: Mapping[str, Any],
    controller: OrderController,
    device: torch.device,
    dtype: Optional[torch.dtype],
    iteration: int,
    max_batches: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    saved_order_rng_state = controller.generator.get_state()
    total_samples = 0
    total_step_nll = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_l2_error = 0.0
    original_squared_error = 0.0
    original_absolute_error = 0.0
    original_l2_error = 0.0
    target_std: Optional[torch.Tensor] = None
    normalization = dataset.metadata.get("normalization")
    if isinstance(normalization, Mapping):
        target_info = normalization.get("targets")
        raw_std: Any = None
        if isinstance(target_info, Mapping) and "std" in target_info:
            raw_std = target_info["std"]
        elif "target_std" in normalization:
            raw_std = normalization["target_std"]
        elif "std" in normalization:
            raw_std = normalization["std"]
        if raw_std is not None:
            std_array = np.asarray(raw_std, dtype=np.float32)
            expected_shape = (dataset.num_variables, dataset.value_dim)
            if std_array.shape == (dataset.value_dim,):
                std_array = np.broadcast_to(std_array, expected_shape).copy()
            elif std_array.shape == () and dataset.value_dim == 1:
                std_array = np.full(expected_shape, float(std_array), dtype=np.float32)
            if std_array.shape == expected_shape and np.isfinite(std_array).all():
                std_array = std_array[dataset.variable_permutation.numpy()]
                target_std = torch.from_numpy(np.ascontiguousarray(std_array)).to(device)
    try:
        for batch_id, batch in enumerate(dataset.iter_batches(split, int(config["batch_size"]), device=device)):
            if max_batches > 0 and batch_id >= max_batches:
                break
            orders = controller.make(int(batch.targets.shape[0]), device, iteration, training=False)
            with autocast_context(device, dtype):
                output = model(batch.targets, batch.conditions, orders)
            difference = output.means_current.float() - batch.targets.float()
            samples = int(batch.targets.shape[0])
            total_samples += samples
            total_step_nll += float(output.step_losses.float().sum().item())
            total_squared_error += float(difference.square().sum().item())
            total_absolute_error += float(difference.abs().sum().item())
            total_l2_error += float(torch.linalg.vector_norm(difference, dim=-1).sum().item())
            if target_std is not None:
                original_difference = difference * target_std.unsqueeze(0)
                original_squared_error += float(original_difference.square().sum().item())
                original_absolute_error += float(original_difference.abs().sum().item())
                original_l2_error += float(
                    torch.linalg.vector_norm(original_difference, dim=-1).sum().item()
                )
    finally:


        controller.generator.set_state(saved_order_rng_state)
        if was_training:
            model.train()
    if total_samples == 0:
        raise ValueError(f"evaluation split {split!r} yielded no batches")
    variables = total_samples * dataset.num_variables
    scalars = variables * dataset.value_dim
    metrics = {
        "teacher_forced_nll_per_variable": total_step_nll / variables,
        "teacher_forced_nll_per_scalar": total_step_nll / scalars,
        "teacher_forced_rmse_normalized": math.sqrt(total_squared_error / scalars),
        "teacher_forced_mae_normalized": total_absolute_error / scalars,
        "teacher_forced_mean_l2_normalized": total_l2_error / variables,
        "num_samples": float(total_samples),
    }
    if target_std is not None:
        metrics.update(
            {
                "teacher_forced_rmse_original_units": math.sqrt(
                    original_squared_error / scalars
                ),
                "teacher_forced_mae_original_units": original_absolute_error / scalars,
                "teacher_forced_mean_l2_original_units": original_l2_error / variables,
            }
        )
    return metrics


def _attention_tensor(attentions: List[torch.Tensor], layers: str) -> torch.Tensor:
    if not attentions:
        raise RuntimeError("model returned no attention matrices")
    if str(layers) != "all":
        raise ValueError("OrderGPT attention capture requires every layer")
    return torch.cat(attentions, dim=1)


@dataclass(frozen=True)
class OrientedFiedlerSupervision:
    spectral: FiedlerOrderResult
    orientation: Optional[OrientationResult]
    orientation_gap: Optional[float]


@torch.no_grad()
def compute_oriented_fiedler_supervision(
    matrix: torch.Tensor,
) -> OrientedFiedlerSupervision:
    spectral = recover_fiedler_order(matrix)
    if spectral.degenerate:
        return OrientedFiedlerSupervision(spectral, None, None)
    orientation = orient_raw_reverse_from_attention(spectral.raw_order, matrix)
    gap = abs(orientation.raw_attention_score - orientation.reverse_attention_score)
    return OrientedFiedlerSupervision(spectral, orientation, gap)


def _validate_completed_teacher_window(
    config: Mapping[str, Any], window: CompletedAttentionWindow
) -> None:
    target_samples = int(config["attention_window_samples"])
    if int(window.num_samples) != target_samples:
        raise RuntimeError(
            "completed Teacher attention window did not contain exactly "
            f"{target_samples} samples"
        )
    if int(window.last_source_sample) >= int(
        config["_anneal_end_samples"]
    ):
        raise RuntimeError(
            "Teacher attention window contains a source sample at/after update_stop"
        )


def _attention_window_record(
    config: Mapping[str, Any],
    iteration: int,
    window: CompletedAttentionWindow,
    train_samples_seen_after_step: int,
) -> Dict[str, Any]:
    samples_per_step = int(config["batch_size"]) * int(
        config["gradient_accumulation_steps"]
    )
    return {
        "iteration": int(iteration),
        "attention_split": "train",
        "attention_samples": int(window.num_samples),
        "attention_trigger_mode": "eligible_sample_window",
        "attention_source": "ordinary_training_forward/all_training_orders",
        "attention_layers": "all",
        "attention_view": "target_to_observed",
        "attention_matrices_per_sample": int(window.num_heads),
        "window_index": int(window.window_index),
        "window_first_source_sample": int(window.first_source_sample),
        "window_last_source_sample": int(window.last_source_sample),
        "window_start_iteration": int(
            window.first_source_sample // samples_per_step
        ),
        "window_end_iteration": int(window.last_source_sample // samples_per_step),
        "window_exact_target_met": True,
        "eligible_samples_seen": int(window.eligible_samples_seen_at_completion),
        "train_samples_seen_after_step": int(train_samples_seen_after_step),
        "final_window_before_freeze": bool(
            int(window.last_source_sample) + 1
            == int(config["_anneal_end_samples"])
        ),
        "updated": False,
    }


@torch.no_grad()
def update_teacher_from_attention_window(
    config: Mapping[str, Any],
    controller: OrderController,
    iteration: int,
    window: CompletedAttentionWindow,
    *,
    train_samples_seen_after_step: int,
    update_callback: Optional[
        Callable[[Dict[str, Any], torch.Tensor, torch.Tensor], None]
    ] = None,
) -> Dict[str, Any]:


    if str(config["method"]) != "teacher":
        raise ValueError("direct attention-window update requires method='teacher'")
    _validate_completed_teacher_window(config, window)
    matrix = window.matrix.detach().cpu().to(dtype=torch.float64)
    supervision = compute_oriented_fiedler_supervision(matrix)
    record = _attention_window_record(
        config, iteration, window, train_samples_seen_after_step
    )
    record["spectral"] = supervision.spectral.diagnostics
    if supervision.orientation is None:
        record["reason"] = "degenerate_fiedler"
    else:
        record["orientation_source"] = "same_attention_window"
        record["orientation"] = supervision.orientation.to_dict()
        record["orientation_extra_forward_passes"] = 0
        record["orientation_gap"] = supervision.orientation_gap
        controller.cached_order = tuple(
            supervision.orientation.selected_order
        )
        record["updated"] = True
        record["order_current_ids"] = list(controller.cached_order)
    controller.teacher_history.append(record)
    if update_callback is not None:
        update_callback(record, matrix, supervision.spectral.fiedler_vector)
    return record


def update_teacher_mlp_from_attention_window(
    config: Mapping[str, Any],
    controller: OrderController,
    runtime: FiedlerMLPRuntime,
    device: torch.device,
    iteration: int,
    window: CompletedAttentionWindow,
    *,
    train_samples_seen_after_step: int,
    update_callback: Optional[
        Callable[[Dict[str, Any], torch.Tensor], None]
    ] = None,
) -> Dict[str, Any]:


    if str(config["method"]) != "mlp":
        raise ValueError("MLP attention-window update requires method='mlp'")
    _validate_completed_teacher_window(config, window)
    matrix = window.matrix.detach().cpu().to(dtype=torch.float64)
    record = _attention_window_record(
        config, iteration, window, train_samples_seen_after_step
    )
    record["method"] = "mlp"
    distill_start = int(config["_distill_start_samples"])
    distill_stop = int(config["_distill_end_samples"])
    first = int(window.first_source_sample)
    last = int(window.last_source_sample)
    if first < distill_start <= last or first < distill_stop <= last:
        raise RuntimeError("teacher MLP distillation boundary splits a window")
    supervision_active = bool(first >= distill_start and last < distill_stop)
    record["supervision_active"] = supervision_active
    record["supervision_valid"] = False
    record["orientation_extra_forward_passes"] = 0
    target_priority: Optional[torch.Tensor] = None
    loss_value: Optional[float] = None
    grad_norm_value: Optional[float] = None

    mlp_matrix = matrix.to(device=device, dtype=torch.float32)
    if supervision_active:
        if runtime.parameters_frozen or runtime.freeze_transition_done:
            raise RuntimeError("teacher MLP is frozen inside distillation interval")
        supervision = compute_oriented_fiedler_supervision(matrix)
        record["spectral"] = supervision.spectral.diagnostics
        if supervision.orientation is None:


            record["supervision_status"] = "degenerate_fiedler_skipped"
        else:
            record["orientation_source"] = "same_attention_window"
            record["orientation"] = supervision.orientation.to_dict()
            record["orientation_extra_forward_passes"] = 0
            record["orientation_gap"] = supervision.orientation_gap
            target = build_fiedler_priority_target(
                supervision.spectral, supervision.orientation.selected_order
            )
            target_priority = target.priority.to(
                device=device, dtype=torch.float32
            )
            record["target"] = target.diagnostics
            record["supervision_valid"] = True
            record["supervision_status"] = "mse_updated"
            runtime.model.train()
            runtime.optimizer.zero_grad(set_to_none=True)
            loss = runtime.model.mse_loss(mlp_matrix, target_priority)
            loss.backward()
            if float(config["mlp_grad_clip"]) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    runtime.model.parameters(),
                    float(config["mlp_grad_clip"]),
                )
                grad_norm_value = float(grad_norm.detach().cpu().item())
            runtime.optimizer.step()
            runtime.parameter_updates += 1
            runtime.last_parameter_update_train_samples = last + 1
            loss_value = float(loss.detach().cpu().item())
    else:
        record["supervision_status"] = "inactive"

    was_mlp_training = runtime.model.training
    runtime.model.eval()
    with torch.no_grad():
        logits = runtime.model.forward_logits(mlp_matrix).float()
        priority = torch.sigmoid(logits)
        order_tensor = priority_to_order(priority)
    if was_mlp_training and not runtime.parameters_frozen:
        runtime.model.train()
    predicted_order = tuple(int(value) for value in order_tensor.cpu().tolist())
    controller.cached_order = predicted_order
    runtime.order_refreshes += 1
    runtime.last_window_index = int(window.window_index)
    runtime.last_order_refresh_train_samples = last + 1
    runtime.last_logits = logits.detach().cpu()
    runtime.last_priority = priority.detach().cpu()
    runtime.last_order = predicted_order
    metrics: Dict[str, float] = {
        "prediction_mean": float(priority.mean().detach().cpu().item()),
        "prediction_std": float(
            priority.std(unbiased=False).detach().cpu().item()
        ),
    }
    if target_priority is not None:
        runtime.last_target_priority = target_priority.detach().cpu()
        metrics.update(priority_mse_diagnostics(priority, target_priority))
        runtime.last_metrics = dict(metrics)
    record["updated"] = True
    record["order_current_ids"] = list(predicted_order)
    record["mlp"] = {
        "parameter_count": int(runtime.model.parameter_count),
        "parameter_updates": int(runtime.parameter_updates),
        "order_refreshes": int(runtime.order_refreshes),
        "parameters_frozen": bool(runtime.parameters_frozen),
        "learning_rate": float(runtime.optimizer.param_groups[0]["lr"]),
        "distill_loss_pre_step": loss_value,
        "grad_norm_pre_clip": grad_norm_value,
        **metrics,
    }
    controller.teacher_history.append(record)
    if update_callback is not None:
        update_callback(record, matrix)
    return record


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _teacher_mlp_model_config(
    config: Mapping[str, Any], num_variables: int
) -> Dict[str, Any]:
    return {
        "num_variables": int(num_variables),
        "hidden_dims": tuple(int(value) for value in config["mlp_hidden_dims"]),
    }


def _normalize_teacher_mlp_state_for_load(
    state: Mapping[str, Any],
    expected_model_config: Mapping[str, Any],
) -> Dict[str, Any]:


    saved_model_config = state.get("model_config")
    if not isinstance(saved_model_config, Mapping):
        raise ValueError("resume checkpoint teacher MLP model config is missing")
    saved = dict(saved_model_config)
    legacy_fixed = {
        "dropout": ORDERGPT_MLP_PROTOCOL["teacher_mlp_dropout"],
        "activation": ORDERGPT_MLP_PROTOCOL["teacher_mlp_activation"],
        "input_normalization": ORDERGPT_MLP_PROTOCOL[
            "teacher_mlp_input_normalization"
        ],
    }
    allowed = set(expected_model_config).union(legacy_fixed)
    unknown = sorted(set(saved).difference(allowed))
    if unknown:
        raise ValueError(
            "resume checkpoint teacher MLP model config has unknown keys: "
            + ", ".join(unknown)
        )
    normalized_expected = {
        "num_variables": int(expected_model_config["num_variables"]),
        "hidden_dims": tuple(
            int(value) for value in expected_model_config["hidden_dims"]
        ),
    }
    normalized_saved = {
        "num_variables": int(saved.get("num_variables", -1)),
        "hidden_dims": tuple(int(value) for value in saved.get("hidden_dims", ())),
    }
    if normalized_saved != normalized_expected:
        raise ValueError("resume checkpoint teacher MLP model config does not match")
    for key, expected in legacy_fixed.items():
        if key in saved and saved[key] != expected:
            raise ValueError(
                f"resume checkpoint uses unsupported fixed MLP protocol {key}"
            )
    normalized_state = dict(state)
    normalized_state["model_config"] = normalized_expected
    return normalized_state


def build_teacher_mlp_runtime(
    config: Mapping[str, Any],
    num_variables: int,
    device: torch.device,
    seed: int,
) -> FiedlerMLPRuntime:


    model_config = _teacher_mlp_model_config(config, num_variables)


    init_generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(config["mlp_seed_offset"])
    )
    with torch.random.fork_rng(devices=[], enabled=True):


        torch.set_rng_state(init_generator.get_state())
        policy = AlignedContinuousOrderMLP(
            num_variables=int(num_variables),
            hidden_dims=model_config["hidden_dims"],
        )
    policy = policy.to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["mlp_learning_rate"]),
        betas=(
            float(config["mlp_beta1"]),
            float(config["mlp_beta2"]),
        ),
        weight_decay=float(config["mlp_weight_decay"]),
    )
    return FiedlerMLPRuntime(
        policy,
        optimizer,
        model_config=model_config,
    )


def _sample_teacher_schedule_signature(
    config: Mapping[str, Any], dataset: AlignedContinuousDataset
) -> Dict[str, Any]:


    signature = {
        "teacher_trigger_mode": ORDERGPT_PROTOCOL["teacher_trigger_mode"],
        "teacher_attention_source": ORDERGPT_PROTOCOL["teacher_attention_source"],
        "attention_window_samples": int(
            config["attention_window_samples"]
        ),
        "teacher_attention_aggregation_backend": ORDERGPT_PROTOCOL[
            "teacher_attention_aggregation_backend"
        ],
        "teacher_attention_observer_chunk_size": ORDERGPT_PROTOCOL[
            "teacher_attention_observer_chunk_size"
        ],
        "teacher_orientation_score": ORDERGPT_PROTOCOL["teacher_orientation_score"],
        "teacher_orientation_tie_break": ORDERGPT_PROTOCOL["teacher_orientation_tie_break"],
        "teacher_attention_layers": ORDERGPT_PROTOCOL["teacher_attention_layers"],
        "teacher_attention_view": ORDERGPT_PROTOCOL["teacher_attention_view"],
        "teacher_affinity_symmetrization": ORDERGPT_PROTOCOL[
            "teacher_affinity_symmetrization"
        ],
        "teacher_affinity_row_topk": ORDERGPT_PROTOCOL[
            "teacher_affinity_row_topk"
        ],
        "distill_start": int(config["distill_start"]),
        "distill_end": int(config["distill_end"]),
        "anneal_end": int(config["anneal_end"]),
        "teacher_require_cache_at_freeze": ORDERGPT_PROTOCOL[
            "teacher_require_cache_at_freeze"
        ],
        "teacher_isolate_rng": ORDERGPT_PROTOCOL["teacher_isolate_rng"],
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
        "num_variables": int(dataset.num_variables),
        "variable_permutation_seed": int(config["variable_permutation_seed"]),
        "current_to_canonical_audit_only": dataset.variable_permutation.tolist(),
    }
    if str(config["method"]) == "mlp":
        signature.update(
            {


                "teacher_policy_mode": "fiedler_mlp",
                "mlp_seed_offset": int(config["mlp_seed_offset"]),
                "mlp_hidden_dims": tuple(
                    int(value) for value in config["mlp_hidden_dims"]
                ),
                "teacher_mlp_dropout": ORDERGPT_MLP_PROTOCOL[
                    "teacher_mlp_dropout"
                ],
                "teacher_mlp_activation": ORDERGPT_MLP_PROTOCOL[
                    "teacher_mlp_activation"
                ],
                "teacher_mlp_input_normalization": ORDERGPT_MLP_PROTOCOL[
                    "teacher_mlp_input_normalization"
                ],
                "mlp_learning_rate": float(
                    config["mlp_learning_rate"]
                ),
                "mlp_weight_decay": float(
                    config["mlp_weight_decay"]
                ),
                "mlp_beta1": float(config["mlp_beta1"]),
                "mlp_beta2": float(config["mlp_beta2"]),
                "mlp_grad_clip": float(config["mlp_grad_clip"]),
                "_distill_start_samples": int(
                    config["_distill_start_samples"]
                ),
                "_distill_end_samples": int(
                    config["_distill_end_samples"]
                ),
                "teacher_mlp_target": ORDERGPT_MLP_PROTOCOL[
                    "teacher_mlp_target"
                ],
                "teacher_mlp_order_mode": ORDERGPT_MLP_PROTOCOL[
                    "teacher_mlp_order_mode"
                ],
            }
        )
    return signature


def _normalize_teacher_schedule_signature(
    signature: Mapping[str, Any],
) -> Dict[str, Any]:


    return dict(signature)


def _checkpoint_payload(
    model: AlignedContinuousOrderGPT,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    controller: OrderController,
    scaler: torch.amp.GradScaler,
    np_rng: np.random.Generator,
    iteration: int,
    best_val: float,
    teacher_np_rng: Optional[np.random.Generator] = None,
    teacher_order_generator: Optional[torch.Generator] = None,
    teacher_runtime_state: Optional[Mapping[str, Any]] = None,
    teacher_mlp_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": np_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "order_generator": controller.generator.get_state(),
    }
    if teacher_np_rng is not None and teacher_np_rng is not np_rng:
        rng_state["teacher_numpy_generator"] = teacher_np_rng.bit_generator.state
    if (
        teacher_order_generator is not None
        and teacher_order_generator is not controller.generator
    ):
        rng_state["teacher_order_generator"] = teacher_order_generator.get_state()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": dict(config),
        "iter_num": int(iteration),
        "best_val_nll": float(best_val),
        "cached_order": None if controller.cached_order is None else list(controller.cached_order),
        "order_frozen": config["method"] != "random" and (iteration + 1 >= int(config["anneal_end"]) or config["method"] in {"canonical", "fixed"}),
        "teacher_history": controller.teacher_history,
        "teacher_runtime_state": (
            None
            if teacher_runtime_state is None
            else dict(teacher_runtime_state)
        ),
        "scaler": scaler.state_dict(),
        "rng_state": rng_state,
    }


    if teacher_mlp_state is not None:
        payload["teacher_mlp_state"] = dict(teacher_mlp_state)
    return payload


def _restore_rng_state(
    state: Mapping[str, Any],
    np_rng: np.random.Generator,
    controller: OrderController,
    teacher_np_rng: Optional[np.random.Generator] = None,
    teacher_order_generator: Optional[torch.Generator] = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    np_rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"].cpu())
    controller.generator.set_state(state["order_generator"].cpu())
    if (
        teacher_np_rng is not None
        and teacher_np_rng is not np_rng
        and "teacher_numpy_generator" in state
    ):
        teacher_np_rng.bit_generator.state = state["teacher_numpy_generator"]
    if (
        teacher_order_generator is not None
        and teacher_order_generator is not controller.generator
        and "teacher_order_generator" in state
    ):
        teacher_order_generator.set_state(
            state["teacher_order_generator"].cpu()
        )
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _init_wandb(config: Mapping[str, Any]):
    if not bool(config["wandb_log"]):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb_log=True requires the wandb package") from error
    run = wandb.init(
        project=str(config["wandb_project"]),
        name=str(config["wandb_run_name"]) or None,
        mode=str(config["wandb_mode"]),
        config=dict(config),
        dir=str(config["out_dir"]),
    )


    run.define_metric("iter")
    run.define_metric("*", step_metric="iter")
    return run


def main(
    dataset_name: Optional[str] = None,
    argv: Optional[Sequence[str]] = None,
    *,
    task_adapter: Optional[TrainingTaskAdapter] = None,
) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("H3WB/CENSUS use one GPU per run; launch independent seeds on separate GPUs")
    config = load_config(sys.argv[1:] if argv is None else argv)
    if task_adapter is not None and task_adapter.validate_config is not None:
        task_adapter.validate_config(config)
    if bool(config["eval_only"]) and str(config["init_from"]) != "resume":
        raise ValueError("eval_only requires init_from='resume' to avoid evaluating random initialization")
    configured_name = str(config["dataset_name"]).strip().lower()
    expected_name = "" if dataset_name is None else str(dataset_name).strip().lower()
    if expected_name and configured_name != expected_name:
        raise ValueError(f"{expected_name} entrypoint requires dataset_name={expected_name!r}, got {configured_name!r}")
    if not str(config["data_dir"]):
        raise ValueError("data_dir must be configured")
    if task_adapter is None:
        dataset = AlignedContinuousDataset(
            config["data_dir"], int(config["variable_permutation_seed"])
        )
    else:
        dataset = task_adapter.build_dataset(config)
    if configured_name and dataset.dataset_name.strip().lower() != configured_name:
        raise ValueError(f"prepared dataset is {dataset.dataset_name!r}, config requests {configured_name!r}")
    validate_dataset_condition_mode(config, dataset)
    if "train" not in dataset.split_names or "val" not in dataset.split_names:
        raise ValueError("prepared data must contain train and val splits")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np_rng = np.random.default_rng(seed)
    order_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    device = resolve_device(str(config["device"]))
    autocast_dtype = resolve_autocast_dtype(str(config["dtype"]), device)

    if task_adapter is None:
        model_config = AlignedContinuousConfig(
            num_variables=dataset.num_variables, value_dim=dataset.value_dim,
            condition_dim=dataset.condition_dim, n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]), n_embd=int(config["n_embd"]),
            condition_layers=int(config["condition_layers"]), dropout=float(config["dropout"]),
            bias=bool(config["bias"]),
            min_log_scale=float(config["min_log_scale"]), max_log_scale=float(config["max_log_scale"]),
        )
        model = AlignedContinuousOrderGPT(model_config).to(device)
        evaluate_task = evaluate
    else:
        model = task_adapter.build_model(dataset, config).to(device)
        evaluate_task = task_adapter.evaluate
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": float(config["weight_decay"])},
            {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=(float(config["beta1"]), float(config["beta2"])),
    )
    controller = OrderController(
        config,
        dataset.num_variables,
        order_generator,
        current_to_canonical=dataset.variable_permutation.tolist(),
    )
    sample_teacher_mode = str(config["method"]) in {"teacher", "mlp"}
    direct_window_teacher_mode = str(config["method"]) == "teacher"
    mlp_window_teacher_mode = str(config["method"]) == "mlp"
    teacher_mlp_runtime: Optional[FiedlerMLPRuntime] = None
    if mlp_window_teacher_mode:
        teacher_mlp_runtime = build_teacher_mlp_runtime(
            config, dataset.num_variables, device, seed
        )
    teacher_accumulator: Optional[AttentionSampleAccumulator] = None
    if sample_teacher_mode:
        teacher_accumulator = AttentionSampleAccumulator(
            dataset.num_variables,
            target_samples=int(config["attention_window_samples"]),
        )
    teacher_np_rng = None
    teacher_order_generator = torch.Generator(device="cpu").manual_seed(
        seed + 2_000_003
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))
    out_dir = Path(str(config["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(out_dir / "resolved_config.json", dict(config))
    start_iter = 0
    best_val = float("inf")
    train_samples_seen = 0
    teacher_cache_frozen = False
    teacher_freeze_transition_done = False
    resume_rng_state: Optional[Mapping[str, Any]] = None
    if str(config["init_from"]) == "resume":
        checkpoint_path = Path(str(config["resume_checkpoint"]) or str(out_dir / "last.pt"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        locked = ("method", "dataset_name", "variable_permutation_seed", "seed", "batch_size",
                  "gradient_accumulation_steps", "distill_start", "distill_end", "anneal_end",
                  "attention_window_samples", "learning_rate", "min_lr", "warmup_iters", "lr_decay_iters",
                  "weight_decay", "beta1", "beta2", "grad_clip", "n_layer", "n_head", "n_embd",
                  "condition_layers", "dropout", "bias", "mlp_hidden_dims")
        for key in locked:
            if key not in saved_config or saved_config[key] != config[key]:
                raise ValueError(f"resume contract mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = int(checkpoint["iter_num"]) + 1
        best_val = float(checkpoint.get("best_val_nll", best_val))
        cached = checkpoint.get("cached_order")
        if cached is not None:
            controller.cached_order = validate_order(cached, dataset.num_variables)
        controller.teacher_history = list(checkpoint.get("teacher_history", []))
        runtime_state = checkpoint.get("teacher_runtime_state")
        if sample_teacher_mode:
            if runtime_state is None:
                raise ValueError(
                    "sample-window resume requires teacher_runtime_state; "
                    "a legacy checkpoint cannot reconstruct the partial window"
                )
            saved_signature = runtime_state.get("schedule_signature")
            if not isinstance(saved_signature, Mapping):
                raise ValueError(
                    "resume checkpoint sample-teacher schedule signature is missing"
                )
            if _normalize_teacher_schedule_signature(
                saved_signature
            ) != _normalize_teacher_schedule_signature(
                _sample_teacher_schedule_signature(config, dataset)
            ):
                raise ValueError(
                    "resume checkpoint sample-teacher schedule does not match"
                )
            train_samples_seen = int(runtime_state["train_samples_seen"])
            expected_samples_seen = start_iter * int(config["batch_size"]) * int(
                config["gradient_accumulation_steps"]
            )
            if train_samples_seen != expected_samples_seen:
                raise ValueError(
                    "resume checkpoint train_samples_seen does not match iter_num"
                )
            teacher_cache_frozen = bool(
                runtime_state.get("cache_frozen", False)
            )
            teacher_freeze_transition_done = bool(
                runtime_state.get("freeze_transition_done", False)
            )
            if teacher_accumulator is None:
                raise RuntimeError("sample Teacher accumulator is not initialized")
            teacher_accumulator.load_state_dict(runtime_state["accumulator"])
            if mlp_window_teacher_mode:
                mlp_state = checkpoint.get("teacher_mlp_state")
                if not isinstance(mlp_state, Mapping):
                    raise ValueError(
                        "method='mlp' resume requires teacher_mlp_state"
                    )
                if teacher_mlp_runtime is None:
                    raise RuntimeError("teacher MLP runtime is not initialized")
                teacher_mlp_runtime.load_state_dict(
                    _normalize_teacher_mlp_state_for_load(
                        mlp_state,
                        teacher_mlp_runtime.model_config,
                    )
                )
                expected_mlp_frozen = train_samples_seen >= int(
                    config["_distill_end_samples"]
                )
                if bool(teacher_mlp_runtime.parameters_frozen) != bool(
                    expected_mlp_frozen
                ):
                    raise ValueError(
                        "resume checkpoint teacher MLP freeze state does not "
                        "match train_samples_seen"
                    )
            elif checkpoint.get("teacher_mlp_state") is not None:
                raise ValueError(
                    "method='teacher' resume cannot load a teacher MLP checkpoint"
                )
        else:
            train_samples_seen = start_iter * int(config["batch_size"]) * int(
                config["gradient_accumulation_steps"]
            )
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        resume_rng_state = checkpoint.get("rng_state")
    elif str(config["init_from"]) != "scratch":
        raise ValueError("init_from must be scratch or resume")

    def current_teacher_runtime_state() -> Optional[Dict[str, Any]]:
        if not sample_teacher_mode:
            return None
        if teacher_accumulator is None or controller.sample_schedule is None:
            raise RuntimeError("sample Teacher runtime is not initialized")
        return {
            "schema_version": 1,
            "trigger_mode": "eligible_sample_window",
            "train_samples_seen": int(train_samples_seen),
            "cache_frozen": bool(teacher_cache_frozen),
            "freeze_transition_done": bool(teacher_freeze_transition_done),
            "accumulator": teacher_accumulator.state_dict(),
            "schedule_signature": _sample_teacher_schedule_signature(
                config, dataset
            ),
            "schedule": controller.sample_schedule.to_dict(),
            "schedule_snapshot": controller.sample_schedule.snapshot(
                train_samples_seen
            ).to_dict(),
            "forbidden_priors_used": False,
        }

    def current_teacher_mlp_state() -> Optional[Dict[str, Any]]:
        if not mlp_window_teacher_mode:
            return None
        if teacher_mlp_runtime is None:
            raise RuntimeError("teacher MLP runtime is not initialized")
        return teacher_mlp_runtime.state_dict()

    run = _init_wandb(config)
    if resume_rng_state is not None:
        _restore_rng_state(
            resume_rng_state,
            np_rng,
            controller,
            teacher_np_rng,
            teacher_order_generator,
        )
    if bool(config["eval_only"]):
        split = str(config["eval_only_split"])
        if split not in dataset.split_names:
            raise ValueError(f"eval_only_split={split!r} is not present in prepared data")
        metrics = evaluate_task(model, dataset, split, config, controller, device, autocast_dtype, start_iter, int(config["eval_batches"]))
        _json_dump(out_dir / f"eval_{split}.json", metrics)
        print(json.dumps({split: metrics}, sort_keys=True))
        if run is not None:
            run.finish()
        return

    model_for_train = torch.compile(model) if bool(config["compile_model"]) else model
    model.train()
    started = time.time()

    def _retain_final_attention(
        record: Dict[str, Any], matrix: torch.Tensor
    ) -> None:
        if record.get("final_window_before_freeze", False):
            record["input_attention_file"] = save_attention_matrix(
                out_dir, int(record["iteration"]), matrix
            )

    def teacher_update_callback(
        record: Dict[str, Any],
        matrix: torch.Tensor,
        _fiedler_vector: torch.Tensor,
    ) -> None:
        _retain_final_attention(record, matrix)
        if run is None:
            return
        order_version = sum(
            int(bool(item.get("updated", False)))
            for item in controller.teacher_history
        )
        event_iteration = int(record["iteration"])
        if controller.sample_schedule is None:
            raise RuntimeError("sample Teacher schedule is not initialized")
        snapshot = controller.sample_schedule.snapshot(
            int(record.get("train_samples_seen_after_step", train_samples_seen))
        )
        event_phase = snapshot.phase
        if record.get("final_window_before_freeze", False):
            event_phase = "final_refresh_and_freeze"
        log_teacher_update(
            run,
            record,
            matrix,
            phase=event_phase,
            scheduled_probability=snapshot.learned_order_probability,
            effective_probability=(
                snapshot.learned_order_probability
                if controller.cached_order is not None
                else 0.0
            ),
            cache_frozen=bool(
                record.get("final_window_before_freeze", False)
                or not snapshot.cache_updates_allowed
            ),
            order_version=order_version,
        )

    def teacher_mlp_update_callback(
        record: Dict[str, Any], matrix: torch.Tensor
    ) -> None:
        _retain_final_attention(record, matrix)
        if run is None:
            return
        if controller.sample_schedule is None:
            raise RuntimeError("sample Teacher schedule is not initialized")
        snapshot = controller.sample_schedule.snapshot(
            int(record.get("train_samples_seen_after_step", train_samples_seen))
        )
        event_phase = snapshot.phase
        if record.get("final_window_before_freeze", False):
            event_phase = "final_refresh_and_freeze"
        log_teacher_mlp_update(
            run,
            record,
            matrix,
            phase=event_phase,
            scheduled_probability=snapshot.learned_order_probability,
            cache_frozen=bool(
                record.get("final_window_before_freeze", False)
                or not snapshot.cache_updates_allowed
            ),
        )

    def persist_teacher_record(record: Mapping[str, Any]) -> None:
        _json_dump(out_dir / "last_teacher_update.json", record)
        if controller.cached_order is not None:
            _json_dump(
                out_dir / "learned_order.json",
                {
                    "schema_version": 1,
                    "coordinate_frame": "current_variable_ids",
                    "dataset": dataset.dataset_name,
                    "num_variables": dataset.num_variables,
                    "variable_permutation_seed": dataset.variable_permutation_seed,
                    "current_to_canonical_audit_only": (
                        dataset.variable_permutation.tolist()
                    ),
                    "order_current_ids": list(controller.cached_order),
                    "selected_with": (
                        "ordinary_training_attention_to_frozen_or_trainable_mlp"
                        if mlp_window_teacher_mode
                        else "ordinary_training_attention_and_attention_orientation"
                    ),
                    "last_teacher_update": dict(record),
                },
            )

    for iteration in range(start_iter, int(config["max_iters"])):
        lr = learning_rate(iteration, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        pending_teacher_windows: List[CompletedAttentionWindow] = []
        for _ in range(int(config["gradient_accumulation_steps"])):
            batch = dataset.sample_batch("train", int(config["batch_size"]), rng=np_rng, device=device)
            microbatch_sample_start = int(train_samples_seen)
            orders = controller.make(int(batch.targets.shape[0]), device, iteration, training=True)
            capture_teacher_attention = False
            if sample_teacher_mode:
                if teacher_accumulator is None or controller.sample_schedule is None:
                    raise RuntimeError("sample Teacher runtime is not initialized")
                snapshot = controller.sample_schedule.snapshot(
                    microbatch_sample_start
                )
                capture_teacher_attention = bool(
                    snapshot.collect_attention and not teacher_cache_frozen
                )
            with autocast_context(device, autocast_dtype):
                output = model_for_train(
                    batch.targets,
                    batch.conditions,
                    orders,
                    return_attentions=capture_teacher_attention,
                )
                loss = output.loss / int(config["gradient_accumulation_steps"])
            if capture_teacher_attention:
                attention = _attention_tensor(
                    output.attentions, "all"
                )
                source_sample_ids = torch.arange(
                    microbatch_sample_start,
                    microbatch_sample_start + int(batch.targets.shape[0]),
                    dtype=torch.long,
                    device=device,
                )
                completed = teacher_accumulator.add(
                    attention,
                    orders,
                    source_sample_ids=source_sample_ids,
                )
                pending_teacher_windows.extend(completed)
                del attention
            detached_loss = float(loss.detach().float().item())
            scaler.scale(loss).backward()
            accumulated_loss += detached_loss
            del loss, output
            train_samples_seen += int(batch.targets.shape[0])
        if float(config["grad_clip"]) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()


        for window in pending_teacher_windows:
            if teacher_cache_frozen:
                raise RuntimeError("Teacher window completed after cache freeze")
            if direct_window_teacher_mode:
                teacher_record = update_teacher_from_attention_window(
                    config,
                    controller,
                    iteration,
                    window,
                    train_samples_seen_after_step=train_samples_seen,
                    update_callback=teacher_update_callback,
                )
            elif mlp_window_teacher_mode:
                if teacher_mlp_runtime is None:
                    raise RuntimeError("teacher MLP runtime is not initialized")
                teacher_record = update_teacher_mlp_from_attention_window(
                    config,
                    controller,
                    teacher_mlp_runtime,
                    device,
                    iteration,
                    window,
                    train_samples_seen_after_step=train_samples_seen,
                    update_callback=teacher_mlp_update_callback,
                )
            else:
                raise RuntimeError("unknown sample-window Teacher policy")
            persist_teacher_record(teacher_record)

        if mlp_window_teacher_mode:
            if teacher_mlp_runtime is None:
                raise RuntimeError("teacher MLP runtime is not initialized")
            mlp_distill_stop = int(
                config["_distill_end_samples"]
            )
            if (
                not teacher_mlp_runtime.freeze_transition_done
                and train_samples_seen >= mlp_distill_stop
            ):
                if train_samples_seen != mlp_distill_stop:
                    raise RuntimeError(
                        "teacher MLP freeze transition overshot its sample boundary"
                    )
                if teacher_mlp_runtime.freeze_parameters():
                    mlp_freeze_payload = {
                        "iteration": int(iteration),
                        "train_samples_seen": int(train_samples_seen),
                        "parameter_updates": int(
                            teacher_mlp_runtime.parameter_updates
                        ),
                        "order_refreshes": int(
                            teacher_mlp_runtime.order_refreshes
                        ),
                        "parameters_frozen": True,
                        "forbidden_priors_used": False,
                    }
                    _json_dump(
                        out_dir / "teacher_mlp_freeze_transition.json",
                        mlp_freeze_payload,
                    )
                    torch.save(
                        _checkpoint_payload(
                            model,
                            optimizer,
                            config,
                            controller,
                            scaler,
                            np_rng,
                            iteration,
                            best_val,
                            teacher_np_rng,
                            teacher_order_generator,
                            current_teacher_runtime_state(),
                            current_teacher_mlp_state(),
                        ),
                        out_dir / "teacher_mlp_frozen.pt",
                    )
                    if run is not None:
                        run.log(
                            {
                                "iter": int(iteration),
                                "teacher_mlp/parameters_frozen": 1.0,
                                "teacher_mlp/parameter_updates": int(
                                    teacher_mlp_runtime.parameter_updates
                                ),
                                "teacher_mlp/order_refreshes": int(
                                    teacher_mlp_runtime.order_refreshes
                                ),
                            }
                        )

        if (
            sample_teacher_mode
            and not teacher_freeze_transition_done
            and train_samples_seen
            >= int(config["_anneal_end_samples"])
        ):
            if teacher_accumulator is None:
                raise RuntimeError("sample Teacher accumulator is not initialized")
            discarded_partial = teacher_accumulator.discard_partial()
            if discarded_partial != 0:
                raise RuntimeError(
                    "exact ordinary-training sample windows left a partial tail "
                    "at the freeze boundary"
                )
            if controller.cached_order is None:
                raise RuntimeError(
                    "Teacher reached freeze without a valid complete cached order"
                )
            teacher_cache_frozen = True
            teacher_freeze_transition_done = True
            freeze_payload = {
                "iteration": int(iteration),
                "train_samples_seen": int(train_samples_seen),
                "partial_samples_discarded": int(discarded_partial),
                "completed_windows": int(teacher_accumulator.completed_windows),
                "cached_order": (
                    None
                    if controller.cached_order is None
                    else list(controller.cached_order)
                ),
                "forbidden_priors_used": False,
            }
            _json_dump(out_dir / "teacher_freeze_transition.json", freeze_payload)
            _json_dump(out_dir / "teacher_history.json", controller.teacher_history)
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    controller,
                    scaler,
                    np_rng,
                    iteration,
                    best_val,
                    teacher_np_rng,
                    teacher_order_generator,
                    current_teacher_runtime_state(),
                    current_teacher_mlp_state(),
                ),
                out_dir / "teacher_frozen.pt",
            )



        if iteration % int(config["log_interval"]) == 0:
            elapsed = max(time.time() - started, 1e-9)
            payload = {
                "iter": iteration,
                "train/loss": accumulated_loss,
                "lr": lr,
                "teacher/probability": controller.teacher_probability(iteration),
                "teacher/scheduled_probability": controller.teacher_scheduled_probability(iteration),
                "teacher/cache_frozen": float(controller.teacher_cache_frozen(iteration)),
                "teacher/phase": controller.teacher_phase(iteration),
                "samples_per_second": ((iteration - start_iter + 1) * int(config["batch_size"]) * int(config["gradient_accumulation_steps"])) / elapsed,
            }
            if sample_teacher_mode:
                if teacher_accumulator is None or controller.sample_schedule is None:
                    raise RuntimeError("sample Teacher runtime is not initialized")
                after_step = controller.sample_schedule.snapshot(train_samples_seen)
                payload.update(
                    {
                        "teacher/probability": (
                            after_step.learned_order_probability
                            if controller.cached_order is not None
                            else 0.0
                        ),
                        "teacher/scheduled_probability": (
                            after_step.learned_order_probability
                        ),
                        "teacher/cache_frozen": float(teacher_cache_frozen),
                        "teacher/phase": after_step.phase,
                        "teacher/window_samples": int(
                            teacher_accumulator.sample_count
                        ),
                        "teacher/window_target_samples": int(
                            teacher_accumulator.target_samples
                        ),
                        "teacher/completed_windows": int(
                            teacher_accumulator.completed_windows
                        ),
                        "teacher/eligible_samples_seen": int(
                            teacher_accumulator.eligible_samples_seen
                        ),
                        "train/samples_seen": int(train_samples_seen),
                    }
                )
            if mlp_window_teacher_mode:
                if teacher_mlp_runtime is None:
                    raise RuntimeError("teacher MLP runtime is not initialized")
                payload.update(
                    {
                        "teacher_mlp/parameter_updates": int(
                            teacher_mlp_runtime.parameter_updates
                        ),
                        "teacher_mlp/order_refreshes": int(
                            teacher_mlp_runtime.order_refreshes
                        ),
                        "teacher_mlp/parameters_frozen": float(
                            teacher_mlp_runtime.parameters_frozen
                        ),
                    }
                )
            print(json.dumps(payload, sort_keys=True), flush=True)
            if run is not None:
                run.log(payload)

        should_eval = (iteration + 1) % int(config["eval_interval"]) == 0 or iteration + 1 == int(config["max_iters"])
        if should_eval:
            metrics = evaluate_task(model, dataset, "val", config, controller, device, autocast_dtype, iteration, int(config["eval_batches"]))
            metric_payload = {f"val/{key}": value for key, value in metrics.items()}
            metric_payload["iter"] = iteration
            teacher_fixed = (
                config["method"] in {"random", "canonical", "fixed"}
                or (controller.cached_order is not None and iteration + 1 >= controller.teacher_fixed_start_iteration())
            )
            selection_eligible = teacher_fixed and iteration + 1 >= int(config["best_selection_start_iter"])
            metric_payload["selection/eligible_fixed_protocol"] = float(
                selection_eligible
            )
            print(json.dumps(metric_payload, sort_keys=True), flush=True)
            if run is not None:
                run.log(metric_payload)
            improved = (
                selection_eligible
                and metrics["teacher_forced_nll_per_variable"] < best_val
            )
            if improved:
                best_val = metrics["teacher_forced_nll_per_variable"]
                torch.save(
                    _checkpoint_payload(
                        model, optimizer, config, controller, scaler, np_rng,
                        iteration, best_val,
                        teacher_np_rng, teacher_order_generator,
                        current_teacher_runtime_state(),
                        current_teacher_mlp_state(),
                    ),
                    out_dir / "best.pt",
                )
                _json_dump(out_dir / "best_val_metrics.json", metrics)
            if improved or bool(config["always_save_checkpoint"]) or iteration + 1 == int(config["max_iters"]):
                torch.save(
                    _checkpoint_payload(
                        model, optimizer, config, controller, scaler, np_rng,
                        iteration, best_val,
                        teacher_np_rng, teacher_order_generator,
                        current_teacher_runtime_state(),
                        current_teacher_mlp_state(),
                    ),
                    out_dir / "last.pt",
                )

    if int(config["test_eval_batches"]) > 0 and "test" in dataset.split_names:
        test_metrics = evaluate_task(model, dataset, "test", config, controller, device, autocast_dtype, int(config["max_iters"]), int(config["test_eval_batches"]))
        _json_dump(out_dir / "test_metrics_monitor_only.json", test_metrics)
    if sample_teacher_mode:
        _json_dump(out_dir / "teacher_history.json", controller.teacher_history)
    if run is not None:
        run.finish()


__all__ = [
    "OrderController",
    "OrientedFiedlerSupervision",
    "TrainingTaskAdapter",
    "build_teacher_mlp_runtime",
    "compute_oriented_fiedler_supervision",
    "evaluate",
    "load_config",
    "main",
    "random_orders",
    "update_teacher_from_attention_window",
    "update_teacher_mlp_from_attention_window",
    "validate_config",
]
