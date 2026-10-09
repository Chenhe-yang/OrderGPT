from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aligned_continuous.order_teacher import FiedlerOrderResult
from ordergpt.order_mlp import OrderMLP, priority_to_order


Order = Tuple[int, ...]


def _as_order(values: Sequence[int], num_variables: int) -> Order:
    order = tuple(int(value) for value in values)
    if len(order) != int(num_variables) or sorted(order) != list(
        range(int(num_variables))
    ):
        raise ValueError("order must be a complete current-variable permutation")
    return order


def minmax_fiedler_priority(
    fiedler_vector: torch.Tensor,
    *,
    selected_reverse: bool,
    eps: float = 1e-12,
) -> torch.Tensor:


    if not torch.is_tensor(fiedler_vector):
        fiedler_vector = torch.as_tensor(fiedler_vector)
    if fiedler_vector.ndim != 1 or int(fiedler_vector.numel()) < 2:
        raise ValueError("fiedler_vector must be one-dimensional with D >= 2")
    values = fiedler_vector.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("fiedler_vector contains non-finite values")
    span = values.max() - values.min()
    if float(span.item()) <= float(eps):
        raise ValueError("fiedler_vector is degenerate and cannot define priority")
    normalized = (values - values.min()) / span
    priority = normalized if bool(selected_reverse) else 1.0 - normalized
    return priority.clamp(0.0, 1.0)


@dataclass(frozen=True)
class FiedlerPriorityTarget:


    priority: torch.Tensor
    order: Order
    selected_reverse: bool
    diagnostics: Dict[str, Any]


def build_fiedler_priority_target(
    result: FiedlerOrderResult,
    selected_order: Sequence[int],
) -> FiedlerPriorityTarget:


    if not isinstance(result, FiedlerOrderResult):
        raise TypeError("result must be a FiedlerOrderResult")
    num_variables = int(result.fiedler_vector.numel())
    selected = _as_order(selected_order, num_variables)
    raw = _as_order(result.raw_order, num_variables)
    reverse = _as_order(result.reverse_order, num_variables)
    if selected == raw:
        selected_reverse = False
    elif selected == reverse:
        selected_reverse = True
    else:
        raise ValueError("selected_order must equal the raw or reverse Fiedler order")
    if bool(result.diagnostics.get("degenerate", False)):
        raise ValueError("degenerate Fiedler result cannot supervise the order MLP")


    priority = minmax_fiedler_priority(
        result.fiedler_vector,
        selected_reverse=selected_reverse,
    )
    diagnostics: Dict[str, Any] = {
        "target_source": "oriented_minmax_fiedler_vector_priority",
        "target_uses_fiedler_eigenvalue": False,
        "selected_reverse": bool(selected_reverse),
        "priority_min": float(priority.min().item()),
        "priority_max": float(priority.max().item()),
        "priority_std": float(priority.std(unbiased=False).item()),
        "fiedler_eigenvalue_diagnostic_only": result.diagnostics.get(
            "fiedler_eigenvalue"
        ),
        "fiedler_eigengap_diagnostic_only": result.diagnostics.get(
            "fiedler_eigengap"
        ),
        "forbidden_priors_used": False,
    }
    return FiedlerPriorityTarget(
        priority=priority,
        order=selected,
        selected_reverse=selected_reverse,
        diagnostics=diagnostics,
    )


def _priority_pair_tau(prediction: torch.Tensor, target: torch.Tensor) -> float:


    pred = prediction.detach().float().reshape(-1)
    truth = target.detach().float().reshape(-1)
    rows, cols = torch.triu_indices(int(pred.numel()), int(pred.numel()), offset=1)
    truth_sign = torch.sign(truth[rows] - truth[cols])
    keep = truth_sign != 0
    if not bool(keep.any()):
        return float("nan")
    pred_sign = torch.sign(pred[rows] - pred[cols])
    return float((pred_sign[keep] * truth_sign[keep]).mean().item())


def priority_mse_diagnostics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:


    pred = prediction.detach().float().reshape(-1)
    truth = target.detach().float().reshape(-1)
    if pred.shape != truth.shape or int(pred.numel()) < 2:
        raise ValueError("prediction and target must have the same non-trivial shape")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(truth).all()):
        raise ValueError("prediction/target contains non-finite values")
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * truth_centered.square().sum()
    )
    pearson = (
        float((pred_centered * truth_centered).sum().div(denominator).item())
        if float(denominator.item()) > 1e-12
        else float("nan")
    )
    pred_order = priority_to_order(pred)
    target_order = priority_to_order(truth)
    order_match = float((pred_order == target_order).float().mean().item())
    return {
        "mse": float(F.mse_loss(pred, truth).item()),
        "mae": float(F.l1_loss(pred, truth).item()),
        "pearson": pearson,
        "priority_pair_tau": _priority_pair_tau(pred, truth),
        "exact_position_fraction": order_match,
        "prediction_mean": float(pred.mean().item()),
        "prediction_std": float(pred.std(unbiased=False).item()),
        "target_mean": float(truth.mean().item()),
        "target_std": float(truth.std(unbiased=False).item()),
    }


class AlignedContinuousOrderMLP(nn.Module):


    def __init__(
        self,
        num_variables: int,
        hidden_dims: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        self.num_variables = int(num_variables)
        if self.num_variables < 2:
            raise ValueError("num_variables must be at least two")
        self.policy = OrderMLP(
            num_variables=self.num_variables,
            hidden_dims=tuple(int(value) for value in hidden_dims),
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    @property
    def parameters_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.parameters())

    def forward_logits(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        return self.policy.forward_logits(attention_matrix)

    def forward(self, attention_matrix: torch.Tensor) -> torch.Tensor:


        return self.policy(attention_matrix)

    @torch.no_grad()
    def order_from_attention(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        return self.policy.order_from_attention(attention_matrix)

    def mse_loss(
        self,
        attention_matrix: torch.Tensor,
        target_priority: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy.mse_loss(attention_matrix, target_priority)

    @torch.no_grad()
    def diagnostics(
        self,
        attention_matrix: torch.Tensor,
        target_priority: torch.Tensor,
    ) -> Dict[str, float]:
        prediction = self(attention_matrix)
        target = target_priority.detach().to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return priority_mse_diagnostics(prediction, target)

    def set_parameters_frozen(self, frozen: bool = True) -> "AlignedContinuousOrderMLP":
        for parameter in self.parameters():
            parameter.requires_grad_(not bool(frozen))
        if frozen:
            self.eval()
        else:
            self.train()
        return self


class FiedlerMLPRuntime:


    SCHEMA_VERSION = 1

    def __init__(
        self,
        model: AlignedContinuousOrderMLP,
        optimizer: torch.optim.Optimizer,
        *,
        model_config: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.model_config = dict(model_config)
        self.parameter_updates = 0
        self.order_refreshes = 0
        self.freeze_transition_done = False
        self.last_window_index: Optional[int] = None
        self.last_parameter_update_train_samples: Optional[int] = None
        self.last_order_refresh_train_samples: Optional[int] = None
        self.last_logits: Optional[torch.Tensor] = None
        self.last_priority: Optional[torch.Tensor] = None
        self.last_target_priority: Optional[torch.Tensor] = None
        self.last_order: Optional[Order] = None
        self.last_metrics: Dict[str, float] = {}

    @property
    def parameters_frozen(self) -> bool:
        return self.model.parameters_frozen

    def freeze_parameters(self) -> bool:


        if self.freeze_transition_done:
            if not self.parameters_frozen:
                raise RuntimeError(
                    "MLP freeze transition is recorded but parameters are trainable"
                )
            return False
        self.model.set_parameters_frozen(True)
        self.freeze_transition_done = True
        return True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "model_config": dict(self.model_config),
            "parameter_count": int(self.model.parameter_count),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "parameter_updates": int(self.parameter_updates),
            "order_refreshes": int(self.order_refreshes),
            "parameters_frozen": bool(self.parameters_frozen),
            "freeze_transition_done": bool(self.freeze_transition_done),
            "last_window_index": self.last_window_index,
            "last_parameter_update_train_samples": (
                self.last_parameter_update_train_samples
            ),
            "last_order_refresh_train_samples": (
                self.last_order_refresh_train_samples
            ),
            "last_logits": (
                None if self.last_logits is None else self.last_logits.detach().cpu()
            ),
            "last_priority": (
                None
                if self.last_priority is None
                else self.last_priority.detach().cpu()
            ),
            "last_target_priority": (
                None
                if self.last_target_priority is None
                else self.last_target_priority.detach().cpu()
            ),
            "last_order": None if self.last_order is None else list(self.last_order),
            "last_metrics": dict(self.last_metrics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported teacher MLP checkpoint schema")
        if dict(state.get("model_config", {})) != self.model_config:
            raise ValueError("resume checkpoint teacher MLP model config does not match")
        expected_count = int(self.model.parameter_count)
        if int(state.get("parameter_count", -1)) != expected_count:
            raise ValueError("resume checkpoint teacher MLP parameter count does not match")
        model_state = state.get("model_state_dict")
        optimizer_state = state.get("optimizer_state_dict")
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise ValueError("teacher MLP checkpoint lacks model/optimizer state")
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        self.parameter_updates = int(state.get("parameter_updates", 0))
        self.order_refreshes = int(state.get("order_refreshes", 0))
        if min(self.parameter_updates, self.order_refreshes) < 0:
            raise ValueError("teacher MLP counters must be non-negative")
        self.freeze_transition_done = bool(
            state.get("freeze_transition_done", False)
        )
        parameters_frozen = bool(state.get("parameters_frozen", False))
        if self.freeze_transition_done != parameters_frozen:
            raise ValueError(
                "teacher MLP frozen flag and freeze transition disagree"
            )
        self.model.set_parameters_frozen(parameters_frozen)
        self.last_window_index = _optional_int(state.get("last_window_index"))
        self.last_parameter_update_train_samples = _optional_int(
            state.get("last_parameter_update_train_samples")
        )
        self.last_order_refresh_train_samples = _optional_int(
            state.get("last_order_refresh_train_samples")
        )
        self.last_logits = _optional_tensor(state.get("last_logits"))
        self.last_priority = _optional_tensor(state.get("last_priority"))
        self.last_target_priority = _optional_tensor(
            state.get("last_target_priority")
        )
        raw_order = state.get("last_order")
        self.last_order = (
            None
            if raw_order is None
            else _as_order(raw_order, self.model.num_variables)
        )
        raw_metrics = state.get("last_metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("teacher MLP last_metrics must be a mapping")
        self.last_metrics = {
            str(key): float(value) for key, value in raw_metrics.items()
        }


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise ValueError("serialized teacher MLP artifact must be a tensor")
    return value.detach().cpu().clone()


__all__ = [
    "AlignedContinuousOrderMLP",
    "FiedlerMLPRuntime",
    "FiedlerPriorityTarget",
    "build_fiedler_priority_target",
    "minmax_fiedler_priority",
    "priority_mse_diagnostics",
    "priority_to_order",
]
