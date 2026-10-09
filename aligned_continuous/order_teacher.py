from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


OrderLike = Union[torch.Tensor, Sequence[int]]


def _as_order_tuple(order: OrderLike, num_variables: Optional[int] = None) -> Tuple[int, ...]:
    if torch.is_tensor(order):
        if order.ndim != 1:
            raise ValueError("order must be one-dimensional")
        values = [int(value) for value in order.detach().cpu().tolist()]
    else:
        values = [int(value) for value in order]
    if num_variables is None:
        num_variables = len(values)
    num_variables = int(num_variables)
    if len(values) != num_variables or sorted(values) != list(range(num_variables)):
        raise ValueError(
            "order must be a complete permutation of current variable IDs "
            "0..{}".format(max(0, num_variables - 1))
        )
    return tuple(values)


def _validate_reveal_orders(reveal_orders: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not torch.is_tensor(reveal_orders):
        reveal_orders = torch.as_tensor(reveal_orders, dtype=torch.long)
    if reveal_orders.ndim != 2:
        raise ValueError(
            "reveal_orders must have shape [batch, num_variables], got {}".format(
                tuple(reveal_orders.shape)
            )
        )
    if int(reveal_orders.shape[0]) != int(batch_size):
        raise ValueError(
            "attention batch size {} does not match reveal_orders batch size {}".format(
                int(batch_size), int(reveal_orders.shape[0])
            )
        )
    num_variables = int(reveal_orders.shape[1])
    expected = torch.arange(num_variables, dtype=torch.long, device=reveal_orders.device)
    sorted_orders = torch.sort(reveal_orders.to(dtype=torch.long), dim=1).values
    if not torch.equal(sorted_orders, expected.view(1, -1).expand_as(sorted_orders)):
        raise ValueError("every reveal order must be a complete current-variable permutation")
    return reveal_orders.to(dtype=torch.long)


def invert_permutation(order: OrderLike) -> torch.Tensor:


    order_tuple = _as_order_tuple(order)
    order_tensor = torch.tensor(order_tuple, dtype=torch.long)
    inverse = torch.empty_like(order_tensor)
    inverse[order_tensor] = torch.arange(len(order_tuple), dtype=torch.long)
    return inverse


@dataclass(frozen=True)
class AttentionAggregation:


    attention: torch.Tensor
    attention_sum: torch.Tensor
    exposure_count: torch.Tensor
    num_samples: int
    num_heads: int

    @property
    def matrix(self) -> torch.Tensor:
        return self.attention


def _extract_all_layer_attention(attention: Any) -> torch.Tensor:
    if isinstance(attention, (list, tuple)):
        if not attention:
            raise ValueError("attention collection is empty")
        if not all(torch.is_tensor(layer) for layer in attention):
            raise TypeError("every attention layer must be a tensor")
        attention = torch.cat(tuple(attention), dim=1)
    if not torch.is_tensor(attention):
        raise TypeError(
            "attention must be an all-layer tensor or a non-empty layer sequence"
        )
    if attention.ndim != 4:
        raise ValueError(
            "attention must have shape [batch, layer_heads, D+1, D+1], got {}".format(
                tuple(attention.shape)
            )
        )
    if int(attention.shape[-1]) != int(attention.shape[-2]):
        raise ValueError("attention must be square in its last two dimensions")
    if int(attention.shape[1]) <= 0:
        raise ValueError("attention must contain at least one layer/head matrix")
    if not torch.isfinite(attention).all():
        raise ValueError("attention contains non-finite values")
    return attention


@dataclass(frozen=True)
class CompletedAttentionWindow:


    matrix: torch.Tensor
    attention_sum: torch.Tensor
    exposure_count: torch.Tensor
    num_samples: int
    window_index: int
    first_source_sample: int
    last_source_sample: int
    eligible_samples_seen_at_completion: int
    num_heads: int


@torch.no_grad()
def aggregate_attention_to_current(
    attention: Any,
    orders: torch.Tensor,
) -> AttentionAggregation:


    stacked = _extract_all_layer_attention(attention)
    batch_size, num_heads, full_rows, full_cols = [
        int(value) for value in stacked.shape
    ]
    validated_orders = _validate_reveal_orders(orders, batch_size=batch_size).to(
        device=stacked.device, dtype=torch.long
    )
    num_variables = int(validated_orders.shape[1])
    if full_rows != num_variables + 1 or full_cols != num_variables + 1:
        raise ValueError(
            "attention has sequence length {}, but reveal_orders imply D+1={}".format(
                full_rows, num_variables + 1
            )
        )

    reveal_view = stacked[:, :, :-1, 1:]


    per_sample = reveal_view.mean(dim=1, dtype=torch.float64)


    inverse = torch.argsort(validated_orders, dim=1)
    row_index = inverse.unsqueeze(-1).expand(-1, -1, num_variables)
    current = torch.gather(per_sample, 1, row_index)
    column_index = inverse.unsqueeze(1).expand(-1, num_variables, -1)
    current = torch.gather(current, 2, column_index)


    legal = inverse.unsqueeze(2) > inverse.unsqueeze(1)
    attention_sum = (current * legal.to(dtype=torch.float64)).sum(dim=0)
    exposure_count = legal.to(dtype=torch.float64).sum(dim=0)
    aggregate = torch.zeros_like(attention_sum)
    observed = exposure_count > 0
    aggregate[observed] = attention_sum[observed] / exposure_count[observed]
    return AttentionAggregation(
        attention=aggregate,
        attention_sum=attention_sum,
        exposure_count=exposure_count,
        num_samples=batch_size,
        num_heads=num_heads,
    )


class AttentionSampleAccumulator:


    def __init__(
        self,
        num_variables: int,
        target_samples: int = 2048,
    ) -> None:
        self.num_variables = int(num_variables)
        self.target_samples = int(target_samples)
        if self.num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if self.target_samples <= 0:
            raise ValueError("target_samples must be positive")
        self.completed_windows = 0
        self.eligible_samples_seen = 0
        self.discarded_partial_samples = 0
        self._reset_partial()

    def _reset_partial(self) -> None:
        shape = (self.num_variables, self.num_variables)
        self.attention_sum = torch.zeros(shape, dtype=torch.float64, device="cpu")
        self.exposure_count = torch.zeros_like(self.attention_sum)
        self.sample_count = 0
        self.first_source_sample: Optional[int] = None
        self.last_source_sample: Optional[int] = None
        self.num_heads: Optional[int] = None

    @property
    def remaining_samples(self) -> int:
        return self.target_samples - self.sample_count

    @torch.no_grad()
    def add(
        self,
        attention: torch.Tensor,
        orders: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
        source_sample_ids: Optional[torch.Tensor] = None,
    ) -> List[CompletedAttentionWindow]:


        stacked = _extract_all_layer_attention(attention)
        batch_size = int(stacked.shape[0])
        validated_orders = _validate_reveal_orders(orders, batch_size=batch_size)
        if eligible_mask is None:
            mask = torch.ones(batch_size, dtype=torch.bool, device=orders.device)
        else:
            if not torch.is_tensor(eligible_mask) or eligible_mask.ndim != 1:
                raise ValueError("eligible_mask must be a one-dimensional tensor")
            if int(eligible_mask.numel()) != batch_size:
                raise ValueError("eligible_mask length must match attention batch size")
            mask = eligible_mask.to(device=orders.device, dtype=torch.bool)
        if source_sample_ids is None:
            source_ids = torch.arange(batch_size, dtype=torch.long, device=orders.device)
        else:
            if not torch.is_tensor(source_sample_ids) or source_sample_ids.ndim != 1:
                raise ValueError("source_sample_ids must be a one-dimensional tensor")
            if int(source_sample_ids.numel()) != batch_size:
                raise ValueError("source_sample_ids length must match attention batch size")
            source_ids = source_sample_ids.to(device=orders.device, dtype=torch.long)
            if bool((source_ids < 0).any()):
                raise ValueError("source_sample_ids must be non-negative")

        eligible_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if int(eligible_indices.numel()) == 0:
            return []
        if int(eligible_indices.numel()) == batch_size:


            selected_attention = stacked.detach()
            selected_orders = validated_orders
            selected_source_ids = source_ids.detach().cpu()
        else:
            selected_attention = stacked.index_select(
                0, eligible_indices.to(device=stacked.device)
            ).detach()
            selected_orders = validated_orders.index_select(0, eligible_indices)
            selected_source_ids = (
                source_ids.index_select(0, eligible_indices).detach().cpu()
            )

        completed: List[CompletedAttentionWindow] = []
        offset = 0
        selected_count = int(eligible_indices.numel())
        while offset < selected_count:
            take = min(self.remaining_samples, selected_count - offset)
            stop = offset + take
            chunk_attention = selected_attention[offset:stop]
            chunk_orders = selected_orders[offset:stop]
            chunk_source_ids = selected_source_ids[offset:stop]
            aggregate = aggregate_attention_to_current(
                chunk_attention,
                chunk_orders,
            )
            chunk_heads = int(aggregate.num_heads)
            if self.num_heads is None:
                self.num_heads = chunk_heads
            elif self.num_heads != chunk_heads:
                raise ValueError("attention head count changed within a sample window")
            self.attention_sum += aggregate.attention_sum.detach().cpu()
            self.exposure_count += aggregate.exposure_count.detach().cpu()
            if self.first_source_sample is None:
                self.first_source_sample = int(chunk_source_ids[0].item())
            self.last_source_sample = int(chunk_source_ids[-1].item())
            self.sample_count += take
            self.eligible_samples_seen += take
            offset = stop

            if self.sample_count == self.target_samples:
                matrix = torch.zeros_like(self.attention_sum)
                observed = self.exposure_count > 0
                matrix[observed] = (
                    self.attention_sum[observed] / self.exposure_count[observed]
                )
                completed.append(
                    CompletedAttentionWindow(
                        matrix=matrix.clone(),
                        attention_sum=self.attention_sum.clone(),
                        exposure_count=self.exposure_count.clone(),
                        num_samples=self.sample_count,
                        window_index=self.completed_windows,
                        first_source_sample=int(self.first_source_sample),
                        last_source_sample=int(self.last_source_sample),
                        eligible_samples_seen_at_completion=int(
                            self.eligible_samples_seen
                        ),
                        num_heads=int(self.num_heads or 0),
                    )
                )
                self.completed_windows += 1
                self._reset_partial()

        return completed

    def discard_partial(self) -> int:


        discarded = int(self.sample_count)
        self.discarded_partial_samples += discarded
        self._reset_partial()
        return discarded

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "num_variables": self.num_variables,
            "target_samples": self.target_samples,
            "view": "target_to_observed",
            "backend": "vectorized",
            "completed_windows": self.completed_windows,
            "eligible_samples_seen": self.eligible_samples_seen,
            "discarded_partial_samples": self.discarded_partial_samples,
            "sample_count": self.sample_count,
            "first_source_sample": self.first_source_sample,
            "last_source_sample": self.last_source_sample,
            "num_heads": self.num_heads,
            "attention_sum": self.attention_sum.clone(),
            "exposure_count": self.exposure_count.clone(),
            "forbidden_priors_used": False,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != 1:
            raise ValueError("unsupported AttentionSampleAccumulator schema_version")
        if int(state["num_variables"]) != self.num_variables:
            raise ValueError("accumulator num_variables does not match")
        if int(state["target_samples"]) != self.target_samples:
            raise ValueError("accumulator target_samples does not match")
        if str(state.get("view", "target_to_observed")) != "target_to_observed":
            raise ValueError("accumulator attention view does not match")
        if str(state.get("backend", "vectorized")) != "vectorized":
            raise ValueError("accumulator aggregation backend does not match")
        shape = (self.num_variables, self.num_variables)
        attention_sum = torch.as_tensor(
            state["attention_sum"], dtype=torch.float64, device="cpu"
        )
        exposure_count = torch.as_tensor(
            state["exposure_count"], dtype=torch.float64, device="cpu"
        )
        if tuple(attention_sum.shape) != shape or tuple(exposure_count.shape) != shape:
            raise ValueError("serialized accumulator matrices have the wrong shape")
        if not torch.isfinite(attention_sum).all() or not torch.isfinite(exposure_count).all():
            raise ValueError("serialized accumulator contains non-finite values")
        sample_count = int(state["sample_count"])
        if not 0 <= sample_count < self.target_samples:
            raise ValueError("serialized partial sample_count is invalid")
        first = state.get("first_source_sample")
        last = state.get("last_source_sample")
        if sample_count == 0 and (first is not None or last is not None):
            raise ValueError("empty serialized accumulator has source sample bounds")
        if sample_count > 0 and (first is None or last is None):
            raise ValueError("non-empty serialized accumulator lacks source sample bounds")
        self.completed_windows = int(state.get("completed_windows", 0))
        self.eligible_samples_seen = int(state.get("eligible_samples_seen", 0))
        self.discarded_partial_samples = int(
            state.get("discarded_partial_samples", 0)
        )
        if min(
            self.completed_windows,
            self.eligible_samples_seen,
            self.discarded_partial_samples,
        ) < 0:
            raise ValueError("serialized accumulator counters must be non-negative")
        self.sample_count = sample_count
        self.first_source_sample = None if first is None else int(first)
        self.last_source_sample = None if last is None else int(last)
        heads = state.get("num_heads")
        self.num_heads = None if heads is None else int(heads)
        if self.num_heads is not None and self.num_heads <= 0:
            raise ValueError("serialized accumulator num_heads must be positive")
        self.attention_sum = attention_sum.clone()
        self.exposure_count = exposure_count.clone()


@dataclass(frozen=True)
class FiedlerOrderResult:
    affinity: torch.Tensor
    laplacian: torch.Tensor
    eigenvalues: torch.Tensor
    fiedler_vector: torch.Tensor
    raw_order: Tuple[int, ...]
    reverse_order: Tuple[int, ...]
    diagnostics: Dict[str, Any]

    @property
    def degenerate(self) -> bool:
        return bool(self.diagnostics.get("degenerate", False))

    @property
    def eigengap(self) -> Optional[float]:
        value = self.diagnostics.get("fiedler_eigengap")
        return None if value is None else float(value)

    def oriented_vector(self, selected_order: OrderLike) -> torch.Tensor:
        selected = _as_order_tuple(selected_order, len(self.raw_order))
        if selected == self.raw_order:
            return self.fiedler_vector.clone()
        if selected == self.reverse_order:
            return -self.fiedler_vector.clone()
        raise ValueError("selected_order must be the raw Fiedler order or its reverse")


@torch.no_grad()
def recover_fiedler_order(
    attention_matrix: torch.Tensor,
    degeneracy_tolerance: float = 1e-10,
) -> FiedlerOrderResult:


    if not torch.is_tensor(attention_matrix):
        attention_matrix = torch.as_tensor(attention_matrix)
    if attention_matrix.ndim != 2 or int(attention_matrix.shape[0]) != int(attention_matrix.shape[1]):
        raise ValueError(
            "attention_matrix must be square [D, D], got {}".format(tuple(attention_matrix.shape))
        )
    num_variables = int(attention_matrix.shape[0])
    if num_variables < 2:
        raise ValueError("Fiedler ordering requires at least two variables")
    values = attention_matrix.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(values).all():
        raise ValueError("attention_matrix contains non-finite values")
    tolerance = max(float(degeneracy_tolerance), 1e-15)
    minimum = float(values.min().item())
    if minimum < -tolerance:
        raise ValueError("attention_matrix contains negative affinity values")

    directed_affinity = values.clamp_min(0.0)
    directed_affinity.fill_diagonal_(0.0)
    directed_sum = float(directed_affinity.sum().item())
    directed_density = float(
        (directed_affinity > 0).to(dtype=torch.float64).mean().item()
    )

    transpose = directed_affinity.transpose(0, 1)
    affinity = torch.sqrt(directed_affinity * transpose)
    affinity_rule = "sqrt(A*A.T)"
    affinity = affinity.clamp_min(0.0)
    affinity.fill_diagonal_(0.0)
    degree = affinity.sum(dim=1)
    laplacian = torch.diag(degree) - affinity
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    except RuntimeError as exc:
        raise ValueError("Laplacian eigendecomposition failed: {}".format(exc)) from exc
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(eigenvectors).all():
        raise ValueError("Laplacian eigendecomposition produced non-finite values")

    vector = eigenvectors[:, 1].clone()
    vector_values = [float(value) for value in vector.tolist()]
    raw_order = tuple(
        sorted(range(num_variables), key=lambda index: (vector_values[index], int(index)))
    )
    reverse_order = tuple(reversed(raw_order))

    spectral_scale = max(1.0, float(eigenvalues.abs().max().item()))
    eig_tolerance = tolerance * spectral_scale
    zero_multiplicity = int((eigenvalues.abs() <= eig_tolerance).sum().item())
    fiedler_value = float(eigenvalues[1].item())
    next_value = float(eigenvalues[2].item()) if num_variables > 2 else None
    eigengap = (next_value - fiedler_value) if next_value is not None else None
    vector_std = float(vector.std(unbiased=False).item())
    vector_range = float((vector.max() - vector.min()).item())
    squared_mass = vector.square()
    inverse_participation_ratio = float(squared_mass.square().sum().item())
    effective_support = float(
        1.0 / max(inverse_participation_ratio, torch.finfo(torch.float64).eps)
    )
    sorted_squared_mass = torch.sort(squared_mass, descending=True).values
    top1_squared_mass = float(sorted_squared_mass[0].item())
    top2_squared_mass = (
        float(sorted_squared_mass[1].item()) if num_variables > 1 else 0.0
    )
    affinity_sum = float(affinity.sum().item())

    reasons: List[str] = []
    if affinity_sum <= tolerance:
        reasons.append("zero_affinity")
    if zero_multiplicity != 1:
        reasons.append("laplacian_zero_eigenvalue_multiplicity_{}".format(zero_multiplicity))
    if vector_std <= tolerance or vector_range <= tolerance:
        reasons.append("constant_fiedler_vector")
    if eigengap is not None and eigengap <= eig_tolerance:
        reasons.append("repeated_or_unseparated_fiedler_eigenspace")

    diagnostics: Dict[str, Any] = {
        "num_variables": num_variables,
        "affinity_rule": affinity_rule,
        "dense_affinity": True,
        "directed_density": directed_density,
        "directed_attention_sum": directed_sum,
        "laplacian_rule": "D-W_unnormalized",
        "affinity_sum": affinity_sum,
        "affinity_density": float((affinity > 0).to(dtype=torch.float64).mean().item()),
        "algebraic_connectivity": fiedler_value,
        "fiedler_eigenvalue": fiedler_value,
        "next_eigenvalue": next_value,
        "fiedler_eigengap": eigengap,
        "relative_fiedler_eigengap": (
            None if eigengap is None else float(eigengap / spectral_scale)
        ),
        "zero_eigenvalue_multiplicity": zero_multiplicity,
        "connected_by_spectrum": bool(zero_multiplicity == 1),
        "fiedler_vector_std": vector_std,
        "fiedler_vector_range": vector_range,
        "fiedler_inverse_participation_ratio": inverse_participation_ratio,
        "fiedler_effective_support": effective_support,
        "fiedler_top1_squared_mass": top1_squared_mass,
        "fiedler_top2_squared_mass": top2_squared_mass,
        "degeneracy_tolerance": tolerance,
        "degenerate": bool(reasons),
        "degeneracy_reasons": reasons,
        "forbidden_priors_used": False,
    }
    return FiedlerOrderResult(
        affinity=affinity,
        laplacian=laplacian,
        eigenvalues=eigenvalues,
        fiedler_vector=vector,
        raw_order=raw_order,
        reverse_order=reverse_order,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class OrientationResult:
    raw_order: Tuple[int, ...]
    reverse_order: Tuple[int, ...]
    selected_order: Tuple[int, ...]
    selected_orientation: str
    raw_attention_score: float
    reverse_attention_score: float
    tied: bool
    score_name: str = "directed_attention_strict_triangular_sum"
    forbidden_priors_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_order": list(self.raw_order),
            "reverse_order": list(self.reverse_order),
            "selected_order": list(self.selected_order),
            "selected_orientation": self.selected_orientation,
            "raw_attention_score": self.raw_attention_score,
            "reverse_attention_score": self.reverse_attention_score,
            "tied": self.tied,
            "score_name": self.score_name,
            "tie_break": "lexicographic_current_ids",
            "forbidden_priors_used": self.forbidden_priors_used,
        }


@torch.no_grad()
def attention_flow_score(attention_matrix: torch.Tensor, order: OrderLike) -> float:
    matrix = torch.as_tensor(attention_matrix).detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention_matrix must be square [D, D]")
    if not torch.isfinite(matrix).all() or bool((matrix < 0).any()):
        raise ValueError("attention_matrix must be finite and nonnegative")
    permutation = _as_order_tuple(order, int(matrix.shape[0]))
    indices = torch.tensor(permutation, dtype=torch.long)
    ordered = matrix[indices[:, None], indices[None, :]]
    lower = torch.tril_indices(len(permutation), len(permutation), offset=-1)
    return math.fsum(ordered[lower[0], lower[1]].tolist())


@torch.no_grad()
def orient_raw_reverse_from_attention(
    raw_order: OrderLike,
    attention_matrix: torch.Tensor,
) -> OrientationResult:
    raw = _as_order_tuple(raw_order)
    reverse = tuple(reversed(raw))
    raw_score = attention_flow_score(attention_matrix, raw)
    reverse_score = attention_flow_score(attention_matrix, reverse)
    tied = raw_score == reverse_score
    if tied:
        selected = min(raw, reverse)
    else:
        selected = raw if raw_score > reverse_score else reverse
    return OrientationResult(
        raw_order=raw,
        reverse_order=reverse,
        selected_order=selected,
        selected_orientation="raw" if selected == raw else "reverse",
        raw_attention_score=raw_score,
        reverse_attention_score=reverse_score,
        tied=tied,
    )


__all__ = [
    "AttentionAggregation",
    "AttentionSampleAccumulator",
    "CompletedAttentionWindow",
    "FiedlerOrderResult",
    "OrientationResult",
    "aggregate_attention_to_current",
    "attention_flow_score",
    "invert_permutation",
    "orient_raw_reverse_from_attention",
    "recover_fiedler_order",
]
