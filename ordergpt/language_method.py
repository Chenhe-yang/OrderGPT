from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def batched_inverse_permutations(permutations: torch.Tensor) -> torch.Tensor:


    if permutations.ndim != 2:
        raise ValueError(
            "permutations must have shape [batch, tokens], got "
            f"{tuple(permutations.shape)}"
        )
    batch, tokens = permutations.shape
    expected = torch.arange(
        tokens, dtype=permutations.dtype, device=permutations.device
    ).expand(batch, -1)
    if not torch.equal(torch.sort(permutations, dim=1).values, expected):
        raise ValueError("every row must be a permutation of current token IDs")
    inverse = torch.empty_like(permutations)
    reveal_positions = torch.arange(
        tokens, dtype=permutations.dtype, device=permutations.device
    ).expand(batch, -1)
    inverse.scatter_(1, permutations, reveal_positions)
    return inverse


def premean_all_layer_head_attention(
    attentions: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:


    if torch.is_tensor(attentions):
        if attentions.ndim != 4:
            raise ValueError(
                "attention summary must have shape [batch, heads, query, key]"
            )
        if attentions.shape[-1] != attentions.shape[-2]:
            raise ValueError(
                f"attention must be square, got {tuple(attentions.shape)}"
            )
        return attentions.detach().float().mean(dim=1, keepdim=True)
    if not attentions:
        raise ValueError("the ordinary training forward returned no attention")
    running_sum = None
    reference = None
    layer_count = 0
    head_count = 0
    for attention in attentions:
        if not torch.is_tensor(attention) or attention.ndim != 4:
            raise ValueError(
                "attention tensors must have shape [batch, heads, query, key]"
            )
        shape = tuple(attention.shape)
        if shape[-1] != shape[-2]:
            raise ValueError(f"attention must be square, got {shape}")
        signature = (shape[0], shape[1], shape[2], shape[3])
        if reference is None:
            reference = signature
            head_count = int(shape[1])
        elif signature != reference:
            raise ValueError(
                f"all attention layers must share one shape: {reference} != {signature}"
            )
        layer_sum = attention.detach().float().sum(dim=1)
        running_sum = layer_sum if running_sum is None else running_sum.add(layer_sum)
        layer_count += 1
    denominator = float(layer_count * head_count)
    return running_sum.div(denominator).unsqueeze(1)


def target_to_observed_contributions(
    attentions: torch.Tensor | Sequence[torch.Tensor],
    orders: torch.Tensor,
    *,
    accumulator_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:


    if accumulator_dtype not in {torch.float32, torch.float64}:
        raise ValueError("attention accumulator must be float32 or float64")
    fused = premean_all_layer_head_attention(attentions)
    reveal_attention = fused[:, 0, :-1, 1:]
    batch, rows, columns = reveal_attention.shape
    if rows != columns:
        raise ValueError(
            "target-to-observed attention must be square after removing None"
        )
    if tuple(orders.shape) != (batch, rows):
        raise ValueError(
            f"orders {tuple(orders.shape)} do not match attention {(batch, rows)}"
        )
    inverse = batched_inverse_permutations(
        orders.detach().to(device=reveal_attention.device, dtype=torch.long)
    )
    current = reveal_attention.gather(
        1, inverse.unsqueeze(2).expand(batch, rows, columns)
    )
    current = current.gather(
        2, inverse.unsqueeze(1).expand(batch, rows, columns)
    )
    exposure = inverse.unsqueeze(2) > inverse.unsqueeze(1)
    numerator = (current * exposure).sum(dim=0, dtype=accumulator_dtype)
    counts = exposure.sum(dim=0, dtype=accumulator_dtype)
    numerator.fill_diagonal_(0.0)
    counts.fill_diagonal_(0.0)
    return numerator, counts


@dataclass(frozen=True)
class AttentionWindow:
    numerator: torch.Tensor
    exposure_count: torch.Tensor
    samples: int

    def directed_matrix(self) -> torch.Tensor:
        matrix = self.numerator / self.exposure_count.clamp_min(1.0)
        matrix = torch.where(
            self.exposure_count > 0.0, matrix, torch.zeros_like(matrix)
        )
        matrix.fill_diagonal_(0.0)
        return matrix


class AttentionWindowAccumulator:


    def __init__(
        self,
        num_tokens: int,
        target_samples: int,
        device: torch.device | str,
    ) -> None:
        self.num_tokens = int(num_tokens)
        self.target_samples = int(target_samples)
        if self.num_tokens < 2:
            raise ValueError("OrderGPT requires at least two target tokens")
        if self.target_samples <= 0:
            raise ValueError("attention target_samples must be positive")
        self.device = torch.device(device)
        self.numerator = torch.zeros(
            self.num_tokens,
            self.num_tokens,
            dtype=torch.float64,
            device=self.device,
        )
        self.exposure_count = torch.zeros_like(self.numerator)
        self.samples = 0

    @torch.no_grad()
    def add(
        self,
        attentions: torch.Tensor | Sequence[torch.Tensor],
        orders: torch.Tensor,
    ) -> bool:
        incoming = int(orders.shape[0])
        if incoming <= 0:
            return False
        if self.samples + incoming > self.target_samples:
            raise ValueError(
                "attention_window_samples must be divisible into complete "
                f"training microbatches: buffered={self.samples}, "
                f"incoming={incoming}, target={self.target_samples}"
            )
        numerator, counts = target_to_observed_contributions(
            attentions, orders, accumulator_dtype=torch.float64
        )
        self.numerator.add_(numerator.to(self.device))
        self.exposure_count.add_(counts.to(self.device))
        self.samples += incoming
        return self.samples == self.target_samples

    def pop(self) -> AttentionWindow:
        if self.samples != self.target_samples:
            raise RuntimeError(
                f"attention window is incomplete: {self.samples}/{self.target_samples}"
            )
        window = AttentionWindow(
            numerator=self.numerator.clone(),
            exposure_count=self.exposure_count.clone(),
            samples=self.samples,
        )
        self.reset()
        return window

    def reset(self) -> None:
        self.numerator.zero_()
        self.exposure_count.zero_()
        self.samples = 0

    def state_dict(self) -> dict:
        return {
            "num_tokens": self.num_tokens,
            "target_samples": self.target_samples,
            "numerator": self.numerator.detach().cpu(),
            "exposure_count": self.exposure_count.detach().cpu(),
            "samples": self.samples,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["num_tokens"]) != self.num_tokens:
            raise ValueError("attention accumulator num_tokens mismatch")
        if int(state["target_samples"]) != self.target_samples:
            raise ValueError("attention accumulator target_samples mismatch")
        self.numerator.copy_(state["numerator"].to(self.device, torch.float64))
        self.exposure_count.copy_(
            state["exposure_count"].to(self.device, torch.float64)
        )
        self.samples = int(state["samples"])
        if not 0 <= self.samples < self.target_samples:
            raise ValueError("invalid restored attention sample count")


@dataclass(frozen=True)
class MutualFiedlerResult:
    directed_attention: torch.Tensor
    mutual_affinity: torch.Tensor
    laplacian: torch.Tensor
    eigenvalues: torch.Tensor
    vector: torch.Tensor
    raw_order: torch.Tensor
    reverse_order: torch.Tensor


def recover_mutual_fiedler(
    directed_attention: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> MutualFiedlerResult:


    directed = torch.as_tensor(
        directed_attention, dtype=torch.float64, device="cpu"
    ).clone()
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError("directed attention must be a square matrix")
    if int(directed.shape[0]) < 2:
        raise ValueError("Fiedler recovery requires at least two tokens")
    directed = torch.nan_to_num(directed, nan=0.0, posinf=0.0, neginf=0.0)
    directed.clamp_(min=0.0)
    directed.fill_diagonal_(0.0)
    affinity = torch.sqrt(directed * directed.T)
    affinity.fill_diagonal_(0.0)
    if float(affinity.sum().item()) <= tolerance:
        raise ValueError("mutual attention graph is empty")
    degree = affinity.sum(dim=1)
    laplacian = torch.diag(degree) - affinity
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    scale = max(1.0, float(eigenvalues.abs().max().item()))
    eig_tolerance = max(float(tolerance), 1e-15) * scale
    zero_multiplicity = int((eigenvalues.abs() <= eig_tolerance).sum().item())
    if zero_multiplicity != 1:
        raise ValueError(
            "mutual attention graph must be connected; "
            f"zero eigenvalue multiplicity={zero_multiplicity}"
        )
    vector = eigenvectors[:, 1].contiguous()
    if not torch.isfinite(vector).all():
        raise ValueError("Fiedler vector contains non-finite values")
    if float((vector.max() - vector.min()).item()) <= tolerance:
        raise ValueError("Fiedler vector is degenerate")
    if len(eigenvalues) > 2:
        gap = float((eigenvalues[2] - eigenvalues[1]).item())
        if gap <= eig_tolerance:
            raise ValueError("Fiedler eigenspace is not uniquely separated")
    raw = torch.tensor(
        sorted(
            range(int(vector.numel())),
            key=lambda index: (float(vector[index].item()), int(index)),
        ),
        dtype=torch.long,
    )
    return MutualFiedlerResult(
        directed_attention=directed,
        mutual_affinity=affinity,
        laplacian=laplacian,
        eigenvalues=eigenvalues,
        vector=vector,
        raw_order=raw,
        reverse_order=torch.flip(raw, dims=(0,)),
    )


@dataclass(frozen=True)
class OrientationResult:
    selected_order: torch.Tensor
    selected_reverse: bool
    raw_score: float
    reverse_score: float
    tied: bool


def orient_raw_reverse(fiedler: MutualFiedlerResult) -> OrientationResult:
    directed = fiedler.directed_attention.to(dtype=torch.float64, device="cpu")
    raw = fiedler.raw_order.to(dtype=torch.long, device="cpu")
    reverse = fiedler.reverse_order.to(dtype=torch.long, device="cpu")
    reordered = directed[raw][:, raw]
    rows, columns = torch.tril_indices(raw.numel(), raw.numel(), offset=-1)
    raw_score = math.fsum(reordered[rows, columns].tolist())
    reverse_score = math.fsum(reordered[columns, rows].tolist())
    tied = raw_score == reverse_score
    selected_reverse = reverse_score > raw_score or (
        tied and tuple(reverse.tolist()) < tuple(raw.tolist())
    )
    selected = reverse if selected_reverse else raw
    return OrientationResult(
        selected_order=selected.clone(),
        selected_reverse=selected_reverse,
        raw_score=raw_score,
        reverse_score=reverse_score,
        tied=tied,
    )


def oriented_minmax_priority(
    fiedler: MutualFiedlerResult,
    *,
    selected_reverse: bool,
) -> torch.Tensor:


    vector = fiedler.vector.to(dtype=torch.float64)
    span = vector.max() - vector.min()
    if float(span.item()) <= 1e-12:
        raise ValueError("cannot normalize a degenerate Fiedler vector")
    normalized = (vector - vector.min()) / span
    priority = normalized if selected_reverse else 1.0 - normalized
    expected_order = (
        fiedler.reverse_order if selected_reverse else fiedler.raw_order
    )
    ordered_priority = priority[expected_order]
    if bool((ordered_priority[:-1] < ordered_priority[1:]).any()):
        raise RuntimeError("oriented Fiedler priorities must be nonincreasing along the selected order")
    return priority.to(dtype=torch.float32)


__all__ = [
    "AttentionWindow",
    "AttentionWindowAccumulator",
    "MutualFiedlerResult",
    "OrientationResult",
    "batched_inverse_permutations",
    "orient_raw_reverse",
    "oriented_minmax_priority",
    "premean_all_layer_head_attention",
    "recover_mutual_fiedler",
    "target_to_observed_contributions",
]
