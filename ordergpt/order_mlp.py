from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def priority_to_order(priority: torch.Tensor) -> torch.Tensor:


    if not torch.is_tensor(priority):
        raise TypeError("priority must be a tensor")
    if priority.ndim not in (1, 2) or int(priority.shape[-1]) < 2:
        raise ValueError("priority must have shape [D] or [B, D], with D >= 2")
    if not bool(torch.isfinite(priority).all()):
        raise ValueError("priority contains non-finite values")
    return torch.argsort(
        priority.detach().float(),
        dim=-1,
        descending=True,
        stable=True,
    )


def priority_mse_loss(
    predicted_priority: torch.Tensor,
    target_priority: torch.Tensor,
) -> torch.Tensor:


    if not torch.is_tensor(predicted_priority) or not torch.is_tensor(
        target_priority
    ):
        raise TypeError("predicted_priority and target_priority must be tensors")
    target = target_priority.detach().to(
        device=predicted_priority.device,
        dtype=predicted_priority.dtype,
    )
    if predicted_priority.shape != target.shape:
        raise ValueError(
            "target priority shape {} does not match prediction shape {}".format(
                tuple(target.shape), tuple(predicted_priority.shape)
            )
        )
    if not bool(torch.isfinite(predicted_priority).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise ValueError("predicted or target priority contains non-finite values")
    if bool((target < 0.0).any()) or bool((target > 1.0).any()):
        raise ValueError("target priority must lie in [0, 1]")
    return F.mse_loss(predicted_priority, target)


class OrderMLP(nn.Module):


    def __init__(
        self,
        num_variables: int,
        hidden_dims: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        self.num_variables = int(num_variables)
        if self.num_variables < 2:
            raise ValueError("num_variables must be at least two")
        hidden = tuple(int(value) for value in hidden_dims)
        if not hidden or any(value <= 0 for value in hidden):
            raise ValueError("hidden_dims must contain positive dimensions")
        dimensions = (
            self.num_variables * self.num_variables,
            *hidden,
            self.num_variables,
        )
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def _normalized_attention(
        self, directed_attention: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(directed_attention):
            raise TypeError("directed_attention must be a tensor")
        if directed_attention.ndim not in (2, 3):
            raise ValueError(
                "directed_attention must have shape [D, D] or [B, D, D]"
            )
        expected = (self.num_variables, self.num_variables)
        if tuple(directed_attention.shape[-2:]) != expected:
            raise ValueError(
                f"expected directed attention suffix {expected}, got "
                f"{tuple(directed_attention.shape[-2:])}"
            )
        matrix = directed_attention.detach().float()
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("directed_attention contains non-finite values")
        mean = matrix.mean(dim=(-2, -1), keepdim=True)
        scale = matrix.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        return (matrix - mean) / scale

    def forward_logits(self, directed_attention: torch.Tensor) -> torch.Tensor:
        normalized = self._normalized_attention(directed_attention)
        if normalized.ndim == 2:
            return self.net(normalized.reshape(1, -1)).squeeze(0)
        return self.net(normalized.reshape(normalized.shape[0], -1))

    def forward(self, directed_attention: torch.Tensor) -> torch.Tensor:


        return torch.sigmoid(self.forward_logits(directed_attention).float())

    def mse_loss(
        self,
        directed_attention: torch.Tensor,
        target_priority: torch.Tensor,
    ) -> torch.Tensor:
        return priority_mse_loss(self(directed_attention), target_priority)

    @torch.no_grad()
    def order_from_attention(
        self, directed_attention: torch.Tensor
    ) -> torch.Tensor:
        return priority_to_order(self(directed_attention))


__all__ = [
    "OrderMLP",
    "priority_mse_loss",
    "priority_to_order",
]
