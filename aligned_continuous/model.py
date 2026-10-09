from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from ordergpt.transformer import Block, FinalLayer


@dataclass
class AlignedContinuousConfig:
    num_variables: int
    value_dim: int
    condition_dim: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    condition_layers: int = 1
    dropout: float = 0.0
    bias: bool = True
    min_log_scale: float = -7.0
    max_log_scale: float = 5.0


@dataclass
class AlignedContinuousOutput:
    loss: torch.Tensor
    step_losses: torch.Tensor
    current_variable_losses: torch.Tensor
    means: torch.Tensor
    log_scales: torch.Tensor
    means_current: torch.Tensor
    log_scales_current: torch.Tensor
    attentions: List[torch.Tensor]
    target_variable_ids: torch.Tensor


class AlignedContinuousOrderGPT(nn.Module):


    target_condition_dim = 128

    def __init__(self, config: AlignedContinuousConfig):
        super().__init__()
        self.config = config
        self.num_variables = int(config.num_variables)
        self.value_dim = int(config.value_dim)
        self.condition_dim = int(config.condition_dim)
        self._validate_config()


        config.block_size = self.num_variables

        self.value_projection = nn.Linear(self.value_dim, config.n_embd)
        self.observed_variable_id_embedding = nn.Embedding(
            self.num_variables, config.n_embd
        )
        self.target_variable_id_embedding = nn.Embedding(
            self.num_variables, self.target_condition_dim
        )
        self.condition_projection = nn.Linear(self.condition_dim, config.n_embd)
        self.condition_variable_id_embedding = nn.Embedding(
            self.num_variables, config.n_embd
        )
        if int(config.condition_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(config.n_embd),
                nhead=int(config.n_head),
                dim_feedforward=4 * int(config.n_embd),
                dropout=float(config.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=bool(config.bias),
            )
            self.condition_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=int(config.condition_layers)
            )
        else:
            self.condition_encoder = nn.Identity()
        self.condition_to_observed = nn.Linear(config.n_embd, config.n_embd)
        self.condition_to_target = nn.Linear(
            config.n_embd, self.target_condition_dim
        )

        self.start_embedding = nn.Parameter(torch.empty(1, 1, config.n_embd))
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_layer = FinalLayer(config)
        self.mean_head = nn.Linear(config.n_embd, self.value_dim)
        self.log_scale_head = nn.Linear(config.n_embd, self.value_dim)
        self.per_variable_mean_bias = nn.Parameter(
            torch.zeros(self.num_variables, self.value_dim)
        )
        self.per_variable_log_scale_bias = nn.Parameter(
            torch.zeros(self.num_variables, self.value_dim)
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(
            self.start_embedding, mean=0.0, std=0.02, a=-0.06, b=0.06
        )
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                std = 0.02 / math.sqrt(2.0 * float(config.n_layer))
                nn.init.trunc_normal_(
                    parameter,
                    mean=0.0,
                    std=std,
                    a=-3.0 * std,
                    b=3.0 * std,
                )

    def _validate_config(self) -> None:
        for name in ("num_variables", "value_dim", "condition_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.config.n_layer) <= 0 or int(self.config.n_head) <= 0:
            raise ValueError("n_layer and n_head must be positive")
        if int(self.config.n_embd) <= 0:
            raise ValueError("n_embd must be positive")
        if int(self.config.n_embd) % int(self.config.n_head) != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if int(self.config.condition_layers) < 0:
            raise ValueError("condition_layers must be non-negative")
        if float(self.config.min_log_scale) >= float(self.config.max_log_scale):
            raise ValueError("min_log_scale must be smaller than max_log_scale")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06
            )

    def _validate_inputs(
        self, values: torch.Tensor, conditions: torch.Tensor, orders: torch.Tensor
    ) -> None:
        expected_values = (int(values.shape[0]), self.num_variables, self.value_dim)
        if tuple(values.shape) != expected_values:
            raise ValueError(
                f"values must have shape [B,{self.num_variables},{self.value_dim}]"
            )
        expected_conditions = (
            int(values.shape[0]),
            self.num_variables,
            self.condition_dim,
        )
        if tuple(conditions.shape) != expected_conditions:
            raise ValueError(
                "conditions must have shape "
                f"[B,{self.num_variables},{self.condition_dim}]"
            )
        if tuple(orders.shape) != (int(values.shape[0]), self.num_variables):
            raise ValueError(f"orders must have shape [B,{self.num_variables}]")
        if not torch.is_floating_point(values) or not torch.is_floating_point(
            conditions
        ):
            raise TypeError("values and conditions must be floating-point tensors")
        if orders.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("orders must use an integer dtype")
        if not bool(torch.isfinite(values).all()) or not bool(
            torch.isfinite(conditions).all()
        ):
            raise ValueError("values/conditions contain non-finite entries")
        if values.device != conditions.device or values.device != orders.device:
            raise ValueError("values, conditions and orders must share a device")
        expected = torch.arange(
            self.num_variables, device=orders.device, dtype=torch.long
        ).view(1, -1).expand_as(orders.long())
        if not torch.equal(torch.sort(orders.long(), dim=1).values, expected):
            raise ValueError("each orders row must be a complete current-ID permutation")

    def forward(
        self,
        values: torch.Tensor,
        conditions: torch.Tensor,
        orders: torch.Tensor,
        return_attentions: bool = False,
    ) -> AlignedContinuousOutput:
        self._validate_inputs(values, conditions, orders)

        batch_size = int(values.shape[0])
        orders_long = orders.long()
        batch_ids = torch.arange(
            batch_size, device=values.device, dtype=torch.long
        ).view(-1, 1).expand(-1, self.num_variables)
        canonical_ids = torch.arange(
            self.num_variables, device=values.device, dtype=torch.long
        ).view(1, -1).expand(batch_size, -1)

        condition_hidden = self.condition_projection(conditions)
        condition_hidden = condition_hidden + self.condition_variable_id_embedding(
            canonical_ids
        )
        condition_hidden = self.condition_encoder(condition_hidden)
        reveal_condition = condition_hidden[batch_ids, orders_long]
        reveal_values = values[batch_ids, orders_long]

        observed = self.value_projection(reveal_values)
        observed = observed + self.observed_variable_id_embedding(orders_long)
        observed = observed + self.condition_to_observed(reveal_condition)
        start = self.start_embedding.to(dtype=observed.dtype).expand(
            batch_size, -1, -1
        )
        hidden = self.input_dropout(torch.cat((start, observed), dim=1))

        target_condition = self.target_variable_id_embedding(orders_long)
        target_condition = target_condition + self.condition_to_target(reveal_condition)
        no_target = target_condition.new_zeros(
            batch_size, 1, self.target_condition_dim
        )
        target_condition = torch.cat((target_condition, no_target), dim=1)

        attentions: List[torch.Tensor] = []
        for block in self.blocks:
            if return_attentions:
                hidden, attention = block(
                    hidden, target_condition, return_attn=True
                )
                attentions.append(attention)
            else:
                hidden = block(hidden, target_condition)
        hidden = self.final_layer(hidden, target_condition)
        predictor_hidden = hidden[:, : self.num_variables]

        means = self.mean_head(predictor_hidden)
        means = means + self.per_variable_mean_bias[orders_long]
        log_scales = self.log_scale_head(predictor_hidden)
        log_scales = log_scales + self.per_variable_log_scale_bias[orders_long]
        log_scales = log_scales.clamp(
            min=float(self.config.min_log_scale),
            max=float(self.config.max_log_scale),
        )
        inv_variance = torch.exp(-2.0 * log_scales)
        element_nll = 0.5 * (reveal_values - means).square() * inv_variance
        element_nll = element_nll + log_scales + 0.5 * math.log(2.0 * math.pi)
        step_losses = element_nll.sum(dim=-1)
        loss = step_losses.mean()

        current_losses = torch.empty_like(step_losses)
        current_losses.scatter_(1, orders_long, step_losses)
        means_current = torch.empty_like(means)
        means_current.scatter_(
            1, orders_long.unsqueeze(-1).expand_as(means), means
        )
        log_scales_current = torch.empty_like(log_scales)
        log_scales_current.scatter_(
            1, orders_long.unsqueeze(-1).expand_as(log_scales), log_scales
        )
        return AlignedContinuousOutput(
            loss=loss,
            step_losses=step_losses,
            current_variable_losses=current_losses,
            means=means,
            log_scales=log_scales,
            means_current=means_current,
            log_scales_current=log_scales_current,
            attentions=attentions,
            target_variable_ids=orders_long,
        )


__all__ = [
    "AlignedContinuousOrderGPT",
    "AlignedContinuousConfig",
    "AlignedContinuousOutput",
]
