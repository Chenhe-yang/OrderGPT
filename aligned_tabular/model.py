import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

from ordergpt.transformer import Block, FinalLayer


@dataclass
class AlignedTabularConfig:


    num_variables: int
    max_states: int
    cardinalities: Optional[Sequence[int]] = None
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    condition_dim: int = 0


@dataclass
class AlignedTabularOutput:


    loss: torch.Tensor
    step_losses: torch.Tensor
    current_variable_losses: torch.Tensor
    logits: torch.Tensor
    attentions: List[torch.Tensor]
    logits_current: torch.Tensor
    target_variable_ids: torch.Tensor


class AlignedTabularOrderGPT(nn.Module):


    target_condition_dim = 128

    def __init__(self, config: AlignedTabularConfig):
        super().__init__()
        self.config = config
        self.num_variables = int(config.num_variables)
        self.max_states = int(config.max_states)
        self._validate_config()

        cardinalities = self._normalized_cardinalities(config.cardinalities)
        self.register_buffer(
            "cardinalities",
            torch.tensor(cardinalities, dtype=torch.long),
            persistent=True,
        )
        state_ids = torch.arange(self.max_states, dtype=torch.long).view(1, -1)
        cardinality_tensor = torch.tensor(cardinalities, dtype=torch.long).view(-1, 1)
        self.register_buffer(
            "valid_state_mask",
            state_ids < cardinality_tensor,
            persistent=True,
        )


        config.block_size = self.num_variables

        self.state_embedding = nn.Embedding(self.max_states, config.n_embd)
        self.observed_variable_id_embedding = nn.Embedding(self.num_variables, config.n_embd)
        self.target_variable_id_embedding = nn.Embedding(
            self.num_variables,
            self.target_condition_dim,
        )
        if int(config.condition_dim) > 0:
            self.condition_projection = nn.Sequential(
                nn.Linear(int(config.condition_dim), self.target_condition_dim),
                nn.SiLU(),
                nn.Linear(self.target_condition_dim, self.target_condition_dim),
            )
        else:
            self.condition_projection = None
        self.start_embedding = nn.Parameter(torch.empty(1, 1, config.n_embd))
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_layer = FinalLayer(config)


        self.state_head = nn.Linear(config.n_embd, self.max_states, bias=True)
        self.per_variable_state_bias = nn.Parameter(
            torch.zeros(self.num_variables, self.max_states)
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.start_embedding, mean=0.0, std=0.02, a=-0.06, b=0.06)
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

    def _validate_config(self):
        if self.num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if self.max_states <= 0:
            raise ValueError("max_states must be positive")
        if int(self.config.n_layer) <= 0:
            raise ValueError("n_layer must be positive")
        if int(self.config.n_head) <= 0:
            raise ValueError("n_head must be positive")
        if int(self.config.n_embd) <= 0:
            raise ValueError("n_embd must be positive")
        if int(self.config.n_embd) % int(self.config.n_head) != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if int(self.config.condition_dim) < 0:
            raise ValueError("condition_dim must be non-negative")

    def _normalized_cardinalities(self, cardinalities: Optional[Sequence[int]]) -> List[int]:
        if cardinalities is None:
            values = [self.max_states] * self.num_variables
        else:
            values = [int(value) for value in cardinalities]
        if len(values) != self.num_variables:
            raise ValueError(
                "cardinalities must have one entry per variable: expected %d, got %d"
                % (self.num_variables, len(values))
            )
        for variable_id, value in enumerate(values):
            if value <= 0 or value > self.max_states:
                raise ValueError(
                    "cardinalities[%d]=%d must be in [1, max_states=%d]"
                    % (variable_id, value, self.max_states)
                )
        return values

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    @staticmethod
    def _require_integer_tensor(name: str, value: Any):
        if not torch.is_tensor(value):
            raise TypeError("%s must be a torch.Tensor" % name)
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if value.dtype not in integer_dtypes:
            raise TypeError("%s must use an integer dtype, got %s" % (name, value.dtype))

    def _validate_values_and_orders(self, values: torch.Tensor, orders: torch.Tensor):
        self._require_integer_tensor("values", values)
        self._require_integer_tensor("orders", orders)
        if values.ndim != 2:
            raise ValueError("values must have shape [B, D]")
        if orders.ndim != 2:
            raise ValueError("orders must have shape [B, D]")
        if tuple(values.shape) != tuple(orders.shape):
            raise ValueError("values and orders must have identical [B, D] shapes")
        if int(values.size(0)) <= 0:
            raise ValueError("values must contain at least one sample")
        if int(values.size(1)) != self.num_variables:
            raise ValueError(
                "values/orders width must equal num_variables=%d" % self.num_variables
            )
        if values.device != orders.device:
            raise ValueError("values and orders must be on the same device")

        orders_long = orders.long()
        expected = torch.arange(
            self.num_variables,
            device=orders.device,
            dtype=torch.long,
        ).view(1, -1).expand_as(orders_long)
        if not torch.equal(torch.sort(orders_long, dim=1).values, expected):
            raise ValueError(
                "each orders row must be a complete permutation of 0..%d"
                % (self.num_variables - 1)
            )

        values_long = values.long()
        if bool((values_long < 0).any()) or bool((values_long >= self.max_states).any()):
            raise ValueError("values contain a state id outside [0, max_states)")

    def _effective_valid_state_mask(
        self,
        values: torch.Tensor,
        valid_state_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        base_mask = self.valid_state_mask.to(device=values.device)
        if valid_state_mask is None:
            mask = base_mask
        else:
            if not torch.is_tensor(valid_state_mask):
                raise TypeError("valid_state_mask must be a torch.Tensor")
            if valid_state_mask.dtype != torch.bool:
                raise TypeError("valid_state_mask must use dtype=torch.bool")
            if tuple(valid_state_mask.shape) != (self.num_variables, self.max_states):
                raise ValueError(
                    "valid_state_mask must have shape (%d, %d)"
                    % (self.num_variables, self.max_states)
                )


            mask = base_mask & valid_state_mask.to(device=values.device)
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("every variable must have at least one valid state")
        return mask

    def _validate_values_against_mask(
        self,
        values: torch.Tensor,
        valid_state_mask: torch.Tensor,
    ):
        batch_size = int(values.size(0))
        variable_ids = torch.arange(
            self.num_variables,
            device=values.device,
            dtype=torch.long,
        ).view(1, -1).expand(batch_size, -1)
        legal = valid_state_mask[variable_ids, values.long()]
        if not bool(legal.all()):
            first_bad = torch.nonzero(~legal, as_tuple=False)[0]
            sample_id = int(first_bad[0].item())
            variable_id = int(first_bad[1].item())
            state_id = int(values[sample_id, variable_id].item())
            raise ValueError(
                "values[%d, %d]=%d is illegal for that variable"
                % (sample_id, variable_id, state_id)
            )

    def _condition_vector(
        self,
        values: torch.Tensor,
        conditions: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = int(values.size(0))
        condition_dim = int(self.config.condition_dim)
        if condition_dim == 0:
            if conditions is not None and tuple(conditions.shape) != (batch_size, 0):
                raise ValueError(
                    "conditions must have shape [B,0] when condition_dim=0"
                )
            return torch.zeros(
                batch_size,
                self.target_condition_dim,
                device=values.device,
                dtype=dtype,
            )
        if conditions is None:
            conditions = torch.zeros(
                batch_size,
                condition_dim,
                device=values.device,
                dtype=dtype,
            )
        else:
            if not torch.is_tensor(conditions):
                raise TypeError("conditions must be a torch.Tensor or None")
            if not torch.is_floating_point(conditions):
                raise TypeError("conditions must use a floating-point dtype")
            if tuple(conditions.shape) != (batch_size, condition_dim):
                raise ValueError(
                    "conditions must have shape (%d, %d)"
                    % (batch_size, condition_dim)
                )
            if conditions.device != values.device:
                raise ValueError("conditions and values must be on the same device")
            if not bool(torch.isfinite(conditions).all()):
                raise ValueError("conditions contain non-finite values")
        projected = self.condition_projection(conditions.to(dtype=dtype))
        return projected

    def forward(
        self,
        values: torch.Tensor,
        conditions: Optional[torch.Tensor],
        orders: torch.Tensor,
        return_attentions: bool = False,
        valid_state_mask: Optional[torch.Tensor] = None,
    ) -> AlignedTabularOutput:


        self._validate_values_and_orders(values, orders)
        effective_mask = self._effective_valid_state_mask(values, valid_state_mask)
        self._validate_values_against_mask(values, effective_mask)

        values_long = values.long()
        orders_long = orders.long()
        batch_size = int(values.size(0))
        batch_indices = torch.arange(
            batch_size,
            device=values.device,
            dtype=torch.long,
        ).view(-1, 1).expand(-1, self.num_variables)
        reveal_values = values_long[batch_indices, orders_long]

        observed = self.state_embedding(reveal_values)
        observed = observed + self.observed_variable_id_embedding(orders_long)
        start = self.start_embedding.to(dtype=observed.dtype).expand(batch_size, -1, -1)
        hidden = torch.cat([start, observed], dim=1)
        hidden = self.input_dropout(hidden)

        target_condition = self.target_variable_id_embedding(orders_long)
        no_target = target_condition.new_zeros(
            batch_size,
            1,
            self.target_condition_dim,
        )
        target_condition = torch.cat([target_condition, no_target], dim=1)
        global_condition = self._condition_vector(
            values,
            conditions,
            dtype=target_condition.dtype,
        )
        target_condition = target_condition + global_condition.unsqueeze(1)

        attentions = []
        for block in self.blocks:
            if return_attentions:
                hidden, attention = block(
                    hidden,
                    target_condition,
                    return_attn=True,
                )
                attentions.append(attention)
            else:
                hidden = block(hidden, target_condition)
        hidden = self.final_layer(hidden, target_condition)

        predictor_hidden = hidden[:, : self.num_variables, :]
        logits = self.state_head(predictor_hidden)
        logits = logits + self.per_variable_state_bias[orders_long]
        reveal_valid_mask = effective_mask[orders_long]
        logits = logits.masked_fill(~reveal_valid_mask, float("-inf"))

        step_losses = F.cross_entropy(
            logits.reshape(-1, self.max_states),
            reveal_values.reshape(-1),
            reduction="none",
        ).view(batch_size, self.num_variables)
        loss = step_losses.mean()

        current_variable_losses = torch.empty_like(step_losses)
        current_variable_losses.scatter_(1, orders_long, step_losses)
        logits_current = torch.empty_like(logits)
        logits_current.scatter_(
            1,
            orders_long.unsqueeze(-1).expand(-1, -1, self.max_states),
            logits,
        )

        return AlignedTabularOutput(
            loss=loss,
            step_losses=step_losses,
            current_variable_losses=current_variable_losses,
            logits=logits,
            attentions=attentions,
            logits_current=logits_current,
            target_variable_ids=orders_long,
        )
