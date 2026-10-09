from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

from .transformer import Block, FinalLayer


@dataclass(frozen=True)
class LanguageModelConfig:


    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


_CORE_CONFIG_FIELDS = tuple(field.name for field in fields(LanguageModelConfig))


def filter_legacy_language_model_args(
    model_args: Mapping[str, Any],
) -> dict[str, Any]:


    if not isinstance(model_args, Mapping):
        raise TypeError("checkpoint model_args must be a mapping")
    if int(model_args.get("block_order_block_len", 1)) != 1:
        raise ValueError("legacy checkpoint is not token-level")
    if str(model_args.get("order_impl", "token")) != "token":
        raise ValueError("legacy checkpoint is not a token-order model")
    if str(model_args.get("position_encoding_mode", "absolute")) != "absolute":
        raise ValueError("legacy checkpoint does not use absolute identity")
    return {
        key: model_args[key]
        for key in _CORE_CONFIG_FIELDS
        if key in model_args
    }


class OrderGPTLanguageModel(nn.Module):


    def __init__(self, config: LanguageModelConfig):
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),

                "wpe": nn.Embedding(config.block_size + 1, config.n_embd),
                "wtpe": nn.Embedding(config.block_size, 128),
                "wnonee": nn.Embedding(1, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                "final_layer": FinalLayer(config),
            }
        )
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=True,
        )

        self.apply(self._init_weights)
        projection_std = 0.02 / math.sqrt(2.0 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.trunc_normal_(
                    parameter,
                    mean=0.0,
                    std=projection_std,
                    a=-3.0 * projection_std,
                    b=3.0 * projection_std,
                )

    @staticmethod
    def _validate_config(config: LanguageModelConfig) -> None:
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
            if int(getattr(config, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(config.n_embd) % int(config.n_head) != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= float(config.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            )

    def get_num_params(self, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.transformer.wpe.weight.numel()
        return count

    @staticmethod
    def _batch_gather(values: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
        batch_ids = torch.arange(
            values.size(0),
            device=values.device,
            dtype=torch.long,
        ).view(-1, 1)
        return values[batch_ids, orders]

    @staticmethod
    def _unshuffle(
        shuffled_values: torch.Tensor,
        orders: torch.Tensor,
    ) -> torch.Tensor:
        unshuffled = torch.zeros_like(shuffled_values)
        batch_ids = torch.arange(
            shuffled_values.size(0),
            device=shuffled_values.device,
            dtype=torch.long,
        ).view(-1, 1)
        unshuffled[batch_ids, orders] = shuffled_values
        return unshuffled

    @staticmethod
    def _validate_orders(idx: torch.Tensor, orders: torch.Tensor) -> None:
        if orders.shape != idx.shape:
            raise ValueError("orders must have the same [batch, tokens] shape as idx")
        if orders.device != idx.device:
            raise ValueError("orders and idx must be on the same device")
        if orders.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("orders must use an integer dtype")
        expected = torch.arange(
            idx.size(1),
            device=idx.device,
            dtype=torch.long,
        ).view(1, -1).expand_as(orders.long())
        if not torch.equal(torch.sort(orders.long(), dim=1).values, expected):
            raise ValueError("each orders row must be a complete token permutation")

    def sample_random_orders(self, idx: torch.Tensor) -> torch.Tensor:


        return torch.stack(
            [
                torch.randperm(idx.size(1), device=idx.device)
                for _ in range(idx.size(0))
            ],
            dim=0,
        )

    @staticmethod
    def _format_outputs(
        logits,
        loss,
        token_losses=None,
        attentions=None,
    ) -> tuple:
        outputs = [logits, loss]
        if token_losses is not None:
            outputs.append(token_losses)
        if attentions is not None:
            outputs.append(attentions)
        return tuple(outputs)

    def forward(
        self,
        idx: torch.Tensor,
        targets=None,
        mode: str | None = "Random",
        orders: torch.Tensor | None = None,
        return_token_loss: bool = False,
        return_attentions: bool = False,
        return_logits: bool = True,
    ) -> tuple:


        del targets
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, tokens]")
        if idx.size(1) > self.config.block_size:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds "
                f"block_size={self.config.block_size}"
            )
        if mode == "Random":
            if orders is not None:
                raise ValueError("orders must be omitted in random-order mode")
            orders = self.sample_random_orders(idx)
        elif mode is None:
            if orders is None:
                raise ValueError("explicit-order mode requires orders")
        else:
            raise ValueError("mode must be 'Random' or None (explicit orders)")
        orders = orders.long()
        self._validate_orders(idx, orders)
        return self._forward_explicit(
            idx,
            orders,
            return_token_loss=return_token_loss,
            return_attentions=return_attentions,
            return_logits=return_logits,
        )

    def _forward_explicit(
        self,
        idx: torch.Tensor,
        orders: torch.Tensor,
        *,
        return_token_loss: bool,
        return_attentions: bool,
        return_logits: bool,
    ) -> tuple:
        batch_size, sequence_length = idx.shape
        device = idx.device
        shuffled_tokens = self._batch_gather(idx, orders)
        targets = shuffled_tokens

        token_hidden = self.transformer.wte(shuffled_tokens)
        start_hidden = self.transformer.wnonee(
            torch.zeros((1, 1), dtype=torch.long, device=device)
        ).expand(batch_size, -1, -1)
        token_hidden = torch.cat([start_hidden, token_hidden], dim=1)

        identity_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=device,
        )
        observed_identity = self.transformer.wpe(
            torch.arange(
                sequence_length + 1,
                dtype=torch.long,
                device=device,
            )
        ).unsqueeze(0).expand(batch_size, -1, -1)
        observed_identity = torch.cat(
            [
                observed_identity[:, :1],
                self._batch_gather(observed_identity[:, 1:], orders),
            ],
            dim=1,
        )
        target_identity = self.transformer.wtpe(identity_ids)
        target_identity = target_identity.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        target_identity = self._batch_gather(target_identity, orders)
        target_identity = torch.cat(
            [target_identity, torch.zeros_like(target_identity[:, :1])],
            dim=1,
        )

        hidden = self.transformer.drop(token_hidden + observed_identity)
        attention_sum = None
        for block in self.transformer.h:
            if return_attentions:
                hidden, attention = block(
                    hidden,
                    target_identity,
                    return_attn=True,
                )
                layer_mean = attention.mean(dim=1, keepdim=True)
                attention_sum = (
                    layer_mean
                    if attention_sum is None
                    else attention_sum.add_(layer_mean)
                )
            else:
                hidden = block(hidden, target_identity)
        attentions = (
            attention_sum.div_(float(len(self.transformer.h)))
            if attention_sum is not None
            else None
        )
        hidden = self.transformer.final_layer(hidden, target_identity)

        logits = None
        loss = None
        token_output = None
        if return_logits:
            logits = self.lm_head(hidden)
            token_losses = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1),
                ignore_index=-1,
                reduction="none",
            ).view_as(targets)
            loss = token_losses.mean()
            if return_token_loss:
                token_output = token_losses.detach()
        elif return_token_loss:
            predictor_hidden = hidden[:, :-1].contiguous()
            flat_hidden = predictor_hidden.view(-1, predictor_hidden.size(-1))
            flat_targets = targets.contiguous().view(-1)
            chunks = []
            for start in range(0, flat_hidden.size(0), 16384):
                stop = min(start + 16384, flat_hidden.size(0))
                chunks.append(
                    F.cross_entropy(
                        self.lm_head(flat_hidden[start:stop]),
                        flat_targets[start:stop],
                        ignore_index=-1,
                        reduction="none",
                    )
                )
            token_losses = torch.cat(chunks).view_as(targets)
            loss = token_losses.mean()
            token_output = token_losses.detach()
        return self._format_outputs(
            logits,
            loss,
            token_losses=token_output,
            attentions=attentions,
        )

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        optimizer_groups = [
            {
                "params": [
                    parameter
                    for parameter in parameters.values()
                    if parameter.dim() >= 2
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    parameter
                    for parameter in parameters.values()
                    if parameter.dim() < 2
                ],
                "weight_decay": 0.0,
            },
        ]
        fused_available = "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters
        extra_args = (
            {"fused": True}
            if fused_available and device_type == "cuda"
            else {}
        )
        return torch.optim.AdamW(
            optimizer_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )
