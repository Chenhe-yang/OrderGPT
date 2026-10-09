from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, ndim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(inputs, self.weight.shape, self.weight, 1e-5)


def _causal_attention_probabilities(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:


    with torch.no_grad():
        query = query.detach()
        key = key.detach()
        sequence_length = int(query.size(-2))
        scores = (query @ key.transpose(-2, -1)) * (
            1.0 / math.sqrt(float(key.size(-1)))
        )
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores.masked_fill_(~causal_mask, float("-inf"))
        return F.softmax(scores, dim=-1)


def _mean_causal_attention_probabilities(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:


    with torch.no_grad():
        query = query.detach()
        key = key.detach()
        sequence_length = int(query.size(-2))
        head_count = int(query.size(1))
        if head_count <= 0:
            raise ValueError("attention must contain at least one head")
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        running = torch.zeros(
            query.size(0),
            1,
            sequence_length,
            sequence_length,
            dtype=torch.float32,
            device=query.device,
        )
        scale = 1.0 / math.sqrt(float(key.size(-1)))
        for head_index in range(head_count):
            scores = (
                query[:, head_index : head_index + 1]
                @ key[:, head_index : head_index + 1].transpose(-2, -1)
            ) * scale
            scores.masked_fill_(~causal_mask, float("-inf"))
            running.add_(F.softmax(scores, dim=-1).float())
        return running.div_(float(head_count))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if int(config.n_embd) % int(config.n_head) != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_dim = self.n_embd // self.n_head
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        batch_size, sequence_length, embedding_dim = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_head,
                self.head_dim,
            ).transpose(1, 2)

        query = self.q_norm(split_heads(query))
        key = self.k_norm(split_heads(key))
        value = split_heads(value)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            probabilities = _causal_attention_probabilities(query, key)
            attended = probabilities @ value

        attention = None
        if return_attn:
            attention_mean = _mean_causal_attention_probabilities(query, key)


            attention = attention_mean.expand(
                -1, self.n_head, -1, -1
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_dim,
        )
        attended = self.resid_dropout(self.c_proj(attended))
        if return_attn:
            return attended, attention
        return attended


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 6 * config.n_embd, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        return_attn: bool = False,
    ):
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(condition).chunk(6, dim=-1)
        attention_output = self.attn(
            modulate(self.ln_1(x), shift_attention, scale_attention),
            return_attn=return_attn,
        )
        if return_attn:
            attention_hidden, attention = attention_output
        else:
            attention_hidden = attention_output
            attention = None
        x = x + gate_attention * attention_hidden
        x = x + gate_mlp * self.mlp(
            modulate(self.ln_2(x), shift_mlp, scale_mlp)
        )
        if return_attn:
            return x, attention
        return x


class FinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = RMSNorm(config.n_embd)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 2 * config.n_embd, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaLN_final(condition).chunk(2, dim=-1)
        return modulate(self.ln_f(x), shift, scale)
