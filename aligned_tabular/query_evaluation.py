from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from aligned_tabular.model import AlignedTabularOrderGPT
from ordergpt.transformer import modulate


OPS = ("<=", ">=", "=")
OP_TO_CODE = {operator: code for code, operator in enumerate(OPS)}


def generate_varskip_queries(
    full_records: np.ndarray,
    cardinalities: Sequence[int],
    *,
    rng_seed: int = 1234,
    num_queries: int = 1000,
) -> list[dict[str, Any]]:


    records = np.asarray(full_records)
    if records.ndim != 2 or int(records.shape[0]) <= 0:
        raise ValueError("full_records must have non-empty shape [N, D]")
    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.shape != (int(records.shape[1]),):
        raise ValueError("cardinalities must have one entry per column")
    if int(num_queries) <= 0:
        raise ValueError("num_queries must be positive")

    rng = np.random.RandomState(int(rng_seed))
    num_rows, num_columns = map(int, records.shape)
    queries: list[dict[str, Any]] = []
    for query_id in range(int(num_queries)):
        num_filters = int(rng.randint(5, 12))
        anchor_row = int(rng.randint(0, num_rows))
        columns = rng.choice(
            num_columns,
            replace=False,
            size=num_filters,
        ).astype(np.int64, copy=False)
        operators = rng.choice(OPS, size=num_filters)
        predicates = []
        for column, sampled_operator in zip(columns.tolist(), operators.tolist()):
            operator = "=" if int(cards[column]) < 10 else str(sampled_operator)
            predicates.append(
                {
                    "column": int(column),
                    "operator": operator,
                    "value": int(records[anchor_row, column]),
                }
            )
        queries.append(
            {
                "query_id": int(query_id),
                "anchor_row": anchor_row,
                "num_filters": num_filters,
                "columns": [int(value) for value in columns.tolist()],
                "operators": [
                    str(predicate["operator"]) for predicate in predicates
                ],
                "values": [int(predicate["value"]) for predicate in predicates],
                "predicates": predicates,
            }
        )
    return queries


def _query_arrays(
    queries: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not queries:
        raise ValueError("queries must be non-empty")
    max_predicates = max(len(query["predicates"]) for query in queries)
    columns = np.full((len(queries), max_predicates), -1, dtype=np.int64)
    operators = np.full((len(queries), max_predicates), -1, dtype=np.int8)
    thresholds = np.zeros((len(queries), max_predicates), dtype=np.int64)
    counts = np.zeros(len(queries), dtype=np.int64)
    for query_id, query in enumerate(queries):
        predicates = query["predicates"]
        counts[query_id] = len(predicates)
        for predicate_id, predicate in enumerate(predicates):
            operator = str(predicate["operator"])
            if operator not in OP_TO_CODE:
                raise ValueError(f"unsupported query operator {operator!r}")
            columns[query_id, predicate_id] = int(predicate["column"])
            operators[query_id, predicate_id] = OP_TO_CODE[operator]
            thresholds[query_id, predicate_id] = int(predicate["value"])
    return columns, operators, thresholds, counts


def exact_query_cardinalities(
    full_records: np.ndarray,
    queries: Sequence[Mapping[str, Any]],
    *,
    row_chunk_size: int = 100_000,
    query_block_size: int = 32,
) -> np.ndarray:


    records = np.asarray(full_records)
    if records.ndim != 2:
        raise ValueError("full_records must have shape [N, D]")
    if int(row_chunk_size) <= 0 or int(query_block_size) <= 0:
        raise ValueError("chunk sizes must be positive")
    columns, operators, thresholds, predicate_counts = _query_arrays(queries)
    if bool((columns[columns >= 0] >= int(records.shape[1])).any()):
        raise ValueError("query column is outside the table")

    totals = np.zeros(len(queries), dtype=np.int64)
    max_predicates = int(columns.shape[1])
    for row_start in range(0, int(records.shape[0]), int(row_chunk_size)):
        row_stop = min(row_start + int(row_chunk_size), int(records.shape[0]))
        chunk = records[row_start:row_stop]
        rows = row_stop - row_start
        for query_start in range(0, len(queries), int(query_block_size)):
            query_stop = min(query_start + int(query_block_size), len(queries))
            block_size = query_stop - query_start
            matches = np.ones((block_size, rows), dtype=np.bool_)
            block_counts = predicate_counts[query_start:query_stop]
            for predicate_id in range(max_predicates):
                active_local = np.flatnonzero(block_counts > predicate_id)
                if not len(active_local):
                    continue
                global_ids = query_start + active_local
                block_columns = columns[global_ids, predicate_id]
                block_thresholds = thresholds[global_ids, predicate_id]
                block_operators = operators[global_ids, predicate_id]
                values = chunk[:, block_columns]
                for operator_code in range(len(OPS)):
                    op_local = np.flatnonzero(block_operators == operator_code)
                    if not len(op_local):
                        continue
                    selected = values[:, op_local]
                    selected_thresholds = block_thresholds[op_local][None, :]
                    if operator_code == OP_TO_CODE["<="]:
                        predicate_matches = selected <= selected_thresholds
                    elif operator_code == OP_TO_CODE[">="]:
                        predicate_matches = selected >= selected_thresholds
                    else:
                        predicate_matches = selected == selected_thresholds
                    matches[active_local[op_local]] &= predicate_matches.T
            totals[query_start:query_stop] += matches.sum(axis=1, dtype=np.int64)
    return totals


def build_allowed_masks(
    queries: Sequence[Mapping[str, Any]],
    cardinalities: Sequence[int],
    max_states: Optional[int] = None,
) -> np.ndarray:


    cards = np.asarray(cardinalities, dtype=np.int64)
    if cards.ndim != 1 or bool((cards <= 0).any()):
        raise ValueError("cardinalities must be a positive one-dimensional array")
    width = int(cards.size)
    state_width = int(cards.max()) if max_states is None else int(max_states)
    if state_width < int(cards.max()):
        raise ValueError("max_states is smaller than a column cardinality")
    states = np.arange(state_width, dtype=np.int64)
    valid = states[None, :] < cards[:, None]
    masks = np.broadcast_to(valid, (len(queries), width, state_width)).copy()
    for query_id, query in enumerate(queries):
        for predicate in query["predicates"]:
            column = int(predicate["column"])
            if column < 0 or column >= width:
                raise ValueError("query column is outside cardinalities")
            threshold = int(predicate["value"])
            operator = str(predicate["operator"])
            if operator == "<=":
                predicate_mask = states <= threshold
            elif operator == ">=":
                predicate_mask = states >= threshold
            elif operator == "=":
                predicate_mask = states == threshold
            else:
                raise ValueError(f"unsupported query operator {operator!r}")
            masks[query_id, column] &= predicate_mask
        if bool((~masks[query_id].any(axis=1)).any()):
            raise ValueError(f"query {query_id} leaves a column with no legal state")
    return masks


def q_error(estimate: int | float, actual: int | float) -> float:


    estimate_value = float(estimate)
    actual_value = float(actual)
    if (
        not math.isfinite(estimate_value)
        or not math.isfinite(actual_value)
        or estimate_value < 0
        or actual_value < 0
    ):
        raise ValueError("cardinalities must be finite and non-negative")
    if actual_value == 0.0 and estimate_value == 0.0:
        return 1.0
    if actual_value == 0.0:
        return estimate_value
    if estimate_value == 0.0:
        return actual_value
    return max(estimate_value / actual_value, actual_value / estimate_value)


def summarize_q_errors(values: Sequence[float]) -> dict[str, float]:


    errors = np.asarray(values, dtype=np.float64)
    if errors.ndim != 1 or int(errors.size) == 0:
        raise ValueError("q-error values must be a non-empty vector")
    if bool((~np.isfinite(errors)).any()) or bool((errors < 1.0).any()):
        raise ValueError("q-errors must be finite and at least one")
    p50 = float(np.median(errors))
    p99 = float(np.quantile(errors, 0.99))
    maximum = float(np.max(errors))
    return {
        "count": int(errors.size),
        "p50": p50,
        "p99": p99,
        "max": maximum,
        "p50_q_error": p50,
        "p99_q_error": p99,
        "max_q_error": maximum,
    }


@dataclass
class IncrementalKVCache:


    batch_size: int
    max_steps: int
    keys: list[Optional[torch.Tensor]]
    values: list[Optional[torch.Tensor]]
    next_step: int = 0


class IncrementalAlignedTabularSampler:


    def __init__(self, model: AlignedTabularOrderGPT):
        if not isinstance(model, AlignedTabularOrderGPT):
            raise TypeError("model must be an AlignedTabularOrderGPT")
        self.model = model

    def new_cache(self, batch_size: int) -> IncrementalKVCache:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        return IncrementalKVCache(
            batch_size=int(batch_size),
            max_steps=self.model.num_variables,
            keys=[None] * len(self.model.blocks),
            values=[None] * len(self.model.blocks),
        )

    def step_logits(
        self,
        values: torch.Tensor,
        conditions: Optional[torch.Tensor],
        orders: torch.Tensor,
        step: int,
    ) -> torch.Tensor:


        step = int(step)
        self.model._validate_values_and_orders(values, orders)
        if step < 0 or step >= self.model.num_variables:
            raise ValueError("step is outside the model width")
        effective_mask = self.model._effective_valid_state_mask(values, None)
        self.model._validate_values_against_mask(values, effective_mask)

        batch_size = int(values.shape[0])
        orders_long = orders.long()
        if step == 0:
            hidden = self.model.start_embedding.expand(batch_size, -1, -1)
        else:
            prefix_order = orders_long[:, :step]
            batch_indices = torch.arange(
                batch_size,
                dtype=torch.long,
                device=values.device,
            ).view(-1, 1)
            prefix_values = values.long()[batch_indices, prefix_order]
            observed = self.model.state_embedding(prefix_values)
            observed = observed + self.model.observed_variable_id_embedding(
                prefix_order
            )
            hidden = torch.cat(
                [self.model.start_embedding.expand(batch_size, -1, -1), observed],
                dim=1,
            )
        hidden = self.model.input_dropout(hidden)
        target_ids = orders_long[:, : step + 1]
        condition = self.model.target_variable_id_embedding(target_ids)
        global_condition = self.model._condition_vector(
            values,
            conditions,
            dtype=condition.dtype,
        )
        condition = condition + global_condition.unsqueeze(1)
        for block in self.model.blocks:
            hidden = block(hidden, condition)
        hidden = self.model.final_layer(hidden, condition)
        current_target = target_ids[:, -1]
        logits = self.model.state_head(hidden[:, -1, :])
        logits = logits + self.model.per_variable_state_bias[current_target]
        valid = effective_mask[current_target]
        return logits.masked_fill(~valid, float("-inf"))

    def next_logits(
        self,
        orders: torch.Tensor,
        step: int,
        previous_values: Optional[torch.Tensor],
        cache: IncrementalKVCache,
    ) -> torch.Tensor:


        step = int(step)
        if orders.ndim != 2 or int(orders.shape[1]) != self.model.num_variables:
            raise ValueError("orders must have shape [B, num_variables]")
        batch_size = int(orders.shape[0])
        if batch_size != int(cache.batch_size):
            raise ValueError("orders batch size does not match cache")
        if step != int(cache.next_step) or step < 0 or step >= cache.max_steps:
            raise ValueError(
                f"cache expects step {cache.next_step}, received step {step}"
            )

        target_ids = orders[:, step].long()
        if step == 0:
            if previous_values is not None:
                raise ValueError("previous_values must be None at step zero")
            hidden = self.model.start_embedding.expand(batch_size, -1, -1)
        else:
            if previous_values is None or tuple(previous_values.shape) != (batch_size,):
                raise ValueError("previous_values must have shape [B] after step zero")
            previous_variable_ids = orders[:, step - 1].long()
            hidden = self.model.state_embedding(previous_values.long()).unsqueeze(1)
            hidden = hidden + self.model.observed_variable_id_embedding(
                previous_variable_ids
            ).unsqueeze(1)
        hidden = self.model.input_dropout(hidden)
        condition = self.model.target_variable_id_embedding(target_ids).unsqueeze(1)

        for layer_id, block in enumerate(self.model.blocks):
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = block.adaLN(condition).chunk(6, dim=-1)
            attention_input = modulate(
                block.ln_1(hidden),
                shift_msa,
                scale_msa,
            )
            q, k, v = block.attn.c_attn(attention_input).split(
                block.attn.n_embd,
                dim=2,
            )
            head_dim = block.attn.n_embd // block.attn.n_head
            q = q.view(batch_size, 1, block.attn.n_head, head_dim).transpose(1, 2)
            k = k.view(batch_size, 1, block.attn.n_head, head_dim).transpose(1, 2)
            v = v.view(batch_size, 1, block.attn.n_head, head_dim).transpose(1, 2)
            q = block.attn.q_norm(q)
            k = block.attn.k_norm(k)

            if cache.keys[layer_id] is None:
                cache.keys[layer_id] = torch.empty(
                    batch_size,
                    block.attn.n_head,
                    cache.max_steps,
                    head_dim,
                    dtype=k.dtype,
                    device=k.device,
                )
                cache.values[layer_id] = torch.empty(
                    batch_size,
                    block.attn.n_head,
                    cache.max_steps,
                    head_dim,
                    dtype=v.dtype,
                    device=v.device,
                )
            key_cache = cache.keys[layer_id]
            value_cache = cache.values[layer_id]
            assert key_cache is not None and value_cache is not None
            key_cache[:, :, step : step + 1, :].copy_(k)
            value_cache[:, :, step : step + 1, :].copy_(v)

            if hasattr(F, "scaled_dot_product_attention"):
                attention_output = F.scaled_dot_product_attention(
                    q,
                    key_cache[:, :, : step + 1, :],
                    value_cache[:, :, : step + 1, :],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                scores = (q @ key_cache[:, :, : step + 1, :].transpose(-2, -1))
                scores = scores * (1.0 / math.sqrt(float(head_dim)))
                probabilities = F.softmax(scores, dim=-1)
                attention_output = probabilities @ value_cache[
                    :, :, : step + 1, :
                ]
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(
                batch_size,
                1,
                block.attn.n_embd,
            )
            attention_output = block.attn.resid_dropout(
                block.attn.c_proj(attention_output)
            )
            hidden = hidden + gate_msa * attention_output
            hidden = hidden + gate_mlp * block.mlp(
                modulate(block.ln_2(hidden), shift_mlp, scale_mlp)
            )

        hidden = self.model.final_layer(hidden, condition)
        logits = self.model.state_head(hidden[:, 0, :])
        logits = logits + self.model.per_variable_state_bias[target_ids]
        valid = self.model.valid_state_mask[target_ids]
        cache.next_step += 1
        return logits.masked_fill(~valid, float("-inf"))


@dataclass
class ProgressiveSampleOutput:
    log_weights: torch.Tensor
    sampled_values_by_step: torch.Tensor


def progressive_sample_log_weights(
    *,
    next_logits: Callable[[int, Optional[torch.Tensor]], torch.Tensor],
    orders: torch.Tensor,
    allowed_masks: torch.Tensor,
    path_query_ids: Optional[torch.Tensor] = None,
    uniforms: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> ProgressiveSampleOutput:


    if orders.ndim != 2:
        raise ValueError("orders must have shape [B, D]")
    batch_size, num_variables = map(int, orders.shape)
    if allowed_masks.ndim != 3 or int(allowed_masks.shape[1]) != num_variables:
        raise ValueError("allowed_masks must have shape [Q, D, Cmax]")
    if path_query_ids is None:
        if int(allowed_masks.shape[0]) not in {1, batch_size}:
            raise ValueError("allowed mask batch must be one or match paths")
        if int(allowed_masks.shape[0]) == 1:
            path_query_ids = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=orders.device,
            )
        else:
            path_query_ids = torch.arange(
                batch_size,
                dtype=torch.long,
                device=orders.device,
            )
    if tuple(path_query_ids.shape) != (batch_size,):
        raise ValueError("path_query_ids must have shape [B]")
    if uniforms is not None and tuple(uniforms.shape) != (
        batch_size,
        num_variables,
    ):
        raise ValueError("uniforms must have shape [B, D]")

    log_weights = torch.zeros(
        batch_size,
        dtype=torch.float64,
        device=orders.device,
    )
    sampled_values = torch.empty_like(orders, dtype=torch.long)
    previous_values: Optional[torch.Tensor] = None
    for step in range(num_variables):
        logits = next_logits(step, previous_values).float()
        if tuple(logits.shape[:1]) != (batch_size,):
            raise ValueError("next_logits returned a wrong batch size")
        target_ids = orders[:, step].long()
        allowed = allowed_masks[
            path_query_ids.long(),
            target_ids,
            :,
        ]
        if tuple(allowed.shape) != tuple(logits.shape):
            raise ValueError("allowed mask and logits have different shapes")
        if bool((~allowed.any(dim=1)).any()):
            raise ValueError("a path has no allowed state at the current step")

        allowed_logits = logits.masked_fill(~allowed, float("-inf"))
        log_mass = torch.logsumexp(allowed_logits, dim=-1) - torch.logsumexp(
            logits,
            dim=-1,
        )
        log_weights += log_mass.double()
        conditional_probabilities = F.softmax(allowed_logits, dim=-1)
        if uniforms is None:
            uniform = torch.rand(
                batch_size,
                dtype=conditional_probabilities.dtype,
                device=conditional_probabilities.device,
                generator=generator,
            )
        else:
            uniform = uniforms[:, step].to(
                device=conditional_probabilities.device,
                dtype=conditional_probabilities.dtype,
            )
        if bool((uniform < 0).any()) or bool((uniform >= 1).any()):
            raise ValueError("uniform variates must lie in [0, 1)")
        cumulative = conditional_probabilities.cumsum(dim=-1)


        cumulative = cumulative / cumulative[:, -1:].clamp_min(
            torch.finfo(cumulative.dtype).tiny
        )
        sampled = (cumulative <= uniform.unsqueeze(1)).sum(dim=1)
        sampled = sampled.clamp_max(int(logits.shape[1]) - 1).long()
        if not bool(allowed.gather(1, sampled.unsqueeze(1)).all()):
            raise RuntimeError("inverse-CDF sampling selected a masked state")
        sampled_values[:, step] = sampled
        previous_values = sampled
    return ProgressiveSampleOutput(
        log_weights=log_weights,
        sampled_values_by_step=sampled_values,
    )


__all__ = [
    "IncrementalAlignedTabularSampler",
    "IncrementalKVCache",
    "OPS",
    "ProgressiveSampleOutput",
    "build_allowed_masks",
    "exact_query_cardinalities",
    "generate_varskip_queries",
    "progressive_sample_log_weights",
    "q_error",
    "summarize_q_errors",
]
