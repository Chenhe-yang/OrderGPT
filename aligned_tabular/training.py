from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from aligned_continuous.training import (
    OrderController,
    TrainingTaskAdapter,
    autocast_context,
    main as shared_main,
)
from aligned_tabular.data import AlignedTabularDataset
from aligned_tabular.model import AlignedTabularConfig, AlignedTabularOrderGPT


def validate_tabular_config(config: Mapping[str, Any]) -> None:


    if int(config["variable_permutation_seed"]) < -1:
        raise ValueError(
            "tabular variable_permutation_seed must be -1 or >= 0; -1 is "
            "canonical identity and seed 0 is a real global column permutation"
        )
    if int(config["condition_layers"]) != 0:
        raise ValueError("unconditional categorical tables require condition_layers=0")


def build_dataset(config: Mapping[str, Any]) -> AlignedTabularDataset:
    return AlignedTabularDataset(
        str(config["data_dir"]),
        int(config["variable_permutation_seed"]),
    )


def build_model(
    dataset: AlignedTabularDataset,
    config: Mapping[str, Any],
) -> AlignedTabularOrderGPT:
    model_config = AlignedTabularConfig(
        num_variables=dataset.num_variables,
        max_states=dataset.max_states,
        cardinalities=dataset.cardinalities.tolist(),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        dropout=float(config["dropout"]),
        bias=bool(config["bias"]),
        condition_dim=0,
    )
    return AlignedTabularOrderGPT(model_config)


def evaluate_tabular(
    model: AlignedTabularOrderGPT,
    dataset: AlignedTabularDataset,
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
    total_nll = 0.0
    total_correct = 0
    try:
        with torch.no_grad():
            for batch_id, batch in enumerate(
                dataset.iter_batches(
                    split,
                    int(config["batch_size"]),
                    device=device,
                )
            ):
                if max_batches > 0 and batch_id >= max_batches:
                    break
                orders = controller.make(
                    int(batch.targets.shape[0]),
                    device,
                    iteration,
                    training=False,
                )
                with autocast_context(device, dtype):
                    output = model(batch.targets, batch.conditions, orders)
                samples = int(batch.targets.shape[0])
                total_samples += samples
                total_nll += float(output.step_losses.float().sum().item())
                predictions = output.logits_current.argmax(dim=-1)
                total_correct += int((predictions == batch.targets).sum().item())
    finally:
        controller.generator.set_state(saved_order_rng_state)
        if was_training:
            model.train()
    if total_samples == 0:
        raise ValueError(f"evaluation split {split!r} yielded no batches")
    variables = total_samples * dataset.num_variables
    nll = total_nll / variables
    return {
        "teacher_forced_nll_per_variable": nll,
        "teacher_forced_bits_per_variable": nll / math.log(2.0),
        "teacher_forced_state_accuracy": total_correct / variables,
        "num_samples": float(total_samples),
    }


TABULAR_TASK = TrainingTaskAdapter(
    build_dataset=build_dataset,
    build_model=build_model,
    evaluate=evaluate_tabular,
    validate_config=validate_tabular_config,
)


def main(
    dataset_name: Optional[str] = None,
    argv: Optional[Sequence[str]] = None,
) -> None:
    shared_main(
        dataset_name=dataset_name,
        argv=argv,
        task_adapter=TABULAR_TASK,
    )


__all__ = [
    "TABULAR_TASK",
    "build_dataset",
    "build_model",
    "evaluate_tabular",
    "main",
    "validate_tabular_config",
]
