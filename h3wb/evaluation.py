from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from h3wb.data import LEFT_HIP_INDEX, NUM_VARIABLES, RIGHT_HIP_INDEX


BODY_AND_FEET_SLICE = slice(0, 23)
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)
HANDS_SLICE = slice(91, 133)
NOSE_INDEX = 0
LEFT_HAND_WRIST_INDEX = 91
RIGHT_HAND_WRIST_INDEX = 112


def _validate_pose_pair(
    predictions_mm: torch.Tensor, targets_mm: torch.Tensor
) -> None:
    expected = (int(predictions_mm.shape[0]), NUM_VARIABLES, 3)
    if tuple(predictions_mm.shape) != expected or tuple(targets_mm.shape) != expected:
        raise ValueError(
            f"H3WB poses must both have shape [N,{NUM_VARIABLES},3]"
        )
    if not torch.is_floating_point(predictions_mm) or not torch.is_floating_point(
        targets_mm
    ):
        raise TypeError("H3WB poses must be floating-point tensors")
    if predictions_mm.device != targets_mm.device:
        raise ValueError("predictions and targets must share a device")
    if not bool(torch.isfinite(predictions_mm).all()) or not bool(
        torch.isfinite(targets_mm).all()
    ):
        raise ValueError("predictions/targets contain NaN or Inf")


def official_mpjpe_metrics(
    predictions_mm: torch.Tensor, targets_mm: torch.Tensor
) -> Dict[str, float]:


    _validate_pose_pair(predictions_mm, targets_mm)
    if int(predictions_mm.shape[0]) == 0:
        raise ValueError("at least one H3WB pose is required")
    residual = predictions_mm.float() - targets_mm.float()
    pelvis_residual = 0.5 * (
        residual[:, LEFT_HIP_INDEX] + residual[:, RIGHT_HIP_INDEX]
    )
    pelvis_aligned = residual - pelvis_residual[:, None, :]
    pelvis_distance = torch.linalg.vector_norm(pelvis_aligned, dim=-1)

    nose_aligned_face = pelvis_aligned - pelvis_aligned[:, NOSE_INDEX : NOSE_INDEX + 1]
    left_wrist_aligned = pelvis_aligned - pelvis_aligned[
        :, LEFT_HAND_WRIST_INDEX : LEFT_HAND_WRIST_INDEX + 1
    ]
    right_wrist_aligned = pelvis_aligned - pelvis_aligned[
        :, RIGHT_HAND_WRIST_INDEX : RIGHT_HAND_WRIST_INDEX + 1
    ]
    left_hand_local = torch.linalg.vector_norm(
        left_wrist_aligned[:, LEFT_HAND_SLICE], dim=-1
    ).mean()
    right_hand_local = torch.linalg.vector_norm(
        right_wrist_aligned[:, RIGHT_HAND_SLICE], dim=-1
    ).mean()

    return {
        "free_running_mpjpe_whole_body_mm": float(pelvis_distance.mean().item()),
        "free_running_mpjpe_body_mm": float(
            pelvis_distance[:, BODY_AND_FEET_SLICE].mean().item()
        ),
        "free_running_mpjpe_face_pelvis_aligned_mm": float(
            pelvis_distance[:, FACE_SLICE].mean().item()
        ),
        "free_running_mpjpe_hands_pelvis_aligned_mm": float(
            pelvis_distance[:, HANDS_SLICE].mean().item()
        ),
        "free_running_mpjpe_face_nose_aligned_mm": float(
            torch.linalg.vector_norm(
                nose_aligned_face[:, FACE_SLICE], dim=-1
            ).mean().item()
        ),
        "free_running_mpjpe_hands_wrist_aligned_mm": float(
            (0.5 * (left_hand_local + right_hand_local)).item()
        ),
    }


def restore_canonical_ids(
    current_id_poses: torch.Tensor, current_to_canonical: Sequence[int]
) -> torch.Tensor:


    if current_id_poses.ndim != 3:
        raise ValueError("current_id_poses must have shape [N,D,V]")
    width = int(current_id_poses.shape[1])
    mapping = tuple(int(value) for value in current_to_canonical)
    if len(mapping) != width or sorted(mapping) != list(range(width)):
        raise ValueError("current_to_canonical must be a complete permutation")
    canonical = torch.empty_like(current_id_poses)
    indices = torch.tensor(mapping, dtype=torch.long, device=current_id_poses.device)
    canonical[:, indices] = current_id_poses
    return canonical


def target_normalization_in_current_ids(
    metadata: Mapping[str, object], current_to_canonical: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:


    normalization = metadata.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("H3WB metadata has no normalization object")
    mean_value = normalization.get("target_mean")
    std_value = normalization.get("target_std")
    if mean_value is None or std_value is None:
        target_info = normalization.get("targets")
        if isinstance(target_info, Mapping):
            mean_value = target_info.get("mean")
            std_value = target_info.get("std")
    width = len(current_to_canonical)

    def expand(value: object, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (3,):
            array = np.broadcast_to(array, (width, 3)).copy()
        if array.shape != (width, 3) or not np.isfinite(array).all():
            raise ValueError(f"target {name} must be finite [3] or [{width},3]")
        return array

    canonical_mean = expand(mean_value, "mean")
    canonical_std = expand(std_value, "std")
    if np.any(canonical_std <= 0.0):
        raise ValueError("target standard deviations must be positive")
    permutation = np.asarray(tuple(current_to_canonical), dtype=np.int64)
    return (
        torch.from_numpy(np.ascontiguousarray(canonical_mean[permutation])),
        torch.from_numpy(np.ascontiguousarray(canonical_std[permutation])),
    )


def denormalize_current_id_poses(
    normalized_poses: torch.Tensor,
    mean_current: torch.Tensor,
    std_current: torch.Tensor,
) -> torch.Tensor:
    if normalized_poses.ndim != 3:
        raise ValueError("normalized_poses must have shape [N,D,V]")
    expected = tuple(normalized_poses.shape[1:])
    if tuple(mean_current.shape) != expected or tuple(std_current.shape) != expected:
        raise ValueError("normalization tensors do not match pose dimensions")
    return normalized_poses * std_current.to(normalized_poses) + mean_current.to(
        normalized_poses
    )


@torch.no_grad()
def greedy_mean_rollout(
    model: torch.nn.Module,
    conditions: torch.Tensor,
    orders: torch.Tensor,
    *,
    autocast_context,
) -> torch.Tensor:


    if conditions.ndim != 3 or orders.ndim != 2:
        raise ValueError("conditions/orders must have shapes [B,D,C] and [B,D]")
    batch_size, width = int(orders.shape[0]), int(orders.shape[1])
    if tuple(conditions.shape[:2]) != (batch_size, width):
        raise ValueError("conditions and orders disagree on batch/variable dimensions")
    value_dim = int(getattr(model, "value_dim"))
    generated = torch.zeros(
        batch_size,
        width,
        value_dim,
        dtype=conditions.dtype,
        device=conditions.device,
    )
    batch_ids = torch.arange(batch_size, device=conditions.device)
    for step in range(width):
        with autocast_context():
            output = model(generated, conditions, orders)
        next_values = output.means[:, step].to(dtype=generated.dtype)
        generated[batch_ids, orders[:, step].long()] = next_values
    return generated


__all__ = [
    "denormalize_current_id_poses",
    "greedy_mean_rollout",
    "official_mpjpe_metrics",
    "restore_canonical_ids",
    "target_normalization_in_current_ids",
]
