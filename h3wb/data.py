from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import numpy as np


NUM_VARIABLES = 133
VALUE_DIM = 3
CONDITION_DIM = 3


LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)


@dataclass(frozen=True)
class RawH3WB:


    targets: np.ndarray
    coordinates_2d: np.ndarray
    group_ids: np.ndarray
    source_schema: str


def _archive_keys(path: Path) -> Sequence[str]:


    try:
        with np.load(path, allow_pickle=False) as archive:
            return tuple(archive.files)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"{path} is not a complete readable NPZ archive; the official "
            "Google Drive download may be truncated or may be an HTML page. "
            "Re-run data/h3wb/download.py with --force."
        ) from error


def _pose_array(
    value: object,
    *,
    dimensions: int,
    name: str,
) -> np.ndarray:


    array = np.asarray(value)
    if array.ndim < 3:
        raise ValueError(f"{name} must have at least three dimensions, got {array.shape}")
    if int(array.shape[-1]) < int(dimensions):
        raise ValueError(
            f"{name} last dimension must contain {dimensions} coordinates, "
            f"got {array.shape}"
        )
    joint_axes = [
        axis
        for axis, size in enumerate(array.shape[:-1])
        if int(size) == NUM_VARIABLES
    ]
    if not joint_axes:
        raise ValueError(
            f"{name} has no {NUM_VARIABLES}-keypoint axis: {array.shape}"
        )


    array = np.moveaxis(array, joint_axes[-1], -2)
    array = array[..., :dimensions]
    array = array.reshape(-1, NUM_VARIABLES, dimensions)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _load_dense_fixture(path: Path, keys: Sequence[str]) -> RawH3WB:
    target_aliases = ("targets", "pose_3d", "keypoints_3d", "positions_3d")
    condition_aliases = ("conditions", "pose_2d", "keypoints_2d", "positions_2d")
    target_key = next((name for name in target_aliases if name in keys), None)
    condition_key = next((name for name in condition_aliases if name in keys), None)
    if target_key is None or condition_key is None:
        raise ValueError(
            "unsupported dense H3WB fixture schema; expected one of "
            f"{target_aliases} and one of {condition_aliases}"
        )
    with np.load(path, allow_pickle=False) as archive:
        targets = _pose_array(
            archive[target_key], dimensions=VALUE_DIM, name=target_key
        )
        coordinates_2d = _pose_array(
            archive[condition_key], dimensions=2, name=condition_key
        )
        if targets.shape[0] != coordinates_2d.shape[0]:
            raise ValueError(
                "dense H3WB fixture targets and conditions have different sample counts"
            )
        if "group_ids" in keys:
            group_ids = np.asarray(archive["group_ids"])
            if group_ids.shape != (targets.shape[0],):
                raise ValueError(
                    "dense H3WB fixture group_ids must have shape [N]"
                )
            if group_ids.dtype.kind not in "iu":
                raise TypeError("dense H3WB fixture group_ids must be integers")
            group_ids = np.asarray(group_ids, dtype=np.int64)
        else:
            group_ids = np.arange(targets.shape[0], dtype=np.int64)
    return RawH3WB(
        targets=np.ascontiguousarray(targets),
        coordinates_2d=np.ascontiguousarray(coordinates_2d),
        group_ids=np.ascontiguousarray(group_ids),
        source_schema="dense_fixture",
    )


def _flatten_official_nested(
    nested_data: Mapping[object, object],
    *,
    source_schema: str,
) -> RawH3WB:


    target_chunks = []
    condition_chunks = []
    group_chunks = []
    next_group = 0
    for subject in sorted(nested_data, key=str):
        actions = nested_data[subject]
        if not isinstance(actions, Mapping):
            raise ValueError(f"official H3WB subject {subject!r} is not a mapping")
        for action in sorted(actions, key=str):
            action_data = actions[action]
            if not isinstance(action_data, Mapping):
                raise ValueError(
                    f"official H3WB action {subject}/{action} is not a mapping"
                )
            camera_sequences = []
            for camera_id in sorted(action_data, key=str):
                camera_data = action_data[camera_id]
                if not isinstance(camera_data, Mapping):
                    continue
                if "pose_2d" not in camera_data or "camera_3d" not in camera_data:
                    continue
                coordinates_2d = _pose_array(
                    camera_data["pose_2d"],
                    dimensions=2,
                    name=f"{subject}/{action}/{camera_id}/pose_2d",
                )
                targets = _pose_array(
                    camera_data["camera_3d"],
                    dimensions=VALUE_DIM,
                    name=f"{subject}/{action}/{camera_id}/camera_3d",
                )
                if targets.shape[0] != coordinates_2d.shape[0]:
                    raise ValueError(
                        f"official H3WB camera {subject}/{action}/{camera_id} "
                        "has mismatched 2D and 3D frame counts"
                    )
                camera_sequences.append((targets, coordinates_2d))
            if not camera_sequences:
                raise ValueError(
                    f"official H3WB action {subject}/{action} has no paired "
                    "pose_2d/camera_3d cameras"
                )
            frame_counts = {int(targets.shape[0]) for targets, _ in camera_sequences}
            if len(frame_counts) != 1:
                raise ValueError(
                    f"official H3WB action {subject}/{action} has unaligned "
                    f"camera frame counts: {sorted(frame_counts)}"
                )
            num_frames = frame_counts.pop()


            action_group_ids = np.full(num_frames, next_group, dtype=np.int64)
            for targets, coordinates_2d in camera_sequences:
                target_chunks.append(targets)
                condition_chunks.append(coordinates_2d)
                group_chunks.append(action_group_ids.copy())
            next_group += 1

    if not target_chunks:
        raise ValueError("official H3WB train_data contains no paired camera samples")
    return RawH3WB(
        targets=np.ascontiguousarray(
            np.concatenate(target_chunks, axis=0), dtype=np.float32
        ),
        coordinates_2d=np.ascontiguousarray(
            np.concatenate(condition_chunks, axis=0), dtype=np.float32
        ),
        group_ids=np.ascontiguousarray(np.concatenate(group_chunks), dtype=np.int64),
        source_schema=source_schema,
    )


def _load_official_nested(path: Path) -> RawH3WB:


    try:
        with np.load(path, allow_pickle=True) as archive:
            metadata = archive["metadata"].item()
            train_data = archive["train_data"].item()
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"failed to load official H3WB object archive: {path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError("official H3WB metadata entry must contain a mapping")
    if not isinstance(train_data, Mapping):
        raise ValueError("official H3WB train_data entry must contain a mapping")
    return _flatten_official_nested(
        train_data, source_schema="official_nested_train_data"
    )


def load_h3wb_train_npz(path: str | Path) -> RawH3WB:


    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"H3WB source archive does not exist: {source}")
    keys = _archive_keys(source)
    if {"metadata", "train_data"}.issubset(keys):
        return _load_official_nested(source)


    return _load_dense_fixture(source, keys)


def load_h3wb_i2d_test_zip(path: str | Path) -> RawH3WB:


    return _load_h3wb_test_zip(
        path,
        task_name="I2D",
        source_schema="official_i2d_test_zip",
    )


def _load_h3wb_test_zip(
    path: str | Path,
    *,
    task_name: str,
    source_schema: str,
) -> RawH3WB:


    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"H3WB {task_name} test archive does not exist: {source}"
        )
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            members = archive.namelist()
            if members != ["data.npy"]:
                raise ValueError(
                    f"official H3WB {task_name} test ZIP must contain only data.npy; "
                    f"got {members}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"H3WB {task_name} test ZIP failed CRC validation at "
                    f"{bad_member!r}"
                )
            payload = archive.read("data.npy")
        nested_data = np.load(BytesIO(payload), allow_pickle=True).item()
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"failed to load official H3WB {task_name} test ZIP: {source}"
        ) from error
    if not isinstance(nested_data, Mapping):
        raise ValueError(
            f"official H3WB {task_name} data.npy must contain a mapping"
        )
    return _flatten_official_nested(
        nested_data, source_schema=source_schema
    )


def pelvis_center_in_place(array: np.ndarray) -> np.ndarray:


    expected_prefix = (int(array.shape[0]), NUM_VARIABLES)
    if array.ndim != 3 or array.shape[:2] != expected_prefix:
        raise ValueError(f"pose array must have shape [N,{NUM_VARIABLES},C]")
    pelvis = 0.5 * (
        array[:, LEFT_HIP_INDEX, :] + array[:, RIGHT_HIP_INDEX, :]
    )
    array -= pelvis[:, None, :]
    return array


def apply_official_i2d_mask(
    normalized_coordinates_2d: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:


    coordinates = np.asarray(normalized_coordinates_2d, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (NUM_VARIABLES, 2):
        raise ValueError(
            f"normalized 2D coordinates must have shape [N,{NUM_VARIABLES},2]"
        )
    rng = np.random.default_rng(int(seed))
    num_samples = int(coordinates.shape[0])
    category = rng.random(num_samples)
    visible = np.ones((num_samples, NUM_VARIABLES), dtype=np.float32)

    random_rows = np.flatnonzero(category < 0.4)
    if random_rows.size:
        random_missing = rng.random((random_rows.size, NUM_VARIABLES)) < 0.25
        visible[random_rows] = (~random_missing).astype(np.float32)
    face_rows = np.flatnonzero((category >= 0.4) & (category < 0.6))
    visible[face_rows, FACE_SLICE] = 0.0
    left_hand_rows = np.flatnonzero((category >= 0.6) & (category < 0.8))
    visible[left_hand_rows, LEFT_HAND_SLICE] = 0.0
    right_hand_rows = np.flatnonzero(category >= 0.8)
    visible[right_hand_rows, RIGHT_HAND_SLICE] = 0.0

    masked = np.array(coordinates, dtype=np.float32, copy=True)
    masked[visible == 0.0] = 0.0
    return np.ascontiguousarray(
        np.concatenate((masked, visible[..., None]), axis=-1), dtype=np.float32
    )


def prepare_released_i2d_conditions(
    coordinates_2d: np.ndarray,
    *,
    train_axis_mean: np.ndarray,
    train_axis_std: np.ndarray,
) -> np.ndarray:


    coordinates = np.asarray(coordinates_2d, dtype=np.float32)
    mean = np.asarray(train_axis_mean, dtype=np.float32)
    std = np.asarray(train_axis_std, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (NUM_VARIABLES, 2):
        raise ValueError(f"released I2D coordinates must have shape [N,{NUM_VARIABLES},2]")
    if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0.0):
        raise ValueError("train condition mean/std must be valid two-vectors")
    visible = np.any(coordinates != 0.0, axis=-1)
    normalized = (coordinates - mean[None, None, :]) / std[None, None, :]
    normalized[~visible] = 0.0
    return np.ascontiguousarray(
        np.concatenate((normalized, visible[..., None].astype(np.float32)), axis=-1),
        dtype=np.float32,
    )


__all__ = [
    "CONDITION_DIM",
    "FACE_SLICE",
    "LEFT_HAND_SLICE",
    "LEFT_HIP_INDEX",
    "NUM_VARIABLES",
    "RIGHT_HAND_SLICE",
    "RIGHT_HIP_INDEX",
    "RawH3WB",
    "VALUE_DIM",
    "apply_official_i2d_mask",
    "load_h3wb_i2d_test_zip",
    "load_h3wb_train_npz",
    "pelvis_center_in_place",
    "prepare_released_i2d_conditions",
]
