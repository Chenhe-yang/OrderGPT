"""Prepare grouped, train-normalized H3WB masked-I2D arrays for Appendix C."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h3wb.data import (
    CONDITION_DIM,
    LEFT_HIP_INDEX,
    NUM_VARIABLES,
    RIGHT_HIP_INDEX,
    VALUE_DIM,
    apply_official_i2d_mask,
    load_h3wb_i2d_test_zip,
    load_h3wb_train_npz,
    pelvis_center_in_place,
    prepare_released_i2d_conditions,
)


FORMAT_VERSION = 1
DEFAULT_SOURCE = Path("data/h3wb/raw/h3wb_train.npz")
DEFAULT_TEST_ZIP = Path("data/h3wb/raw/h3wb_i2d_test.zip")
DEFAULT_OUTPUT = Path("data/h3wb/processed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array, dtype=np.float32))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _split_indices(
    group_ids: np.ndarray,
    *,
    val_fraction: float,
    discovery_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    if not 0.0 < float(discovery_fraction) < 1.0:
        raise ValueError("discovery_fraction must be in (0,1)")
    if float(val_fraction) + float(discovery_fraction) >= 1.0:
        raise ValueError("val_fraction + discovery_fraction must be smaller than 1")
    groups = np.unique(np.asarray(group_ids, dtype=np.int64))
    if groups.size < 3:
        raise ValueError("at least three independent groups are required")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    group_sizes = {
        int(group): int(np.sum(group_ids == group)) for group in groups
    }

    def prefix_nearest_sample_target(
        candidates: np.ndarray,
        target_samples: float,
        *,
        minimum_remaining_groups: int,
    ) -> int:
        maximum_take = int(candidates.size) - int(minimum_remaining_groups)
        if maximum_take < 1:
            raise ValueError("split fractions leave too few independent groups")
        cumulative = np.cumsum(
            [group_sizes[int(group)] for group in candidates[:maximum_take]],
            dtype=np.int64,
        )
        return int(np.argmin(np.abs(cumulative - float(target_samples)))) + 1

    total_samples = int(group_ids.size)
    num_val_groups = prefix_nearest_sample_target(
        groups,
        total_samples * float(val_fraction),
        minimum_remaining_groups=2,
    )
    remaining = groups[num_val_groups:]
    num_discovery_groups = prefix_nearest_sample_target(
        remaining,
        total_samples * float(discovery_fraction),
        minimum_remaining_groups=1,
    )

    group_sets = {
        "val": groups[:num_val_groups],
        "discovery": remaining[:num_discovery_groups],
        "train": remaining[num_discovery_groups:],
    }
    result: Dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "discovery"):
        indices = np.flatnonzero(np.isin(group_ids, group_sets[split_name]))
        if indices.size == 0:
            raise ValueError(f"derived {split_name} split is empty")
        rng.shuffle(indices)
        result[split_name] = np.asarray(indices, dtype=np.int64)
    return result


def _axis_statistics(array: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = array[indices]
    mean = selected.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = selected.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("training normalization statistics contain NaN or Inf")
    if np.any(std < 1e-8):
        bad = np.flatnonzero(std < 1e-8).tolist()
        raise ValueError(f"training normalization has near-zero axes: {bad}")
    return mean, std


def _existing_outputs(output_dir: Path) -> Sequence[Path]:
    names = ["metadata.json"]
    for split_name in ("train", "val", "discovery", "test"):
        names.extend(
            [f"{split_name}_targets.npy", f"{split_name}_conditions.npy"]
        )
    return tuple(output_dir / name for name in names if (output_dir / name).exists())


def _prepare_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.1,
    discovery_fraction: float = 0.1,
    split_seed: int = 1337,
    mask_seed: int = 7331,
    test_zip_path: Optional[str | Path] = None,
    overwrite: bool = False,
) -> Dict[str, object]:


    source = Path(source_path)
    destination = Path(output_dir)
    existing = _existing_outputs(destination)
    if existing and not overwrite:
        raise FileExistsError(
            "prepared H3WB outputs already exist; pass overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)

    raw = load_h3wb_train_npz(source)
    targets = np.array(raw.targets, dtype=np.float32, copy=True, order="C")
    coordinates_2d = np.array(
        raw.coordinates_2d, dtype=np.float32, copy=True, order="C"
    )
    if targets.shape != (targets.shape[0], NUM_VARIABLES, VALUE_DIM):
        raise ValueError(
            f"H3WB targets must have shape [N,{NUM_VARIABLES},{VALUE_DIM}]"
        )
    if coordinates_2d.shape != (targets.shape[0], NUM_VARIABLES, 2):
        raise ValueError(
            f"H3WB 2D coordinates must have shape [N,{NUM_VARIABLES},2]"
        )
    if raw.group_ids.shape != (targets.shape[0],):
        raise ValueError("H3WB group_ids must have one entry per sample")

    pelvis_center_in_place(targets)
    split_indices = _split_indices(
        raw.group_ids,
        val_fraction=float(val_fraction),
        discovery_fraction=float(discovery_fraction),
        seed=int(split_seed),
    )

    target_mean, target_std = _axis_statistics(targets, split_indices["train"])
    condition_mean, condition_std = _axis_statistics(
        coordinates_2d, split_indices["train"]
    )
    targets -= target_mean[None, None, :]
    targets /= target_std[None, None, :]
    coordinates_2d -= condition_mean[None, None, :]
    coordinates_2d /= condition_std[None, None, :]

    split_metadata: Dict[str, object] = {}
    split_mask_seeds = {
        "train": int(mask_seed),
        "val": int(mask_seed) + 1,
        "discovery": int(mask_seed) + 2,
    }
    for split_name in ("train", "val", "discovery"):
        indices = split_indices[split_name]
        split_targets = np.ascontiguousarray(targets[indices], dtype=np.float32)
        split_conditions = apply_official_i2d_mask(
            coordinates_2d[indices], seed=split_mask_seeds[split_name]
        )
        targets_filename = f"{split_name}_targets.npy"
        conditions_filename = f"{split_name}_conditions.npy"
        _atomic_npy(destination / targets_filename, split_targets)
        _atomic_npy(destination / conditions_filename, split_conditions)
        split_metadata[split_name] = {
            "targets_file": targets_filename,
            "conditions_file": conditions_filename,
            "num_samples": int(indices.size),
        }

    test_source_metadata: Optional[Dict[str, object]] = None
    if test_zip_path is not None:
        test_source = Path(test_zip_path)
        test_raw = load_h3wb_i2d_test_zip(test_source)
        test_targets = np.array(
            test_raw.targets, dtype=np.float32, copy=True, order="C"
        )
        pelvis_center_in_place(test_targets)
        test_targets -= target_mean[None, None, :]
        test_targets /= target_std[None, None, :]
        test_conditions = prepare_released_i2d_conditions(
            test_raw.coordinates_2d,
            train_axis_mean=condition_mean,
            train_axis_std=condition_std,
        )
        _atomic_npy(destination / "test_targets.npy", test_targets)
        _atomic_npy(destination / "test_conditions.npy", test_conditions)
        split_metadata["test"] = {
            "targets_file": "test_targets.npy",
            "conditions_file": "test_conditions.npy",
            "num_samples": int(test_targets.shape[0]),
        }
        test_source_metadata = {
            "file": str(test_source.resolve()),
            "sha256": _sha256(test_source),
            "schema": test_raw.source_schema,
            "scope": "formal_evaluation_only",
            "used_for_normalization": False,
            "used_for_discovery_or_orientation": False,
            "used_for_model_selection": False,
        }

    metadata: Dict[str, object] = {
        "dataset": "h3wb",
        "data_kind": "aligned_continuous",
        "format_version": FORMAT_VERSION,
        "condition_mode": "official_i2d",
        "num_variables": NUM_VARIABLES,
        "value_dim": VALUE_DIM,
        "condition_dim": CONDITION_DIM,
        "splits": split_metadata,
        "source": {
            "file": str(source.resolve()),
            "sha256": _sha256(source),
            "schema": raw.source_schema,
            "scope": (
                "official_train_only" if raw.source_schema == "official_nested_train_data"
                else "synthetic_fixture_only"
            ),
        },
        "split_policy": {
            "seed": int(split_seed),
            "val_fraction": float(val_fraction),
            "discovery_fraction": float(discovery_fraction),
            "grouped_camera_views": raw.source_schema
            == "official_nested_train_data",
            "split_unit": (
                "subject_action_all_cameras"
                if raw.source_schema == "official_nested_train_data"
                else "fixture_group_id"
            ),
        },
        "normalization": {
            "fit_split": "train",
            "pelvis_definition": "mean_of_coco_left_hip_11_and_right_hip_12",
            "left_hip_index": LEFT_HIP_INDEX,
            "right_hip_index": RIGHT_HIP_INDEX,
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "condition_frame": "absolute_image_xy_no_hidden_keypoint_centering",
            "condition_xy_mean": condition_mean.tolist(),
            "condition_xy_std": condition_std.tolist(),
        },
        "condition_channels": ["masked_2d_x", "masked_2d_y", "visibility"],
        "masking": {
            "protocol": "official_h3wb_i2d",
            "base_seed": int(mask_seed),
            "split_seeds": split_mask_seeds,
            "mixture": {
                "random_keypoints": {
                    "probability": 0.4,
                    "per_keypoint_mask_probability": 0.25,
                },
                "whole_face": 0.2,
                "whole_left_hand": 0.2,
                "whole_right_hand": 0.2,
            },
        },
    }
    if test_source_metadata is not None:
        metadata["test_source"] = test_source_metadata
        metadata["evaluation_boundary"] = {
            "selection_splits": ["train", "discovery", "val"],
            "formal_evaluation_split": "test",
            "test_access": "final_evaluation_only",
        }


    _atomic_json(destination / "metadata.json", metadata)
    return metadata


def prepare_dataset(source_path, output_dir, *, val_fraction=0.1,
                    discovery_fraction=0.1, split_seed=1337, mask_seed=7331,
                    test_zip_path=None, overwrite=False):
    """Publish a complete preparation only after every requested split succeeds."""
    destination = Path(output_dir)
    if _existing_outputs(destination) and not overwrite:
        raise FileExistsError(f"Prepared H3WB outputs exist in {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".h3wb-prepare-", dir=destination.parent) as staging:
        stage = Path(staging)
        metadata = _prepare_dataset(
            source_path, stage, val_fraction=val_fraction,
            discovery_fraction=discovery_fraction, split_seed=split_seed,
            mask_seed=mask_seed, test_zip_path=test_zip_path,
        )
        destination.mkdir(parents=True, exist_ok=True)
        for split in metadata["splits"].values():
            for key in ("targets_file", "conditions_file"):
                os.replace(stage / split[key], destination / split[key])
        os.replace(stage / "metadata.json", destination / "metadata.json")
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--discovery-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--mask-seed", type=int, default=7331)
    parser.add_argument(
        "--test-zip",
        type=Path,
        default=None,
        help=(
            "Optional official I2D test ZIP. It is transformed only after "
            "all train-only splits and statistics are frozen."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = prepare_dataset(
        args.source,
        args.output_dir,
        val_fraction=args.val_fraction,
        discovery_fraction=args.discovery_fraction,
        split_seed=args.split_seed,
        mask_seed=args.mask_seed,
        test_zip_path=args.test_zip,
        overwrite=args.overwrite,
    )
    print(
        "Prepared H3WB aligned-continuous records: "
        + ", ".join(
            f"{name}={info['num_samples']}"
            for name, info in metadata["splits"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FORMAT_VERSION", "prepare_dataset"]
