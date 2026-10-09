from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import torch


@dataclass
class ContinuousBatch:
    targets: torch.Tensor
    conditions: torch.Tensor

    def to(self, device: torch.device | str) -> "ContinuousBatch":
        return ContinuousBatch(
            targets=self.targets.to(device),
            conditions=self.conditions.to(device),
        )


class AlignedContinuousDataset:


    format_version = 1

    def __init__(
        self,
        data_dir: str | Path,
        variable_permutation_seed: int = -1,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing prepared metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain a JSON object")
        self.metadata: Dict[str, object] = metadata
        self._validate_metadata()

        self.dataset_name = str(metadata["dataset"])
        self.num_variables = int(metadata["num_variables"])
        self.value_dim = int(metadata["value_dim"])
        self.condition_dim = int(metadata["condition_dim"])
        self.splits: Mapping[str, Mapping[str, object]] = metadata["splits"]

        seed = int(variable_permutation_seed)
        if seed < -1:
            raise ValueError("variable_permutation_seed must be -1 (identity) or >= 0")
        if seed == -1:
            permutation = np.arange(self.num_variables, dtype=np.int64)
        else:
            permutation = np.random.default_rng(seed).permutation(
                self.num_variables
            ).astype(np.int64, copy=False)
        self.variable_permutation_seed = seed
        self.variable_permutation = torch.from_numpy(permutation.copy()).long()

        self._arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for split in self.splits:
            self._arrays[str(split)] = self._open_split(str(split))

    def _validate_metadata(self) -> None:
        required = {
            "dataset",
            "data_kind",
            "format_version",
            "num_variables",
            "value_dim",
            "condition_dim",
            "splits",
        }
        missing = sorted(required.difference(self.metadata))
        if missing:
            raise ValueError(f"metadata.json missing fields: {', '.join(missing)}")
        if self.metadata["data_kind"] != "aligned_continuous":
            raise ValueError("metadata data_kind must be 'aligned_continuous'")
        if int(self.metadata["format_version"]) != self.format_version:
            raise ValueError(
                f"unsupported format_version={self.metadata['format_version']}; "
                f"expected {self.format_version}"
            )
        for name in ("num_variables", "value_dim", "condition_dim"):
            if int(self.metadata[name]) <= 0:
                raise ValueError(f"metadata {name} must be positive")
        splits = self.metadata["splits"]
        if not isinstance(splits, dict) or not splits:
            raise ValueError("metadata splits must be a non-empty object")
        for split_name, split in splits.items():
            if not isinstance(split, dict):
                raise ValueError(f"metadata split {split_name!r} must be an object")
            needed = {"targets_file", "conditions_file", "num_samples"}
            split_missing = sorted(needed.difference(split))
            if split_missing:
                raise ValueError(
                    f"metadata split {split_name!r} missing fields: "
                    + ", ".join(split_missing)
                )
            if int(split["num_samples"]) <= 0:
                raise ValueError(f"split {split_name!r} must contain samples")

    def _safe_path(self, relative: object) -> Path:
        path = (self.data_dir / str(relative)).resolve()
        try:
            path.relative_to(self.data_dir)
        except ValueError as error:
            raise ValueError(f"prepared array escapes data directory: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared array: {path}")
        return path

    def _open_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        info = self.splits[split]
        targets = np.load(self._safe_path(info["targets_file"]), mmap_mode="r")
        conditions = np.load(
            self._safe_path(info["conditions_file"]), mmap_mode="r"
        )
        expected_targets = (
            int(info["num_samples"]),
            self.num_variables,
            self.value_dim,
        )
        expected_conditions = (
            int(info["num_samples"]),
            self.num_variables,
            self.condition_dim,
        )
        if targets.shape != expected_targets:
            raise ValueError(
                f"{split} targets shape {targets.shape} != {expected_targets}"
            )
        if conditions.shape != expected_conditions:
            raise ValueError(
                f"{split} conditions shape {conditions.shape} != {expected_conditions}"
            )
        if targets.dtype != np.float32 or conditions.dtype != np.float32:
            raise ValueError(f"{split} arrays must use float32")
        return targets, conditions

    @property
    def split_names(self) -> Tuple[str, ...]:
        return tuple(self._arrays)

    def num_samples(self, split: str) -> int:
        self._require_split(split)
        return int(self.splits[split]["num_samples"])

    def _require_split(self, split: str) -> None:
        if split not in self._arrays:
            available = ", ".join(self.split_names)
            raise KeyError(f"unknown split {split!r}; available: {available}")

    def _batch_from_indices(
        self,
        split: str,
        indices: np.ndarray,
        device: Optional[torch.device | str] = None,
    ) -> ContinuousBatch:
        self._require_split(split)
        targets_array, conditions_array = self._arrays[split]
        permutation = self.variable_permutation.numpy()


        targets = np.asarray(targets_array[indices][:, permutation], dtype=np.float32)
        conditions = np.asarray(
            conditions_array[indices][:, permutation], dtype=np.float32
        )
        batch = ContinuousBatch(
            targets=torch.from_numpy(np.ascontiguousarray(targets)),
            conditions=torch.from_numpy(np.ascontiguousarray(conditions)),
        )
        return batch if device is None else batch.to(device)

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device | str] = None,
    ) -> ContinuousBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self._require_split(split)
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.integers(
            0, self.num_samples(split), size=int(batch_size), dtype=np.int64
        )
        return self._batch_from_indices(split, indices, device=device)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> Iterator[ContinuousBatch]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self._require_split(split)
        indices = np.arange(self.num_samples(split), dtype=np.int64)
        if shuffle:
            np.random.default_rng(int(seed)).shuffle(indices)
        for start in range(0, len(indices), int(batch_size)):
            stop = min(start + int(batch_size), len(indices))
            if drop_last and stop - start < int(batch_size):
                break
            yield self._batch_from_indices(split, indices[start:stop], device=device)

    def validate_finite(self, split: str, chunk_size: int = 4096) -> None:


        self._require_split(split)
        targets, conditions = self._arrays[split]
        for start in range(0, self.num_samples(split), int(chunk_size)):
            stop = min(start + int(chunk_size), self.num_samples(split))
            if not np.isfinite(targets[start:stop]).all():
                raise ValueError(f"{split} targets contain non-finite values")
            if not np.isfinite(conditions[start:stop]).all():
                raise ValueError(f"{split} conditions contain non-finite values")


__all__ = ["AlignedContinuousDataset", "ContinuousBatch"]
