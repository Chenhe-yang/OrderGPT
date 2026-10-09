from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import torch


@dataclass
class TabularBatch:


    targets: torch.Tensor
    conditions: torch.Tensor

    def to(self, device: torch.device | str) -> "TabularBatch":
        return TabularBatch(
            targets=self.targets.to(device),
            conditions=self.conditions.to(device),
        )


class AlignedTabularDataset:


    format_version = 1

    def __init__(
        self,
        data_dir: str | Path,
        variable_permutation_seed: int = 0,
        *,
        validate_values: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing fixed-record metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain a JSON object")
        self.metadata: Dict[str, object] = metadata
        self._validate_metadata()

        self.dataset_name = str(metadata["dataset"])
        self.num_variables = int(metadata["num_variables"])
        self.value_dim = 1
        self.condition_dim = 0
        self.splits: Mapping[str, Mapping[str, object]] = metadata["splits"]

        seed = int(variable_permutation_seed)
        if seed < -1:
            raise ValueError(
                "categorical variable_permutation_seed must be -1 or >= 0; "
                "-1 preserves canonical column order and seed 0 is a real "
                "shuffled permutation"
            )
        if seed == -1:
            permutation = np.arange(self.num_variables, dtype=np.int64)
        else:
            permutation = np.random.default_rng(seed).permutation(
                self.num_variables
            ).astype(np.int64, copy=False)
        inverse = np.empty(self.num_variables, dtype=np.int64)
        inverse[permutation] = np.arange(self.num_variables, dtype=np.int64)
        canonical_cardinalities = np.asarray(metadata["cardinalities"], dtype=np.int64)

        self.variable_permutation_seed = seed
        self._variable_permutation_np = permutation
        self.variable_permutation = torch.from_numpy(permutation.copy()).long()
        self.current_to_canonical = self.variable_permutation
        self.canonical_to_current = torch.from_numpy(inverse.copy()).long()
        self.canonical_cardinalities = torch.from_numpy(
            canonical_cardinalities.copy()
        ).long()
        self.cardinalities = torch.from_numpy(
            canonical_cardinalities[permutation].copy()
        ).long()
        self.max_states = int(self.cardinalities.max().item())

        self._records: Dict[str, np.memmap] = {}
        for split in self.splits:
            self._records[str(split)] = self._open_split(str(split))
        if validate_values:
            for split in self.split_names:
                self.validate_values(split)

    def _validate_metadata(self) -> None:
        required = {
            "dataset",
            "format_version",
            "data_record_mode",
            "record_dtype",
            "record_layout",
            "num_variables",
            "cardinalities",
            "splits",
        }
        missing = sorted(required.difference(self.metadata))
        if missing:
            raise ValueError(f"metadata.json missing fields: {', '.join(missing)}")
        if int(self.metadata["format_version"]) != self.format_version:
            raise ValueError(
                f"unsupported format_version={self.metadata['format_version']}; "
                f"expected {self.format_version}"
            )
        if self.metadata["data_record_mode"] != "fixed":
            raise ValueError("metadata data_record_mode must be 'fixed'")
        if self.metadata["record_dtype"] != "uint8":
            raise ValueError("fixed records must use uint8")
        if self.metadata["record_layout"] != "row_major_canonical_variables":
            raise ValueError(
                "fixed records must use row_major_canonical_variables layout"
            )
        width = int(self.metadata["num_variables"])
        if width <= 0:
            raise ValueError("metadata num_variables must be positive")
        cardinalities = np.asarray(self.metadata["cardinalities"], dtype=np.int64)
        if cardinalities.shape != (width,):
            raise ValueError("metadata cardinalities must have num_variables entries")
        if np.any(cardinalities <= 0) or np.any(cardinalities > 256):
            raise ValueError("uint8 cardinalities must be in [1, 256]")
        splits = self.metadata["splits"]
        if not isinstance(splits, dict) or not splits:
            raise ValueError("metadata splits must be a non-empty object")
        for split_name, split in splits.items():
            if not isinstance(split, dict):
                raise ValueError(f"metadata split {split_name!r} must be an object")
            missing_split = sorted({"file", "num_records"}.difference(split))
            if missing_split:
                raise ValueError(
                    f"metadata split {split_name!r} missing fields: "
                    + ", ".join(missing_split)
                )
            if int(split["num_records"]) <= 0:
                raise ValueError(f"split {split_name!r} must contain records")

    def _safe_path(self, relative: object) -> Path:
        path = (self.data_dir / str(relative)).resolve()
        try:
            path.relative_to(self.data_dir)
        except ValueError as error:
            raise ValueError(f"prepared record escapes data directory: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared record file: {path}")
        return path

    def _open_split(self, split: str) -> np.memmap:
        info = self.splits[split]
        path = self._safe_path(info["file"])
        count = int(info["num_records"])
        expected_bytes = count * self.num_variables
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"split {split!r} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes}"
            )
        return np.memmap(
            path,
            dtype=np.uint8,
            mode="r",
            shape=(count, self.num_variables),
        )

    @property
    def split_names(self) -> Tuple[str, ...]:
        return tuple(self._records)

    def num_samples(self, split: str) -> int:
        self._require_split(split)
        return int(self.splits[split]["num_records"])

    def _require_split(self, split: str) -> None:
        if split not in self._records:
            raise KeyError(
                f"unknown split {split!r}; available: {', '.join(self.split_names)}"
            )

    def _batch_from_indices(
        self,
        split: str,
        indices: np.ndarray,
        device: Optional[torch.device | str] = None,
    ) -> TabularBatch:
        self._require_split(split)
        canonical = np.asarray(self._records[split][indices], dtype=np.uint8)
        current = canonical[:, self._variable_permutation_np]
        targets = torch.from_numpy(np.ascontiguousarray(current)).long()
        conditions = torch.empty((len(indices), 0), dtype=torch.float32)
        batch = TabularBatch(targets=targets, conditions=conditions)
        return batch if device is None else batch.to(device)

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device | str] = None,
    ) -> TabularBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self._require_split(split)
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.integers(
            0,
            self.num_samples(split),
            size=int(batch_size),
            dtype=np.int64,
        )
        return self._batch_from_indices(split, indices, device=device)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        device: Optional[torch.device | str] = None,
    ) -> Iterator[TabularBatch]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self._require_split(split)
        for start in range(0, self.num_samples(split), int(batch_size)):
            stop = min(start + int(batch_size), self.num_samples(split))
            indices = np.arange(start, stop, dtype=np.int64)
            yield self._batch_from_indices(split, indices, device=device)

    def validate_values(self, split: str, chunk_size: int = 100_000) -> None:


        self._require_split(split)
        cardinalities = self.canonical_cardinalities.numpy()
        records = self._records[split]
        for start in range(0, self.num_samples(split), int(chunk_size)):
            stop = min(start + int(chunk_size), self.num_samples(split))
            chunk = np.asarray(records[start:stop])
            invalid = chunk >= cardinalities[None, :]
            if bool(np.any(invalid)):
                row_offset, variable_id = np.argwhere(invalid)[0]
                row = start + int(row_offset)
                value = int(chunk[int(row_offset), int(variable_id)])
                raise ValueError(
                    f"invalid state {value} at {split} row {row}, canonical "
                    f"variable {int(variable_id)}"
                )


__all__ = ["AlignedTabularDataset", "TabularBatch"]
