"""Prepare and hash-verify the pinned UCI Census table with the Appendix C split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SOURCE_BYTES = 361_344_227
SOURCE_SHA256 = "38e99f50855ddc03167b715b36519ef40b77295ae4e85dfef0c6429ac79ed799"
ARCHIVE_SHA256 = "a9a1b84168456ab65b0f2fba7df2c1025aac15b61cd9667c6459f83dcf30b61a"
NUM_SOURCE_ROWS = 2_458_285
NUM_VARIABLES = 68
PROTOCOL = "census68_dense_sorted_unique_contiguous_90_5_5_v1"

CANONICAL_VARIABLE_NAMES: Tuple[str, ...] = (
    "dAge",
    "dAncstry1",
    "dAncstry2",
    "iAvail",
    "iCitizen",
    "iClass",
    "dDepart",
    "iDisabl1",
    "iDisabl2",
    "iEnglish",
    "iFeb55",
    "iFertil",
    "dHispanic",
    "dHour89",
    "dHours",
    "iImmigr",
    "dIncome1",
    "dIncome2",
    "dIncome3",
    "dIncome4",
    "dIncome5",
    "dIncome6",
    "dIncome7",
    "dIncome8",
    "dIndustry",
    "iKorean",
    "iLang1",
    "iLooking",
    "iMarital",
    "iMay75880",
    "iMeans",
    "iMilitary",
    "iMobility",
    "iMobillim",
    "dOccup",
    "iOthrserv",
    "iPerscare",
    "dPOB",
    "dPoverty",
    "dPwgt1",
    "iRagechld",
    "dRearning",
    "iRelat1",
    "iRelat2",
    "iRemplpar",
    "iRiders",
    "iRlabor",
    "iRownchld",
    "dRpincome",
    "iRPOB",
    "iRrelchld",
    "iRspouse",
    "iRvetserv",
    "iSchool",
    "iSept80",
    "iSex",
    "iSubfam1",
    "iSubfam2",
    "iTmpabsnt",
    "dTravtime",
    "iVietnam",
    "dWeek89",
    "iWork89",
    "iWorklwk",
    "iWWII",
    "iYearsch",
    "iYearwrk",
    "dYrsserv",
)
EXPECTED_HEADER: Tuple[str, ...] = ("caseid",) + CANONICAL_VARIABLE_NAMES

EXPECTED_CARDINALITIES: Tuple[int, ...] = (
    8, 12, 12, 5, 5, 10, 6, 3, 3, 5, 2, 14, 10, 6, 6, 11, 5,
    2, 2, 2, 2, 2, 2, 2, 13, 2, 3, 3, 5, 2, 13, 5, 3, 3, 9, 2,
    3, 7, 3, 4, 5, 6, 14, 10, 16, 9, 7, 2, 6, 14, 2, 7, 12, 4,
    2, 2, 4, 4, 4, 7, 2, 3, 3, 3, 2, 18, 8, 3,
)

SPLIT_SPECS: Tuple[Tuple[str, int, int], ...] = (
    ("train", 0, 2_212_456),
    ("val", 2_212_456, 2_335_370),
    ("test", 2_335_370, 2_458_285),
)


EXPECTED_RECORD_SHA256: Mapping[str, str] = {
    "train": "681378537b613752930010ee39a3e4fcf5477c0f2e88d62ebc6d577e6f86a2b4",
    "val": "bccbe549214027b23ebe1556da17eaa4cba79265f4a22bee2717ed6a2418d2b7",
    "test": "354fc9d96c4af92643c761c6f2c8dd93e2c2f9632ecd131f22821ebe8976983b",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed Census table: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != SOURCE_BYTES:
        raise ValueError(
            f"Source byte-length mismatch: expected {SOURCE_BYTES}, "
            f"got {actual_bytes}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {SOURCE_SHA256}, "
            f"got {actual_sha256}"
        )


def _pandas():
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Preparing Census records requires pandas; install the project's "
            "data-processing dependencies first"
        ) from error
    return pd


def _iter_source_chunks(path: Path, chunk_size: int):
    pd = _pandas()
    for chunk in pd.read_csv(path, chunksize=chunk_size, dtype="int32"):
        observed_header = tuple(str(value) for value in chunk.columns)
        if observed_header != EXPECTED_HEADER:
            raise ValueError(
                "Census header differs from the pinned 69-field schema"
            )
        if int(chunk.shape[1]) != NUM_VARIABLES + 1:
            raise ValueError(
                f"Expected 69 source fields, got {int(chunk.shape[1])}"
            )
        yield chunk


def discover_state_values(
    source_path: Path,
    chunk_size: int,
) -> Tuple[List[List[int]], int]:
    observed = [set() for _ in range(NUM_VARIABLES)]
    num_rows = 0
    for chunk in _iter_source_chunks(source_path, chunk_size):
        features = chunk.iloc[:, 1:]
        for variable_id in range(NUM_VARIABLES):
            observed[variable_id].update(
                int(value) for value in features.iloc[:, variable_id].unique()
            )
        num_rows += int(len(chunk))

    if num_rows != NUM_SOURCE_ROWS:
        raise ValueError(
            f"Expected {NUM_SOURCE_ROWS} source rows, observed {num_rows}"
        )
    state_values = [sorted(values) for values in observed]
    cardinalities = tuple(len(values) for values in state_values)
    if cardinalities != EXPECTED_CARDINALITIES:
        raise ValueError(
            "Observed Census cardinalities differ from the pinned 68-column "
            f"schema: {cardinalities}"
        )
    if any(not values for values in state_values):
        raise ValueError("Every Census variable must have at least one state")
    if any(len(values) > 256 for values in state_values):
        raise ValueError("Dense Census state ids do not fit uint8")
    return state_values, num_rows


def _dense_lookup_tables(state_values: Sequence[Sequence[int]]):
    import numpy as np

    tables = []
    offsets = []
    for values in state_values:
        minimum = int(values[0])
        maximum = int(values[-1])
        table = np.full(maximum - minimum + 1, 256, dtype=np.int16)
        for state_id, source_value in enumerate(values):
            table[int(source_value) - minimum] = int(state_id)
        tables.append(table)
        offsets.append(minimum)
    return tables, offsets


def _encode_features(features, lookup_tables, offsets):
    import numpy as np

    raw = features.to_numpy(dtype=np.int32, copy=False)
    encoded = np.empty((int(raw.shape[0]), NUM_VARIABLES), dtype=np.uint8)
    for variable_id, (table, offset) in enumerate(
        zip(lookup_tables, offsets)
    ):
        indices = raw[:, variable_id].astype(np.int64, copy=False) - int(offset)
        if bool((indices < 0).any()) or bool((indices >= len(table)).any()):
            raise ValueError(
                f"Out-of-vocabulary value in variable {variable_id}"
            )
        dense = table[indices]
        if bool((dense >= 256).any()):
            raise ValueError(
                f"Unknown sparse state in variable {variable_id}"
            )
        encoded[:, variable_id] = dense.astype(np.uint8, copy=False)
    return encoded


def _write_records(
    source_path: Path,
    temporary_dir: Path,
    state_values: Sequence[Sequence[int]],
    chunk_size: int,
) -> Dict[str, Dict[str, object]]:
    temporary_dir.mkdir(parents=True, exist_ok=False)
    handles = {
        split: (temporary_dir / f"{split}.bin").open("wb")
        for split, _, _ in SPLIT_SPECS
    }
    digests = {split: hashlib.sha256() for split, _, _ in SPLIT_SPECS}
    records_written = {split: 0 for split, _, _ in SPLIT_SPECS}
    lookup_tables, offsets = _dense_lookup_tables(state_values)
    source_row = 0

    try:
        for chunk in _iter_source_chunks(source_path, chunk_size):
            encoded = _encode_features(
                chunk.iloc[:, 1:],
                lookup_tables,
                offsets,
            )
            chunk_start = source_row
            chunk_stop = source_row + int(encoded.shape[0])
            for split, split_start, split_stop in SPLIT_SPECS:
                overlap_start = max(chunk_start, split_start)
                overlap_stop = min(chunk_stop, split_stop)
                if overlap_start >= overlap_stop:
                    continue
                local_start = overlap_start - chunk_start
                local_stop = overlap_stop - chunk_start
                payload = encoded[local_start:local_stop].tobytes(order="C")
                handles[split].write(payload)
                digests[split].update(payload)
                records_written[split] += overlap_stop - overlap_start
            source_row = chunk_stop
    finally:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    if source_row != NUM_SOURCE_ROWS:
        raise ValueError(
            f"Expected to encode {NUM_SOURCE_ROWS} rows, encoded {source_row}"
        )

    split_metadata: Dict[str, Dict[str, object]] = {}
    for split, start, stop in SPLIT_SPECS:
        expected_records = stop - start
        if records_written[split] != expected_records:
            raise ValueError(
                f"Split {split} wrote {records_written[split]} rows; "
                f"expected {expected_records}"
            )
        path = temporary_dir / f"{split}.bin"
        expected_bytes = expected_records * NUM_VARIABLES
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Split {split} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes}"
            )
        split_metadata[split] = {
            "file": f"{split}.bin",
            "num_records": expected_records,
            "row_start_in_official_table": start,
            "row_stop_in_official_table": stop,
            "sha256": digests[split].hexdigest(),
        }
    return split_metadata


def _metadata_payload(
    source_path: Path,
    state_values: Sequence[Sequence[int]],
    splits: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    return {
        "dataset": "census",
        "format_version": 1,
        "data_record_mode": "fixed",
        "record_dtype": "uint8",
        "record_layout": "row_major_canonical_variables",
        "num_variables": NUM_VARIABLES,
        "canonical_variable_names": list(CANONICAL_VARIABLE_NAMES),
        "cardinalities": list(EXPECTED_CARDINALITIES),
        "state_names": [
            [str(value) for value in values] for values in state_values
        ],
        "state_values": [
            [int(value) for value in values] for values in state_values
        ],
        "source": {
            "file": str(source_path),
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "archive_sha256": ARCHIVE_SHA256,
            "num_rows": NUM_SOURCE_ROWS,
            "num_fields_including_caseid": NUM_VARIABLES + 1,
            "ignored_columns": ["caseid"],
        },
        "preparation": {
            "protocol": PROTOCOL,
            "feature_selection": "all_68_non_caseid_features",
            "categorical_encoding": (
                "per_column_sorted_unique_source_integer_to_dense_zero_based_id"
            ),
            "state_vocabulary_scope": "complete_pinned_table_before_split",
            "split_basis": (
                "contiguous_rows_of_official_table_randomized_by_dataset_creators"
            ),
            "split_fractions": {
                "train": "90%",
                "val": "5%",
                "test": "5%",
            },
        },
        "splits": {key: dict(value) for key, value in splits.items()},
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _require_equal(
    metadata: Mapping[str, object],
    key: str,
    expected: object,
) -> None:
    observed = metadata.get(key)
    if observed != expected:
        raise ValueError(
            f"metadata {key!r} mismatch: expected {expected!r}, "
            f"got {observed!r}"
        )


def verify_records(output_dir: Path) -> Dict[str, object]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing prepared metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")

    required_scalars = {
        "dataset": "census",
        "format_version": 1,
        "data_record_mode": "fixed",
        "record_dtype": "uint8",
        "record_layout": "row_major_canonical_variables",
        "num_variables": NUM_VARIABLES,
        "canonical_variable_names": list(CANONICAL_VARIABLE_NAMES),
        "cardinalities": list(EXPECTED_CARDINALITIES),
    }
    for key, expected in required_scalars.items():
        _require_equal(metadata, key, expected)

    source = metadata.get("source")
    if not isinstance(source, dict):
        raise ValueError("metadata source must be an object")
    if source.get("sha256") != SOURCE_SHA256:
        raise ValueError("metadata source SHA-256 does not match the pinned table")
    if source.get("ignored_columns") != ["caseid"]:
        raise ValueError("metadata must record caseid as the only ignored column")

    preparation = metadata.get("preparation")
    if not isinstance(preparation, dict):
        raise ValueError("metadata preparation must be an object")
    if preparation.get("protocol") != PROTOCOL:
        raise ValueError("metadata preparation protocol mismatch")

    state_values = metadata.get("state_values")
    state_names = metadata.get("state_names")
    if not isinstance(state_values, list) or len(state_values) != NUM_VARIABLES:
        raise ValueError("metadata state_values must have 68 entries")
    if not isinstance(state_names, list) or len(state_names) != NUM_VARIABLES:
        raise ValueError("metadata state_names must have 68 entries")
    for variable_id, (values, names, cardinality) in enumerate(
        zip(state_values, state_names, EXPECTED_CARDINALITIES)
    ):
        if not isinstance(values, list) or len(values) != cardinality:
            raise ValueError(
                f"metadata state_values[{variable_id}] has wrong cardinality"
            )
        normalized = [int(value) for value in values]
        if normalized != sorted(set(normalized)):
            raise ValueError(
                f"metadata state_values[{variable_id}] is not sorted unique"
            )
        if names != [str(value) for value in normalized]:
            raise ValueError(
                f"metadata state_names[{variable_id}] disagrees with state_values"
            )

    splits = metadata.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("metadata splits must be an object")
    for split, start, stop in SPLIT_SPECS:
        info = splits.get(split)
        if not isinstance(info, dict):
            raise ValueError(f"metadata split {split!r} is missing")
        expected_records = stop - start
        expected_file = f"{split}.bin"
        if info.get("file") != expected_file:
            raise ValueError(f"metadata split {split!r} file mismatch")
        if int(info.get("num_records", -1)) != expected_records:
            raise ValueError(f"metadata split {split!r} record count mismatch")
        if int(info.get("row_start_in_official_table", -1)) != start:
            raise ValueError(f"metadata split {split!r} row start mismatch")
        if int(info.get("row_stop_in_official_table", -1)) != stop:
            raise ValueError(f"metadata split {split!r} row stop mismatch")

        path = output_dir / expected_file
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared split: {path}")
        expected_bytes = expected_records * NUM_VARIABLES
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Split {split} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes}"
            )
        actual_sha256 = file_sha256(path)
        if info.get("sha256") != actual_sha256:
            raise ValueError(f"Split {split} does not match its metadata SHA-256")
        pinned_sha256 = EXPECTED_RECORD_SHA256.get(split)
        if pinned_sha256 is not None and actual_sha256 != pinned_sha256:
            raise ValueError(
                f"Split {split} SHA-256 differs from the pinned preparation"
            )
    return metadata


def prepare_records(
    source_path: Path,
    output_dir: Path,
    *,
    chunk_size: int,
    force: bool,
) -> Dict[str, object]:
    verify_source(source_path)
    known_outputs = [
        output_dir / "metadata.json",
        *(output_dir / f"{split}.bin" for split, _, _ in SPLIT_SPECS),
    ]
    if any(path.exists() for path in known_outputs):
        try:
            metadata = verify_records(output_dir)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            if not force:
                raise RuntimeError(
                    "Prepared Census outputs exist but are incomplete or invalid; "
                    "inspect them or rerun with --force"
                ) from error
        else:
            if not force:
                print(f"Prepared Census records already valid: {output_dir}")
                return metadata

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / (
        f".{output_dir.name}.prepare-tmp-{os.getpid()}"
    )
    if temporary_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse preparation temporary directory: {temporary_dir}"
        )

    print("Pass 1/2: discovering deterministic per-column state vocabularies")
    state_values, _ = discover_state_values(source_path, chunk_size)
    print("Pass 2/2: encoding and writing fixed contiguous splits")
    try:
        splits = _write_records(
            source_path,
            temporary_dir,
            state_values,
            chunk_size,
        )
        metadata = _metadata_payload(source_path, state_values, splits)
        _write_json(temporary_dir / "metadata.json", metadata)
        verify_records(temporary_dir)


        for split, _, _ in SPLIT_SPECS:
            os.replace(
                temporary_dir / f"{split}.bin",
                output_dir / f"{split}.bin",
            )
        os.replace(
            temporary_dir / "metadata.json",
            output_dir / "metadata.json",
        )
        temporary_dir.rmdir()
    except Exception:
        for filename in (
            "metadata.json",
            *(f"{split}.bin" for split, _, _ in SPLIT_SPECS),
        ):
            path = temporary_dir / filename
            if path.is_file():
                path.unlink()
        if temporary_dir.is_dir():
            temporary_dir.rmdir()
        raise

    verified = verify_records(output_dir)
    print(f"Prepared and verified Census records: {output_dir}")
    return verified


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/census/raw/official/USCensus1990.data.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/census/records"),
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    if args.verify_only:
        verify_source(args.source)
        metadata = verify_records(args.output_dir)
        print(
            "Census verification passed: "
            f"{metadata['num_variables']} variables, "
            f"{sum(int(info['num_records']) for info in metadata['splits'].values())} "
            "records"
        )
        return 0

    prepare_records(
        args.source,
        args.output_dir,
        chunk_size=int(args.chunk_size),
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_VARIABLE_NAMES",
    "EXPECTED_CARDINALITIES",
    "EXPECTED_RECORD_SHA256",
    "NUM_SOURCE_ROWS",
    "NUM_VARIABLES",
    "PROTOCOL",
    "SPLIT_SPECS",
    "discover_state_values",
    "file_sha256",
    "prepare_records",
    "verify_records",
    "verify_source",
]
