"""Download and verify the complete official UCI Census 1990 release."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import socket
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Sequence


DATASET_PAGE = "https://archive.ics.uci.edu/dataset/116/us+census+data+1990"
ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/116/"
    "us%2Bcensus%2Bdata%2B1990.zip"
)
ARCHIVE_FILENAME = "us_census_data_1990.zip"
ARCHIVE_BYTES = 168_626_076
ARCHIVE_SHA256 = "a9a1b84168456ab65b0f2fba7df2c1025aac15b61cd9667c6459f83dcf30b61a"

EXPECTED_MEMBERS: Mapping[str, int] = {
    "ReadMe.txt": 1_226,
    "USCensus1990-desc.html": 8_230,
    "USCensus1990-task.html": 1_401,
    "USCensus1990.attributes.txt": 814,
    "USCensus1990.data.txt": 361_344_227,
    "USCensus1990.html": 1_217,
    "USCensus1990.mapping.sql": 8_381,
    "USCensus1990.readme.txt": 8_697,
    "USCensus1990.task.txt": 975,
    "USCensus1990raw-desc.html": 3_273,
    "USCensus1990raw.attributes.txt": 84_401,
    "USCensus1990raw.coding.htm": 171_125,
    "USCensus1990raw.data.txt": 862_858_036,
    "USCensus1990raw.html": 1_341,
    "USCensus1990raw.readme.txt": 2_841,
}

EXPECTED_DATA_SHA256: Mapping[str, str] = {
    "USCensus1990.data.txt": (
        "38e99f50855ddc03167b715b36519ef40b77295ae4e85dfef0c6429ac79ed799"
    ),
    "USCensus1990raw.data.txt": (
        "6ac3f3986c6586ddc7c08fc842021c617fd085c106f38527993299b3e3d7386b"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Census archive: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != ARCHIVE_BYTES:
        raise ValueError(
            f"Byte-length mismatch for {path}: "
            f"expected {ARCHIVE_BYTES}, got {actual_bytes}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != ARCHIVE_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: "
            f"expected {ARCHIVE_SHA256}, got {actual_sha256}"
        )

    with zipfile.ZipFile(path) as archive:
        actual_members = {
            member.filename: member.file_size
            for member in archive.infolist()
            if not member.is_dir()
        }
        if actual_members != EXPECTED_MEMBERS:
            raise ValueError(
                "ZIP member manifest differs from the pinned UCI release"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC check failed for {bad_member}")


def verify_extracted(directory: Path) -> None:
    for filename, expected_bytes in EXPECTED_MEMBERS.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing extracted Census file: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Byte-length mismatch for {path}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )

    for filename, expected_sha256 in EXPECTED_DATA_SHA256.items():
        path = directory / filename
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )


def _download_once(destination: Path, timeout: int) -> None:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "OrderGPT/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())


def download_archive(
    destination: Path,
    *,
    force: bool,
    retries: int,
    timeout: int,
) -> None:
    if destination.exists() and not force:
        verify_archive(destination)
        print(f"Pinned archive already exists and is valid: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    retryable_errors = (
        http.client.HTTPException,
        OSError,
        socket.timeout,
        urllib.error.URLError,
        ValueError,
        zipfile.BadZipFile,
    )

    for attempt in range(1, retries + 1):
        if temporary.exists():
            temporary.unlink()
        try:
            print(f"Downloading Census archive (attempt {attempt}/{retries})")
            _download_once(temporary, timeout)
            verify_archive(temporary)
            os.replace(temporary, destination)
            print(f"Downloaded and verified: {destination}")
            return
        except retryable_errors as error:
            if temporary.exists():
                temporary.unlink()
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download the pinned Census archive after "
                    f"{retries} attempts"
                ) from error
            print(f"Attempt {attempt} failed: {error}; retrying")
            time.sleep(min(attempt, 5))


def extract_archive(archive_path: Path, output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and not force:
        try:
            verify_extracted(output_dir)
        except (FileNotFoundError, ValueError):
            pass
        else:
            print(f"Pinned release already extracted and valid: {output_dir}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if resolved_output not in target.parents and target != resolved_output:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        archive.extractall(output_dir)

    verify_extracted(output_dir)
    print(f"Extracted and verified complete UCI release: {output_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/census/raw"),
        help="Directory containing the archive and official/ extraction",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")

    archive_path = args.output_dir / ARCHIVE_FILENAME
    extracted_dir = args.output_dir / "official"

    if args.verify_only:
        verify_archive(archive_path)
        print(f"Archive verification passed: {archive_path}")
        if not args.no_extract:
            verify_extracted(extracted_dir)
            print(f"Extracted release verification passed: {extracted_dir}")
        return 0

    download_archive(
        archive_path,
        force=bool(args.force),
        retries=int(args.retries),
        timeout=int(args.timeout),
    )
    if not args.no_extract:
        extract_archive(archive_path, extracted_dir, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_BYTES",
    "ARCHIVE_FILENAME",
    "ARCHIVE_SHA256",
    "ARCHIVE_URL",
    "DATASET_PAGE",
    "EXPECTED_DATA_SHA256",
    "EXPECTED_MEMBERS",
    "download_archive",
    "extract_archive",
    "file_sha256",
    "verify_archive",
    "verify_extracted",
]
