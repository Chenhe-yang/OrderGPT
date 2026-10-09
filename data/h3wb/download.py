"""Download and hash-verify the official H3WB training and optional I2D test archives."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence
import urllib.request
import zipfile


OFFICIAL_FILE_ID = "1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC"
OFFICIAL_SHA256 = "d423224a9e3c47aa8320687a8fcb58e732aa8d3aabe589e01ce390030d123b1d"
OFFICIAL_VIEW_URL = (
    "https://drive.google.com/file/d/"
    + OFFICIAL_FILE_ID
    + "/view?usp=share_link"
)
OFFICIAL_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id="
    + OFFICIAL_FILE_ID
    + "&export=download&confirm=t"
)
DEFAULT_OUTPUT = Path("data/h3wb/raw/h3wb_train.npz")
REQUIRED_MEMBERS = {"metadata.npy", "train_data.npy"}

OFFICIAL_I2D_TEST_FILE_ID = "1GBOCjzIW0GLHHLHhGmUcB32Ke8PVjDXH"
OFFICIAL_I2D_TEST_SHA256 = "e9e69e9c9315196be561f74df7976385c5bd7eb04fe6f4433a77bd8572576eb7"
OFFICIAL_I2D_TEST_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id="
    + OFFICIAL_I2D_TEST_FILE_ID
    + "&export=download&confirm=t"
)
DEFAULT_I2D_TEST_OUTPUT = Path("data/h3wb/raw/h3wb_i2d_test.zip")

def _looks_like_html(prefix: bytes, headers: Mapping[str, str]) -> bool:
    content_type = str(headers.get("Content-Type", "")).lower()
    stripped = prefix.lstrip().lower()
    return (
        "text/html" in content_type
        or stripped.startswith(b"<!doctype html")
        or stripped.startswith(b"<html")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_npz(
    path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
) -> None:


    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"H3WB archive does not exist: {artifact}")
    with artifact.open("rb") as handle:
        prefix = handle.read(512)
    if _looks_like_html(prefix, {}):
        raise ValueError(
            f"H3WB download is HTML rather than an NPZ archive: {artifact}"
        )
    try:
        with zipfile.ZipFile(artifact, mode="r") as archive:
            members = set(archive.namelist())
            missing = sorted(REQUIRED_MEMBERS.difference(members))
            if missing:
                raise ValueError(
                    "H3WB NPZ is missing official members: " + ", ".join(missing)
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"H3WB NPZ failed CRC validation at member {bad_member!r}"
                )
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"H3WB artifact is not a complete ZIP/NPZ archive: {artifact}"
        ) from error
    if expected_sha256 is not None:
        observed = _sha256(artifact)
        if observed != str(expected_sha256):
            raise ValueError(
                "H3WB NPZ SHA-256 mismatch: expected "
                f"{expected_sha256}, got {observed}"
            )


def validate_i2d_test_zip(
    path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
) -> None:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"H3WB I2D test archive does not exist: {artifact}")
    try:
        with zipfile.ZipFile(artifact, mode="r") as archive:
            if archive.namelist() != ["data.npy"]:
                raise ValueError("H3WB I2D test ZIP must contain only data.npy")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"H3WB I2D test ZIP failed CRC validation at {bad_member!r}"
                )
    except zipfile.BadZipFile as error:
        raise ValueError(f"invalid H3WB I2D test ZIP: {artifact}") from error
    if expected_sha256 is not None:
        observed = _sha256(artifact)
        if observed != str(expected_sha256):
            raise ValueError(
                "H3WB I2D test SHA-256 mismatch: expected "
                f"{expected_sha256}, got {observed}"
            )


def download(
    destination: str | Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
    url: str = OFFICIAL_DOWNLOAD_URL,
    timeout: int = 120,
) -> Path:
    output = Path(destination)
    if output.exists() and not force:
        validate_official_npz(output, expected_sha256=OFFICIAL_SHA256)
        print(f"Official H3WB archive already exists and is valid: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OrderGPT/1.0",
            "Accept": "application/octet-stream,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=int(timeout)) as response:
            first = response.read(64 * 1024)
            if _looks_like_html(first, response.headers):
                raise ValueError(
                    "Google Drive returned HTML instead of h3wb_train.npz; "
                    "check public access or quota and retry"
                )
            with temporary.open("wb") as handle:
                handle.write(first)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        validate_official_npz(temporary, expected_sha256=OFFICIAL_SHA256)
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Downloaded and validated official H3WB archive: {output} "
        f"(source: {OFFICIAL_VIEW_URL})"
    )
    return output


def download_i2d_test(
    destination: str | Path = DEFAULT_I2D_TEST_OUTPUT,
    *,
    force: bool = False,
    timeout: int = 120,
) -> Path:
    output = Path(destination)
    if output.exists() and not force:
        validate_i2d_test_zip(
            output, expected_sha256=OFFICIAL_I2D_TEST_SHA256
        )
        print(f"Official H3WB I2D test archive already exists and is valid: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        OFFICIAL_I2D_TEST_DOWNLOAD_URL,
        headers={"User-Agent": "OrderGPT/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=int(timeout)) as response:
            first = response.read(64 * 1024)
            if _looks_like_html(first, response.headers):
                raise ValueError("Google Drive returned HTML instead of the H3WB I2D test ZIP")
            with temporary.open("wb") as handle:
                handle.write(first)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        validate_i2d_test_zip(
            temporary, expected_sha256=OFFICIAL_I2D_TEST_SHA256
        )
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Downloaded and validated official H3WB I2D test archive: {output}")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--include-i2d-test", action="store_true")
    parser.add_argument(
        "--test-output", type=Path, default=DEFAULT_I2D_TEST_OUTPUT
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing file without making a network request.",
    )
    args = parser.parse_args(argv)
    if args.verify_only:
        validate_official_npz(args.output, expected_sha256=OFFICIAL_SHA256)
        print(f"Official H3WB NPZ validation passed: {args.output}")
        if args.include_i2d_test:
            validate_i2d_test_zip(
                args.test_output, expected_sha256=OFFICIAL_I2D_TEST_SHA256
            )
            print(f"Official H3WB I2D test validation passed: {args.test_output}")
    else:
        download(args.output, force=args.force, timeout=args.timeout)
        if args.include_i2d_test:
            download_i2d_test(
                args.test_output, force=args.force, timeout=args.timeout
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_I2D_TEST_OUTPUT",
    "OFFICIAL_DOWNLOAD_URL",
    "OFFICIAL_FILE_ID",
    "OFFICIAL_SHA256",
    "OFFICIAL_VIEW_URL",
    "download",
    "download_i2d_test",
    "validate_i2d_test_zip",
    "validate_official_npz",
]
