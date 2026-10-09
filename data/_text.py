"""Bounded-memory GPT-2 tokenization shared by the language datasets."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def get_encoder():
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("Install tiktoken before preparing language data.") from error
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as error:
        raise RuntimeError(
            "Cannot load GPT-2 BPE. On offline servers copy a populated "
            "TIKTOKEN_CACHE_DIR from an online machine; see data/README.md."
        ) from error


def file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def text_records(path: Path):
    """JSONL preserves exact record text; TXT preserves line terminators."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing source text file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if path.suffix.lower() == ".jsonl":
                try:
                    text = json.loads(line)["text"]
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: expected a JSON object with a text field"
                    ) from error
            else:
                text = line
            if not isinstance(text, str):
                raise TypeError(f"{path}:{line_number}: text must be a string")
            yield text


def write_streams(records, output_dir: Path, metadata: dict, *, encoder=None,
                  num_threads: int = 8, overwrite: bool = False,
                  split_names=("train", "val")) -> dict:
    """Write each EOT-terminated record once, without storing tokenized corpora."""
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")
    split_names = tuple(split_names)
    if split_names not in {("train",), ("train", "val")}:
        raise ValueError("Expected train-only or train/val output streams")
    output_dir = Path(output_dir)
    names = tuple(f"{split}.bin" for split in split_names)
    managed = ("train.bin", "val.bin", "metadata.json")
    if not overwrite and any((output_dir / name).exists() or (output_dir / name).is_symlink()
                             for name in managed):
        raise FileExistsError(f"Prepared outputs exist in {output_dir}; use --overwrite")
    stale_validation = output_dir / "val.bin" if split_names == ("train",) else None
    if stale_validation is not None and stale_validation.is_dir() and not stale_validation.is_symlink():
        raise ValueError("val.bin is a directory; choose a clean output directory")
    if (output_dir / "test.bin").exists():
        raise ValueError("Language preparation excludes test.bin; choose a clean output directory")
    encoder = get_encoder() if encoder is None else encoder
    if int(encoder.eot_token) != 50256:
        raise ValueError("Expected GPT-2 EOT token 50256")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {name: {"documents": 0, "tokens": 0} for name in split_names}
    hashes = {name: hashlib.sha256() for name in stats}
    source_hashes = {name: hashlib.sha256() for name in stats}
    iterator = iter(records)
    with TemporaryDirectory(prefix=".prepare-", dir=output_dir) as staging:
        stage = Path(staging)
        with ExitStack() as stack:
            handles = {name: stack.enter_context((stage / f"{name}.bin").open("wb"))
                       for name in stats}
            while batch := list(islice(iterator, 256)):
                texts = [text for _, text in batch]
                if any(not isinstance(text, str) for text in texts):
                    raise TypeError("Every source record must contain a string text field")
                encoded = encoder.encode_ordinary_batch(texts, num_threads=num_threads)
                if len(encoded) != len(batch):
                    raise RuntimeError("Tokenizer returned an unexpected record count")
                for (split, text), ids in zip(batch, encoded):
                    if split not in stats:
                        raise ValueError(f"Unsupported language split: {split}")
                    ids = [*ids, encoder.eot_token]
                    if any(token < 0 or token >= 50257 for token in ids):
                        raise ValueError("Tokenizer returned IDs outside the GPT-2 vocabulary")
                    payload = np.asarray(ids, dtype="<u2").tobytes()
                    handles[split].write(payload)
                    hashes[split].update(payload)
                    raw_text = text.encode("utf-8")
                    source_hashes[split].update(len(raw_text).to_bytes(8, "little"))
                    source_hashes[split].update(raw_text)
                    stats[split]["documents"] += 1
                    stats[split]["tokens"] += len(ids)
            for name, handle in handles.items():
                if stats[name]["documents"] == 0:
                    raise ValueError(f"The {name} split has no source records")
                handle.flush()
                os.fsync(handle.fileno())
        for name in stats:
            stats[name].update(file=f"{name}.bin", sha256=hashes[name].hexdigest(),
                               source_records_sha256=source_hashes[name].hexdigest())
        result = {**metadata, "format_version": 1, "tokenizer": "gpt2",
                  "vocab_size": 50257, "eot_token": 50256,
                  "dtype": "uint16", "byte_order": "little",
                  "record_separator": "one_eot_after_every_record_including_empty",
                  "splits": stats}
        (stage / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in names:
            os.replace(stage / name, output_dir / name)
        if stale_validation is not None and (stale_validation.exists() or stale_validation.is_symlink()):
            stale_validation.unlink()
            print(f"Removed obsolete prepared validation entry: {stale_validation}")
        os.replace(stage / "metadata.json", output_dir / "metadata.json")
    for name, info in stats.items():
        print(f"{name}: {info['documents']:,} documents, {info['tokens']:,} tokens")
    return result
