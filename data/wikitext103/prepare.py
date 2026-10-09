"""Prepare raw WikiText103 training and official validation token streams."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data._text import file_info, text_records, write_streams


def prepare_dataset(train_records, validation_records, output_dir, *, source=None,
                    encoder=None, num_threads=8, overwrite=False):
    def records():
        for text in train_records:
            yield "train", text
        for text in validation_records:
            yield "val", text
    return write_streams(records(), Path(output_dir), {
        "dataset": "wikitext103", "subset": "wikitext-103-raw-v1",
        "validation": "official_validation", "excluded_splits": ["test"],
        "source": source or {"kind": "provided_records"},
    }, encoder=encoder, num_threads=num_threads, overwrite=overwrite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, help="Raw train TXT or JSONL (text field)")
    parser.add_argument("--validation-file", type=Path, help="Official validation TXT or JSONL")
    parser.add_argument("--download", action="store_true", help="Explicitly download from Salesforce/wikitext")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/wikitext103")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    local = args.train_file is not None or args.validation_file is not None
    if args.download == local:
        parser.error("Choose --download OR both --train-file and --validation-file")
    if local and (args.train_file is None or args.validation_file is None):
        parser.error("Local preparation needs both --train-file and --validation-file")
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if local:
        source = {"kind": "local", "train": file_info(args.train_file),
                  "validation": file_info(args.validation_file)}
        train, val = text_records(args.train_file), text_records(args.validation_file)
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Install datasets for --download; local files do not require it") from error
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                               split=["train", "validation"], cache_dir=args.cache_dir)
        train, val = ((row["text"] for row in split) for split in dataset)
        source = {"kind": "huggingface", "dataset": "Salesforce/wikitext",
                  "fingerprints": [split._fingerprint for split in dataset]}
    prepare_dataset(train, val, args.output_dir, source=source,
                    num_threads=args.num_threads, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
