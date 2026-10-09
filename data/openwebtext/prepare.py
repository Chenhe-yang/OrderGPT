"""Prepare every supplied OpenWebText document into one GPT-2 training stream."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data._text import file_info, text_records, write_streams

PARQUET_SOURCE = "hf://datasets/Skylion007/openwebtext@refs/convert/parquet/plain_text/train/*.parquet"


def arrow_files(directory):
    files = sorted(Path(directory).glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No *.arrow shards found in {directory}")
    return files


def arrow_batches(files):
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("Install pyarrow to read existing Arrow shards") from error
    for path in files:
        with pa.memory_map(str(path), "r") as handle:
            try:
                reader = pa.ipc.open_stream(handle)
            except pa.ArrowInvalid:
                handle.seek(0)
                reader = pa.ipc.open_file(handle)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            else:
                batches = iter(reader)
            for batch in batches:
                if batch.schema.get_field_index("text") < 0:
                    raise ValueError(f"Arrow shard has no text column: {path}")
                yield batch


def prepare_dataset(documents, num_documents, output_dir, *, source=None,
                    encoder=None, num_threads=8, overwrite=False):
    count = int(num_documents)
    if count < 1:
        raise ValueError("OpenWebText preparation requires at least one document")

    def records():
        seen = 0
        for index, text in enumerate(documents):
            if index >= count:
                raise ValueError("Source gained documents after its counting pass")
            yield "train", text
            seen += 1
        if seen != count:
            raise ValueError(f"Processed {seen} documents; expected {count}")

    return write_streams(records(), Path(output_dir), {
        "dataset": "openwebtext", "source": source or {"kind": "provided_records"},
        "preparation": {"assignment": "all_documents_to_train",
                        "document_order": "source_order", "documents": count},
    }, encoder=encoder, num_threads=num_threads, overwrite=overwrite,
       split_names=("train",))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-jsonl", type=Path, help="One full document per JSON line, text field")
    source_group.add_argument("--input-arrow-dir", type=Path, help="Read all *.arrow shards in this directory, sorted by filename")
    source_group.add_argument("--input-dataset-dir", type=Path, help="Read all rows/splits from a datasets.save_to_disk directory")
    source_group.add_argument("--download", action="store_true", help="Download upstream HF Parquet conversion")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/openwebtext")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Replace prepared outputs and unlink obsolete val.bin without following symlinks")
    args = parser.parse_args(argv)
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if args.input_jsonl is not None:
        if args.input_jsonl.suffix.lower() != ".jsonl":
            parser.error("--input-jsonl must end in .jsonl; lines are documents, not paragraphs")
        count = sum(1 for _ in text_records(args.input_jsonl))
        documents = text_records(args.input_jsonl)
        source = {"kind": "jsonl", **file_info(args.input_jsonl)}
    elif args.input_arrow_dir is not None:
        files = arrow_files(args.input_arrow_dir)
        count = sum(batch.num_rows for batch in arrow_batches(files))
        documents = (text for batch in arrow_batches(files) for text in batch.column("text").to_pylist())
        source = {"kind": "arrow", "files": [file_info(path) for path in files]}
    else:
        try:
            from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
        except ImportError as error:
            raise RuntimeError("Install datasets for --download or --input-dataset-dir") from error
        if args.input_dataset_dir is not None:
            dataset = load_from_disk(str(args.input_dataset_dir))
            source = {"kind": "saved_dataset", "path": str(args.input_dataset_dir.resolve())}
            if isinstance(dataset, DatasetDict):
                if not dataset:
                    raise ValueError("OpenWebText source DatasetDict contains no splits")
                source["split_order"] = list(dataset)
                source["split_documents"] = {name: len(split) for name, split in dataset.items()}
                source["split_fingerprints"] = {name: split._fingerprint for name, split in dataset.items()}
                if any("text" not in split.column_names for split in dataset.values()):
                    raise ValueError("Every OpenWebText source split must contain a text column")
                dataset = concatenate_datasets([split.select_columns(["text"]) for split in dataset.values()])
        else:
            dataset = load_dataset("parquet", data_files={"train": PARQUET_SOURCE},
                                   split="train", cache_dir=args.cache_dir)
            source = {"kind": "huggingface_parquet", "url": PARQUET_SOURCE}
        if "text" not in dataset.column_names:
            raise ValueError("OpenWebText source must contain a text column")
        count = len(dataset)
        documents = (row["text"] for row in dataset)
        source["fingerprint"] = dataset._fingerprint
    prepare_dataset(documents, count, args.output_dir, source=source,
                    num_threads=args.num_threads, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
