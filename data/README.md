# Data Preparation

Install the project dependencies and run the commands below from the repository root. Raw datasets are downloaded separately.
Prepared data includes `metadata.json` with source information, sample counts, and normalization statistics or categorical vocabularies.

| Dataset | Default output directory | Format |
| --- | --- | --- |
| WikiText103 | `data/wikitext103/` | `train.bin`, `val.bin`: little-endian uint16 token streams |
| OpenWebText | `data/openwebtext/` | `train.bin`: a little-endian uint16 token stream |
| H3WB | `data/h3wb/processed/` | float32 NPY arrays: targets `[N,133,3]`, conditions `[N,133,3]` |
| CENSUS | `data/census/records/` | `train.bin`, `val.bin`, `test.bin`: row-major uint8 records `[N,68]` |

Use `--output-dir` to choose an output directory, then point training to that directory.
Use `--overwrite` to replace language or H3WB outputs. CENSUS validates existing records; use `--force` to replace them.
All scripts support `--help` without network access.

## WikiText103

Use the `wikitext-103-raw-v1` configuration of [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext):

```bash
python data/wikitext103/prepare.py --download
```

Alternatively, prepare local raw files:

```bash
python data/wikitext103/prepare.py \
  --train-file /path/to/wikitext-103-raw/wiki.train.raw \
  --validation-file /path/to/wikitext-103-raw/wiki.valid.raw
```

The official training and validation splits are written to `train.bin` and `val.bin`, respectively. The test split is not processed.
Each TXT line, including its newline, is one record. JSONL files use `{"text": "..."}` per line, preserving the full string and empty records.
Each record is tokenized with GPT-2 BPE and followed by one EOT token (`50256`). The default training window is 80 tokens.

## OpenWebText

The source is [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext); the downloader uses its [Parquet version](https://huggingface.co/datasets/Skylion007/openwebtext/tree/refs%2Fconvert%2Fparquet).
All input documents are tokenized with GPT-2 BPE, followed by EOT, and written to `train.bin` in source order. No validation split is held out.

Choose one input method:

```bash
python data/openwebtext/prepare.py --download --cache-dir /path/to/hf-cache
python data/openwebtext/prepare.py --input-arrow-dir /path/to/openwebtext/arrow
python data/openwebtext/prepare.py --input-dataset-dir /path/to/openwebtext/saved
python data/openwebtext/prepare.py --input-jsonl /path/to/openwebtext/documents.jsonl
```

- Arrow: reads all `*.arrow` files in the directory in filename order, supporting both stream and file formats. The directory should contain only shards from the intended corpus.
- Saved dataset: accepts a `datasets.save_to_disk` directory. All `DatasetDict` splits are combined in saved key order, preserving row order within each split.
- JSONL: one `{"text": "complete document"}` per line. Keep each document intact instead of splitting it into paragraphs.

Allow enough disk space for the download cache and token stream.
Tokenization runs in batches; `--num-threads` controls parallelism. The default training window is 1,024 tokens.

## H3WB masked-I2D

The [official H3WB repository](https://github.com/wholebody3d/wholebody3d) provides the [training NPZ](https://drive.google.com/file/d/1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC/view) and [I2D test archive](https://drive.google.com/file/d/1GBOCjzIW0GLHHLHhGmUcB32Ke8PVjDXH/view).
The upstream test archive is named `task2_test_3d.npz`; the downloader saves it as `h3wb_i2d_test.zip`.

```bash
python data/h3wb/download.py --include-i2d-test
python data/h3wb/prepare.py \
  --source data/h3wb/raw/h3wb_train.npz \
  --test-zip data/h3wb/raw/h3wb_i2d_test.zip
```

For existing local files, set `--source` and `--test-zip` to their paths.
The downloader checks SHA-256 hashes. To verify local files offline:

```bash
python data/h3wb/download.py --verify-only --include-i2d-test \
  --output /path/to/h3wb/h3wb_train.npz \
  --test-output /path/to/h3wb/h3wb_i2d_test.zip
```

The official training data is grouped by subject and action, keeping all camera views of a group in the same split.
Approximately 10% is used for validation, 10% for discovery, and the rest for training.
The 3D targets use pelvis-relative coordinates. Conditions contain masked 2D coordinates and visibility.
Normalization statistics are fitted on the training split and applied to the other splits. The official test set retains its supplied visibility.
The `--test-zip` argument is optional for training preparation and required to prepare the official 10,000 test poses for evaluation.

## CENSUS 1990

Use the processed `USCensus1990.data.txt` file from [UCI US Census Data (1990)](https://archive.ics.uci.edu/dataset/116/us+census+data+1990). The raw `USCensus1990raw.data.txt` and Census Income/Adult datasets are not compatible inputs.

```bash
python data/census/download.py
python data/census/prepare.py
```

To use a local processed source table or verify its prepared records:

```bash
python data/census/prepare.py --source /path/to/census/USCensus1990.data.txt
python data/census/prepare.py --source /path/to/census/USCensus1990.data.txt --verify-only
```

The source table contains 2,458,285 rows and 68 columns after dropping `caseid`.
Each column uses a fixed vocabulary of sorted unique values, encoded as consecutive category IDs starting at 0.
Rows are split consecutively in source order into 2,212,456 training, 122,914 validation, and 122,915 test records.
Preparation scans the source twice in `pandas` chunks. The default `--chunk-size` is 100,000; both source and output files are checked using hashes.

## Offline Tokenizer Cache

WikiText103 and OpenWebText use the GPT-2 vocabulary from `tiktoken`.
Cache it on a machine with internet access:

```bash
export TIKTOKEN_CACHE_DIR=/path/to/tiktoken-cache
python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
```

Copy the cache directory to the offline machine and set `TIKTOKEN_CACHE_DIR` to its new path.
Local JSONL/TXT input requires `numpy` and `tiktoken`. Arrow input also requires `pyarrow`; downloading or reading saved datasets requires `datasets`.
