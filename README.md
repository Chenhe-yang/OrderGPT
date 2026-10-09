# OrderGPT

OrderGPT learns autoregressive generation orders from model attention and distills a Spectral Teacher into a lightweight Order MLP.
It supports WikiText103, OpenWebText, H3WB, and CENSUS. The final order is fixed when the annealing probability reaches 1.

## Installation

Use Python 3.10 or 3.11 with CUDA-enabled PyTorch 2.4 or later.

```bash
conda create -n ordergpt python=3.10 -y
conda activate ordergpt
pip install -r requirements.txt
```

Run all commands from the repository root. W&B is disabled by default; to enable it, install `wandb` and set `--wandb_log=True`.

## Data Preparation

```bash
# WikiText103
python data/wikitext103/prepare.py --download

# OpenWebText: use all documents for training, without a validation split
python data/openwebtext/prepare.py --download

# H3WB
python data/h3wb/download.py --include-i2d-test
python data/h3wb/prepare.py \
  --source data/h3wb/raw/h3wb_train.npz \
  --test-zip data/h3wb/raw/h3wb_i2d_test.zip

# CENSUS
python data/census/download.py
python data/census/prepare.py
```

To use prepared data, pass `--data_dir=/path/to/data` to the training script. OpenWebText requires only `train.bin`.
See [Data Preparation](data/README.md) for local JSONL, Arrow, and other input formats.

## Training

```bash
python train.py config/wikitext103/mlp.py
python train.py config/h3wb/mlp.py
python train.py config/census/mlp.py

# OpenWebText: multi-GPU training
python scripts/launch.py --dataset openwebtext --devices 0,1 --batch-size 8 \
  --no-compile --out-dir out/ordergpt/openwebtext/mlp
```

Select devices with `--devices` and set the per-device `--batch-size` to fit available memory. The launcher computes gradient accumulation to preserve the global batch size.
Use `--dry-run` to preview the command and `--data-dir` to specify the data directory.

The default method is `mlp` (OrderGPT). Other options are `teacher`, `canonical`, `random`, `fixed`, and `fixed_curriculum`, for example `--method=random`.
Configurations are in `config/<dataset>/`; override training parameters with `--field=value`.

Language tasks save `ckpt.pt`; H3WB and CENSUS save `last.pt` and `best.pt`.
To resume WikiText103 training:

```bash
python train.py config/wikitext103/mlp.py \
  --init_from=resume --resume_checkpoint=out/ordergpt/wikitext103/mlp/ckpt.pt
```

For multi-GPU runs, add `--resume /path/to/ckpt.pt` to the original launch command. Keep all per-rank sidecar files and use the same device count, batch size, and data settings.

## Evaluation

```bash
# WikiText103 perplexity
python scripts/eval/eval_language.py \
  --checkpoint out/ordergpt/wikitext103/mlp/ckpt.pt \
  --records data/wikitext103/val.bin --order saved --output out/eval/wikitext103.json

# OpenWebText: evaluate on independent benchmark records
python scripts/eval/eval_language.py \
  --checkpoint out/ordergpt/openwebtext/mlp/ckpt.pt \
  --records /path/to/lambada_records.npy --order saved --output out/eval/lambada.json

# H3WB NLL and MPJPE
python scripts/eval/eval_h3wb_metrics.py \
  --checkpoint out/ordergpt/h3wb/mlp/best.pt --split test --output-dir out/eval/h3wb

# CENSUS NLL
python scripts/eval/eval_census_full_table_nll.py \
  --checkpoint out/ordergpt/census/mlp/best.pt --output-dir out/eval/census_nll

# CENSUS Q-error: prepare a shared query workload for all methods
python scripts/eval/prepare_census_workload.py \
  --data-dir data/census/records --output data/census/workload.json
python scripts/eval/eval_census_progressive_sampling.py \
  --checkpoint out/ordergpt/census/mlp/best.pt --workload data/census/workload.json \
  --num-queries 1000 --budget 10000 --output-dir out/eval/census_qerror
```

Prepare external evaluation data for OpenWebText separately, as integer NPY arrays of shape `[N, D]` or uint16 BIN files.
Run each script with `--help` for additional options.

## Acknowledgements

We thank the authors of [AO-GPT-MDM](https://github.com/scxue/AO-GPT-MDM) for sharing their code and providing a reference for this project.
