"""Launch dataset configurations while preserving the global optimizer batch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_BATCH = {"wikitext103": 512, "openwebtext": 480, "h3wb": 160, "census": 512}
METHODS = ("mlp", "teacher", "canonical", "random", "fixed", "fixed_curriculum")


def build_command(args):
    devices = [item.strip() for item in args.devices.split(",")]
    if not devices or any(not item.isascii() or not item.isdecimal() for item in devices):
        raise ValueError("--devices must contain distinct physical GPU indices, e.g. 0,1")
    devices = [str(int(item)) for item in devices]
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct physical GPU indices, e.g. 0,1")
    world = len(devices)
    language = args.dataset in {"wikitext103", "openwebtext"}
    if not language and world != 1:
        raise ValueError("H3WB/CENSUS use one GPU; select exactly one --devices index")
    batch = args.batch_size
    global_batch = GLOBAL_BATCH[args.dataset]
    if batch <= 0 or global_batch % (batch * world):
        raise ValueError(f"micro-batch {batch} × {world} GPUs must divide dataset global batch {global_batch}")
    accumulation = global_batch // (batch * world)
    cmd = [sys.executable]
    if world > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"]
    cmd += ["train.py", f"config/{args.dataset}/{'teacher' if args.method == 'teacher' else 'mlp'}.py",
            f"--method={args.method}", f"--batch_size={batch}", f"--gradient_accumulation_steps={accumulation}",
            f"--out_dir={args.out_dir or 'out/' + args.dataset + '/' + args.method}"]
    if language:
        cmd += [f"--expected_world_size={world}"]
    if args.data_dir:
        cmd += [f"--data_dir={Path(args.data_dir).expanduser().resolve()}"]
    if args.seed is not None:
        cmd += [f"--seed={args.seed}"]
    if args.resume:
        cmd += ["--init_from=resume", f"--resume_checkpoint={Path(args.resume).expanduser().resolve()}"]
    if args.no_compile:
        cmd += ["--compile_model=False"]
    environment = {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
    return cmd, environment, {
        "dataset": args.dataset, "method": args.method,
        "world_size": world, "micro_batch": batch, "accumulation": accumulation,
        "global_batch": global_batch,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=GLOBAL_BATCH, required=True)
    parser.add_argument("--devices", required=True, help="Explicit available physical GPU indices; inspect nvidia-smi first")
    parser.add_argument("--method", choices=METHODS, default="mlp")
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--batch-size", type=int, required=True, help="Per-GPU micro-batch; accumulation is derived from the dataset global batch")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        command, overrides, settings = build_command(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(settings, indent=2), flush=True)
    print(" ".join(key + "=" + shlex.quote(value) for key, value in overrides.items()) + " " + shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env={**os.environ, **overrides}, check=True)


if __name__ == "__main__":
    main()
