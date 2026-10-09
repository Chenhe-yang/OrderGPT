from __future__ import annotations

import ast
import json
import math
import os
import pickle
import runpy
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ordergpt.language_model import (
    LanguageModelConfig,
    OrderGPTLanguageModel,
    filter_legacy_language_model_args,
)
from ordergpt.language_method import (
    AttentionWindow,
    AttentionWindowAccumulator,
    orient_raw_reverse,
    oriented_minmax_priority,
    recover_mutual_fiedler,
)
from ordergpt.order_mlp import OrderMLP, priority_to_order
from ordergpt.schedule import OrderSchedule


@dataclass
class LanguageConfig:

    method: str = "teacher"
    dataset_name: str = "wikitext103"
    data_dir: str = "data/wikitext103"
    out_dir: str = "out/ordergpt/wikitext103/teacher"
    init_from: str = "scratch"
    resume_checkpoint: str = ""
    always_save_checkpoint: bool = True
    eval_only: bool = False


    variable_permutation_seed: int = 42
    variable_permutation_backend: str = "numpy_default_rng"
    seed: int = 2053


    block_size: int = 80
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


    batch_size: int = 256
    gradient_accumulation_steps: int = 2
    max_iters: int = 50_000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 0
    lr_decay_iters: int = 50_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0


    distill_start: int = 10_000
    distill_end: int = 18_000
    anneal_end: int = 32_000
    attention_window_samples: int = 1024


    mlp_hidden_dims: tuple[int, ...] = (32, 32)
    mlp_learning_rate: float = 2e-4
    mlp_weight_decay: float = 0.01
    mlp_beta1: float = 0.9
    mlp_beta2: float = 0.99
    mlp_grad_clip: float = 1.0
    mlp_seed_offset: int = 3_000_003


    eval_interval: int = 250
    eval_iters: int = 200
    eval_batch_size: int = 64
    log_interval: int = 10
    wandb_log: bool = False
    wandb_project: str = "OrderGPT"
    wandb_run_name: str = ""
    wandb_mode: str = "online"


    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    ddp_backend: str = "nccl"
    expected_world_size: int = 0

    def validate(self) -> None:
        if self.method not in {"teacher", "mlp", "random", "canonical", "fixed", "fixed_curriculum"}:
            raise ValueError("unsupported method")
        if self.dataset_name not in {"wikitext103", "openwebtext"}:
            raise ValueError(
                "dataset_name must be 'wikitext103' or 'openwebtext'"
            )
        if self.dataset_name == "openwebtext" and self.eval_only:
            raise ValueError("OpenWebText uses the complete corpus for training and has no validation split; use scripts/eval/eval_language.py with an independent benchmark")
        if self.init_from not in {"scratch", "resume"}:
            raise ValueError("init_from must be 'scratch' or 'resume'")
        if self.block_size < 2:
            raise ValueError("block_size must contain at least two tokens")
        for name in (
            "vocab_size",
            "n_layer",
            "n_head",
            "n_embd",
            "batch_size",
            "gradient_accumulation_steps",
            "max_iters",
            "attention_window_samples",
            "eval_interval",
            "log_interval",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dataset_name == "wikitext103" and (self.eval_iters <= 0 or self.eval_batch_size <= 0):
            raise ValueError("eval_iters and eval_batch_size must be positive for WikiText103")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.variable_permutation_seed < -1:
            raise ValueError("variable_permutation_seed must be -1 or non-negative")
        if self.variable_permutation_backend not in {
            "numpy_default_rng",
            "torch_randperm",
        }:
            raise ValueError(
                "variable_permutation_backend must be "
                "'numpy_default_rng' or 'torch_randperm'"
            )
        if self.expected_world_size < 0:
            raise ValueError("expected_world_size must be zero or positive")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if not 0 <= self.distill_start < self.distill_end < self.anneal_end:
            raise ValueError("require 0 <= distill_start < distill_end < anneal_end")
        if self.method in {"teacher", "mlp", "fixed_curriculum"} and self.anneal_end > self.max_iters:
            raise ValueError("anneal_end must not exceed max_iters")
        if not self.mlp_hidden_dims or any(int(v) <= 0 for v in self.mlp_hidden_dims):
            raise ValueError("mlp_hidden_dims must contain positive dimensions")
        if self.dropout != 0.0:
            raise ValueError(
                "the core trainer fixes dropout=0 so attention observation and "
                "attention-based orientation are deterministic"
            )


def learned_order_probability(config: LanguageConfig, iteration: int) -> float:


    iteration = int(iteration)
    if config.method in {"canonical", "fixed"}:
        return 1.0
    if config.method == "random":
        return 0.0
    return OrderSchedule(config.distill_start, config.distill_end, config.anneal_end).probability(iteration)


def _literal(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(argv: Optional[Sequence[str]] = None) -> LanguageConfig:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("--"):
        raise SystemExit(
            "usage: python train.py CONFIG.py [--field=value ...]"
        )
    config_path = Path(args.pop(0))
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    namespace = runpy.run_path(str(config_path))
    allowed = {field.name for field in fields(LanguageConfig)}
    values: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in namespace.items():
        if key.startswith("_"):
            continue
        if key in allowed:
            values[key] = value
        elif not callable(value) and not hasattr(value, "__spec__"):
            unknown.append(key)
    if unknown:
        raise ValueError(
            "flat OrderGPT config contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    for raw in args:
        if not raw.startswith("--") or "=" not in raw:
            raise ValueError(f"override must be --field=value, got {raw!r}")
        key, text = raw[2:].split("=", 1)
        if key not in allowed:
            raise ValueError(f"unsupported OrderGPT config field {key!r}")
        values[key] = _literal(text)
    if "mlp_hidden_dims" in values:
        raw_dims = values["mlp_hidden_dims"]
        if isinstance(raw_dims, str):
            raw_dims = [part for part in raw_dims.split(",") if part.strip()]
        values["mlp_hidden_dims"] = tuple(int(value) for value in raw_dims)
    config = LanguageConfig(**values)
    config.validate()
    return config


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    master: bool
    device: torch.device


def _distributed_context(config: LanguageConfig) -> DistributedContext:
    enabled = int(os.environ.get("RANK", "-1")) >= 0
    if enabled:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = config.ddp_backend
        if not torch.cuda.is_available() and backend == "nccl":
            backend = "gloo"
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        process_group_args = {"backend": backend}
        if backend == "nccl":
            process_group_args["device_id"] = device
        dist.init_process_group(**process_group_args)
        return DistributedContext(
            True, rank, local_rank, world_size, rank == 0, device
        )
    requested = config.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return DistributedContext(False, 0, 0, 1, True, device)


def _autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=torch_dtype)


class TokenMemmapData:


    def __init__(self, config: LanguageConfig) -> None:
        self.data_dir = Path(config.data_dir)
        self.block_size = int(config.block_size)
        self.split_bytes: dict[str, int] = {}
        splits = ("train",) if config.dataset_name == "openwebtext" else ("train", "val")
        for split in splits:
            path = self.data_dir / f"{split}.bin"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing {config.dataset_name} token file: {path}"
                )
            if path.stat().st_size // np.dtype(np.uint16).itemsize <= self.block_size:
                raise ValueError(f"{path} is shorter than block_size")
            self.split_bytes[split] = int(path.stat().st_size)
        if "val" in self.split_bytes and (self.data_dir / "train.bin").samefile(self.data_dir / "val.bin"):
            raise ValueError("train.bin and val.bin refer to the same file; prepare an independent validation split")
        if config.variable_permutation_seed == -1:
            permutation = np.arange(self.block_size, dtype=np.int64)
        elif config.variable_permutation_backend == "torch_randperm":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.variable_permutation_seed)
            permutation = torch.randperm(
                self.block_size,
                generator=generator,
                device="cpu",
            ).numpy()
        else:
            permutation = np.random.default_rng(
                config.variable_permutation_seed
            ).permutation(self.block_size)
        self.current_to_original = torch.from_numpy(permutation.copy()).long()

    def fingerprint(self) -> dict[str, Any]:


        return {
            "dtype": "uint16",
            **{f"{split}_bytes": size for split, size in self.split_bytes.items()},
        }

    def get_batch(
        self,
        split: str,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        if split not in self.split_bytes:
            raise ValueError(f"split {split!r} is unavailable for this dataset; available: {tuple(self.split_bytes)}")
        data = np.memmap(
            self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r"
        )
        starts = torch.randint(
            0,
            len(data) - self.block_size,
            (int(batch_size),),
            generator=generator,
        ).tolist()
        values = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        data[start : start + self.block_size],
                        dtype=np.int64,
                    ).copy()
                )
                for start in starts
            ]
        )
        values = values[:, self.current_to_original]
        if device.type == "cuda":
            return values.pin_memory().to(device, non_blocking=True)
        return values.to(device)


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _random_orders(
    batch_size: int,
    num_tokens: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(
                int(num_tokens),
                device=device,
                generator=generator,
            )
            for _ in range(int(batch_size))
        ],
        dim=0,
    )


def _training_orders(
    batch_size: int,
    config: LanguageConfig,
    iteration: int,
    cached_order: Optional[torch.Tensor],
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    probability = learned_order_probability(config, iteration)
    if cached_order is None:
        if probability >= 1.0:
            raise RuntimeError("final order is missing at anneal_end")
        probability = 0.0
    if probability >= 1.0:
        return cached_order.to(device).expand(batch_size, -1), probability
    orders = _random_orders(batch_size, config.block_size, device, generator)
    if probability > 0.0:
        use_learned = torch.rand(batch_size, device=device, generator=generator) < probability
        orders[use_learned] = cached_order.to(device)
    return orders, probability


def _extract_attentions(
    outputs: tuple,
) -> torch.Tensor | Sequence[torch.Tensor]:
    value = outputs[-1] if outputs else None
    if torch.is_tensor(value) and value.ndim == 4:
        return value
    if isinstance(value, (list, tuple)) and value:
        return value
    raise RuntimeError(
        "ordinary training forward did not return all-layer/all-head attention"
    )


def _clean_model_state(state: Mapping[str, torch.Tensor]) -> dict:
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("_orig_mod.", "module."):
            while name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def _model_args(config: LanguageConfig, vocab_size: int) -> dict[str, Any]:
    return {
        "block_size": config.block_size,
        "vocab_size": int(vocab_size),
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "bias": config.bias,
    }


_RESUME_LOCKED_FIELDS = (
    "variable_permutation_seed",
    "variable_permutation_backend",
    "block_size",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "batch_size",
    "gradient_accumulation_steps",
    "distill_start",
    "distill_end",
    "anneal_end",
    "attention_window_samples",
    "expected_world_size",
    "seed", "learning_rate", "min_lr", "warmup_iters", "lr_decay_iters",
    "weight_decay", "beta1", "beta2", "grad_clip", "mlp_learning_rate",
    "mlp_weight_decay", "mlp_beta1", "mlp_beta2", "mlp_grad_clip", "mlp_seed_offset",
)


def _resume_dataset_name(saved_config: Mapping[str, Any]) -> str:
    dataset = saved_config.get(
        "dataset_name", saved_config.get("dataset")
    )
    if dataset is None and saved_config.get("data_dir"):
        dataset = Path(saved_config["data_dir"]).name


    return "wikitext103" if dataset is None else str(dataset)


def _normalized_resume_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_resume_value(item) for item in value)
    return value


def _validate_resume_contract(
    saved_config: Mapping[str, Any],
    config: LanguageConfig,
) -> None:
    legacy = {"attention_start_iter", "order_freeze_iter", "anneal_start_iter", "mlp_distill_stop_iter"}
    if legacy.intersection(saved_config):
        raise ValueError("legacy checkpoints use a different lifecycle; start a fresh run with the three-milestone config")
    if saved_config.get("method", config.method) != config.method:
        raise ValueError("cannot resume a different OrderGPT method")
    saved_dataset = _resume_dataset_name(saved_config)
    if saved_dataset != config.dataset_name:
        raise ValueError("cannot resume a different language dataset")

    mismatches = []
    for name in _RESUME_LOCKED_FIELDS:
        if name == "variable_permutation_backend":
            saved_value = saved_config.get(name)
            if saved_value is None:
                saved_value = (
                    "torch_randperm"
                    if saved_dataset == "openwebtext"
                    else "numpy_default_rng"
                )
        elif name in saved_config:
            saved_value = saved_config[name]
        else:
            continue
        current_value = getattr(config, name)
        if _normalized_resume_value(saved_value) != _normalized_resume_value(
            current_value
        ):
            mismatches.append(
                f"{name}: checkpoint={saved_value!r}, "
                f"config={current_value!r}"
            )
    if config.method == "mlp":
        for name in ("distill_end", "mlp_hidden_dims"):
            if name not in saved_config:
                continue
            saved_value = saved_config[name]
            current_value = getattr(config, name)
            if _normalized_resume_value(
                saved_value
            ) != _normalized_resume_value(current_value):
                mismatches.append(
                    f"{name}: checkpoint={saved_value!r}, "
                    f"config={current_value!r}"
                )
    if mismatches:
        raise ValueError(
            "resume contract mismatch: " + "; ".join(mismatches)
        )


def _metadata_vocab_size(config: LanguageConfig) -> int:
    # Tokenizer size and padded output-head size are different quantities.
    json_path = Path(config.data_dir) / "metadata.json"
    pickle_path = Path(config.data_dir) / "meta.pkl"
    if json_path.is_file():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    elif pickle_path.is_file():
        with pickle_path.open("rb") as handle:
            metadata = pickle.load(handle)
    else:
        metadata = {}
    if int(metadata.get("vocab_size", config.vocab_size)) > config.vocab_size:
        raise ValueError("configured vocabulary is smaller than the prepared tokenizer vocabulary")
    return int(config.vocab_size)


def _learning_rate(config: LanguageConfig, iteration: int) -> float:
    if iteration < config.warmup_iters:
        return config.learning_rate * (iteration + 1) / max(
            1, config.warmup_iters
        )
    if iteration >= config.lr_decay_iters:
        return config.min_lr
    ratio = (iteration - config.warmup_iters) / max(
        1, config.lr_decay_iters - config.warmup_iters
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coefficient * (
        config.learning_rate - config.min_lr
    )


def _runtime_sample_validation(
    config: LanguageConfig, context: DistributedContext
) -> int:
    if (
        config.expected_world_size
        and context.world_size != config.expected_world_size
    ):
        raise ValueError(
            f"configuration expects world_size={config.expected_world_size}, "
            f"got {context.world_size}"
        )
    if config.method not in {"teacher", "mlp"}:
        return config.attention_window_samples
    if config.attention_window_samples % context.world_size != 0:
        raise ValueError(
            "attention_window_samples must be divisible by DDP world size"
        )
    local_window = config.attention_window_samples // context.world_size
    if local_window % config.batch_size != 0:
        raise ValueError(
            "each rank's attention window must contain whole microbatches"
        )
    local_samples_per_iteration = (
        config.batch_size * config.gradient_accumulation_steps
    )
    discovery_samples = (
        config.anneal_end - config.distill_start
    ) * local_samples_per_iteration
    if discovery_samples % local_window != 0:
        raise ValueError(
            "attention start/freeze boundaries must contain complete windows"
        )
    if config.method == "mlp":
        distill_samples = (
            config.distill_end - config.distill_start
        ) * local_samples_per_iteration
        if distill_samples % local_window != 0:
            raise ValueError(
                "MLP distillation boundary must end on an attention window"
            )
    return local_window


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    data: TokenMemmapData,
    cached_order: Optional[torch.Tensor],
    config: LanguageConfig,
    context: DistributedContext,
    iteration: int,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 9_000_001)
    order_generator = _device_generator(
        context.device, config.seed + 9_000_101 + context.rank
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=context.device)
    sample_count = torch.zeros_like(loss_sum)
    try:
        for _ in range(config.eval_iters):
            values = data.get_batch(
                "val", config.eval_batch_size, generator, context.device
            )
            if cached_order is None:
                orders = _random_orders(
                    config.eval_batch_size,
                    config.block_size,
                    context.device,
                    order_generator,
                )
            else:
                orders = cached_order.to(context.device).expand(
                    config.eval_batch_size, -1
                )
            with _autocast_context(context.device, config.dtype):
                outputs = model(values, mode=None, orders=orders)
            loss_sum.add_(
                outputs[1].detach().double()
                * float(config.eval_batch_size)
            )
            sample_count.add_(float(config.eval_batch_size))
    finally:
        if was_training:
            model.train()
    if context.enabled:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    return float((loss_sum / sample_count).item())


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class LanguageTrainer:
    def __init__(
        self, config: LanguageConfig, context: DistributedContext
    ) -> None:
        self.config = config
        self.context = context
        self.out_dir = Path(config.out_dir)
        if context.master:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        if context.enabled:
            dist.barrier()
        local_window = _runtime_sample_validation(config, context)
        if context.master:
            _json_write(self.out_dir / "resolved_config.json", {
                **asdict(config), "world_size": context.world_size,
                "global_batch_size": config.batch_size * config.gradient_accumulation_steps * context.world_size,
            })
        self.data = TokenMemmapData(config)


        torch.manual_seed(config.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self.checkpoint_path = self.out_dir / "ckpt.pt"
        self.resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else self.checkpoint_path
        checkpoint = None
        vocab_size = _metadata_vocab_size(config)
        model_args = _model_args(config, vocab_size)
        if config.init_from == "resume":
            checkpoint = torch.load(
                self.resume_path, map_location="cpu", weights_only=False
            )
            saved_config = checkpoint.get("config", {})
            _validate_resume_contract(saved_config, config)
            saved_fingerprint = checkpoint.get("data_fingerprint")
            if saved_fingerprint is not None:
                saved_fingerprint = dict(saved_fingerprint)
                if config.dataset_name == "openwebtext":
                    saved_fingerprint.pop("val_bytes", None)
            if (
                saved_fingerprint is not None
                and saved_fingerprint != self.data.fingerprint()
            ):
                raise ValueError("checkpoint token-stream fingerprint mismatch")
            saved_mapping = checkpoint.get("current_to_original")
            if saved_mapping is not None and not torch.equal(
                torch.as_tensor(saved_mapping, dtype=torch.long),
                self.data.current_to_original,
            ):
                raise ValueError(
                    "checkpoint current-ID coordinate mapping mismatch"
                )
            saved_world_size = checkpoint.get("world_size")
            if (
                saved_world_size is not None
                and int(saved_world_size) != context.world_size
            ):
                raise ValueError("cannot resume with a different world size")
            model_args = filter_legacy_language_model_args(
                checkpoint["model_args"]
            )
            if int(model_args["block_size"]) != config.block_size:
                raise ValueError("checkpoint/config block_size mismatch")

        self.model_args = model_args
        self.model = OrderGPTLanguageModel(
            LanguageModelConfig(**model_args)
        ).to(context.device)
        if checkpoint is not None:
            self.model.load_state_dict(
                _clean_model_state(checkpoint["model"]), strict=True
            )
        self.optimizer = self.model.configure_optimizers(
            config.weight_decay,
            config.learning_rate,
            (config.beta1, config.beta2),
            context.device.type,
        )
        scaler_enabled = (
            context.device.type == "cuda" and config.dtype == "float16"
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=scaler_enabled
        )
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scaler"):
                self.scaler.load_state_dict(checkpoint["scaler"])

        self.order_mlp: Optional[OrderMLP] = None
        self.mlp_optimizer: Optional[torch.optim.Optimizer] = None
        self.mlp_frozen = False
        if config.method == "mlp":
            fork_devices = (
                [context.device]
                if context.device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=fork_devices, enabled=True):
                torch.manual_seed(config.seed + config.mlp_seed_offset)
                if context.device.type == "cuda":
                    torch.cuda.manual_seed_all(
                        config.seed + config.mlp_seed_offset
                    )
                self.order_mlp = OrderMLP(
                    num_variables=config.block_size,
                    hidden_dims=config.mlp_hidden_dims,
                ).to(context.device)
            self.mlp_optimizer = torch.optim.AdamW(
                self.order_mlp.parameters(),
                lr=config.mlp_learning_rate,
                betas=(config.mlp_beta1, config.mlp_beta2),
                weight_decay=config.mlp_weight_decay,
            )
            if checkpoint is not None:
                self.order_mlp.load_state_dict(checkpoint["order_mlp"])
                saved_mlp_optimizer = checkpoint.get("mlp_optimizer")
                if saved_mlp_optimizer is None:
                    self.mlp_optimizer = None
                else:
                    self.mlp_optimizer.load_state_dict(
                        saved_mlp_optimizer
                    )

        self.accumulator = AttentionWindowAccumulator(
            config.block_size, local_window, context.device
        )
        self.data_generator = torch.Generator(device="cpu")
        self.data_generator.manual_seed(config.seed + 101 + context.rank)
        self.order_generator = _device_generator(
            context.device, config.seed + 303 + context.rank
        )
        self.cached_order: Optional[torch.Tensor] = None
        if config.method == "canonical":
            self.cached_order = torch.argsort(self.data.current_to_original).to(context.device)
        elif config.method in {"fixed", "fixed_curriculum"}:
            fixed_generator = torch.Generator().manual_seed(config.seed + 2_000_003)
            self.cached_order = torch.randperm(config.block_size, generator=fixed_generator).to(context.device)
        self.iteration = 0
        self.best_val_loss = float("inf")
        self.teacher_updates = 0
        self.mlp_updates = 0
        self.order_refreshes = 0
        if checkpoint is not None:
            self.iteration = int(checkpoint["iter_num"])
            if config.dataset_name == "wikitext103":
                self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            self.teacher_updates = int(checkpoint.get("teacher_updates", 0))
            self.mlp_updates = int(checkpoint.get("mlp_updates", 0))
            self.order_refreshes = int(checkpoint.get("order_refreshes", 0))
            self._restore_runtime(self._load_runtime(checkpoint))

        if (
            config.method == "mlp"
            and self.iteration >= config.distill_end
        ):
            self._freeze_mlp()

        train_model: torch.nn.Module = self.model
        if config.compile_model:
            train_model = torch.compile(train_model)
        if context.enabled:
            device_ids = (
                [context.local_rank]
                if context.device.type == "cuda"
                else None
            )
            train_model = DistributedDataParallel(
                train_model, device_ids=device_ids
            )
        self.train_model = train_model
        self.run = self._init_wandb()

    def _init_wandb(self):
        if not self.config.wandb_log or not self.context.master:
            return None
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb_log=True but wandb is not installed") from exc
        return wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name or None,
            mode=self.config.wandb_mode,
            config=asdict(self.config),
            dir=str(self.out_dir),
        )

    def _freeze_mlp(self) -> None:
        if self.order_mlp is None or self.mlp_frozen:
            return
        self.order_mlp.eval()
        for parameter in self.order_mlp.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


        self.mlp_optimizer = None
        self.mlp_frozen = True

    def _runtime_state(self) -> dict[str, Any]:
        return {
            "iter_num": self.iteration,
            "cached_order": (
                None
                if self.cached_order is None
                else self.cached_order.detach().cpu()
            ),
            "accumulator": self.accumulator.state_dict(),
            "data_generator": self.data_generator.get_state(),
            "order_generator": self.order_generator.get_state(),
            "mlp_frozen": self.mlp_frozen,
        }

    def _restore_runtime(self, state: Mapping[str, Any]) -> None:
        if int(state.get("iter_num", self.iteration)) != self.iteration:
            raise ValueError("runtime sidecar does not match checkpoint iteration")
        cached = state.get("cached_order")
        if cached is not None:
            cached = torch.as_tensor(cached, dtype=torch.long)
            if (
                cached.numel() != self.config.block_size
                or not torch.equal(
                    torch.sort(cached).values,
                    torch.arange(self.config.block_size),
                )
            ):
                raise ValueError("checkpoint cached_order is not a permutation")
            self.cached_order = cached.to(self.context.device)
        self.accumulator.load_state_dict(state["accumulator"])
        self.data_generator.set_state(state["data_generator"].cpu())
        self.order_generator.set_state(state["order_generator"].cpu())
        if bool(state.get("mlp_frozen", False)):
            self._freeze_mlp()
        if self.iteration >= self.config.anneal_end and self.config.method in {"teacher", "mlp", "fixed_curriculum"}:
            if self.cached_order is None or self.accumulator.samples:
                raise ValueError("frozen checkpoint must have a valid order and no partial attention window")

    def _runtime_path(self, *, resume: bool = False) -> Path:
        return Path(
            str(self.resume_path if resume else self.checkpoint_path)
            + f".rank{self.context.rank}.runtime.pt"
        )

    def _load_runtime(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.context.enabled:
            return checkpoint["runtime"]
        path = self._runtime_path(resume=True)
        if not path.is_file():
            raise FileNotFoundError(f"missing per-rank runtime checkpoint: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def _save_checkpoint(self) -> None:
        runtime = self._runtime_state()
        if self.context.master:
            payload = {
                "model": self.model.state_dict(),
                "model_args": self.model_args,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "order_mlp": (
                    None
                    if self.order_mlp is None
                    else self.order_mlp.state_dict()
                ),
                "mlp_optimizer": (
                    None
                    if self.mlp_optimizer is None
                    else self.mlp_optimizer.state_dict()
                ),
                "config": asdict(self.config),
                "data_fingerprint": self.data.fingerprint(),
                "current_to_original": self.data.current_to_original.clone(),
                "world_size": self.context.world_size,
                "iter_num": self.iteration,
                "cached_order": runtime["cached_order"],
                "order_frozen": self.config.method != "random" and (self.iteration >= self.config.anneal_end or self.config.method in {"canonical", "fixed"}),
                "teacher_updates": self.teacher_updates,
                "mlp_updates": self.mlp_updates,
                "order_refreshes": self.order_refreshes,
            }
            if self.config.dataset_name == "wikitext103":
                payload["best_val_loss"] = self.best_val_loss
            if not self.context.enabled:
                payload["runtime"] = runtime
            torch.save(payload, self.checkpoint_path)
        if self.context.enabled:
            dist.barrier()
            torch.save(runtime, self._runtime_path())
            dist.barrier()

    def _write_order_artifact(
        self,
        iteration: int,
        record: Mapping[str, Any],
    ) -> None:
        if not self.context.master or self.cached_order is None:
            return
        _json_write(
            self.out_dir / "learned_order.json",
            {
                "schema_version": 1,
                "method": self.config.method,
                "dataset": self.config.dataset_name,
                "coordinate_frame": "current_token_ids",
                "current_to_original_audit_only": (
                    self.data.current_to_original.tolist()
                ),
                "order_current_ids": self.cached_order.detach().cpu().tolist(),
                "iteration": int(iteration),
                "last_update": dict(record),
            },
        )

    def _distributed_window(
        self, local_window: AttentionWindow
    ) -> torch.Tensor:
        numerator = local_window.numerator
        counts = local_window.exposure_count
        if self.context.enabled:
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return AttentionWindow(
            numerator=numerator,
            exposure_count=counts,
            samples=self.config.attention_window_samples,
        ).directed_matrix()

    def _refresh_order(
        self, directed_attention: torch.Tensor, iteration: int
    ) -> dict[str, Any]:
        matrix_cpu = directed_attention.detach().cpu().to(torch.float64)
        record: dict[str, Any] = {
            "iteration": int(iteration),
            "method": self.config.method,
            "attention_samples": self.config.attention_window_samples,
            "attention_source": "same_ordinary_training_forward",
            "attention_aggregation": (
                "all_layer_all_head_premean_current_id_"
                "target_to_observed_exposure_corrected"
            ),
            "affinity": "dense_mutual_geometric",
        }
        if self.config.method == "teacher":
            fiedler = recover_mutual_fiedler(matrix_cpu)
            orientation = orient_raw_reverse(fiedler)
            self.cached_order = orientation.selected_order.to(
                self.context.device
            )
            self.teacher_updates += 1
            record.update(
                {
                    "selected_reverse": orientation.selected_reverse,
                    "orientation_tied": orientation.tied,
                    "raw_attention_score": orientation.raw_score,
                    "reverse_attention_score": orientation.reverse_score,
                    "order_current_ids": (
                        orientation.selected_order.tolist()
                    ),
                    "fiedler_eigenvalue": float(
                        fiedler.eigenvalues[1].item()
                    ),
                }
            )
        else:
            if self.order_mlp is None:
                raise RuntimeError("MLP runtime is not initialized")
            matrix_device = matrix_cpu.to(
                self.context.device, dtype=torch.float32
            ).detach()
            supervision_active = (
                iteration < self.config.distill_end
            )
            if supervision_active:
                if self.mlp_optimizer is None:
                    raise RuntimeError(
                        "MLP optimizer is missing during distillation"
                    )
                fiedler = recover_mutual_fiedler(matrix_cpu)
                orientation = orient_raw_reverse(fiedler)
                target = oriented_minmax_priority(
                    fiedler,
                    selected_reverse=orientation.selected_reverse,
                ).to(self.context.device)
                self.order_mlp.train()
                self.mlp_optimizer.zero_grad(set_to_none=True)
                mse = self.order_mlp.mse_loss(matrix_device, target)
                mse.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.order_mlp.parameters(),
                    self.config.mlp_grad_clip,
                )
                self.mlp_optimizer.step()
                self.mlp_updates += 1
                record.update(
                    {
                        "supervision_active": True,
                        "selected_reverse": orientation.selected_reverse,
                        "orientation_tied": orientation.tied,
                        "raw_attention_score": orientation.raw_score,
                        "reverse_attention_score": orientation.reverse_score,
                        "distillation_mse": float(mse.detach().item()),
                        "mlp_grad_norm": float(grad_norm.detach().item()),
                    }
                )
            else:
                self._freeze_mlp()
                record["supervision_active"] = False
            self.order_mlp.eval()
            with torch.no_grad():
                priorities = self.order_mlp(matrix_device)
                predicted = priority_to_order(priorities)
            self.cached_order = predicted.to(self.context.device)
            self.order_refreshes += 1
            record["order_current_ids"] = predicted.cpu().tolist()
            record["mlp_frozen"] = self.mlp_frozen
        self._write_order_artifact(iteration, record)
        if self.context.master:
            _append_jsonl(self.out_dir / "order_updates.jsonl", record)
        return record

    def _train_iteration(self, iteration: int) -> tuple[float, list[dict]]:
        config = self.config
        lr = _learning_rate(config, iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if (
            config.method == "mlp"
            and iteration >= config.distill_end
        ):
            self._freeze_mlp()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        ready_windows: list[AttentionWindow] = []
        capture_attention = (
            config.method in {"teacher", "mlp"}
            and config.distill_start
            <= iteration
            < config.anneal_end
        )
        for micro_step in range(config.gradient_accumulation_steps):
            values = self.data.get_batch(
                "train",
                config.batch_size,
                self.data_generator,
                self.context.device,
            )
            orders, _ = _training_orders(
                config.batch_size,
                config,
                iteration,
                self.cached_order,
                self.context.device,
                self.order_generator,
            )
            synchronize = (
                not self.context.enabled
                or micro_step == config.gradient_accumulation_steps - 1
            )
            sync_context = (
                nullcontext()
                if synchronize
                else self.train_model.no_sync()
            )
            with sync_context:
                with _autocast_context(
                    self.context.device, config.dtype
                ):
                    outputs = self.train_model(
                        values,
                        mode=None,
                        orders=orders,
                        return_attentions=capture_attention,
                    )
                    loss = (
                        outputs[1]
                        / config.gradient_accumulation_steps
                    )
                if capture_attention:
                    attentions = _extract_attentions(outputs)
                    if self.accumulator.add(attentions, orders):
                        ready_windows.append(self.accumulator.pop())
                accumulated_loss += float(loss.detach().item())
                self.scaler.scale(loss).backward()
        records: list[dict] = []

        for local_window in ready_windows:
            matrix = self._distributed_window(local_window)
            try:
                records.append(self._refresh_order(matrix, iteration))
            except ValueError as exc:
                record = {
                    "iteration": int(iteration),
                    "updated": False,
                    "reason": f"degenerate_mutual_fiedler:{exc}",
                }
                records.append(record)
                if self.context.master:
                    _append_jsonl(
                        self.out_dir / "order_updates.jsonl", record
                    )
        if config.grad_clip > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return accumulated_loss, records

    def run_training(self) -> None:
        config = self.config
        if config.eval_only:
            if config.init_from != "resume":
                raise ValueError("eval_only requires init_from=resume")
            val_loss = _evaluate(
                self.model,
                self.data,
                self.cached_order,
                config,
                self.context,
                self.iteration,
            )
            if self.context.master:
                print(json.dumps({"val_loss": val_loss}, sort_keys=True))
            return
        started = time.time()
        while self.iteration < config.max_iters:
            iteration = self.iteration
            train_loss, order_records = self._train_iteration(iteration)
            self.iteration = iteration + 1
            if config.method == "mlp" and self.iteration >= config.distill_end:
                self._freeze_mlp()
            if config.method in {"teacher", "mlp"} and self.iteration == config.anneal_end:
                if self.cached_order is None or self.accumulator.samples:
                    raise RuntimeError("anneal_end requires a complete frozen order and no partial attention window")
                self._write_order_artifact(iteration, {"order_frozen": True, "probability": 1.0})
            if (
                self.context.master
                and self.iteration % config.log_interval == 0
            ):
                elapsed = max(time.time() - started, 1e-9)
                samples = (
                    config.batch_size
                    * config.gradient_accumulation_steps
                    * self.context.world_size
                    * config.log_interval
                )
                payload = {
                    "iter": self.iteration,
                    "train_loss": train_loss,
                    "learning_rate": _learning_rate(config, iteration),
                    "learned_order_probability": learned_order_probability(
                        config, iteration
                    ),
                    "has_cached_order": self.cached_order is not None,
                    "samples_per_second": samples / elapsed,
                }
                print(json.dumps(payload, sort_keys=True), flush=True)
                _append_jsonl(self.out_dir / "metrics.jsonl", payload)
                if self.run is not None:
                    self.run.log(payload, step=self.iteration)
                started = time.time()
            periodic_save = (
                self.iteration % config.eval_interval == 0
                or self.iteration == config.max_iters
            )
            if periodic_save:
                if config.dataset_name == "openwebtext":
                    self._save_checkpoint()
                else:
                    val_loss = _evaluate(
                        self.model,
                        self.data,
                        self.cached_order,
                        config,
                        self.context,
                        self.iteration,
                    )
                    improved = val_loss < self.best_val_loss
                    self.best_val_loss = min(self.best_val_loss, val_loss)
                    if self.context.master:
                        payload = {
                            "iter": self.iteration,
                            "val_loss": val_loss,
                            "best_val_loss": self.best_val_loss,
                        }
                        print(json.dumps(payload, sort_keys=True), flush=True)
                        _append_jsonl(self.out_dir / "metrics.jsonl", payload)
                        if self.run is not None:
                            self.run.log(payload, step=self.iteration)
                    if improved or config.always_save_checkpoint:
                        self._save_checkpoint()
        if self.config.method == "mlp" and self.context.master:
            torch.save(
                {
                    "state_dict": self.order_mlp.state_dict(),
                    "config": {
                        "num_variables": self.config.block_size,
                        "hidden_dims": list(
                            self.config.mlp_hidden_dims
                        ),
                    },
                },
                self.out_dir / "order_mlp.pt",
            )
        if self.run is not None:
            self.run.finish()


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = load_config(argv)
    context = _distributed_context(config)
    try:
        trainer = LanguageTrainer(config, context)
        trainer.run_training()
    finally:
        if context.enabled and dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "LanguageConfig",
    "LanguageTrainer",
    "TokenMemmapData",
    "learned_order_probability",
    "load_config",
    "main",
]
