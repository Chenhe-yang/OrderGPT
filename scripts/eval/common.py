from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]




def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_checkpoint(path: str | Path) -> Tuple[Path, Dict[str, Any]]:
    resolved = resolve_repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        payload = torch.load(
            resolved,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), Mapping):
        raise ValueError(f"{resolved} is not an OrderGPT checkpoint")
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError(f"{resolved} has no embedded resolved config")
    return resolved, payload


def validate_order(values: Sequence[int] | torch.Tensor, width: int) -> Tuple[int, ...]:
    if torch.is_tensor(values):
        sequence = values.detach().cpu().flatten().tolist()
    else:
        sequence = list(values)
    order = tuple(int(value) for value in sequence)
    if len(order) != int(width) or sorted(order) != list(range(int(width))):
        raise ValueError(f"cached_order must be a permutation of 0..{int(width) - 1}")
    return order


def checkpoint_method(checkpoint: Mapping[str, Any]) -> str:
    config = checkpoint["config"]
    method = str(config.get("method", ""))
    if not method:
        method = {"direct_fiedler": "teacher", "fiedler_mlp": "mlp"}.get(
            str(config.get("teacher_policy_mode", "")), ""
        )
    if method not in {"mlp", "teacher", "random", "canonical", "fixed", "fixed_curriculum"}:
        raise ValueError(f"unsupported checkpoint method {method!r}")
    return method


def validate_core_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    dataset_name: str,
    width: int,
    expected_policy: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Tuple[int, ...]]]:
    config = dict(checkpoint["config"])
    observed_dataset = str(config.get("dataset_name", "")).lower()
    if observed_dataset != str(dataset_name).lower():
        raise ValueError(f"expected a {dataset_name} checkpoint, got {observed_dataset!r}")
    method = checkpoint_method(checkpoint)
    if expected_policy is not None:
        expected_method = {"direct_fiedler": "teacher", "fiedler_mlp": "mlp"}.get(
            str(expected_policy), str(expected_policy)
        )
        if method != expected_method:
            raise ValueError(f"expected method {expected_method!r}, checkpoint contains {method!r}")
    # Random evaluation samples an order for each example or particle.
    if method == "random":
        return config, None
    if method in {"mlp", "teacher", "fixed_curriculum"}:
        frozen = checkpoint.get("order_frozen")
        if frozen is None:
            anneal_end = config.get("anneal_end")
            completed = int(checkpoint.get("iter_num", -1))
            if str(dataset_name).lower() in {"h3wb", "census"}:
                completed += 1
            frozen = anneal_end is not None and completed >= int(anneal_end)
        if not frozen:
            raise ValueError("final fixed-order evaluation requires a checkpoint after anneal_end with order_frozen=True")
    cached = checkpoint.get("cached_order")
    if cached is None:
        raise ValueError(f"{method} checkpoint is missing its saved cached_order")
    return config, validate_order(cached, int(width))


def evaluation_orders(
    order: Optional[Sequence[int]],
    batch_size: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if order is not None:
        return repeated_order(order, batch_size, device)
    return torch.rand(int(batch_size), int(width), device=device, generator=generator).argsort(dim=1)


def evaluation_generator(device: torch.device, seed: int) -> torch.Generator:
    target = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=target).manual_seed(int(seed))


def selected_checkpoints(args) -> Dict[str, Path]:
    selected = {name: value for name, value in {
        "teacher": args.teacher_checkpoint,
        "mlp": args.mlp_checkpoint,
    }.items() if value is not None}
    if args.checkpoint is not None:
        _, payload = load_checkpoint(args.checkpoint)
        method = checkpoint_method(payload)
        if method in selected:
            raise ValueError(f"method {method!r} was supplied more than once")
        selected[method] = args.checkpoint
    return selected


def repeated_order(
    order: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.tensor(tuple(order), dtype=torch.long, device=device)
        .view(1, -1)
        .expand(int(batch_size), -1)
    )


def resolve_device(name: str) -> torch.device:
    requested = str(name)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_autocast_dtype(
    name: str,
    device: torch.device,
) -> Optional[torch.dtype]:
    selected = str(name)
    if selected == "float32":
        return None
    if selected == "float16":
        if device.type != "cuda":
            raise ValueError("float16 evaluation requires CUDA")
        return torch.float16
    if selected == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {selected!r}")


def autocast_context(
    device: torch.device,
    dtype: Optional[torch.dtype],
):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


__all__ = [
    "REPO_ROOT",
    "autocast_context",
    "checkpoint_method",
    "evaluation_orders",
    "evaluation_generator",
    "selected_checkpoints",
    "jsonable",
    "load_checkpoint",
    "repeated_order",
    "resolve_autocast_dtype",
    "resolve_device",
    "resolve_repo_path",
    "sha256_file",
    "validate_core_checkpoint",
    "validate_order",
    "write_json",
]
