from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence


SUPPORTED_DATASETS = (
    "wikitext103",
    "openwebtext",
    "h3wb",
    "census",
)

_RUNNERS = {
    "wikitext103": ("ordergpt.language_training", False),
    "openwebtext": ("ordergpt.language_training", False),
    "h3wb": ("aligned_continuous.training", True),
    "census": ("aligned_tabular.training", True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Train any supported OrderGPT task from one flat Python "
            "configuration. The configuration's dataset_name selects the "
            "language, H3WB, or CENSUS runtime."
        ),
        epilog=(
            "examples:\n"
            "  python train.py config/wikitext103/teacher.py\n"
            "  python train.py config/h3wb/mlp.py --device=cuda:1\n"
            "  python train.py config/census/teacher.py\n"
            "  torchrun --standalone --nproc_per_node=6 "
            "train.py config/openwebtext/mlp.py\n\n"
            "supported datasets: " + ", ".join(SUPPORTED_DATASETS)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "config",
        metavar="CONFIG.py",
        help="flat experiment configuration",
    )
    return parser


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ValueError(f"configuration not found: {path}")
    return path.resolve()


def _assigned_dataset_node(statement: ast.stmt) -> Optional[ast.expr]:
    if isinstance(statement, ast.Assign):
        if any(
            isinstance(target, ast.Name) and target.id == "dataset_name"
            for target in statement.targets
        ):
            return statement.value
    elif (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "dataset_name"
    ):
        return statement.value
    return None


def read_declared_dataset(config_path: Path) -> str:
    try:
        source = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"cannot read configuration {config_path}: {error}"
        ) from error
    try:
        tree = ast.parse(source, filename=str(config_path))
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno else "unknown line"
        raise ValueError(
            f"cannot parse configuration {config_path} at {location}: "
            f"{error.msg}"
        ) from error

    value: object = None
    found = False
    for statement in tree.body:
        value_node = _assigned_dataset_node(statement)
        if value_node is None:
            continue
        found = True
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                f"dataset_name in {config_path} must be a string literal"
            ) from error

    if not found:
        raise ValueError(
            f"configuration {config_path} must declare dataset_name"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"dataset_name in {config_path} must be a non-empty string"
        )
    return value.strip().lower()


def validate_overrides(overrides: Sequence[str]) -> None:
    for override in overrides:
        if not override.startswith("--") or "=" not in override[2:]:
            raise ValueError(
                "configuration overrides must have form --field=value; "
                f"got {override!r}"
            )
        key, _ = override[2:].split("=", 1)
        if not key:
            raise ValueError(f"configuration override has no field: {override!r}")


def resolve_dataset(
    declared_dataset: str,
    overrides: Sequence[str],
) -> str:


    dataset_name = declared_dataset
    for override in overrides:
        key, raw_value = override[2:].split("=", 1)
        if key != "dataset_name":
            continue
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("--dataset_name must be a non-empty string")
        dataset_name = value.strip().lower()

    if dataset_name not in _RUNNERS:
        raise ValueError(
            f"unsupported dataset_name {dataset_name!r}; expected one of: "
            + ", ".join(SUPPORTED_DATASETS)
        )
    return dataset_name


def dispatch(dataset_name: str, forwarded_argv: Sequence[str]) -> None:
    module_name, pass_dataset_name = _RUNNERS[dataset_name]
    runner = importlib.import_module(module_name).main
    kwargs: dict[str, object] = {"argv": list(forwarded_argv)}
    if pass_dataset_name:
        kwargs["dataset_name"] = dataset_name
    runner(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args(
        sys.argv[1:] if argv is None else list(argv)
    )
    try:
        validate_overrides(overrides)
        config_path = resolve_config_path(args.config)
        declared_dataset = read_declared_dataset(config_path)
        dataset_name = resolve_dataset(declared_dataset, overrides)
    except ValueError as error:
        parser.error(str(error))

    dispatch(dataset_name, [str(config_path), *overrides])


if __name__ == "__main__":
    main()
