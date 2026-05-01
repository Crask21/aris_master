"""Validate path-like values in config.json files.

The script scans one or more config.json files, walks nested dictionaries and
lists, and checks only fields whose names look like paths. Existing paths are
printed in green with a checkmark; missing paths are printed in yellow with a
warning.

Usage:
    python scripts/validate_config_paths.py
    python scripts/validate_config_paths.py path/to/config.json
    python scripts/validate_config_paths.py path/to/config_dir/
    python scripts/validate_config_paths.py --config-dir path/to/config_dir/
    python scripts/validate_config_paths.py --real-count path/to/config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE_BOLD = "\033[1;37m"
RESET = "\033[0m"
CHECK = "✓"
WARNING = "⚠"

REAL_COUNT_FIELDS = {
    "logging.output_dir",
    "data.synthetic_data_dir",
    "data.classes.impregnated_wood.data_dir",
    "data.classes.normal_wood.data_dir",
    "data.classes.soft_plastic.data_dir",
    "data.classes.hard_plastic.data_dir",
    "evaluation.evaluation_id",
}

REAL_COUNT_PATH_FIELDS = {
    "logging.output_dir",
    "data.synthetic_data_dir",
    "data.classes.impregnated_wood.data_dir",
    "data.classes.normal_wood.data_dir",
    "data.classes.soft_plastic.data_dir",
    "data.classes.hard_plastic.data_dir",
}

PATH_FIELD_NAMES = {
    "checkpoint",
    "checkpoints",
    "checkpoint_dir",
    "data_dir",
    "directory",
    "directories",
    "dir",
    "dirs",
    "file",
    "files",
    "folder",
    "folders",
    "path",
    "paths",
    "root",
    "roots",
    "source",
    "sources",
    "synthetic_data_dir",
    "train_data_dir",
    "output_dir",
    "output_path",
    "output_directory",
}


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def highlight_count(value: str, real_count: int, line_color: str) -> str:
    pattern = re.compile(rf"(?<!\d){re.escape(str(real_count))}(?!\d)")
    return pattern.sub(f"{WHITE_BOLD}{real_count}{RESET}{line_color}", value)


def normalize_field_name(field_path: str) -> str:
    return field_path.lower().replace(".", "_")


def looks_like_path_field(field_path: str) -> bool:
    parts = [part for part in normalize_field_name(field_path).split("_") if part]
    return any(part in PATH_FIELD_NAMES for part in parts)


def iter_config_files(inputs: list[str]) -> list[Path]:
    discovered: set[Path] = set()

    if not inputs:
        inputs = ["."]

    for raw_input in inputs:
        input_path = Path(raw_input)
        if input_path.is_file():
            if input_path.suffix.lower() == ".json":
                discovered.add(input_path.resolve())
            continue

        if input_path.is_dir():
            for config_path in input_path.rglob("*.json"):
                if config_path.is_file():
                    discovered.add(config_path.resolve())

    return sorted(discovered)


def validate_value(config_path: Path, field_path: str, value: str) -> tuple[bool, Path]:
    resolved_path = Path(value).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (config_path.parent / resolved_path).resolve()
    return resolved_path.exists(), resolved_path


def iter_leaf_values(node: object, prefix: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(node, dict):
        for key, value in node.items():
            field_path = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaf_values(value, field_path)
        return

    if isinstance(node, list):
        for index, item in enumerate(node):
            field_path = f"{prefix}[{index}]"
            yield from iter_leaf_values(item, field_path)
        return

    if prefix:
        yield prefix, node


def extract_real_count(config_path: Path) -> int | None:
    match = re.search(r"(\d+)_real", config_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def contains_exact_count(value: str, real_count: int) -> bool:
    return re.search(rf"(?<!\d){re.escape(str(real_count))}(?!\d)", value) is not None


def validate_real_count(config_path: Path, config_data: object, real_count: int) -> int:
    failure_count = 0

    for field_path, value in iter_leaf_values(config_data):
        if field_path not in REAL_COUNT_FIELDS:
            continue

        if field_path == "data.synthetic_data_dir" and value is None:
            continue

        if field_path in REAL_COUNT_PATH_FIELDS:
            if not isinstance(value, str):
                failure_count += 1
                print(
                    colorize(
                        f"{WARNING} {config_path.name} | {field_path} -> expected a string containing {real_count}, found {value!r}",
                        YELLOW,
                    )
                )
                continue

            resolved_path = Path(value).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = (config_path.parent / resolved_path).resolve()

            line_color = GREEN if resolved_path.exists() and contains_exact_count(value, real_count) else YELLOW
            has_count = contains_exact_count(value, real_count)
            if not resolved_path.exists() or not has_count:
                failure_count += 1

            count_value = highlight_count(value, real_count, line_color)
            if resolved_path.exists() and has_count:
                print(f"{line_color}{CHECK} {field_path} -> {count_value}{RESET}")
            else:
                print(f"{line_color}{WARNING} {config_path.name} | {field_path} -> {count_value}{RESET}")
            continue

        if not isinstance(value, str):
            failure_count += 1
            print(
                colorize(
                    f"{WARNING} {config_path.name} | {field_path} -> expected a string containing {real_count}, found {value!r}",
                    YELLOW,
                )
            )
            continue

        if contains_exact_count(value, real_count):
            print(f"{GREEN}{CHECK} {field_path} -> {highlight_count(value, real_count, GREEN)}{RESET}")
        else:
            failure_count += 1
            print(f"{YELLOW}{WARNING} {config_path.name} | {field_path} -> {highlight_count(value, real_count, YELLOW)}{RESET}")

    return failure_count


def walk_config(config_path: Path, node: object, prefix: str = "") -> Iterable[tuple[str, str, bool, Path]]:
    if isinstance(node, dict):
        for key, value in node.items():
            field_path = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_config(config_path, value, field_path)
        return

    if isinstance(node, list):
        for index, item in enumerate(node):
            field_path = f"{prefix}[{index}]"
            if isinstance(item, (dict, list)):
                yield from walk_config(config_path, item, field_path)
            elif isinstance(item, str) and looks_like_path_field(prefix):
                exists, resolved_path = validate_value(config_path, field_path, item)
                yield field_path, item, exists, resolved_path
        return

    if isinstance(node, str) and prefix and looks_like_path_field(prefix):
        exists, resolved_path = validate_value(config_path, prefix, node)
        yield prefix, node, exists, resolved_path


def validate_config(config_path: Path, validate_real_count_flag: bool = False) -> int:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    except json.JSONDecodeError as exc:
        print(colorize(f"{WARNING} {config_path}: invalid JSON ({exc})", YELLOW), file=sys.stderr)
        return 1

    failure_count = 0

    for field_path, raw_value, exists, resolved_path in walk_config(config_path, config_data):
        if validate_real_count_flag and field_path in REAL_COUNT_FIELDS:
            continue
        if exists:
            print(colorize(f"{CHECK} {field_path} -> {resolved_path}", GREEN))
        else:
            failure_count += 1
            print(colorize(f"{WARNING} {field_path} -> {raw_value}", YELLOW))

    if validate_real_count_flag:
        real_count = extract_real_count(config_path)
        if real_count is None:
            print(
                colorize(
                    f"{WARNING} {config_path.name}: could not extract real count from filename",
                    YELLOW,
                ),
                file=sys.stderr,
            )
            failure_count += 1
        else:
            failure_count += validate_real_count(config_path, config_data, real_count)

    return 1 if failure_count else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate path fields in JSON config files.")
    parser.add_argument(
        "--config-dir",
        dest="config_dirs",
        action="append",
        default=[],
        help="Directory to scan recursively for config.json files. Can be passed multiple times.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional JSON config files or directories to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--real-count",
        action="store_true",
        help="Validate that the filename-derived real count appears in selected config fields.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_files = iter_config_files([*args.paths, *args.config_dirs])
    if not config_files:
        print(colorize("No config.json files found.", YELLOW), file=sys.stderr)
        return 1

    exit_code = 0
    for config_path in config_files:
        print("-" * 80)
        print(colorize(f"Validating {config_path}...", GREEN))
        exit_code |= validate_config(config_path, validate_real_count_flag=args.real_count)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())