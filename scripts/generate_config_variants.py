#!/usr/bin/env python3
"""Generate JSON config variants by find/replace on selected fields.

Example:
  python scripts/generate_config_variants.py \
    queue/planning/Synth_to_Real_ratio_tests/500_real/baseline_500_real.json \
    --change data.normal_wood.data_dir logging.output_dir \
    --change_filename \
    --find 500 \
    --replace 1000 250
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
from importlib import import_module
from pathlib import Path
from typing import Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one new JSON config per --replace value by applying find/replace "
            "on selected fields."
        )
    )
    parser.add_argument(
        "template",
        type=Path,
        help="Path to the template JSON config.",
    )
    parser.add_argument(
        "--change",
        nargs="+",
        default=None,
        help=(
            "Dot-path fields to update (e.g. logging.output_dir data.normal_wood.data_dir). "
            "For class shorthand, data.<class>.x is auto-resolved to data.classes.<class>.x."
        ),
    )
    parser.add_argument(
        "--change_filename",
        action="store_false",
        help="Apply the same find/replace on the output filename.",
    )
    parser.add_argument(
        "--find",
        default=None,
        help="Substring to find in the selected fields.",
    )
    parser.add_argument(
        "--replace",
        nargs="+",
        default=None,
        help="One or more replacement values. One config is generated for each value.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt interactively for fields/find/replace selections.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for generated configs. Defaults to the template file directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated file if it already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview output without writing files.",
    )
    return parser.parse_args()


def get_inquirer_module():
    try:
        module = import_module("inquirer")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'inquirer'. Install with: uv add inquirer "
            "or pip install inquirer"
        ) from exc

    return module


def collect_leaf_paths(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            leaves.extend(collect_leaf_paths(value, path))
        return leaves

    if isinstance(data, list):
        # Lists are treated as leaf values for this find/replace workflow.
        leaves.append((prefix, data))
        return leaves

    leaves.append((prefix, data))
    return leaves


def prompt_interactive_inputs(
    template_data: dict[str, Any],
    change_from_flags: list[str] | None,
    find_from_flags: str | None,
    replace_from_flags: list[str] | None,
) -> tuple[list[str], str, list[str]]:
    inquirer = get_inquirer_module()

    leaf_paths = collect_leaf_paths(template_data)
    if not leaf_paths:
        raise ValueError("Template config has no leaf fields to select.")

    change_paths = change_from_flags
    if not change_paths:
        choices = [path for path, _ in leaf_paths]
        answers = inquirer.prompt(
            [
                inquirer.Checkbox(
                    "selected_fields",
                    message="Select one or more fields to change",
                    choices=choices,
                )
            ]
        )
        selected = answers.get("selected_fields", []) if answers else []
        if not selected:
            raise ValueError("No fields selected.")
        change_paths = selected

    find_text = find_from_flags
    if find_text is None:
        answers = inquirer.prompt(
            [
                inquirer.Text(
                    "find_text",
                    message="Find text (substring)",
                )
            ]
        )
        find_text = (answers.get("find_text") if answers else "") or ""

    replace_values = replace_from_flags
    if not replace_values:
        replace_values = []
        while True:
            answers = inquirer.prompt(
                [
                    inquirer.Text(
                        "replace_text",
                        message="Replacement value (val/N)",
                    )
                ]
            )
            replacement = ((answers.get("replace_text") if answers else "") or "").strip()

            if replacement.lower() in {"n", "no"}:
                break

            if replacement == "":
                continue

            replace_values.append(replacement)

        if not replace_values:
            raise ValueError("At least one replacement value must be provided.")

    return (change_paths, find_text, replace_values)


def path_exists(data: Any, keys: List[str]) -> bool:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def resolve_path(raw_path: str, template_data: dict[str, Any]) -> List[str]:
    keys = raw_path.split(".")
    if path_exists(template_data, keys):
        return keys

    # Convenience shorthand: data.<class>.* -> data.classes.<class>.*
    if len(keys) >= 2 and keys[0] == "data":
        expanded = ["data", "classes", *keys[1:]]
        if path_exists(template_data, expanded):
            return expanded

    raise KeyError(f"Field not found: {raw_path}")


def get_nested(data: dict[str, Any], keys: List[str]) -> Any:
    current: Any = data
    for key in keys:
        current = current[key]
    return current


def set_nested(data: dict[str, Any], keys: List[str], value: Any) -> None:
    current: Any = data
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def build_output_filename(
    template_path: Path,
    find_text: str,
    replace_text: str,
    change_filename: bool,
) -> str:
    if change_filename:
        new_name = template_path.name.replace(find_text, replace_text)
        if new_name != template_path.name:
            return new_name

    # Fallback name when filename is unchanged or --change_filename is not set.
    return f"{template_path.stem}_{replace_text}{template_path.suffix}"


def build_rerun_command(
    template: Path,
    change_paths: list[str],
    find_text: str,
    replace_values: list[str],
    change_filename: bool,
    output_dir: Path | None,
    overwrite: bool,
    dry_run: bool,
) -> str:
    parts: list[str] = [
        "python3",
        Path(__file__).name,
        shlex.quote(str(template)),
        "--change",
    ]

    parts.extend(shlex.quote(path) for path in change_paths)
    parts.extend(["--find", shlex.quote(find_text), "--replace"])
    parts.extend(shlex.quote(value) for value in replace_values)

    if change_filename:
        parts.append("--change_filename")
    if output_dir is not None:
        parts.extend(["--output_dir", shlex.quote(str(output_dir))])
    if overwrite:
        parts.append("--overwrite")
    if dry_run:
        parts.append("--dry_run")

    return " ".join(parts)


def main() -> int:
    args = parse_args()

    if not args.template.exists():
        raise FileNotFoundError(f"Template file not found: {args.template}")

    output_dir = args.output_dir if args.output_dir is not None else args.template.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.template.open("r", encoding="utf-8") as f:
        template_data = json.load(f)

    change_paths = args.change
    find_text = args.find
    replace_values = args.replace

    if args.interactive:
        change_paths, find_text, replace_values = prompt_interactive_inputs(
            template_data=template_data,
            change_from_flags=change_paths,
            find_from_flags=find_text,
            replace_from_flags=replace_values,
        )

    if not change_paths:
        raise ValueError("--change is required unless provided through --interactive prompts.")
    if find_text is None:
        raise ValueError("--find is required unless provided through --interactive prompts.")
    if find_text == "":
        raise ValueError("Find text must be a non-empty string.")
    if not replace_values:
        raise ValueError("--replace is required unless provided through --interactive prompts.")

    resolved_change_paths: list[tuple[str, List[str]]] = []
    for raw_path in change_paths:
        resolved_change_paths.append((raw_path, resolve_path(raw_path, template_data)))

    generated_paths: list[Path] = []
    for replacement in replace_values:
        data_variant = copy.deepcopy(template_data)

        for raw_path, keys in resolved_change_paths:
            original_value = get_nested(data_variant, keys)
            if not isinstance(original_value, str):
                joined = ".".join(keys)
                raise TypeError(
                    f"Field '{raw_path}' (resolved to '{joined}') is not a string: "
                    f"{type(original_value).__name__}"
                )
            new_value = original_value.replace(find_text, replacement)
            set_nested(data_variant, keys, new_value)

        output_name = build_output_filename(
            template_path=args.template,
            find_text=find_text,
            replace_text=replacement,
            change_filename=args.change_filename,
        )
        output_path = output_dir / output_name

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )

        generated_paths.append(output_path)
        if args.dry_run:
            continue

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data_variant, f, indent=2)
            f.write("\n")

    for path in generated_paths:
        prefix = "[dry-run] Would write" if args.dry_run else "Wrote"
        print(f"{prefix}: {path}")

    rerun_command = build_rerun_command(
        template=args.template,
        change_paths=change_paths,
        find_text=find_text,
        replace_values=replace_values,
        change_filename=args.change_filename,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print("\nRe-run command (non-interactive):")
    print(rerun_command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
