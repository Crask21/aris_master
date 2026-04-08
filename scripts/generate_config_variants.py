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
        required=True,
        help=(
            "Dot-path fields to update (e.g. logging.output_dir data.normal_wood.data_dir). "
            "For class shorthand, data.<class>.x is auto-resolved to data.classes.<class>.x."
        ),
    )
    parser.add_argument(
        "--change_filename",
        action="store_true",
        help="Apply the same find/replace on the output filename.",
    )
    parser.add_argument(
        "--find",
        required=True,
        help="Substring to find in the selected fields.",
    )
    parser.add_argument(
        "--replace",
        nargs="+",
        required=True,
        help="One or more replacement values. One config is generated for each value.",
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


def main() -> int:
    args = parse_args()

    if not args.template.exists():
        raise FileNotFoundError(f"Template file not found: {args.template}")

    output_dir = args.output_dir if args.output_dir is not None else args.template.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.template.open("r", encoding="utf-8") as f:
        template_data = json.load(f)

    resolved_change_paths: list[tuple[str, List[str]]] = []
    for raw_path in args.change:
        resolved_change_paths.append((raw_path, resolve_path(raw_path, template_data)))

    generated_paths: list[Path] = []
    for replacement in args.replace:
        data_variant = copy.deepcopy(template_data)

        for raw_path, keys in resolved_change_paths:
            original_value = get_nested(data_variant, keys)
            if not isinstance(original_value, str):
                joined = ".".join(keys)
                raise TypeError(
                    f"Field '{raw_path}' (resolved to '{joined}') is not a string: "
                    f"{type(original_value).__name__}"
                )
            new_value = original_value.replace(args.find, replacement)
            set_nested(data_variant, keys, new_value)

        output_name = build_output_filename(
            template_path=args.template,
            find_text=args.find,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
