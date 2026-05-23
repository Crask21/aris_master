#!/usr/bin/env python3
"""Merge testing results.json files by evaluation_id search.

Usage example:
    python scripts/merge_results_by_eval_id.py \
        --test-dir /home/ap/cloud/master/aris_master/Testing \
        --evaluation-id Diff-Mix
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable


@dataclass
class ResultFile:
    evaluation_id: str
    file_path: Path
    payload: dict


def get_inquirer_module():
    try:
        module = import_module("inquirer")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'inquirer'. Install with: uv add inquirer "
            "or pip install inquirer"
        ) from exc

    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a testing directory for results.json files matching an evaluation_id "
            "pattern, then merge selected results into one master results.json."
        )
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Root testing directory to search recursively for results.json files.",
    )
    parser.add_argument(
        "--evaluation-id",
        required=True,
        help="Substring pattern used to match evaluation_id values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the merged results.json (prompted if omitted).",
    )
    return parser.parse_args()


def discover_results_files(test_dir: Path) -> list[ResultFile]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    discovered: list[ResultFile] = []

    for result_path in sorted(test_dir.rglob("results.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping invalid JSON: {result_path} ({exc})")
            continue

        evaluation_id = payload.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id.strip():
            print(f"[WARN] Missing evaluation_id in {result_path}")
            continue

        discovered.append(
            ResultFile(
                evaluation_id=evaluation_id.strip(),
                file_path=result_path,
                payload=payload,
            )
        )

    return discovered


def filter_results_by_evaluation_id(results: Iterable[ResultFile], pattern: str) -> list[ResultFile]:
    lowered = pattern.lower()
    return [item for item in results if lowered in item.evaluation_id.lower()]


def prompt_for_evaluation_ids(results: list[ResultFile]) -> list[ResultFile]:
    inquirer = get_inquirer_module()

    choices = [
        (f"{item.evaluation_id}  ({item.file_path.parent})", item.evaluation_id)
        for item in results
    ]
    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "selected_ids",
                message="Select one or more evaluation_id values",
                choices=choices,
            )
        ]
    )
    selected_ids = answers.get("selected_ids", []) if answers else []

    if not selected_ids:
        raise ValueError("No evaluation_id selected.")

    selected_set = set(selected_ids)
    return [item for item in results if item.evaluation_id in selected_set]


def prompt_for_output_dir(default_dir: Path | None) -> Path:
    inquirer = get_inquirer_module()

    default_text = str(default_dir) if default_dir else ""
    answers = inquirer.prompt(
        [
            inquirer.Text(
                "output_dir",
                message="Enter output directory for merged results.json",
                default=default_text,
            )
        ]
    )
    output_dir = answers.get("output_dir") if answers else None
    if not output_dir:
        raise ValueError("No output directory provided.")

    return Path(output_dir)


def prompt_for_new_evaluation_id(default_id: str) -> str:
    inquirer = get_inquirer_module()

    answers = inquirer.prompt(
        [
            inquirer.Text(
                "evaluation_id",
                message="Enter new evaluation_id for merged results",
                default=default_id,
            )
        ]
    )
    evaluation_id = answers.get("evaluation_id") if answers else None
    if not evaluation_id or not str(evaluation_id).strip():
        raise ValueError("No evaluation_id provided.")

    return str(evaluation_id).strip()


def extract_first_run(split: dict) -> dict | None:
    runs = split.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return None
    first = runs[0]
    return first if isinstance(first, dict) else None


def merge_splits(selected_results: list[ResultFile]) -> list[dict]:
    merged_by_name: dict[str, dict] = {}
    run_order: dict[str, list[dict]] = {}

    for item in selected_results:
        splits = item.payload.get("splits", [])
        if not isinstance(splits, list):
            print(f"[WARN] Skipping splits for {item.file_path} (invalid splits list)")
            continue

        for split in splits:
            if not isinstance(split, dict):
                print(f"[WARN] Skipping non-dict split in {item.file_path}")
                continue

            split_name = split.get("split_name")
            if not isinstance(split_name, str) or not split_name.strip():
                print(f"[WARN] Skipping split with missing split_name in {item.file_path}")
                continue

            first_run = extract_first_run(split)
            if first_run is None:
                print(f"[WARN] No valid runs for split '{split_name}' in {item.file_path}")
                continue

            if split_name not in merged_by_name:
                merged_split = {k: v for k, v in split.items() if k != "runs"}
                merged_split["runs"] = []
                merged_by_name[split_name] = merged_split
                run_order[split_name] = []

            run_order[split_name].append(first_run)

    merged: list[dict] = []
    for split_name, merged_split in merged_by_name.items():
        ordered_runs = run_order.get(split_name, [])
        normalized_runs: list[dict] = []
        for idx, run in enumerate(ordered_runs, start=1):
            normalized = dict(run)
            normalized["run_id"] = f"run{idx}"
            normalized_runs.append(normalized)
        merged_split["runs"] = normalized_runs
        merged.append(merged_split)

    return merged


def build_master_payload(
    selected_results: list[ResultFile],
    evaluation_id: str,
) -> dict:
    base = selected_results[0].payload

    master = {
        "evaluation_id": evaluation_id,
        "independent_variable": base.get("independent_variable"),
        "independent_variable_value": base.get("independent_variable_value"),
        "metrics": base.get("metrics", []),
        "splits": merge_splits(selected_results),
        "plots": base.get("plots", {}),
    }

    return master


def main() -> None:
    args = parse_args()

    discovered = discover_results_files(args.test_dir)
    if not discovered:
        raise SystemExit("No valid results.json files with evaluation_id were found.")

    filtered = filter_results_by_evaluation_id(discovered, args.evaluation_id)
    if not filtered:
        raise SystemExit(
            f"No results.json files matched evaluation_id pattern: {args.evaluation_id}"
        )

    selected_results = prompt_for_evaluation_ids(filtered)
    selected_results = sorted(selected_results, key=lambda item: item.evaluation_id.lower())

    new_evaluation_id = prompt_for_new_evaluation_id(args.evaluation_id)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = prompt_for_output_dir(args.test_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    master_payload = build_master_payload(selected_results, new_evaluation_id)
    output_path.write_text(json.dumps(master_payload, indent=2), encoding="utf-8")

    print(f"Merged {len(selected_results)} files into: {output_path}")


if __name__ == "__main__":
    main()
