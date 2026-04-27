#!/usr/bin/env python3
"""Interactive summary tables for testing results.

Usage example:
    python scripts/summarize_training/interactive_results_summary.py \
        --test-dir /home/ap/cloud/master/aris_master/Testing \
        --sort alpha \
        --error std
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable
from scipy.stats import t
import numpy as np

ALLOWED_METRICS = [
    "val_accuracy",
    "val_f1_score",
    "test_accuracy",
    "test_f1_score",
]

REAL_IMAGES_PER_CLASS_DIVISOR = 4


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


def get_tabulate_function():
    try:
        module = import_module("tabulate")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'tabulate'. Install with: uv add tabulate "
            "or pip install tabulate"
        ) from exc

    return module.tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a testing directory for results.json files, select evaluation IDs and "
            "split names interactively, then print summary tables."
        )
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Root testing directory to search recursively for results.json files.",
    )
    parser.add_argument(
        "--sort",
        choices=["alpha", "date"],
        default="alpha",
        help="Sort evaluation IDs alphabetically or by date found in evaluation_id.",
    )
    parser.add_argument(
        "--error",
        choices=["std", "ci95"],
        default="std",
        help="Error term displayed after mean: standard deviation or 95%% confidence interval.",
    )
    parser.add_argument(
        "--metric",
        "--metrics",
        dest="metrics",
        action="append",
        nargs="+",
        choices=ALLOWED_METRICS,
        default=None,
        help=(
            "Metric(s) to display. If omitted, metrics are selected interactively. "
            "Can be repeated, e.g. --metric val_accuracy --metric test_f1_score"
        ),
    )
    parser.add_argument(
        "--baseline-evaluation-id",
        "--baseline-evaluation-ids",
        dest="baseline_evaluation_ids",
        action="append",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional baseline evaluation_id value(s). Can be repeated. In expansion ratio mode, "
            "multiple baselines can be provided (for different real image counts)."
        ),
    )
    parser.add_argument(
        "--evaluation-id",
        "--evaluation-ids",
        dest="evaluation_ids",
        action="append",
        nargs="+",
        default=None,
        help=(
            "DA method evaluation_id values to use without interactive selection. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--split-name",
        "--split-names",
        dest="split_names",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Split names to use without interactive split selection. Can be repeated."
        ),
    )
    parser.add_argument(
        "--filter",
        "-f",
        dest="evaluation_id_filter",
        type=str,
        default=None,
        help=(
            "Optional comma-separated keywords to filter interactive evaluation_id choices, "
            "e.g. --filter '1000_real, lr_scheduler'."
        ),
    )
    parser.add_argument(
        "--use-all-samples",
        action="store_true",
        help=(
            "Use all available run samples per method. If omitted, DA methods are "
            "aligned to the same sample count per split/metric by truncating to the "
            "minimum count (run1..runN)."
        ),
    )
    parser.add_argument(
        "--to_latex",
        "--to-latex",
        dest="to_latex",
        action="store_true",
        help="Output summary tables as LaTeX table figures.",
    )
    parser.add_argument(
        "--expansion_ratio",
        "--expansion-ratio",
        dest="expansion_ratio",
        action="store_true",
        help=(
            "Output LaTeX tables pivoted by real dataset size (rows) and expansion ratio "
            "columns derived from synthetic/real image counts."
        ),
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help=(
            "Optional JSON export path for generated tables, e.g. "
            "--export /path/to/output-dir/table_export.json"
        ),
    )
    return parser.parse_args()


def discover_results_files(test_dir: Path) -> list[ResultFile]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    discovered: list[ResultFile] = []

    for result_path in sorted(test_dir.rglob("results.json")):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping invalid JSON file: {result_path} ({exc})")
            continue

        evaluation_id = payload.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id.strip():
            continue

        discovered.append(
            ResultFile(
                evaluation_id=evaluation_id.strip(),
                file_path=result_path,
                payload=payload,
            )
        )

    return discovered


def extract_sort_datetime(evaluation_id: str) -> datetime | None:
    patterns = [
        # Example: 2026-04-06T02-42
        re.compile(r"(\d{4})-(\d{2})-(\d{2})[T_](\d{2})[-:](\d{2})"),
        # Example: 04-06T02-42 (year assumed to current year)
        re.compile(r"(\d{2})-(\d{2})[T_](\d{2})[-:](\d{2})"),
    ]

    now = datetime.now()

    for i, pattern in enumerate(patterns):
        match = pattern.search(evaluation_id)
        if not match:
            continue

        try:
            if i == 0:
                year, month, day, hour, minute = map(int, match.groups())
            else:
                month, day, hour, minute = map(int, match.groups())
                year = now.year

            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    return None


def sort_results(results: list[ResultFile], mode: str) -> list[ResultFile]:
    if mode == "alpha":
        return sorted(results, key=lambda r: r.evaluation_id.lower())

    def date_key(item: ResultFile) -> tuple[int, datetime, str]:
        parsed = extract_sort_datetime(item.evaluation_id)
        if parsed is None:
            return (1, datetime.max, item.evaluation_id.lower())
        return (0, parsed, item.evaluation_id.lower())

    return sorted(results, key=date_key)


def ensure_unique_evaluation_ids(results: list[ResultFile]) -> list[ResultFile]:
    unique: dict[str, ResultFile] = {}
    duplicates: dict[str, list[Path]] = {}

    for item in results:
        if item.evaluation_id in unique:
            duplicates.setdefault(item.evaluation_id, [unique[item.evaluation_id].file_path]).append(
                item.file_path
            )
            continue
        unique[item.evaluation_id] = item

    if duplicates:
        print("[WARN] Duplicate evaluation_id values found. Keeping first occurrence for each duplicate:")
        for eval_id, paths in sorted(duplicates.items()):
            print(f"  - {eval_id}")
            for path in paths:
                print(f"    {path}")

    return list(unique.values())


def prompt_for_evaluation_ids(results: list[ResultFile]) -> list[ResultFile]:
    inquirer = get_inquirer_module()

    choices = [item.evaluation_id for item in results]
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


def parse_filter_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []

    keywords: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        keyword = token.strip()
        if not keyword:
            continue
        normalized = keyword.lower()
        if normalized in seen:
            continue
        keywords.append(keyword)
        seen.add(normalized)

    return keywords


def filter_results_by_evaluation_id_keywords(
    results: list[ResultFile],
    keywords: list[str],
) -> list[ResultFile]:
    if not keywords:
        return list(results)

    lowered_keywords = [keyword.lower() for keyword in keywords]
    return [
        item
        for item in results
        if any(keyword in item.evaluation_id.lower() for keyword in lowered_keywords)
    ]


def prompt_for_evaluation_id_filter_keywords() -> list[str]:
    inquirer = get_inquirer_module()

    use_filter_answers = inquirer.prompt(
        [
            inquirer.Confirm(
                "use_filter",
                message="Filter evaluation_id choices with comma-separated keywords?",
                default=False,
            )
        ]
    )
    use_filter = bool(use_filter_answers.get("use_filter")) if use_filter_answers else False
    if not use_filter:
        return []

    keyword_answers = inquirer.prompt(
        [
            inquirer.Text(
                "keywords",
                message="Enter comma-separated keywords",
            )
        ]
    )
    raw_keywords = keyword_answers.get("keywords", "") if keyword_answers else ""
    return parse_filter_keywords(raw_keywords)


def prompt_for_baseline_evaluation_id(
    results: list[ResultFile],
    selected_results: list[ResultFile],
    baseline_evaluation_id: str | None,
    filter_keywords: list[str] | None = None,
) -> ResultFile:
    inquirer = get_inquirer_module()

    selected_ids = {item.evaluation_id for item in selected_results}

    by_id = {item.evaluation_id: item for item in results}
    if baseline_evaluation_id:
        if baseline_evaluation_id not in by_id:
            raise ValueError(
                f"Baseline evaluation_id '{baseline_evaluation_id}' was not found among discovered results."
            )
        if baseline_evaluation_id in selected_ids:
            raise ValueError(
                "Baseline evaluation_id must be different from selected DA method evaluation_id values."
            )
        return by_id[baseline_evaluation_id]

    candidate_results = [item for item in results if item.evaluation_id not in selected_ids]
    if filter_keywords:
        candidate_results = filter_results_by_evaluation_id_keywords(candidate_results, filter_keywords)
    if not candidate_results:
        raise ValueError("No candidate baseline evaluation_id remains after DA method selection.")

    choices = [
        (f"{item.evaluation_id}  ({item.file_path.parent})", item.evaluation_id)
        for item in candidate_results
    ]
    answers = inquirer.prompt(
        [
            inquirer.List(
                "baseline_id",
                message="Select baseline evaluation_id",
                choices=choices,
            )
        ]
    )
    baseline_id = answers.get("baseline_id") if answers else None
    if not baseline_id:
        raise ValueError("No baseline evaluation_id selected.")

    return by_id[baseline_id]


def extract_split_names(item: ResultFile) -> list[str]:
    splits = item.payload.get("splits", [])
    names: list[str] = []

    if not isinstance(splits, list):
        return names

    for split in splits:
        if not isinstance(split, dict):
            continue
        split_name = split.get("split_name")
        if isinstance(split_name, str) and split_name.strip():
            names.append(split_name.strip())

    return names


def split_real_image_count(item: ResultFile, split_name: str) -> int | None:
    split = find_split(item, split_name)
    if split is None:
        return None

    value = split.get("real_image_count")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def split_synthetic_image_count(item: ResultFile, split_name: str) -> int | None:
    split = find_split(item, split_name)
    if split is None:
        return None

    value = split.get("synthetic_image_count")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def split_image_counts(item: ResultFile, split_name: str) -> tuple[int, int] | None:
    real_count = split_real_image_count(item, split_name)
    synthetic_count = split_synthetic_image_count(item, split_name)
    if real_count is None or synthetic_count is None:
        return None
    return (real_count, synthetic_count)


def expansion_split_metadata(selected_results: list[ResultFile]) -> dict[str, tuple[int, int, float]]:
    metadata: dict[str, tuple[int, int, float]] = {}

    for item in selected_results:
        splits = item.payload.get("splits", [])
        if not isinstance(splits, list):
            continue

        for split in splits:
            if not isinstance(split, dict):
                continue

            split_name = split.get("split_name")
            real_count = split.get("real_image_count")
            synthetic_count = split.get("synthetic_image_count")

            if not isinstance(split_name, str) or not split_name.strip():
                continue
            if not isinstance(real_count, (int, float)):
                continue
            if not isinstance(synthetic_count, (int, float)):
                continue

            real_int = int(real_count)
            synthetic_int = int(synthetic_count)
            if real_int <= 0:
                continue

            split_name = split_name.strip()
            ratio = synthetic_int / real_int
            candidate = (real_int, synthetic_int, ratio)

            existing = metadata.get(split_name)
            if existing is not None and existing[:2] != candidate[:2]:
                raise ValueError(
                    f"Split '{split_name}' has conflicting image counts across selected DA methods: "
                    f"{existing[:2]} vs {(real_int, synthetic_int)}."
                )

            metadata[split_name] = candidate

    return metadata


def get_expansion_selectable_split_choices(
    selected_results: list[ResultFile],
) -> list[tuple[str, str]]:
    metadata = expansion_split_metadata(selected_results)
    ordered = sorted(
        metadata.items(),
        key=lambda item: (item[1][0], item[1][2], item[0].lower()),
    )

    choices: list[tuple[str, str]] = []
    for split_name, (real_count, synthetic_count, ratio) in ordered:
        label = (
            f"{split_name}  [real={real_count}, synthetic={synthetic_count}, "
            f"x({format_compact_number(ratio)}+1)]"
        )
        choices.append((label, split_name))

    return choices


def prompt_for_expansion_split_names(
    selected_results: list[ResultFile],
) -> list[str]:
    inquirer = get_inquirer_module()

    choices = get_expansion_selectable_split_choices(selected_results)
    if not choices:
        raise ValueError("No selectable split_name values found in selected DA methods.")

    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "selected_splits",
                message="Select one or more split_name values",
                choices=choices,
            )
        ]
    )
    selected_splits = answers.get("selected_splits", []) if answers else []

    if not selected_splits:
        raise ValueError("No split_name selected.")

    return selected_splits


def method_name_for_expansion(evaluation_id: str) -> str:
    match = re.search(r"_\d+_real_(.+)$", evaluation_id)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return evaluation_id


def shared_split_image_counts(
    selected_results: list[ResultFile],
    split_name: str,
) -> tuple[int, int] | None:
    real_counts: set[int] = set()
    synthetic_counts: set[int] = set()

    for item in selected_results:
        real_count = split_real_image_count(item, split_name)
        synthetic_count = split_synthetic_image_count(item, split_name)
        if real_count is None or synthetic_count is None:
            return None
        real_counts.add(real_count)
        synthetic_counts.add(synthetic_count)

    if len(real_counts) != 1 or len(synthetic_counts) != 1:
        return None

    return (next(iter(real_counts)), next(iter(synthetic_counts)))


def shared_real_image_count_for_split(selected_results: list[ResultFile], split_name: str) -> int | None:
    counts: set[int] = set()
    for item in selected_results:
        count = split_real_image_count(item, split_name)
        if count is None:
            return None
        counts.add(count)

    if len(counts) != 1:
        return None
    return next(iter(counts))


def choose_baseline_split_for_real_count(
    baseline_result: ResultFile,
    real_image_count: int,
) -> dict | None:
    splits = baseline_result.payload.get("splits", [])
    if not isinstance(splits, list):
        return None

    matching: list[dict] = []
    for split in splits:
        if not isinstance(split, dict):
            continue
        real_count = split.get("real_image_count")
        if not isinstance(real_count, (int, float)):
            continue
        if int(real_count) != int(real_image_count):
            continue
        matching.append(split)

    if not matching:
        return None

    def baseline_sort_key(split: dict) -> tuple[float, str]:
        synthetic = split.get("synthetic_image_count")
        synthetic_key = float(synthetic) if isinstance(synthetic, (int, float)) else float("inf")
        split_name = split.get("split_name")
        name_key = split_name if isinstance(split_name, str) else ""
        return (synthetic_key, name_key)

    matching.sort(key=baseline_sort_key)
    return matching[0]


def prompt_for_split_names(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
) -> list[str]:
    inquirer = get_inquirer_module()

    split_sets = [set(extract_split_names(item)) for item in selected_results]

    if len(selected_results) == 1:
        candidate_splits = sorted(split_sets[0])
    else:
        candidate_splits = sorted(set.intersection(*split_sets)) if split_sets else []

    if not candidate_splits:
        raise ValueError("No common split_name values found across selected evaluation_id files.")

    filtered_choices = get_selectable_split_choices(selected_results, baseline_result)
    
    if not filtered_choices:
        raise ValueError(
            "No selectable split_name remains after enforcing shared real_image_count with baseline."
        )

    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "selected_splits",
                message="Select one or more split_name values",
                choices=filtered_choices,
            )
        ]
    )
    selected_splits = answers.get("selected_splits", []) if answers else []

    if not selected_splits:
        raise ValueError("No split_name selected.")

    return selected_splits


def get_selectable_split_choices(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
) -> list[tuple[str, str]]:
    split_sets = [set(extract_split_names(item)) for item in selected_results]

    if len(selected_results) == 1:
        candidate_splits = sorted(split_sets[0])
    else:
        candidate_splits = sorted(set.intersection(*split_sets)) if split_sets else []

    if not candidate_splits:
        return []

    filtered_choices: list[tuple[str, str]] = []
    for split_name in candidate_splits:
        real_count = shared_real_image_count_for_split(selected_results, split_name)
        if real_count is None:
            continue

        baseline_split = choose_baseline_split_for_real_count(baseline_result, real_count)
        if baseline_split is None:
            continue

        baseline_split_name = baseline_split.get("split_name")
        if not isinstance(baseline_split_name, str):
            continue

        label = (
            f"{split_name}  [real={real_count}]  "
            f"(baseline split: {baseline_split_name})"
        )
        filtered_choices.append((label, split_name))

    return filtered_choices


def build_baseline_split_name_map(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
    selected_splits: list[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for split_name in selected_splits:
        real_count = shared_real_image_count_for_split(selected_results, split_name)
        if real_count is None:
            raise ValueError(
                f"Selected split '{split_name}' does not have a shared real_image_count across DA methods."
            )

        baseline_split = choose_baseline_split_for_real_count(baseline_result, real_count)
        if baseline_split is None:
            raise ValueError(
                f"No baseline split found with real_image_count={real_count} for selected split '{split_name}'."
            )

        baseline_split_name = baseline_split.get("split_name")
        if not isinstance(baseline_split_name, str) or not baseline_split_name:
            raise ValueError(
                f"Baseline split for selected split '{split_name}' is missing a valid split_name."
            )

        mapping[split_name] = baseline_split_name

    return mapping


def find_split(item: ResultFile, split_name: str) -> dict | None:
    splits = item.payload.get("splits", [])
    if not isinstance(splits, list):
        return None

    for split in splits:
        if isinstance(split, dict) and split.get("split_name") == split_name:
            return split

    return None


def _ordered_metric_values_from_runs(runs: list[dict], metric: str) -> list[float]:
    run_id_pattern = re.compile(r"^run(\d+)$", re.IGNORECASE)

    numeric_runs: dict[int, float] = {}
    fallback_runs: list[tuple[int, float]] = []

    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue

        raw_value = run.get(metric)
        if not isinstance(raw_value, (int, float)):
            continue

        run_id = run.get("run_id")
        if isinstance(run_id, str):
            match = run_id_pattern.fullmatch(run_id.strip())
        else:
            match = None

        value = float(raw_value)
        if match:
            run_index = int(match.group(1))
            # Keep first occurrence for duplicated run labels.
            if run_index not in numeric_runs:
                numeric_runs[run_index] = value
        else:
            fallback_runs.append((idx, value))

    ordered_values = [numeric_runs[k] for k in sorted(numeric_runs)]
    ordered_values.extend(v for _, v in sorted(fallback_runs, key=lambda t: t[0]))
    return ordered_values


def metric_values_for_split(
    item: ResultFile,
    split_name: str,
    metric: str,
    sample_limit: int | None = None,
) -> list[float]:
    split = find_split(item, split_name)
    if split is None:
        return []

    runs = split.get("runs", [])
    if not isinstance(runs, list):
        return []

    values = _ordered_metric_values_from_runs(runs, metric)
    if sample_limit is not None:
        return values[: max(sample_limit, 0)]
    return values


def da_sample_limit_for_split_metric(
    selected_results: list[ResultFile],
    split_name: str,
    metric: str,
) -> int:
    counts = [
        len(metric_values_for_split(item, split_name, metric))
        for item in selected_results
    ]
    if not counts:
        return 0
    return min(counts)

    return values


def da_sample_limit_for_split_metric_present(
    selected_results: list[ResultFile],
    split_name: str,
    metric: str,
) -> int:
    counts: list[int] = []
    for item in selected_results:
        values = metric_values_for_split(item, split_name, metric)
        if values:
            counts.append(len(values))

    if not counts:
        return 0
    return min(counts)


def common_metrics_for_split(selected_results: list[ResultFile], split_name: str) -> list[str]:
    per_file_metrics: list[set[str]] = []

    for item in selected_results:
        available = {
            metric
            for metric in ALLOWED_METRICS
            if metric_values_for_split(item, split_name, metric)
        }
        per_file_metrics.append(available)

    if not per_file_metrics:
        return []

    common = set.intersection(*per_file_metrics)
    return [metric for metric in ALLOWED_METRICS if metric in common]


def common_metrics_for_split_with_baseline(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
    split_name: str,
    baseline_split_name: str,
) -> list[str]:
    selected_common = set(common_metrics_for_split(selected_results, split_name))
    baseline_available = {
        metric
        for metric in ALLOWED_METRICS
        if metric_values_for_split(baseline_result, baseline_split_name, metric)
    }

    common = selected_common.intersection(baseline_available)
    return [metric for metric in ALLOWED_METRICS if metric in common]


def common_metrics_across_selected_splits(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
    split_to_baseline_split: dict[str, str],
) -> list[str]:
    running_common: set[str] | None = None

    for split_name, baseline_split_name in split_to_baseline_split.items():
        split_common = set(
            common_metrics_for_split_with_baseline(
                selected_results=selected_results,
                baseline_result=baseline_result,
                split_name=split_name,
                baseline_split_name=baseline_split_name,
            )
        )
        if running_common is None:
            running_common = split_common
        else:
            running_common = running_common.intersection(split_common)

    if not running_common:
        return []

    return [metric for metric in ALLOWED_METRICS if metric in running_common]


def resolve_metric_selection(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
    split_to_baseline_split: dict[str, str],
    metrics_from_flags: list[str] | None,
) -> list[str]:
    inquirer = get_inquirer_module()

    available_metrics = common_metrics_across_selected_splits(
        selected_results=selected_results,
        baseline_result=baseline_result,
        split_to_baseline_split=split_to_baseline_split,
    )

    if not available_metrics:
        raise ValueError(
            "No metrics are shared across selected DA methods, selected splits, and mapped baseline splits."
        )

    if metrics_from_flags:
        selected: list[str] = []
        seen: set[str] = set()
        for metric in metrics_from_flags:
            if metric not in seen:
                selected.append(metric)
                seen.add(metric)

        invalid = [metric for metric in selected if metric not in available_metrics]
        if invalid:
            raise ValueError(
                "Metric flag selection is not valid for the chosen methods/splits/baseline. "
                f"Invalid: {invalid}. Available: {available_metrics}."
            )
        return selected

    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "selected_metrics",
                message="Select one or more metrics to display",
                choices=available_metrics,
            )
        ]
    )
    selected_metrics = answers.get("selected_metrics", []) if answers else []
    if not selected_metrics:
        raise ValueError("No metric selected.")

    return selected_metrics


def resolve_expansion_baselines_by_real_count(
    results: list[ResultFile],
    selected_results: list[ResultFile],
    required_real_counts: list[int],
    baseline_ids_from_flags: list[str] | None,
    filter_keywords: list[str] | None = None,
    show_result_paths_in_prompt: bool = True,
) -> dict[int, ResultFile]:
    by_id = {item.evaluation_id: item for item in results}
    selected_ids = {item.evaluation_id for item in selected_results}
    candidate_results = [item for item in results if item.evaluation_id not in selected_ids]
    if filter_keywords:
        candidate_results = filter_results_by_evaluation_id_keywords(candidate_results, filter_keywords)

    if not candidate_results:
        raise ValueError("No candidate baseline evaluation_id remains after DA method selection.")

    normalized_flag_ids: list[str] = []
    if baseline_ids_from_flags:
        seen: set[str] = set()
        for baseline_id in baseline_ids_from_flags:
            if baseline_id in seen:
                continue
            if baseline_id not in by_id:
                raise ValueError(f"Baseline evaluation_id '{baseline_id}' from flags was not found.")
            if baseline_id in selected_ids:
                raise ValueError(
                    f"Baseline evaluation_id '{baseline_id}' cannot also be selected as a DA method."
                )
            normalized_flag_ids.append(baseline_id)
            seen.add(baseline_id)

    baseline_candidates = (
        [by_id[baseline_id] for baseline_id in normalized_flag_ids]
        if normalized_flag_ids
        else candidate_results
    )

    mapping: dict[int, ResultFile] = {}

    inquirer = get_inquirer_module() if not normalized_flag_ids else None

    for real_count in sorted(required_real_counts):
        matches = [
            item for item in baseline_candidates if choose_baseline_split_for_real_count(item, real_count) is not None
        ]

        if not matches:
            candidate_ids = [item.evaluation_id for item in baseline_candidates]
            raise ValueError(
                "No baseline evaluation_id contains a matching split for "
                f"real_image_count={real_count}. Available baseline IDs: {candidate_ids}"
            )

        if normalized_flag_ids:
            mapping[real_count] = matches[0]
            continue

        if show_result_paths_in_prompt:
            choices = [
                (f"{item.evaluation_id}  ({item.file_path.parent})", item.evaluation_id)
                for item in matches
            ]
        else:
            choices = [item.evaluation_id for item in matches]
        answers = inquirer.prompt(
            [
                inquirer.List(
                    f"baseline_id_{real_count}",
                    message=f"Select baseline evaluation_id for real_image_count={real_count}",
                    choices=choices,
                )
            ]
        )
        baseline_id = answers.get(f"baseline_id_{real_count}") if answers else None
        if not baseline_id:
            raise ValueError(f"No baseline evaluation_id selected for real_image_count={real_count}.")

        mapping[real_count] = by_id[baseline_id]

    return mapping


def available_metrics_for_expansion(
    selected_results: list[ResultFile],
    baseline_by_real_count: dict[int, ResultFile],
    selected_splits: list[str],
) -> list[str]:
    available: list[str] = []

    for metric in ALLOWED_METRICS:
        has_da_data = any(
            metric_values_for_split(item, split_name, metric)
            for split_name in selected_splits
            for item in selected_results
        )
        if not has_da_data:
            continue

        baseline_ok = True
        for real_count, baseline_item in baseline_by_real_count.items():
            baseline_split = choose_baseline_split_for_real_count(baseline_item, real_count)
            if baseline_split is None:
                baseline_ok = False
                break

            baseline_split_name = baseline_split.get("split_name")
            if not isinstance(baseline_split_name, str) or not baseline_split_name:
                baseline_ok = False
                break

            if not metric_values_for_split(baseline_item, baseline_split_name, metric):
                baseline_ok = False
                break

        if baseline_ok:
            available.append(metric)

    return available


def resolve_metric_selection_expansion(
    selected_results: list[ResultFile],
    baseline_by_real_count: dict[int, ResultFile],
    selected_splits: list[str],
    metrics_from_flags: list[str] | None,
) -> list[str]:
    available_metrics = available_metrics_for_expansion(
        selected_results=selected_results,
        baseline_by_real_count=baseline_by_real_count,
        selected_splits=selected_splits,
    )

    if not available_metrics:
        raise ValueError(
            "No metrics are shared across selected DA methods, selected splits, and selected baseline mappings."
        )

    if metrics_from_flags:
        selected: list[str] = []
        seen: set[str] = set()
        for metric in metrics_from_flags:
            if metric not in seen:
                selected.append(metric)
                seen.add(metric)

        invalid = [metric for metric in selected if metric not in available_metrics]
        if invalid:
            raise ValueError(
                "Metric flag selection is not valid for expansion ratio mode. "
                f"Invalid: {invalid}. Available: {available_metrics}."
            )
        return selected

    inquirer = get_inquirer_module()

    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "selected_metrics",
                message="Select one or more metrics to display",
                choices=available_metrics,
            )
        ]
    )
    selected_metrics = answers.get("selected_metrics", []) if answers else []
    if not selected_metrics:
        raise ValueError("No metric selected.")

    return selected_metrics


def uncertainty(values: Iterable[float], mode: str) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return (math.nan, math.nan)

    mean_val = fmean(vals)

    if len(vals) == 1:
        return (mean_val, 0.0)

    std = pstdev(vals)
    if mode == "std":
        return (mean_val, std)

    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    h = t.ppf(0.975, df=len(vals)-1) * se   # 95% CI
    # print(f"[DEBUG] mean={mean_val:.4f}, std={std:.4f}, se={se:.4f}, h={h:.4f} for values: {vals}")
    confidence_interval = 2.045 * (np.std(vals) / np.sqrt(len(vals)))
    # print(f"[DEBUG] Calculated 95% confidence interval: ±{confidence_interval:.4f} for values: {vals}")
    return (mean_val, h)


def compute_common_prefix(strings: list[str]) -> str:
    if len(strings) < 2:
        return ""

    prefix = strings[0]
    for candidate in strings[1:]:
        while not candidate.startswith(prefix) and prefix:
            prefix = prefix[:-1]
        if not prefix:
            break

    return prefix


def format_method_names(evaluation_ids: list[str]) -> tuple[dict[str, str], str]:
    prefix = compute_common_prefix(evaluation_ids)

    if not prefix:
        return ({eid: eid for eid in evaluation_ids}, "")

    display_map: dict[str, str] = {}
    for eid in evaluation_ids:
        remainder = eid[len(prefix) :]
        display_map[eid] = "*" if remainder == "" else f"*{remainder}"

    return (display_map, prefix)


def metric_title(metric: str) -> str:
    return metric.replace("_", " ").title()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_label_fragment(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "value"


def format_metric_summary_cell(
    mean_val: float,
    err_val: float,
    run_count: int,
    to_latex: bool,
) -> str:
    if to_latex:
        return f"{mean_val * 100:.2f}\\% $\\pm$ {err_val * 100:.2f}\\% ({run_count} runs)"

    return f"{mean_val * 100:.2f}% ± {err_val * 100:.2f}% ({run_count} runs)"


def format_plain_percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"

    as_percent = value * 100.0
    text = f"{as_percent:.2f}".rstrip("0").rstrip(".")
    return f"{text}\\%"


def format_plain_percent_with_uncertainty(
    mean_value: float | None,
    uncertainty_value: float | None,
) -> str:
    if mean_value is None or math.isnan(mean_value):
        return "-"

    mean_text = format_plain_percent(mean_value)
    if uncertainty_value is None or math.isnan(uncertainty_value):
        return mean_text

    uncertainty_text = format_plain_percent(abs(uncertainty_value))
    return f"{mean_text} $\\pm$ {uncertainty_text}"


def format_compact_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_real_images_display(real_image_count: int) -> str:
    per_class = real_image_count / REAL_IMAGES_PER_CLASS_DIVISOR
    if per_class.is_integer():
        return str(int(per_class))
    return f"{per_class:.2f}".rstrip("0").rstrip(".")


def render_latex_table_figure(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{ll}",
        r"\toprule",
        f" \\textbf{{{latex_escape(headers[0])}}} & \\textbf{{{latex_escape(headers[1])}}} \\\\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        method_name, metric_summary = row
        lines.append(f" {latex_escape(method_name)} & {metric_summary} \\\\")
        if idx == 0 and len(rows) > 1:
            lines.append("")
            lines.append(r" \hline")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{tab:{label}}}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def render_expansion_ratio_latex_table(
    method_name: str,
    metric: str,
    real_counts: list[int],
    ratio_columns: list[float],
    baseline_by_real_count: dict[int, tuple[float, float]],
    method_values: dict[tuple[int, float], tuple[float, float]],
) -> str:
    ratio_column_count = 1 + len(ratio_columns)
    total_columns = 1 + ratio_column_count
    column_spec = "c" * total_columns

    header_ratio_cells = ["1"]
    header_ratio_cells.extend(
        f"\\textbf{{$\\times({format_compact_number(ratio)}+1)$}}"
        for ratio in ratio_columns
    )
    header_ratio_line = " & " + " & ".join(header_ratio_cells) + r" \\" 

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\makebox[\textwidth][c]{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        (
            r"\multirow{2}{*}{\textbf{Real Images}}"
            f" & \\multicolumn{{{ratio_column_count}}}{{c}}{{\\textbf{{Expansion Ratio}}}} \\\\" 
        ),
        f"\\cline{{2-{total_columns}}}",
        header_ratio_line,
        r"\hline",
    ]

    for real_count in real_counts:
        row_cells = [format_real_images_display(real_count)]
        baseline_mean, baseline_err = baseline_by_real_count.get(real_count, (math.nan, math.nan))
        row_cells.append(format_plain_percent_with_uncertainty(baseline_mean, baseline_err))
        for ratio in ratio_columns:
            value_mean, value_err = method_values.get((real_count, ratio), (math.nan, math.nan))
            row_cells.append(format_plain_percent_with_uncertainty(value_mean, value_err))
        lines.append(" " + " & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            (
                "\\caption{"
                f"Real dataset size vs. expansion ratio for \\textbf{{{latex_escape(method_name)}}}"
                "."
                "}"
            ),
            (
                "\\label{tab:"
                "real_dataset_size_vs_expansion_ratio"
                f"_{latex_label_fragment(method_name)}"
                f"_{latex_label_fragment(metric)}"
                "}"
            ),
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def render_zero_synthetic_expansion_latex_table(
    metric: str,
    real_counts: list[int],
    method_names: list[str],
    baseline_by_real_count: dict[int, tuple[float, float]],
    method_values_by_name: dict[str, dict[int, tuple[float, float]]],
) -> str:
    ratio_column_count = 1 + len(method_names)
    total_columns = 1 + ratio_column_count
    column_spec = "c" * total_columns

    header_method_cells = ["1"]
    header_method_cells.extend(
        f"\\textbf{{{latex_escape(method_name)}}}" for method_name in method_names
    )
    header_method_line = " & " + " & ".join(header_method_cells) + r" \\" 

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        (
            r"\multirow{2}{*}{\textbf{Real Images}}"
            f" & \\multicolumn{{{ratio_column_count}}}{{c}}{{\\textbf{{Expansion Ratio}}}} \\\\" 
        ),
        f"\\cline{{2-{total_columns}}}",
        header_method_line,
        r"\hline",
    ]

    for real_count in real_counts:
        row_cells = [format_real_images_display(real_count)]
        baseline_mean, baseline_err = baseline_by_real_count.get(real_count, (math.nan, math.nan))
        row_cells.append(format_plain_percent_with_uncertainty(baseline_mean, baseline_err))
        for method_name in method_names:
            method_mean, method_err = method_values_by_name.get(method_name, {}).get(
                real_count,
                (math.nan, math.nan),
            )
            row_cells.append(format_plain_percent_with_uncertainty(method_mean, method_err))
        lines.append(" " + " & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                "\\caption{"
                "Real dataset size comparison across methods with zero synthetic images"
                f" for \\textbf{{{latex_escape(metric_title(metric))}}}."
                "}"
            ),
            (
                "\\label{tab:"
                "real_dataset_size_vs_method_zero_synth"
                f"_{latex_label_fragment(metric)}"
                "}"
            ),
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def build_rerun_command(
    args: argparse.Namespace,
    selected_results: list[ResultFile],
    baseline_evaluation_ids: list[str],
    selected_splits: list[str],
    selected_metrics: list[str],
) -> str:
    parts: list[str] = [
        "python",
        "scripts/interactive_results_summary.py",
        "--test-dir",
        shlex.quote(str(args.test_dir)),
        "--sort",
        shlex.quote(args.sort),
        "--error",
        shlex.quote(args.error),
    ]

    for baseline_id in baseline_evaluation_ids:
        parts.extend(["--baseline-evaluation-id", shlex.quote(baseline_id)])

    for item in selected_results:
        parts.extend(["--evaluation-id", shlex.quote(item.evaluation_id)])

    for split_name in selected_splits:
        parts.extend(["--split-name", shlex.quote(split_name)])

    for metric in selected_metrics:
        parts.extend(["--metric", shlex.quote(metric)])

    if args.use_all_samples:
        parts.append("--use-all-samples")

    if args.to_latex:
        parts.append("--to_latex")

    if args.expansion_ratio:
        parts.append("--expansion_ratio")

    return " ".join(parts)


def write_table_export_json(export_path: Path, payload: dict) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Exported table JSON to: {export_path}")


def print_expansion_ratio_report(
    selected_results: list[ResultFile],
    baseline_by_real_count_items: dict[int, ResultFile],
    selected_splits: list[str],
    selected_metrics: list[str],
    error_mode: str,
    use_all_samples: bool,
    rerun_command: str,
    to_latex: bool = False,
) -> list[dict]:
    if to_latex:
        print("\n% Re-run Command (No Prompts)")
    else:
        print("\nRe-run Command (No Prompts)")
    print(rerun_command)

    exported_tables: list[dict] = []

    split_metadata = expansion_split_metadata(selected_results)

    missing = [split_name for split_name in selected_splits if split_name not in split_metadata]
    if missing:
        raise ValueError(
            "Selected split_name values were not found in selected DA methods: "
            f"{missing}."
        )

    split_info = [
        (split_name, *split_metadata[split_name])
        for split_name in selected_splits
    ]

    real_counts = sorted({real_count for _, real_count, _, _ in split_info})
    ratio_columns = sorted({ratio for _, _, _, ratio in split_info if ratio > 0})
    zero_ratio_present = any(ratio == 0 for _, _, _, ratio in split_info)
    zero_synthetic_mode = not ratio_columns and zero_ratio_present

    if not real_counts:
        raise ValueError("No valid real_image_count values were found for expansion ratio tables.")

    if not ratio_columns and not zero_synthetic_mode:
        raise ValueError(
            "No positive expansion ratios were found. Select at least one split with synthetic_image_count > 0."
        )

    split_names_by_real_ratio: dict[tuple[int, float], list[str]] = {}
    split_names_by_real_zero: dict[int, list[str]] = {}
    for split_name, real_count, _, ratio in sorted(split_info, key=lambda x: (x[1], x[3], x[0])):
        if ratio > 0:
            key = (real_count, ratio)
            split_names_by_real_ratio.setdefault(key, []).append(split_name)
            continue

        if zero_synthetic_mode and ratio == 0:
            split_names_by_real_zero.setdefault(real_count, []).append(split_name)

    if zero_synthetic_mode:
        duplicates_zero = {
            real_count: names
            for real_count, names in split_names_by_real_zero.items()
            if len(names) > 1
        }
        if duplicates_zero:
            print("% [WARN] Multiple split_name values found for same real count in zero-synthetic mode:")
            for real_count, split_names in sorted(duplicates_zero.items(), key=lambda item: item[0]):
                print("% " f"real={real_count} -> {split_names}")
    else:
        duplicates = {
            key: names for key, names in split_names_by_real_ratio.items() if len(names) > 1
        }
        if duplicates:
            print("% [WARN] Multiple split_name values found for same real count and expansion ratio:")
            for (real_count, ratio), split_names in sorted(duplicates.items(), key=lambda item: (item[0][0], item[0][1])):
                print(
                    "% "
                    f"real={real_count}, ratio={format_compact_number(ratio)} "
                    f"-> {split_names}"
                )

    method_groups: dict[str, list[ResultFile]] = {}
    for item in selected_results:
        method_name = method_name_for_expansion(item.evaluation_id)
        method_groups.setdefault(method_name, []).append(item)

    sample_limits: dict[tuple[str, str], int] = {}
    if not use_all_samples:
        for split_name in selected_splits:
            for metric in selected_metrics:
                sample_limits[(split_name, metric)] = da_sample_limit_for_split_metric_present(
                    selected_results,
                    split_name,
                    metric,
                )

    tabulate = get_tabulate_function()

    for metric in selected_metrics:
        baseline_by_real_count: dict[int, tuple[float, float]] = {}
        for real_count in real_counts:
            baseline_item = baseline_by_real_count_items.get(real_count)
            if baseline_item is None:
                raise ValueError(
                    f"No baseline evaluation_id selected for real_image_count={real_count}."
                )

            baseline_split = choose_baseline_split_for_real_count(baseline_item, real_count)
            if baseline_split is None:
                raise ValueError(
                    f"No baseline split found with real_image_count={real_count} for expansion ratio mode."
                )

            baseline_split_name = baseline_split.get("split_name")
            if not isinstance(baseline_split_name, str) or not baseline_split_name:
                raise ValueError(
                    f"Baseline split with real_image_count={real_count} does not provide a valid split_name."
                )

            baseline_values = metric_values_for_split(baseline_item, baseline_split_name, metric)
            baseline_by_real_count[real_count] = (
                uncertainty(baseline_values, error_mode) if baseline_values else (math.nan, math.nan)
            )

        if zero_synthetic_mode:
            method_names = list(method_groups.keys())
            method_values_by_name: dict[str, dict[int, tuple[float, float]]] = {}

            for method_name, method_items in method_groups.items():
                method_values_by_real_count: dict[int, tuple[float, float]] = {}

                for real_count in real_counts:
                    candidate_splits = split_names_by_real_zero.get(real_count, [])

                    found_value = False
                    for split_name in candidate_splits:
                        sample_limit: int | None = None
                        if not use_all_samples:
                            sample_limit = sample_limits.get((split_name, metric), 0)

                        for item in method_items:
                            vals = metric_values_for_split(item, split_name, metric, sample_limit=sample_limit)
                            if vals:
                                method_values_by_real_count[real_count] = uncertainty(vals, error_mode)
                                found_value = True
                                break

                        if found_value:
                            break

                    if not found_value:
                        method_values_by_real_count[real_count] = (math.nan, math.nan)

                method_values_by_name[method_name] = method_values_by_real_count

            if to_latex:
                table = render_zero_synthetic_expansion_latex_table(
                    metric=metric,
                    real_counts=real_counts,
                    method_names=method_names,
                    baseline_by_real_count=baseline_by_real_count,
                    method_values_by_name=method_values_by_name,
                )
                print("\n" + table)
            else:
                headers = ["Real Images", "Baseline"] + method_names
                rows: list[list[str]] = []

                for real_count in real_counts:
                    row_cells = [format_real_images_display(real_count)]
                    baseline_mean, baseline_err = baseline_by_real_count.get(real_count, (math.nan, math.nan))
                    row_cells.append(format_plain_percent_with_uncertainty(baseline_mean, baseline_err))
                    for method_name in method_names:
                        method_mean, method_err = method_values_by_name.get(method_name, {}).get(
                            real_count,
                            (math.nan, math.nan),
                        )
                        row_cells.append(format_plain_percent_with_uncertainty(method_mean, method_err))
                    rows.append(row_cells)

                print(f"\nMetric: {metric}")
                print(tabulate(rows, headers=headers, tablefmt="github"))

            exported_rows: list[dict] = []
            for real_count in real_counts:
                baseline_mean, baseline_err = baseline_by_real_count.get(real_count, (math.nan, math.nan))
                methods: dict[str, str] = {}
                for method_name in method_names:
                    method_mean, method_err = method_values_by_name.get(method_name, {}).get(
                        real_count,
                        (math.nan, math.nan),
                    )
                    methods[method_name] = format_plain_percent_with_uncertainty(method_mean, method_err)

                exported_rows.append(
                    {
                        "real_images": format_real_images_display(real_count),
                        "baseline": format_plain_percent_with_uncertainty(baseline_mean, baseline_err),
                        "methods": methods,
                    }
                )

            exported_tables.append(
                {
                    "mode": "expansion_ratio_zero_synthetic",
                    "metric": metric,
                    "headers": ["Real Images", "Baseline"] + method_names,
                    "rows": exported_rows,
                }
            )

            continue

        for method_name, method_items in method_groups.items():
            method_values: dict[tuple[int, float], tuple[float, float]] = {}

            for real_count in real_counts:
                for ratio in ratio_columns:
                    candidate_splits = split_names_by_real_ratio.get((real_count, ratio), [])

                    found_value = False
                    for split_name in candidate_splits:
                        sample_limit: int | None = None
                        if not use_all_samples:
                            split_limit = sample_limits.get((split_name, metric), 0)
                            sample_limit = split_limit

                        for item in method_items:
                            vals = metric_values_for_split(item, split_name, metric, sample_limit=sample_limit)
                            if vals:
                                method_values[(real_count, ratio)] = uncertainty(vals, error_mode)
                                found_value = True
                                break

                        if found_value:
                            break

                    if not found_value:
                        method_values[(real_count, ratio)] = (math.nan, math.nan)

            if to_latex:
                table = render_expansion_ratio_latex_table(
                    method_name=method_name,
                    metric=metric,
                    real_counts=real_counts,
                    ratio_columns=ratio_columns,
                    baseline_by_real_count=baseline_by_real_count,
                    method_values=method_values,
                )

                print("\n" + table)
            else:
                # Plain-text table per method
                headers = ["Real Images"]
                # Baseline column header
                headers.append("Baseline")
                # Ratio columns
                headers.extend([f"x({format_compact_number(r)}+1)" for r in ratio_columns])

                rows: list[list[str]] = []
                for real_count in real_counts:
                    row_cells = [format_real_images_display(real_count)]
                    baseline_mean, baseline_err = baseline_by_real_count.get(real_count, (math.nan, math.nan))
                    row_cells.append(format_plain_percent_with_uncertainty(baseline_mean, baseline_err))
                    for ratio in ratio_columns:
                        value_mean, value_err = method_values.get((real_count, ratio), (math.nan, math.nan))
                        row_cells.append(format_plain_percent_with_uncertainty(value_mean, value_err))
                    rows.append(row_cells)

                print(f"\nMethod: {method_name} — Metric: {metric}")
                print(tabulate(rows, headers=headers, tablefmt="github"))

            exported_rows: list[dict] = []
            for real_count in real_counts:
                baseline_mean, baseline_err = baseline_by_real_count.get(real_count, (math.nan, math.nan))
                ratio_values: dict[str, str] = {}
                for ratio in ratio_columns:
                    value_mean, value_err = method_values.get((real_count, ratio), (math.nan, math.nan))
                    ratio_values[f"x({format_compact_number(ratio)}+1)"] = format_plain_percent_with_uncertainty(
                        value_mean,
                        value_err,
                    )

                exported_rows.append(
                    {
                        "real_images": format_real_images_display(real_count),
                        "baseline": format_plain_percent_with_uncertainty(baseline_mean, baseline_err),
                        "ratios": ratio_values,
                    }
                )

            exported_tables.append(
                {
                    "mode": "expansion_ratio",
                    "metric": metric,
                    "method_name": method_name,
                    "headers": ["Real Images", "Baseline"]
                    + [f"x({format_compact_number(r)}+1)" for r in ratio_columns],
                    "rows": exported_rows,
                }
            )

    return exported_tables


def print_report(
    selected_results: list[ResultFile],
    baseline_result: ResultFile,
    selected_splits: list[str],
    split_to_baseline_split: dict[str, str],
    selected_metrics: list[str],
    error_mode: str,
    use_all_samples: bool,
    rerun_command: str,
    to_latex: bool,
) -> list[dict]:
    tabulate = get_tabulate_function()
    exported_tables: list[dict] = []

    da_evaluation_ids = [item.evaluation_id for item in selected_results]
    method_name_map, shared_prefix = format_method_names(da_evaluation_ids)

    if to_latex:
        print("\n% Re-run Command (No Prompts)")
    else:
        print("\nRe-run Command (No Prompts)")
    print(rerun_command)

    if to_latex:
        print("\n% Selected DA Methods")
    else:
        print("\n" + "=" * 88)
        print("Selected DA Methods")
        print("=" * 88)
    for item in selected_results:
        if to_latex:
            print(f"% - {method_name_map[item.evaluation_id]} -> {item.evaluation_id}")
        else:
            print(f"- {method_name_map[item.evaluation_id]} -> {item.evaluation_id}")

    if to_latex:
        print("\n% Baseline")
        print(f"% - {baseline_result.evaluation_id} -> {baseline_result.evaluation_id}")
    else:
        print("\nBaseline")
        print(f"- {baseline_result.evaluation_id} -> {baseline_result.evaluation_id}")

    if shared_prefix:
        if to_latex:
            print(f"\n% * = {shared_prefix}")
        else:
            print(f"\n* = {shared_prefix}")

    error_label = "STD" if error_mode == "std" else "95% CI"

    for split_name in selected_splits:
        baseline_split_name = split_to_baseline_split[split_name]
        real_count = shared_real_image_count_for_split(selected_results, split_name)

        if to_latex:
            print(f"\n% Split: {split_name}")
            print(f"% Baseline split used: {baseline_split_name}")
            if real_count is not None:
                print(f"% Shared real_image_count: {real_count}")
        else:
            print("\n" + "#" * 88)
            print(f"Split: {split_name}")
            print(f"Baseline split used: {baseline_split_name}")
            if real_count is not None:
                print(f"Shared real_image_count: {real_count}")
            print("#" * 88)

        common_metrics = common_metrics_for_split_with_baseline(
            selected_results=selected_results,
            baseline_result=baseline_result,
            split_name=split_name,
            baseline_split_name=baseline_split_name,
        )
        metrics_to_show = [metric for metric in selected_metrics if metric in common_metrics]

        if not metrics_to_show:
            print("No common metrics found across selected DA methods for this split.")
            continue

        for metric in metrics_to_show:
            header = f"{metric_title(metric)} ({error_label})"
            if to_latex:
                print(f"\n% {header}")
            else:
                print(f"\n{header}")
                print("-" * len(header))

            da_sample_limit: int | None = None
            if not use_all_samples:
                da_sample_limit = da_sample_limit_for_split_metric(selected_results, split_name, metric)

            if use_all_samples:
                sample_mode_line = "DA sample mode: using all available runs per method"
            else:
                sample_mode_line = f"DA sample mode: aligned to {da_sample_limit} runs per method"

            if to_latex:
                print(f"% {sample_mode_line}")
            else:
                print(sample_mode_line)

            rows = []

            baseline_vals = metric_values_for_split(baseline_result, baseline_split_name, metric)
            baseline_mean, baseline_err = uncertainty(baseline_vals, error_mode)
            baseline_row = [
                f"{baseline_result.evaluation_id} (baseline)",
                format_metric_summary_cell(
                    mean_val=baseline_mean,
                    err_val=baseline_err,
                    run_count=len(baseline_vals),
                    to_latex=to_latex,
                ),
            ]
            rows.append(baseline_row)

            if not to_latex:
                # Visual separator requested below the baseline row.
                rows.append(["-" * 16, "-" * 20])

            da_rows: list[tuple[float, list[str]]] = []
            for item in selected_results:
                vals = metric_values_for_split(item, split_name, metric, sample_limit=da_sample_limit)
                mean_val, err_val = uncertainty(vals, error_mode)
                da_rows.append(
                    (
                        mean_val,
                        [
                            method_name_map[item.evaluation_id],
                            format_metric_summary_cell(
                                mean_val=mean_val,
                                err_val=err_val,
                                run_count=len(vals),
                                to_latex=to_latex,
                            ),
                        ],
                    )
                )

            da_rows.sort(key=lambda x: x[0], reverse=True)
            rows.extend(row for _, row in da_rows)

            if to_latex:
                latex_table = render_latex_table_figure(
                    headers=["DA Method", metric_title(metric)],
                    rows=rows,
                    caption=", ".join(
                        [
                            f"{metric_title(metric)} ({error_label})",
                            f"split={split_name}",
                            f"baseline_split={baseline_split_name}",
                        ]
                        + ([f"real_image_count={real_count}"] if real_count is not None else [])
                        + (
                            ["da_sample_mode=all_runs"]
                            if use_all_samples
                            else [f"da_sample_mode=aligned_{da_sample_limit}_runs"]
                        )
                    ),
                    label="_".join(
                        [
                            "da_summary",
                            latex_label_fragment(split_name),
                            latex_label_fragment(metric),
                            latex_label_fragment(error_mode),
                        ]
                    ),
                )
                print(latex_table)
            else:
                print(tabulate(rows, headers=["DA Method", metric_title(metric)], tablefmt="github"))

            export_rows = [
                {
                    "method": baseline_result.evaluation_id,
                    "is_baseline": True,
                    "summary": baseline_row[1],
                }
            ]
            for _, da_row in da_rows:
                export_rows.append(
                    {
                        "method": da_row[0],
                        "is_baseline": False,
                        "summary": da_row[1],
                    }
                )

            exported_tables.append(
                {
                    "mode": "standard",
                    "split_name": split_name,
                    "baseline_split_name": baseline_split_name,
                    "metric": metric,
                    "error_mode": error_mode,
                    "headers": ["DA Method", metric_title(metric)],
                    "rows": export_rows,
                }
            )

    return exported_tables


def main() -> None:
    args = parse_args()

    metrics_from_flags: list[str] | None = None
    if args.metrics:
        metrics_from_flags = [metric for group in args.metrics for metric in group]

    evaluation_ids_from_flags: list[str] | None = None
    if args.evaluation_ids:
        evaluation_ids_from_flags = [eid for group in args.evaluation_ids for eid in group]

    split_names_from_flags: list[str] | None = None
    if args.split_names:
        split_names_from_flags = [s for group in args.split_names for s in group]

    baseline_ids_from_flags: list[str] | None = None
    if args.baseline_evaluation_ids:
        baseline_ids_from_flags = [eid for group in args.baseline_evaluation_ids for eid in group]

    filter_keywords = parse_filter_keywords(args.evaluation_id_filter)
    interactive_filter_active = False

    discovered = discover_results_files(args.test_dir)
    if not discovered:
        raise SystemExit("No valid results.json files with evaluation_id were found.")

    unique_results = ensure_unique_evaluation_ids(discovered)
    sorted_results = sort_results(unique_results, mode=args.sort)

    if evaluation_ids_from_flags:
        by_id = {item.evaluation_id: item for item in sorted_results}
        selected_results = []
        seen: set[str] = set()
        for eid in evaluation_ids_from_flags:
            if eid in seen:
                continue
            if eid not in by_id:
                raise ValueError(f"evaluation_id '{eid}' from flags was not found.")
            selected_results.append(by_id[eid])
            seen.add(eid)
        if not selected_results:
            raise ValueError("No valid DA method evaluation_id provided via flags.")
    else:
        if not filter_keywords:
            filter_keywords = prompt_for_evaluation_id_filter_keywords()
            interactive_filter_active = bool(filter_keywords)

        filtered_results = filter_results_by_evaluation_id_keywords(sorted_results, filter_keywords)
        if filter_keywords and not filtered_results:
            raise ValueError(
                "No evaluation_id values matched the filter keywords: "
                f"{', '.join(filter_keywords)}"
            )

        if filter_keywords and (interactive_filter_active or args.evaluation_id_filter):
            print(f"Using evaluation_id filter keywords: {', '.join(filter_keywords)}")

        selected_results = prompt_for_evaluation_ids(filtered_results)

    if args.expansion_ratio:
        if split_names_from_flags:
            selectable = {value for _, value in get_expansion_selectable_split_choices(selected_results)}
            selected_splits = []
            seen_splits: set[str] = set()
            for split_name in split_names_from_flags:
                if split_name in seen_splits:
                    continue
                if split_name not in selectable:
                    raise ValueError(
                        f"split_name '{split_name}' from flags is not selectable for expansion ratio mode."
                    )
                selected_splits.append(split_name)
                seen_splits.add(split_name)
            if not selected_splits:
                raise ValueError("No valid split_name provided via flags.")
        else:
            selected_splits = prompt_for_expansion_split_names(selected_results)

        split_metadata = expansion_split_metadata(selected_results)
        required_real_counts = sorted({split_metadata[split_name][0] for split_name in selected_splits})

        baseline_by_real_count_items = resolve_expansion_baselines_by_real_count(
            results=sorted_results,
            selected_results=selected_results,
            required_real_counts=required_real_counts,
            baseline_ids_from_flags=baseline_ids_from_flags,
            filter_keywords=filter_keywords,
            show_result_paths_in_prompt=False,
        )

        selected_metrics = resolve_metric_selection_expansion(
            selected_results=selected_results,
            baseline_by_real_count=baseline_by_real_count_items,
            selected_splits=selected_splits,
            metrics_from_flags=metrics_from_flags,
        )

        baseline_ids_for_command: list[str] = []
        seen_baseline_ids: set[str] = set()
        for real_count in required_real_counts:
            baseline_id = baseline_by_real_count_items[real_count].evaluation_id
            if baseline_id in seen_baseline_ids:
                continue
            baseline_ids_for_command.append(baseline_id)
            seen_baseline_ids.add(baseline_id)

        rerun_command = build_rerun_command(
            args=args,
            selected_results=selected_results,
            baseline_evaluation_ids=baseline_ids_for_command,
            selected_splits=selected_splits,
            selected_metrics=selected_metrics,
        )

        exported_tables = print_expansion_ratio_report(
            selected_results=selected_results,
            baseline_by_real_count_items=baseline_by_real_count_items,
            selected_splits=selected_splits,
            selected_metrics=selected_metrics,
            error_mode=args.error,
            use_all_samples=args.use_all_samples,
            rerun_command=rerun_command,
            to_latex=args.to_latex,
        )

        if args.export:
            export_payload = {
                "generated_at": datetime.now().isoformat(),
                "mode": "expansion_ratio",
                "to_latex": args.to_latex,
                "error_mode": args.error,
                "rerun_command": rerun_command,
                "selected_evaluation_ids": [item.evaluation_id for item in selected_results],
                "baseline_evaluation_ids": baseline_ids_for_command,
                "selected_splits": selected_splits,
                "selected_metrics": selected_metrics,
                "tables": exported_tables,
            }
            write_table_export_json(args.export, export_payload)
        return

    baseline_evaluation_id: str | None = None
    if baseline_ids_from_flags:
        unique_baselines: list[str] = []
        seen_baselines: set[str] = set()
        for baseline_id in baseline_ids_from_flags:
            if baseline_id in seen_baselines:
                continue
            unique_baselines.append(baseline_id)
            seen_baselines.add(baseline_id)

        if len(unique_baselines) != 1:
            raise ValueError(
                "Non-expansion mode expects exactly one baseline evaluation_id. "
                "Provide a single --baseline-evaluation-id or use --expansion_ratio for multiple baselines."
            )

        baseline_evaluation_id = unique_baselines[0]

    baseline_result = prompt_for_baseline_evaluation_id(
        results=sorted_results,
        selected_results=selected_results,
        baseline_evaluation_id=baseline_evaluation_id,
        filter_keywords=filter_keywords,
    )

    if split_names_from_flags:
        selectable = {value for _, value in get_selectable_split_choices(selected_results, baseline_result)}
        selected_splits = []
        seen_splits: set[str] = set()
        for split_name in split_names_from_flags:
            if split_name in seen_splits:
                continue
            if split_name not in selectable:
                raise ValueError(
                    f"split_name '{split_name}' from flags is not selectable for current methods/baseline."
                )
            selected_splits.append(split_name)
            seen_splits.add(split_name)
        if not selected_splits:
            raise ValueError("No valid split_name provided via flags.")
    else:
        selected_splits = prompt_for_split_names(selected_results, baseline_result)

    split_to_baseline_split = build_baseline_split_name_map(
        selected_results=selected_results,
        baseline_result=baseline_result,
        selected_splits=selected_splits,
    )

    selected_metrics = resolve_metric_selection(
        selected_results=selected_results,
        baseline_result=baseline_result,
        split_to_baseline_split=split_to_baseline_split,
        metrics_from_flags=metrics_from_flags,
    )

    rerun_command = build_rerun_command(
        args=args,
        selected_results=selected_results,
        baseline_evaluation_ids=[baseline_result.evaluation_id],
        selected_splits=selected_splits,
        selected_metrics=selected_metrics,
    )

    exported_tables = print_report(
        selected_results=selected_results,
        baseline_result=baseline_result,
        selected_splits=selected_splits,
        split_to_baseline_split=split_to_baseline_split,
        selected_metrics=selected_metrics,
        error_mode=args.error,
        use_all_samples=args.use_all_samples,
        rerun_command=rerun_command,
        to_latex=args.to_latex,
    )

    if args.export:
        export_payload = {
            "generated_at": datetime.now().isoformat(),
            "mode": "standard",
            "to_latex": args.to_latex,
            "error_mode": args.error,
            "rerun_command": rerun_command,
            "selected_evaluation_ids": [item.evaluation_id for item in selected_results],
            "baseline_evaluation_ids": [baseline_result.evaluation_id],
            "selected_splits": selected_splits,
            "selected_metrics": selected_metrics,
            "tables": exported_tables,
        }
        write_table_export_json(args.export, export_payload)


if __name__ == "__main__":
    main()
