#!/usr/bin/env python3
"""Statistical tests between two methods per split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from tabulate import tabulate

# Edit these defaults if you want to run without CLI args.
METHOD1_RESULTS = (
    "/home/ap/cloud/master/aris_master/Testing/redo/reevaluated_results/1000_real/"
    "conditional_diffusion/results.json"
)
METHOD2_RESULTS = (
    "/home/ap/cloud/master/aris_master/Testing/redo/1000_real/"
    "unconditioned_diffusion/results.json"
)
DEFAULT_METRIC = "test_f1_macro"
DEFAULT_STAT_TEST = "welch_ttest"
DEFAULT_ONE_SIDED_DIRECTION = "greater"


@dataclass(frozen=True)
class TestResult:
    statistic_name: str
    statistic: float
    p_value: float
    df: float | None = None


def load_metric_by_split(path: str, metric: str) -> Tuple[str, Dict[str, Dict[str, object]]]:
    data = json.loads(Path(path).read_text())
    splits: Dict[str, Dict[str, object]] = {}
    method_name = data.get("evaluation_id", Path(path).parent.name)

    for split in data.get("splits", []):
        values: List[float] = []
        for run in split.get("runs", []):
            if metric in run:
                values.append(float(run[metric]))

        if not values:
            continue

        splits[split["split_name"]] = {
            "values": np.array(values, dtype=float),
            "synthetic_image_count": split.get("synthetic_image_count"),
        }

    return method_name, splits


def normalize_stat_test(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "wilcoxon": "wilcoxon",
        "wilcoxon_signed_rank": "wilcoxon",
        "wilcoxon_signed_rank_test": "wilcoxon",
        "signed_rank": "wilcoxon",
        "normal": "student_ttest",
        "normal_ttest": "student_ttest",
        "normal_t_test": "student_ttest",
        "student": "student_ttest",
        "student_ttest": "student_ttest",
        "student_t_test": "student_ttest",
        "independent_ttest": "student_ttest",
        "independent_t_test": "student_ttest",
        "ttest": "student_ttest",
        "t_test": "student_ttest",
        "ttest_ind": "student_ttest",
        "paired": "paired_ttest",
        "paired_ttest": "paired_ttest",
        "paired_t_test": "paired_ttest",
        "ttest_rel": "paired_ttest",
        "welch": "welch_ttest",
        "welch_ttest": "welch_ttest",
        "welch_t_test": "welch_ttest",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(set(aliases.values())))
        raise argparse.ArgumentTypeError(
            f"Unknown statistical test '{value}'. Valid tests: {valid}"
        ) from exc


def welch_ttest(a: np.ndarray, b: np.ndarray, alternative: str) -> TestResult:
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)

    # Welch-Satterthwaite degrees of freedom
    sa2 = np.var(a, ddof=1)
    sb2 = np.var(b, ddof=1)
    na, nb = len(a), len(b)
    num = (sa2 / na + sb2 / nb) ** 2
    den = (sa2**2) / (na**2 * (na - 1)) + (sb2**2) / (nb**2 * (nb - 1))
    df = num / den if den != 0 else float("inf")

    return TestResult("t", float(t_stat), float(p_val), float(df))


def student_ttest(a: np.ndarray, b: np.ndarray, alternative: str) -> TestResult:
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=True, alternative=alternative)
    df = len(a) + len(b) - 2
    return TestResult("t", float(t_stat), float(p_val), float(df))


def paired_ttest(a: np.ndarray, b: np.ndarray, alternative: str) -> TestResult:
    if len(a) != len(b):
        raise ValueError(
            "Paired t-test requires paired samples with equal lengths "
            f"(got {len(a)} and {len(b)})."
        )

    t_stat, p_val = stats.ttest_rel(a, b, alternative=alternative)
    df = len(a) - 1
    return TestResult("t", float(t_stat), float(p_val), float(df))


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray, alternative: str) -> TestResult:
    if len(a) != len(b):
        raise ValueError(
            "Wilcoxon signed-rank test requires paired samples with equal lengths "
            f"(got {len(a)} and {len(b)})."
        )

    if len(a) == 0:
        raise ValueError("Wilcoxon signed-rank test requires at least one paired sample.")

    if np.allclose(a, b):
        return TestResult("W", 0.0, 1.0)

    statistic, p_val = stats.wilcoxon(a, b, alternative=alternative)
    return TestResult("W", float(statistic), float(p_val))


def run_stat_test(
    stat_test: str, a: np.ndarray, b: np.ndarray, alternative: str
) -> TestResult:
    if stat_test == "wilcoxon":
        return wilcoxon_signed_rank(a, b, alternative)
    if stat_test == "student_ttest":
        return student_ttest(a, b, alternative)
    if stat_test == "paired_ttest":
        return paired_ttest(a, b, alternative)
    if stat_test == "welch_ttest":
        return welch_ttest(a, b, alternative)
    raise ValueError(f"Unhandled statistical test: {stat_test}")


def summary_stats(values: np.ndarray) -> Tuple[int, float, float]:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return n, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical tests between two results.json files",
    )
    parser.add_argument("--method1", default=METHOD1_RESULTS)
    parser.add_argument("--method2", default=METHOD2_RESULTS)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--stat-test",
        type=normalize_stat_test,
        default=DEFAULT_STAT_TEST,
        help=(
            "Statistical test to run per split. Accepted aliases include: wilcoxon, "
            "wilcoxon_signed_rank, welch_ttest, welch, normal_ttest, student_ttest, "
            "paired_ttest. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--one-sided",
        action="store_true",
        help=(
            "Use a one-sided alternative hypothesis. By default this tests whether "
            "method1 is greater than method2."
        ),
    )
    parser.add_argument(
        "--one-sided-direction",
        choices=["greater", "less"],
        default=DEFAULT_ONE_SIDED_DIRECTION,
        help=(
            "Direction to use with --one-sided: 'greater' tests method1 > method2, "
            "'less' tests method1 < method2. Defaults to %(default)s."
        ),
    )
    args = parser.parse_args()
    alternative = args.one_sided_direction if args.one_sided else "two-sided"

    method1_name, method1 = load_metric_by_split(args.method1, args.metric)
    method2_name, method2 = load_metric_by_split(args.method2, args.metric)

    shared_splits = [s for s in method1.keys() if s in method2]
    shared_splits.sort(
        key=lambda s: (
            method1[s].get("synthetic_image_count")
            if method1[s].get("synthetic_image_count") is not None
            else s
        )
    )

    rows = []
    has_df = False
    statistic_name = ""
    for split_name in shared_splits:
        a = method1[split_name]["values"]
        b = method2[split_name]["values"]
        n1, mean1, std1 = summary_stats(a)
        n2, mean2, std2 = summary_stats(b)
        print(f"Running {args.stat_test} on split {a}")
        print(f"Running {args.stat_test} on split {b}")
        result = run_stat_test(args.stat_test, a, b, alternative)
        has_df = has_df or result.df is not None
        statistic_name = result.statistic_name

        row = [
            split_name,
            n1,
            f"{mean1:.6f} +/- {std1:.6f}",
            n2,
            f"{mean2:.6f} +/- {std2:.6f}",
            f"{result.statistic:.4f}",
        ]
        if result.df is not None:
            row.append(f"{result.df:.2f}")
        row.append(f"{result.p_value:.6f}")
        rows.append(row)

    if not rows:
        print("No overlapping splits found or metric missing.")
        return

    stat_test_label = {
        "wilcoxon": "Wilcoxon signed-rank",
        "student_ttest": "Student t-test",
        "paired_ttest": "Paired t-test",
        "welch_ttest": "Welch t-test",
    }[args.stat_test]
    if alternative != "two-sided":
        stat_test_label = f"{stat_test_label}, one-sided {alternative}"

    headers = [
        "Split",
        f"N ({method1_name})",
        f"Mean +/- Std ({method1_name})",
        f"N ({method2_name})",
        f"Mean +/- Std ({method2_name})",
        f"{statistic_name} ({stat_test_label})",
    ]
    if has_df:
        headers.append("df")
    headers.append("p")
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
