#!/usr/bin/env python3
"""Plot TensorBoard metrics across multiple runs.

Example:
  python scripts/plot_tensorboard_metrics.py \
    --runs /path/to/run1 /path/to/run2 \
    --tags train/acc val/acc train/loss val/loss \
    --title "ResNet Training Curves" \
    --out figures/training_curves.png
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class ScalarSeries:
    steps: List[int]
    values: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard scalar metrics across multiple runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to run directories (each containing TensorBoard event files).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["train/acc", "val/acc", "train/loss", "val/loss"],
        help="Scalar tags to plot (default: train/acc val/acc train/loss val/loss).",
    )
    parser.add_argument(
        "--title",
        default="Training Curves",
        help="Figure title.",
    )
    parser.add_argument(
        "--out",
        default="training_curves.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--font",
        default="/home/ap/cloud/master/aris_master/figures/font/cmunbx.ttf",
        help="Path to CMU Serif font file.",
    )
    parser.add_argument(
        "--separate",
        "--seperate",
        dest="separate",
        action="store_true",
        help="Also export each tag as a separate figure.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="EMA smoothing factor in (0,1]; 0 disables smoothing.",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.3,
        help="Opacity for raw curves when EMA is enabled.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print summary stats for each run/tag.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=8.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI.",
    )
    return parser.parse_args()


def has_event_files(dir_path: str) -> bool:
    try:
        for name in os.listdir(dir_path):
            if "tfevents" in name:
                return True
    except FileNotFoundError:
        return False
    return False


def discover_event_dirs(run_dir: str) -> List[str]:
    if has_event_files(run_dir):
        return [run_dir]

    event_dirs: List[str] = []
    for dirpath, _, filenames in os.walk(run_dir):
        if any("tfevents" in name for name in filenames):
            event_dirs.append(dirpath)
    return event_dirs


def normalize_run_dirs(run_dirs: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    normalized: List[str] = []
    labels: Dict[str, str] = {}

    for run_dir in run_dirs:
        if os.path.isfile(run_dir) and "tfevents" in os.path.basename(run_dir):
            run_dir = os.path.dirname(run_dir)

        discovered = discover_event_dirs(run_dir)
        if not discovered:
            continue

        for event_dir in discovered:
            if event_dir in labels:
                continue
            normalized.append(event_dir)
            rel_label = os.path.relpath(event_dir, start=run_dir)
            if rel_label == ".":
                rel_label = os.path.basename(os.path.normpath(event_dir))

            seed_match = re.search(r"run\s*([0-9]+)", rel_label, re.IGNORECASE)
            if seed_match:
                labels[event_dir] = f"Seed {seed_match.group(1)}"
            else:
                labels[event_dir] = rel_label

    return normalized, labels


def resolve_tag(available: Set[str], requested: str) -> str | None:
    if requested in available:
        return requested

    tokens = [
        token
        for token in re.split(r"[\s/_-]+", requested.lower())
        if token
    ]
    if not tokens:
        return None

    alias_map = {
        "acc": ["acc", "accuracy"],
        "accuracy": ["accuracy", "acc"],
        "val": ["val", "valid", "validation"],
        "valid": ["val", "valid", "validation"],
        "validation": ["val", "valid", "validation"],
        "train": ["train", "training"],
        "training": ["train", "training"],
        "loss": ["loss"],
    }

    groups = [alias_map.get(token, [token]) for token in tokens]

    matches = []
    for tag in available:
        lower = tag.lower()
        if all(any(alias in lower for alias in group) for group in groups):
            matches.append(tag)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return sorted(matches, key=len)[0]


def load_scalars(
    run_dir: str, tags: Iterable[str]
) -> Tuple[Dict[str, ScalarSeries], Set[str], Dict[str, str]]:
    accumulator = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    series: Dict[str, ScalarSeries] = {}
    resolved: Dict[str, str] = {}
    for tag in tags:
        resolved_tag = resolve_tag(available, tag)
        if resolved_tag is None:
            continue
        if resolved_tag != tag:
            resolved[tag] = resolved_tag
        events = accumulator.Scalars(resolved_tag)
        series[tag] = ScalarSeries(
            steps=[event.step for event in events],
            values=[event.value for event in events],
        )
    return series, available, resolved


def ema_smooth(values: List[float], alpha: float) -> List[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def display_label(tag: str) -> str:
    label_map = {
        "train/acc": "Training Accuracy",
        "val/acc": "Validation Accuracy",
        "train/loss": "Training Loss",
        "val/loss": "Validation Loss",
    }
    if tag in label_map:
        return label_map[tag]
    return tag.replace("_", " ").replace("/", " ").title()


def run_sort_key(run_dir: str, run_labels: Dict[str, str]) -> Tuple[int, int, str]:
    label = run_labels.get(run_dir, "")
    match = re.search(r"seed\s*([0-9]+)", label, re.IGNORECASE)
    if match:
        return (1, int(match.group(1)), label.lower())
    return (0, 0, label.lower())


def configure_matplotlib(font_path: str) -> None:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams.update(
            {
                "font.family": font_name,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )


def plot_runs(
    run_dirs: List[str],
    run_labels: Dict[str, str],
    tags: List[str],
    title: str,
    out_path: str,
    width: float,
    height: float,
    dpi: int,
    font_path: str,
    separate: bool,
    ema_alpha: float,
    raw_alpha: float,
    verbose: bool,
) -> None:
    configure_matplotlib(font_path)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(width, height),
        constrained_layout=True,
    )
    axes_map = {
        "train/acc": axes[0, 0],
        "val/acc": axes[0, 1],
        "train/loss": axes[1, 0],
        "val/loss": axes[1, 1],
    }

    total_series = 0
    for run_dir in run_dirs:
        run_name = run_labels.get(
            run_dir, os.path.basename(os.path.normpath(run_dir))
        )
        series_map, available, resolved = load_scalars(run_dir, tags)

        if verbose:
            print(f"Run: {run_name}")
            if available:
                print(f"  Available tags: {sorted(available)}")
            else:
                print("  Available tags: none")
            for requested, actual in resolved.items():
                print(f"  Resolved tag: {requested} -> {actual}")

        for tag in tags:
            if tag not in series_map:
                if verbose:
                    print(f"  Missing tag: {tag}")
                continue
            axis = axes_map.get(tag)
            if axis is None:
                continue
            series = series_map[tag]
            if ema_alpha > 0.0:
                axis.plot(
                    series.steps,
                    series.values,
                    label="_nolegend_",
                    alpha=raw_alpha,
                )
                smoothed = ema_smooth(series.values, ema_alpha)
                axis.plot(series.steps, smoothed, label=run_name)
            else:
                axis.plot(series.steps, series.values, label=run_name)
            total_series += 1
            if verbose:
                max_val = max(series.values)
                min_val = min(series.values)
                last_val = series.values[-1]
                if "loss" in tag:
                    print(
                        f"  {tag}: min={min_val:.6f} max={max_val:.6f} last={last_val:.6f}"
                    )
                elif "acc" in tag or "accuracy" in tag:
                    print(
                        f"  {tag}: max={max_val:.6f} min={min_val:.6f} last={last_val:.6f}"
                    )
                else:
                    print(
                        f"  {tag}: min={min_val:.6f} max={max_val:.6f} last={last_val:.6f}"
                    )

    if total_series == 0:
        raise RuntimeError(
            "No matching scalar data found. Check run paths and tag names."
        )

    for tag, axis in axes_map.items():
        if tag not in tags:
            axis.set_visible(False)
            continue
        axis.set_title(display_label(tag))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy (%)" if "acc" in tag else "Loss")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(frameon=False)

    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if separate:
        base, ext = os.path.splitext(out_path)
        for tag in tags:
            axis = axes_map.get(tag)
            if axis is None:
                continue
            if not axis.get_lines():
                continue
            fig_sep, ax_sep = plt.subplots(figsize=(width / 1.2, height / 1.2))
            for line in axis.get_lines():
                ax_sep.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=line.get_label(),
                )
            ax_sep.set_title(display_label(tag))
            ax_sep.set_xlabel("Epoch")
            ax_sep.set_ylabel("Accuracy (%)" if "acc" in tag else "Loss")
            ax_sep.grid(True, linestyle="--", alpha=0.4)
            ax_sep.legend(frameon=False)
            safe_tag = tag.replace("/", "_")
            fig_sep.savefig(f"{base}_{safe_tag}{ext}", dpi=dpi, bbox_inches="tight")



def main() -> None:
    args = parse_args()
    run_dirs, run_labels = normalize_run_dirs(args.runs)
    if not run_dirs:
        raise RuntimeError("No TensorBoard event files found under --runs paths.")

    run_dirs = sorted(
        run_dirs,
        key=lambda run_dir: run_sort_key(run_dir, run_labels),
        reverse=True,
    )

    plot_runs(
        run_dirs=run_dirs,
        run_labels=run_labels,
        tags=args.tags,
        title=args.title,
        out_path=args.out,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        font_path=args.font,
        separate=args.separate,
        ema_alpha=args.ema_alpha,
        raw_alpha=args.raw_alpha,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
