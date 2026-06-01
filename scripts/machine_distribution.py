#!/usr/bin/env python3
"""Build class/machine image distribution and save a grouped bar chart."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np


MACHINE_ALIASES = {
    "smiling-donkey2": "smiling-donkey",
}

COMPARISON_CLASS_ORDER = [
    "normal_wood",
    "impregnated_wood",
    "hard_plastic",
    "soft_plastic",
]


def normalize_machine_name(machine_name: str) -> str:
    """Normalize machine aliases so equivalent machines are grouped together."""
    return MACHINE_ALIASES.get(machine_name, machine_name)


def order_class_names(class_names: list[str]) -> list[str]:
    """Apply domain-specific class ordering for easier wood/plastic comparisons."""
    required_classes = set(COMPARISON_CLASS_ORDER)
    present_classes = set(class_names)

    if required_classes.issubset(present_classes):
        remaining = [name for name in class_names if name not in required_classes]
        return COMPARISON_CLASS_ORDER + remaining

    return class_names


def extract_machine_name(file_path: Path) -> str | None:
    """Extract machine name from a filename prefixed as img_<machine>_...png."""
    stem = file_path.stem
    if not stem.startswith("img_"):
        return None

    remainder = stem[len("img_") :]
    machine = remainder.split("_", maxsplit=1)[0]
    if not machine:
        return None
    return machine


def extract_class_name(file_path: Path) -> str | None:
    """Determine class from parent folder, or parent of images/ folder."""
    parent = file_path.parent
    if parent.name == "images":
        if parent.parent == parent:
            return None
        return parent.parent.name
    return parent.name


def collect_distribution(input_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Collect png files into class -> machine -> list[path]."""
    distribution: DefaultDict[str, DefaultDict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    png_paths = sorted(input_dir.rglob("*.png"))
    png_paths.extend(sorted(input_dir.rglob("*.PNG")))

    for png_path in png_paths:
        class_name = extract_class_name(png_path)
        machine_name = extract_machine_name(png_path)

        if class_name is None or machine_name is None:
            continue

        machine_name = normalize_machine_name(machine_name)

        distribution[class_name][machine_name].append(str(png_path.resolve()))

    # Convert nested defaultdicts to plain dicts for cleaner output handling.
    return {
        class_name: {
            machine_name: paths
            for machine_name, paths in sorted(machine_map.items(), key=lambda item: item[0])
        }
        for class_name, machine_map in sorted(distribution.items(), key=lambda item: item[0])
    }


def plot_distribution(
    distribution: dict[str, dict[str, list[str]]], output_path: Path
) -> None:
    """Create grouped bar chart for machine counts in each class."""
    if not distribution:
        raise ValueError("No valid .png files were found to plot.")

    class_names = order_class_names(list(distribution.keys()))
    machine_names = sorted(
        {
            machine
            for machine_map in distribution.values()
            for machine in machine_map.keys()
        }
    )

    if not machine_names:
        raise ValueError("No machine names could be extracted from filenames.")

    x_positions = np.arange(len(class_names), dtype=float)
    group_width = 0.8
    bar_width = group_width / len(machine_names)

    fig_width = max(10, int(len(class_names) * 1.7))
    fig, ax = plt.subplots(figsize=(fig_width + 4, 6))

    # Ensure palette has at least 20 colors so it does not loop for large machine sets.
    palette_size = max(20, len(machine_names))
    palette = plt.colormaps["nipy_spectral"](np.linspace(0.05, 0.95, palette_size))

    for idx, machine_name in enumerate(machine_names):
        counts = [len(distribution[class_name].get(machine_name, [])) for class_name in class_names]
        offset = (idx - (len(machine_names) - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset,
            counts,
            width=bar_width * 0.95,
            label=machine_name,
            color=palette[idx],
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Image Distribution by Class and Machine")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.legend(
        title="Machine",
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan .png images, group by class and machine name, "
            "and save a grouped bar chart."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory to recursively search for .png files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the chart will be saved (default: current working directory).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="machine_distribution.png",
        help="Name of the output chart file (default: machine_distribution.png).",
    )
    parser.add_argument(
        "--print-dict",
        action="store_true",
        help="Print the full class -> machine -> paths dictionary.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    distribution = collect_distribution(input_dir)
    output_path = args.output_dir.resolve() / args.output_name
    plot_distribution(distribution, output_path)

    total_classes = len(distribution)
    total_machines = len(
        {
            machine
            for machine_map in distribution.values()
            for machine in machine_map.keys()
        }
    )
    total_images = sum(
        len(paths)
        for machine_map in distribution.values()
        for paths in machine_map.values()
    )

    print(f"Saved chart: {output_path}")
    print(f"Classes: {total_classes}")
    print(f"Machines: {total_machines}")
    print(f"Images counted: {total_images}")
    if args.print_dict:
        print("\nDictionary structure:")
        print(distribution)


if __name__ == "__main__":
    main()
