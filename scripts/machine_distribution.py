#!/usr/bin/env python3
"""Build class/machine image distribution and save a grouped bar chart."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


# Plot style configuration.
CMU_SERIF_FONT_PATH = Path("figures/font/cmunrm.ttf")
TITLE_STRING = "Image Distribution by Class and Machine"
Y_LABEL_STRING = "Number of Images"
TICK_FONT_SIZE = 15
AXIS_TITLE_FONT_SIZE = 17
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = TICK_FONT_SIZE
LEGEND_TITLE_FONT_SIZE = TICK_FONT_SIZE

# Use a matplotlib colormap name, or replace with a list of colors such as
# ["#1b9e77", "#d95f02", "#7570b3"].
COLOR_PALETTE = "nipy_spectral"

MACHINE_ALIASES = {
    "smiling-donkey2": "smiling-donkey",
}

COMPARISON_CLASS_ORDER = [
    "normal_wood",
    "impregnated_wood",
    "hard_plastic",
    "soft_plastic",
]

CLASS_DISPLAY_NAMES = {
    "normal_wood": "Normal Wood",
    "impregnated_wood": "Impregnated Wood",
    "hard_plastic": "Hard Plastic",
    "soft_plastic": "Soft Plastic",
}


def repo_root() -> Path:
    """Return repository root based on this script's location."""
    return Path(__file__).resolve().parents[1]


def configure_plot_style() -> None:
    """Configure matplotlib font defaults for the generated chart."""
    font_path = CMU_SERIF_FONT_PATH
    if not font_path.is_absolute():
        font_path = repo_root() / font_path

    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams.update(
            {
                "font.family": font_name,
                "font.serif": [font_name],
            }
        )
    else:
        plt.rcParams.update({"font.family": "serif"})


def make_palette(machine_count: int):
    """Return enough colors for all machines from the configured palette."""
    palette_size = max(20, machine_count)

    if isinstance(COLOR_PALETTE, str):
        return plt.colormaps[COLOR_PALETTE](np.linspace(0.05, 0.95, palette_size))

    return [COLOR_PALETTE[idx % len(COLOR_PALETTE)] for idx in range(machine_count)]


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


def display_class_name(class_name: str) -> str:
    """Return publication-friendly class label for a class folder name."""
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)


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


def extract_machine_name_from_annot(file_path: Path) -> str | None:
    """Extract machine name from annot_<machine>_...json."""
    stem = file_path.stem
    if not stem.startswith("annot_"):
        return None

    remainder = stem[len("annot_") :]
    machine = remainder.split("_", maxsplit=1)[0]
    if not machine:
        return None
    return machine


def extract_class_name(file_path: Path) -> str | None:
    """Determine class from parent folder, or parent of images/annots folder."""
    parent = file_path.parent
    if parent.name in {"images", "annots"}:
        if parent.parent == parent:
            return None
        return parent.parent.name
    return parent.name


def collect_distribution(input_dir: Path, use_annots: bool = False) -> dict[str, dict[str, list[str]]]:
    """Collect image or annotation files into class -> machine -> list[path]."""
    distribution: DefaultDict[str, DefaultDict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    if use_annots:
        file_paths = sorted(input_dir.rglob("*.json"))
        extract_machine = extract_machine_name_from_annot
    else:
        file_paths = sorted(input_dir.rglob("*.png"))
        file_paths.extend(sorted(input_dir.rglob("*.PNG")))
        extract_machine = extract_machine_name

    for file_path in file_paths:
        class_name = extract_class_name(file_path)
        machine_name = extract_machine(file_path)

        if class_name is None or machine_name is None:
            continue

        machine_name = normalize_machine_name(machine_name)

        distribution[class_name][machine_name].append(str(file_path.resolve()))

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
        raise ValueError("No valid files were found to plot.")

    configure_plot_style()
    class_names = order_class_names(list(distribution.keys()))
    class_display_names = [display_class_name(class_name) for class_name in class_names]
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

    palette = make_palette(len(machine_names))

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

    ax.set_xlabel("Class", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel(Y_LABEL_STRING, fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_title(TITLE_STRING, fontsize=TITLE_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_display_names, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    legend = ax.legend(
        title="Machine",
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    legend.get_title().set_fontsize(LEGEND_TITLE_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_distribution_matrix(distribution: dict[str, dict[str, list[str]]]) -> None:
    """Print machine rows against class columns with row totals."""
    if not distribution:
        print("No valid files were found.")
        return

    class_names = order_class_names(list(distribution.keys()))
    class_display_names = [display_class_name(class_name) for class_name in class_names]
    machine_names = sorted(
        {
            machine
            for machine_map in distribution.values()
            for machine in machine_map.keys()
        }
    )

    counts_by_machine = {
        machine_name: [
            len(distribution[class_name].get(machine_name, []))
            for class_name in class_names
        ]
        for machine_name in machine_names
    }

    headers = ["machine", *class_display_names, "total"]
    rows = [
        [
            machine_name,
            *[str(count) for count in counts],
            str(sum(counts)),
        ]
        for machine_name, counts in counts_by_machine.items()
    ]
    # blushing-lion  brave-panther  curious-fox     energetic-kangaroo  gleaming-tiger  majestic-horse  playful-monkey  smiling-donkey   smoldering-whale   bold-eagle     cheerful-swan  daring-leopard  gallant-stag        jolly-giraffe   mysterious-owl  radiant-bear    smiling-donkey2

    # --sources blushing-lion --sources brave-panther --sources curious-fox --sources energetic-kangaroo --sources gleaming-tiger --sources majestic-horse --sources playful-monkey --sources smiling-donkey --sources smoldering-whale --sources bold-eagle --sources cheerful-swan --sources daring-leopard --sources gallant-stag --sources jolly-giraffe --sources mysterious-owl --sources radiant-bear --sources smiling-donkey2

    total_counts = [
        sum(counts[class_idx] for counts in counts_by_machine.values())
        for class_idx in range(len(class_names))
    ]
    rows.append(["total", *[str(count) for count in total_counts], str(sum(total_counts))])

    column_widths = [
        max(len(row[column_idx]) for row in [headers, *rows])
        for column_idx in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return "  ".join(
            value.ljust(column_widths[idx])
            if idx == 0
            else value.rjust(column_widths[idx])
            for idx, value in enumerate(row)
        )

    print(format_row(headers))
    print(format_row(["-" * width for width in column_widths]))
    for row in rows:
        print(format_row(row))


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
        "--use-annots",
        action="store_true",
        help="Build distribution from annot_*.json files instead of images.",
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

    distribution = collect_distribution(input_dir, use_annots=args.use_annots)
    print_distribution_matrix(distribution)

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
