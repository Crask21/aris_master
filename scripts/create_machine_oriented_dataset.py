#!/usr/bin/env python3
"""Copy images into a machine/category-oriented dataset layout."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

MACHINE_NAME_RE = re.compile(r"^img_(?P<machine>[A-Za-z0-9]+-[A-Za-z0-9]+)_")


@dataclass(frozen=True)
class ImageCopyPlan:
    source: Path
    destination: Path
    machine: str
    category: str


def extract_machine_name(image_path: Path) -> str | None:
    """Extract the adjective-animal machine name from img_<machine>_<timestamp>."""
    match = MACHINE_NAME_RE.match(image_path.name)
    if match is None:
        return None
    return match.group("machine")


def extract_category(image_path: Path, input_dir: Path) -> str | None:
    """Use the grandparent folder when the parent is images, otherwise the parent."""
    parent = image_path.parent
    category_path = parent.parent if parent.name == "images" else parent

    try:
        category_path.relative_to(input_dir)
    except ValueError:
        return None

    if category_path == input_dir:
        return None

    return category_path.name


def iter_image_paths(input_dir: Path) -> list[Path]:
    """Return all supported image files under input_dir."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_destination(
    output_dir: Path,
    machine: str,
    category: str,
    source: Path,
    overwrite: bool,
) -> Path:
    """Build the target path, adding a numeric suffix if needed."""
    destination = output_dir / machine / category / source.name
    if overwrite or not destination.exists():
        return destination

    counter = 2
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def build_copy_plan(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> tuple[list[ImageCopyPlan], Counter[str]]:
    """Scan input_dir and prepare copy operations."""
    skipped: Counter[str] = Counter()
    plans: list[ImageCopyPlan] = []

    for image_path in iter_image_paths(input_dir):
        machine = extract_machine_name(image_path)
        if machine is None:
            skipped["missing_machine_name"] += 1
            continue

        category = extract_category(image_path, input_dir)
        if category is None:
            skipped["missing_category"] += 1
            continue

        destination = resolve_destination(
            output_dir=output_dir,
            machine=machine,
            category=category,
            source=image_path,
            overwrite=overwrite,
        )
        plans.append(
            ImageCopyPlan(
                source=image_path,
                destination=destination,
                machine=machine,
                category=category,
            )
        )

    return plans, skipped


def copy_images(plans: list[ImageCopyPlan], dry_run: bool) -> None:
    """Execute a list of image copy operations."""
    for plan in plans:
        if dry_run:
            print(f"{plan.source} -> {plan.destination}")
            continue

        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.source, plan.destination)


def print_summary(plans: list[ImageCopyPlan], skipped: Counter[str], dry_run: bool) -> None:
    """Print counts for copied images, machines, categories, and skipped files."""
    machines = sorted({plan.machine for plan in plans})
    categories = sorted({plan.category for plan in plans})

    action = "Would copy" if dry_run else "Copied"
    print(f"\n{action} {len(plans)} images")
    print(f"Machines: {len(machines)}")
    print(f"Categories: {len(categories)}")

    if machines:
        print("Machine names:")
        for machine in machines:
            print(f"  {machine}")

    if categories:
        print("Categories:")
        for category in categories:
            print(f"  {category}")

    if skipped:
        print("Skipped files:")
        for reason, count in skipped.items():
            print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy images from a class-oriented dataset into machine/category folders. "
            "Machine names are extracted from filenames like "
            "img_brave-panther_2024-12-30T12-38-45-481.png."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory containing the current dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the machine-oriented dataset will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of adding a numeric suffix.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different.")

    plans, skipped = build_copy_plan(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    copy_images(plans, dry_run=args.dry_run)
    print_summary(plans, skipped, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
