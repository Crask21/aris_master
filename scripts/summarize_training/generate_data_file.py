"""
Generate data_files.json from UNet config file.

This script reads a UNet configuration JSON file and generates a data_files.json
file that maps subcategories to their image file paths across train/val/test splits.

Key differences from generate_data_yaml_with_log.py:
- Input: Single UNet config JSON instead of separate input_folders and class_separation YAMLs
- Categories: Uses subcategories from data.categories in config, ignoring parent category keys
- Splits: Discovers train/val/test folders automatically from the source directory
- Output: JSON format instead of YAML, defaults to config.directory/data_files.json

Usage:
    python generate_data_json.py -c path/to/config.json [--out output.json] [--verbose]

Example:
    # Generate data_files.json in the config.directory location
    python generate_data_json.py -c training/unet-wood/config.json --verbose
    
    # Generate with custom output path
    python generate_data_json.py -c config.json --out /tmp/data_files.json

The UNet config should have this structure:
    {
        "config": {
            "directory": "/path/to/training/output"
        },
        "data": {
            "source": "/path/to/dataset",
            "categories": {
                "parent_category1": ["subcategory1", "subcategory2", ...],
                "parent_category2": ["subcategory3", "subcategory4", ...]
            }
        }
    }

The source directory should have this structure:
    source/
        train/
            subcategory1/
                images/
                    img1.png
                    img2.png
            subcategory2/
                images/
                    img3.png
        val/
            subcategory1/
                images/
                    img4.png
        test/
            ...
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Set
import re
import datetime

# ---------------------- JSON loading & helpers ----------------------

def load_json(path: Path):
    """Load JSON configuration file."""
    if not path.exists():
        sys.exit(f"ERROR: File not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: Failed to parse JSON '{path}': {e}")

def normalize(name: str) -> str:
    """Normalize directory/class names for matching."""
    s = name.strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
    s = re.sub(r"_+", "_", s)
    return s

def extract_subcategories(categories: Dict[str, List[str]]) -> List[str]:
    """Extract all subcategories from the categories dict, ignoring parent category keys."""
    subcategories = []
    for parent_cat, subcat_list in categories.items():
        if subcat_list and isinstance(subcat_list, list):
            subcategories.extend(subcat_list)
    return subcategories

def list_images_in_dir(d: Path, allowed_exts: Set[str], recursive: bool) -> List[Path]:
    """List all image files in a directory."""
    if not d.exists() or not d.is_dir():
        return []
    if recursive:
        it = d.rglob("*")
    else:
        it = d.iterdir()
    return [p.resolve() for p in it if p.is_file() and p.suffix.lower() in allowed_exts]

# ---------------------- Main data building ----------------------

def build_data(
    base_dir: Path,
    subcategories: List[str],
    splits: List[str],
    allowed_exts: Set[str],
    prefer_images_subdir: str,
    ignore_subdirs: Set[str],
    recursive_within_images: bool
) -> Dict[str, Dict[str, List[str]]]:
    """Build the data structure by scanning for subcategories in train/val/test folders."""
    
    # Create normalized lookup for subcategories
    subcat_normalized = {normalize(sc): sc for sc in subcategories}
    
    # Initialize data structure
    data = {split: {sc: [] for sc in subcategories} for split in splits}
    
    for split in splits:
        split_path = (base_dir / split).resolve()
        if not split_path.exists() or not split_path.is_dir():
            print(f"WARNING: {split} folder not found or not a directory: {split_path}", file=sys.stderr)
            continue
        
        # Iterate through subdirectories in the split folder
        for subdir in split_path.iterdir():
            if not subdir.is_dir():
                continue
            
            subdir_name = subdir.name
            key = normalize(subdir_name)
            
            # Check if this directory matches any subcategory
            if key not in subcat_normalized:
                continue
            
            subcategory = subcat_normalized[key]
            
            # Look for images
            images_root = subdir / prefer_images_subdir if prefer_images_subdir else subdir
            if prefer_images_subdir and images_root.exists() and images_root.is_dir():
                imgs = list_images_in_dir(images_root, allowed_exts, recursive_within_images)
            else:
                # Look in subdir directly and subdirs (except ignored ones)
                imgs = [p.resolve() for p in subdir.iterdir() 
                       if p.is_file() and p.suffix.lower() in allowed_exts]
                for child in subdir.iterdir():
                    if child.is_dir() and normalize(child.name) not in ignore_subdirs:
                        imgs.extend(list_images_in_dir(child, allowed_exts, recursive=False))
            
            if imgs:
                data[split][subcategory].extend(str(p) for p in imgs)
        
        # Sort and deduplicate
        for sc, paths in data[split].items():
            data[split][sc] = sorted(set(paths))
    
    return data

# ---------------------- Summary helpers ----------------------

def compute_summary(data: Dict[str, Dict[str, List[str]]]):
    """Compute summary statistics for the dataset."""
    splits = list(data.keys())
    totals_by_split = {split: sum(len(v) for v in data[split].values()) for split in splits}
    overall = sum(totals_by_split.values()) or 1
    
    split_summary = []
    for split in splits:
        split_summary.append({
            "split": split,
            "count": totals_by_split[split],
            "share_overall": totals_by_split[split] / overall
        })
    
    class_summary = []
    for split in splits:
        split_total = totals_by_split[split] or 1
        split_classes = []
        for cls, items in data[split].items():
            count = len(items)
            if count > 0:  # Filter out zero-count categories
                split_classes.append({
                    "split": split,
                    "class": cls,
                    "count": count,
                    "share_of_split": (count / split_total)
                })
        # Sort by count descending
        split_classes.sort(key=lambda x: x["count"], reverse=True)
        class_summary.extend(split_classes)
    
    return split_summary, class_summary

def format_pct(x: float) -> str:
    """Format percentage."""
    return f"{x*100:.2f}%"

def _print_table(headers: List[str], rows: List[List[str]], logf=None):
    """Print a formatted table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    
    lines = []
    lines.append(fmt_row(headers))
    lines.append("  ".join("-"*w for w in widths))
    for r in rows:
        lines.append(fmt_row(r))
    
    for line in lines:
        print(line)
        if logf:
            logf.write(line + "\n")

def print_summary(split_summary, class_summary, splits):
    """Print summary statistics and return formatted string."""
    lines = []
    
    # Split level
    hdr = ["Split", "Count", "Share of overall"]
    rows = [[r["split"], str(r["count"]), format_pct(r["share_overall"])] for r in split_summary]
    print("\n=== Summary: by split ===")
    lines.append("=== Summary: by split ===")
    
    # Capture table for both print and return
    widths = [len(h) for h in hdr]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    def fmt_row(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    
    header_line = fmt_row(hdr)
    separator = "  ".join("-"*w for w in widths)
    print(header_line)
    print(separator)
    lines.append(header_line)
    lines.append(separator)
    
    for row in rows:
        row_line = fmt_row(row)
        print(row_line)
        lines.append(row_line)
    
    # Class level
    for split in splits:
        hdr = ["Class", "Count", "Share of split"]
        split_rows = []
        for r in class_summary:
            if r["split"] == split:
                split_rows.append([r["class"], str(r["count"]), format_pct(r["share_of_split"])])
        
        print(f"\n=== Summary: by class (within {split}) ===")
        lines.append("")
        lines.append(f"=== Summary: by class (within {split}) ===")
        
        widths = [len(h) for h in hdr]
        for row in split_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        header_line = fmt_row(hdr)
        separator = "  ".join("-"*w for w in widths)
        print(header_line)
        print(separator)
        lines.append(header_line)
        lines.append(separator)
        
        for row in split_rows:
            row_line = fmt_row(row)
            print(row_line)
            lines.append(row_line)
    
    return "\n".join(lines)

def write_summary_to_notes(summary_text: str, config_dir: Path, timestamp: str):
    """Write or update summary in notes.md file under ## Data Summary section."""
    notes_path = config_dir / "notes.md"
    
    # Format the summary in a code block
    summary_block = f"\n\n```\n{summary_text}\n```\n\nGenerated: {timestamp}\n"
    
    if notes_path.exists():
        # Read existing content
        content = notes_path.read_text(encoding="utf-8")
        
        # Check if ## Data Summary exists
        if "## Data Summary" in content:
            # Find the section and replace/append to it
            lines = content.split("\n")
            new_lines = []
            in_data_summary = False
            summary_written = False
            
            for i, line in enumerate(lines):
                if line.strip() == "## Data Summary":
                    in_data_summary = True
                    new_lines.append(line)
                    new_lines.append(summary_block)
                    summary_written = True
                elif in_data_summary and line.startswith("## "):
                    # Next section started
                    in_data_summary = False
                    new_lines.append(line)
                elif in_data_summary and not summary_written:
                    # Skip old content in Data Summary section until we hit next section
                    continue
                else:
                    new_lines.append(line)
            
            content = "\n".join(new_lines)
        else:
            # Append new section at the end
            content += f"\n\n## Data Summary{summary_block}"
        
        # Write back
        notes_path.write_text(content, encoding="utf-8")
    else:
        # Create new notes.md
        content = f"## Data Summary{summary_block}"
        notes_path.write_text(content, encoding="utf-8")
    
    print(f"\nUpdated: {notes_path}")

# ---------------------- Main entry ----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate data_files.json from UNet config file"
    )
    parser.add_argument(
        "-c", "--config", 
        required=True, 
        type=Path,
        help="Path to the UNet config JSON file"
    )
    parser.add_argument(
        "--out", 
        default=None, 
        type=Path,
        help="Output JSON file path (default: <config.directory>/data_files.json)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Hardcoded values (previously bloat arguments)
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ignore_subdirs = {"annots", "annotations", "labels"}
    splits = ["train", "val", "test"]
    prefer_images_subdir = "images"
    recursive_within_images = False
    
    # Load config
    config = load_json(args.config)
    
    # Validate config structure
    if "data" not in config or "source" not in config["data"]:
        sys.exit("ERROR: Config must have 'data.source' field")
    if "data" not in config or "categories" not in config["data"]:
        sys.exit("ERROR: Config must have 'data.categories' field")
    
    base_dir = Path(config["data"]["source"]).resolve()
    if not base_dir.exists():
        sys.exit(f"ERROR: Source directory does not exist: {base_dir}")
    
    # Determine output path
    if args.out is None:
        if "config" in config and "directory" in config["config"]:
            output_dir = Path(config["config"]["directory"]).resolve()
            args.out = output_dir / "data_files.json"
        else:
            sys.exit("ERROR: No --out specified and config.directory not found in config")
    
    # Extract subcategories (ignore parent category keys)
    subcategories = extract_subcategories(config["data"]["categories"])
    
    if not subcategories:
        sys.exit("ERROR: No subcategories found in config")
    
    if args.verbose:
        print(f"Base directory: {base_dir}")
        print(f"Subcategories to find: {subcategories}")
        print(f"Splits: {splits}")
    
    # Build data structure
    data = build_data(
        base_dir=base_dir,
        subcategories=subcategories,
        splits=splits,
        allowed_exts=allowed_exts,
        prefer_images_subdir=prefer_images_subdir,
        ignore_subdirs=ignore_subdirs,
        recursive_within_images=recursive_within_images
    )
    
    # Write output
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated: {args.out}")
    
    # Print summary
    split_summary, class_summary = compute_summary(data)
    summary_text = print_summary(split_summary, class_summary, splits)
    
    # Write summary to notes.md if we have a config directory
    if "config" in config and "directory" in config["config"]:
        config_dir = Path(config["config"]["directory"]).resolve()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_summary_to_notes(summary_text, config_dir, timestamp)

if __name__ == "__main__":
    main()
