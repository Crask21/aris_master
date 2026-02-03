#!/usr/bin/env python3
"""
Generate or update configuration summary in notes.md file.

This script scans a dataset directory and creates/updates a notes.md file with
a configuration summary including dataset statistics and training parameters.
It handles two directory structures:

1. Train/Val split structure:
   dataset/
     train/
       category1/
       category2/
     val/
       category1/
       category2/

2. Simple category structure:
   dataset/
     category1/
     category2/

Usage:
    # Generate summary for train/val structure
    python generate_config_summary.py --dataset /path/to/dataset --output /path/to/notes.md
    
    # Generate summary in current directory
    python generate_config_summary.py --dataset /path/to/dataset
    
    # Use custom image subfolder name
    python generate_config_summary.py --dataset /path/to/dataset --images-subdir images

Examples:
    python generate_config_summary.py --dataset data/waste_preview
    python generate_config_summary.py --dataset /tmp/master2025 --output training/session/notes.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


def _find_dataset_path(config: Dict) -> Optional[Path]:
    """Find dataset path from various possible config keys."""
    possible_names = [
        'dataset_path', 'dataset', 'data_path', 'data_dir', 'data_source',
        'train_data_dir', 'input', 'input_dir', 'input_path', 'source'
    ]
    for name in possible_names:
        if name in config and config[name] is not None:
            return Path(config[name])
    return None


def get_allowed_extensions() -> Set[str]:
    """Return set of allowed image file extensions."""
    return {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def list_images_in_dir(directory: Path, allowed_exts: Set[str], recursive: bool = False) -> List[Path]:
    """List all image files in a directory."""
    if not directory.exists() or not directory.is_dir():
        return []
    
    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.iterdir()
    
    return [p for p in iterator if p.is_file() and p.suffix.lower() in allowed_exts]


def find_images_in_category(category_dir: Path, images_subdir: str, allowed_exts: Set[str]) -> List[Path]:
    """
    Find images in a category directory, handling multiple layouts:
    - category/image1.png
    - category/images/image1.png
    - Both mixed (some images at root, some in subdirs)
    
    Always scans recursively to find all images within the category.
    """
    # Scan category directory recursively to find all images
    # This handles all cases: images directly in category, in subfolders, or mixed
    images = list_images_in_dir(category_dir, allowed_exts, recursive=True)
    return images


def analyze_dataset_structure(dataset_path: Path, images_subdir: str = "images") -> Tuple[str, Dict]:
    """
    Analyze dataset structure and return type and statistics.
    
    Handles multiple structures:
    1. Train/Val/Test splits with categories
    2. Simple category folders
    3. Direct images folder (single category)
    
    Returns:
        Tuple of (structure_type, data_dict)
        structure_type: "train_val" or "categories"
        data_dict: Contains image counts organized by structure
    """
    allowed_exts = get_allowed_extensions()
    
    # Check for train/val structure
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    test_dir = dataset_path / "test"
    
    has_train = train_dir.exists() and train_dir.is_dir()
    has_val = val_dir.exists() and val_dir.is_dir()
    has_test = test_dir.exists() and test_dir.is_dir()
    
    if has_train or has_val or has_test:
        # Train/Val/Test structure
        data = {}
        
        for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            if not split_dir.exists():
                continue
            
            split_data = {}
            
            # Scan for category subdirectories
            for category_dir in sorted(split_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                
                category_name = category_dir.name
                images = find_images_in_category(category_dir, images_subdir, allowed_exts)
                
                if images:
                    split_data[category_name] = len(images)
            
            if split_data:
                data[split_name] = split_data
        
        return "train_val", data
    
    else:
        # Simple category structure or single category
        data = {}
        
        # Common names for image-only subdirectories (not category names)
        image_folder_names = {"images", "imgs", "img", "image", "data", "files"}
        
        # First, check if there are subdirectories that look like categories
        # (not just image storage folders)
        category_subdirs = []
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name.lower() not in image_folder_names:
                category_subdirs.append(item)
        
        if category_subdirs:
            # Scan subdirectories as categories
            for category_dir in sorted(category_subdirs):
                category_name = category_dir.name
                images = find_images_in_category(category_dir, images_subdir, allowed_exts)
                
                if images:
                    data[category_name] = len(images)
        
        # If no categories found with subdirs, check if dataset_path itself contains images
        if not data:
            direct_images = list_images_in_dir(dataset_path, allowed_exts, recursive=True)
            if direct_images:
                # Dataset path itself is the category
                category_name = dataset_path.parent.name if dataset_path.name == images_subdir else dataset_path.name
                data[category_name] = len(direct_images)
        
        return "categories", data


def load_config(config_path: Path) -> Optional[Dict]:
    """Load config.json if it exists."""
    if not config_path.exists():
        return None
    
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"WARNING: Failed to load config.json: {e}", file=sys.stderr)
        return None


def format_config_section(config: Dict) -> str:
    """Format config settings into a markdown table."""
    lines = []
    lines.append("### Script Config")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    
    # Flatten config for display
    def add_config_items(data: Dict, prefix: str = ""):
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict) and key != "categories":
                # Recursively flatten nested dicts
                add_config_items(value, f"{full_key}.")
            elif isinstance(value, list):
                # Format lists nicely
                if len(value) > 3:
                    value_str = f"{len(value)} items"
                else:
                    value_str = ", ".join(str(v) for v in value)
                lines.append(f"| {full_key} | {value_str} |")
            elif isinstance(value, dict) and key == "categories":
                # Special handling for categories
                total_categories = sum(len(v) if isinstance(v, list) else 0 for v in value.values())
                lines.append(f"| {full_key} | {len(value)} groups, {total_categories} total |")
            else:
                # Format value
                if value is None:
                    value_str = "_not set_"
                elif isinstance(value, bool):
                    value_str = "Yes" if value else "No"
                else:
                    value_str = str(value)
                lines.append(f"| {full_key} | {value_str} |")
    
    # Process config items
    add_config_items(config)
    
    lines.append("")
    return "\n".join(lines)


def format_data_summary(structure_type: str, data: Dict, dataset_path: Path, config: Optional[Dict] = None) -> str:
    """Format the data summary section for notes.md."""
    lines = []
    lines.append("## Configuration Summary")
    lines.append("")
    lines.append(f"**Dataset Path:** `{dataset_path}`")
    lines.append("")
    
    # Add config section if available
    if config:
        lines.append(format_config_section(config))
    
    if structure_type == "train_val":
        # Calculate totals
        split_totals = {}
        all_categories = set()
        
        for split_name, categories in data.items():
            split_totals[split_name] = sum(categories.values())
            all_categories.update(categories.keys())
        
        overall_total = sum(split_totals.values())
        
        # Split distribution
        lines.append("### Split Distribution")
        lines.append("")
        lines.append("| Split | Images | Percentage |")
        lines.append("|-------|--------|------------|")
        
        for split_name in ["train", "val", "test"]:
            if split_name in split_totals:
                count = split_totals[split_name]
                percentage = (count / overall_total * 100) if overall_total > 0 else 0
                lines.append(f"| {split_name.capitalize()} | {count:,} | {percentage:.1f}% |")
        
        lines.append(f"| **Total** | **{overall_total:,}** | **100.0%** |")
        lines.append("")
        
        # Category distribution per split
        lines.append("### Category Distribution")
        lines.append("")
        
        for split_name in ["train", "val", "test"]:
            if split_name not in data:
                continue
            
            categories = data[split_name]
            split_total = split_totals[split_name]
            
            lines.append(f"#### {split_name.capitalize()} Set")
            lines.append("")
            lines.append("| Category | Images | Percentage |")
            lines.append("|----------|--------|------------|")
            
            # Sort by count descending
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            
            for category, count in sorted_categories:
                percentage = (count / split_total * 100) if split_total > 0 else 0
                lines.append(f"| {category} | {count:,} | {percentage:.1f}% |")
            
            lines.append(f"| **Total** | **{split_total:,}** | **100.0%** |")
            lines.append("")
    
    else:  # categories structure
        # Calculate total
        total_images = sum(data.values())
        
        lines.append("### Category Distribution")
        lines.append("")
        lines.append("| Category | Images | Percentage |")
        lines.append("|----------|--------|------------|")
        
        # Sort by count descending
        sorted_categories = sorted(data.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            percentage = (count / total_images * 100) if total_images > 0 else 0
            lines.append(f"| {category} | {count:,} | {percentage:.1f}% |")
        
        lines.append(f"| **Total** | **{total_images:,}** | **100.0%** |")
        lines.append("")
    
    return "\n".join(lines)


def update_or_create_notes(output_path: Path, data_summary: str, dataset_path: Path, verbose: bool = True):
    """
    Update existing notes.md with new data summary or create a new file.
    Uses the template from training/template/notes.md when creating a new file.
    """
    # If output_path is a directory, look for notes.md inside it
    if output_path.exists() and output_path.is_dir():
        notes_path = output_path / "notes.md"
    else:
        notes_path = output_path
    
    if notes_path.exists():
        # Read existing content
        content = notes_path.read_text(encoding="utf-8")
        
        # Update the date with current timestamp
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        import re
        
        # Update Creation date only if it's empty (first time)
        if "**Creation:**" in content:
            # Check if Creation is empty
            creation_match = re.search(r'\*\*Creation:\*\*\s*\n', content)
            if creation_match:
                # Creation is empty, fill it with current date
                content = re.sub(
                    r'\*\*Creation:\*\*\s*\n',
                    f'**Creation:** {current_date}\n',
                    content,
                    count=1
                )
        
        # Always update Last Training session
        if "**Last Training session:**" in content or "**Last training session:**" in content:
            # Update last training session (case insensitive)
            content = re.sub(
                r'\*\*Last [Tt]raining session:\*\*.*?\n',
                f'**Last Training session:** {current_date}\n',
                content,
                count=1
            )
        
        # Legacy support: if old "Date:" field exists, migrate to new format
        elif "**Date:**" in content:
            content = re.sub(
                r'\*\*Date:\*\*.*?\n',
                f'**Creation:** {current_date}\n\n**Last Training session:** {current_date}\n',
                content,
                count=1
            )
        
        # Check if Configuration Summary section exists
        if "## Configuration Summary" in content:
            # Replace existing Configuration Summary section
            # Find the start of Configuration Summary
            start_marker = "## Configuration Summary"
            start_idx = content.find(start_marker)
            
            # Find the next section (starts with ## and is not on the same line)
            # We need to find "\n## " to ensure we're finding the next section header
            search_start = start_idx + len(start_marker)
            next_section_idx = content.find("\n## ", search_start)
            
            if next_section_idx == -1:
                # Data Summary is the last section
                new_content = content[:start_idx] + data_summary
            else:
                # Insert new summary and preserve everything after the next section
                new_content = content[:start_idx] + data_summary + "\n" + content[next_section_idx + 1:]
            
            notes_path.write_text(new_content, encoding="utf-8")
            if verbose:
                print(f"Updated existing Configuration Summary in: {notes_path}")
        else:
            # Append Configuration Summary section
            if not content.endswith("\n"):
                content += "\n"
            content += "\n" + data_summary
            
            notes_path.write_text(content, encoding="utf-8")
            if verbose:
                print(f"Added Configuration Summary to existing file: {notes_path}")
    else:
        # Create new notes.md using template
        # Find template directory
        script_dir = Path(__file__).resolve().parent
        training_root = script_dir.parent.parent / "training"
        template_path = training_root / "template" / "notes.md"
        
        if template_path.exists():
            # Read template and fill in Creation and Last Training session
            template_content = template_path.read_text(encoding="utf-8")
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fill in Creation if it's a placeholder
            if "**Creation:**" in template_content:
                template_content = template_content.replace("**Creation:**", f"**Creation:** {current_date}")
            
            # Fill in Last Training session if it's a placeholder
            if "**Last Training session:**" in template_content or "**Last training session:**" in template_content:
                import re
                template_content = re.sub(
                    r'\*\*Last [Tt]raining session:\*\*',
                    f'**Last Training session:** {current_date}',
                    template_content
                )
            
            # Legacy support for old Date field
            if "**Date:**" in template_content and "**Creation:**" not in template_content:
                template_content = template_content.replace("**Date:**", f"**Creation:** {current_date}\n\n**Last Training session:** {current_date}")
            
            # Check if template has Configuration Summary section
            if "## Configuration Summary" in template_content:
                # Replace/append to Configuration Summary section in template
                start_marker = "## Configuration Summary"
                start_idx = template_content.find(start_marker)
                search_start = start_idx + len(start_marker)
                next_section_idx = template_content.find("\n## ", search_start)
                
                if next_section_idx == -1:
                    # Data Summary is the last section
                    new_content = template_content[:start_idx] + data_summary
                else:
                    # Insert new summary and preserve everything after
                    new_content = template_content[:start_idx] + data_summary + "\n" + template_content[next_section_idx + 1:]
            else:
                # Append Data Summary to template
                if not template_content.endswith("\n"):
                    template_content += "\n"
                new_content = template_content + "\n" + data_summary
        else:
            # Fallback: create basic template if template file not found
            print(f"WARNING: Template not found at {template_path}, using fallback template", file=sys.stderr)
            exit 
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_content = f"""# Model Training Notes

**Creation:** {current_date}

**Last Training session:** {current_date}

## Why was this model trained?

Because...

- A
- B
- C

{data_summary}
"""
        
        notes_path.write_text(new_content, encoding="utf-8")
        if verbose:
            print(f"Created new notes.md with Configuration Summary: {notes_path}")


def generate_data_summary(
    dataset_path: Path,
    output_path: Path = None,
    images_subdir: str = "images",
    verbose: bool = True
) -> Path:
    """
    Generate or update configuration summary in notes.md file.
    
    This function can be imported and used in other scripts to automatically
    generate dataset summaries with configuration details.
    
    The function is agnostic to argument naming conventions - it will automatically
    detect dataset paths from various common argument names (train_data_dir, input,
    data_dir, etc.) in the config file.
    
    Args:
        dataset_path: Path to dataset directory
        output_path: Output path for notes.md file (default: ./notes.md)
        images_subdir: Name of images subdirectory within categories (default: "images")
        verbose: Print progress messages (default: True)
    
    Returns:
        Path to the generated/updated notes.md file
    
    Raises:
        FileNotFoundError: If dataset_path doesn't exist
        ValueError: If no images found in dataset
    
    Example:
        from scripts.training_session.generate_config_summary import generate_data_summary
        
        # Generate summary
        notes_path = generate_data_summary(
            dataset_path=Path("/path/to/dataset"),
            output_path=Path("training/session/notes.md")
        )
    """
    # Validate dataset path
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {dataset_path}")
    
    # Set default output path
    if output_path is None:
        output_path = Path("notes.md")
    output_path = Path(output_path).resolve()
    
    if verbose:
        print(f"Analyzing dataset: {dataset_path}")
    
    # Analyze dataset structure
    structure_type, data = analyze_dataset_structure(dataset_path, images_subdir)
    
    if not data:
        raise ValueError(f"No images found in dataset: {dataset_path}")
    
    if verbose:
        print(f"Detected structure: {structure_type}")
    
    # Try to load config from output directory
    config = None
    if output_path.is_dir():
        config_path = output_path / "config.json"
    else:
        config_path = output_path.parent / "config.json"
    
    # Load config if it exists
    if config_path.exists():
        config = load_config(config_path)
        if verbose and config:
            print(f"Loaded config from: {config_path}")
    
    # Format data summary
    data_summary = format_data_summary(structure_type, data, dataset_path, config)
    
    # Update or create notes.md
    update_or_create_notes(output_path, data_summary, dataset_path, verbose)
    
    if verbose:
        print(f"✓ Configuration summary generated successfully: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate or update configuration summary in notes.md file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary in current directory
  python generate_config_summary.py --dataset /path/to/dataset
  
  # Specify output location
  python generate_config_summary.py --dataset /path/to/dataset --output training/notes.md
  
  # Use custom images subfolder name
  python generate_config_summary.py --dataset /path/to/dataset --images-subdir imgs
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for notes.md file (default: ./notes.md)"
    )
    
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Name of images subdirectory within categories (default: images)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print progress messages (default: True)"
    )
    
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    try:
        verbose = args.verbose and not args.quiet
        
        generate_data_summary(
            dataset_path=args.dataset,
            output_path=args.output,
            images_subdir=args.images_subdir,
            verbose=verbose
        )
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
