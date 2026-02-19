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


def analyze_data_dict(data_dict: List[Dict]) -> Tuple[str, Dict]:
    """
    Analyze dataset structure from a data dictionary.
    
    Args:
        data_dict: List of dictionaries with format:
            [
                {'filepath': '/path/to/image.png', 'class': 'class_name', 'split': 'train'},
                ...
            ]
    
    Returns:
        Tuple of (structure_type, data_dict)
        structure_type: "train_val" or "categories"
        data_dict: Contains image counts organized by structure
    """
    # Group by split and class
    split_data = {}
    
    for item in data_dict:
        split = item['split']
        class_name = item['class']
        
        if split not in split_data:
            split_data[split] = {}
        
        if class_name not in split_data[split]:
            split_data[split][class_name] = 0
        
        split_data[split][class_name] += 1
    
    # Check if we have splits
    has_splits = len(split_data) > 1 or any(s in split_data for s in ['train', 'val', 'test', 'synth'])
    
    if has_splits:
        return "train_val", split_data
    else:
        # Single category structure (only one split, no train/val/test)
        # Flatten to just category counts
        category_data = {}
        for split_name, categories in split_data.items():
            for class_name, count in categories.items():
                if class_name not in category_data:
                    category_data[class_name] = 0
                category_data[class_name] += count
        return "categories", category_data


def analyze_dataset_from_config(class_dict: Dict, images_subdir: str = "images") -> Tuple[str, Dict]:
    """
    Analyze dataset structure from a config class dictionary.
    
    Args:
        class_dict: Dictionary with format:
            {
                'parent_category': {
                    'data_dir': '/path/to/data',
                    'sub_categories': ['subcat1', 'subcat2'],
                    'train_img_count': 1000  # optional
                },
                ...
            }
        images_subdir: Name of images subdirectory within categories (default: "images")
    
    Returns:
        Tuple of (structure_type, data_dict)
        structure_type: "train_val" or "categories"
        data_dict: Contains image counts organized by structure
    """
    allowed_exts = get_allowed_extensions()
    
    # First, check if any data_dir has train/val/test splits at root level
    # We need to check the first data_dir to determine the structure
    sample_data_dir = Path(list(class_dict.values())[0]['data_dir'])
    train_dir = sample_data_dir / "train"
    val_dir = sample_data_dir / "val"
    test_dir = sample_data_dir / "test"
    
    has_splits = (train_dir.exists() and train_dir.is_dir()) or \
                 (val_dir.exists() and val_dir.is_dir()) or \
                 (test_dir.exists() and test_dir.is_dir())
    
    if has_splits:
        # Structure: data_dir/train/sub_category/images
        split_data = {"train": {}, "val": {}, "test": {}}
        
        for parent_category, config in class_dict.items():
            data_dir = Path(config['data_dir'])
            sub_categories = config.get('sub_categories', [parent_category])
            
            parent_split_counts = {"train": 0, "val": 0, "test": 0}
            
            # For each split, check if subcategories exist
            for split_name in ["train", "val", "test"]:
                split_path = data_dir / split_name
                if not split_path.exists():
                    continue
                
                # Count images in all subcategories for this parent category
                for sub_cat in sub_categories:
                    sub_cat_path = split_path / sub_cat
                    if not sub_cat_path.exists():
                        continue
                    
                    images = find_images_in_category(sub_cat_path, images_subdir, allowed_exts)
                    parent_split_counts[split_name] += len(images)
            
            # Add counts to split_data
            for split_name, count in parent_split_counts.items():
                if count > 0:
                    split_data[split_name][parent_category] = count
        
        # Remove empty splits
        result = {k: v for k, v in split_data.items() if v}
        return "train_val", result
    
    else:
        # Structure: data_dir/sub_category/images (no splits)
        category_data = {}
        
        for parent_category, config in class_dict.items():
            data_dir = Path(config['data_dir'])
            sub_categories = config.get('sub_categories', [parent_category])
            
            parent_total = 0
            
            for sub_cat in sub_categories:
                sub_cat_path = data_dir / sub_cat
                if not sub_cat_path.exists():
                    continue
                
                images = find_images_in_category(sub_cat_path, images_subdir, allowed_exts)
                parent_total += len(images)
            
            if parent_total > 0:
                category_data[parent_category] = parent_total
        
        return "categories", category_data


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


def format_data_summary(structure_type: str, data: Dict, dataset_path=None, config: Optional[Dict] = None, class_dict: Optional[Dict] = None) -> str:
    """Format the data summary section for notes.md."""
    lines = []
    lines.append("## Configuration Summary")
    lines.append("")
    
    # Handle dataset paths
    if class_dict:
        # Extract unique data directories from class_dict
        data_dirs = set()
        for class_config in class_dict.values():
            data_dirs.add(str(Path(class_config['data_dir']).resolve()))
        
        if len(data_dirs) == 1:
            lines.append(f"**Dataset Path:** `{list(data_dirs)[0]}`")
        else:
            lines.append("**Dataset Paths:**")
            for data_dir in sorted(data_dirs):
                lines.append(f"- `{data_dir}`")
    elif dataset_path:
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
        
        # Sort splits to show in a consistent order: train, val, test, synth, then others
        split_order = ["train", "val", "test", "synth"]
        sorted_splits = [s for s in split_order if s in split_totals]
        sorted_splits.extend([s for s in sorted(split_totals.keys()) if s not in split_order])
        
        for split_name in sorted_splits:
            count = split_totals[split_name]
            percentage = (count / overall_total * 100) if overall_total > 0 else 0
            lines.append(f"| {split_name.capitalize()} | {count:,} | {percentage:.1f}% |")
        
        lines.append(f"| **Total** | **{overall_total:,}** | **100.0%** |")
        lines.append("")
        
        # Category distribution per split
        lines.append("### Category Distribution")
        lines.append("")
        
        for split_name in sorted_splits:
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


def update_or_create_notes(output_path: Path, data_summary: str, dataset_path: Path, comment: str = "", verbose: bool = True):
    """
    Update existing notes.md with new data summary or create a new file.
    Uses the template from training/template/notes.md when creating a new file.
    """
    # If output_path is a directory, look for notes.md inside it
    if output_path.exists() and output_path.is_dir():
        notes_path = output_path / "notes.md"
    else:
        notes_path = output_path
    
    # Ensure parent directory exists before writing
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        # ----------------------- Configuration Summary section ---------------------- #
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
                
                
        # -------------------- "Why was this model trained?" section ------------------- #
                
        if "## Why was this model trained?" in content:
            # Replace existing Configuration Summary section
            # Find the start of Configuration Summary
            start_marker = "## Why was this model trained?"
            start_idx = content.find(start_marker)
            
            # Find the next section (starts with ## and is not on the same line)
            # We need to find "\n## " to ensure we're finding the next section header
            search_start = start_idx + len(start_marker)
            next_section_idx = content.find("\n## ", search_start)
            
            if next_section_idx == -1:
                # Data Summary is the last section
                new_content = content[:start_idx]  +start_marker+"\n\n" + comment
            else:
                # Insert new summary and preserve everything after the next section
                
                new_content = content[:start_idx] +start_marker+"\n\n" + comment + "\n" + content[next_section_idx + 1:]
            
            notes_path.write_text(new_content, encoding="utf-8")
            if verbose:
                print(f"Updated existing Why was this model trained? section in: {notes_path}")
        else:
            # Append Why was this model trained? section
            if not content.endswith("\n"):
                content += "\n"
            content += "\n" + comment
            print("Appending Why was this model trained? section")
            
            notes_path.write_text(content, encoding="utf-8")
            if verbose:
                print(f"Added Why was this model trained? section to existing file: {notes_path}")
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
            
            if "## Why was this model trained?" in template_content:
                # Fill in comment if provided
                if comment:
                    template_content = template_content.replace("## Why was this model trained?\n\n", f"## Why was this model trained?\n\n{comment}\n\n")
                else:
                    template_content = template_content.replace("## Why was this model trained?\n\n", "## Why was this model trained?\n\n_Because...\n\n")
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
            # Create the notes.md content with the data summary and comment
            new_content = f"""# Model Training Notes

**Creation:** {current_date}

**Last Training session:** {current_date}

## Why was this model trained?

{comment}

{data_summary}
"""
        
        notes_path.write_text(new_content, encoding="utf-8")
        if verbose:
            print(f"Created new notes.md with Configuration Summary: {notes_path}")



def generate_data_summary_from_config(
    config_path: Path,
    data_dict: List[Dict] = None,
    images_subdir: str = "images",
    verbose: bool = True
) -> Path:
    """
    Generate or update configuration summary in notes.md file from a config file.
    
    This function reads a config file with class definitions and analyzes datasets
    based on the data_dir and sub_categories for each class, OR uses a provided
    data_dict to generate the summary.
    
    Args:
        config_path: Path to the config.json file
        data_dict: Optional list of dicts with 'filepath', 'class', 'split' keys (default: None)
                   If provided, will use this instead of scanning filesystem
        images_subdir: Name of images subdirectory within categories (default: "images")
        verbose: Print progress messages (default: True)
    
    Returns:
        Path to the generated/updated notes.md file
    
    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If no images found in dataset or invalid config format
    
    Example:
        from src.summarize_training.generate_config_summary import generate_data_summary_from_config
        
        # Generate summary from config
        notes_path = generate_data_summary_from_config(
            config_path=Path("/path/to/config.json")
        )
        
        # Or from data_dict
        notes_path = generate_data_summary_from_config(
            config_path=Path("/path/to/config.json"),
            data_dict=[{'filepath': '/path/img.png', 'class': 'wood', 'split': 'train'}, ...]
        )
    """
    # Load config
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    
    cfg = load_config(config_path)
    if not cfg:
        raise ValueError(f"Failed to load config from: {config_path}")
    
    comment = cfg["logging"]["comment"] if "logging" in cfg and "comment" in cfg["logging"] else ""
    
    output_path = str(Path(cfg["logging"]["output_dir"]) / "notes.md") if "logging" in cfg and "output_dir" in cfg["logging"] else None
    
    # Extract class_dict from config
    if "data" not in cfg or "classes" not in cfg["data"]:
        raise ValueError(f"Config file must contain 'data.classes' section: {config_path}")
    
    class_dict = cfg["data"]["classes"]
    
    if not class_dict:
        raise ValueError(f"No classes defined in config: {config_path}")
    
    # Set default output path (same directory as config)
    if output_path is None:
        output_path = config_path.parent / "notes.md"
    output_path = Path(output_path).resolve()
    
    # Analyze dataset structure
    if data_dict is not None:
        # Use provided data_dict
        if verbose:
            print(f"Analyzing dataset from provided data_dict")
            print(f"Found {len(data_dict)} total samples")
        
        structure_type, data = analyze_data_dict(data_dict)
        
        # Extract unique data directories from data_dict filepaths for display
        unique_dirs = set()
        for item in data_dict:
            filepath = Path(item['filepath'])
            # Try to find the data directory (usually 2-3 levels up from image)
            parts = filepath.parts
            for i, part in enumerate(parts):
                if part in ['train', 'val', 'test', 'synth']:
                    if i > 0:
                        unique_dirs.add(str(Path(*parts[:i])))
                    break
        
        # If we couldn't extract dirs from paths, fall back to class_dict
        if not unique_dirs:
            unique_dirs = {class_config['data_dir'] for class_config in class_dict.values() if 'data_dir' in class_config}
    else:
        # Scan filesystem using class_dict
        if verbose:
            print(f"Analyzing datasets from config: {config_path}")
            print(f"Found {len(class_dict)} classes: {', '.join(class_dict.keys())}")
        
        # Validate that all data_dirs exist
        for parent_category, class_config in class_dict.items():
            if "data_dir" not in class_config:
                raise ValueError(f"Missing 'data_dir' for class '{parent_category}'")
            
            data_dir = Path(class_config["data_dir"])
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory does not exist for class '{parent_category}': {data_dir}")
        
        structure_type, data = analyze_dataset_from_config(class_dict, images_subdir)
    
    if not data:
        raise ValueError(f"No images found in any dataset")
    
    if verbose:
        print(f"Detected structure: {structure_type}")
        total_images = 0
        if structure_type == "train_val":
            for split_name, categories in data.items():
                split_total = sum(categories.values())
                total_images += split_total
                print(f"  {split_name}: {split_total:,} images across {len(categories)} categories")
        else:
            total_images = sum(data.values())
            print(f"  Total: {total_images:,} images across {len(data)} categories")
    
    # Format data summary with class_dict for multiple dataset paths
    data_summary = format_data_summary(structure_type, data, dataset_path=None, config=cfg, class_dict=class_dict)
    
    # Update or create notes.md (use first data_dir for legacy dataset_path parameter)
    first_data_dir = Path(list(class_dict.values())[0]['data_dir']) if 'data_dir' in list(class_dict.values())[0] else Path(output_path).parent
    update_or_create_notes(output_path, data_summary, first_data_dir, comment, verbose)
    
    if verbose:
        print(f"✓ Configuration summary generated successfully: {output_path}")
    
    return output_path


def generate_data_summary(
    dataset_path: Path,
    output_path: Path = None,
    images_subdir: str = "images",
    comment: str = "",
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
    update_or_create_notes(output_path, data_summary, dataset_path, comment, verbose)
    
    if verbose:
        print(f"✓ Configuration summary generated successfully: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate or update configuration summary in notes.md file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary from config file
  python generate_config_summary.py --config /path/to/config.json
  
  # Generate summary from dataset directory (legacy)
  python generate_config_summary.py --dataset /path/to/dataset
  
  # Specify output location
  python generate_config_summary.py --config /path/to/config.json --output training/notes.md
  
  # Use custom images subfolder name
  python generate_config_summary.py --config /path/to/config.json --images-subdir imgs
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json file (preferred method)"
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset directory (legacy method, use --config instead)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for notes.md file (default: ./notes.md or config directory)"
    )
    
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Name of images subdirectory within categories (default: images)"
    )
    
    parser.add_argument(
        "--comment",
        "-c",
        type=str,
        default="",
        help="Comment to add to 'Why was this model trained?' section"
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
        
        # Validate arguments
        if not args.config and not args.dataset:
            parser.error("Either --config or --dataset must be provided")
        
        if args.config and args.dataset:
            parser.error("Cannot use both --config and --dataset, use --config for the new workflow")
        
        # Use config-based analysis if config is provided
        if args.config:
            generate_data_summary_from_config(
                config_path=args.config
            )
        else:
            # Legacy dataset-based analysis
            generate_data_summary(
                dataset_path=args.dataset,
                output_path=args.output,
                images_subdir=args.images_subdir,
                comment=args.comment,
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

