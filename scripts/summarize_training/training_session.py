#!/usr/bin/env python3
"""
Manage training sessions: create new sessions or load existing ones.

This script can either create a new training session or load an existing one:

CREATE MODE (--create):
- Generates a new training session folder under training/ directory
- Creates checkpoints/ and results/ subdirectories
- Copies and auto-fills config.json from template
- Copies and timestamps notes.md
- Optionally auto-fills data fields if --data-source provided

LOAD MODE (--load):
- Loads an existing training session by folder path or config.json path
- Displays current configuration
- Can update data source with --data-source (prompts before replacing existing)

Usage:
    # Create new session (default name: unet-wood)
    python training_session.py --create
    
    # Create with custom name
    python training_session.py --create my-experiment --data-source /path/to/dataset
    
    # Load existing session
    python training_session.py --load training/unet-wood_01-27T13-24
    
    # Load and update data source
    python training_session.py --load training/unet-wood_01-27T13-24 --data-source /new/path
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_image_dimensions(data_source: Path) -> Optional[Tuple[int, int]]:
    """
    Recursively search for an image in data_source and return its dimensions.
    
    Args:
        data_source: Path to the dataset directory
        
    Returns:
        Tuple of (width, height) or None if no image found
    """
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    
    # Search for first image file
    for img_path in data_source.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in allowed_exts:
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    return img.size  # Returns (width, height)
            except Exception as e:
                print(f"Warning: Failed to read image {img_path}: {e}", file=sys.stderr)
                continue
    
    return None


def create_training_session(
    name: str,
    data_source: Optional[Path] = None,
    model_type: Optional[str] = None,
    training_root: Path = None
) -> Path:
    """
    Create a new training session directory with templates.
    
    Args:
        name: Base name for the training session
        data_source: Optional path to dataset (auto-fills config fields)
        model_type: Optional model type string
        training_root: Root directory for training sessions
        
    Returns:
        Path to the created training session directory
    """
    if training_root is None:
        # Default to training/ in script's parent directory
        script_dir = Path(__file__).resolve().parent.parent
        training_root = script_dir.parent / "training"
    
    template_dir = training_root / "template"
    
    # Validate template directory exists
    if not template_dir.exists():
        sys.exit(f"ERROR: Template directory not found: {template_dir}")
    
    template_config = template_dir / "config.json"
    template_notes = template_dir / "notes.md"
    
    if not template_config.exists():
        sys.exit(f"ERROR: Template config.json not found: {template_config}")
    if not template_notes.exists():
        sys.exit(f"ERROR: Template notes.md not found: {template_notes}")
    
    # Generate timestamp and session directory name
    timestamp = datetime.now().strftime("%m-%dT%H-%M")
    session_name = f"{name}_{timestamp}"
    session_dir = training_root / session_name
    
    # Check if directory already exists
    if session_dir.exists():
        sys.exit(f"ERROR: Training session directory already exists: {session_dir}")
    
    # Create session directory and subdirectories
    print(f"Creating training session: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "checkpoints").mkdir(exist_ok=True)
    (session_dir / "results").mkdir(exist_ok=True)
    
    # Load template config
    with template_config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Fill in config fields
    config["config"]["directory"] = str(session_dir.resolve())
    
    if model_type:
        config["config"]["model_type"] = model_type
    
    # Handle data source if provided
    if data_source:
        data_source = data_source.resolve()
        
        if not data_source.exists():
            print(f"WARNING: Data source does not exist: {data_source}", file=sys.stderr)
        else:
            config["data"]["source"] = str(data_source)
            
            # Extract image dimensions
            print(f"Scanning for images in: {data_source}")
            dimensions = get_image_dimensions(data_source)
            
            if dimensions:
                width, height = dimensions
                config["data"]["width"] = width
                config["data"]["height"] = height
                print(f"Detected image dimensions: {width}x{height}")
            else:
                print("WARNING: No images found in data source", file=sys.stderr)
    
    # Write config.json
    config_path = session_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Created: {config_path}")
    
    # Copy and update notes.md
    notes_content = template_notes.read_text(encoding="utf-8")
    
    # Replace **Date:** line with actual date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notes_content = notes_content.replace("**Date:**", f"**Date:** {current_date}")
    
    notes_path = session_dir / "notes.md"
    notes_path.write_text(notes_content, encoding="utf-8")
    print(f"Created: {notes_path}")
    
    print(f"\nTraining session created: {session_dir}")
    
    # Run generate_data_file.py if data source was provided
    if data_source and data_source.exists():
        print(f"\nRunning generate_data_file.py...")
        script_path = Path(__file__).resolve().parent / "generate_data_file.py"
        
        if not script_path.exists():
            print(f"WARNING: generate_data_file.py not found at {script_path}", file=sys.stderr)
            print("Skipping data file generation.", file=sys.stderr)
        else:
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), "-c", str(config_path), "--verbose"],
                    check=True,
                    capture_output=False,
                    text=True
                )
                print("Data file generation completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: generate_data_file.py failed with exit code {e.returncode}", file=sys.stderr)
                print("Training session created, but data file generation failed.", file=sys.stderr)
    
    return session_dir


def load_training_session(
    session_path: Path,
    data_source: Optional[Path] = None
) -> Path:
    """
    Load an existing training session and optionally update data source.
    
    Args:
        session_path: Path to training session directory or config.json
        data_source: Optional new data source (prompts before replacing)
        
    Returns:
        Path to the training session directory
    """
    # Resolve session directory
    session_path = session_path.resolve()
    
    if session_path.is_file() and session_path.name == "config.json":
        # User provided config.json path
        session_dir = session_path.parent
        config_path = session_path
    elif session_path.is_dir():
        # User provided directory
        session_dir = session_path
        config_path = session_dir / "config.json"
    else:
        sys.exit(f"ERROR: Invalid session path: {session_path}")
    
    # Validate config exists
    if not config_path.exists():
        sys.exit(f"ERROR: config.json not found in session: {config_path}")
    
    # Load existing config
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"Loaded training session: {session_dir}")
    print(f"\nCurrent configuration:")
    print(f"  Directory: {config.get('config', {}).get('directory', 'N/A')}")
    print(f"  Model type: {config.get('config', {}).get('model_type', 'N/A')}")
    print(f"  Data source: {config.get('data', {}).get('source', 'N/A')}")
    print(f"  Image dimensions: {config.get('data', {}).get('width', 'N/A')}x{config.get('data', {}).get('height', 'N/A')}")
    
    # Handle data source update
    if data_source:
        data_source = data_source.resolve()
        
        if not data_source.exists():
            sys.exit(f"ERROR: Data source does not exist: {data_source}")
        
        # Check if data source already exists
        existing_source = config.get('data', {}).get('source')
        
        if existing_source and existing_source != "null" and existing_source is not None:
            print(f"\n⚠️  WARNING: This training session already has a data source:")
            print(f"   Current: {existing_source}")
            print(f"   New: {data_source}")
            
            response = input("\nReplace existing data source? [y/N]: ").strip().lower()
            
            if response not in ['y', 'yes']:
                print("Cancelled. Data source not updated.")
                return session_dir
        
        # Update data source
        print(f"\nUpdating data source to: {data_source}")
        config["data"]["source"] = str(data_source)
        
        # Extract and update image dimensions
        print(f"Scanning for images in: {data_source}")
        dimensions = get_image_dimensions(data_source)
        
        if dimensions:
            width, height = dimensions
            config["data"]["width"] = width
            config["data"]["height"] = height
            print(f"Detected image dimensions: {width}x{height}")
        else:
            print("WARNING: No images found in data source", file=sys.stderr)
        
        # Write updated config
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Updated: {config_path}")
        
        # Run generate_data_file.py
        print(f"\nRunning generate_data_file.py...")
        script_path = Path(__file__).resolve().parent / "generate_data_file.py"
        
        if not script_path.exists():
            print(f"WARNING: generate_data_file.py not found at {script_path}", file=sys.stderr)
            print("Skipping data file generation.", file=sys.stderr)
        else:
            try:
                subprocess.run(
                    [sys.executable, str(script_path), "-c", str(config_path), "--verbose"],
                    check=True,
                    capture_output=False,
                    text=True
                )
                print("Data file generation completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: generate_data_file.py failed with exit code {e.returncode}", file=sys.stderr)
    
    return session_dir


def main():
    parser = argparse.ArgumentParser(
        description="Manage training sessions: create new or load existing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new session (default name: unet-wood)
  python training_session.py --create
  
  # Create with custom name and data source
  python training_session.py --create my-experiment --data-source /path/to/dataset
  
  # Load existing session
  python training_session.py --load training/unet-wood_01-27T13-24
  
  # Load and update data source
  python training_session.py --load training/unet-wood_01-27T13-24/config.json --data-source /new/path
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--create",
        nargs="?",
        const="unet-wood",
        default=None,
        type=str,
        help="Create new training session (optionally specify name, default: unet-wood)"
    )
    mode_group.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Load existing training session (provide directory or config.json path)"
    )
    
    # Common arguments
    parser.add_argument(
        "--data-source",
        type=Path,
        default=None,
        help="Path to dataset directory (auto-fills config fields)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type string (only for --create mode)"
    )
    
    parser.add_argument(
        "--training-root",
        type=Path,
        default=None,
        help="Root directory for training sessions (default: training/ in project root)"
    )
    
    args = parser.parse_args()
    
    try:
        # Check for PIL if data source is provided
        if args.data_source:
            try:
                from PIL import Image
            except ImportError:
                sys.exit("ERROR: PIL (Pillow) is required when using --data-source. Install with: pip install Pillow")
        
        if args.create is not None:
            # CREATE MODE
            session_dir = create_training_session(
                name=args.create,
                data_source=args.data_source,
                model_type=args.model_type,
                training_root=args.training_root
            )
            print(f"\n✓ Training session ready: {session_dir}")
            
        elif args.load is not None:
            # LOAD MODE
            session_dir = load_training_session(
                session_path=args.load,
                data_source=args.data_source
            )
            print(f"\n✓ Training session loaded: {session_dir}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
