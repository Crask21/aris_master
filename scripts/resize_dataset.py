#!/usr/bin/env python3
"""
Script to resize all images in a dataset directory while preserving folder structure.
Shows a preview comparison between original and resized images.
"""

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_image(image_path: Path, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize an image to the target size using smart cropping to preserve aspect ratio.
    
    The image is resized to match the target height, then center-cropped to match
    the target width. This prevents squishing when aspect ratios differ.
    
    Args:
        image_path: Path to the input image
        target_size: Tuple of (width, height) for the output image
        
    Returns:
        Resized and cropped PIL Image
    """
    img = Image.open(image_path)
    # Convert to RGB if image has alpha channel or is in a different mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    target_width, target_height = target_size
    original_width, original_height = img.size
    
    # Calculate the scaling factor to match target height
    scale = target_height / original_height
    new_width = int(original_width * scale)
    
    # Resize to match target height
    img = img.resize((new_width, target_height), Image.LANCZOS)
    
    # Center crop to match target width
    if new_width > target_width:
        # Image is wider than target, crop the sides
        left = (new_width - target_width) // 2
        right = left + target_width
        img = img.crop((left, 0, right, target_height))
    elif new_width < target_width:
        # Image is narrower than target, pad with black or resize width
        # For now, we'll resize to fill (slight stretch on width only)
        img = img.resize((target_width, target_height), Image.LANCZOS)
    
    return img


def show_preview(original_paths: list, resized_images: list, num_samples: int = 4):
    """
    Show a comparison between original and resized images.
    
    Args:
        original_paths: List of paths to original images
        resized_images: List of resized PIL Images
        num_samples: Number of samples to show in preview
    """
    num_samples = min(num_samples, len(original_paths))
    
    if num_samples == 0:
        print("No images to preview.")
        return
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    # Handle case when there's only one sample
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        original_img = Image.open(original_paths[idx])
        
        # Show original
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original: {original_img.size[0]}x{original_img.size[1]}')
        axes[idx, 0].axis('off')
        
        # Show resized
        axes[idx, 1].imshow(resized_images[idx])
        axes[idx, 1].set_title(f'Resized: {resized_images[idx].size[0]}x{resized_images[idx].size[1]}')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_dataset(input_dir: Path, output_dir: Path, width: int, height: int, preview: bool = True):
    """
    Process all images in the dataset directory and resize them.
    
    Args:
        input_dir: Path to input dataset directory
        output_dir: Path to output dataset directory
        width: Target width for resized images
        height: Target height for resized images
        preview: Whether to show preview comparison
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all image files (common extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(input_dir.rglob(f'*{ext}'))
        image_paths.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    print(f"Target size: {width}x{height}")
    
    target_size = (width, height)
    preview_originals = []
    preview_resized = []
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Resizing images", unit="image"):
        # Calculate relative path from input directory
        rel_path = img_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resize and save image
        try:
            resized_img = resize_image(img_path, target_size)
            resized_img.save(output_path, quality=95)
            
            # Collect samples for preview
            if preview and len(preview_originals) < 4:
                preview_originals.append(img_path)
                preview_resized.append(resized_img)
        except Exception as e:
            tqdm.write(f"Error processing {rel_path}: {e}")
    
    # Copy non-image files (like annotations) to preserve folder structure
    non_image_files = [f for f in input_dir.rglob('*') if f.is_file() and f.suffix.lower() not in image_extensions]
    
    if non_image_files:
        for file_path in tqdm(non_image_files, desc="Copying non-image files", unit="file"):
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, output_path)
    
    print(f"\nDataset processing complete!")
    print(f"Output saved to: {output_dir}")
    
    # Show preview
    if preview and preview_originals:
        print("\nShowing preview comparison...")
        show_preview(preview_originals, preview_resized)


def main():
    parser = argparse.ArgumentParser(
        description='Resize all images in a dataset while preserving folder structure'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset directory path'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output dataset directory path'
    )
    parser.add_argument(
        '--width',
        type=int,
        required=True,
        help='Target width for resized images'
    )
    parser.add_argument(
        '--height',
        type=int,
        required=True,
        help='Target height for resized images'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        default=True,
        help='Show preview comparison (enabled by default)'
    )
    parser.add_argument(
        '--no-preview',
        action='store_false',
        dest='preview',
        help='Disable preview comparison'
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input,
        output_dir=args.out,
        width=args.width,
        height=args.height,
        preview=args.preview
    )


if __name__ == '__main__':
    main()
