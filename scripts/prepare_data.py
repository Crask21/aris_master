#!/usr/bin/env python3
"""
Data preparation script for waste segmentation dataset.
"""

import os
import argparse
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocessing import create_data_splits, preprocess_image, preprocess_mask
from utils.config import load_config


def prepare_data(config_path: str, source_dir: str):
    """
    Prepare data for training.
    
    Args:
        config_path: Path to configuration file
        source_dir: Directory containing raw images and masks
    """
    config = load_config(config_path)
    
    print("Preparing data for waste segmentation...")
    
    # Create necessary directories
    os.makedirs(config.data.raw_data_dir, exist_ok=True)
    os.makedirs(config.data.processed_data_dir, exist_ok=True)
    
    # Copy data to raw directory if needed
    if source_dir != config.data.raw_data_dir:
        print(f"Copying data from {source_dir} to {config.data.raw_data_dir}")
        if os.path.exists(os.path.join(source_dir, 'images')):
            shutil.copytree(
                os.path.join(source_dir, 'images'),
                os.path.join(config.data.raw_data_dir, 'images'),
                dirs_exist_ok=True
            )
        if os.path.exists(os.path.join(source_dir, 'masks')):
            shutil.copytree(
                os.path.join(source_dir, 'masks'),
                os.path.join(config.data.raw_data_dir, 'masks'),
                dirs_exist_ok=True
            )
    
    # Create data splits
    print("Creating train/validation/test splits...")
    create_data_splits(
        config.data.raw_data_dir,
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio
    )
    
    # Move split data to processed directory
    for split in ['train', 'val', 'test']:
        src_split_dir = os.path.join(config.data.raw_data_dir, split)
        dst_split_dir = os.path.join(config.data.processed_data_dir, split)
        
        if os.path.exists(src_split_dir):
            shutil.move(src_split_dir, dst_split_dir)
    
    print("Data preparation completed!")
    print(f"Processed data saved to: {config.data.processed_data_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source data directory')
    
    args = parser.parse_args()
    
    prepare_data(args.config, args.source)


if __name__ == '__main__':
    main()