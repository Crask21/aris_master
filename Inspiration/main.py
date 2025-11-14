"""
Main entry point for the Synthetic Data Generation for Waste Segmentation project.

This script provides a command-line interface for various tasks including:
- Data preparation
- Model training (segmentation and GAN)
- Synthetic data generation
- Model evaluation
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.config import load_config, create_directories


def setup_project(config_path: str):
    """
    Set up project directories and configuration.
    
    Args:
        config_path: Path to configuration file
    """
    print("Setting up project directories...")
    config = load_config(config_path)
    create_directories(config)
    print("Project setup completed!")


def train_segmentation(config_path: str):
    """Train segmentation model."""
    from training.train_segmentation import SegmentationTrainer
    
    trainer = SegmentationTrainer(config_path)
    trainer.train()


def train_gan(config_path: str):
    """Train GAN model."""
    from training.train_gan import GANTrainer
    
    trainer = GANTrainer(config_path)
    trainer.train()


def generate_data(config_path: str, model_path: str, num_samples: int = None):
    """Generate synthetic data."""
    from scripts.generate_synthetic_data import generate_synthetic_data
    
    generate_synthetic_data(config_path, model_path, num_samples)


def evaluate_model(config_path: str, model_path: str, split: str = 'test'):
    """Evaluate trained model."""
    from scripts.evaluate_model import evaluate_model as eval_model
    
    eval_model(config_path, model_path, split)


def prepare_data(config_path: str, source_dir: str):
    """Prepare data for training."""
    from scripts.prepare_data import prepare_data as prep_data
    
    prep_data(config_path, source_dir)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Synthetic Data Generation for Waste Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup --config config/config.yaml
  python main.py prepare-data --config config/config.yaml --source data/raw
  python main.py train-segmentation --config config/config.yaml
  python main.py train-gan --config config/config.yaml
  python main.py generate --config config/config.yaml --model outputs/models/final_generator.pth
  python main.py evaluate --config config/config.yaml --model outputs/models/best_model.pth
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up project directories')
    setup_parser.add_argument('--config', type=str, required=True,
                             help='Path to configuration file')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare-data', help='Prepare data for training')
    prepare_parser.add_argument('--config', type=str, required=True,
                               help='Path to configuration file')
    prepare_parser.add_argument('--source', type=str, required=True,
                               help='Path to source data directory')
    
    # Train segmentation command
    train_seg_parser = subparsers.add_parser('train-segmentation', help='Train segmentation model')
    train_seg_parser.add_argument('--config', type=str, required=True,
                                 help='Path to configuration file')
    
    # Train GAN command
    train_gan_parser = subparsers.add_parser('train-gan', help='Train GAN model')
    train_gan_parser.add_argument('--config', type=str, required=True,
                                 help='Path to configuration file')
    
    # Generate data command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    generate_parser.add_argument('--config', type=str, required=True,
                                help='Path to configuration file')
    generate_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained generator model')
    generate_parser.add_argument('--num-samples', type=int, default=None,
                                help='Number of samples to generate')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument('--config', type=str, required=True,
                                help='Path to configuration file')
    evaluate_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model')
    evaluate_parser.add_argument('--split', type=str, default='test',
                                choices=['train', 'val', 'test'],
                                help='Dataset split to evaluate on')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        # Execute command
        if args.command == 'setup':
            setup_project(args.config)
        elif args.command == 'prepare-data':
            prepare_data(args.config, args.source)
        elif args.command == 'train-segmentation':
            train_segmentation(args.config)
        elif args.command == 'train-gan':
            train_gan(args.config)
        elif args.command == 'generate':
            generate_data(args.config, args.model, args.num_samples)
        elif args.command == 'evaluate':
            evaluate_model(args.config, args.model, args.split)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
