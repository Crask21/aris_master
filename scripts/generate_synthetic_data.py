#!/usr/bin/env python3
"""
Generate synthetic data using trained GAN model.
"""

import os
import torch
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.generators import Generator
from utils.config import load_config


def generate_synthetic_data(config_path: str, model_path: str, num_samples: int = None):
    """
    Generate synthetic waste images using trained GAN.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained generator model
        num_samples: Number of samples to generate (if None, uses config value)
    """
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    generator = Generator(
        config.model.latent_dim,
        config.model.img_channels,
        config.model.feature_map_size
    ).to(device)
    
    # Load trained weights
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded generator from: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    generator.eval()
    
    # Set number of samples
    if num_samples is None:
        num_samples = config.synthetic.num_samples
    
    # Create output directory
    output_dir = os.path.join(config.data.synthetic_data_dir, 'generated_images')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic images...")
    
    # Generate samples in batches
    batch_size = config.synthetic.generation_batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            # Calculate batch size for this iteration
            current_batch_size = min(batch_size, num_samples - sample_count)
            
            # Generate noise
            noise = torch.randn(current_batch_size, config.model.latent_dim, device=device)
            
            # Generate images
            fake_images = generator(noise)
            
            # Convert to PIL images and save
            for i in range(current_batch_size):
                # Convert tensor to numpy
                img_tensor = fake_images[i].cpu()
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                # Denormalize from [-1, 1] to [0, 1]
                img_np = (img_np + 1) / 2
                img_np = np.clip(img_np, 0, 1)
                
                # Convert to [0, 255] and uint8
                img_np = (img_np * 255).astype(np.uint8)
                
                # Save image
                img_pil = Image.fromarray(img_np)
                filename = f"synthetic_{sample_count:06d}.png"
                img_pil.save(os.path.join(output_dir, filename))
                
                sample_count += 1
    
    print(f"Generated {sample_count} synthetic images")
    print(f"Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained generator model')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    generate_synthetic_data(args.config, args.model, args.num_samples)


if __name__ == '__main__':
    main()