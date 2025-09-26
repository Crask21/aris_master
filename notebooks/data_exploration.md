# Data Exploration and Visualization

This notebook provides tools for exploring and visualizing the waste segmentation dataset.

## Imports and Setup

```python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import seaborn as sns
from pathlib import Path

# Add src to path
project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data.dataset import WasteSegmentationDataset, get_transforms
from utils.config import load_config
from utils.visualization import visualize_segmentation, plot_training_history

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## Load Configuration

```python
# Load configuration
config_path = project_root / "config" / "config.yaml"
config = load_config(str(config_path))
print("Configuration loaded successfully!")
print(f"Image size: {config.data.image_size}")
print(f"Batch size: {config.data.batch_size}")
print(f"Number of classes: {config.model.n_classes}")
```

## Dataset Overview

```python
# Check data directories
data_dir = project_root / "data"
print("Data directory structure:")
for item in data_dir.rglob("*"):
    if item.is_dir():
        print(f"📁 {item.relative_to(project_root)}")
    elif item.suffix in ['.jpg', '.jpeg', '.png']:
        print(f"📄 {item.relative_to(project_root)}")
```

## Load and Visualize Sample Data

```python
# Create dataset
try:
    transform, target_transform = get_transforms(tuple(config.data.image_size), is_training=False)
    
    dataset = WasteSegmentationDataset(
        str(project_root / "data" / "processed"),
        split='train',
        transform=transform,
        target_transform=target_transform
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Visualize first few samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(min(4, len(dataset))):
        image, mask = dataset[i]
        
        # Convert to numpy for visualization
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.array(image) / 255.0
            
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().numpy()
        else:
            mask_np = np.array(mask)
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_np, cmap='jet')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure you have prepared your data using: python main.py prepare-data")
```

## Dataset Statistics

```python
# Analyze dataset statistics if data is available
try:
    if len(dataset) > 0:
        print("Dataset Statistics:")
        print(f"Total samples: {len(dataset)}")
        
        # Sample a few images to analyze
        sample_size = min(100, len(dataset))
        mask_pixels = []
        waste_ratios = []
        
        for i in range(sample_size):
            _, mask = dataset[i]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = np.array(mask)
            
            total_pixels = mask_np.size
            waste_pixels = np.sum(mask_np > 0)
            waste_ratio = waste_pixels / total_pixels
            
            mask_pixels.append(total_pixels)
            waste_ratios.append(waste_ratio)
        
        print(f"Average waste ratio: {np.mean(waste_ratios):.3f} ± {np.std(waste_ratios):.3f}")
        print(f"Min waste ratio: {np.min(waste_ratios):.3f}")
        print(f"Max waste ratio: {np.max(waste_ratios):.3f}")
        
        # Plot waste ratio distribution
        plt.figure(figsize=(10, 6))
        plt.hist(waste_ratios, bins=20, alpha=0.7)
        plt.xlabel('Waste Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Waste Ratios in Dataset')
        plt.grid(True, alpha=0.3)
        plt.show()
        
except Exception as e:
    print(f"Error analyzing dataset: {e}")
```

## Model Architecture Visualization

```python
# Visualize model architectures
from models.segmentation import UNet
from models.generators import Generator, Discriminator

# Create model instances for visualization
try:
    # Segmentation model
    seg_model = UNet(
        n_channels=config.model.n_channels,
        n_classes=config.model.n_classes
    )
    
    # Count parameters
    seg_params = sum(p.numel() for p in seg_model.parameters())
    print(f"Segmentation model parameters: {seg_params:,}")
    
    # GAN models
    generator = Generator(
        latent_dim=config.model.latent_dim,
        img_channels=config.model.img_channels,
        feature_map_size=config.model.feature_map_size
    )
    
    discriminator = Discriminator(
        img_channels=config.model.img_channels,
        feature_map_size=config.model.feature_map_size
    )
    
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Model summary visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Segmentation\n(U-Net)', 'Generator\n(GAN)', 'Discriminator\n(GAN)']
    params = [seg_params, gen_params, disc_params]
    
    bars = ax.bar(models, params, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Complexity Comparison')
    ax.set_yscale('log')  # Log scale for better visualization
    
    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error creating models: {e}")
```

## Training Progress Visualization

```python
# Placeholder for training progress visualization
# This section would be filled with actual training results

print("Training Progress Visualization")
print("This section will show training curves when models are trained.")

# Example of how training curves would look
epochs = np.arange(1, 51)
train_loss = 0.8 * np.exp(-epochs/20) + 0.1 + 0.05 * np.random.randn(50)
val_loss = 0.9 * np.exp(-epochs/18) + 0.15 + 0.05 * np.random.randn(50)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Example Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
train_acc = 1 - train_loss + 0.1
val_acc = 1 - val_loss + 0.05
plt.plot(epochs, train_acc, label='Training Accuracy', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Example Accuracy Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Next Steps

```python
print("Next Steps:")
print("1. Prepare your dataset using: python main.py prepare-data --config config/config.yaml --source your_data_path")
print("2. Train segmentation model: python main.py train-segmentation --config config/config.yaml")
print("3. Train GAN model: python main.py train-gan --config config/config.yaml")
print("4. Generate synthetic data: python main.py generate --config config/config.yaml --model outputs/models/final_generator.pth")
print("5. Evaluate results: python main.py evaluate --config config/config.yaml --model outputs/models/best_model.pth")
```