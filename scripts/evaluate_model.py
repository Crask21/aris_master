#!/usr/bin/env python3
"""
Evaluate trained segmentation model.
"""

import os
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import WasteSegmentationDataset, get_transforms
from models.segmentation import get_segmentation_model
from evaluation.metrics import calculate_metrics
from utils.config import load_config
from utils.visualization import save_sample_predictions


def evaluate_model(config_path: str, model_path: str, split: str = 'test'):
    """
    Evaluate trained segmentation model.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model
        split: Dataset split to evaluate on
    """
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = get_segmentation_model(
        config.model.segmentation_name,
        config.model.n_channels,
        config.model.n_classes
    ).to(device)
    
    # Load trained weights
    if os.path.exists(model_path):
        if model_path.endswith('.pth') and 'checkpoint' in model_path:
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint: {model_path}")
        else:
            # Load state dict directly
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    
    # Create data loader
    _, target_transform = get_transforms(
        tuple(config.data.image_size), is_training=False
    )
    
    dataset = WasteSegmentationDataset(
        config.data.processed_data_dir,
        split=split,
        transform=None,  # We'll handle transforms manually for visualization
        target_transform=target_transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    print(f"Evaluating on {len(dataset)} {split} samples...")
    
    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_gts = []
    sample_preds = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(data_loader, desc=f"Evaluating {split}")):
            images = images.to(device)
            masks = masks.to(device).squeeze(1).long()
            
            # Forward pass
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            # Collect for metrics
            all_predictions.append(outputs.cpu())
            all_targets.append(masks.cpu())
            
            # Collect samples for visualization (first batch only)
            if batch_idx == 0 and len(sample_images) < 8:
                for i in range(min(8, images.size(0))):
                    # Convert image back to numpy for visualization
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    
                    sample_images.append(img_np)
                    sample_gts.append(masks[i].cpu().numpy())
                    sample_preds.append(predictions[i].cpu().numpy())
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_predictions, all_targets, config.model.n_classes)
    
    # Print results
    print(f"\n=== Evaluation Results on {split.upper()} set ===")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Per-class metrics
    print("\nPer-class metrics:")
    for class_id in range(config.model.n_classes):
        class_name = f"Class {class_id}" if class_id == 0 else "Waste"
        if class_id == 0:
            class_name = "Background"
        
        iou_key = f'iou_class_{class_id}'
        dice_key = f'dice_class_{class_id}'
        
        if iou_key in metrics:
            print(f"  {class_name}:")
            print(f"    IoU: {metrics[iou_key]:.4f}")
            print(f"    Dice: {metrics[dice_key]:.4f}")
    
    # Save sample predictions
    if sample_images:
        sample_dir = os.path.join(config.output.visualization_save_dir, f'{split}_samples')
        save_sample_predictions(
            sample_images[:8],
            sample_gts[:8],
            sample_preds[:8],
            sample_dir,
            prefix=f'{split}_sample'
        )
        print(f"\nSample predictions saved to: {sample_dir}")
    
    # Save metrics to file
    import json
    metrics_file = os.path.join(config.output.results_save_dir, f'{split}_metrics.json')
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    evaluate_model(args.config, args.model, args.split)


if __name__ == '__main__':
    main()