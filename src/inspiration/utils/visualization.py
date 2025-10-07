"""
Utility functions for visualization and plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import seaborn as sns
from typing import List, Tuple, Optional, Union
import cv2


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        train_metrics: Training metric values
        val_metrics: Validation metric values
        metric_name: Name of the metric
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics if provided
    if train_metrics is not None and val_metrics is not None:
        axes[1].plot(train_metrics, label=f'Train {metric_name}', color='blue')
        axes[1].plot(val_metrics, label=f'Validation {metric_name}', color='red')
        axes[1].set_title(f'Training and Validation {metric_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_segmentation(
    image: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    prediction: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize image, ground truth mask, and prediction.
    
    Args:
        image: Input image
        mask: Ground truth mask
        prediction: Predicted mask (optional)
        class_names: Names of the classes
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if prediction is not None and isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Handle different image formats
    if image.ndim == 3 and image.shape[0] == 3:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    
    # Normalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Set up the plot
    n_cols = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    if n_cols == 2:
        axes = [axes[0], axes[1]]
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(mask, cmap='jet', alpha=0.8)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction if provided
    if prediction is not None:
        axes[2].imshow(prediction, cmap='jet', alpha=0.8)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gan_samples(
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    n_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Plot real and generated samples side by side.
    
    Args:
        real_samples: Real images
        fake_samples: Generated images
        n_samples: Number of samples to display
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    for i in range(n_samples):
        # Real samples
        real_img = real_samples[i].detach().cpu()
        if real_img.shape[0] == 3:  # CHW format
            real_img = real_img.permute(1, 2, 0)
        real_img = (real_img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        axes[0, i].imshow(real_img)
        axes[0, i].set_title('Real')
        axes[0, i].axis('off')
        
        # Fake samples
        fake_img = fake_samples[i].detach().cpu()
        if fake_img.shape[0] == 3:  # CHW format
            fake_img = fake_img.permute(1, 2, 0)
        fake_img = (fake_img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        axes[1, i].imshow(fake_img)
        axes[1, i].set_title('Generated')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0])
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_class_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    class_colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay of segmentation mask on original image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Segmentation mask (H, W)
        class_colors: Colors for each class
        alpha: Transparency of the overlay
        
    Returns:
        Image with overlay
    """
    if class_colors is None:
        class_colors = [(0, 0, 0), (255, 0, 0)]  # Black for background, red for waste
    
    overlay = np.zeros_like(image)
    
    for class_id, color in enumerate(class_colors):
        overlay[mask == class_id] = color
    
    # Blend original image with overlay
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def save_sample_predictions(
    images: List[np.ndarray],
    ground_truths: List[np.ndarray],
    predictions: List[np.ndarray],
    save_dir: str,
    prefix: str = "sample"
) -> None:
    """
    Save sample predictions as images.
    
    Args:
        images: List of original images
        ground_truths: List of ground truth masks
        predictions: List of predicted masks
        save_dir: Directory to save images
        prefix: Prefix for saved files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (img, gt, pred) in enumerate(zip(images, ground_truths, predictions)):
        # Create overlay images
        gt_overlay = create_class_overlay(img, gt)
        pred_overlay = create_class_overlay(img, pred)
        
        # Save individual images
        Image.fromarray(img).save(os.path.join(save_dir, f"{prefix}_{i}_original.png"))
        Image.fromarray(gt_overlay).save(os.path.join(save_dir, f"{prefix}_{i}_ground_truth.png"))
        Image.fromarray(pred_overlay).save(os.path.join(save_dir, f"{prefix}_{i}_prediction.png"))
        
        # Create combined visualization
        combined = np.hstack([img, gt_overlay, pred_overlay])
        Image.fromarray(combined).save(os.path.join(save_dir, f"{prefix}_{i}_combined.png"))