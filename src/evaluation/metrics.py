"""
Evaluation metrics for segmentation tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Union, Optional


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate pixel accuracy.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        
    Returns:
        Pixel accuracy
    """
    pred = pred.argmax(dim=1) if pred.dim() > 3 else pred
    target = target.long()
    
    correct = (pred == target).sum().item()
    total = target.numel()
    
    return correct / total


def intersection_over_union(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    num_classes: int,
    ignore_index: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate IoU for each class and mean IoU.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        ignore_index: Index to ignore in calculation
        
    Returns:
        Dictionary with IoU scores
    """
    pred = pred.argmax(dim=1) if pred.dim() > 3 else pred
    target = target.long()
    
    ious = {}
    valid_ious = []
    
    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue
            
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
            
        ious[f'iou_class_{class_id}'] = iou
        valid_ious.append(iou)
    
    ious['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0
    
    return ious


def dice_coefficient(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    num_classes: int,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Calculate Dice coefficient for each class and mean Dice.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dictionary with Dice scores
    """
    pred = pred.argmax(dim=1) if pred.dim() > 3 else pred
    target = target.long()
    
    dices = {}
    valid_dices = []
    
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()
        
        intersection = (pred_mask * target_mask).sum()
        dice = (2.0 * intersection + smooth) / (pred_mask.sum() + target_mask.sum() + smooth)
        
        dices[f'dice_class_{class_id}'] = dice.item()
        valid_dices.append(dice.item())
    
    dices['mean_dice'] = np.mean(valid_dices)
    
    return dices


def calculate_metrics(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: List of predicted masks
        targets: List of ground truth masks
        num_classes: Number of classes
        
    Returns:
        Dictionary with all metrics
    """
    all_metrics = {
        'pixel_accuracy': [],
        'mean_iou': [],
        'mean_dice': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    # Add per-class metrics
    for class_id in range(num_classes):
        all_metrics[f'iou_class_{class_id}'] = []
        all_metrics[f'dice_class_{class_id}'] = []
    
    for pred, target in zip(predictions, targets):
        # Pixel accuracy
        acc = pixel_accuracy(pred, target)
        all_metrics['pixel_accuracy'].append(acc)
        
        # IoU metrics
        iou_metrics = intersection_over_union(pred, target, num_classes)
        for key, value in iou_metrics.items():
            if key in all_metrics:
                all_metrics[key].append(value)
        
        # Dice metrics
        dice_metrics = dice_coefficient(pred, target, num_classes)
        for key, value in dice_metrics.items():
            if key in all_metrics:
                all_metrics[key].append(value)
        
        # Convert to numpy for sklearn metrics
        pred_np = pred.argmax(dim=1).flatten().cpu().numpy() if pred.dim() > 3 else pred.flatten().cpu().numpy()
        target_np = target.flatten().cpu().numpy()
        
        # Precision, Recall, F1 (macro average)
        precision = precision_score(target_np, pred_np, average='macro', zero_division=0)
        recall = recall_score(target_np, pred_np, average='macro', zero_division=0)
        f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)
        
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1_score'].append(f1)
    
    # Calculate mean of all metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        final_metrics[key] = np.mean(values)
    
    return final_metrics


class SegmentationMetrics:
    """
    Class to compute and track segmentation metrics during training.
    """
    
    def __init__(self, num_classes: int = 2, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth mask
        """
        self.predictions.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with computed metrics
        """
        if not self.predictions:
            return {}
        
        return calculate_metrics(self.predictions, self.targets, self.num_classes)
    
    def compute_batch(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for a single batch.
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth mask
            
        Returns:
            Dictionary with computed metrics
        """
        return calculate_metrics([pred], [target], self.num_classes)


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        pred: Predicted logits (N, C, H, W)
        target: Ground truth labels (N, H, W)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: Reduction method
        
    Returns:
        Focal loss
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Dice Loss for segmentation.
    
    Args:
        pred: Predicted probabilities (N, C, H, W)
        target: Ground truth labels (N, H, W)
        smooth: Smoothing factor
        
    Returns:
        Dice loss
    """
    pred_probs = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
    
    intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()
    
    return dice_loss