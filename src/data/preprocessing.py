"""
Data preprocessing utilities for waste segmentation.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional
import os


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single image for model input.
    
    Args:
        image: Input image as numpy array
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Preprocessed image
    """
    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize if requested
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def preprocess_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    num_classes: int = 2
) -> np.ndarray:
    """
    Preprocess a segmentation mask.
    
    Args:
        mask: Input mask as numpy array
        target_size: Target size (height, width)
        num_classes: Number of segmentation classes
    
    Returns:
        Preprocessed mask
    """
    # Resize mask using nearest neighbor interpolation
    mask = cv2.resize(
        mask, 
        (target_size[1], target_size[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Ensure mask values are in valid range
    mask = np.clip(mask, 0, num_classes - 1)
    
    return mask


def augment_conveyor_data(
    image: np.ndarray,
    mask: np.ndarray,
    augment_prob: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply conveyor belt specific augmentations.
    
    Args:
        image: Input image
        mask: Input mask
        augment_prob: Probability of applying each augmentation
    
    Returns:
        Augmented image and mask
    """
    # Horizontal flip (common for conveyor belt scenarios)
    if np.random.random() < augment_prob:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    # Add slight rotation (conveyor belts are usually straight)
    if np.random.random() < augment_prob * 0.3:  # Less likely
        angle = np.random.uniform(-5, 5)  # Small rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image = cv2.warpAffine(image, matrix, (w, h))
        mask = cv2.warpAffine(mask, matrix, (w, h))
    
    # Brightness and contrast adjustments
    if np.random.random() < augment_prob:
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-20, 20)    # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image, mask


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> None:
    """
    Split data into train/validation/test sets.
    
    Args:
        data_dir: Directory containing images and masks
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Get all image files
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle files
    np.random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create split directories and move files
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'masks'), exist_ok=True)
    
    print(f"Data split complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")