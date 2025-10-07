"""
Dataset classes for loading and processing waste segmentation data.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import Optional, Callable, Tuple, List


class WasteSegmentationDataset(Dataset):
    """
    Dataset class for waste segmentation on conveyor belt images.
    
    Args:
        data_dir: Path to the data directory
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to be applied on images
        target_transform: Optional transform to be applied on masks
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image and mask paths
        self.image_paths, self.mask_paths = self._load_paths()
        
    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """Load image and mask file paths."""
        split_dir = os.path.join(self.data_dir, self.split)
        image_dir = os.path.join(split_dir, 'images')
        mask_dir = os.path.join(split_dir, 'masks')
        
        image_paths = []
        mask_paths = []
        
        if os.path.exists(image_dir):
            for filename in sorted(os.listdir(image_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, filename)
                    mask_name = filename.rsplit('.', 1)[0] + '.png'
                    mask_path = os.path.join(mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
        
        return image_paths, mask_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale for masks
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return image, mask


def get_transforms(image_size: Tuple[int, int] = (512, 512), is_training: bool = True):
    """
    Get standard transforms for waste segmentation.
    
    Args:
        image_size: Target image size (height, width)
        is_training: Whether this is for training (includes augmentations)
    
    Returns:
        Tuple of (image_transform, mask_transform)
    """
    if is_training:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    mask_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform