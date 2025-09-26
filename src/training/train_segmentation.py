"""
Training script for segmentation models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import argparse
from typing import Dict, Any

# Import local modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import WasteSegmentationDataset, get_transforms
from models.segmentation import get_segmentation_model
from evaluation.metrics import SegmentationMetrics, focal_loss, dice_loss
from utils.config import load_config, create_directories
from utils.visualization import plot_training_history, visualize_segmentation


class SegmentationTrainer:
    """Trainer class for segmentation models."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        create_directories(self.config)
        
        # Set device
        if self.config.hardware.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.hardware.device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_segmentation_model(
            self.config.model.segmentation_name,
            self.config.model.n_channels,
            self.config.model.n_classes
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Initialize scheduler
        if self.config.training.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        else:
            self.scheduler = None
        
        # Initialize loss function
        if self.config.training.loss_function == 'cross_entropy':
            class_weights = torch.tensor(self.config.training.class_weights, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.config.training.loss_function == 'focal':
            self.criterion = focal_loss
        elif self.config.training.loss_function == 'dice':
            self.criterion = dice_loss
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.train_metrics = SegmentationMetrics(self.config.model.n_classes, self.device)
        self.val_metrics = SegmentationMetrics(self.config.model.n_classes, self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_iou = 0.0
        
        # Initialize wandb if enabled
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=self.config.logging.experiment_name,
                config=self.config.__dict__
            )
    
    def create_data_loaders(self) -> tuple:
        """Create train and validation data loaders."""
        # Get transforms
        train_transform, train_target_transform = get_transforms(
            tuple(self.config.data.image_size), is_training=True
        )
        val_transform, val_target_transform = get_transforms(
            tuple(self.config.data.image_size), is_training=False
        )
        
        # Create datasets
        train_dataset = WasteSegmentationDataset(
            self.config.data.processed_data_dir,
            split='train',
            transform=train_transform,
            target_transform=train_target_transform
        )
        
        val_dataset = WasteSegmentationDataset(
            self.config.data.processed_data_dir,
            split='val',
            transform=val_transform,
            target_transform=val_target_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.training.epochs}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device).squeeze(1).long()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log batch metrics
            if (batch_idx + 1) % self.config.logging.log_frequency == 0:
                batch_metrics = self.train_metrics.compute_batch(outputs, masks)
                if self.config.logging.use_wandb:
                    wandb.log({
                        'train_batch_loss': loss.item(),
                        'train_batch_accuracy': batch_metrics.get('pixel_accuracy', 0),
                        'batch': epoch * num_batches + batch_idx
                    })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        epoch_metrics = self.train_metrics.compute()
        
        return {'loss': avg_loss, **epoch_metrics}
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device).squeeze(1).long()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                self.val_metrics.update(outputs, masks)
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        epoch_metrics = self.val_metrics.compute()
        
        return {'loss': avg_loss, **epoch_metrics}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_iou': self.best_val_iou,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config.output.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                self.config.output.model_save_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output.model_save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with IoU: {self.best_val_iou:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.config.training.epochs):
            # Train
            train_results = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_results['loss'])
            self.train_accuracies.append(train_results.get('pixel_accuracy', 0))
            
            # Validate
            if (epoch + 1) % self.config.evaluation.eval_frequency == 0:
                val_results = self.validate_epoch(val_loader, epoch)
                self.val_losses.append(val_results['loss'])
                self.val_accuracies.append(val_results.get('pixel_accuracy', 0))
                
                # Check if best model
                val_iou = val_results.get('mean_iou', 0)
                is_best = val_iou > self.best_val_iou
                if is_best:
                    self.best_val_iou = val_iou
                
                # Save checkpoint
                if self.config.evaluation.save_best_model:
                    self.save_checkpoint(epoch, is_best)
                
                # Log results
                print(f"Epoch {epoch+1}/{self.config.training.epochs}:")
                print(f"  Train Loss: {train_results['loss']:.4f}, "
                      f"Train Acc: {train_results.get('pixel_accuracy', 0):.4f}")
                print(f"  Val Loss: {val_results['loss']:.4f}, "
                      f"Val Acc: {val_results.get('pixel_accuracy', 0):.4f}, "
                      f"Val IoU: {val_iou:.4f}")
                
                # Log to wandb
                if self.config.logging.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train_loss': train_results['loss'],
                        'val_loss': val_results['loss'],
                        'train_accuracy': train_results.get('pixel_accuracy', 0),
                        'val_accuracy': val_results.get('pixel_accuracy', 0),
                        'val_iou': val_iou,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    wandb.log(log_dict)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
        
        # Save final model
        final_path = os.path.join(self.config.output.model_save_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_path)
        
        # Plot training history
        if len(self.val_losses) > 0:
            plot_path = os.path.join(
                self.config.output.visualization_save_dir,
                'training_history.png'
            )
            plot_training_history(
                self.train_losses[::self.config.evaluation.eval_frequency],
                self.val_losses,
                self.train_accuracies[::self.config.evaluation.eval_frequency],
                self.val_accuracies,
                save_path=plot_path
            )
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()
    
    trainer = SegmentationTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()