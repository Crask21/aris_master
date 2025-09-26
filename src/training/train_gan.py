"""
Training script for GAN models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import argparse
import numpy as np
from typing import Dict, Tuple

# Import local modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import WasteSegmentationDataset, get_transforms
from models.generators import WasteGAN, Generator, Discriminator
from utils.config import load_config, create_directories
from utils.visualization import plot_gan_samples


class GANTrainer:
    """Trainer class for GAN models."""
    
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
        
        # Initialize models
        self.generator = Generator(
            self.config.model.latent_dim,
            self.config.model.img_channels,
            self.config.model.feature_map_size
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.config.model.img_channels,
            self.config.model.feature_map_size
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.training.g_lr,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.training.d_lr,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.d_real_acc = []
        self.d_fake_acc = []
        
        # Fixed noise for visualization
        self.fixed_noise = torch.randn(16, self.config.model.latent_dim, device=self.device)
        
        # Initialize wandb if enabled
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=f"{self.config.logging.experiment_name}_gan",
                config=self.config.__dict__
            )
    
    def create_data_loader(self) -> DataLoader:
        """Create data loader for real images."""
        # Get transforms (only for images, no masks needed for GAN)
        transform, _ = get_transforms(
            tuple(self.config.data.image_size), is_training=True
        )
        
        # Create dataset (we'll only use images)
        dataset = WasteSegmentationDataset(
            self.config.data.processed_data_dir,
            split='train',
            transform=transform
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        return data_loader
    
    def train_discriminator(self, real_images: torch.Tensor) -> Tuple[float, float, float]:
        """
        Train discriminator for one step.
        
        Args:
            real_images: Batch of real images
            
        Returns:
            Tuple of (loss, real_accuracy, fake_accuracy)
        """
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)
        
        # Train on real images
        self.d_optimizer.zero_grad()
        
        real_outputs = self.discriminator(real_images)
        real_loss = self.criterion(real_outputs, real_labels)
        
        # Train on fake images
        noise = torch.randn(batch_size, self.config.model.latent_dim, device=self.device)
        fake_images = self.generator(noise).detach()  # Detach to avoid training generator
        fake_outputs = self.discriminator(fake_images)
        fake_loss = self.criterion(fake_outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Calculate accuracies
        real_acc = (real_outputs > 0.5).float().mean().item()
        fake_acc = (fake_outputs < 0.5).float().mean().item()
        
        return d_loss.item(), real_acc, fake_acc
    
    def train_generator(self, batch_size: int) -> float:
        """
        Train generator for one step.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Generator loss
        """
        # Labels for generator training (we want discriminator to think fake images are real)
        real_labels = torch.ones(batch_size, device=self.device)
        
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, self.config.model.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        
        # Get discriminator's opinion on fake images
        fake_outputs = self.discriminator(fake_images)
        g_loss = self.criterion(fake_outputs, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate samples for visualization."""
        with torch.no_grad():
            noise = torch.randn(num_samples, self.config.model.latent_dim, device=self.device)
            samples = self.generator(noise)
        return samples
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'config': self.config
        }
        
        if (epoch + 1) % self.config.output.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                self.config.output.model_save_dir,
                f'gan_checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
    
    def train(self):
        """Main training loop."""
        print("Starting GAN training...")
        
        # Create data loader
        data_loader = self.create_data_loader()
        print(f"Training samples: {len(data_loader.dataset)}")
        
        for epoch in range(self.config.training.epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            epoch_d_real_acc = []
            epoch_d_fake_acc = []
            
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.config.training.epochs}')
            
            for batch_idx, (real_images, _) in enumerate(progress_bar):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Train Discriminator
                d_loss, d_real_acc, d_fake_acc = self.train_discriminator(real_images)
                epoch_d_losses.append(d_loss)
                epoch_d_real_acc.append(d_real_acc)
                epoch_d_fake_acc.append(d_fake_acc)
                
                # Train Generator
                g_loss = self.train_generator(batch_size)
                epoch_g_losses.append(g_loss)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'D_loss': d_loss,
                    'G_loss': g_loss,
                    'D_real_acc': d_real_acc,
                    'D_fake_acc': d_fake_acc
                })
                
                # Log batch metrics
                if (batch_idx + 1) % self.config.logging.log_frequency == 0:
                    if self.config.logging.use_wandb:
                        wandb.log({
                            'batch_d_loss': d_loss,
                            'batch_g_loss': g_loss,
                            'batch_d_real_acc': d_real_acc,
                            'batch_d_fake_acc': d_fake_acc,
                            'batch': epoch * len(data_loader) + batch_idx
                        })
            
            # Calculate epoch averages
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            avg_d_real_acc = np.mean(epoch_d_real_acc)
            avg_d_fake_acc = np.mean(epoch_d_fake_acc)
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.d_real_acc.append(avg_d_real_acc)
            self.d_fake_acc.append(avg_d_fake_acc)
            
            print(f"Epoch {epoch+1}/{self.config.training.epochs}:")
            print(f"  Generator Loss: {avg_g_loss:.4f}")
            print(f"  Discriminator Loss: {avg_d_loss:.4f}")
            print(f"  D Real Accuracy: {avg_d_real_acc:.4f}")
            print(f"  D Fake Accuracy: {avg_d_fake_acc:.4f}")
            
            # Log to wandb
            if self.config.logging.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'generator_loss': avg_g_loss,
                    'discriminator_loss': avg_d_loss,
                    'd_real_accuracy': avg_d_real_acc,
                    'd_fake_accuracy': avg_d_fake_acc
                }
                wandb.log(log_dict)
            
            # Generate and save samples
            if (epoch + 1) % self.config.evaluation.eval_frequency == 0:
                # Generate samples with fixed noise
                with torch.no_grad():
                    fake_samples = self.generator(self.fixed_noise)
                
                # Save samples
                sample_path = os.path.join(
                    self.config.output.visualization_save_dir,
                    f'generated_samples_epoch_{epoch+1}.png'
                )
                
                # Get some real samples for comparison
                real_samples = next(iter(data_loader))[0][:16].to(self.device)
                
                plot_gan_samples(
                    real_samples,
                    fake_samples,
                    n_samples=8,
                    save_path=sample_path
                )
                
                if self.config.logging.use_wandb:
                    wandb.log({
                        'generated_samples': wandb.Image(sample_path),
                        'epoch': epoch + 1
                    })
            
            # Save checkpoint
            self.save_checkpoint(epoch)
        
        # Save final models
        torch.save(self.generator.state_dict(), 
                  os.path.join(self.config.output.model_save_dir, 'final_generator.pth'))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(self.config.output.model_save_dir, 'final_discriminator.pth'))
        
        print("GAN training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train GAN model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()
    
    trainer = GANTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()