"""
Generative models for synthetic data generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    """
    DCGAN-style Generator for synthetic waste image generation.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 3,
        feature_map_size: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        
        # Initial projection and reshape
        self.fc = nn.Linear(latent_dim, feature_map_size * 8 * 4 * 4)
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_map_size, feature_map_size // 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size // 2),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(feature_map_size // 2, feature_map_size // 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size // 4),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(feature_map_size // 4, feature_map_size // 8, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size // 8),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(feature_map_size // 8, img_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, -1, 4, 4)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    """
    DCGAN-style Discriminator for synthetic waste image generation.
    """
    
    def __init__(
        self,
        img_channels: int = 3,
        feature_map_size: int = 64
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(img_channels, feature_map_size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256x256 -> 128x128
            nn.Conv2d(feature_map_size // 8, feature_map_size // 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(feature_map_size // 4, feature_map_size // 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(feature_map_size // 2, feature_map_size, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1).squeeze()


class WasteGAN(nn.Module):
    """
    Complete GAN model for waste image generation.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 3,
        feature_map_size: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, img_channels, feature_map_size)
        self.discriminator = Discriminator(img_channels, feature_map_size)
    
    def generate_samples(self, num_samples: int, device: str = 'cpu'):
        """Generate synthetic samples."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.generator(z)
        return samples
    
    def discriminate(self, images):
        """Discriminate real vs fake images."""
        return self.discriminator(images)