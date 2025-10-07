"""
Segmentation models for waste classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation of waste on conveyor belts.
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes
        bilinear: Use bilinear upsampling instead of transposed convolutions
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2, bilinear: bool = True):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class DeepLabV3Plus(nn.Module):
    """
    Simplified DeepLabV3+ implementation for waste segmentation.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (simplified ResNet-like backbone)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Block 1
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(512, 256, 3, padding=6, dilation=6),
            nn.Conv2d(512, 256, 3, padding=12, dilation=12),
            nn.Conv2d(512, 256, 3, padding=18, dilation=18),
        ])
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(512, 256, 1)
        
        # Fusion
        self.fusion_conv = nn.Conv2d(256 * 5, 256, 1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
    
    def forward(self, x):
        input_size = x.size()
        
        # Encoder
        features = self.encoder(x)
        
        # ASPP
        aspp_features = []
        for aspp_layer in self.aspp:
            aspp_features.append(aspp_layer(features))
        
        # Global Average Pooling
        gap_feature = self.global_avg_pool(features)
        gap_feature = self.gap_conv(gap_feature)
        gap_feature = F.interpolate(
            gap_feature, size=features.size()[2:], 
            mode='bilinear', align_corners=False
        )
        aspp_features.append(gap_feature)
        
        # Concatenate ASPP features
        x = torch.cat(aspp_features, dim=1)
        x = self.fusion_conv(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(
            x, size=input_size[2:], 
            mode='bilinear', align_corners=False
        )
        
        return x


def get_segmentation_model(
    model_name: str,
    n_channels: int = 3,
    n_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to get segmentation models.
    
    Args:
        model_name: Name of the model ('unet', 'deeplabv3plus')
        n_channels: Number of input channels
        n_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Segmentation model
    """
    if model_name.lower() == 'unet':
        return UNet(n_channels, n_classes, **kwargs)
    elif model_name.lower() == 'deeplabv3plus':
        return DeepLabV3Plus(n_channels, n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")