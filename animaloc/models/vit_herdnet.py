__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: December 16, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.3.0"


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .register import MODELS
from .vit_backbone import vit_base, vit_small, vit_large


class ViTUpsampler(nn.Module):
    """Upsampling module for ViT features to match original resolution"""
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Use bilinear interpolation instead of transposed convolution to control size better
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Use bilinear interpolation for more controlled upsampling
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class ViTDecoder(nn.Module):
    """Decoder module that aggregates ViT multi-scale features"""
    
    def __init__(self, channels: list, target_channels: int = 64):
        super().__init__()
        self.channels = channels
        self.target_channels = target_channels
        
        # Upsampling modules for each scale
        self.upsamplers = nn.ModuleList()
        
        # Process features from deepest to shallowest
        # Limit the number of upsampling stages to prevent excessive upsampling
        num_stages = min(len(channels), 3)  # Maximum 3 upsampling stages
        
        for i, ch in enumerate(reversed(channels[-num_stages:])):
            if i == 0:
                # First upsampler (deepest features) - use smaller scale factor
                upsampler = ViTUpsampler(ch, target_channels, scale_factor=2)
            else:
                # Subsequent upsamplers with skip connections - use smaller scale factor
                upsampler = ViTUpsampler(target_channels + ch, target_channels, scale_factor=2)
            
            self.upsamplers.append(upsampler)
        
        # Final projection to target channels
        self.final_conv = nn.Conv2d(target_channels, target_channels, kernel_size=1)
        
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of ViT features from shallow to deep
        Returns:
            Upsampled feature map
        """
        # Reverse to process from deep to shallow and limit to the number we can handle
        num_stages = min(len(features), 3)  # Match the number from __init__
        features = list(reversed(features[-num_stages:]))
        
        x = features[0]  # Start with deepest features
        
        # Progressive upsampling with skip connections
        for i, upsampler in enumerate(self.upsamplers):
            if i == 0:
                x = upsampler(x)
            else:
                # Add skip connection from shallower feature
                skip_feat = features[i]
                # Resize skip connection to match current resolution
                if skip_feat.shape[-2:] != x.shape[-2:]:
                    # Add bounds check for interpolation
                    target_h, target_w = x.shape[-2:]
                    max_elements = 2**31 - 1
                    total_elements = skip_feat.shape[0] * skip_feat.shape[1] * target_h * target_w
                    
                    if total_elements > max_elements:
                        # Skip this feature if it would cause overflow
                        continue
                    
                    skip_feat = F.interpolate(
                        skip_feat, size=x.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                x = torch.cat([x, skip_feat], dim=1)
                x = upsampler(x)
                
                # Add additional bounds check after upsampling
                max_elements = 2**31 - 1  
                total_elements = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
                if total_elements > max_elements:
                    # Limit the size if it gets too large
                    max_dim = int((max_elements // (x.shape[0] * x.shape[1])) ** 0.5)
                    if x.shape[2] > max_dim or x.shape[3] > max_dim:
                        new_h = min(x.shape[2], max_dim)
                        new_w = min(x.shape[3], max_dim)
                        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        x = self.final_conv(x)
        return x


@MODELS.register()
class ViTHerdNet(nn.Module):
    """HerdNet with Vision Transformer backbone"""
    
    def __init__(
        self,
        vit_size: str = 'base',  # 'small', 'base', 'large'
        img_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: Optional[int] = 2,
        head_conv: int = 64
    ):
        """
        Args:
            vit_size (str): Size of ViT backbone ('small', 'base', 'large')
            img_size (int): Input image size
            patch_size (int): ViT patch size
            num_classes (int): Number of output classes, background included
            pretrained (bool): Use pretrained ViT weights
            down_ratio (int): Downsample ratio for output
            head_conv (int): Channels in detection heads
        """
        super().__init__()
        
        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.patch_size = patch_size
        
        # ViT backbone selection
        if vit_size == 'small':
            self.backbone = vit_small(pretrained=pretrained, img_size=img_size, patch_size=patch_size)
        elif vit_size == 'base':
            self.backbone = vit_base(pretrained=pretrained, img_size=img_size, patch_size=patch_size)
        elif vit_size == 'large':
            self.backbone = vit_large(pretrained=pretrained, img_size=img_size, patch_size=patch_size)
        else:
            raise ValueError(f"Unsupported ViT size: {vit_size}")
        
        # Get channel configuration from backbone
        self.channels = self.backbone.channels
        
        # Decoder for aggregating multi-scale ViT features
        self.decoder = ViTDecoder(self.channels, target_channels=head_conv)
        
        # Classification head (operates on deepest features)
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.channels[-1], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        # Localization head (operates on aggregated features)
        self.loc_head = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        # Initialize head weights
        self._init_heads()
        
        # Calculate the number of upsampling steps needed
        feature_map_size = img_size // patch_size
        target_size = img_size // down_ratio
        self.final_upsample_factor = target_size // feature_map_size
        
    def _init_heads(self):
        """Initialize detection head weights"""
        self.loc_head[-2].bias.data.fill_(0.00)
        self.cls_head[-1].bias.data.fill_(0.00)
        
        for m in [self.loc_head, self.cls_head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, std=0.001)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, input: torch.Tensor):
        """
        Forward pass
        
        Args:
            input: Input tensor of shape (B, 3, H, W)
            
        Returns:
            heatmap: Localization heatmap (B, 1, H//down_ratio, W//down_ratio)
            clsmap: Classification map (B, num_classes, H//patch_size, W//patch_size)
        """
        # Extract multi-scale features from ViT backbone
        features = self.backbone(input)  # List of feature maps
        
        # Use deepest features for classification
        cls_features = features[-1]  # Deepest features
        clsmap = self.cls_head(cls_features)
        
        # Aggregate features for localization
        loc_features = self.decoder(features)
        
        # Apply final upsampling if needed to match target resolution
        target_height = input.shape[2] // self.down_ratio
        target_width = input.shape[3] // self.down_ratio
        
        # Check if upsampling is needed and if the target size is reasonable
        if loc_features.shape[2] != target_height or loc_features.shape[3] != target_width:
            # Add bounds check to prevent memory overflow
            max_elements = 2**31 - 1  # INT_MAX
            total_elements = loc_features.shape[0] * loc_features.shape[1] * target_height * target_width
            
            if total_elements > max_elements:
                # If target size would be too large, use a smaller intermediate size
                max_dim = int((max_elements // (loc_features.shape[0] * loc_features.shape[1])) ** 0.5)
                target_height = min(target_height, max_dim)
                target_width = min(target_width, max_dim)
            
            loc_features = F.interpolate(
                loc_features, 
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
        
        heatmap = self.loc_head(loc_features)
        
        return heatmap, clsmap
    
    def freeze(self, layers: list) -> None:
        """Freeze specified layers"""
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        """Freeze all parameters in a specific layer"""
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    
    def reshape_classes(self, num_classes: int) -> None:
        """Reshape classification head for new number of classes"""
        self.cls_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes


# Factory functions for different configurations
@MODELS.register()
def ViTHerdNet_Small(**kwargs):
    """ViT-Small HerdNet"""
    kwargs.pop('vit_size', None)  # Remove vit_size if present to avoid duplicate
    return ViTHerdNet(vit_size='small', **kwargs)


@MODELS.register()
def ViTHerdNet_Base(**kwargs):
    """ViT-Base HerdNet"""
    kwargs.pop('vit_size', None)  # Remove vit_size if present to avoid duplicate
    return ViTHerdNet(vit_size='base', **kwargs)


@MODELS.register()
def ViTHerdNet_Large(**kwargs):
    """ViT-Large HerdNet"""
    kwargs.pop('vit_size', None)  # Remove vit_size if present to avoid duplicate
    return ViTHerdNet(vit_size='large', **kwargs)