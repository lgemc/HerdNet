__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: September 16, 2025
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from typing import Optional, List
from .register import MODELS


class ConvNeXtFeatureExtractor(nn.Module):
    """ConvNeXt feature extractor that returns multi-scale features"""

    def __init__(self, model_name: str = 'convnext_base', pretrained: bool = True):
        super().__init__()

        # Load ConvNeXt model
        if model_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
            self.feature_channels = [96, 96, 192, 384, 768]
        elif model_name == 'convnext_small':
            self.backbone = models.convnext_small(weights='DEFAULT' if pretrained else None)
            self.feature_channels = [96, 96, 192, 384, 768]
        elif model_name == 'convnext_base':
            self.backbone = models.convnext_base(weights='DEFAULT' if pretrained else None)
            self.feature_channels = [128, 128, 256, 512, 1024]
        elif model_name == 'convnext_large':
            self.backbone = models.convnext_large(weights='DEFAULT' if pretrained else None)
            self.feature_channels = [192, 192, 384, 768, 1536]
        else:
            raise ValueError(f"Unsupported ConvNeXt model: {model_name}")

        # Extract only the feature extractor (remove classifier)
        self.features = self.backbone.features

        # Define which stages to extract features from
        self.extract_stages = [1, 3, 5, 7]  # After downsampling stages

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from ConvNeXt backbone"""
        features = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.extract_stages:
                features.append(x)

        return features


class SimpleUpsampling(nn.Module):
    """Simple upsampling module that reaches target output resolution"""

    def __init__(self, channels: List[int], target_stride: int = 2):
        super().__init__()
        self.channels = channels
        self.target_stride = target_stride

        # Calculate how much upsampling we need
        # ConvNeXt features start at different strides: [4, 8, 16, 32] for stages [1, 3, 5, 7]
        # We need to reach stride = target_stride (e.g., 2)

        # Create upsampling layers to progressively reduce stride
        self.up_layers = nn.ModuleList()

        # Start from deepest features and work backwards
        current_channels = channels[-1]

        # First, upsample to match the stride of the second-to-last features
        for i in range(len(channels) - 1, 0, -1):
            next_channels = channels[i - 1]

            up_layer = nn.Sequential(
                nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True)
            )
            self.up_layers.append(up_layer)
            current_channels = next_channels

        # Add final upsampling layers to reach target stride
        # ConvNeXt stage 1 has stride 4, we need to reach target_stride
        final_upsample_factor = 4 // target_stride
        if final_upsample_factor > 1:
            # Use multiple 2x upsampling steps for better results
            while final_upsample_factor > 1:
                upsample_step = min(2, final_upsample_factor)
                final_up = nn.Sequential(
                    nn.ConvTranspose2d(current_channels, current_channels,
                                       kernel_size=4, stride=upsample_step, padding=1, bias=False),
                    nn.BatchNorm2d(current_channels),
                    nn.ReLU(inplace=True)
                )
                self.up_layers.append(final_up)
                final_upsample_factor //= upsample_step

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Progressively upsample and combine features"""
        x = features[-1]  # Start with deepest features

        # Upsample through all feature levels
        feature_idx = len(features) - 2
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x)

            # Add skip connection if we have a corresponding feature level
            if feature_idx >= 0 and i < len(features) - 1:
                skip_feat = features[feature_idx]

                # Ensure spatial dimensions match
                if x.shape[2:] != skip_feat.shape[2:]:
                    x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)

                # Add skip connection if channels match
                if x.shape[1] == skip_feat.shape[1]:
                    x = x + skip_feat

                feature_idx -= 1

        return x


@MODELS.register()
class HerdNetConvNeXt(nn.Module):
    """HerdNet architecture with ConvNeXt backbone"""

    def __init__(
        self,
        model_name: str = 'convnext_base',
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: Optional[int] = 2,
        head_conv: int = 64
    ):
        """
        Args:
            model_name (str): ConvNeXt model variant ('convnext_tiny', 'convnext_small',
                             'convnext_base', 'convnext_large'). Defaults to 'convnext_base'.
            num_classes (int): number of output classes, background included. Defaults to 2.
            pretrained (bool): set False to disable pretrained weights from ImageNet. Defaults to True.
            down_ratio (int): downsample ratio. Possible values are 1, 2, 4, 8, or 16.
                             Set to 1 to get output of the same size as input. Defaults to 2.
            head_conv (int): number of channels in detection heads. Defaults to 64.
        """
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.model_name = model_name

        # ConvNeXt backbone
        self.backbone = ConvNeXtFeatureExtractor(model_name, pretrained)

        # Calculate which features to use based on down_ratio
        # ConvNeXt features have strides [4, 8, 16, 32], we need to choose starting point
        feature_strides = [4, 8, 16, 32]
        start_idx = 0  # Always start from stride 4 feature
        for i, stride in enumerate(feature_strides):
            if stride >= down_ratio * 2:  # Start from a feature with enough resolution
                start_idx = i
                break

        # Get feature channels for upsampling
        feature_channels = self.backbone.feature_channels[1:]  # Skip first layer [128, 256, 512, 1024]
        channels = feature_channels[start_idx:]
        self.start_idx = start_idx  # Store for forward method

        # Upsampling module
        self.upsampler = SimpleUpsampling(channels, target_stride=down_ratio)

        # Localization head (for heatmap)
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[0], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Classification head - should output same resolution as heatmap
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels[0], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Initialize biases
        self.loc_head[-2].bias.data.fill_(0.00)
        self.cls_head[-1].bias.data.fill_(0.00)

    def forward(self, input: torch.Tensor):
        """Forward pass"""
        # Extract multi-scale features
        features = self.backbone(input)

        # Use features from start_idx onwards for both tasks
        loc_features = features[self.start_idx:]

        # Upsample features for both localization and classification
        upsampled = self.upsampler(loc_features)

        # Generate outputs
        heatmap = self.loc_head(upsampled)
        clsmap = self.cls_head(upsampled)

        return heatmap, clsmap

    def freeze(self, layers: List[str]) -> None:
        """Freeze specified layers"""
        for layer_name in layers:
            if hasattr(self, layer_name):
                for param in getattr(self, layer_name).parameters():
                    param.requires_grad = False
            else:
                print(f"Warning: Layer '{layer_name}' not found")

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture according to a new number of classes"""
        self.cls_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes