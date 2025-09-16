__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from .register import MODELS


class DINOv3Backbone(nn.Module):
    """DINOv3 Vision Transformer backbone for feature extraction"""

    def __init__(
        self,
        model_name: str = 'dinov3_vitb16',
        pretrained_weights: Optional[str] = None,
        repo_path: Optional[str] = None,
        return_levels: bool = True,
        feature_layers: List[int] = [3, 6, 9, 12],
        img_size: int = 224
    ):
        """
        Args:
            model_name (str): DINOv3 model variant (dinov3_vitb16, dinov3_vitl16, etc.)
            pretrained_weights (str, optional): Path or URL to pretrained weights
            repo_path (str, optional): Path to local DINOv3 repository
            return_levels (bool): Whether to return multi-level features
            feature_layers (List[int]): Which transformer layers to extract features from
            img_size (int): Input image size for patch embedding calculation
        """
        super(DINOv3Backbone, self).__init__()

        self.model_name = model_name
        self.return_levels = return_levels
        self.feature_layers = feature_layers
        self.img_size = img_size

        # Load DINOv3 model from local repository or torch hub
        # Check if local DINOv3 repo exists
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(current_dir))
        local_dinov3_path = os.path.join(repo_root, 'dinov3')

        if repo_path:
            dinov3_repo = repo_path
        elif os.path.exists(local_dinov3_path):
            dinov3_repo = local_dinov3_path
        else:
            dinov3_repo = 'facebookresearch/dinov3'

        # Model mapping for torch hub
        hub_model_mapping = {
            'dinov3_vitb16': 'dinov3_vitb16',
            'dinov3_vitl16': 'dinov3_vitl16',
            'dinov3_vits16': 'dinov3_vits16',
            'dinov3_vith16plus': 'dinov3_vith16plus'
        }

        hub_model_name = hub_model_mapping.get(model_name, 'dinov3_vitb16')

        # Load from local repo if available, otherwise fallback to remote
        if os.path.exists(dinov3_repo):
            self.backbone = torch.hub.load(dinov3_repo, hub_model_name, source='local', pretrained=False)
        else:
            self.backbone = torch.hub.load('facebookresearch/dinov3', hub_model_name)

        # Load pretrained weights if provided
        if pretrained_weights and os.path.exists(pretrained_weights):
            print(f"Loading DINOv3 pretrained weights from: {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Load the state dict into the backbone
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys[:5]}...")  # Show first 5
        elif pretrained_weights:
            print(f"Warning: Pretrained weights path does not exist: {pretrained_weights}")

        # Get model dimensions
        self._setup_model_info()

        # Feature extraction hooks
        self.features = {}
        self._register_hooks()

    def _setup_model_info(self):
        """Setup model-specific information"""
        # Model dimension mapping
        model_dims = {
            'dinov3_vits16': 384,
            'dinov3_vitb16': 768,
            'dinov3_vitl16': 1024,
            'dinov3_vith16plus': 1280
        }

        self.embed_dim = model_dims.get(self.model_name, 768)
        self.patch_size = 16  # All DINOv3 models use patch size 16
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # Calculate channels for multi-level features (matching DLA style)
        if self.return_levels:
            # Create channel progression similar to DLA
            base_channels = [64, 128, 256, 512, 1024, 2048]
            self.channels = base_channels[:len(self.feature_layers)]
        else:
            self.channels = [self.embed_dim]

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        if not self.return_levels:
            return

        def get_activation(name):
            def hook(model, input, output):
                # Handle different output formats
                if isinstance(output, (list, tuple)):
                    # Some models return (hidden_states, attention_weights)
                    self.features[name] = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    # HuggingFace transformer output
                    self.features[name] = output.last_hidden_state
                else:
                    # Standard tensor output
                    self.features[name] = output
            return hook

        # Register hooks on transformer blocks
        if hasattr(self.backbone, 'blocks'):
            # Standard DINOv3 model structure
            for i, layer_idx in enumerate(self.feature_layers):
                if layer_idx < len(self.backbone.blocks):
                    self.backbone.blocks[layer_idx].register_forward_hook(
                        get_activation(f'layer_{i}')
                    )
        elif hasattr(self.backbone, 'encoder'):
            # HuggingFace model structure
            for i, layer_idx in enumerate(self.feature_layers):
                if layer_idx < len(self.backbone.encoder.layer):
                    self.backbone.encoder.layer[layer_idx].register_forward_hook(
                        get_activation(f'layer_{i}')
                    )

    def _reshape_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape 1D patch features to 2D spatial features"""
        # Handle different input types
        if isinstance(features, (list, tuple)):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(features)}")

        # Ensure we have a 3D tensor [B, N, C]
        if features.dim() == 2:
            # Add batch dimension if missing
            features = features.unsqueeze(0)
        elif features.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, N, C], got shape {features.shape}")

        B, N, C = features.shape

        # Handle CLS token removal - check if we have N = H*W + 1 (with CLS token)
        # Calculate expected number of patches dynamically
        has_cls_token = False
        if int(math.sqrt(N - 1)) ** 2 == (N - 1):
            # We have CLS token
            has_cls_token = True
            features = features[:, 1:]  # Remove CLS token
            N = N - 1
        elif int(math.sqrt(N)) ** 2 != N:
            # Check if it's a different format - try to infer dimensions
            print(f"Warning: Unexpected number of patches: {N}, trying to infer dimensions")
            # Try to find the closest square number
            H = W = int(math.sqrt(N))
            if H * W != N:
                # If not perfect square, use the tensor size to compute
                total_elements = B * N * C
                # Assume square spatial dimensions
                spatial_dim = int(math.sqrt(N))
                H = W = spatial_dim
                if H * W != N:
                    raise ValueError(f"Cannot reshape features with {N} patches into square spatial dimensions")

        # Reshape to spatial dimensions
        if not has_cls_token or int(math.sqrt(N)) ** 2 == N:
            H = W = int(math.sqrt(N))

        # Verify the reshape will work
        expected_size = B * C * H * W
        actual_size = features.numel()
        if expected_size != actual_size:
            raise ValueError(f"Cannot reshape tensor: expected size {expected_size} (B={B}, C={C}, H={H}, W={W}), got {actual_size}")

        features = features.transpose(1, 2).reshape(B, C, H, W)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINOv3 backbone"""
        self.features.clear()

        # Update actual input dimensions for dynamic reshaping
        B, C, H, W = x.shape
        self.actual_img_size = H  # Assume square input
        self.actual_num_patches = (H // self.patch_size) ** 2

        # Forward through backbone
        if hasattr(self.backbone, 'forward_features'):
            # Standard DINOv3 model
            output = self.backbone.forward_features(x)
        else:
            # HuggingFace model
            output = self.backbone(x)
            if hasattr(output, 'last_hidden_state'):
                output = output.last_hidden_state

        if self.return_levels:
            # Extract and reshape multi-level features
            level_features = []
            for i in range(len(self.feature_layers)):
                if f'layer_{i}' in self.features:
                    feat = self.features[f'layer_{i}']
                    if hasattr(feat, 'hidden_states'):
                        feat = feat.hidden_states
                    feat = self._reshape_features_dynamic(feat)
                    level_features.append(feat)

            # Add final output
            final_feat = self._reshape_features_dynamic(output)
            level_features.append(final_feat)

            return level_features
        else:
            return self._reshape_features_dynamic(output)

    def _reshape_features_dynamic(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape features based on actual input dimensions"""
        # Handle different input types
        if isinstance(features, (list, tuple)):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(features)}")

        # Ensure we have a 3D tensor [B, N, C]
        if features.dim() == 2:
            features = features.unsqueeze(0)
        elif features.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, N, C], got shape {features.shape}")

        B, N, C = features.shape

        # Calculate spatial dimensions based on actual input
        expected_patches = self.actual_num_patches

        # Check for CLS token and handle various scenarios
        has_cls_token = False
        if N == expected_patches + 1:
            # Standard case: CLS token present
            has_cls_token = True
            features = features[:, 1:]
            N = N - 1
        elif N != expected_patches:
            # Non-standard case - check if removing CLS makes it a perfect square
            if int(math.sqrt(N - 1)) ** 2 == (N - 1):
                # CLS token present but expected_patches calculation was wrong
                has_cls_token = True
                features = features[:, 1:]
                N = N - 1
                expected_patches = N
            else:
                # No CLS token, use N as is
                expected_patches = N

        # Calculate spatial dimensions
        H = W = int(math.sqrt(expected_patches))

        # If not a perfect square, try to find the closest dimensions or interpolate
        if H * W != expected_patches:
            # Try to find reasonable dimensions
            print(f"Warning: Non-square patch count {expected_patches}, trying to handle gracefully")

            # Option 1: Find closest square that's slightly smaller and pad/interpolate
            H = W = int(math.sqrt(expected_patches))
            target_patches = H * W

            if target_patches < expected_patches:
                # Take first target_patches patches
                features = features[:, :target_patches, :]
                expected_patches = target_patches
            else:
                raise ValueError(f"Cannot create square spatial dimensions from {expected_patches} patches")

        # Reshape
        features = features.transpose(1, 2).reshape(B, C, H, W)
        return features


class FeatureAdaptationModule(nn.Module):
    """Adapt DINOv3 features to match HerdNet's expected channels and scales"""

    def __init__(
        self,
        dinov3_channels: List[int],
        target_channels: List[int],
        embed_dim: int = 768
    ):
        """
        Args:
            dinov3_channels: Expected output channels from each DINOv3 level
            target_channels: Target channels to match HerdNet DLA structure
            embed_dim: DINOv3 embedding dimension
        """
        super(FeatureAdaptationModule, self).__init__()

        self.dinov3_channels = dinov3_channels
        self.target_channels = target_channels

        # Create adaptation layers for each level
        self.adaptations = nn.ModuleList()

        for i, (in_ch, out_ch) in enumerate(zip([embed_dim] * len(target_channels), target_channels)):
            adaptation = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.adaptations.append(adaptation)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Adapt DINOv3 features to target channels"""
        adapted_features = []

        for i, (feat, adaptation) in enumerate(zip(features, self.adaptations)):
            adapted_feat = adaptation(feat)
            adapted_features.append(adapted_feat)

        return adapted_features


@MODELS.register()
class HerdNetDinoV3(nn.Module):
    """HerdNet with DINOv3 backbone for animal detection in aerial imagery"""

    def __init__(
        self,
        model_name: str = 'dinov3_vitb16',
        num_classes: int = 2,
        pretrained_weights: Optional[str] = None,
        repo_path: Optional[str] = None,
        down_ratio: Optional[int] = 2,
        head_conv: int = 64,
        img_size: int = 224,
        feature_layers: List[int] = [3, 6, 9, 12]
    ):
        """
        Args:
            model_name (str): DINOv3 model variant
            num_classes (int): Number of output classes, background included
            pretrained_weights (str, optional): Path to pretrained DINOv3 weights
            repo_path (str, optional): Path to local DINOv3 repository
            down_ratio (int): Downsample ratio for output
            head_conv (int): Number of channels in head convolutions
            img_size (int): Input image size
            feature_layers (List[int]): Which transformer layers to use
        """
        super(HerdNetDinoV3, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'

        self.model_name = model_name
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.img_size = img_size

        # DINOv3 backbone
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            pretrained_weights=pretrained_weights,
            repo_path=repo_path,
            return_levels=True,
            feature_layers=feature_layers,
            img_size=img_size
        )

        # Target channels similar to DLA structure
        target_channels = [64, 128, 256, 512, 1024][:len(feature_layers)+1]

        # Feature adaptation
        self.feature_adapter = FeatureAdaptationModule(
            dinov3_channels=[self.backbone.embed_dim] * len(target_channels),
            target_channels=target_channels,
            embed_dim=self.backbone.embed_dim
        )

        # Import DLA upsampling (reuse existing implementation)
        from . import dla as dla_modules

        # Setup upsampling similar to HerdNet
        self.first_level = int(math.log2(down_ratio))
        scales = [2 ** i for i in range(len(target_channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(
            target_channels[self.first_level:],
            scales=scales
        )

        # Bottleneck convolution
        self.bottleneck_conv = nn.Conv2d(
            target_channels[-1], target_channels[-1],
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        # Localization head (density map)
        self.loc_head = nn.Sequential(
            nn.Conv2d(target_channels[self.first_level], head_conv,
                     kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, 1,
                kernel_size=1, stride=1,
                padding=0, bias=True
            ),
            nn.Sigmoid()
        )

        self.loc_head[-2].bias.data.fill_(0.00)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(target_channels[-1], head_conv,
                     kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, self.num_classes,
                kernel_size=1, stride=1,
                padding=0, bias=True
            )
        )

        self.cls_head[-1].bias.data.fill_(0.00)

    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Extract features from DINOv3
        features = self.backbone(input)

        # Adapt features to target channels
        adapted_features = self.feature_adapter(features)

        # Apply bottleneck to final features
        bottleneck = self.bottleneck_conv(adapted_features[-1])
        adapted_features[-1] = bottleneck

        # Upsample features
        decode_hm = self.dla_up(adapted_features[self.first_level:])

        # Generate outputs
        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)

        return heatmap, clsmap

    def freeze(self, layers: list) -> None:
        """Freeze specified layers"""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        """Freeze a specific layer"""
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture for new number of classes"""
        self.cls_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes