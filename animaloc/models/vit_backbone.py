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
import math
from typing import Optional, Tuple, List

from .register import MODELS
from .pretrained_utils import load_pretrained_vit_weights, get_model_config, list_available_models


class PatchEmbedding(nn.Module):
    """Vision Transformer patch embedding layer"""
    
    def __init__(self, img_size: int = 512, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, num_patches)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x, (H, W)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        
        return x


class MLP(nn.Module):
    """MLP block for transformer"""
    
    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for dense prediction tasks"""
    
    def __init__(
        self, 
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        return_levels: bool = True
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.return_levels = return_levels
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-scale feature channels for compatibility with HerdNet decoder
        self.channels = [embed_dim // 4, embed_dim // 2, embed_dim, embed_dim]
        
        # Feature projection layers for multi-scale outputs
        self.feature_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, ch, kernel_size=1) for ch in self.channels
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding
        x, (H, W) = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Store intermediate features for multi-scale output
        features = []
        num_blocks = len(self.blocks)
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Collect features at different depths for multi-scale processing
            if i in [num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4, num_blocks - 1] or i == num_blocks - 1:
                # Remove class token and reshape to spatial format
                feat = x[:, 1:, :]  # Remove cls token
                feat = feat.transpose(1, 2).reshape(B, self.embed_dim, H, W)
                features.append(feat)
        
        x = self.norm(x)
        
        if not self.return_levels:
            # Return only the final feature map
            final_feat = x[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, H, W)
            return final_feat
        
        # Project features to different channel dimensions for compatibility
        multi_scale_features = []
        for i, feat in enumerate(features[-4:]):  # Take last 4 features
            if i < len(self.feature_projs):
                projected = self.feature_projs[i](feat)
                multi_scale_features.append(projected)
        
        return multi_scale_features


@MODELS.register()
class ViTBackbone(nn.Module):
    """ViT backbone compatible with HerdNet architecture"""
    
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        pretrained: bool = True,
        pretrained_model: Optional[str] = None,
        return_levels: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            return_levels=return_levels
        )
        
        # Channel configuration for compatibility with HerdNet
        self.channels = self.vit.channels
        
        if pretrained:
            self._load_pretrained_weights(pretrained_model)
    
    def _load_pretrained_weights(self, pretrained_model: Optional[str] = None):
        """Load pretrained weights if available"""
        try:
            # Auto-select model based on architecture if not specified
            if pretrained_model is None:
                pretrained_model = self._get_default_pretrained_model()
            
            if pretrained_model is None:
                print("Warning: No suitable pretrained model found. Using random initialization.")
                return
            
            print(f"Loading pretrained weights from {pretrained_model}")
            loaded_count, total_count = load_pretrained_vit_weights(
                self.vit, pretrained_model, strict=False
            )
            
            if loaded_count > 0:
                print(f"Successfully loaded {loaded_count}/{total_count} parameters from pretrained weights")
            else:
                print("Warning: No parameters were loaded from pretrained weights")
                
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            print("Using random initialization instead.")
    
    def _get_default_pretrained_model(self) -> Optional[str]:
        """Get default pretrained model based on architecture parameters"""
        available_models = list_available_models()
        
        # Find best matching model based on embed_dim, depth, num_heads
        for model_name in available_models:
            config = get_model_config(model_name)
            if (config['embed_dim'] == self.embed_dim and 
                config['depth'] == self.depth and 
                config['num_heads'] == self.num_heads and
                config['patch_size'] == self.patch_size):
                return model_name
        
        # Fallback: find closest match by embed_dim
        if self.embed_dim == 384:
            return 'vit_small_patch16_224'
        elif self.embed_dim == 768:
            return 'vit_base_patch16_224'
        elif self.embed_dim == 1024:
            return 'vit_large_patch16_224'
        
        return None
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.vit(x)


# Factory functions for different ViT sizes
def vit_base(pretrained: bool = True, pretrained_model: Optional[str] = None, **kwargs):
    """ViT-Base model"""
    if pretrained_model is None and pretrained:
        pretrained_model = 'vit_base_patch16_224'
    
    return ViTBackbone(
        embed_dim=768,
        depth=12,
        num_heads=12,
        pretrained=pretrained,
        pretrained_model=pretrained_model,
        **kwargs
    )


def vit_large(pretrained: bool = True, pretrained_model: Optional[str] = None, **kwargs):
    """ViT-Large model"""
    if pretrained_model is None and pretrained:
        pretrained_model = 'vit_large_patch16_224'
    
    return ViTBackbone(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        pretrained=pretrained,
        pretrained_model=pretrained_model,
        **kwargs
    )


def vit_small(pretrained: bool = True, pretrained_model: Optional[str] = None, **kwargs):
    """ViT-Small model"""
    if pretrained_model is None and pretrained:
        pretrained_model = 'vit_small_patch16_224'
    
    return ViTBackbone(
        embed_dim=384,
        depth=12,
        num_heads=6,
        pretrained=pretrained,
        pretrained_model=pretrained_model,
        **kwargs
    )