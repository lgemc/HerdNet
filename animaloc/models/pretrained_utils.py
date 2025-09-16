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


import os
import hashlib
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import json


# Pretrained model configurations
PRETRAINED_MODELS = {
    'vit_small_patch16_224': {
        'url': 'https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'patch_size': 16,
        'img_size': 224,
        'num_classes': 21843,  # ImageNet-21k
    },
    'vit_base_patch16_224': {
        'url': 'https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'patch_size': 16,
        'img_size': 224,
        'num_classes': 21843,  # ImageNet-21k
    },
    'vit_large_patch16_224': {
        'url': 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'patch_size': 16,
        'img_size': 224,
        'num_classes': 21843,  # ImageNet-21k
    },
    # ImageNet-1k variants (fine-tuned)
    'vit_base_patch16_224_in1k': {
        'url': 'https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'patch_size': 16,
        'img_size': 224,
        'num_classes': 1000,  # ImageNet-1k
    }
}


def get_cache_dir() -> Path:
    """Get the cache directory for pretrained models"""
    cache_dir = Path.home() / '.cache' / 'animaloc' / 'pretrained'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def download_file(url: str, filepath: Path, progress_callback=None) -> None:
    """Download a file from URL with optional progress callback"""
    def report_progress(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            downloaded = block_num * block_size
            progress = min(downloaded / total_size, 1.0)
            progress_callback(progress)
    
    print(f"Downloading {url} to {filepath}")
    urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
    print(f"Download completed: {filepath}")


def download_pretrained_weights(model_name: str, force_download: bool = False) -> Path:
    """
    Download pretrained weights for a given model
    
    Args:
        model_name: Name of the pretrained model
        force_download: Force re-download even if file exists
        
    Returns:
        Path to the downloaded weights file
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(PRETRAINED_MODELS.keys())}")
    
    config = PRETRAINED_MODELS[model_name]
    cache_dir = get_cache_dir()
    
    # Create filename from URL
    filename = Path(config['url']).name
    filepath = cache_dir / filename
    
    # Download if file doesn't exist or force_download is True
    if not filepath.exists() or force_download:
        download_file(config['url'], filepath)
    else:
        print(f"Using cached weights: {filepath}")
    
    return filepath


def load_npz_weights(filepath: Path) -> Dict[str, np.ndarray]:
    """Load weights from NPZ file"""
    with np.load(filepath) as data:
        return dict(data)


def convert_jax_to_pytorch_key(jax_key: str) -> str:
    """Convert JAX parameter names to PyTorch parameter names"""
    # Common conversions for ViT weights
    key_mapping = {
        'embedding/kernel': 'patch_embed.projection.weight',
        'embedding/bias': 'patch_embed.projection.bias',
        'cls': 'cls_token',
        'Transformer/posembed_input/pos_embedding': 'pos_embed',
        'head/kernel': 'head.weight',
        'head/bias': 'head.bias',
    }
    
    # Handle transformer blocks
    if 'encoderblock_' in jax_key:
        # Extract block number
        parts = jax_key.split('/')
        block_idx = None
        for part in parts:
            if part.startswith('encoderblock_'):
                block_idx = part.split('_')[1]
                break
        
        if block_idx is not None:
            # Map different components
            if 'LayerNorm_0' in jax_key:
                if 'scale' in jax_key:
                    return f'blocks.{block_idx}.norm1.weight'
                elif 'bias' in jax_key:
                    return f'blocks.{block_idx}.norm1.bias'
            elif 'LayerNorm_2' in jax_key:
                if 'scale' in jax_key:
                    return f'blocks.{block_idx}.norm2.weight'
                elif 'bias' in jax_key:
                    return f'blocks.{block_idx}.norm2.bias'
            elif 'MultiHeadDotProductAttention_1' in jax_key:
                if 'query/kernel' in jax_key:
                    return f'blocks.{block_idx}.attention.qkv.weight'  # Will need special handling
                elif 'key/kernel' in jax_key:
                    return f'blocks.{block_idx}.attention.qkv.weight'  # Will need special handling
                elif 'value/kernel' in jax_key:
                    return f'blocks.{block_idx}.attention.qkv.weight'  # Will need special handling
                elif 'out/kernel' in jax_key:
                    return f'blocks.{block_idx}.attention.projection.weight'
                elif 'out/bias' in jax_key:
                    return f'blocks.{block_idx}.attention.projection.bias'
            elif 'MlpBlock_3' in jax_key:
                if 'Dense_0/kernel' in jax_key:
                    return f'blocks.{block_idx}.mlp.fc1.weight'
                elif 'Dense_0/bias' in jax_key:
                    return f'blocks.{block_idx}.mlp.fc1.bias'
                elif 'Dense_1/kernel' in jax_key:
                    return f'blocks.{block_idx}.mlp.fc2.weight'
                elif 'Dense_1/bias' in jax_key:
                    return f'blocks.{block_idx}.mlp.fc2.bias'
    
    # Check direct mappings
    for jax_pattern, pytorch_pattern in key_mapping.items():
        if jax_pattern in jax_key:
            return jax_key.replace(jax_pattern, pytorch_pattern)
    
    # Default: return as-is (might need manual handling)
    return jax_key


def convert_jax_weights_to_pytorch(jax_weights: Dict[str, np.ndarray], model_config: Dict) -> Dict[str, torch.Tensor]:
    """
    Convert JAX/Flax weights to PyTorch format

    Args:
        jax_weights: Dictionary of JAX weights
        model_config: Model configuration

    Returns:
        Dictionary of PyTorch weights
    """
    pytorch_weights = {}
    embed_dim = model_config['embed_dim']
    num_heads = model_config['num_heads']
    head_dim = embed_dim // num_heads

    # Handle special cases that need concatenation or reshaping
    qkv_weights = {}  # Collect Q, K, V weights to concatenate

    for jax_key, jax_weight in jax_weights.items():
        pytorch_key = convert_jax_to_pytorch_key(jax_key)

        # Convert numpy to torch tensor
        weight = torch.from_numpy(jax_weight)

        # Handle different weight types
        if 'embedding/kernel' in jax_key:
            # Conv2d weight: need to permute dimensions
            weight = weight.permute(3, 2, 0, 1)  # (H, W, in_ch, out_ch) -> (out_ch, in_ch, H, W)
        elif 'kernel' in jax_key and weight.dim() == 2:
            # Linear layer weight: transpose
            weight = weight.transpose(0, 1)
        elif jax_key == 'cls':
            # Class token: correct shape for PyTorch ViT (1, 1, embed_dim)
            weight = weight.unsqueeze(0)  # Add batch dimension only
        elif 'pos_embedding' in jax_key:
            # Position embedding: already correct shape
            pass
        elif 'query/kernel' in jax_key or 'key/kernel' in jax_key or 'value/kernel' in jax_key:
            # Collect QKV weights for concatenation
            block_num = None
            for part in jax_key.split('/'):
                if part.startswith('encoderblock_'):
                    block_num = part.split('_')[1]
                    break

            if block_num is not None:
                if block_num not in qkv_weights:
                    qkv_weights[block_num] = {}

                # JAX weights come as (embed_dim, num_heads, head_dim)
                # We need to reshape and transpose them properly
                if 'query/kernel' in jax_key:
                    # Reshape from (embed_dim, num_heads, head_dim) to (embed_dim, embed_dim)
                    qkv_weights[block_num]['q'] = weight.reshape(embed_dim, embed_dim).transpose(0, 1)
                elif 'key/kernel' in jax_key:
                    qkv_weights[block_num]['k'] = weight.reshape(embed_dim, embed_dim).transpose(0, 1)
                elif 'value/kernel' in jax_key:
                    qkv_weights[block_num]['v'] = weight.reshape(embed_dim, embed_dim).transpose(0, 1)
            continue  # Skip adding individual Q, K, V weights
        elif 'out/kernel' in jax_key:
            # Attention projection weights: reshape from (num_heads, head_dim, embed_dim) to (embed_dim, embed_dim)
            if weight.dim() == 3 and weight.shape[0] == num_heads:
                weight = weight.reshape(embed_dim, embed_dim).transpose(0, 1)
            else:
                weight = weight.transpose(0, 1)

        pytorch_weights[pytorch_key] = weight

    # Concatenate Q, K, V weights
    for block_num, qkv_dict in qkv_weights.items():
        if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
            concatenated_qkv = torch.cat([qkv_dict['q'], qkv_dict['k'], qkv_dict['v']], dim=0)
            pytorch_weights[f'blocks.{block_num}.attention.qkv.weight'] = concatenated_qkv

    return pytorch_weights


def resize_pos_embedding(pos_embed: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Resize positional embedding to match target size

    Args:
        pos_embed: Original positional embedding (1, num_patches + 1, embed_dim)
        target_size: Target number of patches (without class token)

    Returns:
        Resized positional embedding
    """
    if pos_embed.size(1) - 1 == target_size:
        return pos_embed

    # Separate class token and position embeddings
    cls_token_embed = pos_embed[:, 0:1, :]  # (1, 1, embed_dim)
    pos_embed_patches = pos_embed[:, 1:, :]  # (1, num_patches, embed_dim)

    # Get original grid size
    num_patches = pos_embed_patches.size(1)
    grid_size = int(num_patches ** 0.5)

    # Get target grid size
    target_grid_size = int(target_size ** 0.5)

    if grid_size * grid_size != num_patches:
        raise ValueError(f"Position embedding size {num_patches} is not a perfect square")

    if target_grid_size * target_grid_size != target_size:
        raise ValueError(f"Target size {target_size} is not a perfect square")

    # Reshape to 2D grid
    embed_dim = pos_embed_patches.size(-1)
    pos_embed_grid = pos_embed_patches.reshape(1, grid_size, grid_size, embed_dim)
    pos_embed_grid = pos_embed_grid.permute(0, 3, 1, 2)  # (1, embed_dim, H, W)

    # Interpolate to target size
    pos_embed_resized = F.interpolate(
        pos_embed_grid,
        size=(target_grid_size, target_grid_size),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back to sequence format
    pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1)  # (1, H, W, embed_dim)
    pos_embed_resized = pos_embed_resized.reshape(1, target_size, embed_dim)

    # Concatenate class token back
    return torch.cat([cls_token_embed, pos_embed_resized], dim=1)


def load_pretrained_vit_weights(model: nn.Module, model_name: str, strict: bool = False) -> Tuple[int, int]:
    """
    Load pretrained weights into a ViT model

    Args:
        model: PyTorch ViT model
        model_name: Name of pretrained model
        strict: Whether to strictly match all parameters

    Returns:
        Tuple of (loaded_params, total_params) counts
    """
    # Download weights
    weights_path = download_pretrained_weights(model_name)

    # Load JAX weights
    jax_weights = load_npz_weights(weights_path)

    # Get model config
    model_config = PRETRAINED_MODELS[model_name]

    # Convert to PyTorch format
    pytorch_weights = convert_jax_weights_to_pytorch(jax_weights, model_config)

    # Filter weights to match model architecture
    model_state = model.state_dict()
    filtered_weights = {}

    for key, weight in pytorch_weights.items():
        if key in model_state:
            if key == 'pos_embed':
                # Handle position embedding resizing
                target_num_patches = model_state[key].size(1) - 1  # Subtract 1 for class token
                if weight.size(1) - 1 != target_num_patches:
                    print(f"Resizing position embedding from {weight.size(1) - 1} to {target_num_patches} patches")
                    weight = resize_pos_embedding(weight, target_num_patches)

            if weight.shape == model_state[key].shape:
                filtered_weights[key] = weight
            else:
                print(f"Shape mismatch for {key}: pretrained {weight.shape} vs model {model_state[key].shape}")
        else:
            print(f"Key not found in model: {key}")

    # Load weights
    loaded_keys = model.load_state_dict(filtered_weights, strict=strict)

    if not strict:
        missing_keys = loaded_keys.missing_keys
        unexpected_keys = loaded_keys.unexpected_keys

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    loaded_count = len(filtered_weights)
    total_count = len(model_state)

    print(f"Loaded {loaded_count}/{total_count} parameters from pretrained weights")

    return loaded_count, total_count


def get_model_config(model_name: str) -> Dict:
    """Get configuration for a pretrained model"""
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return PRETRAINED_MODELS[model_name].copy()


def list_available_models() -> list:
    """List all available pretrained models"""
    return list(PRETRAINED_MODELS.keys())