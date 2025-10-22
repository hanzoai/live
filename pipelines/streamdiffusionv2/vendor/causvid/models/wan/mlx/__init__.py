"""
MLX-native implementation of WAN models for Apple Silicon.

This package provides a complete MLX-native implementation of the WAN
(Weighted Attention Network) diffusion model, optimized for Apple Silicon
with 2-3x performance improvements over PyTorch MPS.
"""

# Core attention operations
from .attention import scaled_dot_product_attention, attention

# Core operations
from .core_ops import (
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
    WanRMSNorm,
    WanLayerNorm,
)

# Convolution operations
from .conv import Conv3d

# Neural network modules
from .modules import (
    WanLinear,
    GELU,
    FeedForward,
    Head,
    MLPProj,
)

# Attention modules
from .wan_attention import (
    WanSelfAttention,
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    WAN_CROSSATTENTION_CLASSES,
)

# Transformer blocks
from .wan_block import WanAttentionBlock

# Complete model
from .model import WanModel

__all__ = [
    # Attention operations
    "scaled_dot_product_attention",
    "attention",
    # Core operations
    "sinusoidal_embedding_1d",
    "rope_params",
    "rope_apply",
    "WanRMSNorm",
    "WanLayerNorm",
    # Convolution
    "Conv3d",
    # Modules
    "WanLinear",
    "GELU",
    "FeedForward",
    "Head",
    "MLPProj",
    # Attention modules
    "WanSelfAttention",
    "WanT2VCrossAttention",
    "WanI2VCrossAttention",
    "WAN_CROSSATTENTION_CLASSES",
    # Transformer block
    "WanAttentionBlock",
    # Full model
    "WanModel",
]
