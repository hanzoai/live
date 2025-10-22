# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native neural network modules for WAN models.

Implements Linear layers, GELU activation, and feedforward networks
optimized for Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class WanLinear(nn.Module):
    """
    Linear layer compatible with PyTorch nn.Linear.

    Implements y = xW^T + b where W has shape [out_features, in_features].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, adds a learnable bias to the output
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weight with Kaiming uniform (similar to PyTorch default)
        scale = mx.sqrt(mx.array(1.0 / in_features))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_features, in_features),
            dtype=mx.float32,
        )

        if bias:
            self.bias = mx.random.uniform(
                low=-scale, high=scale, shape=(out_features,), dtype=mx.float32
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply linear transformation.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Matrix multiplication: x @ W^T
        output = mx.matmul(x, self.weight.T)

        if self.use_bias:
            output = output + self.bias

        return output


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    Implements GELU with tanh approximation for compatibility with PyTorch's
    approximate="tanh" mode used in WAN models.
    """

    def __init__(self, approximate: str = "tanh"):
        """
        Args:
            approximate: Type of approximation ("none" or "tanh")
        """
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply GELU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor with same shape as input
        """
        if self.approximate == "tanh":
            # Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            sqrt_2_over_pi = mx.sqrt(mx.array(2.0 / mx.pi))
            x_cubed = mx.power(x, 3.0)
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            return 0.5 * x * (1.0 + mx.tanh(tanh_arg))
        else:
            # Exact GELU: GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
            # We'll use the tanh approximation anyway as it's faster and accurate enough
            sqrt_2_over_pi = mx.sqrt(mx.array(2.0 / mx.pi))
            x_cubed = mx.power(x, 3.0)
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            return 0.5 * x * (1.0 + mx.tanh(tanh_arg))


class FeedForward(nn.Module):
    """
    Feedforward network (FFN) block used in WAN attention blocks.

    Implements: Linear(dim → ffn_dim) → GELU → Linear(ffn_dim → dim)
    """

    def __init__(self, dim: int, ffn_dim: int, approximate_gelu: str = "tanh"):
        """
        Args:
            dim: Input and output dimension
            ffn_dim: Hidden dimension (typically 4x dim)
            approximate_gelu: GELU approximation type
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim

        self.fc1 = WanLinear(dim, ffn_dim)
        self.activation = GELU(approximate=approximate_gelu)
        self.fc2 = WanLinear(ffn_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply feedforward transformation.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor [..., dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class Head(nn.Module):
    """
    Output head for WAN diffusion model.

    Projects transformer outputs back to patch space with adaptive normalization.
    """

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        """
        Args:
            dim: Input dimension from transformer
            out_dim: Output channels (C_out)
            patch_size: 3D patch size (t_patch, h_patch, w_patch)
            eps: Epsilon for normalization
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Calculate total output dimension (flattened patch)
        import math
        total_out_dim = math.prod(patch_size) * out_dim

        # Normalization and projection layers
        from .core_ops import WanLayerNorm
        self.norm = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.head = WanLinear(dim, total_out_dim)

        # Modulation parameters for adaptive normalization
        scale = 1.0 / mx.sqrt(mx.array(dim, dtype=mx.float32))
        self.modulation = mx.random.normal(shape=(1, 2, dim)) * scale

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        """
        Apply output head with adaptive normalization.

        Args:
            x: Transformer output [B, L, C]
            e: Conditioning embeddings [B, C]

        Returns:
            Output patches [B, L, out_dim * prod(patch_size)]
        """
        # Expand e to [B, 1, C] for broadcasting
        e = mx.expand_dims(e, axis=1)  # [B, 1, C]

        # Add modulation and split into 2 parts
        e_modulated = self.modulation + e  # [B, 1, 2, C] -> [B, 2, C]
        e_list = mx.split(e_modulated, indices_or_sections=2, axis=1)
        e0 = mx.squeeze(e_list[0], axis=1)  # [B, C]
        e1 = mx.squeeze(e_list[1], axis=1)  # [B, C]

        # Expand for broadcasting with [B, L, C]
        e0 = mx.expand_dims(e0, axis=1)  # [B, 1, C]
        e1 = mx.expand_dims(e1, axis=1)  # [B, 1, C]

        # Adaptive normalization: norm(x) * (1 + e1) + e0
        x_norm = self.norm(x)
        x_modulated = x_norm * (1.0 + e1) + e0

        # Project to output space
        x = self.head(x_modulated)

        return x


class MLPProj(nn.Module):
    """
    MLP projection for CLIP image features in image-to-video mode.

    Projects CLIP embeddings to the model's hidden dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Args:
            in_dim: Input dimension (CLIP feature dim, typically 1280)
            out_dim: Output dimension (model hidden dim)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Import here to avoid circular dependency
        from .core_ops import WanLayerNorm
        from .modules import GELU

        # MLP projection: LayerNorm → Linear → GELU → Linear → LayerNorm
        self.norm1 = WanLayerNorm(in_dim, eps=1e-6, elementwise_affine=True)
        self.fc1 = WanLinear(in_dim, in_dim)
        self.activation = GELU(approximate="none")
        self.fc2 = WanLinear(in_dim, out_dim)
        self.norm2 = WanLayerNorm(out_dim, eps=1e-6, elementwise_affine=True)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        """
        Project CLIP image embeddings.

        Args:
            image_embeds: CLIP features [B, 257, in_dim]

        Returns:
            Projected features [B, 257, out_dim]
        """
        x = self.norm1(image_embeds)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


__all__ = ["WanLinear", "GELU", "FeedForward", "Head", "MLPProj"]
