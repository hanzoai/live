# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native core operations for WAN models.

Implements fundamental operations like RoPE (Rotary Position Embeddings),
sinusoidal embeddings, and normalization layers optimized for Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """
    Generate 1D sinusoidal position embeddings.

    Args:
        dim: Embedding dimension (must be even)
        position: Position indices [N]

    Returns:
        Sinusoidal embeddings [N, dim]
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    half = dim // 2

    # Convert to float32 for precision
    position = position.astype(mx.float32)

    # Compute frequency bands: 10000^(-2i/dim) for i in [0, half)
    freq_bands = mx.power(10000.0, -mx.arange(half, dtype=mx.float32) / half)

    # Outer product: position × freq_bands
    # [N, 1] × [1, half] -> [N, half]
    sinusoid = mx.expand_dims(position, axis=1) * mx.expand_dims(freq_bands, axis=0)

    # Concatenate cos and sin
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)

    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> mx.array:
    """
    Compute RoPE (Rotary Position Embedding) frequency parameters.

    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension (must be even)
        theta: Base frequency (default 10000)

    Returns:
        Complex frequency tensor [max_seq_len, dim//2]
    """
    assert dim % 2 == 0, "Embedding dimension must be even"

    # Compute frequency bands: 1 / (theta^(2i/dim)) for i in [0, dim//2)
    freq_bands = 1.0 / mx.power(
        theta, mx.arange(0, dim, 2, dtype=mx.float32) / dim
    )

    # Outer product: positions × freq_bands
    # [max_seq_len, 1] × [1, dim//2] -> [max_seq_len, dim//2]
    positions = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.expand_dims(positions, axis=1) * mx.expand_dims(freq_bands, axis=0)

    # Convert to complex exponentials: e^(i * freqs)
    # In MLX, we'll represent as cos + i*sin = [cos, sin] pairs
    freqs_cos = mx.cos(freqs)
    freqs_sin = mx.sin(freqs)

    # Stack to create complex representation [max_seq_len, dim//2, 2]
    freqs_complex = mx.stack([freqs_cos, freqs_sin], axis=-1)

    return freqs_complex


def rope_apply(
    x: mx.array,
    grid_sizes: mx.array,
    freqs: mx.array
) -> mx.array:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor [batch, max_seq_len, num_heads, head_dim]
        grid_sizes: Grid sizes [batch, 3] containing (frames, height, width)
        freqs: Frequency parameters from rope_params [max_pos, head_dim//2, 2]

    Returns:
        Tensor with RoPE applied [batch, max_seq_len, num_heads, head_dim]
    """
    batch_size, max_seq_len, num_heads, head_dim = x.shape
    c = head_dim // 2  # Complex dimension

    # Split frequencies for 3D structure (frames, height, width)
    # Allocate frequencies: temporal (f), height (h), width (w)
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    freq_splits = []
    start_idx = 0
    for size in split_sizes:
        freq_splits.append(freqs[:, start_idx:start_idx+size, :])
        start_idx += size

    output = []

    # Process each sample in batch
    for i in range(batch_size):
        f, h, w = int(grid_sizes[i, 0]), int(grid_sizes[i, 1]), int(grid_sizes[i, 2])
        seq_len = f * h * w

        # Get valid sequence
        x_i = x[i, :seq_len]  # [seq_len, num_heads, head_dim]

        # Reshape to complex pairs: [seq_len, num_heads, head_dim//2, 2]
        x_i_complex = mx.reshape(x_i, (seq_len, num_heads, c, 2))

        # Build 3D frequency grid
        # freq_f: [f, split_sizes[0], 2]
        # freq_h: [h, split_sizes[1], 2]
        # freq_w: [w, split_sizes[2], 2]

        # Expand and concatenate frequencies for 3D grid
        freq_f = freq_splits[0][:f]  # [f, dim_f, 2]
        freq_h = freq_splits[1][:h]  # [h, dim_h, 2]
        freq_w = freq_splits[2][:w]  # [w, dim_w, 2]

        # Broadcast to full 3D grid and concatenate
        # This is a simplified version - full implementation would properly tile
        # For now, we'll use a simpler approach that matches the semantics

        # Reshape frequencies to match the grid structure
        freq_f_expanded = mx.reshape(freq_f, (f, 1, 1, -1, 2)).broadcast_to((f, h, w, split_sizes[0], 2))
        freq_h_expanded = mx.reshape(freq_h, (1, h, 1, -1, 2)).broadcast_to((f, h, w, split_sizes[1], 2))
        freq_w_expanded = mx.reshape(freq_w, (1, 1, w, -1, 2)).broadcast_to((f, h, w, split_sizes[2], 2))

        # Concatenate along frequency dimension
        freqs_i = mx.concatenate([freq_f_expanded, freq_h_expanded, freq_w_expanded], axis=3)
        freqs_i = mx.reshape(freqs_i, (seq_len, 1, c, 2))  # [seq_len, 1, c, 2]

        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        # x_i_complex: [seq_len, num_heads, c, 2] where [..., 0] is real, [..., 1] is imag
        # freqs_i: [seq_len, 1, c, 2]
        x_real = x_i_complex[..., 0]  # [seq_len, num_heads, c]
        x_imag = x_i_complex[..., 1]
        freq_real = freqs_i[..., 0]  # [seq_len, 1, c]
        freq_imag = freqs_i[..., 1]

        # Rotate: multiply by e^(i*theta)
        rotated_real = x_real * freq_real - x_imag * freq_imag
        rotated_imag = x_real * freq_imag + x_imag * freq_real

        # Stack back to complex format and flatten
        rotated = mx.stack([rotated_real, rotated_imag], axis=-1)
        rotated = mx.reshape(rotated, (seq_len, num_heads, head_dim))

        # Concatenate with unmodified padding tokens
        if seq_len < max_seq_len:
            padding = x[i, seq_len:]
            x_i_rotated = mx.concatenate([rotated, padding], axis=0)
        else:
            x_i_rotated = rotated

        output.append(x_i_rotated)

    return mx.stack(output)


class WanRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More stable and efficient alternative to LayerNorm for transformer models.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: Normalized dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # Compute RMS: sqrt(mean(x^2))
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x_normalized = x * mx.rsqrt(variance + self.eps)

        # Apply learned scale
        return self.weight * x_normalized


class WanLayerNorm(nn.Module):
    """
    Layer Normalization compatible with PyTorch LayerNorm.

    Standard layer normalization that normalizes across the feature dimension,
    computing mean and variance statistics.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        """
        Args:
            dim: Normalized dimension
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn affine parameters (weight and bias)
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply layer normalization.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # Compute mean and variance across the last dimension
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.var(x, axis=-1, keepdims=True)

        # Normalize: (x - mean) / sqrt(variance + eps)
        x_normalized = (x - mean) * mx.rsqrt(variance + self.eps)

        # Apply learned affine transformation if enabled
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


__all__ = [
    "sinusoidal_embedding_1d",
    "rope_params",
    "rope_apply",
    "WanRMSNorm",
    "WanLayerNorm",
]
