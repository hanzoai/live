# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native WAN attention block implementation.

Combines self-attention, cross-attention, and feedforward networks
with adaptive layer normalization (modulation).
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional

from .core_ops import WanLayerNorm
from .modules import FeedForward
from .wan_attention import (
    WanSelfAttention,
    WAN_CROSSATTENTION_CLASSES,
)


class WanAttentionBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and FFN.

    Uses adaptive layer normalization (modulation) to condition on timestep/class embeddings.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            cross_attn_type: Type of cross-attention ("t2v_cross_attn" or "i2v_cross_attn")
            dim: Model dimension
            ffn_dim: Feedforward network hidden dimension (typically 4x dim)
            num_heads: Number of attention heads
            window_size: Sliding window size for local attention
            qk_norm: Whether to apply RMS normalization to queries and keys
            cross_attn_norm: Whether to apply normalization before cross-attention
            eps: Epsilon for normalization layers
        """
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Normalization layers
        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = WanLayerNorm(dim, eps, elementwise_affine=False)

        # Optional normalization before cross-attention
        if cross_attn_norm:
            self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)

        # Self-attention module
        self.self_attn = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
        )

        # Cross-attention module (text-to-video or image-to-video)
        cross_attn_class = WAN_CROSSATTENTION_CLASSES[cross_attn_type]
        self.cross_attn = cross_attn_class(
            dim=dim,
            num_heads=num_heads,
            window_size=(-1, -1),  # No windowing for cross-attention
            qk_norm=qk_norm,
            eps=eps,
        )

        # Feedforward network
        self.ffn = FeedForward(dim=dim, ffn_dim=ffn_dim, approximate_gelu="tanh")

        # Modulation parameters for adaptive normalization
        # Shape: [1, 6, dim] - 6 modulation vectors for different residual connections
        scale = 1.0 / mx.sqrt(mx.array(dim, dtype=mx.float32))
        self.modulation = mx.random.normal(shape=(1, 6, dim)) * scale

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: mx.array,
        grid_sizes: mx.array,
        freqs: mx.array,
        context: mx.array,
        context_lens: mx.array,
        crossattn_cache: Optional[dict] = None,
    ) -> mx.array:
        """
        Apply attention block with adaptive normalization.

        Args:
            x: Input tensor [B, L, C]
            e: Conditioning embeddings [B, 6, C] (timestep/class embeddings)
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] containing (F, H, W)
            freqs: RoPE frequency parameters [max_seq_len, head_dim//2, 2]
            context: Context embeddings for cross-attention [B, L2, C]
            context_lens: Context sequence lengths [B]
            crossattn_cache: Optional cache for cross-attention K, V

        Returns:
            Output tensor [B, L, C]
        """
        # Compute modulated conditioning: e = modulation + e
        # Then split into 6 modulation vectors
        e_modulated = self.modulation + e  # [B, 6, C]

        # Split along dim=1 to get 6 separate modulation vectors
        e_list = mx.split(e_modulated, indices_or_sections=6, axis=1)
        e0 = mx.squeeze(e_list[0], axis=1)  # [B, C]
        e1 = mx.squeeze(e_list[1], axis=1)  # [B, C]
        e2 = mx.squeeze(e_list[2], axis=1)  # [B, C]
        e3 = mx.squeeze(e_list[3], axis=1)  # [B, C]
        e4 = mx.squeeze(e_list[4], axis=1)  # [B, C]
        e5 = mx.squeeze(e_list[5], axis=1)  # [B, C]

        # Expand for broadcasting with [B, L, C]
        e0 = mx.expand_dims(e0, axis=1)  # [B, 1, C]
        e1 = mx.expand_dims(e1, axis=1)  # [B, 1, C]
        e2 = mx.expand_dims(e2, axis=1)  # [B, 1, C]
        e3 = mx.expand_dims(e3, axis=1)  # [B, 1, C]
        e4 = mx.expand_dims(e4, axis=1)  # [B, 1, C]
        e5 = mx.expand_dims(e5, axis=1)  # [B, 1, C]

        # Self-attention with adaptive normalization
        # Modulate: norm(x) * (1 + e1) + e0
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1.0 + e1) + e0

        # Apply self-attention
        y = self.self_attn(
            x=x_modulated,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )

        # Residual connection with modulation: x = x + y * e2
        x = x + y * e2

        # Cross-attention
        if self.cross_attn_norm:
            x_cross = self.norm3(x)
        else:
            x_cross = x

        # Apply cross-attention based on type
        if crossattn_cache is not None and hasattr(self.cross_attn, "__class__") and \
           self.cross_attn.__class__.__name__ == "WanT2VCrossAttention":
            # Text-to-video cross-attention supports caching
            y_cross = self.cross_attn(
                x=x_cross,
                context=context,
                context_lens=context_lens,
                crossattn_cache=crossattn_cache,
            )
        else:
            # Image-to-video cross-attention or no caching
            y_cross = self.cross_attn(
                x=x_cross,
                context=context,
                context_lens=context_lens,
            )

        # Add cross-attention output directly (no modulation)
        x = x + y_cross

        # Feedforward network with adaptive normalization
        # Modulate: norm(x) * (1 + e4) + e3
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1.0 + e4) + e3

        # Apply FFN
        y_ffn = self.ffn(x_modulated)

        # Residual connection with modulation: x = x + y * e5
        x = x + y_ffn * e5

        return x


__all__ = ["WanAttentionBlock"]
