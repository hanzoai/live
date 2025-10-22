# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native attention modules for WAN models.

Implements self-attention and cross-attention mechanisms optimized for Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict

from .attention import attention
from .core_ops import rope_apply, WanRMSNorm
from .modules import WanLinear


class WanSelfAttention(nn.Module):
    """
    Self-attention module with RoPE (Rotary Position Embeddings).

    Used in WAN transformer blocks for modeling spatial-temporal relationships.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        """
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            window_size: Sliding window size for local attention
            qk_norm: Whether to apply RMS normalization to queries and keys
            eps: Epsilon for normalization layers
        """
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Linear projections for Q, K, V, and output
        self.q = WanLinear(dim, dim)
        self.k = WanLinear(dim, dim)
        self.v = WanLinear(dim, dim)
        self.o = WanLinear(dim, dim)

        # Optional normalization for queries and keys
        if qk_norm:
            self.norm_q = WanRMSNorm(dim, eps=eps)
            self.norm_k = WanRMSNorm(dim, eps=eps)

    def __call__(
        self,
        x: mx.array,
        seq_lens: mx.array,
        grid_sizes: mx.array,
        freqs: mx.array,
    ) -> mx.array:
        """
        Apply self-attention with RoPE.

        Args:
            x: Input tensor [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] containing (F, H, W)
            freqs: RoPE frequency parameters [max_seq_len, head_dim//2, 2]

        Returns:
            Attention output [B, L, C]
        """
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim

        # Compute Q, K, V
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Apply normalization if enabled
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Reshape to [B, L, num_heads, head_dim]
        q = mx.reshape(q, (b, s, n, d))
        k = mx.reshape(k, (b, s, n, d))
        v = mx.reshape(v, (b, s, n, d))

        # Apply RoPE to queries and keys
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # Apply attention
        x = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # Reshape back to [B, L, C]
        x = mx.reshape(x, (b, s, -1))

        # Output projection
        x = self.o(x)

        return x


class WanT2VCrossAttention(WanSelfAttention):
    """
    Text-to-video cross-attention module.

    Inherits from WanSelfAttention but performs cross-attention between
    video latents (query) and text embeddings (key, value).
    """

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: mx.array,
        crossattn_cache: Optional[Dict] = None,
    ) -> mx.array:
        """
        Apply text-to-video cross-attention.

        Args:
            x: Video latents [B, L1, C]
            context: Text embeddings [B, L2, C]
            context_lens: Text sequence lengths [B]
            crossattn_cache: Optional cache for K and V tensors

        Returns:
            Attention output [B, L1, C]
        """
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        # Compute query from video latents
        q = self.q(x)
        if self.qk_norm:
            q = self.norm_q(q)
        q = mx.reshape(q, (b, -1, n, d))

        # Compute or retrieve cached key and value from text context
        if crossattn_cache is not None:
            if not crossattn_cache.get("is_init", False):
                # First time: compute and cache K, V
                k = self.k(context)
                v = self.v(context)
                if self.qk_norm:
                    k = self.norm_k(k)
                k = mx.reshape(k, (b, -1, n, d))
                v = mx.reshape(v, (b, -1, n, d))

                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
                crossattn_cache["is_init"] = True
            else:
                # Use cached K, V
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            # No caching: compute K, V each time
            k = self.k(context)
            v = self.v(context)
            if self.qk_norm:
                k = self.norm_k(k)
            k = mx.reshape(k, (b, -1, n, d))
            v = mx.reshape(v, (b, -1, n, d))

        # Apply attention
        x = attention(q, k, v, k_lens=context_lens)

        # Reshape and project
        x = mx.reshape(x, (b, -1, self.dim))
        x = self.o(x)

        return x


class WanI2VCrossAttention(WanSelfAttention):
    """
    Image-to-video cross-attention module.

    Performs dual cross-attention: one for image embeddings and one for text embeddings,
    then combines both outputs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        # Additional projections for image context
        self.k_img = WanLinear(dim, dim)
        self.v_img = WanLinear(dim, dim)

        if qk_norm:
            self.norm_k_img = WanRMSNorm(dim, eps=eps)

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: mx.array,
    ) -> mx.array:
        """
        Apply image-to-video cross-attention.

        Args:
            x: Video latents [B, L1, C]
            context: Combined context [B, L2, C] where first 257 tokens are image, rest are text
            context_lens: Text sequence lengths [B]

        Returns:
            Attention output [B, L1, C]
        """
        # Split context into image and text
        context_img = context[:, :257, :]  # First 257 tokens = image
        context_text = context[:, 257:, :]  # Remaining tokens = text

        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        # Compute query from video latents
        q = self.q(x)
        if self.qk_norm:
            q = self.norm_q(q)
        q = mx.reshape(q, (b, -1, n, d))

        # Compute key and value for text context
        k_text = self.k(context_text)
        v_text = self.v(context_text)
        if self.qk_norm:
            k_text = self.norm_k(k_text)
        k_text = mx.reshape(k_text, (b, -1, n, d))
        v_text = mx.reshape(v_text, (b, -1, n, d))

        # Compute key and value for image context
        k_img = self.k_img(context_img)
        v_img = self.v_img(context_img)
        if self.qk_norm:
            k_img = self.norm_k_img(k_img)
        k_img = mx.reshape(k_img, (b, -1, n, d))
        v_img = mx.reshape(v_img, (b, -1, n, d))

        # Apply attention for both image and text
        img_x = attention(q, k_img, v_img, k_lens=None)
        text_x = attention(q, k_text, v_text, k_lens=context_lens)

        # Reshape and combine
        img_x = mx.reshape(img_x, (b, -1, self.dim))
        text_x = mx.reshape(text_x, (b, -1, self.dim))

        x = img_x + text_x

        # Output projection
        x = self.o(x)

        return x


# Registry of cross-attention classes
WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


__all__ = [
    "WanSelfAttention",
    "WanT2VCrossAttention",
    "WanI2VCrossAttention",
    "WAN_CROSSATTENTION_CLASSES",
]
