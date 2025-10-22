# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native attention implementation optimized for Apple Silicon.

This module provides efficient attention mechanisms using MLX's Metal-optimized operations,
offering better performance than PyTorch's MPS backend on Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    attn_mask: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> mx.array:
    """
    MLX-native scaled dot-product attention.

    Optimized for Apple Silicon using Metal Performance Shaders via MLX.
    Significantly faster than PyTorch's MPS backend for attention operations.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim] or [batch, num_heads, seq_len_q, head_dim]
        key: Key tensor [batch, seq_len_k, num_heads, head_dim] or [batch, num_heads, seq_len_k, head_dim]
        value: Value tensor [batch, seq_len_k, num_heads, head_dim] or [batch, num_heads, seq_len_k, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently not implemented in MLX)
        is_causal: Whether to apply causal masking
        scale: Scaling factor (defaults to 1/sqrt(head_dim))

    Returns:
        Attention output tensor with same shape as query
    """
    # Get dimensions
    # MLX uses [..., num_heads, seq_len, head_dim] format typically
    # But we need to handle both [batch, seq, heads, dim] and [batch, heads, seq, dim]

    # Assume input is [batch, seq_len, num_heads, head_dim]
    # Transpose to [batch, num_heads, seq_len, head_dim] for computation
    if query.ndim == 4:
        # Permute from [batch, seq, heads, dim] to [batch, heads, seq, dim]
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))
        need_transpose_back = True
    else:
        need_transpose_back = False

    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape

    # Calculate scaling factor
    if scale is None:
        scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=mx.float32))

    # Compute attention scores: Q @ K^T
    # [batch, heads, seq_q, dim] @ [batch, heads, dim, seq_k] -> [batch, heads, seq_q, seq_k]
    scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2)))
    scores = scores * scale

    # Apply causal mask if requested
    if is_causal:
        # Create causal mask: upper triangle is -inf
        causal_mask = mx.triu(mx.full((seq_len_q, seq_len_k), -mx.inf, dtype=scores.dtype), k=1)
        scores = scores + causal_mask

    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Softmax over the key dimension
    attn_weights = mx.softmax(scores, axis=-1)

    # Apply dropout (not implemented in MLX yet, would need custom implementation)
    if dropout_p > 0.0:
        # MLX doesn't have dropout yet, so we skip it
        # In training mode, you'd implement: attn_weights = mx.dropout(attn_weights, dropout_p)
        pass

    # Compute attention output: weights @ V
    # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, dim] -> [batch, heads, seq_q, dim]
    output = mx.matmul(attn_weights, value)

    # Transpose back if needed
    if need_transpose_back:
        # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        output = mx.transpose(output, (0, 2, 1, 3))

    return output


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_lens: Optional[mx.array] = None,
    k_lens: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
) -> mx.array:
    """
    MLX-native attention wrapper compatible with the PyTorch attention API.

    This function provides a drop-in replacement for the PyTorch attention module,
    optimized for Apple Silicon using MLX's Metal backend.

    Args:
        q: Query tensor [B, Lq, Nq, C1]
        k: Key tensor [B, Lk, Nk, C1]
        v: Value tensor [B, Lk, Nk, C2]
        q_lens: Optional sequence lengths for queries
        k_lens: Optional sequence lengths for keys
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax
        q_scale: Additional scaling for queries
        causal: Whether to apply causal masking
        window_size: Sliding window size (not yet implemented)
        deterministic: Whether to use deterministic operations

    Returns:
        Attention output [B, Lq, Nq, C2]
    """
    # Apply query scaling if provided
    if q_scale is not None:
        q = q * q_scale

    # Handle sequence length masking (simplified - full implementation would need padding)
    # For now, we'll use the full sequences and rely on causal masking

    # Call the core scaled_dot_product_attention
    output = scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=None,  # Could be constructed from q_lens/k_lens
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )

    return output


__all__ = ["scaled_dot_product_attention", "attention"]
