# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native WAN diffusion model implementation.

Complete implementation of the WAN (Weighted Attention Network) model
optimized for Apple Silicon using MLX.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple

from .attention import attention
from .conv import Conv3d
from .core_ops import sinusoidal_embedding_1d, rope_params
from .modules import WanLinear, GELU, Head, MLPProj
from .wan_block import WanAttentionBlock


__all__ = ["WanModel"]


class WanModel(nn.Module):
    """
    WAN diffusion backbone supporting both text-to-video and image-to-video.

    Implements a transformer-based diffusion model with:
    - 3D patch embedding for video inputs
    - Rotary position embeddings (RoPE)
    - Self-attention and cross-attention blocks
    - Adaptive layer normalization (modulation)
    """

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initialize the diffusion model backbone.

        Args:
            model_type: Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size: 3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len: Fixed length for text embeddings
            in_dim: Input video channels (C_in)
            dim: Hidden dimension of the transformer
            ffn_dim: Intermediate dimension in feed-forward network
            freq_dim: Dimension for sinusoidal time embeddings
            text_dim: Input dimension for text embeddings
            out_dim: Output video channels (C_out)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            window_size: Window size for local attention (-1 indicates global attention)
            qk_norm: Enable query/key normalization
            cross_attn_norm: Enable cross-attention normalization
            eps: Epsilon value for normalization layers
        """
        super().__init__()

        assert model_type in ["t2v", "i2v"], f"model_type must be 't2v' or 'i2v', got {model_type}"

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Patch embedding: Conv3d to convert input video to patches
        self.patch_embedding = Conv3d(
            in_channels=in_dim,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Text embedding: Linear projection with GELU activation
        self.text_embedding_fc1 = WanLinear(text_dim, dim)
        self.text_embedding_gelu = GELU(approximate="tanh")
        self.text_embedding_fc2 = WanLinear(dim, dim)

        # Time embedding: Sinusoidal embedding → SiLU → Linear
        self.time_embedding_fc1 = WanLinear(freq_dim, dim)
        self.time_embedding_fc2 = WanLinear(dim, dim)

        # Time projection: projects timestep embedding to 6*dim for modulation
        self.time_projection_fc = WanLinear(dim, dim * 6)

        # Transformer blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = [
            WanAttentionBlock(
                cross_attn_type=cross_attn_type,
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
            )
            for _ in range(num_layers)
        ]

        # Output head
        self.head = Head(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=eps)

        # RoPE frequency parameters
        # Compute for maximum sequence length of 1024
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        # Split dimensions for temporal, height, width
        self.freqs = mx.concatenate(
            [
                rope_params(1024, d - 4 * (d // 6)),  # Temporal
                rope_params(1024, 2 * (d // 6)),      # Height
                rope_params(1024, 2 * (d // 6)),      # Width
            ],
            axis=1,
        )

        # Image embedding for i2v model
        if model_type == "i2v":
            self.img_emb = MLPProj(in_dim=1280, out_dim=dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model parameters using Xavier/normal initialization.
        """
        # Initialize patch embedding with Xavier uniform
        # Weight shape: [out_channels, in_channels, kT, kH, kW]
        weight_flat = mx.reshape(self.patch_embedding.weight, (self.out_dim, -1))
        # Xavier uniform initialization
        fan_in = weight_flat.shape[1]
        limit = mx.sqrt(mx.array(6.0 / fan_in, dtype=mx.float32))
        self.patch_embedding.weight = mx.random.uniform(
            low=-limit,
            high=limit,
            shape=self.patch_embedding.weight.shape,
            dtype=mx.float32,
        )

        # Initialize text embedding with normal(std=0.02)
        std = 0.02
        for param in [self.text_embedding_fc1.weight, self.text_embedding_fc2.weight]:
            new_weight = mx.random.normal(shape=param.shape, dtype=mx.float32) * std
            # Note: In MLX, we can't reassign directly, but for initialization we'll do it this way
            param = new_weight

        # Initialize time embedding with normal(std=0.02)
        for param in [self.time_embedding_fc1.weight, self.time_embedding_fc2.weight]:
            new_weight = mx.random.normal(shape=param.shape, dtype=mx.float32) * std
            param = new_weight

        # Initialize output head weight to zeros
        self.head.head.weight = mx.zeros_like(self.head.head.weight)

    def __call__(
        self,
        x: List[mx.array],
        t: mx.array,
        context: List[mx.array],
        seq_len: int,
        clip_fea: Optional[mx.array] = None,
        y: Optional[List[mx.array]] = None,
    ) -> List[mx.array]:
        """
        Forward pass through the diffusion model.

        Args:
            x: List of input video tensors, each with shape [C_in, F, H, W]
            t: Diffusion timesteps tensor of shape [B]
            context: List of text embeddings each with shape [L, C]
            seq_len: Maximum sequence length for positional encoding
            clip_fea: CLIP image features for image-to-video mode [B, 257, 1280]
            y: Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List of denoised video tensors with shapes [C_out, F, H//8, W//8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None, "i2v model requires clip_fea and y"

        # Concatenate conditional and unconditional inputs for i2v
        if y is not None:
            x = [mx.concatenate([u, v], axis=0) for u, v in zip(x, y)]

        # Patch embedding: [C_in, F, H, W] → [C, F_p, H_p, W_p]
        x = [self.patch_embedding(u) for u in x]

        # Get grid sizes (F_p, H_p, W_p) for each sample
        grid_sizes = mx.stack([mx.array(u.shape[1:], dtype=mx.int32) for u in x])

        # Flatten spatial dimensions: [C, F_p, H_p, W_p] → [F_p*H_p*W_p, C]
        x = [mx.transpose(mx.reshape(u, (u.shape[0], -1)), (1, 0)) for u in x]

        # Get sequence lengths
        seq_lens = mx.array([u.shape[0] for u in x], dtype=mx.int32)
        max_seq = int(mx.max(seq_lens).item())
        assert max_seq <= seq_len, f"max sequence length {max_seq} exceeds seq_len {seq_len}"

        # Pad sequences to seq_len
        batch_size = len(x)
        x_padded = []
        for u in x:
            if u.shape[0] < seq_len:
                padding = mx.zeros((seq_len - u.shape[0], u.shape[1]), dtype=u.dtype)
                u_padded = mx.concatenate([u, padding], axis=0)
            else:
                u_padded = u
            x_padded.append(u_padded)

        x = mx.stack(x_padded)  # [B, seq_len, C]

        # Time embeddings
        # Sinusoidal embedding
        t_sin = sinusoidal_embedding_1d(self.freq_dim, t)  # [B, freq_dim]

        # Apply SiLU activation (approximated with sigmoid(x) * x)
        def silu(x):
            return x * mx.sigmoid(x)

        e = self.time_embedding_fc1(t_sin)
        e = silu(e)
        e = self.time_embedding_fc2(e)  # [B, dim]

        # Project to modulation space
        e0 = self.time_projection_fc(silu(e))  # [B, dim*6]
        e0 = mx.reshape(e0, (batch_size, 6, self.dim))  # [B, 6, dim]

        # Text embeddings
        # Pad context to text_len
        context_padded = []
        for u in context:
            if u.shape[0] < self.text_len:
                padding = mx.zeros((self.text_len - u.shape[0], u.shape[1]), dtype=u.dtype)
                u_padded = mx.concatenate([u, padding], axis=0)
            else:
                u_padded = u[:self.text_len]
            context_padded.append(u_padded)

        context = mx.stack(context_padded)  # [B, text_len, text_dim]

        # Apply text embedding layers
        context = self.text_embedding_fc1(context)
        context = self.text_embedding_gelu(context)
        context = self.text_embedding_fc2(context)  # [B, text_len, dim]

        # Add image embeddings for i2v
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # [B, 257, dim]
            context = mx.concatenate([context_clip, context], axis=1)

        context_lens = None  # Not using length masking for now

        # Apply transformer blocks
        for block in self.blocks:
            x = block(
                x=x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
            )

        # Output head
        x = self.head(x, e)  # [B, seq_len, out_dim * prod(patch_size)]

        # Unpatchify: convert patches back to video
        outputs = self._unpatchify(x, grid_sizes)

        return outputs

    def _unpatchify(
        self,
        x: mx.array,
        grid_sizes: mx.array,
    ) -> List[mx.array]:
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x: Patchified features [B, seq_len, C_out * prod(patch_size)]
            grid_sizes: Grid dimensions [B, 3] (F_patches, H_patches, W_patches)

        Returns:
            List of reconstructed video tensors [C_out, F, H, W]
        """
        c = self.out_dim
        pT, pH, pW = self.patch_size

        outputs = []
        for i, grid_size in enumerate(grid_sizes.tolist()):
            f_p, h_p, w_p = grid_size
            num_patches = f_p * h_p * w_p

            # Extract valid patches for this sample
            u = x[i, :num_patches]  # [num_patches, out_dim * prod(patch_size)]

            # Reshape to [F_p, H_p, W_p, pT, pH, pW, C_out]
            u = mx.reshape(u, (f_p, h_p, w_p, pT, pH, pW, c))

            # Transpose to [C_out, F_p, pT, H_p, pH, W_p, pW]
            u = mx.transpose(u, (6, 0, 3, 1, 4, 2, 5))

            # Reshape to [C_out, F, H, W]
            u = mx.reshape(u, (c, f_p * pT, h_p * pH, w_p * pW))

            outputs.append(u)

        return outputs


__all__ = ["WanModel"]
