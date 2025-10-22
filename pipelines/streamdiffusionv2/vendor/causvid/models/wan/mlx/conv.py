# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
MLX-native 3D convolution implementation.

Implements Conv3d using manual sliding window operations since MLX doesn't
natively support 3D convolutions yet. This is specifically optimized for
the WAN patch embedding use case.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Union


class Conv3d(nn.Module):
    """
    3D convolution layer using manual sliding window implementation.

    Specifically designed for patch embedding in video diffusion models.
    Implements standard Conv3d operation using MLX primitives.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel (T, H, W)
            stride: Stride of the convolution (T, H, W)
            padding: Zero-padding added to all three sides (T, H, W)
            bias: If True, adds a learnable bias
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert to tuples if single int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Initialize weights: [out_channels, in_channels, kT, kH, kW]
        # Use Kaiming initialization
        kT, kH, kW = kernel_size
        n = in_channels * kT * kH * kW
        scale = mx.sqrt(mx.array(2.0 / n, dtype=mx.float32))

        self.weight = mx.random.normal(
            shape=(out_channels, in_channels, kT, kH, kW),
            dtype=mx.float32
        ) * scale

        if bias:
            self.bias = mx.zeros((out_channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply 3D convolution.

        Args:
            x: Input tensor [C_in, T, H, W] or [batch, C_in, T, H, W]

        Returns:
            Output tensor [C_out, T_out, H_out, W_out] or [batch, C_out, T_out, H_out, W_out]
        """
        # Handle both batched and unbatched inputs
        if x.ndim == 4:
            # Unbatched: [C_in, T, H, W] -> add batch dimension
            x = mx.expand_dims(x, axis=0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, C_in, T, H, W = x.shape
        assert C_in == self.in_channels, f"Expected {self.in_channels} input channels, got {C_in}"

        kT, kH, kW = self.kernel_size
        sT, sH, sW = self.stride
        pT, pH, pW = self.padding

        # Apply padding if needed
        if any(p > 0 for p in self.padding):
            # Pad: [[batch_pad], [channel_pad], [T_pad], [H_pad], [W_pad]]
            pad_width = [
                (0, 0),  # batch
                (0, 0),  # channels
                (pT, pT),  # temporal
                (pH, pH),  # height
                (pW, pW),  # width
            ]
            x = mx.pad(x, pad_width)
            T, H, W = T + 2*pT, H + 2*pH, W + 2*pW

        # Calculate output dimensions
        T_out = (T - kT) // sT + 1
        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1

        # Extract patches using sliding window
        # We'll iterate through the spatial-temporal dimensions
        patches = []

        for t in range(0, T - kT + 1, sT):
            for h in range(0, H - kH + 1, sH):
                for w in range(0, W - kW + 1, sW):
                    # Extract patch: [batch, C_in, kT, kH, kW]
                    patch = x[:, :, t:t+kT, h:h+kH, w:w+kW]
                    # Flatten spatial dimensions: [batch, C_in*kT*kH*kW]
                    patch = mx.reshape(patch, (batch_size, -1))
                    patches.append(patch)

        # Stack all patches: [batch, num_patches, C_in*kT*kH*kW]
        patches = mx.stack(patches, axis=1)

        # Reshape weight: [C_out, C_in*kT*kH*kW]
        weight_flat = mx.reshape(self.weight, (self.out_channels, -1))

        # Apply convolution as matrix multiplication
        # patches: [batch, num_patches, C_in*kT*kH*kW]
        # weight_flat.T: [C_in*kT*kH*kW, C_out]
        # output: [batch, num_patches, C_out]
        output = mx.matmul(patches, weight_flat.T)

        # Add bias if enabled
        if self.use_bias:
            output = output + self.bias

        # Reshape to [batch, T_out, H_out, W_out, C_out]
        output = mx.reshape(output, (batch_size, T_out, H_out, W_out, self.out_channels))

        # Transpose to [batch, C_out, T_out, H_out, W_out]
        output = mx.transpose(output, (0, 4, 1, 2, 3))

        # Remove batch dimension if input was unbatched
        if squeeze_output:
            output = mx.squeeze(output, axis=0)

        return output


__all__ = ["Conv3d"]
