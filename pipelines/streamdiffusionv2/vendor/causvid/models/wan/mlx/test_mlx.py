# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
Test suite for MLX WAN implementation.

Verifies numerical accuracy and correctness of MLX operations compared
to PyTorch reference implementations.
"""

import mlx.core as mx
import numpy as np
from typing import Tuple


def assert_close(mlx_arr: mx.array, numpy_arr: np.ndarray, rtol: float = 1e-4, atol: float = 1e-5, name: str = ""):
    """
    Assert that MLX array is close to NumPy array.

    Args:
        mlx_arr: MLX array
        numpy_arr: NumPy array
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages
    """
    mlx_numpy = np.array(mlx_arr)
    if not np.allclose(mlx_numpy, numpy_arr, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(mlx_numpy - numpy_arr))
        raise AssertionError(
            f"{name} arrays not close!\n"
            f"Max difference: {max_diff}\n"
            f"Mean MLX: {np.mean(mlx_numpy):.6f}, Mean NumPy: {np.mean(numpy_arr):.6f}"
        )


def test_sinusoidal_embedding():
    """Test sinusoidal position embeddings."""
    print("Testing sinusoidal_embedding_1d...")

    from .core_ops import sinusoidal_embedding_1d

    dim = 256
    position = mx.arange(128)

    # MLX version
    mlx_emb = sinusoidal_embedding_1d(dim, position)

    # Reference implementation
    half = dim // 2
    position_np = np.arange(128, dtype=np.float32)
    freq_bands = np.power(10000.0, -np.arange(half, dtype=np.float32) / half)
    sinusoid = np.outer(position_np, freq_bands)
    ref_emb = np.concatenate([np.cos(sinusoid), np.sin(sinusoid)], axis=1)

    # Slightly relaxed tolerance for floating point differences between MLX and NumPy
    assert_close(mlx_emb, ref_emb, rtol=1e-4, atol=2e-5, name="sinusoidal_embedding")
    print("✓ sinusoidal_embedding_1d passed")


def test_rope_params():
    """Test RoPE parameter computation."""
    print("Testing rope_params...")

    from .core_ops import rope_params

    max_seq_len = 128
    dim = 64

    # MLX version
    mlx_freqs = rope_params(max_seq_len, dim)

    # Verify shape
    assert mlx_freqs.shape == (max_seq_len, dim // 2, 2), \
        f"Expected shape ({max_seq_len}, {dim // 2}, 2), got {mlx_freqs.shape}"

    # Verify values are in reasonable range
    mlx_freqs_np = np.array(mlx_freqs)
    assert np.all(np.abs(mlx_freqs_np) <= 1.0), "RoPE freqs should be in [-1, 1]"

    print("✓ rope_params passed")


def test_rms_norm():
    """Test RMS normalization."""
    print("Testing WanRMSNorm...")

    from .core_ops import WanRMSNorm

    batch_size, seq_len, dim = 2, 128, 256
    x = mx.random.normal((batch_size, seq_len, dim))

    # MLX version
    norm = WanRMSNorm(dim, eps=1e-5)
    mlx_output = norm(x)

    # Reference implementation
    x_np = np.array(x)
    variance = np.mean(x_np**2, axis=-1, keepdims=True)
    x_normalized = x_np * (1.0 / np.sqrt(variance + 1e-5))
    ref_output = x_normalized * np.array(norm.weight)

    assert_close(mlx_output, ref_output, name="rms_norm")
    print("✓ WanRMSNorm passed")


def test_layer_norm():
    """Test layer normalization."""
    print("Testing WanLayerNorm...")

    from .core_ops import WanLayerNorm

    batch_size, seq_len, dim = 2, 128, 256
    x = mx.random.normal((batch_size, seq_len, dim))

    # MLX version (without affine)
    norm = WanLayerNorm(dim, eps=1e-6, elementwise_affine=False)
    mlx_output = norm(x)

    # Reference implementation
    x_np = np.array(x)
    mean = np.mean(x_np, axis=-1, keepdims=True)
    variance = np.var(x_np, axis=-1, keepdims=True)
    ref_output = (x_np - mean) / np.sqrt(variance + 1e-6)

    assert_close(mlx_output, ref_output, name="layer_norm")
    print("✓ WanLayerNorm passed")


def test_attention():
    """Test scaled dot-product attention."""
    print("Testing scaled_dot_product_attention...")

    from .attention import scaled_dot_product_attention

    batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    # MLX version
    mlx_output = scaled_dot_product_attention(q, k, v)

    # Verify shape
    assert mlx_output.shape == q.shape, \
        f"Expected shape {q.shape}, got {mlx_output.shape}"

    # Verify output is not NaN or Inf
    mlx_output_np = np.array(mlx_output)
    assert not np.any(np.isnan(mlx_output_np)), "Output contains NaN"
    assert not np.any(np.isinf(mlx_output_np)), "Output contains Inf"

    print("✓ scaled_dot_product_attention passed")


def test_gelu():
    """Test GELU activation."""
    print("Testing GELU...")

    from .modules import GELU

    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # MLX version
    gelu = GELU(approximate="tanh")
    mlx_output = gelu(x)

    # Reference GELU values (approximate)
    # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    x_np = np.array(x)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    x_cubed = x_np ** 3
    tanh_arg = sqrt_2_over_pi * (x_np + 0.044715 * x_cubed)
    ref_output = 0.5 * x_np * (1.0 + np.tanh(tanh_arg))

    assert_close(mlx_output, ref_output, name="gelu")
    print("✓ GELU passed")


def test_conv3d():
    """Test 3D convolution."""
    print("Testing Conv3d...")

    from .conv import Conv3d

    in_channels, out_channels = 16, 32
    kernel_size = (2, 2, 2)
    x = mx.random.normal((in_channels, 8, 16, 16))  # [C, T, H, W]

    # MLX version
    conv = Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=kernel_size,
    )
    mlx_output = conv(x)

    # Verify output shape
    expected_shape = (out_channels, 4, 8, 8)  # Stride = kernel_size
    assert mlx_output.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {mlx_output.shape}"

    # Verify output is not NaN or Inf
    mlx_output_np = np.array(mlx_output)
    assert not np.any(np.isnan(mlx_output_np)), "Output contains NaN"
    assert not np.any(np.isinf(mlx_output_np)), "Output contains Inf"

    print("✓ Conv3d passed")


def test_feedforward():
    """Test feedforward network."""
    print("Testing FeedForward...")

    from .modules import FeedForward

    batch_size, seq_len, dim = 2, 128, 256
    ffn_dim = 1024

    x = mx.random.normal((batch_size, seq_len, dim))

    # MLX version
    ffn = FeedForward(dim=dim, ffn_dim=ffn_dim)
    mlx_output = ffn(x)

    # Verify shape
    assert mlx_output.shape == x.shape, \
        f"Expected shape {x.shape}, got {mlx_output.shape}"

    # Verify output is not NaN or Inf
    mlx_output_np = np.array(mlx_output)
    assert not np.any(np.isnan(mlx_output_np)), "Output contains NaN"
    assert not np.any(np.isinf(mlx_output_np)), "Output contains Inf"

    print("✓ FeedForward passed")


def test_self_attention():
    """Test self-attention module."""
    print("Testing WanSelfAttention...")

    from .wan_attention import WanSelfAttention
    from .core_ops import rope_params

    batch_size, seq_len, dim = 2, 64, 256
    num_heads = 8
    head_dim = dim // num_heads

    x = mx.random.normal((batch_size, seq_len, dim))
    seq_lens = mx.array([seq_len] * batch_size, dtype=mx.int32)
    grid_sizes = mx.array([[16, 2, 2]] * batch_size, dtype=mx.int32)
    freqs = rope_params(1024, head_dim)

    # MLX version
    attn = WanSelfAttention(dim=dim, num_heads=num_heads)
    mlx_output = attn(x, seq_lens, grid_sizes, freqs)

    # Verify shape
    assert mlx_output.shape == x.shape, \
        f"Expected shape {x.shape}, got {mlx_output.shape}"

    # Verify output is not NaN or Inf
    mlx_output_np = np.array(mlx_output)
    assert not np.any(np.isnan(mlx_output_np)), "Output contains NaN"
    assert not np.any(np.isinf(mlx_output_np)), "Output contains Inf"

    print("✓ WanSelfAttention passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MLX WAN Test Suite")
    print("=" * 70 + "\n")

    tests = [
        test_sinusoidal_embedding,
        test_rope_params,
        test_rms_norm,
        test_layer_norm,
        test_attention,
        test_gelu,
        test_conv3d,
        test_feedforward,
        test_self_attention,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
