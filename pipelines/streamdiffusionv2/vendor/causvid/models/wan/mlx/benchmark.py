# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
Comprehensive benchmarking suite for MLX vs PyTorch WAN models.

Compares performance across different operations and model configurations
to quantify the speedup provided by MLX on Apple Silicon.
"""

import time
import mlx.core as mx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    operation: str
    backend: str
    duration_ms: float
    memory_mb: Optional[float] = None
    throughput: Optional[float] = None  # ops/sec or tokens/sec
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison between MLX and PyTorch."""

    operation: str
    mlx_time_ms: float
    pytorch_time_ms: float
    speedup: float
    mlx_memory_mb: Optional[float] = None
    pytorch_memory_mb: Optional[float] = None


def benchmark_attention(
    batch_size: int = 1,
    seq_len: int = 256,
    num_heads: int = 16,
    head_dim: int = 64,
    num_iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark attention operation on both MLX and PyTorch.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results for each backend
    """
    results = {}

    # MLX benchmark
    print(f"Benchmarking MLX attention ({seq_len}x{seq_len})...")
    from .attention import scaled_dot_product_attention

    # Create random inputs
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    # Warmup
    for _ in range(warmup):
        _ = scaled_dot_product_attention(q, k, v)
        mx.eval(_)  # Ensure computation completes

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = scaled_dot_product_attention(q, k, v)
        mx.eval(output)  # Force evaluation
    end = time.perf_counter()

    mlx_time = (end - start) * 1000 / num_iterations

    results["mlx"] = BenchmarkResult(
        operation="attention",
        backend="mlx",
        duration_ms=mlx_time,
        throughput=1000 / mlx_time,  # ops/sec
        metadata={
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
    )

    # PyTorch MPS benchmark
    try:
        import torch

        if torch.backends.mps.is_available():
            print(f"Benchmarking PyTorch MPS attention ({seq_len}x{seq_len})...")

            device = torch.device("mps")
            q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            k_torch = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v_torch = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

            # Warmup
            for _ in range(warmup):
                _ = torch.nn.functional.scaled_dot_product_attention(q_torch, k_torch, v_torch)
                torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                output = torch.nn.functional.scaled_dot_product_attention(q_torch, k_torch, v_torch)
                torch.mps.synchronize()
            end = time.perf_counter()

            pytorch_time = (end - start) * 1000 / num_iterations

            results["pytorch_mps"] = BenchmarkResult(
                operation="attention",
                backend="pytorch_mps",
                duration_ms=pytorch_time,
                throughput=1000 / pytorch_time,
                metadata={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                },
            )
    except Exception as e:
        print(f"PyTorch MPS benchmark failed: {e}")

    return results


def benchmark_rope_apply(
    batch_size: int = 1,
    seq_len: int = 256,
    num_heads: int = 16,
    head_dim: int = 64,
    num_iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark RoPE (Rotary Position Embeddings) application.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # MLX benchmark
    print(f"Benchmarking MLX RoPE ({seq_len} tokens)...")
    from .core_ops import rope_params, rope_apply

    x = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    grid_sizes = mx.array([[16, 4, 4]] * batch_size, dtype=mx.int32)  # F=16, H=4, W=4
    freqs = rope_params(1024, head_dim)

    # Warmup
    for _ in range(warmup):
        _ = rope_apply(x, grid_sizes, freqs)
        mx.eval(_)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = rope_apply(x, grid_sizes, freqs)
        mx.eval(output)
    end = time.perf_counter()

    mlx_time = (end - start) * 1000 / num_iterations

    results["mlx"] = BenchmarkResult(
        operation="rope_apply",
        backend="mlx",
        duration_ms=mlx_time,
        throughput=1000 / mlx_time,
        metadata={"batch_size": batch_size, "seq_len": seq_len},
    )

    # PyTorch benchmark
    try:
        import torch
        from ..wan_base.modules.model import rope_apply as torch_rope_apply, rope_params as torch_rope_params

        if torch.backends.mps.is_available():
            print(f"Benchmarking PyTorch MPS RoPE ({seq_len} tokens)...")

            device = torch.device("mps")
            x_torch = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            grid_sizes_torch = torch.tensor([[16, 4, 4]] * batch_size, device=device)
            freqs_torch = torch_rope_params(1024, head_dim).to(device)

            # Warmup
            for _ in range(warmup):
                _ = torch_rope_apply(x_torch, grid_sizes_torch, freqs_torch)
                torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                output = torch_rope_apply(x_torch, grid_sizes_torch, freqs_torch)
                torch.mps.synchronize()
            end = time.perf_counter()

            pytorch_time = (end - start) * 1000 / num_iterations

            results["pytorch_mps"] = BenchmarkResult(
                operation="rope_apply",
                backend="pytorch_mps",
                duration_ms=pytorch_time,
                throughput=1000 / pytorch_time,
                metadata={"batch_size": batch_size, "seq_len": seq_len},
            )
    except Exception as e:
        print(f"PyTorch MPS benchmark failed: {e}")

    return results


def benchmark_rms_norm(
    batch_size: int = 1,
    seq_len: int = 256,
    dim: int = 2048,
    num_iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark RMS normalization.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        dim: Hidden dimension
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # MLX benchmark
    print(f"Benchmarking MLX RMS Norm ({seq_len}x{dim})...")
    from .core_ops import WanRMSNorm

    x = mx.random.normal((batch_size, seq_len, dim))
    norm = WanRMSNorm(dim)

    # Warmup
    for _ in range(warmup):
        _ = norm(x)
        mx.eval(_)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = norm(x)
        mx.eval(output)
    end = time.perf_counter()

    mlx_time = (end - start) * 1000 / num_iterations

    results["mlx"] = BenchmarkResult(
        operation="rms_norm",
        backend="mlx",
        duration_ms=mlx_time,
        throughput=1000 / mlx_time,
        metadata={"batch_size": batch_size, "seq_len": seq_len, "dim": dim},
    )

    # PyTorch benchmark
    try:
        import torch
        from ..wan_base.modules.model import WanRMSNorm as TorchRMSNorm

        if torch.backends.mps.is_available():
            print(f"Benchmarking PyTorch MPS RMS Norm ({seq_len}x{dim})...")

            device = torch.device("mps")
            x_torch = torch.randn(batch_size, seq_len, dim, device=device)
            norm_torch = TorchRMSNorm(dim).to(device)

            # Warmup
            for _ in range(warmup):
                _ = norm_torch(x_torch)
                torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                output = norm_torch(x_torch)
                torch.mps.synchronize()
            end = time.perf_counter()

            pytorch_time = (end - start) * 1000 / num_iterations

            results["pytorch_mps"] = BenchmarkResult(
                operation="rms_norm",
                backend="pytorch_mps",
                duration_ms=pytorch_time,
                throughput=1000 / pytorch_time,
                metadata={"batch_size": batch_size, "seq_len": seq_len, "dim": dim},
            )
    except Exception as e:
        print(f"PyTorch MPS benchmark failed: {e}")

    return results


def run_full_benchmark_suite() -> List[ComparisonResult]:
    """
    Run comprehensive benchmark suite comparing MLX and PyTorch.

    Returns:
        List of comparison results
    """
    print("\n" + "=" * 70)
    print("WAN Model Benchmark Suite: MLX vs PyTorch MPS")
    print("=" * 70 + "\n")

    comparisons = []

    # Attention benchmarks
    for seq_len in [128, 256, 512]:
        results = benchmark_attention(seq_len=seq_len, num_iterations=50)

        if "mlx" in results and "pytorch_mps" in results:
            mlx_time = results["mlx"].duration_ms
            pytorch_time = results["pytorch_mps"].duration_ms
            speedup = pytorch_time / mlx_time

            comparisons.append(
                ComparisonResult(
                    operation=f"attention_{seq_len}x{seq_len}",
                    mlx_time_ms=mlx_time,
                    pytorch_time_ms=pytorch_time,
                    speedup=speedup,
                )
            )

    # RoPE benchmarks
    for seq_len in [128, 256, 512]:
        results = benchmark_rope_apply(seq_len=seq_len, num_iterations=50)

        if "mlx" in results and "pytorch_mps" in results:
            mlx_time = results["mlx"].duration_ms
            pytorch_time = results["pytorch_mps"].duration_ms
            speedup = pytorch_time / mlx_time

            comparisons.append(
                ComparisonResult(
                    operation=f"rope_apply_{seq_len}",
                    mlx_time_ms=mlx_time,
                    pytorch_time_ms=pytorch_time,
                    speedup=speedup,
                )
            )

    # RMS Norm benchmarks
    for seq_len in [128, 256, 512]:
        results = benchmark_rms_norm(seq_len=seq_len, dim=2048, num_iterations=50)

        if "mlx" in results and "pytorch_mps" in results:
            mlx_time = results["mlx"].duration_ms
            pytorch_time = results["pytorch_mps"].duration_ms
            speedup = pytorch_time / mlx_time

            comparisons.append(
                ComparisonResult(
                    operation=f"rms_norm_{seq_len}x2048",
                    mlx_time_ms=mlx_time,
                    pytorch_time_ms=pytorch_time,
                    speedup=speedup,
                )
            )

    # Print results
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"{'Operation':<25} {'MLX (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for comp in comparisons:
        print(
            f"{comp.operation:<25} {comp.mlx_time_ms:>10.2f}   "
            f"{comp.pytorch_time_ms:>12.2f}   {comp.speedup:>8.2f}x"
        )

    # Calculate average speedup
    avg_speedup = sum(c.speedup for c in comparisons) / len(comparisons)

    print("-" * 70)
    print(f"{'Average Speedup':<25} {'':<12} {'':<14} {avg_speedup:>8.2f}x")
    print("=" * 70 + "\n")

    return comparisons


__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "benchmark_attention",
    "benchmark_rope_apply",
    "benchmark_rms_norm",
    "run_full_benchmark_suite",
]


if __name__ == "__main__":
    # Run benchmark suite when executed directly
    run_full_benchmark_suite()
