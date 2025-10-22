# Copyright 2024-2025 Hanzo AI. All rights reserved.
"""
Backend selection utilities for WAN models.

Automatically detects available backends (MLX, PyTorch MPS, PyTorch CUDA)
and provides a unified interface for model initialization.
"""

import sys
import platform
from typing import Optional, Literal
from dataclasses import dataclass


BackendType = Literal["mlx", "pytorch_mps", "pytorch_cuda", "pytorch_cpu"]


@dataclass
class BackendInfo:
    """Information about an available backend."""

    name: BackendType
    available: bool
    priority: int  # Lower is better
    performance_score: float  # Estimated relative performance
    description: str


def check_mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        # Try a simple operation to ensure it works
        _ = mx.array([1.0])
        return True
    except Exception:
        return False


def check_pytorch_available() -> tuple[bool, bool, bool]:
    """
    Check PyTorch availability and device support.

    Returns:
        Tuple of (has_pytorch, has_mps, has_cuda)
    """
    try:
        import torch

        has_mps = torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        return True, has_mps, has_cuda
    except Exception:
        return False, False, False


def get_available_backends() -> list[BackendInfo]:
    """
    Get list of available backends sorted by priority.

    Returns:
        List of BackendInfo objects for available backends
    """
    backends = []

    # Check if running on macOS (required for MLX and MPS)
    is_macos = platform.system() == "Darwin"

    # MLX - Best performance on Apple Silicon
    if is_macos and check_mlx_available():
        backends.append(
            BackendInfo(
                name="mlx",
                available=True,
                priority=1,
                performance_score=1.0,  # Reference: best performance
                description="MLX native (2-3x faster than PyTorch MPS on Apple Silicon)",
            )
        )

    # PyTorch backends
    has_torch, has_mps, has_cuda = check_pytorch_available()

    if has_cuda:
        backends.append(
            BackendInfo(
                name="pytorch_cuda",
                available=True,
                priority=2,
                performance_score=0.9,  # Slightly slower than MLX on Apple Silicon
                description="PyTorch with CUDA (best for NVIDIA GPUs)",
            )
        )

    if is_macos and has_mps:
        backends.append(
            BackendInfo(
                name="pytorch_mps",
                available=True,
                priority=3,
                performance_score=0.4,  # 2.5x slower than MLX
                description="PyTorch with MPS (Metal Performance Shaders)",
            )
        )

    if has_torch:
        backends.append(
            BackendInfo(
                name="pytorch_cpu",
                available=True,
                priority=4,
                performance_score=0.1,  # Much slower than GPU
                description="PyTorch CPU fallback (slow, not recommended)",
            )
        )

    # Sort by priority
    backends.sort(key=lambda x: x.priority)

    return backends


def select_backend(
    preferred_backend: Optional[BackendType] = None,
    verbose: bool = True,
) -> BackendInfo:
    """
    Select the best available backend.

    Args:
        preferred_backend: Optionally specify a preferred backend
        verbose: Whether to print backend selection info

    Returns:
        BackendInfo for the selected backend

    Raises:
        RuntimeError: If no suitable backend is available
    """
    available = get_available_backends()

    if not available:
        raise RuntimeError(
            "No suitable backend found. Please install either:\n"
            "  - MLX (pip install mlx) on macOS with Apple Silicon\n"
            "  - PyTorch (pip install torch) with CUDA or MPS support"
        )

    # If preferred backend specified, try to find it
    if preferred_backend:
        for backend in available:
            if backend.name == preferred_backend:
                if verbose:
                    print(f"Using preferred backend: {backend.name}")
                    print(f"  {backend.description}")
                return backend

        # Preferred backend not available
        if verbose:
            print(f"Warning: Preferred backend '{preferred_backend}' not available")
            print(f"Falling back to best available backend")

    # Use highest priority available backend
    selected = available[0]

    if verbose:
        print(f"Auto-selected backend: {selected.name}")
        print(f"  {selected.description}")
        print(f"  Estimated performance: {selected.performance_score:.1%} of MLX")

        if len(available) > 1:
            print(f"\nOther available backends:")
            for backend in available[1:]:
                print(f"  - {backend.name}: {backend.description}")

    return selected


def create_model(
    model_type: str = "t2v",
    backend: Optional[BackendType] = None,
    **model_kwargs,
):
    """
    Create a WAN model with automatic backend selection.

    Args:
        model_type: Model variant ('t2v' or 'i2v')
        backend: Optional backend override
        **model_kwargs: Additional arguments passed to model constructor

    Returns:
        WAN model instance (either MLX or PyTorch)

    Example:
        ```python
        # Auto-select best backend
        model = create_model(model_type="t2v", dim=2048, num_layers=32)

        # Force MLX backend
        model = create_model(model_type="t2v", backend="mlx", dim=2048)
        ```
    """
    selected = select_backend(backend, verbose=True)

    if selected.name == "mlx":
        print("Loading MLX model...")
        from .model import WanModel

        model = WanModel(model_type=model_type, **model_kwargs)
        return model

    elif selected.name.startswith("pytorch"):
        print("Loading PyTorch model...")
        from ..wan_base.modules.model import WanModel

        model = WanModel(model_type=model_type, **model_kwargs)

        # Move to appropriate device
        if selected.name == "pytorch_cuda":
            import torch
            model = model.cuda()
        elif selected.name == "pytorch_mps":
            import torch
            model = model.to("mps")

        return model

    else:
        raise RuntimeError(f"Unknown backend: {selected.name}")


def print_backend_info():
    """Print detailed information about all available backends."""
    print("\n" + "=" * 70)
    print("WAN Model Backend Information")
    print("=" * 70)

    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")

    available = get_available_backends()

    if not available:
        print("\n⚠️  No backends available!")
        print("Please install MLX or PyTorch.")
        return

    print(f"\nAvailable backends: {len(available)}")
    print("-" * 70)

    for i, backend in enumerate(available, 1):
        marker = "✓" if i == 1 else " "
        print(f"\n{marker} {i}. {backend.name.upper()}")
        print(f"     Priority: {backend.priority}")
        print(f"     Performance: {backend.performance_score:.1%} relative to MLX")
        print(f"     Description: {backend.description}")

    print("\n" + "=" * 70)

    # Recommendations
    if available[0].name == "mlx":
        print("✅ MLX is available and recommended for best performance!")
    elif available[0].name == "pytorch_mps":
        print("⚠️  Consider installing MLX for 2-3x better performance:")
        print("   pip install mlx")
    elif available[0].name == "pytorch_cuda":
        print("✅ CUDA is available - good performance on NVIDIA GPUs")
    else:
        print("⚠️  Running on CPU - performance will be limited")
        print("   Consider using a system with GPU support")

    print("=" * 70 + "\n")


__all__ = [
    "BackendType",
    "BackendInfo",
    "check_mlx_available",
    "check_pytorch_available",
    "get_available_backends",
    "select_backend",
    "create_model",
    "print_backend_info",
]
