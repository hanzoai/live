"""MLX-native implementation of WAN models for Apple Silicon."""

from .attention import scaled_dot_product_attention

__all__ = ["scaled_dot_product_attention"]
