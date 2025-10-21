"""Self-Forcing pipeline for autoregressive video generation.

Based on "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
https://github.com/guandeh17/Self-Forcing
"""

import logging
import time

import torch

from ..base.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from ..interface import Pipeline, Requirements

logger = logging.getLogger(__name__)


class SelfForcingPipeline(Pipeline):
    """Self-Forcing autoregressive video diffusion pipeline.

    This pipeline implements the Self-Forcing training method which bridges
    the train-test gap in autoregressive video generation by simulating
    the inference process during training with KV caching.

    Features:
    - Real-time streaming generation on RTX 4090
    - Chunk-wise autoregressive rollout
    - KV caching for improved efficiency
    - DMD (Distribution Matching Distillation) training
    """

    def __init__(
        self,
        config,
        low_memory: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the Self-Forcing pipeline.

        Args:
            config: Pipeline configuration from model.yaml
            low_memory: Enable low memory mode
            device: Torch device to use
            dtype: Data type for model weights
        """
        model_dir = getattr(config, "model_dir", None)
        checkpoint_path = getattr(config, "checkpoint_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)

        # Self-Forcing specific settings
        self.chunk_size = getattr(config, "chunk_size", 16)
        self.use_kv_cache = getattr(config, "use_kv_cache", True)
        self.autoregressive = getattr(config, "autoregressive", True)

        # Load base Wan2.1 diffusion model
        start = time.time()
        self.generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}),
            model_dir=model_dir,
            is_causal=self.autoregressive  # Enable causal attention for autoregressive generation
        )
        logger.info(f"Loaded diffusion wrapper in {time.time() - start:.3f}s")

        # Load Self-Forcing checkpoint
        if checkpoint_path:
            start = time.time()
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
            )
            # Load generator state from Self-Forcing checkpoint
            if "generator" in checkpoint:
                self.generator.load_state_dict(checkpoint["generator"])
            elif "model" in checkpoint:
                self.generator.load_state_dict(checkpoint["model"])
            else:
                # Assume the checkpoint itself is the state dict
                self.generator.load_state_dict(checkpoint)
            logger.info(f"Loaded Self-Forcing checkpoint in {time.time() - start:.3f}s")

        # Load text encoder
        start = time.time()
        self.text_encoder = WanTextEncoder(
            model_dir=model_dir,
            text_encoder_path=text_encoder_path
        )
        logger.info(f"Loaded text encoder in {time.time() - start:.3f}s")

        # Load VAE
        start = time.time()
        self.vae = WanVAEWrapper(model_dir=model_dir)
        logger.info(f"Loaded VAE in {time.time() - start:.3f}s")

        # Move models to device
        self.device = device
        self.dtype = dtype

        self.generator = self.generator.to(device=self.device, dtype=self.dtype)
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=self.dtype)
        self.vae = self.vae.to(device=self.device, dtype=self.dtype)

        # Initialize state
        self.seed = getattr(config, "seed", 42)
        self.height = getattr(config, "height", 320)
        self.width = getattr(config, "width", 576)
        self.prompts = None
        self.encoded_prompts = None

        logger.info(f"Self-Forcing pipeline initialized with chunk_size={self.chunk_size}, "
                   f"kv_cache={self.use_kv_cache}, autoregressive={self.autoregressive}")

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        """Prepare the pipeline for generation.

        Args:
            should_prepare: Force preparation even if params haven't changed
            **kwargs: Additional parameters
                - prompts: Text prompts for generation
                - seed: Random seed

        Returns:
            Requirements object or None
        """
        prompts = kwargs.get("prompts", None)
        seed = kwargs.get("seed", self.seed)

        # Check if we need to re-encode prompts
        if prompts is not None and prompts != self.prompts:
            should_prepare = True
            self.prompts = prompts

        if should_prepare and self.prompts:
            # Encode text prompts
            with torch.no_grad():
                self.encoded_prompts = self.text_encoder.encode(self.prompts)
                logger.info(f"Encoded {len(self.prompts)} prompts")

        # Set random seed
        if seed != self.seed:
            self.seed = seed
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)

        return None

    def __call__(self, *args, **kwargs):
        """Generate video frames.

        This is a placeholder for the actual generation logic.
        The full implementation would include:
        1. Latent noise initialization
        2. Chunk-wise autoregressive rollout
        3. KV cache management
        4. VAE decoding to pixels

        Returns:
            Generated video frames
        """
        # Placeholder - actual implementation would generate video frames
        logger.warning("Self-Forcing generation not yet fully implemented")
        return None

    def to(self, device=None, dtype=None):
        """Move pipeline to device/dtype."""
        if device is not None:
            self.device = device
            self.generator = self.generator.to(device=device)
            self.text_encoder = self.text_encoder.to(device=device)
            self.vae = self.vae.to(device=device)

        if dtype is not None:
            self.dtype = dtype
            self.generator = self.generator.to(dtype=dtype)
            self.text_encoder = self.text_encoder.to(dtype=dtype)
            self.vae = self.vae.to(dtype=dtype)

        return self
