"""
Model Manager for FENICE components.
Implements singleton pattern to prevent model reloading.
"""

import logging
from typing import Optional

import torch


class ModelManager:
    """Singleton manager for FENICE models to prevent reloading."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)

            # Model instances (loaded lazily)
            self._claim_extractor = None
            self._nli_aligner = None
            self._coref_model = None

            # Configuration tracking
            self._claim_extractor_config = None
            self._nli_aligner_config = None
            self._coref_model_config = None

            self.logger.info("ModelManager initialized")
            ModelManager._initialized = True

    def get_claim_extractor(self, batch_size: int = 256, device: str = None):
        """Get or create claim extractor model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = (batch_size, device)

        # Check if we need to create/recreate the model
        if self._claim_extractor is None or self._claim_extractor_config != config:
            self.logger.info(
                f"Loading ClaimExtractor (device={device}, batch_size={batch_size})"
            )

            # Clean up old model if exists
            if self._claim_extractor is not None:
                del self._claim_extractor
                torch.cuda.empty_cache()

            # Import and create new model
            from .claim_extractor.claim_extractor import ClaimExtractor

            self._claim_extractor = ClaimExtractor(batch_size=batch_size, device=device)
            self._claim_extractor_config = config

            self.logger.info("ClaimExtractor loaded and cached")

        return self._claim_extractor

    def get_nli_aligner(
        self, batch_size: int = 256, device: str = None, max_length: int = 1024
    ):
        """Get or create NLI aligner model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = (batch_size, device, max_length)

        # Check if we need to create/recreate the model
        if self._nli_aligner is None or self._nli_aligner_config != config:
            self.logger.info(
                f"Loading NLIAligner (device={device}, batch_size={batch_size}, max_length={max_length})"
            )

            # Clean up old model if exists
            if self._nli_aligner is not None:
                del self._nli_aligner
                torch.cuda.empty_cache()

            # Import and create new model
            from .nli.nli_aligner import NLIAligner

            self._nli_aligner = NLIAligner(
                batch_size=batch_size, device=device, max_length=max_length
            )
            self._nli_aligner_config = config

            self.logger.info("NLIAligner loaded and cached")

        return self._nli_aligner

    def get_coref_model(
        self, batch_size: int = 1, device: str = None, load_model: bool = True
    ):
        """Get or create coreference resolution model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = (batch_size, device, load_model)

        # Check if we need to create/recreate the model
        if self._coref_model is None or self._coref_model_config != config:
            self.logger.info(
                f"Loading CoreferenceResolution (device={device}, batch_size={batch_size})"
            )

            # Clean up old model if exists
            if self._coref_model is not None:
                if hasattr(self._coref_model, "model"):
                    del self._coref_model.model
                del self._coref_model
                torch.cuda.empty_cache()

            # Import and create new model
            from .coreference_resolution.coreference_resolution import (
                CoreferenceResolution,
            )

            self._coref_model = CoreferenceResolution(
                batch_size=batch_size, device=device, load_model=load_model
            )
            self._coref_model_config = config

            self.logger.info("CoreferenceResolution loaded and cached")

        return self._coref_model

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self.logger.info("Clearing model cache...")

        if self._claim_extractor is not None:
            del self._claim_extractor
            self._claim_extractor = None
            self._claim_extractor_config = None

        if self._nli_aligner is not None:
            del self._nli_aligner
            self._nli_aligner = None
            self._nli_aligner_config = None

        if self._coref_model is not None:
            if hasattr(self._coref_model, "model"):
                del self._coref_model.model
            del self._coref_model
            self._coref_model = None
            self._coref_model_config = None

        torch.cuda.empty_cache()
        self.logger.info("Model cache cleared")

    def get_memory_info(self):
        """Get information about loaded models and memory usage."""
        info = {
            "claim_extractor_loaded": self._claim_extractor is not None,
            "nli_aligner_loaded": self._nli_aligner is not None,
            "coref_model_loaded": self._coref_model is not None,
        }

        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved()

        return info


# Global instance
model_manager = ModelManager()
