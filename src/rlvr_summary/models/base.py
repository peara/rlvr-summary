"""Base model loading and configuration utilities."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy classes for type hints
    PreTrainedModel = object
    PreTrainedTokenizer = object

try:
    from ..utils.config import load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading and configuring models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model loader.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "ModelLoader":
        """Create ModelLoader from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ModelLoader instance
        """
        if not CONFIG_AVAILABLE:
            raise ImportError("Configuration utilities not available")
        
        config = load_config(config_path)
        if hasattr(config, 'model'):
            return cls(config.model)
        else:
            return cls(config)
    
    def load_model_and_tokenizer(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer.
        
        Args:
            model_name: Model name/path override
            device: Device to load model on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required for model loading")
        
        if not TORCH_AVAILABLE:
            raise ImportError("torch package is required for model loading")
        
        # Use provided model name or get from config
        model_name = model_name or self.config.get("model_name")
        if not model_name:
            raise ValueError("Model name must be provided in config or as argument")
        
        self.logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.get("trust_remote_code", True),
        )
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": self.config.get("trust_remote_code", True),
            "torch_dtype": self._get_torch_dtype(),
            "device_map": self.config.get("device_map", "auto"),
        }
        
        # Add quantization settings
        if self.config.get("load_in_8bit", False):
            model_kwargs["load_in_8bit"] = True
        elif self.config.get("load_in_4bit", False):
            model_kwargs["load_in_4bit"] = True
        
        # Override with provided kwargs
        model_kwargs.update(kwargs)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device if specified
        if device and not self.config.get("device_map"):
            model = model.to(device)
        
        self.logger.info(f"Model loaded successfully on device: {model.device}")
        
        return model, tokenizer
    
    def _get_torch_dtype(self):
        """Get torch dtype from config.
        
        Returns:
            torch.dtype or string
        """
        if not TORCH_AVAILABLE:
            return "auto"
        
        dtype_str = self.config.get("torch_dtype", "auto")
        
        if dtype_str == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        elif dtype_str == "float16":
            return torch.float16
        elif dtype_str == "float32":
            return torch.float32
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        else:
            self.logger.warning(f"Unknown dtype {dtype_str}, using auto")
            return torch.float16 if torch.cuda.is_available() else torch.float32
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration.
        
        Returns:
            Generation configuration dictionary
        """
        generation_config = self.config.get("generation", {})
        
        # Set defaults
        defaults = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,  # Will be set from tokenizer
        }
        
        # Merge with config
        for key, default_value in defaults.items():
            if key not in generation_config:
                generation_config[key] = default_value
        
        return generation_config


def load_model_from_config(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Load model and tokenizer from configuration.
    
    Args:
        config_path: Path to configuration file (uses default if None)
        model_name: Model name override
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, generation_config)
    """
    if not CONFIG_AVAILABLE:
        raise ImportError("Configuration utilities not available")
    
    if config_path:
        loader = ModelLoader.from_config_file(config_path)
    else:
        # Load from default hydra config
        config = load_config(config_name="config")
        loader = ModelLoader(config.model if hasattr(config, 'model') else {})
    
    model, tokenizer = loader.load_model_and_tokenizer(
        model_name=model_name,
        device=device,
    )
    
    generation_config = loader.get_generation_config()
    if generation_config.get("pad_token_id") is None:
        generation_config["pad_token_id"] = tokenizer.pad_token_id
    
    return model, tokenizer, generation_config