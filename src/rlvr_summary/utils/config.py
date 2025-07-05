"""Configuration utilities for RLVR Summary."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[list] = None,
) -> DictConfig:
    """Load configuration using Hydra.

    Args:
        config_path: Path to config directory (relative to cwd or absolute)
        config_name: Name of the config file (without .yaml extension)
        overrides: List of config overrides

    Returns:
        Loaded configuration as DictConfig
    """
    if config_path is None:
        # Default to configs directory relative to current working directory
        config_path = os.path.join(os.getcwd(), "configs")
    else:
        # Convert relative paths to absolute paths
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.getcwd(), config_path)

    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    # Use initialize_config_dir with absolute path for reliability
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        save_path: Path to save the config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config, save_path)
    logging.info(f"Configuration saved to {save_path}")


def update_config(config: DictConfig, updates: Dict[str, Any]) -> DictConfig:
    """Update configuration with new values.

    Args:
        config: Original configuration
        updates: Dictionary of updates to apply

    Returns:
        Updated configuration
    """
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.update(updates)
    return OmegaConf.create(config_dict)


def resolve_paths(config: DictConfig, base_path: Optional[str] = None) -> DictConfig:
    """Resolve relative paths in configuration.

    Args:
        config: Configuration with potentially relative paths
        base_path: Base path for resolving relative paths

    Returns:
        Configuration with resolved absolute paths
    """
    if base_path is None:
        base_path = os.getcwd()

    # Convert to dict for easier manipulation
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Resolve paths in common sections
    if "paths" in config_dict:
        for key, path in config_dict["paths"].items():
            if isinstance(path, str) and not os.path.isabs(path):
                config_dict["paths"][key] = os.path.abspath(
                    os.path.join(base_path, path)
                )

    return OmegaConf.create(config_dict)


def validate_config(config: DictConfig) -> bool:
    """Validate configuration for required fields and types.

    Args:
        config: Configuration to validate

    Returns:
        True if config is valid, raises exception otherwise
    """
    # Check required top-level sections
    required_sections = ["project", "paths", "wandb"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate project section
    if "name" not in config.project:
        raise ValueError("Missing project.name in config")

    # Validate paths
    required_paths = ["data_dir", "output_dir", "log_dir", "checkpoint_dir"]
    for path_key in required_paths:
        if path_key not in config.paths:
            raise ValueError(f"Missing required path: {path_key}")

    # Validate W&B config
    if "project" not in config.wandb:
        raise ValueError("Missing wandb.project in config")

    logging.info("Configuration validation passed")
    return True


def validate_training_config(
    config: DictConfig, check_files: bool = True
) -> Dict[str, Any]:
    """Enhanced validation for training configuration.

    Args:
        config: Configuration to validate
        check_files: Whether to check for file existence

    Returns:
        Dictionary with validation results and warnings
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "missing_files": [],
        "missing_dependencies": [],
    }

    try:
        # Basic config validation
        validate_config(config)
    except ValueError as e:
        results["errors"].append(str(e))
        results["valid"] = False

    # Check training-specific sections
    if "training" in config:
        training_cfg = config.training

        # Check required training parameters
        required_training_params = ["learning_rate", "batch_size", "max_steps"]
        for param in required_training_params:
            if param not in training_cfg:
                results["warnings"].append(f"Missing training parameter: {param}")

        # Validate parameter ranges
        if "learning_rate" in training_cfg:
            lr = training_cfg.learning_rate
            if not (1e-7 <= lr <= 1e-2):
                results["warnings"].append(
                    f"Learning rate {lr} may be outside typical range [1e-7, 1e-2]"
                )

        if "batch_size" in training_cfg:
            bs = training_cfg.batch_size
            if bs <= 0:
                results["errors"].append(f"Batch size must be positive, got {bs}")
                results["valid"] = False

    # Check model configuration
    if "model" in config:
        model_cfg = config.model
        if "model_name" not in model_cfg:
            results["errors"].append("Missing model.model_name in configuration")
            results["valid"] = False

    # Check data configuration
    if "data" in config:
        data_cfg = config.data
        if "dataset_name" not in data_cfg:
            results["warnings"].append("Missing data.dataset_name in configuration")

    # Check file existence if requested
    if check_files and "paths" in config:
        paths = config.paths

        # Check if data directory exists
        if "data_dir" in paths:
            data_dir = Path(paths.data_dir)
            if not data_dir.exists():
                results["missing_files"].append(f"Data directory: {data_dir}")

    # Check for optional dependencies
    try:
        pass
    except ImportError:
        results["missing_dependencies"].append("torch")

    try:
        pass
    except ImportError:
        results["missing_dependencies"].append("transformers")

    try:
        pass
    except ImportError:
        results["missing_dependencies"].append("verl")

    if results["missing_dependencies"]:
        results["errors"].append("Missing required dependencies for training")
        results["valid"] = False

    return results


def setup_directories(config: DictConfig) -> None:
    """Create necessary directories from config.

    Args:
        config: Configuration containing paths
    """
    for path_key, path_value in config.paths.items():
        if isinstance(path_value, str) and path_key.endswith("_dir"):
            os.makedirs(path_value, exist_ok=True)
            logging.debug(f"Created directory: {path_value}")

    logging.info("Project directories set up")
