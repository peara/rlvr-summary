"""Configuration utilities for RLVR Summary."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


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
        # Default to configs directory relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = str(project_root / "configs")
    
    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize(config_path=config_path, version_base=None):
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