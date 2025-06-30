"""Weights & Biases integration for experiment tracking."""

import os
import logging
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available. Install with: pip install wandb")


class WandbLogger:
    """Weights & Biases logger for RLVR Summary experiments."""
    
    def __init__(
        self,
        project: str = "rlvr-summary",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: str = "",
        enabled: bool = True,
    ):
        """Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            name: Experiment name
            config: Configuration dictionary to log
            tags: List of tags for the experiment
            notes: Experiment notes
            enabled: Whether to enable W&B logging
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if not self.enabled:
            if not WANDB_AVAILABLE:
                logging.warning("W&B not available. Logging disabled.")
            else:
                logging.info("W&B logging disabled.")
            return
            
        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags or [],
            notes=notes,
            resume="allow",
        )
        
        logging.info(f"W&B run initialized: {self.run.url}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or self.run is None:
            return
            
        wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration to W&B.
        
        Args:
            config: Configuration dictionary
        """
        if not self.enabled or self.run is None:
            return
            
        wandb.config.update(config)
    
    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        description: str = "",
    ) -> None:
        """Log an artifact to W&B.
        
        Args:
            artifact_path: Path to the artifact
            artifact_name: Name of the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            description: Description of the artifact
        """
        if not self.enabled or self.run is None:
            return
            
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description,
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self.run is None:
            return
            
        wandb.finish()
        logging.info("W&B run finished.")


def setup_wandb_from_config(wandb_config: Dict[str, Any]) -> WandbLogger:
    """Setup W&B logger from configuration.
    
    Args:
        wandb_config: W&B configuration dictionary
        
    Returns:
        Configured WandbLogger instance
    """
    return WandbLogger(
        project=wandb_config.get("project", "rlvr-summary"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("name"),
        config=wandb_config.get("config"),
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes", ""),
        enabled=wandb_config.get("enabled", True),
    )


def is_wandb_available() -> bool:
    """Check if W&B is available and properly configured."""
    if not WANDB_AVAILABLE:
        return False
        
    # Check if API key is configured
    api_key = os.getenv("WANDB_API_KEY") or wandb.api.api_key
    return api_key is not None


def test_wandb_connection() -> bool:
    """Test W&B connection without starting a run.
    
    Returns:
        True if connection is successful, False otherwise
    """
    if not is_wandb_available():
        return False
        
    try:
        # Test connection by checking API
        api = wandb.Api()
        # Try to access user info (will fail if not authenticated)
        _ = api.viewer
        return True
    except Exception as e:
        logging.error(f"W&B connection test failed: {e}")
        return False