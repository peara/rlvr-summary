"""PPO training loop implementation using VERL Ray architecture."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    from omegaconf import DictConfig, OmegaConf
    from transformers import AutoTokenizer
    from verl.single_controller.ray import RayWorkerGroup

    # VERL imports - following the Ray architecture from VERL documentation
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.trainer.ppo.reward import get_custom_reward_fn, load_reward_manager
    from verl.workers.fsdp_workers import (
        ActorRolloutRefWorker,
        CriticWorker,
        RewardModelWorker,
    )

    TORCH_AVAILABLE = True
    VERL_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    VERL_AVAILABLE = False
    missing_deps = str(e)

from ..evaluation import EvaluationPipeline
from ..rewards import create_reward_function
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class VERLPPOTrainingLoop:
    """VERL PPO training loop using Ray architecture for RLVR summary model."""

    def __init__(
        self,
        config: Union[Dict[str, Any], DictConfig],
        wandb_logger=None,
    ):
        """Initialize VERL Ray PPO training loop.

        Args:
            config: Training configuration (dict or DictConfig)
            wandb_logger: W&B logger for experiment tracking
        """
        if not TORCH_AVAILABLE or not VERL_AVAILABLE:
            raise ImportError(f"PyTorch and VERL are required: {missing_deps}")

        # Convert dict to DictConfig if necessary for consistent handling
        if isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            self.config = config

        self.wandb_logger = wandb_logger
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

        # VERL components
        self.trainer = None
        self.tokenizer = None
        self.reward_manager = None
        self.val_reward_manager = None

        # Paths
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup_tokenizer(self):
        """Set up tokenizer for VERL training."""
        model_name = self.config.get("model_name", "distilgpt2")
        self.logger.info(f"Loading tokenizer: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_reward_function(self):
        """Set up reward functions for VERL using our existing reward system."""
        self.logger.info(
            "Setting up VERL reward functions using existing reward system..."
        )

        # Use our existing VERL-compatible reward function
        reward_file_path = Path(__file__).parent.parent / "rewards" / "verl_reward.py"

        if not reward_file_path.exists():
            raise FileNotFoundError(
                f"VERL reward function not found: {reward_file_path}"
            )

        # Configure VERL to use our reward function
        reward_config = {
            "custom_reward_function": {
                "path": str(reward_file_path),
                "name": "compute_score",
            },
            "data": {"reward_fn_key": "ground_truth"},  # VERL expects this field
        }

        # Get the custom reward function using VERL's pattern
        self.reward_manager = get_custom_reward_fn(reward_config)
        self.val_reward_manager = get_custom_reward_fn(reward_config)

        self.logger.info(
            f"Successfully configured VERL to use reward function: {reward_file_path}"
        )

    def setup_workers_and_resources(self):
        """Set up VERL worker roles and resource allocation."""
        self.logger.info("Setting up VERL workers and resource pools...")

        # Define role to worker mapping (using FSDP backend)
        role_worker_mapping = {
            Role.ActorRollout: ActorRolloutRefWorker,
            Role.Critic: CriticWorker,
            Role.RefPolicy: ActorRolloutRefWorker,
        }

        # Add reward model worker if enabled
        if self.config.get("reward_model", {}).get("enable", False):
            role_worker_mapping[Role.RewardModel] = RewardModelWorker

        # Define resource pool - single pool for all roles (co-location)
        global_pool_id = "global_pool"
        n_gpus_per_node = self.config.get("n_gpus_per_node", 1)
        nnodes = self.config.get("nnodes", 1)

        resource_pool_spec = {
            global_pool_id: [n_gpus_per_node] * nnodes,
        }

        # Map all roles to the global resource pool
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        if self.config.get("reward_model", {}).get("enable", False):
            mapping[Role.RewardModel] = global_pool_id

        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        return role_worker_mapping, resource_pool_manager

    def setup_trainer(self):
        """Set up the VERL RayPPOTrainer."""
        if not self.tokenizer:
            self.setup_tokenizer()

        if not self.reward_manager:
            self.setup_reward_function()

        # Set up workers and resources
        role_worker_mapping, resource_pool_manager = self.setup_workers_and_resources()

        # Update config with model path and reward function
        self._update_verl_config()

        self.logger.info("Initializing VERL RayPPOTrainer...")

        # Create the VERL trainer - pass config directly as OmegaConf object
        self.trainer = RayPPOTrainer(
            config=self.config,
            tokenizer=self.tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=self.reward_manager,
            val_reward_fn=self.val_reward_manager,
        )

    def prepare_data(self):
        """Prepare training data in VERL format.

        VERL expects data to be preprocessed and stored in parquet files,
        then loaded with RLHFDataset.
        """
        self.logger.info("Preparing data for VERL training...")

        # Import pandas for parquet file creation
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for VERL data preparation")

        # For now, we'll use the existing data pipeline and convert to parquet
        from ..data import BatchProcessor, CNNDMLoader, DataValidator, TextPreprocessor
        from ..data.batch_processor import create_data_pipeline

        # Configure data pipeline components
        train_loader = CNNDMLoader(
            data_path=self.config.get("data_path"),
            split="train",
            max_samples=self.config.get("train_size", 1000),
        )

        preprocessor = TextPreprocessor(
            use_spacy=self.config.get("use_spacy", False),
            max_length=self.config.get("max_article_length", 10000),
        )

        validator = DataValidator(
            min_article_length=self.config.get("min_article_length", 50),
            max_article_length=self.config.get("max_article_length", 10000),
            min_summary_length=self.config.get("min_summary_length", 10),
            max_summary_length=self.config.get("max_summary_length", 500),
        )

        batch_processor = BatchProcessor(
            batch_size=self.config.get("data_batch_size", 32),
            max_workers=self.config.get("data_workers", 2),
        )

        # Process training data
        self.logger.info("Loading and processing training dataset...")
        train_pipeline_result = create_data_pipeline(
            loader=train_loader,
            preprocessor=preprocessor,
            validator=validator,
            batch_processor=batch_processor,
        )

        # Convert to VERL format
        train_data = []
        for item in train_pipeline_result["data"]:
            processed_item = item.get("processed", item.get("original", {}))

            # Create standardized prompt for VERL
            article = processed_item.get("article", "")
            prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"

            train_data.append(
                {
                    "prompt": prompt,
                    "data_source": "cnn_dailymail",
                    "id": processed_item.get("id", ""),
                    # Include reference for potential reward computation
                    "ground_truth": processed_item.get(
                        "highlights", processed_item.get("summary", "")
                    ),
                }
            )

        # Save as parquet file for VERL
        data_dir = Path(self.config.get("data_dir", "./data/processed"))
        data_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = data_dir / "train_data.parquet"
        df = pd.DataFrame(train_data)
        df.to_parquet(parquet_path, index=False)

        self.logger.info(
            f"Training data saved to {parquet_path}: {len(train_data)} samples"
        )

        # Update config with data path for VERL
        OmegaConf.set_struct(self.config, False)  # Allow adding new keys
        if "data" not in self.config:
            self.config.data = {}
        self.config.data.train_files = [str(parquet_path)]
        self.config.data.val_files = [
            str(parquet_path)
        ]  # Use same for validation for now
        OmegaConf.set_struct(self.config, True)  # Re-enable struct mode

    def train(self):
        """Main training loop using VERL RayPPOTrainer."""
        self.logger.info("Starting VERL PPO training...")

        # Prepare data
        self.prepare_data()

        # Set up trainer
        self.setup_trainer()

        # Initialize workers
        self.logger.info("Initializing VERL workers...")
        self.trainer.init_workers()

        # Start training
        self.logger.info("Starting VERL PPO training loop...")
        self.trainer.fit()

        self.logger.info("VERL PPO training completed!")

    def _update_verl_config(self):
        """Update VERL config with runtime values like model path and reward function."""
        # Update model path if provided through legacy config
        model_name = self.config.get("model_name")
        if model_name and hasattr(self.config, "actor_rollout_ref"):
            self.config.actor_rollout_ref.model.path = model_name

        # Update reward function path
        reward_file_path = Path(__file__).parent.parent / "rewards" / "verl_reward.py"
        if hasattr(self.config, "custom_reward_function"):
            self.config.custom_reward_function.path = str(reward_file_path)

        self.logger.info("Updated VERL config with runtime values")


def train_ppo_model(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> VERLPPOTrainingLoop:
    """Train PPO model with VERL configuration.

    Args:
        config_path: Path to training configuration
        experiment_name: Experiment name for W&B

    Returns:
        Trained VERL PPO training loop instance
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config(config_name="config")

    # Extract training config and ensure it's a dictionary
    if hasattr(config, "training"):
        training_config = config.training
        if hasattr(training_config, "__dict__"):
            # Convert OmegaConf to dict if needed
            training_config = dict(training_config)
    else:
        training_config = {}

    # Set up W&B if enabled
    wandb_logger = None
    wandb_config = getattr(config, "wandb", {})
    if hasattr(wandb_config, "__dict__"):
        wandb_config = dict(wandb_config)

    if wandb_config.get("enabled", True):
        try:
            from ..utils.wandb_logger import WandbLogger

            wandb_logger = WandbLogger(
                project=wandb_config.get("project", "rlvr-summary"),
                name=experiment_name or f"verl-ppo-training-{int(time.time())}",
                config=training_config,
                tags=["verl", "ppo", "baseline"] + wandb_config.get("tags", []),
                notes=wandb_config.get("notes", "VERL PPO baseline training"),
            )
        except ImportError:
            logger.warning("W&B not available, training without logging")

    # Create and run training loop
    training_loop = VERLPPOTrainingLoop(
        config=training_config,
        wandb_logger=wandb_logger,
    )

    training_loop.train()

    if wandb_logger and hasattr(wandb_logger, "finish"):
        wandb_logger.finish()

    return training_loop


# Legacy alias for backward compatibility
PPOTrainingLoop = VERLPPOTrainingLoop
