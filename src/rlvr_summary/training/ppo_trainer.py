"""PPO training loop implementation using VERL Ray architecture."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoTokenizer
    # VERL imports - following the Ray architecture from VERL documentation
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager, get_custom_reward_fn
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
    from verl.single_controller.ray import RayWorkerGroup

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
        config: Dict[str, Any],
        wandb_logger=None,
    ):
        """Initialize VERL Ray PPO training loop.

        Args:
            config: Training configuration
            wandb_logger: W&B logger for experiment tracking
        """
        if not TORCH_AVAILABLE or not VERL_AVAILABLE:
            raise ImportError(f"PyTorch and VERL are required: {missing_deps}")

        self.config = config
        self.wandb_logger = wandb_logger
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

        # VERL components
        self.trainer = None
        self.tokenizer = None
        self.reward_manager = None
        self.val_reward_manager = None

        # Paths
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
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
        self.logger.info("Setting up VERL reward functions using existing reward system...")
        
        # Use our existing VERL-compatible reward function
        reward_file_path = Path(__file__).parent.parent / "rewards" / "verl_reward.py"
        
        if not reward_file_path.exists():
            raise FileNotFoundError(f"VERL reward function not found: {reward_file_path}")
        
        # Configure VERL to use our reward function
        reward_config = {
            "custom_reward_function": {
                "path": str(reward_file_path),
                "name": "compute_score"
            },
            "data": {
                "reward_fn_key": "ground_truth"  # VERL expects this field
            }
        }
        
        # Get the custom reward function using VERL's pattern
        self.reward_manager = get_custom_reward_fn(reward_config)
        self.val_reward_manager = get_custom_reward_fn(reward_config)
        
        self.logger.info(f"Successfully configured VERL to use reward function: {reward_file_path}")



    def setup_workers_and_resources(self):
        """Set up VERL worker roles and resource allocation."""
        self.logger.info("Setting up VERL workers and resource pools...")
        
        # Define role to worker mapping (using FSDP backend)
        role_worker_mapping = {
            Role.ActorRollout: ActorRolloutRefWorker,
            Role.Critic: CriticWorker,
            Role.RefPolicy: ActorRolloutRefWorker
        }
        
        # Add reward model worker if enabled
        if self.config.get("reward_model", {}).get("enable", False):
            role_worker_mapping[Role.RewardModel] = RewardModelWorker
        
        # Define resource pool - single pool for all roles (co-location)
        global_pool_id = 'global_pool'
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
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
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
        
        # Ensure config is in proper VERL format
        verl_config = self._ensure_verl_config_format(self.config)
        
        self.logger.info("Initializing VERL RayPPOTrainer...")
        
        # Create the VERL trainer
        self.trainer = RayPPOTrainer(
            config=verl_config,
            tokenizer=self.tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=self.reward_manager,
            val_reward_fn=self.val_reward_manager
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
            
            train_data.append({
                "prompt": prompt,
                "data_source": "cnn_dailymail",
                "id": processed_item.get("id", ""),
                # Include reference for potential reward computation
                "ground_truth": processed_item.get("highlights", processed_item.get("summary", ""))
            })

        # Save as parquet file for VERL
        data_dir = Path(self.config.get("data_dir", "./data/processed"))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = data_dir / "train_data.parquet"
        df = pd.DataFrame(train_data)
        df.to_parquet(parquet_path, index=False)
        
        self.logger.info(f"Training data saved to {parquet_path}: {len(train_data)} samples")
        
        # Update config with data path for VERL
        self.config["data"] = {
            "train_files": [str(parquet_path)],
            "val_files": [str(parquet_path)],  # Use same for validation for now
        }

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

    def _ensure_verl_config_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure config is in proper format for VERL RayPPOTrainer.
        
        Args:
            config: Input configuration dictionary
            
        Returns:
            Properly formatted VERL configuration
        """
        # Default VERL configuration structure
        verl_config = {
            # Model configuration
            "model": {
                "model_name": config.get("model_name", "distilgpt2"),
                "trust_remote_code": config.get("trust_remote_code", False),
            },
            
            # Training hyperparameters
            "trainer": {
                "total_epochs": config.get("total_epochs", 1),
                "rollout_batch_size": config.get("rollout_batch_size", 16),
                "n_gpus_per_node": config.get("n_gpus_per_node", 1),
                "nnodes": config.get("nnodes", 1),
            },
            
            # PPO algorithm configuration
            "algorithm": {
                "kl_ctrl": {
                    "kl_coef": config.get("kl_coef", 0.1),
                    "adaptive_kl": config.get("adaptive_kl", False),
                },
                "rollout_config": {
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 0.9),
                    "max_new_tokens": config.get("max_new_tokens", 256),
                },
                "ppo_mini_batch_size": config.get("ppo_mini_batch_size", 4),
                "ppo_epochs": config.get("ppo_epochs", 4),
                "clip_range": config.get("clip_range", 0.2),
                "clip_range_value": config.get("clip_range_value", 0.2),
            },
            
            # Data configuration
            "data": config.get("data", {}),
            
            # Reward model configuration
            "reward_model": {
                "enable": config.get("enable_reward_model", False),
                "model_name": config.get("reward_model_name", ""),
            },
            
            # Actor-Rollout-Reference worker configuration
            "actor_rollout_ref": {
                "actor": {
                    "strategy": config.get("strategy", "fsdp"),
                    "optim": {
                        "lr": config.get("learning_rate", 1e-5),
                        "beta1": config.get("beta1", 0.9),
                        "beta2": config.get("beta2", 0.95),
                        "eps": config.get("eps", 1e-5),
                        "weight_decay": config.get("weight_decay", 0.1),
                    },
                },
                "rollout": {
                    "log_prob_micro_batch_size": config.get("log_prob_micro_batch_size", 4),
                    "tensor_model_parallel_size": config.get("tensor_model_parallel_size", 1),
                    "pipeline_model_parallel_size": config.get("pipeline_model_parallel_size", 1),
                },
                "ref": {
                    "log_prob_micro_batch_size": config.get("ref_log_prob_micro_batch_size", 4),
                },
            },
            
            # Critic worker configuration
            "critic": {
                "strategy": config.get("strategy", "fsdp"),
                "optim": {
                    "lr": config.get("critic_learning_rate", 1e-5),
                    "beta1": config.get("beta1", 0.9),
                    "beta2": config.get("beta2", 0.95),
                    "eps": config.get("eps", 1e-5),
                    "weight_decay": config.get("weight_decay", 0.1),
                },
            },
        }
        
        # Merge with any additional config provided
        for key, value in config.items():
            if key not in verl_config:
                verl_config[key] = value
        
        return verl_config

    def _create_custom_reward_function_file(self):
        """Create a custom reward function file for VERL.
        
        VERL expects a Python file with a function that takes specific parameters:
        - data_source: dataset name
        - solution_str: generated response
        - ground_truth: reference/ground truth
        - extra_info: additional information
        """
        reward_file_path = self.checkpoint_dir / "custom_reward.py"
        
        # Create the reward function file content
        reward_function_code = '''
"""Custom reward function for RLVR summary generation."""

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward score for generated summary.
    
    Args:
        data_source: Name of the dataset
        solution_str: Generated summary text
        ground_truth: Reference summary or article text
        extra_info: Additional information (optional)
        
    Returns:
        float: Reward score between 0 and 1
    """
    import re
    
    def calculate_length_reward(summary_text, min_length=10, max_length=500):
        """Calculate reward based on summary length."""
        length = len(summary_text.split())
        
        if length < min_length:
            return 0.1  # Too short
        elif length > max_length:
            return 0.3  # Too long
        else:
            # Optimal length range gets higher reward
            if 20 <= length <= 200:
                return 1.0
            elif 10 <= length <= 300:
                return 0.8
            else:
                return 0.5
    
    def calculate_content_reward(summary_text):
        """Calculate reward based on content quality."""
        # Basic content quality checks
        score = 0.0
        
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', summary_text.strip())
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(valid_sentences) >= 1:
            score += 0.3
        if len(valid_sentences) >= 2:
            score += 0.2
            
        # Check for capitalization
        if summary_text and summary_text[0].isupper():
            score += 0.1
            
        # Check for proper ending
        if summary_text.strip().endswith(('.', '!', '?')):
            score += 0.1
            
        # Avoid repetition
        words = summary_text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            score += min(0.3, uniqueness * 0.5)
            
        return min(1.0, score)
    
    # Basic validation
    if not solution_str or not isinstance(solution_str, str):
        return 0.0
        
    if len(solution_str.strip()) == 0:
        return 0.0
    
    # Calculate component rewards
    length_reward = calculate_length_reward(solution_str)
    content_reward = calculate_content_reward(solution_str)
    
    # Combine rewards with weights
    total_reward = (
        0.4 * length_reward +
        0.6 * content_reward
    )
    
    return float(max(0.0, min(1.0, total_reward)))
'''
        
        # Write the reward function to file
        with open(reward_file_path, 'w') as f:
            f.write(reward_function_code)
            
        self.logger.info(f"Created custom reward function file: {reward_file_path}")

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
    if hasattr(config, 'training'):
        training_config = config.training
        if hasattr(training_config, '__dict__'):
            # Convert OmegaConf to dict if needed
            training_config = dict(training_config)
    else:
        training_config = {}

    # Set up W&B if enabled
    wandb_logger = None
    wandb_config = getattr(config, 'wandb', {})
    if hasattr(wandb_config, '__dict__'):
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
