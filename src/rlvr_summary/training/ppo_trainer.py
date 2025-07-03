"""PPO training loop implementation using HuggingFace TRL."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from trl import PPOConfig, PPOTrainer

    TORCH_AVAILABLE = True
    TRL_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TRL_AVAILABLE = False
    missing_deps = str(e)

from ..evaluation import EvaluationPipeline
from ..models import load_model_from_config
from ..rewards import create_reward_function
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class PPOTrainingLoop:
    """PPO training loop for RLVR summary model."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        wandb_logger=None,
    ):
        """Initialize PPO training loop.

        Args:
            config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            wandb_logger: W&B logger for experiment tracking
        """
        if not TORCH_AVAILABLE or not TRL_AVAILABLE:
            raise ImportError(f"PyTorch and TRL are required: {missing_deps}")

        self.config = config
        self.wandb_logger = wandb_logger
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

        # Initialize components
        self.model = model
        self.tokenizer = tokenizer
        self.ppo_trainer = None
        self.reward_function = None
        self.evaluation_pipeline = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.total_steps = config.get("max_steps", 10000)

        # Paths
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        """Set up training components."""
        self.logger.info("Setting up PPO training loop...")

        # Load model and tokenizer if not provided
        if self.model is None or self.tokenizer is None:
            self.logger.info("Loading model and tokenizer...")
            self.model, self.tokenizer, generation_config = load_model_from_config()
            self.generation_config = generation_config

        # Ensure tokenizer is properly configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create PPO config
        ppo_config = PPOConfig(
            output_dir=str(self.checkpoint_dir),
            learning_rate=self.config.get("learning_rate", 1.41e-5),
            batch_size=self.config.get("batch_size", 16),
            mini_batch_size=self.config.get("mini_batch_size", 4),
            gradient_accumulation_steps=self.config.get(
                "gradient_accumulation_steps", 1
            ),
            num_ppo_epochs=self.config.get("ppo_epochs", 4),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            seed=self.config.get("seed", 42),
            report_to="wandb" if self.wandb_logger else None,
            exp_name=self.config.get("exp_name", "ppo_training"),
            response_length=self.config.get("max_new_tokens", 256),
            temperature=self.config.get("temperature", 0.7),
        )

        # Initialize PPO trainer
        self.logger.info("Initializing PPO trainer...")

        # Memory optimization: Move model to device early to avoid device placement issues
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            torch.cuda.empty_cache()  # Clear cache before creating copies

        # Create reference model (copy of policy model)
        self.logger.info("Creating reference model...")
        ref_model = type(self.model)(self.model.config)
        ref_model.load_state_dict(self.model.state_dict())
        ref_model.eval()  # Reference model should be in eval mode
        if torch.cuda.is_available():
            ref_model = ref_model.cuda()

        # Create a simple reward model (we'll use our rule-based rewards)
        # For now, we'll use the same model architecture but this could be different
        self.logger.info("Creating reward model...")
        reward_model = type(self.model)(self.model.config)
        reward_model.load_state_dict(self.model.state_dict())
        reward_model.eval()
        if torch.cuda.is_available():
            reward_model = reward_model.cuda()
            torch.cuda.empty_cache()  # Clear cache after loading models

        # Create minimal initial dataset for PPOTrainer initialization
        # We'll load the full dataset later in the train method
        self.logger.info("Creating minimal initial dataset for PPOTrainer...")
        initial_dataset = self._create_minimal_dataset()
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            processing_class=self.tokenizer,
            policy=self.model,
            ref_policy=ref_model,
            reward_model=reward_model,
            train_dataset=initial_dataset,
            value_model=self.model,
        )

        # Set up reward function
        self.logger.info("Setting up reward function...")
        self.reward_function = create_reward_function(wandb_logger=self.wandb_logger)

        # Set up evaluation pipeline
        self.evaluation_pipeline = EvaluationPipeline(wandb_logger=self.wandb_logger)

        self.logger.info("PPO training loop setup complete!")

    def _create_minimal_dataset(self) -> Dataset:
        """Create a minimal dataset for PPOTrainer initialization.
        
        This creates a small sample dataset to satisfy TRL's initialization requirements.
        The actual training dataset will be loaded and set later in the train method.
        
        Returns:
            Minimal Dataset object with tokenized prompts for TRL PPOTrainer
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer must be initialized before creating minimal dataset")
            
        # Create one minimal sample
        minimal_data = [{
            "id": "init_sample",
            "article": "This is a minimal initialization sample for the PPOTrainer setup.",
            "summary": "Minimal sample for initialization.",
        }]
        
        return self._convert_to_ppo_format(minimal_data)



    def _convert_to_ppo_format(self, data: List[Dict[str, str]]) -> Dataset:
        """Convert processed data to TRL PPOTrainer expected format.
        
        Args:
            data: List of processed data samples with 'article', 'summary', 'id' keys
            
        Returns:
            Dataset object with tokenized prompts for TRL PPOTrainer
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer must be initialized before converting to PPO format")
            
        ppo_samples = []
        for sample in data:
            # Create standardized prompt
            prompt = f"Summarize the following article:\n\n{sample['article']}\n\nSummary:"
            
            # Tokenize prompt only (not prompt + completion)
            tokenized = self.tokenizer(
                prompt,
                max_length=self.config.get("max_prompt_length", 512),
                truncation=True,
                padding=False,  # TRL handles padding
                return_tensors=None,
            )
            
            ppo_samples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "query": prompt,  # For reward computation
                "reference": sample["summary"],  # For reward computation
                "article": sample["article"],  # Preserve for reward function
                "id": sample["id"],  # Preserve for tracking
            })
        
        return Dataset.from_list(ppo_samples)

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load training and evaluation datasets using the proper data pipeline.

        Returns:
            Tuple of (train_dataset, eval_dataset) as TRL-compatible Dataset objects
        """
        from ..data import BatchProcessor, CNNDMLoader, DataValidator, TextPreprocessor
        from ..data.batch_processor import create_data_pipeline

        # Configure data pipeline components
        train_loader = CNNDMLoader(
            data_path=self.config.get("data_path"),
            split="train",
            max_samples=self.config.get("train_size", 1000),
        )

        eval_loader = CNNDMLoader(
            data_path=self.config.get("data_path"),
            split="validation",
            max_samples=self.config.get("eval_size", 100),
        )

        # Set up preprocessing and validation
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

        # Process evaluation data
        self.logger.info("Loading and processing evaluation dataset...")
        eval_pipeline_result = create_data_pipeline(
            loader=eval_loader,
            preprocessor=preprocessor,
            validator=validator,
            batch_processor=batch_processor,
        )

        # Extract processed data and convert to expected format
        train_dataset = []
        for item in train_pipeline_result["data"]:
            processed_item = item.get("processed", item.get("original", {}))
            # Ensure we have the expected field names
            train_dataset.append(
                {
                    "id": processed_item.get("id", ""),
                    "article": processed_item.get("article", ""),
                    "summary": processed_item.get(
                        "highlights", processed_item.get("summary", "")
                    ),
                }
            )

        eval_dataset = []
        for item in eval_pipeline_result["data"]:
            processed_item = item.get("processed", item.get("original", {}))
            eval_dataset.append(
                {
                    "id": processed_item.get("id", ""),
                    "article": processed_item.get("article", ""),
                    "summary": processed_item.get(
                        "highlights", processed_item.get("summary", "")
                    ),
                }
            )

        # Log dataset statistics
        train_stats = train_pipeline_result["statistics"]
        eval_stats = eval_pipeline_result["statistics"]

        self.logger.info(
            f"Training dataset loaded: {train_stats['total_loaded']} samples, "
            f"validation rate: {(train_stats['total_validated'] - train_stats.get('validation_failures', 0)) / max(train_stats['total_validated'], 1):.1%}"
        )
        self.logger.info(
            f"Evaluation dataset loaded: {eval_stats['total_loaded']} samples, "
            f"validation rate: {(eval_stats['total_validated'] - eval_stats.get('validation_failures', 0)) / max(eval_stats['total_validated'], 1):.1%}"
        )

        # Log additional pipeline info if available
        if self.wandb_logger:
            self.wandb_logger.log(
                {
                    "data/train_samples": len(train_dataset),
                    "data/eval_samples": len(eval_dataset),
                    "data/train_validation_rate": (
                        train_stats["total_validated"]
                        - train_stats.get("validation_failures", 0)
                    )
                    / max(train_stats["total_validated"], 1),
                    "data/eval_validation_rate": (
                        eval_stats["total_validated"]
                        - eval_stats.get("validation_failures", 0)
                    )
                    / max(eval_stats["total_validated"], 1),
                    "data/train_processing_time": train_stats.get("processing_time", 0),
                    "data/eval_processing_time": eval_stats.get("processing_time", 0),
                }
            )

        # Convert to TRL PPOTrainer format
        self.logger.info("Converting datasets to TRL PPOTrainer format...")
        train_trl_dataset = self._convert_to_ppo_format(train_dataset)
        eval_trl_dataset = self._convert_to_ppo_format(eval_dataset)

        self.logger.info(
            f"TRL format conversion complete: "
            f"train={len(train_trl_dataset)} samples, eval={len(eval_trl_dataset)} samples"
        )

        return train_trl_dataset, eval_trl_dataset

    def prepare_batch(self, examples: List[Dict[str, str]]) -> Dict[str, List]:
        """Prepare a batch for training.

        Args:
            examples: List of dataset examples

        Returns:
            Batch dictionary with tokenized inputs
        """
        prompts = []
        references = []

        for example in examples:
            # Simple prompt template
            prompt = (
                f"Summarize the following article:\n\n{example['article']}\n\nSummary:"
            )
            prompts.append(prompt)
            references.append(example["summary"])

        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.get("max_prompt_length", 512),
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "references": references,
            "prompts": prompts,
        }

    def generate_summaries(self, batch: Dict[str, Any]) -> List[str]:
        """Generate summaries for a batch.

        Args:
            batch: Batch with input_ids and attention_mask

        Returns:
            List of generated summaries
        """
        # Ensure batch tensors are on the correct device
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            # Generate responses using the model, not the trainer
            response_tensors = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode summaries (remove prompt part)
        summaries = []
        for i, response in enumerate(response_tensors):
            prompt_length = len(input_ids[i])
            summary_tokens = response[prompt_length:]
            summary = self.tokenizer.decode(summary_tokens, skip_special_tokens=True)
            summaries.append(summary.strip())

        return summaries

    def compute_rewards(
        self,
        prompts: List[str],
        summaries: List[str],
        references: List[str],
    ) -> List[float]:
        """Compute rewards for generated summaries.

        Args:
            prompts: Input prompts
            summaries: Generated summaries
            references: Reference summaries

        Returns:
            List of reward scores
        """
        rewards = []

        for prompt, summary, reference in zip(prompts, summaries, references):
            # Extract article from prompt
            article = prompt.replace(
                "Summarize the following article:\n\n", ""
            ).replace("\n\nSummary:", "")

            # Compute rule-based reward
            rule_reward = self.reward_function(article, summary)

            # Optionally compute ROUGE bonus
            rouge_scores = self.evaluation_pipeline.evaluate_single(summary, reference)
            rouge_bonus = rouge_scores["rouge1"]["f1"] * 0.1  # Small ROUGE bonus

            total_reward = rule_reward + rouge_bonus
            rewards.append(total_reward)

        return rewards

    def compute_batch_rewards(self, prompts: List[str], summaries: List[str]) -> List[float]:
        """Integrate with TRL's reward computation cycle.
        
        Args:
            prompts: List of input prompts
            summaries: List of generated summaries
            
        Returns:
            List of reward scores
        """
        rewards = []
        for prompt, summary in zip(prompts, summaries):
            # Extract article from standardized prompt
            article = self._extract_article_from_prompt(prompt)
            reward = self.reward_function(article, summary)
            rewards.append(reward)
        return rewards

    def _extract_article_from_prompt(self, prompt: str) -> str:
        """Extract article text from standardized prompt format.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Extracted article text
        """
        # Remove the prompt template to get the original article
        article = prompt.replace(
            "Summarize the following article:\n\n", ""
        ).replace("\n\nSummary:", "")
        return article

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one training step using manual PPO implementation.

        NOTE: The TRL PPOTrainer doesn't support individual training steps,
        so this is a simplified manual implementation for demonstration.

        Args:
            batch: Training batch

        Returns:
            Dictionary with training metrics
        """
        # Generate summaries
        summaries = self.generate_summaries(batch)

        # Compute rewards
        rewards = self.compute_rewards(
            batch["prompts"],
            summaries,
            batch["references"],
        )

        # Calculate evaluation metrics
        rouge_scores = self.evaluation_pipeline.evaluate_batch(
            summaries,
            batch["references"],
            step=self.step,
        )

        # Combine metrics (simplified - not actually doing PPO updates here)
        metrics = {
            "train/step": self.step,
            "train/epoch": self.epoch,
            "train/avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "train/max_reward": max(rewards) if rewards else 0.0,
            "train/min_reward": min(rewards) if rewards else 0.0,
            **{f"train/{k}": v for k, v in rouge_scores.items()},
        }

        return metrics

    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint.

        Args:
            metrics: Optional metrics to save with checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-step-{self.step}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "config": self.config,
            "metrics": metrics or {},
        }

        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def evaluate(self, eval_dataset: List[Dict[str, str]]) -> Dict[str, float]:
        """Run evaluation on evaluation dataset.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        self.logger.info("Running evaluation...")

        all_summaries = []
        all_references = []
        all_rewards = []

        # Process in batches
        batch_size = self.config.get("eval_batch_size", 8)
        for i in range(0, len(eval_dataset), batch_size):
            batch_examples = eval_dataset[i : i + batch_size]
            batch = self.prepare_batch(batch_examples)

            # Generate summaries
            summaries = self.generate_summaries(batch)

            # Compute rewards
            rewards = self.compute_rewards(
                batch["prompts"],
                summaries,
                batch["references"],
            )

            all_summaries.extend(summaries)
            all_references.extend(batch["references"])
            all_rewards.extend(rewards)

        # Calculate metrics
        rouge_scores = self.evaluation_pipeline.evaluate_batch(
            all_summaries,
            all_references,
            step=self.step,
            log_to_wandb=False,
        )

        eval_metrics = {
            "eval/avg_reward": sum(all_rewards) / len(all_rewards)
            if all_rewards
            else 0.0,
            "eval/max_reward": max(all_rewards) if all_rewards else 0.0,
            "eval/min_reward": min(all_rewards) if all_rewards else 0.0,
            **{f"eval/{k}": v for k, v in rouge_scores.items()},
        }

        # Log to W&B
        if self.wandb_logger and hasattr(self.wandb_logger, "log"):
            self.wandb_logger.log(eval_metrics, step=self.step)

        self.logger.info(f"Evaluation complete: {eval_metrics}")
        return eval_metrics

    def train(self):
        """Main training loop using TRL PPOTrainer."""
        if not self.ppo_trainer:
            self.setup()

        self.logger.info(f"Starting PPO training for {self.total_steps} steps...")

        # Load datasets (now returns TRL-compatible Dataset objects)
        train_dataset, eval_dataset = self.load_datasets()

        # Update the PPO trainer's dataset with the TRL-formatted dataset
        self.ppo_trainer.train_dataset = train_dataset

        # Set up training configuration for TRL
        self.ppo_trainer.args.max_steps = self.total_steps
        self.ppo_trainer.args.logging_steps = self.config.get("logging_steps", 10)
        self.ppo_trainer.args.save_steps = self.config.get("save_steps", 1000)
        self.ppo_trainer.args.eval_steps = self.config.get("eval_steps", 500)

        # Use TRL's built-in training loop
        self.logger.info("Starting TRL PPO training...")
        self.ppo_trainer.train()

        # Final checkpoint
        self.save_checkpoint()
        self.logger.info("Training completed!")




def train_ppo_model(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> PPOTrainingLoop:
    """Train PPO model with configuration.

    Args:
        config_path: Path to training configuration
        experiment_name: Experiment name for W&B

    Returns:
        Trained PPO training loop instance
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config(config_name="config")

    # Extract training config
    training_config = config.training if hasattr(config, "training") else {}

    # Set up W&B if enabled
    wandb_logger = None
    if config.wandb.get("enabled", True):
        try:
            from ..utils.wandb_logger import WandbLogger

            wandb_logger = WandbLogger(
                project=config.wandb.get("project", "rlvr-summary"),
                name=experiment_name or f"ppo-training-{int(time.time())}",
                config=training_config,
                tags=["ppo", "baseline"] + config.wandb.get("tags", []),
                notes=config.wandb.get("notes", "PPO baseline training"),
            )
        except ImportError:
            logger.warning("W&B not available, training without logging")

    # Create and run training loop
    training_loop = PPOTrainingLoop(
        config=training_config,
        wandb_logger=wandb_logger,
    )

    training_loop.train()

    if wandb_logger and hasattr(wandb_logger, "finish"):
        wandb_logger.finish()

    return training_loop
