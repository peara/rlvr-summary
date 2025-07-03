"""PPO training loop implementation using VERL."""

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
    # VERL imports - following the architecture from VERL documentation
    from verl import PPOTrainer, PPOConfig
    from verl.trainer.ppo import functional_reward

    TORCH_AVAILABLE = True
    VERL_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    VERL_AVAILABLE = False
    missing_deps = str(e)

from ..evaluation import EvaluationPipeline
from ..models import load_model_from_config
from ..rewards import create_reward_function
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class PPOTrainingLoop:
    """PPO training loop for RLVR summary model using VERL."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        wandb_logger=None,
    ):
        """Initialize VERL training loop.

        Args:
            config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            wandb_logger: W&B logger for experiment tracking
        """
        if not TORCH_AVAILABLE or not VERL_AVAILABLE:
            raise ImportError(f"PyTorch and VERL are required: {missing_deps}")

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
        if not TORCH_AVAILABLE or not VERL_AVAILABLE:
            raise ImportError(f"PyTorch and VERL are required: {missing_deps}")

        self.logger.info("Setting up VERL PPO training loop...")

        # Load model and tokenizer if not provided
        if self.model is None or self.tokenizer is None:
            self.logger.info("Loading model and tokenizer...")
            self.model, self.tokenizer, generation_config = load_model_from_config()
            self.generation_config = generation_config

        # Ensure tokenizer is properly configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create VERL PPO config
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
            # VERL-specific configuration
            max_new_tokens=self.config.get("max_new_tokens", 256),
            temperature=self.config.get("temperature", 0.7),
            top_k=self.config.get("top_k", 50),
            top_p=self.config.get("top_p", 0.95),
        )

        # Initialize VERL PPO trainer
        self.logger.info("Initializing VERL PPO trainer...")

        # Memory optimization: Move model to device early
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

        # Create reference model (copy of policy model)
        self.logger.info("Creating reference model...")
        ref_model = type(self.model)(self.model.config)
        ref_model.load_state_dict(self.model.state_dict())
        ref_model.eval()
        if torch.cuda.is_available():
            ref_model = ref_model.cuda()

        # Create VERL PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            policy_model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
        )

        # Set up reward function using VERL's functional_reward interface
        self.logger.info("Setting up reward function...")
        self.reward_function = create_reward_function(wandb_logger=self.wandb_logger)
        
        # Wrap our reward function for VERL's functional_reward interface
        self.verl_reward_function = self._create_verl_reward_function()

        # Set up evaluation pipeline
        self.evaluation_pipeline = EvaluationPipeline(wandb_logger=self.wandb_logger)

        self.logger.info("VERL PPO training loop setup complete!")

    def _create_verl_reward_function(self):
        """Create a VERL-compatible reward function using functional_reward.
        
        This follows VERL's functional_reward interface pattern.
        """
        def verl_reward_fn(prompts: List[str], responses: List[str]) -> List[float]:
            """VERL-compatible reward function.
            
            Args:
                prompts: List of input prompts
                responses: List of generated responses
                
            Returns:
                List of reward scores
            """
            rewards = []
            for prompt, response in zip(prompts, responses):
                # Extract article from prompt (our prompts contain the article)
                article = self._extract_article_from_prompt(prompt)
                # Compute reward using our existing reward function
                reward = self.reward_function(article, response)
                rewards.append(reward)
            return rewards
        
        # Use VERL's functional_reward decorator
        return functional_reward(verl_reward_fn)

    def _create_minimal_dataset(self) -> Dataset:
        """Create a minimal dataset for PPOTrainer initialization.
        
        This creates a small sample dataset to satisfy VERL's initialization requirements.
        The actual training dataset will be loaded and set later in the train method.
        
        Returns:
            Minimal Dataset object with tokenized prompts for VERL PPOTrainer
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer must be initialized before creating minimal dataset")
            
        # Create one minimal sample
        minimal_data = [{
            "id": "init_sample",
            "article": "This is a minimal initialization sample for the PPOTrainer setup.",
            "summary": "Minimal sample for initialization.",
        }]
        
        return self._convert_to_verl_format(minimal_data)

    def _convert_to_verl_format(self, data: List[Dict[str, str]]) -> Dataset:
        """Convert processed data to VERL PPOTrainer expected format.
        
        Args:
            data: List of processed data samples with 'article', 'summary', 'id' keys
            
        Returns:
            Dataset object with tokenized prompts for VERL PPOTrainer
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer must be initialized before converting to VERL format")
            
        verl_samples = []
        for sample in data:
            # Create standardized prompt
            prompt = f"Summarize the following article:\n\n{sample['article']}\n\nSummary:"
            
            # Tokenize prompt only (not prompt + completion)
            tokenized = self.tokenizer(
                prompt,
                max_length=self.config.get("max_prompt_length", 512),
                truncation=True,
                padding=False,  # VERL handles padding
                return_tensors=None,
            )
            
            verl_samples.append({
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

        # Convert to VERL PPOTrainer format
        self.logger.info("Converting datasets to VERL PPOTrainer format...")
        train_verl_dataset = self._convert_to_verl_format(train_dataset)
        eval_verl_dataset = self._convert_to_verl_format(eval_dataset)

        self.logger.info(
            f"VERL format conversion complete: "
            f"train={len(train_verl_dataset)} samples, eval={len(eval_verl_dataset)} samples"
        )

        return train_verl_dataset, eval_verl_dataset

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
        """Main training loop using VERL PPOTrainer."""
        if not self.ppo_trainer:
            self.setup()

        self.logger.info(f"Starting VERL PPO training for {self.total_steps} steps...")

        # Load datasets (now returns VERL-compatible Dataset objects)
        train_dataset, eval_dataset = self.load_datasets()

        # Set up the training with VERL
        self.ppo_trainer.set_reward_function(self.verl_reward_function)
        self.ppo_trainer.set_dataset(train_dataset)

        # Run VERL training loop
        self.logger.info("Starting VERL PPO training...")
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
