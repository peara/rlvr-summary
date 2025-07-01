"""PPO training loop implementation using HuggingFace TRL."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from trl import PPOConfig, PPOTrainer
    TORCH_AVAILABLE = True
    TRL_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TRL_AVAILABLE = False
    missing_deps = str(e)

from ..models import load_model_from_config
from ..rewards import create_reward_function
from ..evaluation import EvaluationPipeline
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
        
        # Create PPO config
        ppo_config = PPOConfig(
            model_name=self.config.get("model_name", "meta-llama/Llama-3.2-1B"),
            learning_rate=self.config.get("learning_rate", 1.41e-5),
            batch_size=self.config.get("batch_size", 16),
            mini_batch_size=self.config.get("mini_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            ppo_epochs=self.config.get("ppo_epochs", 4),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            seed=self.config.get("seed", 42),
            log_with="wandb" if self.wandb_logger else None,
        )
        
        # Initialize PPO trainer
        self.logger.info("Initializing PPO trainer...")
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # Set up reward function
        self.logger.info("Setting up reward function...")
        self.reward_function = create_reward_function(wandb_logger=self.wandb_logger)
        
        # Set up evaluation pipeline
        self.evaluation_pipeline = EvaluationPipeline(wandb_logger=self.wandb_logger)
        
        self.logger.info("PPO training loop setup complete!")
    
    def load_datasets(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Load training and evaluation datasets.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        from ..data import create_data_loader
        
        data_loader = create_data_loader()
        
        # Load datasets
        train_dataset = data_loader.load_data("train", size=self.config.get("train_size", 1000))
        eval_dataset = data_loader.load_data("eval", size=self.config.get("eval_size", 100))
        
        # Log dataset statistics
        train_stats = data_loader.get_dataset_stats(train_dataset)
        eval_stats = data_loader.get_dataset_stats(eval_dataset)
        
        self.logger.info(f"Training dataset: {train_stats}")
        self.logger.info(f"Evaluation dataset: {eval_stats}")
        
        return train_dataset, eval_dataset
    
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
            prompt = f"Summarize the following article:\n\n{example['article']}\n\nSummary:"
            prompts.append(prompt)
            references.append(example['summary'])
        
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
        with torch.no_grad():
            # Generate responses
            response_tensors = self.ppo_trainer.generate(
                batch["input_ids"],
                max_new_tokens=self.config.get("max_new_tokens", 256),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode summaries (remove prompt part)
        summaries = []
        for i, response in enumerate(response_tensors):
            prompt_length = len(batch["input_ids"][i])
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
            article = prompt.replace("Summarize the following article:\n\n", "").replace("\n\nSummary:", "")
            
            # Compute rule-based reward
            rule_reward = self.reward_function(article, summary)
            
            # Optionally compute ROUGE bonus
            rouge_scores = self.evaluation_pipeline.evaluate_single(summary, reference)
            rouge_bonus = rouge_scores["rouge1"]["f1"] * 0.1  # Small ROUGE bonus
            
            total_reward = rule_reward + rouge_bonus
            rewards.append(total_reward)
        
        return rewards
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one training step.
        
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
        
        # Convert rewards to tensors
        reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]
        
        # Update model with PPO
        stats = self.ppo_trainer.step(
            batch["input_ids"],
            [torch.cat([batch["input_ids"][i], torch.tensor(self.tokenizer.encode(summaries[i]))])
             for i in range(len(summaries))],
            reward_tensors,
        )
        
        # Calculate evaluation metrics
        rouge_scores = self.evaluation_pipeline.evaluate_batch(
            summaries,
            batch["references"],
            step=self.step,
        )
        
        # Combine metrics
        metrics = {
            "train/step": self.step,
            "train/epoch": self.epoch,
            "train/avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "train/max_reward": max(rewards) if rewards else 0.0,
            "train/min_reward": min(rewards) if rewards else 0.0,
            **{f"train/{k}": v for k, v in stats.items()},
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
            batch_examples = eval_dataset[i:i + batch_size]
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
            "eval/avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "eval/max_reward": max(all_rewards) if all_rewards else 0.0,
            "eval/min_reward": min(all_rewards) if all_rewards else 0.0,
            **{f"eval/{k}": v for k, v in rouge_scores.items()},
        }
        
        # Log to W&B
        if self.wandb_logger and hasattr(self.wandb_logger, 'log'):
            self.wandb_logger.log(eval_metrics, step=self.step)
        
        self.logger.info(f"Evaluation complete: {eval_metrics}")
        return eval_metrics
    
    def train(self):
        """Main training loop."""
        if not self.ppo_trainer:
            self.setup()
        
        self.logger.info(f"Starting PPO training for {self.total_steps} steps...")
        
        # Load datasets
        train_dataset, eval_dataset = self.load_datasets()
        
        # Training configuration
        batch_size = self.config.get("batch_size", 16)
        logging_steps = self.config.get("logging_steps", 10)
        save_steps = self.config.get("save_steps", 1000)
        eval_steps = self.config.get("eval_steps", 500)
        
        start_time = time.time()
        
        while self.step < self.total_steps:
            # Sample batch
            batch_start = (self.step * batch_size) % len(train_dataset)
            batch_end = min(batch_start + batch_size, len(train_dataset))
            batch_examples = train_dataset[batch_start:batch_end]
            
            if len(batch_examples) < batch_size:
                # Wrap around dataset
                remaining = batch_size - len(batch_examples)
                batch_examples.extend(train_dataset[:remaining])
            
            # Prepare batch
            batch = self.prepare_batch(batch_examples)
            
            # Training step
            try:
                metrics = self.train_step(batch)
                
                # Log metrics
                if self.step % logging_steps == 0:
                    elapsed = time.time() - start_time
                    metrics["train/elapsed_time"] = elapsed
                    metrics["train/steps_per_second"] = self.step / elapsed if elapsed > 0 else 0.0
                    
                    if self.wandb_logger and hasattr(self.wandb_logger, 'log'):
                        self.wandb_logger.log(metrics, step=self.step)
                    
                    self.logger.info(f"Step {self.step}: {metrics}")
                
                # Save checkpoint
                if self.step % save_steps == 0 and self.step > 0:
                    self.save_checkpoint(metrics)
                
                # Run evaluation
                if self.step % eval_steps == 0 and self.step > 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    
                    # Check for milestone
                    rouge1_f1 = eval_metrics.get("eval/rouge1_f1", 0.0)
                    if rouge1_f1 >= 0.2:  # 20% milestone
                        self.logger.info(f"🎉 Milestone achieved! ROUGE-1 F1: {rouge1_f1:.3f}")
                
                self.step += 1
                
            except Exception as e:
                self.logger.error(f"Error in training step {self.step}: {e}")
                self.step += 1
                continue
        
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
    training_config = config.training if hasattr(config, 'training') else {}
    
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
    
    if wandb_logger and hasattr(wandb_logger, 'finish'):
        wandb_logger.finish()
    
    return training_loop