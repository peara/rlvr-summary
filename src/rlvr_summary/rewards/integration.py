"""Integration utilities for the reward system with training loops."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .rule_bundle import RuleBundleRewardSystem, load_rule_bundle_from_config

# Try to import config utilities, but make them optional
try:
    from ..utils.config import load_config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Try to import wandb logger, but make it optional
try:
    from ..utils.wandb_logger import WandbLogger

    WANDB_LOGGER_AVAILABLE = True
except ImportError:
    WANDB_LOGGER_AVAILABLE = False

    # Create dummy WandbLogger class
    class WandbLogger:
        def __init__(self, *args, **kwargs):
            self.enabled = False

        def log(self, *args, **kwargs):
            pass


class RewardSystemIntegrator:
    """Integration wrapper for reward system with training loops."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        """Initialize reward system integrator.

        Args:
            config: Reward system configuration dictionary
            config_path: Path to reward system configuration file
            wandb_logger: Optional W&B logger for metrics tracking
        """
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")
        self.wandb_logger = wandb_logger

        # Load reward system
        if config_path:
            self.reward_system = load_rule_bundle_from_config(config_path)
        elif config:
            self.reward_system = RuleBundleRewardSystem(config)
        else:
            # Try to load from default hydra config
            if CONFIG_AVAILABLE:
                try:
                    cfg = load_config(config_name="config")
                    if hasattr(cfg, "rewards"):
                        reward_config = cfg.rewards
                        if hasattr(reward_config, "_target_"):
                            # Remove hydra target specification
                            reward_dict = {
                                k: v
                                for k, v in reward_config.items()
                                if k != "_target_"
                            }
                        else:
                            reward_dict = dict(reward_config)
                        self.reward_system = RuleBundleRewardSystem(reward_dict)
                    else:
                        self.logger.warning("No reward config found, using default")
                        from .rule_bundle import create_default_rule_bundle

                        self.reward_system = create_default_rule_bundle()
                except Exception as e:
                    self.logger.warning(f"Failed to load config: {e}, using default")
                    from .rule_bundle import create_default_rule_bundle

                    self.reward_system = create_default_rule_bundle()
            else:
                self.logger.warning("Config system not available, using default")
                from .rule_bundle import create_default_rule_bundle

                self.reward_system = create_default_rule_bundle()

        # Initialize metrics tracking
        self._total_evaluations = 0
        self._cumulative_scores = {}
        self._cumulative_pass_rates = {}

        self.logger.info("RewardSystemIntegrator initialized successfully")

    def compute_reward(
        self,
        source: str,
        summary: str,
        log_details: bool = False,
        step: Optional[int] = None,
    ) -> float:
        """Compute reward for a single source-summary pair.

        Args:
            source: Original text
            summary: Generated summary
            log_details: Whether to log detailed evaluation
            step: Optional training step for logging

        Returns:
            Total reward score (0.0 to 1.0)
        """
        result = self.reward_system.evaluate(source, summary, log_details=log_details)

        # Track metrics
        self._update_metrics(result)

        # Log to W&B if available
        if self.wandb_logger and self.wandb_logger.enabled:
            metrics = result.get_metrics()
            if step is not None:
                metrics["step"] = step
            self.wandb_logger.log(metrics, step=step)

        return result.total_score

    def compute_reward_batch(
        self,
        sources: List[str],
        summaries: List[str],
        log_details: bool = False,
        step: Optional[int] = None,
    ) -> List[float]:
        """Compute rewards for a batch of source-summary pairs.

        Args:
            sources: List of original texts
            summaries: List of generated summaries
            log_details: Whether to log detailed evaluation
            step: Optional training step for logging

        Returns:
            List of reward scores
        """
        results = self.reward_system.evaluate_batch(
            sources, summaries, log_details=log_details
        )

        # Track metrics for each result
        for result in results:
            self._update_metrics(result)

        # Log batch statistics to W&B if available
        if self.wandb_logger and self.wandb_logger.enabled and results:
            batch_metrics = self._compute_batch_metrics(results)
            if step is not None:
                batch_metrics["step"] = step
            self.wandb_logger.log(batch_metrics, step=step)

        return [result.total_score for result in results]

    def _update_metrics(self, result) -> None:
        """Update cumulative metrics tracking."""
        self._total_evaluations += 1

        # Update total score
        if "total_score" not in self._cumulative_scores:
            self._cumulative_scores["total_score"] = []
        self._cumulative_scores["total_score"].append(result.total_score)

        # Update rule scores
        for rule_name, score in result.rule_scores.items():
            if rule_name not in self._cumulative_scores:
                self._cumulative_scores[rule_name] = []
            self._cumulative_scores[rule_name].append(score)

        # Update pass rates
        if "total_pass_rate" not in self._cumulative_pass_rates:
            self._cumulative_pass_rates["total_pass_rate"] = []
        self._cumulative_pass_rates["total_pass_rate"].append(result.pass_rate)

        for rule_name, passed in result.rule_passed.items():
            if rule_name not in self._cumulative_pass_rates:
                self._cumulative_pass_rates[rule_name] = []
            self._cumulative_pass_rates[rule_name].append(float(passed))

    def _compute_batch_metrics(self, results) -> Dict[str, float]:
        """Compute batch-level metrics."""
        if not results:
            return {}

        batch_size = len(results)
        metrics = {
            "reward/batch_size": batch_size,
            "reward/batch_avg_score": sum(r.total_score for r in results) / batch_size,
            "reward/batch_avg_pass_rate": sum(r.pass_rate for r in results)
            / batch_size,
        }

        # Add per-rule batch averages
        rule_names = list(results[0].rule_scores.keys())
        for rule_name in rule_names:
            avg_score = (
                sum(r.rule_scores.get(rule_name, 0.0) for r in results) / batch_size
            )
            avg_pass_rate = (
                sum(float(r.rule_passed.get(rule_name, False)) for r in results)
                / batch_size
            )

            metrics[f"reward/batch_{rule_name}_score"] = avg_score
            metrics[f"reward/batch_{rule_name}_pass_rate"] = avg_pass_rate

        return metrics

    def get_cumulative_statistics(self) -> Dict[str, Any]:
        """Get cumulative statistics since initialization.

        Returns:
            Dictionary with cumulative statistics
        """
        if self._total_evaluations == 0:
            return {"total_evaluations": 0}

        stats = {
            "total_evaluations": self._total_evaluations,
            "average_scores": {},
            "average_pass_rates": {},
        }

        # Compute average scores
        for metric_name, scores in self._cumulative_scores.items():
            stats["average_scores"][metric_name] = sum(scores) / len(scores)

        # Compute average pass rates
        for metric_name, pass_rates in self._cumulative_pass_rates.items():
            stats["average_pass_rates"][metric_name] = sum(pass_rates) / len(pass_rates)

        return stats

    def reset_statistics(self) -> None:
        """Reset cumulative statistics."""
        self._total_evaluations = 0
        self._cumulative_scores.clear()
        self._cumulative_pass_rates.clear()
        self.logger.info("Cumulative statistics reset")

    def update_rule_weights(self, new_weights: Dict[str, float]) -> None:
        """Update rule weights in the underlying reward system.

        Args:
            new_weights: Dictionary of new weights for rules
        """
        self.reward_system.update_rule_weights(new_weights)
        self.logger.info(f"Updated rule weights: {new_weights}")

    def get_rule_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about configured rules.

        Returns:
            Dictionary with rule information
        """
        return self.reward_system.get_rule_info()

    def evaluate_milestone_criteria(
        self, target_pass_rate: float = 0.2
    ) -> Dict[str, Any]:
        """Evaluate milestone criteria for rule-pass rate.

        Args:
            target_pass_rate: Target pass rate threshold (default 0.2 for M0 milestone)

        Returns:
            Dictionary with milestone evaluation results
        """
        stats = self.get_cumulative_statistics()

        if stats["total_evaluations"] == 0:
            return {
                "milestone_met": False,
                "current_pass_rate": 0.0,
                "target_pass_rate": target_pass_rate,
                "evaluations": 0,
                "message": "No evaluations performed yet",
            }

        current_pass_rate = stats["average_pass_rates"].get("total_pass_rate", 0.0)
        milestone_met = current_pass_rate >= target_pass_rate

        return {
            "milestone_met": milestone_met,
            "current_pass_rate": current_pass_rate,
            "target_pass_rate": target_pass_rate,
            "evaluations": stats["total_evaluations"],
            "message": (
                f"✅ Milestone achieved! Pass rate {current_pass_rate:.3f} >= {target_pass_rate:.3f}"
                if milestone_met
                else f"❌ Milestone not yet met. Pass rate {current_pass_rate:.3f} < {target_pass_rate:.3f}"
            ),
        }


def create_reward_integrator(
    config_path: Optional[str] = None,
    wandb_logger: Optional[WandbLogger] = None,
) -> RewardSystemIntegrator:
    """Create a reward system integrator with default configuration.

    Args:
        config_path: Optional path to configuration file
        wandb_logger: Optional W&B logger

    Returns:
        Configured RewardSystemIntegrator
    """
    if config_path:
        return RewardSystemIntegrator(
            config_path=config_path, wandb_logger=wandb_logger
        )
    else:
        return RewardSystemIntegrator(wandb_logger=wandb_logger)


def create_reward_function(
    config_path: Optional[str] = None,
    wandb_logger: Optional[WandbLogger] = None,
):
    """Create a simple reward function for use in training loops.

    Args:
        config_path: Optional path to configuration file
        wandb_logger: Optional W&B logger

    Returns:
        Function that takes (source, summary) and returns reward score
    """
    integrator = create_reward_integrator(config_path, wandb_logger)

    def reward_fn(source: str, summary: str) -> float:
        """Simple reward function interface.

        Args:
            source: Original text
            summary: Generated summary

        Returns:
            Reward score (0.0 to 1.0)
        """
        return integrator.compute_reward(source, summary)

    return reward_fn
