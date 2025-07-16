"""Integration utilities for the reward system with training loops."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .rule_bundle import RuleBundleRewardSystem, load_rule_bundle_from_config
from .combined import CombinedRewardSystem, create_combined_reward_system

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
        use_combined_system: bool = True,
    ):
        """Initialize reward system integrator.

        Args:
            config: Reward system configuration dictionary
            config_path: Path to reward system configuration file
            wandb_logger: Optional W&B logger for metrics tracking
            use_combined_system: Whether to use FENICE+Rules combined system (default True)
        """
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")
        self.wandb_logger = wandb_logger
        self.use_combined_system = use_combined_system

        # Load reward system
        if use_combined_system:
            # Use combined FENICE + rule-based system
            if config_path:
                # TODO: Implement config loading for combined system
                self.reward_system = create_combined_reward_system()
                self.logger.info("Using combined reward system (config file support TBD)")
            elif config:
                # Extract weights and configuration
                fenice_weight = config.get("fenice_weight", 0.7)
                rule_weight = config.get("rule_weight", 0.3)
                fenice_enabled = config.get("fenice_enabled", True)
                fenice_config = config.get("fenice_config", {})
                rule_config = config.get("rule_config", {})
                
                self.reward_system = CombinedRewardSystem(
                    fenice_weight=fenice_weight,
                    rule_weight=rule_weight,
                    fenice_config=fenice_config,
                    rule_config=rule_config
                )
            else:
                # Use default combined system
                self.reward_system = create_combined_reward_system()
                self.logger.info("Using default combined reward system")
        else:
            # Fall back to rule-based only system
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

        # Handle different result types
        if hasattr(result, 'total_score'):
            # Combined or rule-based result
            total_score = result.total_score
        else:
            # Fallback for unexpected result format
            total_score = float(result) if isinstance(result, (int, float)) else 0.5

        # Track metrics
        self._update_metrics(result)

        # Log to W&B if available
        if self.wandb_logger and self.wandb_logger.enabled:
            metrics = self._get_metrics_from_result(result)
            if step is not None:
                metrics["step"] = step
            self.wandb_logger.log(metrics, step=step)

        return total_score

    def _get_metrics_from_result(self, result) -> Dict[str, float]:
        """Extract metrics from result object."""
        if hasattr(result, 'get_metrics'):
            return result.get_metrics()
        elif hasattr(result, 'total_score'):
            # Basic metrics for rule-based result
            return {"reward/total_score": result.total_score}
        else:
            # Fallback
            return {"reward/total_score": float(result) if isinstance(result, (int, float)) else 0.5}

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

        return [self._extract_total_score(result) for result in results]

    def _extract_total_score(self, result) -> float:
        """Extract total score from result object."""
        if hasattr(result, 'total_score'):
            return result.total_score
        else:
            return float(result) if isinstance(result, (int, float)) else 0.5

    def _update_metrics(self, result) -> None:
        """Update cumulative metrics tracking."""
        self._total_evaluations += 1

        # Extract scores based on result type
        if hasattr(result, 'total_score'):
            total_score = result.total_score
        else:
            total_score = float(result) if isinstance(result, (int, float)) else 0.5

        # Update total score
        if "total_score" not in self._cumulative_scores:
            self._cumulative_scores["total_score"] = []
        self._cumulative_scores["total_score"].append(total_score)

        # Handle combined system results
        if hasattr(result, 'fenice_score'):
            # Combined system
            if "fenice_score" not in self._cumulative_scores:
                self._cumulative_scores["fenice_score"] = []
            self._cumulative_scores["fenice_score"].append(result.fenice_score)
            
            if "rule_score" not in self._cumulative_scores:
                self._cumulative_scores["rule_score"] = []
            self._cumulative_scores["rule_score"].append(result.rule_score)
            
            # Update pass rates
            if "total_pass_rate" not in self._cumulative_pass_rates:
                self._cumulative_pass_rates["total_pass_rate"] = []
            self._cumulative_pass_rates["total_pass_rate"].append(float(result.passed))
            
        elif hasattr(result, 'rule_scores'):
            # Rule-based system
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
        
        # Calculate total scores
        total_scores = [self._extract_total_score(r) for r in results]
        metrics = {
            "reward/batch_size": batch_size,
            "reward/batch_avg_score": sum(total_scores) / batch_size,
        }

        # Handle different result types
        if results and hasattr(results[0], 'fenice_score'):
            # Combined system metrics
            fenice_scores = [r.fenice_score for r in results]
            rule_scores = [r.rule_score for r in results]
            pass_rates = [float(r.passed) for r in results]
            
            metrics.update({
                "reward/batch_avg_fenice_score": sum(fenice_scores) / batch_size,
                "reward/batch_avg_rule_score": sum(rule_scores) / batch_size,
                "reward/batch_avg_pass_rate": sum(pass_rates) / batch_size,
            })
            
        elif results and hasattr(results[0], 'rule_scores'):
            # Rule-based system metrics  
            pass_rates = [r.pass_rate for r in results]
            metrics["reward/batch_avg_pass_rate"] = sum(pass_rates) / batch_size
            
            # Add per-rule batch averages
            if results[0].rule_scores:
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
