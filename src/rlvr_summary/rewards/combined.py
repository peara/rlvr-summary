"""Combined reward system integrating FENICE and rule-based scoring."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseRule, RuleEvaluationResult
from .fenice import FENICEScorer, create_fenice_scorer
from .rule_bundle import RuleBundleRewardSystem, create_default_rule_bundle


class CombinedRewardResult:
    """Container for combined reward evaluation results."""

    def __init__(
        self,
        total_score: float,
        fenice_score: float,
        rule_score: float,
        fenice_weight: float,
        rule_weight: float,
        fenice_details: Dict[str, Any],
        rule_result: RuleEvaluationResult,
        passed: bool,
    ):
        """Initialize combined result.
        
        Args:
            total_score: Final weighted combination score
            fenice_score: FENICE factual consistency score
            rule_score: Rule-based score
            fenice_weight: Weight applied to FENICE score
            rule_weight: Weight applied to rule score
            fenice_details: Detailed FENICE results
            rule_result: Rule-based evaluation result
            passed: Whether combined score meets threshold
        """
        self.total_score = total_score
        self.fenice_score = fenice_score
        self.rule_score = rule_score
        self.fenice_weight = fenice_weight
        self.rule_weight = rule_weight
        self.fenice_details = fenice_details
        self.rule_result = rule_result
        self.passed = passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_score": self.total_score,
            "fenice_score": self.fenice_score,
            "rule_score": self.rule_score,
            "fenice_weight": self.fenice_weight,
            "rule_weight": self.rule_weight,
            "fenice_details": self.fenice_details,
            "rule_details": self.rule_result.to_dict(),
            "passed": self.passed,
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics suitable for tracking/logging."""
        metrics = {
            "reward/total_score": self.total_score,
            "reward/fenice_score": self.fenice_score,
            "reward/rule_score": self.rule_score,
            "reward/fenice_weight": self.fenice_weight,
            "reward/rule_weight": self.rule_weight,
            "reward/combined_passed": float(self.passed),
        }

        # Add FENICE-specific metrics
        if self.fenice_details:
            metrics["reward/fenice_num_claims"] = self.fenice_details.get("num_claims", 0)

        # Add rule-based metrics
        rule_metrics = self.rule_result.get_metrics()
        metrics.update(rule_metrics)

        return metrics


class CombinedRewardSystem:
    """Combined reward system using FENICE + rule-based scoring.
    
    Implements the weighted combination: R = fenice_weight × FENICE + rule_weight × Rules
    """

    def __init__(
        self,
        fenice_weight: float = 0.7,
        rule_weight: float = 0.3,
        fenice_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ):
        """Initialize combined reward system.
        
        Args:
            fenice_weight: Weight for FENICE score (default 0.7)
            rule_weight: Weight for rule-based score (default 0.3)
            fenice_config: Configuration for FENICE scorer
            rule_config: Configuration for rule-based system
            threshold: Threshold for passing combined score
        """
        self.fenice_weight = fenice_weight
        self.rule_weight = rule_weight
        self.threshold = threshold
        
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")
        
        # Validate weights
        total_weight = fenice_weight + rule_weight
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(
                f"Weights sum to {total_weight:.3f}, expected 1.0. "
                "Normalizing weights."
            )
            self.fenice_weight = fenice_weight / total_weight
            self.rule_weight = rule_weight / total_weight
        
        # Initialize FENICE scorer
        fenice_config = fenice_config or {}
        self.fenice_scorer = create_fenice_scorer(
            weight=self.fenice_weight,
            **fenice_config
        )
        
        # Initialize rule-based system
        if rule_config:
            self.rule_system = RuleBundleRewardSystem(rule_config)
        else:
            self.rule_system = create_default_rule_bundle()
        
        self.logger.info(
            f"CombinedRewardSystem initialized: "
            f"FENICE weight={self.fenice_weight:.3f}, "
            f"Rule weight={self.rule_weight:.3f}, "
            f"threshold={self.threshold:.3f}"
        )

    def evaluate(
        self, 
        source: str, 
        summary: str, 
        log_details: bool = False
    ) -> CombinedRewardResult:
        """Evaluate combined reward for source-summary pair.
        
        Args:
            source: Original text
            summary: Generated summary
            log_details: Whether to log detailed evaluation
            
        Returns:
            CombinedRewardResult with detailed scoring
        """
        # Get FENICE score
        fenice_result = self.fenice_scorer.evaluate(source, summary)
        fenice_score = fenice_result["score"]
        fenice_details = fenice_result["details"]
        
        # Get rule-based score
        rule_result = self.rule_system.evaluate(source, summary, log_details=log_details)
        rule_score = rule_result.total_score
        
        # Compute weighted combination
        total_score = (self.fenice_weight * fenice_score) + (self.rule_weight * rule_score)
        
        # Determine if passed
        passed = total_score >= self.threshold
        
        # Create combined result
        combined_result = CombinedRewardResult(
            total_score=total_score,
            fenice_score=fenice_score,
            rule_score=rule_score,
            fenice_weight=self.fenice_weight,
            rule_weight=self.rule_weight,
            fenice_details=fenice_details,
            rule_result=rule_result,
            passed=passed,
        )
        
        if log_details:
            self.logger.info(
                f"Combined evaluation: total={total_score:.3f} "
                f"(FENICE={fenice_score:.3f}×{self.fenice_weight:.3f} + "
                f"Rules={rule_score:.3f}×{self.rule_weight:.3f}), "
                f"passed={passed}"
            )
            
            # Log FENICE details
            num_claims = fenice_details.get("num_claims", 0)
            self.logger.info(f"FENICE: {num_claims} claims extracted")
        
        return combined_result

    def evaluate_batch(
        self,
        sources: List[str],
        summaries: List[str],
        log_details: bool = False,
    ) -> List[CombinedRewardResult]:
        """Evaluate combined rewards for multiple source-summary pairs.
        
        Args:
            sources: List of source texts
            summaries: List of summaries
            log_details: Whether to log detailed evaluation
            
        Returns:
            List of CombinedRewardResult objects
        """
        if len(sources) != len(summaries):
            raise ValueError("Sources and summaries must have same length")
        
        results = []
        for i, (source, summary) in enumerate(zip(sources, summaries)):
            result = self.evaluate(source, summary, log_details=log_details)
            results.append(result)
        
        if log_details and results:
            avg_total = sum(r.total_score for r in results) / len(results)
            avg_fenice = sum(r.fenice_score for r in results) / len(results)
            avg_rule = sum(r.rule_score for r in results) / len(results)
            
            self.logger.info(
                f"Batch evaluation: {len(results)} items, "
                f"avg_total={avg_total:.3f}, "
                f"avg_fenice={avg_fenice:.3f}, "
                f"avg_rule={avg_rule:.3f}"
            )
        
        return results

    def update_weights(self, fenice_weight: float, rule_weight: float) -> None:
        """Update scoring weights.
        
        Args:
            fenice_weight: New weight for FENICE score
            rule_weight: New weight for rule score
        """
        total_weight = fenice_weight + rule_weight
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(
                f"Weights sum to {total_weight:.3f}, expected 1.0. "
                "Normalizing weights."
            )
            fenice_weight = fenice_weight / total_weight
            rule_weight = rule_weight / total_weight
        
        self.fenice_weight = fenice_weight
        self.rule_weight = rule_weight
        
        self.logger.info(
            f"Updated weights: FENICE={self.fenice_weight:.3f}, "
            f"Rules={self.rule_weight:.3f}"
        )

    def configure_fenice(self, **config) -> None:
        """Update FENICE configuration.
        
        Args:
            **config: Configuration parameters for FENICE scorer
        """
        for key, value in config.items():
            if hasattr(self.fenice_scorer.config, key):
                self.fenice_scorer.config[key] = value
            else:
                self.fenice_scorer.config[key] = value
        
        self.logger.info(f"Updated FENICE config: {config}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the combined system.
        
        Returns:
            Dictionary with system configuration and status
        """
        return {
            "type": "CombinedRewardSystem",
            "fenice_weight": self.fenice_weight,
            "rule_weight": self.rule_weight,
            "threshold": self.threshold,
            "fenice_config": self.fenice_scorer.config,
            "rule_system_info": self.rule_system.get_rule_info(),
        }


def create_combined_reward_system(
    fenice_weight: float = 0.7,
    rule_weight: float = 0.3,
    threshold: float = 0.5,
    **kwargs
) -> CombinedRewardSystem:
    """Create a combined reward system with default configuration.
    
    Args:
        fenice_weight: Weight for FENICE score (default 0.7)
        rule_weight: Weight for rule-based score (default 0.3)  
        threshold: Threshold for passing combined score
        **kwargs: Additional configuration
        
    Returns:
        Configured CombinedRewardSystem
    """
    return CombinedRewardSystem(
        fenice_weight=fenice_weight,
        rule_weight=rule_weight,
        threshold=threshold,
        **kwargs
    )