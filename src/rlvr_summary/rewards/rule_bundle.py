"""Main rule bundle reward system implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseRule, RuleEvaluationResult
from .rules import (
    EntityOverlapRule,
    FluencyRule,
    LengthConstraintRule,
    NumberConsistencyRule,
    ProfanityDetectionRule,
)
from .fenice import FENICEScorer


class RuleBundleRewardSystem:
    """Main orchestrator for rule-based reward system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the rule bundle system.

        Args:
            config: Configuration dictionary containing rule weights and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

        # Initialize rules
        self.rules: Dict[str, BaseRule] = {}
        self._setup_rules()

        # Validate configuration
        self._validate_config()

        self.logger.info(
            f"RuleBundleRewardSystem initialized with {len(self.rules)} rules"
        )

    def _setup_rules(self) -> None:
        """Setup all rules based on configuration."""
        weights = self.config.get("weights", {})

        # Length constraint rule
        if "length_constraint" in weights:
            self.rules["length_constraint"] = LengthConstraintRule(
                weight=weights["length_constraint"],
                config=self.config.get("length", {}),
            )

        # Entity overlap rule
        if "entity_overlap" in weights:
            self.rules["entity_overlap"] = EntityOverlapRule(
                weight=weights["entity_overlap"], config=self.config.get("entity", {})
            )

        # Number consistency rule
        if "number_consistency" in weights:
            self.rules["number_consistency"] = NumberConsistencyRule(
                weight=weights["number_consistency"],
                config=self.config.get("numbers", {}),
            )

        # Profanity detection rule
        if "profanity_penalty" in weights:
            self.rules["profanity_penalty"] = ProfanityDetectionRule(
                weight=weights["profanity_penalty"],
                config=self.config.get("profanity", {}),
            )

        # Fluency rule
        if "fluency" in weights:
            self.rules["fluency"] = FluencyRule(
                weight=weights["fluency"], config=self.config.get("fluency", {})
            )
        
        # FENICE factual consistency rule
        if "fenice_factual_consistency" in weights:
            self.rules["fenice_factual_consistency"] = FENICEScorer(
                weight=weights["fenice_factual_consistency"], 
                config=self.config.get("fenice", {})
            )

    def _validate_config(self) -> None:
        """Validate configuration and rule weights."""
        weights = self.config.get("weights", {})

        if not weights:
            self.logger.warning("No rule weights specified in config")
            return

        # Check weight sum
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(
                f"Rule weights sum to {total_weight:.3f}, expected 1.0. "
                "Scores may not be properly normalized."
            )

        # Check for rules with weights but no implementation
        for rule_name in weights:
            if rule_name not in self.rules:
                self.logger.warning(
                    f"Rule '{rule_name}' has weight but no implementation"
                )

    def evaluate(
        self, source: str, summary: str, log_details: bool = False
    ) -> RuleEvaluationResult:
        """Evaluate a summary using all configured rules.

        Args:
            source: Original text
            summary: Generated summary
            log_details: Whether to log detailed evaluation results

        Returns:
            RuleEvaluationResult containing scores and metrics
        """
        if not self.rules:
            self.logger.warning("No rules configured, returning neutral score")
            return RuleEvaluationResult(
                total_score=0.5,
                rule_scores={},
                rule_details={},
                rule_passed={},
                pass_rate=0.0,
            )

        rule_scores = {}
        rule_details = {}
        rule_passed = {}

        # Evaluate each rule
        for rule_name, rule in self.rules.items():
            try:
                result = rule.evaluate(source, summary)
                rule_scores[rule_name] = result["score"]
                rule_details[rule_name] = result["details"]
                rule_passed[rule_name] = result["passed"]

                if log_details:
                    self.logger.info(
                        f"Rule '{rule_name}': score={result['score']:.3f}, "
                        f"passed={result['passed']}, weight={rule.weight:.3f}"
                    )

            except Exception as e:
                self.logger.error(f"Error evaluating rule '{rule_name}': {e}")
                # Assign neutral score on error
                rule_scores[rule_name] = 0.5
                rule_details[rule_name] = {"error": str(e)}
                rule_passed[rule_name] = False

        # Calculate weighted total score
        total_score = 0.0
        total_weight = 0.0

        for rule_name, rule in self.rules.items():
            if rule_name in rule_scores:
                total_score += rule_scores[rule_name] * rule.weight
                total_weight += rule.weight

        # Normalize by total weight
        if total_weight > 0:
            total_score = total_score / total_weight
        else:
            total_score = 0.5

        # Calculate pass rate
        num_passed = sum(1 for passed in rule_passed.values() if passed)
        pass_rate = num_passed / len(rule_passed) if rule_passed else 0.0

        result = RuleEvaluationResult(
            total_score=total_score,
            rule_scores=rule_scores,
            rule_details=rule_details,
            rule_passed=rule_passed,
            pass_rate=pass_rate,
        )

        if log_details:
            self.logger.info(
                f"Overall evaluation: total_score={total_score:.3f}, "
                f"pass_rate={pass_rate:.3f} ({num_passed}/{len(rule_passed)} rules passed)"
            )

        return result

    def evaluate_batch(
        self, sources: List[str], summaries: List[str], log_details: bool = False
    ) -> List[RuleEvaluationResult]:
        """Evaluate a batch of summaries.

        Args:
            sources: List of original texts
            summaries: List of generated summaries
            log_details: Whether to log detailed evaluation results

        Returns:
            List of RuleEvaluationResult objects
        """
        if len(sources) != len(summaries):
            raise ValueError(
                f"Mismatch in batch sizes: {len(sources)} sources, {len(summaries)} summaries"
            )

        results = []
        for i, (source, summary) in enumerate(zip(sources, summaries)):
            if log_details and i % 10 == 0:
                self.logger.info(f"Processing batch item {i+1}/{len(sources)}")

            result = self.evaluate(source, summary, log_details=False)
            results.append(result)

        if log_details:
            self._log_batch_statistics(results)

        return results

    def _log_batch_statistics(self, results: List[RuleEvaluationResult]) -> None:
        """Log statistics for a batch of results."""
        if not results:
            return

        # Calculate averages
        avg_total_score = sum(r.total_score for r in results) / len(results)
        avg_pass_rate = sum(r.pass_rate for r in results) / len(results)

        # Calculate per-rule averages
        rule_names = list(results[0].rule_scores.keys())
        avg_rule_scores = {}
        avg_rule_pass_rates = {}

        for rule_name in rule_names:
            scores = [r.rule_scores.get(rule_name, 0.0) for r in results]
            passed = [r.rule_passed.get(rule_name, False) for r in results]

            avg_rule_scores[rule_name] = sum(scores) / len(scores)
            avg_rule_pass_rates[rule_name] = sum(passed) / len(passed)

        # Log summary
        self.logger.info(f"Batch statistics (n={len(results)}):")
        self.logger.info(f"  Average total score: {avg_total_score:.3f}")
        self.logger.info(f"  Average pass rate: {avg_pass_rate:.3f}")

        for rule_name in rule_names:
            self.logger.info(
                f"  {rule_name}: avg_score={avg_rule_scores[rule_name]:.3f}, "
                f"pass_rate={avg_rule_pass_rates[rule_name]:.3f}"
            )

    def get_rule_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured rules.

        Returns:
            Dictionary with rule information
        """
        info = {}
        for rule_name, rule in self.rules.items():
            info[rule_name] = {
                "weight": rule.weight,
                "threshold": rule.get_threshold(),
                "config": rule.config,
                "class_name": rule.__class__.__name__,
            }
        return info

    def update_rule_weights(self, new_weights: Dict[str, float]) -> None:
        """Update rule weights.

        Args:
            new_weights: Dictionary of new weights for rules
        """
        for rule_name, weight in new_weights.items():
            if rule_name in self.rules:
                self.rules[rule_name].weight = weight
                self.logger.info(f"Updated weight for '{rule_name}': {weight}")
            else:
                self.logger.warning(
                    f"Cannot update weight for unknown rule: '{rule_name}'"
                )

        # Update config
        if "weights" not in self.config:
            self.config["weights"] = {}
        self.config["weights"].update(new_weights)

        # Re-validate
        self._validate_config()


def load_rule_bundle_from_config(
    config_path: Union[str, Path],
) -> RuleBundleRewardSystem:
    """Load rule bundle system from configuration file.

    Args:
        config_path: Path to configuration file (YAML)

    Returns:
        Configured RuleBundleRewardSystem
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return RuleBundleRewardSystem(config)


def create_default_rule_bundle() -> RuleBundleRewardSystem:
    """Create a rule bundle system with default configuration.

    Returns:
        RuleBundleRewardSystem with default settings
    """
    default_config = {
        "weights": {
            "length_constraint": 0.3,
            "entity_overlap": 0.3,
            "number_consistency": 0.2,
            "profanity_penalty": 0.1,
            "fluency": 0.1,
        },
        "length": {
            "min_words": 20,
            "max_words": 100,
            "optimal_range": [30, 80],
            "penalty_factor": 0.5,
        },
        "entity": {
            "min_overlap": 0.3,
            "optimal_overlap": 0.7,
        },
        "numbers": {
            "exact_match_bonus": 1.0,
            "partial_match_bonus": 0.5,
            "mismatch_penalty": -0.5,
        },
        "profanity": {
            "enabled": True,
            "penalty": -1.0,
        },
        "fluency": {
            "enabled": True,
            "min_score": 0.5,
        },
    }

    return RuleBundleRewardSystem(default_config)
