"""Base classes for reward system components."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRule(ABC):
    """Abstract base class for all reward rules."""

    def __init__(self, weight: float, config: Optional[Dict[str, Any]] = None):
        """Initialize base rule.

        Args:
            weight: Weight of this rule in the overall score
            config: Rule-specific configuration
        """
        self.weight = weight
        self.config = config or {}
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate the rule for given source and summary.

        Args:
            source: Original text
            summary: Generated summary

        Returns:
            Dictionary containing:
                - score: Rule score (0.0 to 1.0)
                - details: Additional details about the evaluation
                - passed: Whether the rule passed (score >= threshold)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this rule."""

    def get_threshold(self) -> float:
        """Get the passing threshold for this rule."""
        return self.config.get("threshold", 0.5)

    def log_evaluation(self, result: Dict[str, Any], source: str, summary: str) -> None:
        """Log evaluation results.

        Args:
            result: Evaluation result from evaluate()
            source: Original text
            summary: Generated summary
        """
        self.logger.debug(
            f"{self.name} evaluation: score={result['score']:.3f}, "
            f"passed={result['passed']}, weight={self.weight}"
        )


class RuleEvaluationResult:
    """Container for rule evaluation results."""

    def __init__(
        self,
        total_score: float,
        rule_scores: Dict[str, float],
        rule_details: Dict[str, Dict[str, Any]],
        rule_passed: Dict[str, bool],
        pass_rate: float,
    ):
        """Initialize evaluation result.

        Args:
            total_score: Weighted aggregate score
            rule_scores: Individual rule scores
            rule_details: Detailed results for each rule
            rule_passed: Whether each rule passed
            pass_rate: Fraction of rules that passed
        """
        self.total_score = total_score
        self.rule_scores = rule_scores
        self.rule_details = rule_details
        self.rule_passed = rule_passed
        self.pass_rate = pass_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_score": self.total_score,
            "rule_scores": self.rule_scores,
            "rule_details": self.rule_details,
            "rule_passed": self.rule_passed,
            "pass_rate": self.pass_rate,
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics suitable for tracking/logging."""
        metrics = {
            "reward/total_score": self.total_score,
            "reward/pass_rate": self.pass_rate,
        }

        # Add individual rule scores
        for rule_name, score in self.rule_scores.items():
            metrics[f"reward/{rule_name}_score"] = score
            metrics[f"reward/{rule_name}_passed"] = float(self.rule_passed[rule_name])

        return metrics


class TextProcessor:
    """Utility class for common text processing operations."""

    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words from text.

        Args:
            text: Input text

        Returns:
            List of words (lowercased, alphanumeric only)
        """
        import re

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return words

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract numbers from text.

        Args:
            text: Input text

        Returns:
            List of number strings (integers, decimals, percentages)
        """
        import re

        numbers = []

        # Find percentages first (number followed by %)
        percentage_pattern = r"(\d+(?:\.\d+)?)%"
        percentage_matches = re.findall(percentage_pattern, text)
        for match in percentage_matches:
            numbers.append(match + "%")

        # Find "percent" pattern
        percent_word_pattern = r"(\d+(?:\.\d+)?)\s*percent"
        percent_word_matches = re.findall(percent_word_pattern, text.lower())
        for match in percent_word_matches:
            numbers.append(match + "%")  # Normalize to % form

        # Find standalone numbers, excluding those already captured as percentages
        standalone_pattern = r"\b\d+(?:\.\d+)?\b"
        standalone_matches = re.findall(standalone_pattern, text)

        # Get the numeric parts of percentages to exclude from standalone
        percentage_bases = set(percentage_matches + percent_word_matches)

        for match in standalone_matches:
            if match not in percentage_bases:
                numbers.append(match)

        return numbers

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract named entities using simple regex patterns.

        Args:
            text: Input text

        Returns:
            List of potential entities (capitalized words/phrases)
        """
        import re

        # Simple pattern for capitalized words (potential proper nouns)
        # This is a basic approach that can be enhanced later with NLP libraries
        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        entities = re.findall(pattern, text)

        # Filter out common words that are often capitalized
        common_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "A",
            "An",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        }

        entities = [e for e in entities if e not in common_words]
        return entities

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not set1 and not set2:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0
