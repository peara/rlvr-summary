"""BART-MNLI based factual consistency rule implementation."""

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseRule


class BartMNLIConsistencyRule(BaseRule):
    """BART-MNLI based factual consistency rule using Natural Language Inference.

    Uses facebook/bart-large-mnli to determine if a summary (hypothesis) is
    entailed by the source text (premise). Returns binary scores based on
    entailment probability threshold.
    """

    def __init__(self, weight: float = 1.0, config: Optional[Dict[str, Any]] = None):
        """Initialize BART-MNLI consistency rule.

        Args:
            weight: Weight of this rule in the overall score
            config: Configuration dictionary with settings:
                - threshold: Entailment probability threshold for binary scoring (default: 0.8)
                - max_length: Maximum sequence length for truncation (default: 1024)
                - device: Device to run model on (default: auto-detect)
        """
        super().__init__(weight, config)

        self.threshold = self.config.get("threshold", 0.8)
        self.max_length = self.config.get("max_length", 1024)
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

        # Initialize model
        self._initialize_model()

        self.logger.info(
            f"BartMNLIConsistencyRule initialized with threshold={self.threshold}, "
            f"max_length={self.max_length}, device={self.device}"
        )

    def _initialize_model(self) -> None:
        """Initialize the BART-MNLI model."""
        model_name = "facebook/bart-large-mnli"

        try:
            # Use manual PyTorch approach for direct NLI evaluation
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.logger.error(f"Failed to initialize BART-MNLI model: {e}")
            raise

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length.

        Args:
            text: Input text to truncate

        Returns:
            Truncated text
        """
        # Simple word-based truncation
        words = text.split()
        if len(words) <= self.max_length:
            return text

        truncated = " ".join(words[: self.max_length])
        self.logger.debug(
            f"Truncated text from {len(words)} to {self.max_length} words"
        )
        return truncated

    def _evaluate_entailment(self, source: str, summary: str) -> float:
        """Evaluate entailment using NLI approach.

        Args:
            source: Source text (premise)
            summary: Summary text (hypothesis)

        Returns:
            Entailment probability
        """
        # Truncate inputs
        source = self._truncate_text(source)
        summary = self._truncate_text(summary)

        # Pose as NLI problem: premise is source, hypothesis is summary
        premise = source
        hypothesis = summary

        # Tokenize and encode
        inputs = self.tokenizer.encode(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        inputs = inputs.to(self.device)

        # Get model predictions
        with torch.no_grad():
            logits = self.model(inputs)[0]

        # BART-MNLI outputs: [contradiction, neutral, entailment]
        # We want the probability of entailment
        probs = torch.softmax(logits, dim=1)
        entailment_prob = probs[0, 2].item()  # Index 2 is entailment

        return entailment_prob

    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate factual consistency using BART-MNLI.

        Args:
            source: Original source text
            summary: Generated summary

        Returns:
            Dictionary with score, details, and pass/fail status
        """
        try:
            # Get entailment probability using direct NLI evaluation
            entailment_prob = self._evaluate_entailment(source, summary)

            # Binary scoring based on threshold
            binary_score = 1.0 if entailment_prob >= self.threshold else 0.0
            passed = entailment_prob >= self.threshold

            return {
                "score": binary_score,
                "details": {
                    "entailment_probability": entailment_prob,
                    "threshold": self.threshold,
                    "binary_score": binary_score,
                    "method": "nli",
                },
                "passed": passed,
            }

        except Exception as e:
            self.logger.error(f"Error in BART-MNLI evaluation: {e}")
            return {
                "score": 0.5,  # Neutral score on error
                "details": {"error": str(e)},
                "passed": False,
            }

    def batch_evaluate(
        self, sources: List[str], summaries: List[str]
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of source-summary pairs.

        Args:
            sources: List of source texts
            summaries: List of summaries

        Returns:
            List of evaluation results
        """
        if len(sources) != len(summaries):
            raise ValueError(
                f"Mismatch in batch sizes: {len(sources)} sources, {len(summaries)} summaries"
            )

        results = []
        for i, (source, summary) in enumerate(zip(sources, summaries)):
            if i % 10 == 0:
                self.logger.debug(f"Processing batch item {i+1}/{len(sources)}")

            result = self.evaluate(source, summary)
            results.append(result)

        return results

    def get_threshold(self) -> float:
        """Get the current threshold value.

        Returns:
            Current threshold for binary scoring
        """
        return self.threshold

    def update_threshold(self, threshold: float) -> None:
        """Update the threshold value.

        Args:
            threshold: New threshold value (should be between 0 and 1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        old_threshold = self.threshold
        self.threshold = threshold
        self.config["threshold"] = threshold

        self.logger.info(f"Updated threshold from {old_threshold} to {threshold}")
