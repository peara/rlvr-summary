"""BertScore-based Factual Consistency Scorer for reward system."""

import logging
from typing import Any, Dict, List, Optional

try:
    from bert_score import score as bertscore_score

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

from .base import BaseRule


class BertScoreConsistencyRule(BaseRule):
    """BertScore-based factual consistency scorer with binary thresholding.

    This scorer uses BertScore to compute similarity between source documents
    and summaries, then applies a binary threshold to provide clear feedback
    for reinforcement learning training.

    Returns 1.0 if BertScore >= threshold, 0.0 otherwise.
    """

    def __init__(self, weight: float = 1.0, config: Optional[Dict[str, Any]] = None):
        """Initialize BertScore consistency scorer.

        Args:
            weight: Weight of this scorer in the overall reward
            config: Configuration including threshold and model settings
        """
        super().__init__(weight, config)

        self.threshold = self.config.get("threshold", 0.8)
        self.model_type = self.config.get(
            "model_type", "distilbert-base-uncased"
        )  # Faster, smaller model
        self.batch_size = self.config.get("batch_size", 32)
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", True)

        # Document embedding cache for efficiency
        self._document_embeddings_cache = {}

        if not BERTSCORE_AVAILABLE:
            self.logger.error(
                "BertScore package not available. Install with: pip install bert-score"
            )
            raise RuntimeError("BertScore package required but not installed")

        self.logger.info(
            f"BertScore scorer initialized with threshold: {self.threshold}, model: {self.model_type}"
        )

    @property
    def name(self) -> str:
        """Return the name of this scorer."""
        return "bertscore_factual_consistency"

    def evaluate(
        self, source: str, summary: str, cache_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate factual consistency using BertScore with binary thresholding.

        Args:
            source: Original text
            summary: Generated summary
            cache_data: Optional cache data (BertScore results can be cached)

        Returns:
            Dictionary containing:
                - score: Binary score (1.0 if >= threshold, 0.0 otherwise)
                - details: Detailed evaluation results including raw BertScore
                - passed: Whether the score meets threshold (always same as score for binary)
        """
        if not summary or not source:
            raise ValueError("Empty input: both summary and source must be provided")

        try:
            # Standard evaluation without cache
            P, R, F1 = bertscore_score(
                [summary],
                [source],
                model_type=self.model_type,
                batch_size=self.batch_size,
                device="cuda" if self.config.get("use_gpu", True) else "cpu",
                verbose=False,
            )

            raw_score = F1.item()

            details = {
                "raw_bertscore_f1": raw_score,
                "raw_bertscore_precision": P.item(),
                "raw_bertscore_recall": R.item(),
                "threshold": self.threshold,
                "model_type": self.model_type,
                "used_cache": False,
            }

            # Apply binary thresholding
            binary_score = 1.0 if raw_score >= self.threshold else 0.0
            passed = binary_score == 1.0

            details["binary_score"] = binary_score

            self.logger.debug(
                f"BertScore evaluation: raw_f1={raw_score:.3f}, "
                f"binary_score={binary_score}, passed={passed}"
            )

            return {"score": binary_score, "details": details, "passed": passed}

        except Exception as e:
            self.logger.error(f"BertScore evaluation failed: {e}")
            raise RuntimeError(f"BertScore evaluation failed: {e}") from e

    def evaluate_with_context(
        self, source: str, summary: str, context: Dict
    ) -> Dict[str, Any]:
        """Evaluate factual consistency with context data.

        Args:
            source: Original text
            summary: Generated summary
            context: Context data including potential cache

        Returns:
            Dictionary containing evaluation results
        """
        # Extract cache data from context if available
        cache_data = context.get("bertscore_cache") if context else None
        return self.evaluate(source, summary, cache_data=cache_data)

    def batch_evaluate(
        self,
        sources: List[str],
        summaries: List[str],
        cache_data: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple source-summary pairs efficiently using BertScore batching.

        Args:
            sources: List of source texts
            summaries: List of summaries
            cache_data: Optional cache data

        Returns:
            List of evaluation results
        """
        if len(sources) != len(summaries):
            raise ValueError("Sources and summaries must have same length")

        if not sources:
            return []

        try:
            # Compute BertScore for entire batch
            P, R, F1 = bertscore_score(
                summaries,  # candidates
                sources,  # references
                model_type=self.model_type,
                batch_size=self.batch_size,
                device="cuda" if self.config.get("use_gpu", True) else "cpu",
                verbose=False,
            )

            # Convert to list of results
            results = []
            for i in range(len(sources)):
                raw_f1 = F1[i].item()
                raw_p = P[i].item()
                raw_r = R[i].item()

                # Apply binary thresholding
                binary_score = 1.0 if raw_f1 >= self.threshold else 0.0
                passed = binary_score == 1.0

                details = {
                    "raw_bertscore_f1": raw_f1,
                    "raw_bertscore_precision": raw_p,
                    "raw_bertscore_recall": raw_r,
                    "threshold": self.threshold,
                    "binary_score": binary_score,
                    "model_type": self.model_type,
                }

                results.append(
                    {"score": binary_score, "details": details, "passed": passed}
                )

            return results

        except Exception as e:
            self.logger.error(f"BertScore batch evaluation failed: {e}")
            raise RuntimeError(f"BertScore batch evaluation failed: {e}") from e


def create_bertscore_scorer(
    weight: float = 1.0, threshold: float = 0.8
) -> BertScoreConsistencyRule:
    """Create a BertScore consistency scorer with default configuration.

    Args:
        weight: Weight in overall reward calculation
        threshold: Threshold for binary scoring (recommend 0.8-0.85 for good summaries)

    Returns:
        Configured BertScoreConsistencyRule instance
    """
    config = {
        "threshold": threshold,
        "model_type": "distilbert-base-uncased",  # Fast and lightweight
        "batch_size": 32,
        "use_gpu": True,
        "use_fast_tokenizer": True,
    }

    return BertScoreConsistencyRule(weight=weight, config=config)
