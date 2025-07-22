"""FENICE Factual Consistency Scorer for reward system."""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from .base import BaseRule


def _get_document_key(document: str) -> str:
    """Generate a stable key for document cache validation."""
    return hashlib.sha256(document.encode('utf-8')).hexdigest()[:16]


def _validate_cache_for_document(cache_data: Dict, document: str) -> bool:
    """Validate that cache data matches the given document."""
    if not cache_data or not isinstance(cache_data, dict):
        return False
    
    # Check if cache has document_key for validation
    expected_key = _get_document_key(document)
    cached_key = cache_data.get('document_key')
    
    return cached_key == expected_key


class FENICEScorer(BaseRule):
    """FENICE Factual Consistency Scorer using the official FENICE package.

    This scorer uses the FENICE (Factuality Evaluation of summarization based on
    Natural Language Inference and Claim Extraction) package to evaluate factual
    consistency between source documents and summaries.
    """

    def __init__(self, weight: float = 1.0, config: Optional[Dict[str, Any]] = None):
        """Initialize FENICE scorer.

        Args:
            weight: Weight of this scorer in the overall reward
            config: Configuration including thresholds
        """
        super().__init__(weight, config)

        self.threshold = self.config.get("threshold", 0.5)
        self.batch_size = self.config.get("batch_size", 8)

        # Initialize FENICE model lazily to avoid loading during import
        self._fenice_model = None
        self._model_loaded = False

        self.logger.info(f"FENICE scorer initialized with threshold: {self.threshold}")

    @property
    def name(self) -> str:
        """Return the name of this scorer."""
        return "fenice_factual_consistency"

    def _load_model(self) -> None:
        """Load FENICE model lazily."""
        if self._model_loaded:
            return

        try:
            from rlvr_summary.fenice import FENICE

            self.logger.info("Loading FENICE model...")
            self._fenice_model = FENICE()
            self._model_loaded = True
            self.logger.info("FENICE model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "FENICE module not found. Please check the installation."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load FENICE model: {e}") from e

    def evaluate_with_context(self, source: str, summary: str, context: Dict) -> Dict[str, Any]:
        """Evaluate factual consistency with context data (e.g., cache).

        Args:
            source: Original text
            summary: Generated summary
            context: Context data including potential FENICE cache

        Returns:
            Dictionary containing:
                - score: Factual consistency score (0.0 to 1.0)
                - details: Detailed evaluation results
                - passed: Whether the score meets threshold
        """
        # Extract cache data from context if available
        cache_data = context.get('fenice_cache') if context else None
        return self.evaluate(source, summary, cache_data=cache_data)

    def evaluate(self, source: str, summary: str, cache_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate factual consistency of summary against source.

        Args:
            source: Original text
            summary: Generated summary
            cache_data: Optional pre-computed document cache data

        Returns:
            Dictionary containing:
                - score: Factual consistency score (0.0 to 1.0)
                - details: Detailed evaluation results
                - passed: Whether the score meets threshold
        """
        if not summary or not source:
            raise ValueError("Empty input: both summary and source must be provided")

        self._load_model()

        # Prepare batch for FENICE
        batch = [{"document": source, "summary": summary}]

        try:
            # Validate cache data if provided
            validated_cache = None
            if cache_data and _validate_cache_for_document(cache_data, source):
                self.logger.debug("Using validated cached document data for FENICE evaluation")
                # Convert single document cache to batch format expected by FENICE
                validated_cache = {0: cache_data}
            elif cache_data:
                self.logger.warning("Cache data validation failed, falling back to runtime computation")
            
            # Get FENICE results with or without cache
            if validated_cache:
                results = self._fenice_model.score_batch(batch, document_cache_data=validated_cache)
            else:
                results = self._fenice_model.score_batch(batch)
                
            fenice_result = results[0]  # Single item in batch

            # Extract score and alignments
            score = fenice_result["score"]
            alignments = fenice_result.get("alignments", [])

            # Determine if passed
            passed = score >= self.threshold

            # Extract detailed information
            num_claims = len(alignments)
            claim_scores = [alignment["score"] for alignment in alignments]

            details = {
                "fenice_score": score,
                "num_claims": num_claims,
                "claim_scores": claim_scores,
                "alignments": alignments,
                "threshold": self.threshold,
                "avg_score": score,  # FENICE already provides aggregated score
            }

            self.logger.debug(
                f"FENICE evaluation: {num_claims} claims, "
                f"score={score:.3f}, passed={passed}"
            )

            return {"score": score, "details": details, "passed": passed}

        except Exception as e:
            self.logger.error(f"FENICE evaluation failed: {e}")
            raise RuntimeError(f"FENICE evaluation failed: {e}") from e

    def batch_evaluate(
        self, sources: List[str], summaries: List[str], cache_data: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple source-summary pairs efficiently.

        Args:
            sources: List of source texts
            summaries: List of summaries
            cache_data: Optional pre-computed document cache data

        Returns:
            List of evaluation results
        """
        if len(sources) != len(summaries):
            raise ValueError("Sources and summaries must have same length")

        self._load_model()

        # Prepare batch for FENICE
        batch = [
            {"document": source, "summary": summary}
            for source, summary in zip(sources, summaries)
        ]

        try:
            # Validate cache data if provided
            validated_cache = None
            if cache_data:
                validated_cache = {}
                for i, source in enumerate(sources):
                    if i in cache_data and _validate_cache_for_document(cache_data[i], source):
                        validated_cache[i] = cache_data[i]
                    elif i in cache_data:
                        self.logger.warning(f"Cache validation failed for document {i}, falling back to runtime computation")
                
                if validated_cache:
                    self.logger.debug(f"Using validated cached document data for {len(validated_cache)} out of {len(sources)} documents in batch")
            
            # Get FENICE results for entire batch with or without cache
            if validated_cache:
                fenice_results = self._fenice_model.score_batch(batch, document_cache_data=validated_cache)
            else:
                fenice_results = self._fenice_model.score_batch(batch)

            # Convert to expected format
            results = []
            for fenice_result in fenice_results:
                score = fenice_result["score"]
                alignments = fenice_result.get("alignments", [])
                passed = score >= self.threshold

                num_claims = len(alignments)
                claim_scores = [alignment["score"] for alignment in alignments]

                details = {
                    "fenice_score": score,
                    "num_claims": num_claims,
                    "claim_scores": claim_scores,
                    "alignments": alignments,
                    "threshold": self.threshold,
                    "avg_score": score,
                }

                results.append({"score": score, "details": details, "passed": passed})

            return results

        except Exception as e:
            self.logger.error(f"FENICE batch evaluation failed: {e}")
            raise RuntimeError(f"FENICE batch evaluation failed: {e}") from e

    def clear_model_cache(self):
        """Clear cached models to free memory."""
        if self._fenice_model is not None:
            self._fenice_model.clear_model_cache()
            self.logger.info("FENICE model cache cleared")

    def get_model_info(self):
        """Get information about loaded models."""
        if self._fenice_model is not None:
            return self._fenice_model.get_model_info()
        return {"models_loaded": False}


def create_fenice_scorer(weight: float = 1.0, threshold: float = 0.5) -> FENICEScorer:
    """Create a FENICE scorer with default configuration.

    Args:
        weight: Weight in overall reward calculation
        threshold: Threshold for passing score

    Returns:
        Configured FENICEScorer instance
    """
    config = {
        "threshold": threshold,
        "batch_size": 8,
    }

    return FENICEScorer(weight=weight, config=config)
