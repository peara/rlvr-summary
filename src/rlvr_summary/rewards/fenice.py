"""FENICE Factual Consistency Scorer for reward system."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseRule


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
            from metric.FENICE import FENICE
            self.logger.info("Loading FENICE model...")
            self._fenice_model = FENICE()
            self._model_loaded = True
            self.logger.info("FENICE model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "FENICE package not found. Please install it with: pip install FENICE"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load FENICE model: {e}") from e

    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate factual consistency of summary against source.
        
        Args:
            source: Original text
            summary: Generated summary
            
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
            # Get FENICE results
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
            
            return {
                "score": score,
                "details": details,
                "passed": passed
            }
            
        except Exception as e:
            self.logger.error(f"FENICE evaluation failed: {e}")
            raise RuntimeError(f"FENICE evaluation failed: {e}") from e

    def batch_evaluate(
        self, 
        sources: List[str], 
        summaries: List[str]
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple source-summary pairs efficiently.
        
        Args:
            sources: List of source texts
            summaries: List of summaries
            
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
            # Get FENICE results for entire batch
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
                
                results.append({
                    "score": score,
                    "details": details,
                    "passed": passed
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"FENICE batch evaluation failed: {e}")
            raise RuntimeError(f"FENICE batch evaluation failed: {e}") from e


def create_fenice_scorer(
    weight: float = 1.0,
    threshold: float = 0.5
) -> FENICEScorer:
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