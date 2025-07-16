"""FENICE Factual Consistency Scorer for reward system."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseRule


class FENICEScorer(BaseRule):
    """FENICE Factual Consistency Scorer using claim extraction and NLI.
    
    This scorer implements the distilled FENICE approach:
    1. Extract claims from the summary
    2. Score each claim against the source using NLI
    3. Aggregate scores to get overall factual consistency
    """

    def __init__(self, weight: float = 1.0, config: Optional[Dict[str, Any]] = None):
        """Initialize FENICE scorer.
        
        Args:
            weight: Weight of this scorer in the overall reward
            config: Configuration including model paths and thresholds
        """
        super().__init__(weight, config)
        
        self.model_name = self.config.get("model_name", "Babelscape/FENICE")
        self.batch_size = self.config.get("batch_size", 8)
        self.max_length = self.config.get("max_length", 512)
        self.threshold = self.config.get("threshold", 0.5)
        
        # Initialize models lazily to avoid loading during import
        self._claim_extractor = None
        self._nli_model = None
        self._tokenizer = None
        self._models_loaded = False
        
        self.logger.info(f"FENICE scorer initialized with model: {self.model_name}")

    @property
    def name(self) -> str:
        """Return the name of this scorer."""
        return "fenice_factual_consistency"

    def _load_models(self) -> None:
        """Load FENICE models lazily."""
        if self._models_loaded:
            return
            
        # Import here to avoid import errors if transformers not available
        from transformers import (
            AutoTokenizer, 
            AutoModelForSequenceClassification,
            AutoModelForSeq2SeqLM,
            pipeline
        )
        
        self.logger.info("Loading FENICE models...")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # For now, use a simplified approach with existing models
        # In a real implementation, we'd use the actual FENICE models
        self._claim_extractor = pipeline(
            "text2text-generation",
            model="t5-small",  # Placeholder - would use FENICE claim extractor
            tokenizer=AutoTokenizer.from_pretrained("t5-small"),
            max_length=self.max_length,
            device_map="auto" if self._has_gpu() else "cpu"
        )
        
        self._nli_model = pipeline(
            "text-classification",
            model="microsoft/deberta-v3-base",  # Placeholder - would use FENICE NLI model
            device_map="auto" if self._has_gpu() else "cpu"
        )
        
        self._models_loaded = True
        self.logger.info("FENICE models loaded successfully")

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _extract_claims(self, summary: str) -> List[str]:
        """Extract claims from summary text.
        
        Args:
            summary: Summary text to extract claims from
            
        Returns:
            List of extracted claims
        """
        self._load_models()
        
        # For this implementation, we'll use simple sentence splitting
        # In a real FENICE implementation, this would use the claim extraction model
        claims = self._simple_sentence_split(summary)
        
        self.logger.debug(f"Extracted {len(claims)} claims from summary")
        return claims

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting as fallback."""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        # Clean and filter
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Minimum claim length
                claims.append(sentence)
        return claims

    def _score_claim_nli(self, claim: str, source: str) -> Dict[str, Any]:
        """Score a claim against source using NLI.
        
        Args:
            claim: Claim to verify
            source: Source text to verify against
            
        Returns:
            Dictionary with NLI score and details
        """
        self._load_models()
        
        # Prepare input for NLI model
        # Format: premise: source, hypothesis: claim
        input_text = f"premise: {source[:500]} hypothesis: {claim}"  # Truncate for efficiency
        
        # Get NLI prediction
        result = self._nli_model(input_text)
        
        # Extract score and label
        label = result[0]['label'] if result else "NEUTRAL"
        confidence = result[0]['score'] if result else 0.0
        
        # Convert to factual consistency score
        if label == "ENTAILMENT":
            score = confidence
        elif label == "NEUTRAL":
            score = 0.5  # Neutral baseline
        else:  # CONTRADICTION
            score = 1.0 - confidence  # Penalty for contradiction
        
        return {
            "score": score,
            "label": label,
            "confidence": confidence
        }

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
        
        # Extract claims from summary
        claims = self._extract_claims(summary)
        
        if not claims:
            return {
                "score": 0.5,  # Neutral for no claims
                "details": {"claims": [], "no_claims": True},
                "passed": True
            }
        
        # Score each claim
        claim_scores = []
        claim_details = []
        
        for claim in claims:
            nli_result = self._score_claim_nli(claim, source)
            claim_scores.append(nli_result["score"])
            claim_details.append({
                "claim": claim,
                "nli_score": nli_result["score"],
                "nli_label": nli_result["label"],
                "nli_confidence": nli_result["confidence"]
            })
        
        # Aggregate scores (average)
        total_score = sum(claim_scores) / len(claim_scores)
        
        # Determine if passed
        passed = total_score >= self.threshold
        
        details = {
            "num_claims": len(claims),
            "claim_scores": claim_scores,
            "claim_details": claim_details,
            "avg_score": total_score,
            "threshold": self.threshold,
        }
        
        self.logger.debug(
            f"FENICE evaluation: {len(claims)} claims, "
            f"avg_score={total_score:.3f}, passed={passed}"
        )
        
        return {
            "score": total_score,
            "details": details,
            "passed": passed
        }

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
        
        results = []
        for source, summary in zip(sources, summaries):
            result = self.evaluate(source, summary)
            results.append(result)
        
        return results


def create_fenice_scorer(
    weight: float = 1.0,
    model_name: str = "Babelscape/FENICE",
    threshold: float = 0.5
) -> FENICEScorer:
    """Create a FENICE scorer with default configuration.
    
    Args:
        weight: Weight in overall reward calculation
        model_name: Hugging Face model name for FENICE
        threshold: Threshold for passing score
        
    Returns:
        Configured FENICEScorer instance
    """
    config = {
        "model_name": model_name,
        "threshold": threshold,
        "batch_size": 8,
        "max_length": 512
    }
    
    return FENICEScorer(weight=weight, config=config)