"""ROUGE evaluation metrics implementation."""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleRougeCalculator:
    """Simple ROUGE calculator implementation.

    This is a basic implementation of ROUGE-1, ROUGE-2, and ROUGE-L metrics
    without external dependencies.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization - split on whitespace and common punctuation
        text = text.lower().strip()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-grams as tuples
        """
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Calculate longest common subsequence length.

        Args:
            x: First sequence
            y: Second sequence

        Returns:
            Length of LCS
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def rouge_n(self, hypothesis: str, reference: str, n: int = 1) -> Dict[str, float]:
        """Calculate ROUGE-N score.

        Args:
            hypothesis: Generated summary
            reference: Reference summary
            n: N-gram size

        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)

        hyp_ngrams = self._get_ngrams(hyp_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not hyp_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Count overlaps
        hyp_ngram_counts = defaultdict(int)
        ref_ngram_counts = defaultdict(int)

        for ngram in hyp_ngrams:
            hyp_ngram_counts[ngram] += 1

        for ngram in ref_ngrams:
            ref_ngram_counts[ngram] += 1

        overlap = 0
        for ngram, count in hyp_ngram_counts.items():
            overlap += min(count, ref_ngram_counts[ngram])

        precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def rouge_l(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE-L score.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)

        if not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not hyp_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        lcs_len = self._lcs_length(hyp_tokens, ref_tokens)

        precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def calculate_rouge_scores(
        self, hypothesis: str, reference: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate all ROUGE scores.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        scores = {
            "rouge1": self.rouge_n(hypothesis, reference, n=1),
            "rouge2": self.rouge_n(hypothesis, reference, n=2),
            "rougeL": self.rouge_l(hypothesis, reference),
        }

        return scores


class EvaluationPipeline:
    """Evaluation pipeline for model performance."""

    def __init__(self, wandb_logger=None):
        """Initialize evaluation pipeline.

        Args:
            wandb_logger: Optional W&B logger for metrics tracking
        """
        self.rouge_calculator = SimpleRougeCalculator()
        self.wandb_logger = wandb_logger
        self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__name__}")

    def evaluate_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        step: Optional[int] = None,
        log_to_wandb: bool = True,
    ) -> Dict[str, float]:
        """Evaluate a batch of hypotheses against references.

        Args:
            hypotheses: Generated summaries
            references: Reference summaries
            step: Training step for logging
            log_to_wandb: Whether to log to W&B

        Returns:
            Dictionary with averaged ROUGE scores
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        all_scores = defaultdict(list)

        for hyp, ref in zip(hypotheses, references):
            rouge_scores = self.rouge_calculator.calculate_rouge_scores(hyp, ref)

            for metric_name, metric_scores in rouge_scores.items():
                for score_type, score_value in metric_scores.items():
                    all_scores[f"{metric_name}_{score_type}"].append(score_value)

        # Calculate averages
        avg_scores = {}
        for metric_name, scores in all_scores.items():
            avg_scores[metric_name] = sum(scores) / len(scores) if scores else 0.0

        # Log to W&B if enabled
        if log_to_wandb and self.wandb_logger and hasattr(self.wandb_logger, "log"):
            eval_metrics = {f"eval/{key}": value for key, value in avg_scores.items()}
            self.wandb_logger.log(eval_metrics, step=step)

        return avg_scores

    def evaluate_single(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a single hypothesis against reference.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with ROUGE scores
        """
        return self.rouge_calculator.calculate_rouge_scores(hypothesis, reference)

    def calculate_pass_rates(
        self,
        scores: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """Calculate pass rates based on thresholds.

        Args:
            scores: ROUGE scores
            thresholds: Score thresholds for passing

        Returns:
            Dictionary indicating which metrics passed
        """
        if thresholds is None:
            thresholds = {
                "rouge1_f1": 0.3,
                "rouge2_f1": 0.15,
                "rougeL_f1": 0.25,
            }

        pass_rates = {}
        for metric, threshold in thresholds.items():
            pass_rates[metric] = scores.get(metric, 0.0) >= threshold

        return pass_rates
