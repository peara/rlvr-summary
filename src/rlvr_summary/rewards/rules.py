"""Individual rule implementations for the reward system."""

import re
import logging
from typing import Dict, Any, List, Set

from .base import BaseRule, TextProcessor


class LengthConstraintRule(BaseRule):
    """Rule for evaluating summary length constraints."""
    
    @property
    def name(self) -> str:
        return "length_constraint"
    
    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate length constraint for the summary.
        
        Args:
            source: Original text
            summary: Generated summary
            
        Returns:
            Score based on word count relative to constraints
        """
        words = TextProcessor.extract_words(summary)
        word_count = len(words)
        
        # Get configuration parameters
        min_words = self.config.get("min_words", 20)
        max_words = self.config.get("max_words", 100)
        optimal_range = self.config.get("optimal_range", [30, 80])
        penalty_factor = self.config.get("penalty_factor", 0.5)
        
        optimal_min, optimal_max = optimal_range
        
        # Calculate score
        if optimal_min <= word_count <= optimal_max:
            # Perfect score for optimal range
            score = 1.0
        elif min_words <= word_count <= max_words:
            # Partial score for acceptable range
            if word_count < optimal_min:
                # Too short
                ratio = (word_count - min_words) / (optimal_min - min_words)
            else:
                # Too long
                ratio = (max_words - word_count) / (max_words - optimal_max)
            score = penalty_factor + (1.0 - penalty_factor) * ratio
        else:
            # Outside acceptable range - very low score
            score = 0.1
        
        passed = word_count >= min_words and word_count <= max_words
        
        result = {
            "score": max(0.0, min(1.0, score)),
            "details": {
                "word_count": word_count,
                "min_words": min_words,
                "max_words": max_words,
                "optimal_range": optimal_range,
                "in_optimal_range": optimal_min <= word_count <= optimal_max,
                "in_acceptable_range": min_words <= word_count <= max_words,
            },
            "passed": passed,
        }
        
        self.log_evaluation(result, source, summary)
        return result


class EntityOverlapRule(BaseRule):
    """Rule for evaluating entity overlap between source and summary."""
    
    @property
    def name(self) -> str:
        return "entity_overlap"
    
    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate entity overlap between source and summary.
        
        Args:
            source: Original text
            summary: Generated summary
            
        Returns:
            Score based on entity overlap using Jaccard similarity
        """
        source_entities = set(TextProcessor.extract_entities(source))
        summary_entities = set(TextProcessor.extract_entities(summary))
        
        # Get configuration parameters
        min_overlap = self.config.get("min_overlap", 0.3)
        optimal_overlap = self.config.get("optimal_overlap", 0.7)
        
        # Calculate Jaccard similarity
        jaccard_score = TextProcessor.jaccard_similarity(source_entities, summary_entities)
        
        # If no entities found, fall back to word overlap
        if not source_entities and not summary_entities:
            source_words = set(TextProcessor.extract_words(source))
            summary_words = set(TextProcessor.extract_words(summary))
            jaccard_score = TextProcessor.jaccard_similarity(source_words, summary_words)
            fallback_used = True
        else:
            fallback_used = False
        
        # Calculate score based on overlap
        if jaccard_score >= optimal_overlap:
            score = 1.0
        elif jaccard_score >= min_overlap:
            # Linear interpolation between min and optimal
            score = (jaccard_score - min_overlap) / (optimal_overlap - min_overlap)
        else:
            # Below minimum - very low score
            score = jaccard_score / min_overlap * 0.3
        
        passed = jaccard_score >= min_overlap
        
        result = {
            "score": max(0.0, min(1.0, score)),
            "details": {
                "jaccard_score": jaccard_score,
                "source_entities": list(source_entities),
                "summary_entities": list(summary_entities),
                "entity_overlap_count": len(source_entities.intersection(summary_entities)),
                "fallback_used": fallback_used,
                "min_overlap": min_overlap,
                "optimal_overlap": optimal_overlap,
            },
            "passed": passed,
        }
        
        self.log_evaluation(result, source, summary)
        return result


class NumberConsistencyRule(BaseRule):
    """Rule for evaluating number consistency between source and summary."""
    
    @property
    def name(self) -> str:
        return "number_consistency"
    
    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate number consistency between source and summary.
        
        Args:
            source: Original text
            summary: Generated summary
            
        Returns:
            Score based on number matching and consistency
        """
        source_numbers = set(TextProcessor.extract_numbers(source))
        summary_numbers = set(TextProcessor.extract_numbers(summary))
        
        # Get configuration parameters
        exact_match_bonus = self.config.get("exact_match_bonus", 1.0)
        partial_match_bonus = self.config.get("partial_match_bonus", 0.5)
        mismatch_penalty = self.config.get("mismatch_penalty", -0.5)
        
        if not summary_numbers:
            # No numbers in summary - neutral score
            score = 0.7
            details = {
                "source_numbers": list(source_numbers),
                "summary_numbers": list(summary_numbers),
                "exact_matches": [],
                "mismatches": [],
                "no_numbers_in_summary": True,
            }
        else:
            exact_matches = source_numbers.intersection(summary_numbers)
            mismatches = summary_numbers - source_numbers
            
            # Calculate score
            num_summary_numbers = len(summary_numbers)
            num_exact_matches = len(exact_matches)
            num_mismatches = len(mismatches)
            
            if num_summary_numbers == 0:
                score = 0.7  # Neutral
            else:
                # Base score from exact matches
                exact_ratio = num_exact_matches / num_summary_numbers
                score = exact_ratio * exact_match_bonus
                
                # Penalty for mismatches
                mismatch_ratio = num_mismatches / num_summary_numbers
                score += mismatch_ratio * mismatch_penalty
                
            details = {
                "source_numbers": list(source_numbers),
                "summary_numbers": list(summary_numbers),
                "exact_matches": list(exact_matches),
                "mismatches": list(mismatches),
                "exact_match_ratio": num_exact_matches / num_summary_numbers if num_summary_numbers > 0 else 0,
                "mismatch_ratio": num_mismatches / num_summary_numbers if num_summary_numbers > 0 else 0,
                "no_numbers_in_summary": False,
            }
        
        # Rule passes if no mismatches or if no numbers in summary
        passed = len(summary_numbers - source_numbers) == 0
        
        result = {
            "score": max(0.0, min(1.0, score)),
            "details": details,
            "passed": passed,
        }
        
        self.log_evaluation(result, source, summary)
        return result


class ProfanityDetectionRule(BaseRule):
    """Rule for detecting and penalizing profanity in summaries."""
    
    # Basic profanity word list (can be extended)
    PROFANITY_WORDS = {
        'damn', 'hell', 'crap', 'shit', 'fuck', 'fucking', 'bitch', 'ass', 'asshole',
        'bastard', 'piss', 'dick', 'cock', 'pussy', 'whore', 'slut', 'fag', 'nigger',
        'retard', 'stupid', 'idiot', 'moron', 'dumb', 'gay', 'lesbian', 'queer'
    }
    
    @property
    def name(self) -> str:
        return "profanity_detection"
    
    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate profanity in the summary.
        
        Args:
            source: Original text (unused for profanity detection)
            summary: Generated summary
            
        Returns:
            Score based on profanity detection (1.0 if clean, penalty if profanity found)
        """
        if not self.config.get("enabled", True):
            # Profanity detection disabled
            return {
                "score": 1.0,
                "details": {"enabled": False, "profanity_found": []},
                "passed": True,
            }
        
        # Extract words and check for profanity
        words = TextProcessor.extract_words(summary)
        word_set = set(words)
        
        # Load custom wordlist if specified
        wordlist_path = self.config.get("wordlist_path")
        if wordlist_path:
            try:
                with open(wordlist_path, 'r') as f:
                    custom_words = {line.strip().lower() for line in f}
                profanity_set = custom_words
            except FileNotFoundError:
                self.logger.warning(f"Profanity wordlist not found: {wordlist_path}, using default")
                profanity_set = self.PROFANITY_WORDS
        else:
            profanity_set = self.PROFANITY_WORDS
        
        # Find profanity matches
        profanity_found = list(word_set.intersection(profanity_set))
        
        # Calculate score
        if not profanity_found:
            score = 1.0
        else:
            penalty = self.config.get("penalty", -1.0)
            # Apply penalty per profane word found
            score = 1.0 + len(profanity_found) * penalty
        
        passed = len(profanity_found) == 0
        
        result = {
            "score": max(0.0, min(1.0, score)),
            "details": {
                "enabled": True,
                "profanity_found": profanity_found,
                "total_words": len(words),
                "profanity_count": len(profanity_found),
            },
            "passed": passed,
        }
        
        self.log_evaluation(result, source, summary)
        return result


class FluencyRule(BaseRule):
    """Rule for evaluating text fluency (basic heuristics)."""
    
    @property
    def name(self) -> str:
        return "fluency"
    
    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        """Evaluate summary fluency using basic heuristics.
        
        Args:
            source: Original text (unused)
            summary: Generated summary
            
        Returns:
            Score based on fluency heuristics
        """
        if not self.config.get("enabled", True):
            return {
                "score": 1.0,
                "details": {"enabled": False},
                "passed": True,
            }
        
        # Basic fluency heuristics
        words = TextProcessor.extract_words(summary)
        sentences = self._split_sentences(summary)
        
        # Calculate metrics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Heuristic scoring
        score = 1.0
        details = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
        }
        
        # Penalize very short or very long sentences
        if avg_sentence_length < 5:
            score *= 0.8
            details["penalty_short_sentences"] = True
        elif avg_sentence_length > 30:
            score *= 0.9
            details["penalty_long_sentences"] = True
        
        # Penalize very short or very long words on average
        if avg_word_length < 3:
            score *= 0.9
            details["penalty_short_words"] = True
        elif avg_word_length > 8:
            score *= 0.9
            details["penalty_long_words"] = True
        
        min_score = self.config.get("min_score", 0.5)
        passed = score >= min_score
        
        result = {
            "score": max(0.0, min(1.0, score)),
            "details": details,
            "passed": passed,
        }
        
        self.log_evaluation(result, source, summary)
        return result
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences