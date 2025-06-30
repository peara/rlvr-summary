"""Data validation and quality control framework."""

import logging
from typing import Any, Dict, List, Optional, Set, Union
import json
import re

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Data validation and quality control for CNN-DM and other datasets.
    
    Provides comprehensive validation for text data quality, format compliance,
    and content requirements for summarization tasks.
    """
    
    def __init__(
        self,
        min_article_length: int = 100,
        max_article_length: int = 50000,
        min_summary_length: int = 10,
        max_summary_length: int = 1000,
        required_fields: Optional[List[str]] = None,
        allow_empty_summaries: bool = False,
        check_language: bool = True,
        check_encoding: bool = True,
    ):
        """Initialize data validator.
        
        Args:
            min_article_length: Minimum article length in characters
            max_article_length: Maximum article length in characters
            min_summary_length: Minimum summary length in characters
            max_summary_length: Maximum summary length in characters
            required_fields: List of required fields in each sample
            allow_empty_summaries: Whether to allow empty summaries
            check_language: Whether to check if text appears to be English
            check_encoding: Whether to check for encoding issues
        """
        self.min_article_length = min_article_length
        self.max_article_length = max_article_length
        self.min_summary_length = min_summary_length
        self.max_summary_length = max_summary_length
        self.required_fields = required_fields or ["id", "article", "highlights"]
        self.allow_empty_summaries = allow_empty_summaries
        self.check_language = check_language
        self.check_encoding = check_encoding
        
        # Statistics tracking
        self.validation_stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "failed_samples": 0,
            "validation_errors": {},
        }
        
    def validate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Validate a single sample.
        
        Args:
            sample: Sample dictionary to validate
            
        Returns:
            Validation result with is_valid flag and error details
        """
        self.validation_stats["total_samples"] += 1
        
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "sample_id": sample.get("id", "unknown"),
        }
        
        try:
            # Check required fields
            missing_fields = self._check_required_fields(sample)
            if missing_fields:
                result["errors"].append(f"Missing required fields: {missing_fields}")
                
            # Check field types
            type_errors = self._check_field_types(sample)
            if type_errors:
                result["errors"].extend(type_errors)
                
            # Check article content
            article_errors = self._validate_article(sample.get("article", ""))
            if article_errors:
                result["errors"].extend(article_errors)
                
            # Check summary content
            summary_errors = self._validate_summary(sample.get("highlights", ""))
            if summary_errors:
                result["errors"].extend(summary_errors)
                
            # Check encoding issues
            if self.check_encoding:
                encoding_errors = self._check_encoding_issues(sample)
                if encoding_errors:
                    result["warnings"].extend(encoding_errors)
                    
            # Check language
            if self.check_language:
                language_warnings = self._check_language(sample)
                if language_warnings:
                    result["warnings"].extend(language_warnings)
                    
            # Check for duplicate content
            duplicate_warnings = self._check_duplicates(sample)
            if duplicate_warnings:
                result["warnings"].extend(duplicate_warnings)
                
        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            
        # Update validation status
        if result["errors"]:
            result["is_valid"] = False
            self.validation_stats["failed_samples"] += 1
            
            # Track error types
            for error in result["errors"]:
                error_type = error.split(":")[0]
                self.validation_stats["validation_errors"][error_type] = (
                    self.validation_stats["validation_errors"].get(error_type, 0) + 1
                )
        else:
            self.validation_stats["valid_samples"] += 1
            
        return result
        
    def _check_required_fields(self, sample: Dict) -> List[str]:
        """Check for required fields in sample.
        
        Args:
            sample: Sample to check
            
        Returns:
            List of missing field names
        """
        missing = []
        for field in self.required_fields:
            if field not in sample or sample[field] is None:
                missing.append(field)
        return missing
        
    def _check_field_types(self, sample: Dict) -> List[str]:
        """Check field types in sample.
        
        Args:
            sample: Sample to check
            
        Returns:
            List of type error messages
        """
        errors = []
        
        # Check that text fields are strings
        text_fields = ["article", "highlights", "id", "url"]
        for field in text_fields:
            if field in sample and sample[field] is not None:
                if not isinstance(sample[field], str):
                    errors.append(f"Field '{field}' must be string, got {type(sample[field])}")
                    
        return errors
        
    def _validate_article(self, article: str) -> List[str]:
        """Validate article content.
        
        Args:
            article: Article text to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not article:
            errors.append("Article is empty")
            return errors
            
        article_length = len(article)
        
        if article_length < self.min_article_length:
            errors.append(f"Article too short: {article_length} < {self.min_article_length}")
            
        if article_length > self.max_article_length:
            errors.append(f"Article too long: {article_length} > {self.max_article_length}")
            
        # Check for suspicious patterns
        if len(article.split()) < 10:
            errors.append("Article has too few words")
            
        # Check for repeated patterns that might indicate corruption
        if self._has_excessive_repetition(article):
            errors.append("Article contains excessive repetition")
            
        return errors
        
    def _validate_summary(self, summary: str) -> List[str]:
        """Validate summary content.
        
        Args:
            summary: Summary text to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not summary:
            if not self.allow_empty_summaries:
                errors.append("Summary is empty")
            return errors
            
        summary_length = len(summary)
        
        if summary_length < self.min_summary_length:
            errors.append(f"Summary too short: {summary_length} < {self.min_summary_length}")
            
        if summary_length > self.max_summary_length:
            errors.append(f"Summary too long: {summary_length} > {self.max_summary_length}")
            
        # Check for suspicious patterns
        if len(summary.split()) < 3:
            errors.append("Summary has too few words")
            
        return errors
        
    def _check_encoding_issues(self, sample: Dict) -> List[str]:
        """Check for text encoding issues.
        
        Args:
            sample: Sample to check
            
        Returns:
            List of encoding warning messages
        """
        warnings = []
        
        text_fields = ["article", "highlights"]
        for field in text_fields:
            if field in sample and sample[field]:
                text = sample[field]
                
                # Check for common encoding issues
                if "�" in text:
                    warnings.append(f"Field '{field}' contains replacement characters (encoding issues)")
                    
                # Check for excessive non-ASCII characters
                non_ascii_count = sum(1 for c in text if ord(c) > 127)
                if non_ascii_count > len(text) * 0.3:  # More than 30% non-ASCII
                    warnings.append(f"Field '{field}' has high non-ASCII character ratio")
                    
        return warnings
        
    def _check_language(self, sample: Dict) -> List[str]:
        """Check if text appears to be in English.
        
        Args:
            sample: Sample to check
            
        Returns:
            List of language warning messages
        """
        warnings = []
        
        # Simple English detection based on common words
        english_indicators = {
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"
        }
        
        text_fields = ["article", "highlights"]
        for field in text_fields:
            if field in sample and sample[field]:
                text = sample[field].lower()
                words = set(re.findall(r'\b\w+\b', text))
                
                # Check overlap with English indicators
                english_word_count = len(words.intersection(english_indicators))
                total_unique_words = len(words)
                
                if total_unique_words > 10:  # Only check if we have enough words
                    english_ratio = english_word_count / min(total_unique_words, len(english_indicators))
                    if english_ratio < 0.1:  # Less than 10% common English words
                        warnings.append(f"Field '{field}' may not be in English")
                        
        return warnings
        
    def _check_duplicates(self, sample: Dict) -> List[str]:
        """Check for duplicate or near-duplicate content.
        
        Args:
            sample: Sample to check
            
        Returns:
            List of duplicate warning messages
        """
        warnings = []
        
        if "article" in sample and "highlights" in sample:
            article = sample["article"].lower()
            highlights = sample["highlights"].lower()
            
            # Check if summary is just a copy of article start
            if highlights and article.startswith(highlights):
                warnings.append("Summary appears to be identical to article beginning")
                
            # Check for excessive overlap
            article_words = set(article.split())
            highlight_words = set(highlights.split())
            
            if highlight_words and len(highlight_words.intersection(article_words)) / len(highlight_words) > 0.9:
                warnings.append("Summary has excessive word overlap with article")
                
        return warnings
        
    def _has_excessive_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text has excessive repetition.
        
        Args:
            text: Text to check
            threshold: Repetition threshold (0-1)
            
        Returns:
            True if text has excessive repetition
        """
        words = text.split()
        if len(words) < 10:
            return False
            
        # Check for repeated phrases
        phrases = []
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            phrases.append(phrase)
            
        unique_phrases = set(phrases)
        repetition_ratio = 1 - (len(unique_phrases) / len(phrases))
        
        return repetition_ratio > threshold
        
    def batch_validate(self, samples: List[Dict]) -> Dict[str, Any]:
        """Validate a batch of samples.
        
        Args:
            samples: List of samples to validate
            
        Returns:
            Batch validation results
        """
        results = []
        
        for sample in samples:
            result = self.validate_sample(sample)
            results.append(result)
            
        # Compile batch statistics
        batch_stats = {
            "total_samples": len(samples),
            "valid_samples": sum(1 for r in results if r["is_valid"]),
            "failed_samples": sum(1 for r in results if not r["is_valid"]),
            "validation_rate": 0.0,
            "results": results,
            "error_summary": self.validation_stats["validation_errors"].copy(),
        }
        
        if batch_stats["total_samples"] > 0:
            batch_stats["validation_rate"] = batch_stats["valid_samples"] / batch_stats["total_samples"]
            
        return batch_stats
        
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report.
        
        Returns:
            Validation statistics and summary
        """
        stats = self.validation_stats.copy()
        
        if stats["total_samples"] > 0:
            stats["validation_rate"] = stats["valid_samples"] / stats["total_samples"]
            stats["failure_rate"] = stats["failed_samples"] / stats["total_samples"]
        else:
            stats["validation_rate"] = 0.0
            stats["failure_rate"] = 0.0
            
        return stats
        
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "failed_samples": 0,
            "validation_errors": {},
        }