"""Text preprocessing utilities for VERL training.

Minimal preprocessing focused on what LLMs actually need.
"""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Minimal text preprocessor for LLM training.

    Provides only essential cleaning and length validation needed for VERL.
    Modern LLMs don't need the heavy NLP preprocessing of classical models.
    """

    def __init__(
        self,
        max_length: Optional[int] = None,
        normalize_whitespace: bool = True,
        remove_excessive_newlines: bool = True,
    ):
        """Initialize minimal text preprocessor.

        Args:
            max_length: Maximum text length (characters)
            normalize_whitespace: Whether to normalize whitespace
            remove_excessive_newlines: Whether to reduce multiple newlines
        """
        self.max_length = max_length
        self.normalize_whitespace = normalize_whitespace
        self.remove_excessive_newlines = remove_excessive_newlines

    def clean_text(self, text: str) -> str:
        """Apply minimal text cleaning for LLM training.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove excessive newlines (keep structure but avoid huge gaps)
        if self.remove_excessive_newlines:
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Normalize whitespace (but preserve intentional formatting)
        if self.normalize_whitespace:
            # Replace tabs with spaces
            text = text.replace("\t", " ")
            # Replace multiple spaces with single space (but preserve newlines)
            text = re.sub(r"[ ]+", " ", text)
            # Remove trailing whitespace from lines
            text = re.sub(r"[ ]+\n", "\n", text)
            # Remove leading/trailing whitespace
            text = text.strip()

        # Apply length limit (break at word boundary)
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(" ", 1)[0]

        return text

    def preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a sample with minimal cleaning.

        Args:
            sample: Sample dictionary with 'article' and 'highlights' fields

        Returns:
            Sample with cleaned text fields
        """
        processed_sample = sample.copy()

        # Clean article
        if "article" in sample:
            processed_sample["article"] = self.clean_text(sample["article"])

        # Clean summary/highlights
        if "highlights" in sample:
            processed_sample["highlights"] = self.clean_text(sample["highlights"])

        return processed_sample

    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Get basic statistics about the text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {"char_count": 0, "word_count": 0, "line_count": 0}

        clean_text = self.clean_text(text)

        stats = {
            "char_count": len(clean_text),
            "word_count": len(clean_text.split()),
            "line_count": clean_text.count("\n") + 1,
        }

        return stats
