"""Text preprocessing utilities for CNN-DailyMail and other text data."""

import logging
import re
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing pipeline for summarization tasks.

    Provides cleaning, normalization, and spaCy-based preprocessing
    for article and summary text.
    """

    def __init__(
        self,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        max_length: Optional[int] = None,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False,
    ):
        """Initialize text preprocessor.

        Args:
            use_spacy: Whether to use spaCy for advanced preprocessing
            spacy_model: spaCy model to use for processing
            max_length: Maximum text length (characters)
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
            remove_special_chars: Whether to remove special characters
        """
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars

        self._nlp = None
        self._init_spacy()

    def _init_spacy(self):
        """Initialize spaCy pipeline if available."""
        if self.use_spacy:
            try:
                import spacy

                self._nlp = spacy.load(self.spacy_model)
                logger.info(f"Loaded spaCy model: {self.spacy_model}")
            except (ImportError, OSError) as e:
                logger.warning(f"spaCy not available or model not found: {e}")
                logger.warning("Falling back to basic text preprocessing")
                self.use_spacy = False
                self._nlp = None

    def clean_text(self, text: str) -> str:
        """Apply basic text cleaning operations.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
            text = re.sub(
                r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )

        # Remove email addresses
        if self.remove_emails:
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
            )

        # Remove special characters (keep basic punctuation)
        if self.remove_special_chars:
            text = re.sub(r"[^\w\s.,!?;:\-\'\"()]", "", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple whitespace with single space
            text = re.sub(r"\s+", " ", text)
            # Remove leading/trailing whitespace
            text = text.strip()

        # Apply length limit
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length].rsplit(" ", 1)[0]  # Break at word boundary

        return text

    def preprocess_with_spacy(self, text: str) -> Dict[str, Union[str, List[str]]]:
        """Preprocess text using spaCy for advanced NLP features.

        Args:
            text: Input text to preprocess

        Returns:
            Dictionary with processed text and extracted features
        """
        if not self._nlp:
            return {"text": self.clean_text(text), "tokens": [], "sentences": []}

        # Clean text first
        clean_text = self.clean_text(text)

        # Process with spaCy
        doc = self._nlp(clean_text)

        return {
            "text": clean_text,
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc],
            "pos_tags": [token.pos_ for token in doc],
            "sentences": [sent.text for sent in doc.sents],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
        }

    def preprocess_article(self, article: str) -> Dict[str, Union[str, List[str]]]:
        """Preprocess article text specifically.

        Args:
            article: Article text to preprocess

        Returns:
            Preprocessed article data
        """
        if self.use_spacy and self._nlp:
            return self.preprocess_with_spacy(article)
        else:
            clean_article = self.clean_text(article)
            return {
                "text": clean_article,
                "tokens": clean_article.split(),
                "sentences": self._split_sentences(clean_article),
            }

    def preprocess_summary(self, summary: str) -> Dict[str, Union[str, List[str]]]:
        """Preprocess summary text specifically.

        Args:
            summary: Summary text to preprocess

        Returns:
            Preprocessed summary data
        """
        if self.use_spacy and self._nlp:
            return self.preprocess_with_spacy(summary)
        else:
            clean_summary = self.clean_text(summary)
            return {
                "text": clean_summary,
                "tokens": clean_summary.split(),
                "sentences": self._split_sentences(clean_summary),
            }

    def preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a complete sample (article + summary).

        Args:
            sample: Sample dictionary with 'article' and 'highlights' fields

        Returns:
            Preprocessed sample with additional fields
        """
        processed_sample = sample.copy()

        # Preprocess article
        if "article" in sample:
            article_data = self.preprocess_article(sample["article"])
            processed_sample.update(
                {
                    "article_clean": article_data["text"],
                    "article_tokens": article_data["tokens"],
                    "article_sentences": article_data["sentences"],
                }
            )

            # Add spaCy-specific fields if available
            if self.use_spacy and self._nlp:
                processed_sample.update(
                    {
                        "article_lemmas": article_data.get("lemmas", []),
                        "article_pos_tags": article_data.get("pos_tags", []),
                        "article_entities": article_data.get("entities", []),
                    }
                )

        # Preprocess summary/highlights
        if "highlights" in sample:
            summary_data = self.preprocess_summary(sample["highlights"])
            processed_sample.update(
                {
                    "highlights_clean": summary_data["text"],
                    "highlights_tokens": summary_data["tokens"],
                    "highlights_sentences": summary_data["sentences"],
                }
            )

            # Add spaCy-specific fields if available
            if self.use_spacy and self._nlp:
                processed_sample.update(
                    {
                        "highlights_lemmas": summary_data.get("lemmas", []),
                        "highlights_pos_tags": summary_data.get("pos_tags", []),
                        "highlights_entities": summary_data.get("entities", []),
                    }
                )

        return processed_sample

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting for fallback when spaCy is not available.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences
        """
        # Simple sentence splitting based on common patterns
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Get basic statistics about the text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {"char_count": 0, "word_count": 0, "sentence_count": 0}

        clean_text = self.clean_text(text)

        stats = {
            "char_count": len(clean_text),
            "word_count": len(clean_text.split()),
            "sentence_count": len(self._split_sentences(clean_text)),
        }

        return stats

    def batch_preprocess(self, samples: List[Dict]) -> List[Dict]:
        """Preprocess a batch of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of preprocessed samples
        """
        processed_samples = []

        for sample in samples:
            try:
                processed_sample = self.preprocess_sample(sample)
                processed_samples.append(processed_sample)
            except Exception as e:
                logger.warning(
                    f"Failed to preprocess sample {sample.get('id', 'unknown')}: {e}"
                )
                # Add original sample with error flag
                error_sample = sample.copy()
                error_sample["preprocessing_error"] = str(e)
                processed_samples.append(error_sample)

        return processed_samples
