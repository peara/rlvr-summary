"""Test BART-MNLI factual consistency rule."""

from unittest.mock import Mock, patch

import pytest
import torch

from rlvr_summary.rewards.bart_mnli import BartMNLIConsistencyRule


class TestBartMNLIConsistencyRule:
    """Test cases for BART-MNLI factual consistency rule."""

    def test_initialization_default_config(self):
        """Test rule initialization with default configuration."""
        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock()

            rule = BartMNLIConsistencyRule(weight=0.5)

            assert rule.weight == 0.5
            assert rule.threshold == 0.8  # Default threshold
            assert rule.max_length == 1024  # Default max length
            assert rule.use_pipeline is True  # Default use pipeline

    def test_initialization_custom_config(self):
        """Test rule initialization with custom configuration."""
        config = {
            "threshold": 0.7,
            "max_length": 512,
            "use_pipeline": False,
            "device": "cpu",
        }

        with patch(
            "rlvr_summary.rewards.bart_mnli.AutoModelForSequenceClassification"
        ) as mock_model:
            with patch(
                "rlvr_summary.rewards.bart_mnli.AutoTokenizer"
            ) as mock_tokenizer:
                mock_model.from_pretrained.return_value = Mock()
                mock_tokenizer.from_pretrained.return_value = Mock()

                rule = BartMNLIConsistencyRule(weight=0.8, config=config)

                assert rule.weight == 0.8
                assert rule.threshold == 0.7
                assert rule.max_length == 512
                assert rule.use_pipeline is False
                assert rule.device == "cpu"

    def test_truncate_text(self):
        """Test text truncation functionality."""
        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock()

            rule = BartMNLIConsistencyRule(config={"max_length": 5})

            # Test short text (no truncation)
            short_text = "This is short"
            assert rule._truncate_text(short_text) == short_text

            # Test long text (should truncate)
            long_text = "This is a very long text that exceeds the limit"
            truncated = rule._truncate_text(long_text)
            assert len(truncated.split()) == 5
            assert truncated == "This is a very long"

    def test_evaluate_with_pipeline_mock(self):
        """Test evaluation using pipeline with mocked results."""
        # Mock pipeline result
        mock_result = {
            "labels": ["entailed by source", "not entailed by source"],
            "scores": [0.85, 0.15],  # High entailment probability
        }

        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = mock_result
            mock_pipeline.return_value = mock_classifier

            rule = BartMNLIConsistencyRule(
                config={"threshold": 0.8, "use_pipeline": True}
            )

            source = "The sky is blue on a clear day."
            summary = "The sky appears blue when it's clear."

            result = rule.evaluate(source, summary)

            assert result["score"] == 1.0  # Above threshold
            assert result["passed"] is True
            assert result["details"]["entailment_probability"] == 0.85
            assert result["details"]["threshold"] == 0.8
            assert result["details"]["binary_score"] == 1.0

    def test_evaluate_below_threshold(self):
        """Test evaluation when entailment probability is below threshold."""
        # Mock pipeline result with low entailment
        mock_result = {
            "labels": ["entailed by source", "not entailed by source"],
            "scores": [0.4, 0.6],  # Low entailment probability
        }

        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = mock_result
            mock_pipeline.return_value = mock_classifier

            rule = BartMNLIConsistencyRule(
                config={"threshold": 0.8, "use_pipeline": True}
            )

            source = "The company made a profit."
            summary = "The company lost money."

            result = rule.evaluate(source, summary)

            assert result["score"] == 0.0  # Below threshold
            assert result["passed"] is False
            assert result["details"]["entailment_probability"] == 0.4

    def test_evaluate_error_handling(self):
        """Test error handling during evaluation."""
        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.side_effect = Exception("Model error")
            mock_pipeline.return_value = mock_classifier

            rule = BartMNLIConsistencyRule(config={"use_pipeline": True})

            result = rule.evaluate("source", "summary")

            assert result["score"] == 0.5  # Neutral score on error
            assert result["passed"] is False
            assert "error" in result["details"]

    def test_batch_evaluate(self):
        """Test batch evaluation functionality."""
        mock_results = [
            {
                "labels": ["entailed by source", "not entailed by source"],
                "scores": [0.9, 0.1],
            },
            {
                "labels": ["entailed by source", "not entailed by source"],
                "scores": [0.3, 0.7],
            },
        ]

        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.side_effect = mock_results
            mock_pipeline.return_value = mock_classifier

            rule = BartMNLIConsistencyRule(
                config={"threshold": 0.8, "use_pipeline": True}
            )

            sources = ["Good source", "Bad source"]
            summaries = ["Good summary", "Bad summary"]

            results = rule.batch_evaluate(sources, summaries)

            assert len(results) == 2
            assert results[0]["score"] == 1.0  # Above threshold
            assert results[1]["score"] == 0.0  # Below threshold

    def test_batch_evaluate_size_mismatch(self):
        """Test batch evaluation with mismatched input sizes."""
        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock()

            rule = BartMNLIConsistencyRule()

            with pytest.raises(ValueError, match="Mismatch in batch sizes"):
                rule.batch_evaluate(["source1", "source2"], ["summary1"])

    def test_get_threshold(self):
        """Test threshold getter."""
        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock()

            rule = BartMNLIConsistencyRule(config={"threshold": 0.75})
            assert rule.get_threshold() == 0.75

    def test_update_threshold(self):
        """Test threshold update functionality."""
        with patch("rlvr_summary.rewards.bart_mnli.pipeline") as mock_pipeline:
            mock_pipeline.return_value = Mock()

            rule = BartMNLIConsistencyRule(config={"threshold": 0.8})

            # Valid threshold update
            rule.update_threshold(0.9)
            assert rule.threshold == 0.9
            assert rule.config["threshold"] == 0.9

            # Invalid threshold (should raise error)
            with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
                rule.update_threshold(1.5)

    def test_manual_pytorch_mode_mock(self):
        """Test manual PyTorch evaluation mode with mocked components."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Mock tokenizer output
        mock_inputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.encode.return_value = mock_inputs

        # Mock model output (logits for [contradiction, neutral, entailment])
        mock_logits = torch.tensor([[0.1, 0.2, 0.7]])  # High entailment
        mock_model.return_value = [mock_logits]

        with patch(
            "rlvr_summary.rewards.bart_mnli.AutoModelForSequenceClassification"
        ) as mock_model_class:
            with patch(
                "rlvr_summary.rewards.bart_mnli.AutoTokenizer"
            ) as mock_tokenizer_class:
                mock_model_class.from_pretrained.return_value = mock_model
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                rule = BartMNLIConsistencyRule(
                    config={"threshold": 0.6, "use_pipeline": False, "device": "cpu"}
                )

                result = rule.evaluate("source text", "summary text")

                # Should pass since entailment probability > threshold
                assert result["score"] == 1.0
                assert result["passed"] is True
                # Entailment probability should be softmax result for index 2
                expected_prob = torch.softmax(mock_logits, dim=1)[0, 2].item()
                assert (
                    abs(result["details"]["entailment_probability"] - expected_prob)
                    < 0.01
                )


def test_integration_with_actual_model():
    """Integration test with actual model (requires internet)."""
    # This test is marked as slow and may be skipped in CI
    pytest.skip("Skipping integration test to avoid model download in CI")

    rule = BartMNLIConsistencyRule(config={"threshold": 0.8, "use_pipeline": True})

    # Test with clearly entailed summary
    source = "The cat sat on the mat."
    summary = "A cat was on a mat."

    result = rule.evaluate(source, summary)

    # Should have reasonable entailment probability
    assert 0.0 <= result["details"]["entailment_probability"] <= 1.0
    assert isinstance(result["score"], float)
    assert result["score"] in [0.0, 1.0]  # Binary score
    assert isinstance(result["passed"], bool)
