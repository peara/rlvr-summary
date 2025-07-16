"""Tests for FENICE scorer and combined reward system."""

import sys
from pathlib import Path
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.rewards import (
    FENICEScorer,
    create_fenice_scorer,
    CombinedRewardSystem,
    create_combined_reward_system,
)


class TestFENICEScorer:
    """Test FENICE factual consistency scorer."""

    def test_scorer_creation(self):
        """Test FENICE scorer creation."""
        scorer = create_fenice_scorer()
        assert scorer.threshold == 0.5
        assert scorer.batch_size == 8
        assert scorer._fenice_model is None  # Model not loaded yet
        assert not scorer._model_loaded

    def test_scorer_fails_fast_without_models(self):
        """Test FENICE scorer fails immediately when models can't be loaded."""
        scorer = create_fenice_scorer()
        
        source = "John Smith works at Microsoft."
        summary = "John Smith is at Microsoft."
        
        # Should fail fast when trying to load FENICE models (network or dependency issue)
        with pytest.raises((OSError, ModuleNotFoundError, ImportError, RuntimeError)):
            scorer.evaluate(source, summary)

    def test_empty_input_validation(self):
        """Test scorer with empty input validates immediately."""
        scorer = create_fenice_scorer()
        
        # Should fail fast on empty input validation before trying to load models
        with pytest.raises(ValueError, match="Empty input"):
            scorer.evaluate("", "summary")
            
        with pytest.raises(ValueError, match="Empty input"):
            scorer.evaluate("source", "")

    def test_batch_evaluation_fails_fast(self):
        """Test batch evaluation fails fast."""
        scorer = create_fenice_scorer()
        
        sources = ["Source 1", "Source 2"]
        summaries = ["Summary 1", "Summary 2"]
        
        # Should fail fast when trying to process
        with pytest.raises((OSError, ModuleNotFoundError, ImportError, RuntimeError)):
            scorer.batch_evaluate(sources, summaries)

    def test_claim_extraction_fallback_removed(self):
        """Test that FENICE now uses the real package instead of fallbacks."""
        scorer = create_fenice_scorer()
        
        # The old _simple_sentence_split method should no longer exist
        assert not hasattr(scorer, '_simple_sentence_split')
        
        # The scorer should use the real FENICE package
        assert hasattr(scorer, '_fenice_model')
        assert hasattr(scorer, '_load_model')


class TestCombinedRewardSystem:
    """Test combined FENICE + rule-based reward system."""

    def test_default_weights(self):
        """Test system creation with default weights."""
        system = create_combined_reward_system()
        
        assert system.fenice_weight == 0.7
        assert system.rule_weight == 0.3
        assert system.threshold == 0.5

    def test_custom_weights(self):
        """Test system creation with custom weights."""
        system = create_combined_reward_system(
            fenice_weight=0.8,
            rule_weight=0.2,
            threshold=0.6
        )
        
        assert system.fenice_weight == 0.8
        assert system.rule_weight == 0.2
        assert system.threshold == 0.6

    def test_weight_normalization(self):
        """Test weight normalization when weights don't sum to 1."""
        system = CombinedRewardSystem(
            fenice_weight=0.6,
            rule_weight=0.6,  # Total = 1.2, should be normalized
        )
        
        # Weights should be normalized to sum to 1.0
        assert abs(system.fenice_weight - 0.5) < 0.01
        assert abs(system.rule_weight - 0.5) < 0.01

    def test_evaluation_fails_fast(self):
        """Test combined evaluation fails fast when FENICE fails."""
        system = create_combined_reward_system()
        
        source = "John Smith works at Microsoft with 1000 employees."
        summary = "John Smith is at Microsoft which has 1000 workers."
        
        # Should fail fast when FENICE scorer fails
        with pytest.raises((OSError, ModuleNotFoundError, ImportError, RuntimeError)):
            system.evaluate(source, summary)

    def test_batch_evaluation_fails_fast(self):
        """Test batch evaluation fails fast."""
        system = create_combined_reward_system()
        
        sources = ["Source 1", "Source 2"]  
        summaries = ["Summary 1", "Summary 2"]
        
        # Should fail fast when FENICE scorer fails
        with pytest.raises((OSError, ModuleNotFoundError, ImportError, RuntimeError)):
            system.evaluate_batch(sources, summaries)

    def test_weight_update(self):
        """Test weight updating functionality."""
        system = create_combined_reward_system()
        
        system.update_weights(0.6, 0.4)
        
        assert system.fenice_weight == 0.6
        assert system.rule_weight == 0.4

    def test_fenice_configuration(self):
        """Test FENICE configuration updating."""
        system = create_combined_reward_system()
        
        system.configure_fenice(threshold=0.8, batch_size=16)
        
        assert system.fenice_scorer.config["threshold"] == 0.8
        assert system.fenice_scorer.config["batch_size"] == 16

    def test_system_info(self):
        """Test system information retrieval."""
        system = create_combined_reward_system()
        
        info = system.get_system_info()
        
        assert info["type"] == "CombinedRewardSystem"
        assert info["fenice_weight"] == 0.7
        assert info["rule_weight"] == 0.3
        assert info["threshold"] == 0.5
        assert "fenice_config" in info
        assert "rule_system_info" in info

    def test_metrics_extraction_structure(self):
        """Test metrics structure without actual evaluation."""
        # We can't test actual metrics extraction because FENICE will fail
        # But we can test the result structure components
        from rlvr_summary.rewards.combined import CombinedRewardResult
        from rlvr_summary.rewards.base import RuleEvaluationResult
        
        # Create a mock result
        rule_result = RuleEvaluationResult(
            total_score=0.8,
            rule_scores={"test_rule": 0.8},
            rule_passed={"test_rule": True},
            rule_details={"test_rule": {}},
            pass_rate=1.0
        )
        
        combined_result = CombinedRewardResult(
            total_score=0.75,
            fenice_score=0.7,
            rule_score=0.8,
            fenice_weight=0.7,
            rule_weight=0.3,
            fenice_details={"num_claims": 3},
            rule_result=rule_result,
            passed=True
        )
        
        metrics = combined_result.get_metrics()
        
        # Check key metrics exist
        assert "reward/total_score" in metrics
        assert "reward/fenice_score" in metrics
        assert "reward/rule_score" in metrics
        assert "reward/fenice_num_claims" in metrics
        assert metrics["reward/fenice_score"] == 0.7
        assert metrics["reward/rule_score"] == 0.8
        assert metrics["reward/fenice_num_claims"] == 3


def run_fenice_tests():
    """Run all FENICE-related tests manually."""
    print("Running FENICE scorer tests...")
    
    # Test FENICEScorer
    print("Testing FENICEScorer...")
    test_fenice = TestFENICEScorer()
    test_fenice.test_scorer_creation()
    print("✓ FENICEScorer creation test passed")
    
    # Test CombinedRewardSystem
    print("Testing CombinedRewardSystem...")
    test_combined = TestCombinedRewardSystem()
    test_combined.test_default_weights()
    test_combined.test_custom_weights()
    test_combined.test_weight_normalization()
    test_combined.test_weight_update()
    test_combined.test_fenice_configuration()
    test_combined.test_system_info()
    test_combined.test_metrics_extraction_structure()
    print("✓ CombinedRewardSystem tests passed")
    
    print("\n✅ All FENICE-related structural tests passed successfully!")
    print("Note: Tests that require model loading will fail fast as expected.")


if __name__ == "__main__":
    run_fenice_tests()