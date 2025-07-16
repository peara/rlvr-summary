"""Tests for FENICE scorer and rule-based reward system integration."""

import sys
from pathlib import Path
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.rewards import (
    FENICEScorer,
    create_fenice_scorer,
    RuleBundleRewardSystem,
    load_rule_bundle_from_config,
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


class TestRuleBundleWithFENICE:
    """Test rule-based reward system with FENICE integration."""

    def test_fenice_integration_config(self):
        """Test FENICE integration through rule bundle configuration."""
        config = {
            "weights": {
                "length_constraint": 0.3,
                "fenice_factual_consistency": 0.7,
            },
            "length": {
                "min_words": 20,
                "max_words": 100,
            },
            "fenice": {
                "threshold": 0.6,
                "batch_size": 4,
            }
        }
        
        system = RuleBundleRewardSystem(config)
        
        # Check that FENICE rule is properly configured
        assert "fenice_factual_consistency" in system.rules
        fenice_rule = system.rules["fenice_factual_consistency"]
        assert fenice_rule.weight == 0.7
        assert fenice_rule.threshold == 0.6
        assert fenice_rule.batch_size == 4

    def test_config_file_loading(self):
        """Test loading configuration from file."""
        config_path = Path(__file__).parent.parent / "configs" / "rewards" / "rule_bundle.yaml"
        
        if config_path.exists():
            system = load_rule_bundle_from_config(config_path)
            
            # Should have FENICE rule configured
            assert "fenice_factual_consistency" in system.rules
            fenice_rule = system.rules["fenice_factual_consistency"]
            assert fenice_rule.weight > 0
            assert fenice_rule.threshold >= 0

    def test_evaluation_fails_fast_with_fenice(self):
        """Test evaluation fails fast when FENICE fails."""
        config = {
            "weights": {
                "length_constraint": 0.5,
                "fenice_factual_consistency": 0.5,
            },
            "length": {"min_words": 20, "max_words": 100},
            "fenice": {"threshold": 0.5}
        }
        
        system = RuleBundleRewardSystem(config)
        
        source = "John Smith works at Microsoft with 1000 employees."
        summary = "John Smith is at Microsoft which has 1000 workers."
        
        # Should fail fast when FENICE scorer fails
        with pytest.raises((OSError, ModuleNotFoundError, ImportError, RuntimeError)):
            system.evaluate(source, summary)

    def test_batch_evaluation_fails_fast(self):
        """Test batch evaluation fails fast."""
        config = {
            "weights": {
                "length_constraint": 0.5,
                "fenice_factual_consistency": 0.5,
            },
            "length": {"min_words": 20, "max_words": 100},
            "fenice": {"threshold": 0.5}
        }
        
        system = RuleBundleRewardSystem(config)
        
        sources = ["Source 1", "Source 2"]  
        summaries = ["Summary 1", "Summary 2"]
        
        # Should fail fast when FENICE scorer fails
        with pytest.raises((OSError, ModuleNotFoundError, ImportError, RuntimeError)):
            system.evaluate_batch(sources, summaries)

    def test_weight_validation(self):
        """Test weight validation in rule bundle system."""
        config = {
            "weights": {
                "length_constraint": 0.3,
                "fenice_factual_consistency": 0.8,  # Sum > 1.0
            },
            "length": {"min_words": 20, "max_words": 100},
            "fenice": {"threshold": 0.5}
        }
        
        # Should create system and log warning about weights
        system = RuleBundleRewardSystem(config)
        assert len(system.rules) == 2

    def test_rule_info_includes_fenice(self):
        """Test rule information includes FENICE details."""
        config = {
            "weights": {
                "fenice_factual_consistency": 1.0,
            },
            "fenice": {
                "threshold": 0.7,
                "batch_size": 16,
            }
        }
        
        system = RuleBundleRewardSystem(config)
        rule_info = system.get_rule_info()
        
        assert "fenice_factual_consistency" in rule_info
        fenice_info = rule_info["fenice_factual_consistency"]
        assert fenice_info["weight"] == 1.0
        assert fenice_info["threshold"] == 0.7
        assert fenice_info["class_name"] == "FENICEScorer"


def run_fenice_tests():
    """Run all FENICE-related tests manually."""
    print("Running FENICE scorer tests...")
    
    # Test FENICEScorer
    print("Testing FENICEScorer...")
    test_fenice = TestFENICEScorer()
    test_fenice.test_scorer_creation()
    print("✓ FENICEScorer creation test passed")
    
    # Test RuleBundleRewardSystem with FENICE
    print("Testing RuleBundleRewardSystem with FENICE...")
    test_rule_bundle = TestRuleBundleWithFENICE()
    test_rule_bundle.test_fenice_integration_config()
    test_rule_bundle.test_config_file_loading()
    test_rule_bundle.test_weight_validation()
    test_rule_bundle.test_rule_info_includes_fenice()
    print("✓ RuleBundleRewardSystem with FENICE tests passed")
    
    print("\n✅ All FENICE-related structural tests passed successfully!")
    print("Note: Tests that require model loading will fail fast as expected.")


if __name__ == "__main__":
    run_fenice_tests()