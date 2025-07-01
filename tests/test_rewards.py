"""Tests for the rule-based reward system."""

import sys
from pathlib import Path
import tempfile
import yaml

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.rewards import (
    RuleBundleRewardSystem,
    LengthConstraintRule,
    EntityOverlapRule,
    NumberConsistencyRule,
    ProfanityDetectionRule,
    FluencyRule,
    create_default_rule_bundle,
    load_rule_bundle_from_config,
    TextProcessor,
)


class TestTextProcessor:
    """Test TextProcessor utility functions."""
    
    def test_extract_words(self):
        """Test word extraction."""
        text = "Hello, World! This is a test 123."
        words = TextProcessor.extract_words(text)
        expected = ["hello", "world", "this", "is", "a", "test"]
        assert words == expected
    
    def test_extract_numbers(self):
        """Test number extraction."""
        text = "There are 42 items, 3.14 pi, and 95% success rate."
        numbers = TextProcessor.extract_numbers(text)
        expected = ["95%", "42", "3.14"]  # Percentages first, then standalone numbers
        assert numbers == expected
    
    def test_extract_entities(self):
        """Test entity extraction."""
        text = "John Smith visited New York City and met with Microsoft."
        entities = TextProcessor.extract_entities(text)
        # Should find capitalized words/phrases
        assert "John Smith" in entities
        assert "New York City" in entities
        assert "Microsoft" in entities
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        similarity = TextProcessor.jaccard_similarity(set1, set2)
        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert abs(similarity - 0.5) < 0.001
        
        # Test empty sets
        similarity = TextProcessor.jaccard_similarity(set(), set())
        assert similarity == 1.0


class TestLengthConstraintRule:
    """Test length constraint rule."""
    
    def test_optimal_length(self):
        """Test scoring for optimal length."""
        config = {
            "min_words": 20,
            "max_words": 100,
            "optimal_range": [30, 50],
            "penalty_factor": 0.5,
        }
        rule = LengthConstraintRule(weight=1.0, config=config)
        
        # Create text with 40 words (in optimal range)
        summary = " ".join(["word"] * 40)
        result = rule.evaluate("source", summary)
        
        assert result["score"] == 1.0
        assert result["passed"] is True
        assert result["details"]["word_count"] == 40
        assert result["details"]["in_optimal_range"] is True
    
    def test_acceptable_length(self):
        """Test scoring for acceptable but not optimal length."""
        config = {
            "min_words": 20,
            "max_words": 100,
            "optimal_range": [30, 50],
            "penalty_factor": 0.5,
        }
        rule = LengthConstraintRule(weight=1.0, config=config)
        
        # Create text with 25 words (acceptable but below optimal)
        summary = " ".join(["word"] * 25)
        result = rule.evaluate("source", summary)
        
        assert 0.5 <= result["score"] < 1.0  # Should have penalty
        assert result["passed"] is True
        assert result["details"]["in_optimal_range"] is False
        assert result["details"]["in_acceptable_range"] is True
    
    def test_unacceptable_length(self):
        """Test scoring for unacceptable length."""
        config = {
            "min_words": 20,
            "max_words": 100,
            "optimal_range": [30, 50],
            "penalty_factor": 0.5,
        }
        rule = LengthConstraintRule(weight=1.0, config=config)
        
        # Create text with 10 words (below minimum)
        summary = " ".join(["word"] * 10)
        result = rule.evaluate("source", summary)
        
        assert result["score"] <= 0.2  # Should have very low score
        assert result["passed"] is False
        assert result["details"]["in_acceptable_range"] is False


class TestEntityOverlapRule:
    """Test entity overlap rule."""
    
    def test_good_entity_overlap(self):
        """Test scoring for good entity overlap."""
        config = {
            "min_overlap": 0.3,
            "optimal_overlap": 0.7,
        }
        rule = EntityOverlapRule(weight=1.0, config=config)
        
        source = "John Smith works at Microsoft in Seattle."
        summary = "John Smith is employed by Microsoft."
        
        result = rule.evaluate(source, summary)
        
        # Should have good overlap for John Smith and Microsoft
        assert result["score"] > 0.5
        assert result["passed"] is True
        assert "John Smith" in result["details"]["source_entities"]
        assert "Microsoft" in result["details"]["source_entities"]
    
    def test_no_entities(self):
        """Test fallback to word overlap when no entities."""
        config = {
            "min_overlap": 0.3,
            "optimal_overlap": 0.7,
        }
        rule = EntityOverlapRule(weight=1.0, config=config)
        
        source = "the quick brown fox jumps over the lazy dog"
        summary = "quick brown fox jumps over dog"
        
        result = rule.evaluate(source, summary)
        
        assert result["details"]["fallback_used"] is True
        assert result["score"] > 0.3  # Should have reasonable word overlap


class TestNumberConsistencyRule:
    """Test number consistency rule."""
    
    def test_exact_number_match(self):
        """Test scoring for exact number matches."""
        config = {
            "exact_match_bonus": 1.0,
            "mismatch_penalty": -0.5,
        }
        rule = NumberConsistencyRule(weight=1.0, config=config)
        
        source = "The company reported 42 million in revenue and 15% growth."
        summary = "Revenue was 42 million with 15% growth."
        
        result = rule.evaluate(source, summary)
        
        assert result["score"] >= 0.8  # Should score well for exact matches
        assert result["passed"] is True
        assert "42" in result["details"]["exact_matches"]
        assert "15%" in result["details"]["exact_matches"]
    
    def test_number_mismatch(self):
        """Test penalty for number mismatches."""
        config = {
            "exact_match_bonus": 1.0,
            "mismatch_penalty": -0.5,
        }
        rule = NumberConsistencyRule(weight=1.0, config=config)
        
        source = "The company reported 42 million in revenue."
        summary = "Revenue was 50 million."  # Wrong number
        
        result = rule.evaluate(source, summary)
        
        assert result["score"] <= 0.5  # Should have penalty
        assert result["passed"] is False
        assert "50" in result["details"]["mismatches"]
    
    def test_no_numbers_in_summary(self):
        """Test neutral score when no numbers in summary."""
        config = {}
        rule = NumberConsistencyRule(weight=1.0, config=config)
        
        source = "The company reported 42 million in revenue."
        summary = "The company reported revenue."
        
        result = rule.evaluate(source, summary)
        
        assert result["score"] == 0.7  # Neutral score
        assert result["passed"] is True  # No mismatches
        assert result["details"]["no_numbers_in_summary"] is True


class TestProfanityDetectionRule:
    """Test profanity detection rule."""
    
    def test_clean_text(self):
        """Test scoring for clean text."""
        config = {"enabled": True, "penalty": -1.0}
        rule = ProfanityDetectionRule(weight=1.0, config=config)
        
        summary = "This is a clean and professional summary."
        result = rule.evaluate("source", summary)
        
        assert result["score"] == 1.0
        assert result["passed"] is True
        assert len(result["details"]["profanity_found"]) == 0
    
    def test_profanity_detected(self):
        """Test penalty for profanity."""
        config = {"enabled": True, "penalty": -1.0}
        rule = ProfanityDetectionRule(weight=1.0, config=config)
        
        summary = "This is a damn bad summary."
        result = rule.evaluate("source", summary)
        
        assert result["score"] < 1.0
        assert result["passed"] is False
        assert "damn" in result["details"]["profanity_found"]
    
    def test_disabled_profanity_check(self):
        """Test disabled profanity detection."""
        config = {"enabled": False}
        rule = ProfanityDetectionRule(weight=1.0, config=config)
        
        summary = "This is a damn bad summary."
        result = rule.evaluate("source", summary)
        
        assert result["score"] == 1.0
        assert result["passed"] is True
        assert result["details"]["enabled"] is False


class TestFluencyRule:
    """Test fluency rule."""
    
    def test_good_fluency(self):
        """Test scoring for good fluency."""
        config = {"enabled": True, "min_score": 0.5}
        rule = FluencyRule(weight=1.0, config=config)
        
        summary = "This is a well-written summary with appropriate length sentences."
        result = rule.evaluate("source", summary)
        
        assert result["score"] >= 0.8
        assert result["passed"] is True
    
    def test_poor_fluency(self):
        """Test penalties for poor fluency indicators."""
        config = {"enabled": True, "min_score": 0.5}
        rule = FluencyRule(weight=1.0, config=config)
        
        # Very short sentences
        summary = "Bad. Text. Here."
        result = rule.evaluate("source", summary)
        
        assert result["score"] < 1.0
        assert "penalty_short_sentences" in result["details"]
    
    def test_disabled_fluency(self):
        """Test disabled fluency checking."""
        config = {"enabled": False}
        rule = FluencyRule(weight=1.0, config=config)
        
        summary = "Bad. Text. Here."
        result = rule.evaluate("source", summary)
        
        assert result["score"] == 1.0
        assert result["passed"] is True


class TestRuleBundleRewardSystem:
    """Test the main rule bundle system."""
    
    def test_default_system(self):
        """Test default rule bundle system."""
        system = create_default_rule_bundle()
        
        assert len(system.rules) == 5
        assert "length_constraint" in system.rules
        assert "entity_overlap" in system.rules
        assert "number_consistency" in system.rules
        assert "profanity_penalty" in system.rules
        assert "fluency" in system.rules
    
    def test_evaluation(self):
        """Test complete system evaluation."""
        system = create_default_rule_bundle()
        
        source = "John Smith, CEO of Microsoft, reported 42 million in revenue with 15% growth in Seattle."
        summary = "Microsoft CEO John Smith announced 42 million revenue and 15% growth."
        
        result = system.evaluate(source, summary, log_details=True)
        
        assert 0.0 <= result.total_score <= 1.0
        assert 0.0 <= result.pass_rate <= 1.0
        assert len(result.rule_scores) == 5
        assert len(result.rule_passed) == 5
        
        # Should pass most rules for this reasonable summary
        assert result.pass_rate >= 0.6
    
    def test_batch_evaluation(self):
        """Test batch evaluation."""
        system = create_default_rule_bundle()
        
        sources = [
            "John Smith works at Microsoft.",
            "The company has 1000 employees.",
        ]
        summaries = [
            "John Smith is at Microsoft.",
            "Company employs 1000 people.",
        ]
        
        results = system.evaluate_batch(sources, summaries, log_details=True)
        
        assert len(results) == 2
        for result in results:
            assert 0.0 <= result.total_score <= 1.0
            assert 0.0 <= result.pass_rate <= 1.0
    
    def test_rule_info(self):
        """Test getting rule information."""
        system = create_default_rule_bundle()
        info = system.get_rule_info()
        
        assert len(info) == 5
        for rule_name, rule_info in info.items():
            assert "weight" in rule_info
            assert "threshold" in rule_info
            assert "config" in rule_info
            assert "class_name" in rule_info
    
    def test_weight_update(self):
        """Test updating rule weights."""
        system = create_default_rule_bundle()
        
        original_weight = system.rules["length_constraint"].weight
        new_weights = {"length_constraint": 0.5}
        
        system.update_rule_weights(new_weights)
        
        assert system.rules["length_constraint"].weight == 0.5
        assert system.rules["length_constraint"].weight != original_weight
    
    def test_config_loading(self):
        """Test loading from config file."""
        config = {
            "weights": {
                "length_constraint": 0.4,
                "entity_overlap": 0.3,
                "number_consistency": 0.3,
            },
            "length": {
                "min_words": 10,
                "max_words": 50,
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            system = load_rule_bundle_from_config(config_path)
            assert len(system.rules) == 3
            assert system.rules["length_constraint"].weight == 0.4
            assert system.rules["length_constraint"].config["min_words"] == 10
        finally:
            Path(config_path).unlink()
    
    def test_error_handling(self):
        """Test error handling in evaluation."""
        # Create system with no rules
        system = RuleBundleRewardSystem({})
        
        result = system.evaluate("source", "summary")
        
        assert result.total_score == 0.5  # Neutral score when no rules
        assert result.pass_rate == 0.0
        assert len(result.rule_scores) == 0


def run_all_tests():
    """Run all tests manually."""
    print("Running reward system tests...")
    
    # Test TextProcessor
    print("Testing TextProcessor...")
    test_tp = TestTextProcessor()
    test_tp.test_extract_words()
    test_tp.test_extract_numbers()
    test_tp.test_extract_entities()
    test_tp.test_jaccard_similarity()
    print("✓ TextProcessor tests passed")
    
    # Test LengthConstraintRule
    print("Testing LengthConstraintRule...")
    test_length = TestLengthConstraintRule()
    test_length.test_optimal_length()
    test_length.test_acceptable_length()
    test_length.test_unacceptable_length()
    print("✓ LengthConstraintRule tests passed")
    
    # Test EntityOverlapRule
    print("Testing EntityOverlapRule...")
    test_entity = TestEntityOverlapRule()
    test_entity.test_good_entity_overlap()
    test_entity.test_no_entities()
    print("✓ EntityOverlapRule tests passed")
    
    # Test NumberConsistencyRule
    print("Testing NumberConsistencyRule...")
    test_number = TestNumberConsistencyRule()
    test_number.test_exact_number_match()
    test_number.test_number_mismatch()
    test_number.test_no_numbers_in_summary()
    print("✓ NumberConsistencyRule tests passed")
    
    # Test ProfanityDetectionRule
    print("Testing ProfanityDetectionRule...")
    test_profanity = TestProfanityDetectionRule()
    test_profanity.test_clean_text()
    test_profanity.test_profanity_detected()
    test_profanity.test_disabled_profanity_check()
    print("✓ ProfanityDetectionRule tests passed")
    
    # Test FluencyRule
    print("Testing FluencyRule...")
    test_fluency = TestFluencyRule()
    test_fluency.test_good_fluency()
    test_fluency.test_poor_fluency()
    test_fluency.test_disabled_fluency()
    print("✓ FluencyRule tests passed")
    
    # Test RuleBundleRewardSystem
    print("Testing RuleBundleRewardSystem...")
    test_system = TestRuleBundleRewardSystem()
    test_system.test_default_system()
    test_system.test_evaluation()
    test_system.test_batch_evaluation()
    test_system.test_rule_info()
    test_system.test_weight_update()
    test_system.test_config_loading()
    test_system.test_error_handling()
    print("✓ RuleBundleRewardSystem tests passed")
    
    print("\n✅ All reward system tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()