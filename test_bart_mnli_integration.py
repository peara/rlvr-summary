"""Simple test to verify BART-MNLI rule works with real model."""

import sys
from pathlib import Path

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent / "src"))

from rlvr_summary.rewards.bart_mnli import BartMNLIConsistencyRule
from rlvr_summary.rewards.rule_bundle import RuleBundleRewardSystem


def test_bart_mnli_rule_basic():
    """Test basic BART-MNLI rule functionality with real model."""
    print("🧪 Testing BART-MNLI Rule Basic Functionality")
    print("⬇️ This will download facebook/bart-large-mnli model on first run...")
    
    # Test rule creation
    rule = BartMNLIConsistencyRule(
        weight=1.0,
        config={"threshold": 0.8}
    )
    
    print(f"✅ Rule created with threshold: {rule.threshold}")
    
    # Test cases with different levels of entailment
    test_cases = [
        {
            "name": "High Entailment",
            "source": "The company reported profits of $1 billion for Q3 2023.",
            "summary": "The company made $1 billion in profits in Q3 2023.",
            "expected_high": True
        },
        {
            "name": "Low Entailment", 
            "source": "The company reported profits of $1 billion for Q3 2023.",
            "summary": "The company lost money and filed for bankruptcy.",
            "expected_high": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        result = rule.evaluate(test_case["source"], test_case["summary"])
        
        entailment_prob = result["details"]["entailment_probability"]
        binary_score = result["score"]
        passed = result["passed"]
        
        print(f"   📊 Entailment Probability: {entailment_prob:.3f}")
        print(f"   📈 Binary Score: {binary_score}")
        print(f"   ✔️ Passed (>= 0.8): {passed}")
        
        # Validate expectations
        if test_case["expected_high"]:
            assert entailment_prob > 0.5, f"Expected high entailment, got {entailment_prob}"
        else:
            assert entailment_prob < 0.8, f"Expected low entailment, got {entailment_prob}"
    
    # Test batch evaluation
    sources = [case["source"] for case in test_cases]
    summaries = [case["summary"] for case in test_cases]
    
    batch_results = rule.batch_evaluate(sources, summaries)
    print(f"\n✅ Batch evaluation: {len(batch_results)} results")
    
    assert len(batch_results) == len(test_cases)


def test_rule_bundle_integration():
    """Test BART-MNLI integration with rule bundle system."""
    print("\n🔧 Testing Rule Bundle Integration")
    
    # Configuration using BART-MNLI
    config = {
        "weights": {
            "length_constraint": 0.30,
            "bart_mnli_factual_consistency": 0.35,
            "entity_overlap": 0.35,
        },
        "bart_mnli": {
            "threshold": 0.8
        },
        "length": {"min_words": 10, "max_words": 50},
        "entity": {"min_overlap": 0.3}
    }
    
    # Create rule bundle
    rule_bundle = RuleBundleRewardSystem(config)
    print(f"✅ Rule bundle created with {len(rule_bundle.rules)} rules")
    
    # Test evaluation
    source = "Tesla sold 100,000 cars last quarter."
    summary = "Tesla's quarterly car sales reached 100,000 units."
    
    result = rule_bundle.evaluate(source, summary)
    
    print(f"✅ Bundle evaluation successful:")
    print(f"   Total Score: {result.total_score:.3f}")
    print(f"   Pass Rate: {result.pass_rate:.3f}")
    print(f"   BART-MNLI Score: {result.rule_scores.get('bart_mnli_factual_consistency', 'N/A')}")
    
    # Verify BART-MNLI rule is present and working
    assert "bart_mnli_factual_consistency" in result.rule_scores
    bart_score = result.rule_scores["bart_mnli_factual_consistency"]
    assert bart_score in [0.0, 1.0], f"Expected binary score, got {bart_score}"


def test_threshold_behavior():
    """Test different threshold values."""
    print("\n🎚️ Testing Threshold Behavior")
    
    source = "Apple reported revenue of $90 billion in Q4 2023."
    summary = "Apple's Q4 2023 revenue was $90 billion."
    
    thresholds = [0.6, 0.8, 0.9]
    
    for threshold in thresholds:
        rule = BartMNLIConsistencyRule(
            weight=1.0,
            config={"threshold": threshold}
        )
        
        result = rule.evaluate(source, summary)
        entailment_prob = result["details"]["entailment_probability"]
        binary_score = result["score"]
        
        print(f"   Threshold {threshold}: prob={entailment_prob:.3f}, score={binary_score}")
        
        # Verify binary scoring logic
        expected_score = 1.0 if entailment_prob >= threshold else 0.0
        assert binary_score == expected_score, f"Binary scoring failed for threshold {threshold}"


if __name__ == "__main__":
    print("🚀 BART-MNLI Rule Integration Test")
    print("Testing with real facebook/bart-large-mnli model...")
    print("Note: This will download the model on first run (~1.6GB)")
    print()
    
    try:
        test_bart_mnli_rule_basic()
        test_rule_bundle_integration()
        test_threshold_behavior()
        
        print("\n✅ All tests passed!")
        print("\n💡 The BART-MNLI rule implementation is working correctly.")
        print("   Ready for integration into VERL training pipeline.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
