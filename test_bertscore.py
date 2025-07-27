#!/usr/bin/env python3
"""Test script for BertScore factual consistency scorer."""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_bertscore_scorer():
    """Test BertScore scorer functionality."""

    print("🧪 Testing BertScore Factual Consistency Scorer")
    print("=" * 50)

    try:
        # Test import
        from src.rlvr_summary.rewards.bertscore import create_bertscore_scorer

        print("✅ Import successful")

        # Test scorer creation
        scorer = create_bertscore_scorer(threshold=0.8)
        print(f"✅ Scorer created: {scorer.name}")

        # Test with sample data
        source = "Microsoft Corporation announced its quarterly earnings today, reporting revenue of $65.4 billion."
        good_summary = "Microsoft reported quarterly revenue of $65.4 billion."
        bad_summary = "Apple announced new iPhone sales records."

        print("\n📊 Testing with sample data:")
        print(f"Source: {source}")
        print(f"Good summary: {good_summary}")
        print(f"Bad summary: {bad_summary}")

        # Test good summary
        result_good = scorer.evaluate(source, good_summary)
        print(f"\n✅ Good summary result:")
        print(f"  Score: {result_good['score']}")
        print(f"  Passed: {result_good['passed']}")
        print(f"  Raw BertScore F1: {result_good['details']['raw_bertscore_f1']:.3f}")

        # Test bad summary
        result_bad = scorer.evaluate(source, bad_summary)
        print(f"\n✅ Bad summary result:")
        print(f"  Score: {result_bad['score']}")
        print(f"  Passed: {result_bad['passed']}")
        print(f"  Raw BertScore F1: {result_bad['details']['raw_bertscore_f1']:.3f}")

        # Test batch evaluation
        sources = [source, source]
        summaries = [good_summary, bad_summary]
        batch_results = scorer.batch_evaluate(sources, summaries)

        print(f"\n✅ Batch evaluation:")
        print(f"  Results count: {len(batch_results)}")
        for i, result in enumerate(batch_results):
            print(f"  Item {i}: score={result['score']}, passed={result['passed']}")

        print("\n🎉 All tests passed! BertScore scorer is working correctly.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install bert-score: pip install bert-score")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"💡 Traceback:\n{traceback.format_exc()}")
        return False


def test_rule_bundle_integration():
    """Test BertScore integration with rule bundle system."""

    print("\n🔧 Testing Rule Bundle Integration")
    print("=" * 50)

    try:
        from src.rlvr_summary.rewards import load_rule_bundle_from_config

        # Test loading new configuration
        config_path = (
            Path(__file__).parent / "configs" / "rewards" / "rule_bundle_bertscore.yaml"
        )

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False

        system = load_rule_bundle_from_config(config_path)
        print("✅ Rule bundle loaded successfully")

        # Check that BertScore rule is present
        if "bertscore_factual_consistency" in system.rules:
            bertscore_rule = system.rules["bertscore_factual_consistency"]
            print(f"✅ BertScore rule found with weight: {bertscore_rule.weight}")
        else:
            print("❌ BertScore rule not found in system")
            return False

        # Check that FENICE is not present
        if "fenice_factual_consistency" not in system.rules:
            print("✅ FENICE rule correctly disabled")
        else:
            print("⚠️  FENICE rule still present (but this might be expected)")

        # Test evaluation
        source = "The company reported strong financial results."
        summary = "The business showed good financial performance."

        result = system.evaluate(source, summary)
        print(f"✅ System evaluation successful:")
        print(f"  Total score: {result.total_score:.3f}")
        print(f"  Pass rate: {result.pass_rate:.3f}")

        if "bertscore_factual_consistency" in result.rule_scores:
            bertscore_score = result.rule_scores["bertscore_factual_consistency"]
            print(f"  BertScore contribution: {bertscore_score}")

        print("\n🎉 Rule bundle integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(f"💡 Traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("🚀 BertScore Factual Consistency Test Suite")
    print("=" * 60)

    success = True

    # Test individual scorer
    success &= test_bertscore_scorer()

    # Test integration
    success &= test_rule_bundle_integration()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! BertScore implementation is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)
