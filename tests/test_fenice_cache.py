#!/usr/bin/env python3
"""
Test script to validate FENICE document caching functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_data_preparation():
    """Test the data preparation with FENICE caching."""
    print("Testing data preparation with FENICE caching...")

    try:
        from scripts.prepare_data_verl import create_fenice_document_cache

        # Test with sample documents
        sample_documents = [
            "John Smith works at Microsoft. He is a software engineer with 5 years of experience.",
            "The weather today is sunny with temperatures reaching 25 degrees Celsius.",
        ]

        print(f"Processing {len(sample_documents)} sample documents...")
        cache = create_fenice_document_cache(sample_documents)

        print(f"Cache created with {len(cache)} entries")

        # Validate cache structure
        for doc_idx, cache_data in cache.items():
            print(f"Document {doc_idx}: {list(cache_data.keys())}")
            if "sentences" in cache_data:
                sentences = cache_data["sentences"]
                print(f"  - {len(sentences)} sentences cached")
                if sentences:
                    print(f"  - First sentence: {sentences[0][0][:50]}...")

        return cache

    except Exception as e:
        print(f"❌ Data preparation test failed: {e}")
        return None


def test_fenice_scorer():
    """Test FENICE scorer with caching."""
    print("\nTesting FENICE scorer with caching...")

    try:
        from rlvr_summary.rewards.fenice import (
            FENICEScorer,
            get_fenice_document_cache,
            set_fenice_document_cache,
        )

        # Test cache setting/getting
        test_cache = {"test": "data"}
        set_fenice_document_cache(test_cache)
        retrieved_cache = get_fenice_document_cache()

        if retrieved_cache == test_cache:
            print("✅ Cache setting/getting works")
        else:
            print("❌ Cache setting/getting failed")

        # Test scorer creation
        scorer = FENICEScorer(weight=1.0, config={"threshold": 0.5})
        print(f"✅ FENICE scorer created: {scorer.name}")

        return True

    except Exception as e:
        print(f"❌ FENICE scorer test failed: {e}")
        return False


def test_verl_reward():
    """Test VERL reward function with cache."""
    print("\nTesting VERL reward function with cache...")

    try:
        from rlvr_summary.rewards.verl_reward import compute_score

        # Test without cache
        score = compute_score(
            data_source="cnn_dailymail",
            solution_str="Test summary.",
            ground_truth="Test article content.",
            extra_info=None,
        )
        print(f"✅ Score without cache: {score}")

        # Test with cache
        cache_data = {"0": {"sentences": [("Test sentence.", 0, 13)]}}
        score_with_cache = compute_score(
            data_source="cnn_dailymail",
            solution_str="Test summary.",
            ground_truth="Test article content.",
            extra_info={"fenice_document_cache": cache_data},
        )
        print(f"✅ Score with cache: {score_with_cache}")

        return True

    except Exception as e:
        print(f"❌ VERL reward test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing FENICE Document Caching Implementation")
    print("=" * 60)

    # Test 1: Data preparation
    cache = test_data_preparation()

    # Test 2: FENICE scorer
    scorer_ok = test_fenice_scorer()

    # Test 3: VERL reward function
    reward_ok = test_verl_reward()

    print("\n" + "=" * 60)
    if cache and scorer_ok and reward_ok:
        print("✅ All tests passed! FENICE caching implementation is working.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    print("\n📋 Summary:")
    print(f"  - Data preparation: {'✅' if cache else '❌'}")
    print(f"  - FENICE scorer: {'✅' if scorer_ok else '❌'}")
    print(f"  - VERL reward: {'✅' if reward_ok else '❌'}")


if __name__ == "__main__":
    main()
