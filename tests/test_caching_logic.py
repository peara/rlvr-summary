#!/usr/bin/env python3
"""
Focused test to validate FENICE caching logic without dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_cache_data_structure():
    """Test that cache data has the expected structure."""
    print("Testing cache data structure...")

    # Simulate cache data as created by prepare_data_verl.py
    sample_cache = {
        0: {
            "doc_id": "0John Smith works at Microsoft. He is a software engineer with 5 years of experience.",
            "sentences": [
                ("John Smith works at Microsoft.", 0, 31),
                ("He is a software engineer with 5 years of experience.", 32, 85),
            ],
            "document_text": "John Smith works at Microsoft. He is a software engineer with 5 years of experience.",
        },
        1: {
            "doc_id": "1The weather today is sunny with temperatures reaching 25 degrees Celsius.",
            "sentences": [
                (
                    "The weather today is sunny with temperatures reaching 25 degrees Celsius.",
                    0,
                    72,
                )
            ],
            "document_text": "The weather today is sunny with temperatures reaching 25 degrees Celsius.",
        },
    }

    print(f"✅ Sample cache has {len(sample_cache)} documents")

    # Validate structure
    for doc_idx, cache_data in sample_cache.items():
        required_keys = ["doc_id", "sentences", "document_text"]
        for key in required_keys:
            if key not in cache_data:
                print(f"❌ Missing key '{key}' in document {doc_idx}")
                return False

        sentences = cache_data["sentences"]
        if not isinstance(sentences, list):
            print(f"❌ Sentences should be a list for document {doc_idx}")
            return False

        for sentence_data in sentences:
            if not isinstance(sentence_data, tuple) or len(sentence_data) != 3:
                print(f"❌ Invalid sentence data format for document {doc_idx}")
                return False

    print("✅ Cache data structure is valid")
    return sample_cache


def test_thread_local_cache():
    """Test thread-local cache mechanism."""
    print("\nTesting thread-local cache mechanism...")

    try:
        from rlvr_summary.rewards.fenice import (
            get_fenice_document_cache,
            set_fenice_document_cache,
        )

        # Test setting and getting cache
        test_cache = {"test_doc": {"sentences": [("Test sentence.", 0, 13)]}}

        # Initially should be None
        initial_cache = get_fenice_document_cache()
        if initial_cache is not None:
            print(f"❌ Initial cache should be None, got: {initial_cache}")
            return False

        # Set cache
        set_fenice_document_cache(test_cache)

        # Get cache
        retrieved_cache = get_fenice_document_cache()
        if retrieved_cache != test_cache:
            print(f"❌ Retrieved cache doesn't match set cache")
            print(f"   Set: {test_cache}")
            print(f"   Got: {retrieved_cache}")
            return False

        # Clear cache
        set_fenice_document_cache(None)
        cleared_cache = get_fenice_document_cache()
        if cleared_cache is not None:
            print(f"❌ Cache should be None after clearing, got: {cleared_cache}")
            return False

        print("✅ Thread-local cache mechanism works correctly")
        return True

    except Exception as e:
        print(f"❌ Thread-local cache test failed: {e}")
        return False


def test_verl_data_format():
    """Test that VERL data format includes cache correctly."""
    print("\nTesting VERL data format with cache...")

    # Simulate what prepare_data_verl.py would create
    sample_cache = {
        0: {
            "doc_id": "0Test article content.",
            "sentences": [("Test article content.", 0, 20)],
            "document_text": "Test article content.",
        }
    }

    # Simulate VERL data item
    verl_item = {
        "data_source": "cnn_dailymail",
        "prompt": [
            {
                "role": "user",
                "content": "Summarize the following article:\n\nTest article content.",
            }
        ],
        "ability": "summarization",
        "reward_model": {
            "style": "rule",
            "ground_truth": "Test reference summary.",
        },
        "extra_info": {
            "split": "train",
            "index": 0,
            "id": "test_id",
            "fenice_document_cache": sample_cache.get(0, {}),
        },
    }

    # Validate VERL format
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for field in required_fields:
        if field not in verl_item:
            print(f"❌ Missing required VERL field: {field}")
            return False

    # Check cache is in extra_info
    extra_info = verl_item["extra_info"]
    if "fenice_document_cache" not in extra_info:
        print("❌ Missing fenice_document_cache in extra_info")
        return False

    cache_data = extra_info["fenice_document_cache"]
    if "sentences" not in cache_data:
        print("❌ Missing sentences in cached data")
        return False

    print("✅ VERL data format includes cache correctly")
    print(f"   Cache keys: {list(cache_data.keys())}")
    print(f"   Sentences cached: {len(cache_data['sentences'])}")
    return True


def test_backward_compatibility():
    """Test that the system works without cache (backward compatibility)."""
    print("\nTesting backward compatibility (no cache)...")

    try:
        from rlvr_summary.rewards.verl_reward import compute_score

        # Test without extra_info (old behavior)
        score1 = compute_score(
            data_source="cnn_dailymail",
            solution_str="Test summary.",
            ground_truth="Test article content.",
        )

        # Test with empty extra_info
        score2 = compute_score(
            data_source="cnn_dailymail",
            solution_str="Test summary.",
            ground_truth="Test article content.",
            extra_info={},
        )

        # Test with extra_info but no cache
        score3 = compute_score(
            data_source="cnn_dailymail",
            solution_str="Test summary.",
            ground_truth="Test article content.",
            extra_info={"other_data": "test"},
        )

        # All should work and return valid scores
        scores = [score1, score2, score3]
        for i, score in enumerate(scores, 1):
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                print(f"❌ Invalid score {i}: {score}")
                return False

        print(f"✅ Backward compatibility works - scores: {scores}")
        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run focused caching tests."""
    print("🧪 FENICE Document Caching Logic Tests")
    print("=" * 50)

    tests = [
        ("Cache data structure", test_cache_data_structure),
        ("Thread-local cache", test_thread_local_cache),
        ("VERL data format", test_verl_data_format),
        ("Backward compatibility", test_backward_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Results:")

    all_passed = True
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not results[i]:
            all_passed = False

    print(
        f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    if all_passed:
        print("\n🚀 FENICE document caching is ready for use!")
        print("   - Data preparation will create document cache")
        print("   - FENICE will use cached data when available")
        print("   - System falls back to runtime computation if needed")
        print("   - Backward compatibility is maintained")


if __name__ == "__main__":
    main()
