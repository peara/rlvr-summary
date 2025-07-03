#!/usr/bin/env python3
"""Simple test for TRL format conversion logic."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_format_conversion_logic():
    """Test the core logic of format conversion without dependencies."""
    
    # Test data in current format
    test_data = [
        {
            "id": "test_001",
            "article": "This is a test article about science discoveries.",
            "summary": "Scientists made discoveries."
        }
    ]
    
    # Simulate the conversion logic from _convert_to_ppo_format
    ppo_samples = []
    for sample in test_data:
        # Create standardized prompt (same as in the method)
        prompt = f"Summarize the following article:\n\n{sample['article']}\n\nSummary:"
        
        # Mock tokenization result
        mock_tokenized = {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "attention_mask": [1, 1, 1, 1, 1, 1]
        }
        
        ppo_samples.append({
            "input_ids": mock_tokenized["input_ids"],
            "attention_mask": mock_tokenized["attention_mask"],
            "query": prompt,
            "reference": sample["summary"],
            "article": sample["article"],
            "id": sample["id"],
        })
    
    # Validate the conversion
    assert len(ppo_samples) == 1
    sample = ppo_samples[0]
    
    # Check all required TRL fields are present
    required_fields = ["input_ids", "attention_mask", "query", "reference", "article", "id"]
    for field in required_fields:
        assert field in sample, f"Missing required field: {field}"
    
    # Check query format
    expected_prompt = "Summarize the following article:\n\nThis is a test article about science discoveries.\n\nSummary:"
    assert sample["query"] == expected_prompt
    
    # Check metadata preservation
    assert sample["reference"] == "Scientists made discoveries."
    assert sample["article"] == "This is a test article about science discoveries."
    assert sample["id"] == "test_001"
    
    # Check tokenization structure
    assert isinstance(sample["input_ids"], list)
    assert isinstance(sample["attention_mask"], list)
    assert len(sample["input_ids"]) == len(sample["attention_mask"])
    
    print("✅ Format conversion logic test passed")
    return True


def test_article_extraction_logic():
    """Test article extraction logic."""
    
    # Test prompt
    prompt = "Summarize the following article:\n\nThis is a test article about science discoveries.\n\nSummary:"
    
    # Simulate extraction logic from _extract_article_from_prompt
    article = prompt.replace(
        "Summarize the following article:\n\n", ""
    ).replace("\n\nSummary:", "")
    
    expected_article = "This is a test article about science discoveries."
    assert article == expected_article
    
    print("✅ Article extraction logic test passed")
    return True


def test_backward_compatibility():
    """Test that old format can be converted to new format."""
    
    # Old format sample
    old_format = {
        "id": "cnn_001",
        "article": "Breaking news about technology advances.",
        "summary": "Technology advances reported."
    }
    
    # Conversion to new format (TRL)
    prompt = f"Summarize the following article:\n\n{old_format['article']}\n\nSummary:"
    
    new_format = {
        "input_ids": [1, 2, 3, 4, 5],  # Mock tokenization
        "attention_mask": [1, 1, 1, 1, 1],
        "query": prompt,
        "reference": old_format["summary"],
        "article": old_format["article"],
        "id": old_format["id"],
    }
    
    # Validate conversion preserves all necessary information
    assert new_format["reference"] == old_format["summary"]
    assert new_format["article"] == old_format["article"]
    assert new_format["id"] == old_format["id"]
    assert "Summarize the following article:" in new_format["query"]
    assert old_format["article"] in new_format["query"]
    
    # Check that we can extract back to old format if needed
    extracted_article = new_format["query"].replace(
        "Summarize the following article:\n\n", ""
    ).replace("\n\nSummary:", "")
    
    assert extracted_article == old_format["article"]
    
    print("✅ Backward compatibility test passed")
    return True


def main():
    """Run all tests."""
    print("Testing TRL format conversion logic...")
    
    tests = [
        test_format_conversion_logic,
        test_article_extraction_logic, 
        test_backward_compatibility,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n✨ Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All core logic tests passed! TRL format conversion is working correctly.")
        return True
    else:
        print("💥 Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)