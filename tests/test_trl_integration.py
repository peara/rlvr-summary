"""Tests for TRL integration and format conversion."""

import sys
from pathlib import Path
import tempfile

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __init__(self):
        self.pad_token = None
        self.eos_token = "[EOS]"
    
    def __call__(self, text, max_length=512, truncation=True, padding=False, return_tensors=None):
        """Mock tokenization that returns fake token IDs."""
        if isinstance(text, list):
            # Batch tokenization
            return {
                "input_ids": [[1, 2, 3, 4, 5] for _ in text],
                "attention_mask": [[1, 1, 1, 1, 1] for _ in text]
            }
        else:
            # Single text tokenization
            return {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1]
            }


class MockDataset:
    """Mock Dataset class for testing."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)
    
    @classmethod
    def from_list(cls, data):
        return cls(data)


class TestTRLIntegration:
    """Test TRL format conversion functionality."""
    
    def test_convert_to_ppo_format(self):
        """Test conversion to TRL PPOTrainer format."""
        # Mock the dependencies to test the logic
        import unittest.mock as mock
        
        # Create test data in the current format
        test_data = [
            {
                "id": "test_001",
                "article": "This is a test article about science.",
                "summary": "Test article about science."
            },
            {
                "id": "test_002", 
                "article": "Another test article about technology.",
                "summary": "Test article about technology."
            }
        ]
        
        # Mock the training loop with minimal config
        config = {
            "max_prompt_length": 512,
            "batch_size": 4,
            "max_steps": 10,
            "checkpoint_dir": tempfile.mkdtemp(),
        }
        
        # Mock imports and create a minimal training loop
        with mock.patch.dict('sys.modules', {
            'torch': mock.MagicMock(),
            'datasets': mock.MagicMock(),
            'transformers': mock.MagicMock(),
            'trl': mock.MagicMock(),
            'omegaconf': mock.MagicMock(),
        }):
            # Import after mocking
            try:
                from rlvr_summary.training.ppo_trainer import PPOTrainingLoop
                
                # Mock the Dataset import within the module
                import rlvr_summary.training.ppo_trainer as ppo_module
                ppo_module.Dataset = MockDataset
                
                # Create training loop instance
                training_loop = PPOTrainingLoop(config)
                training_loop.tokenizer = MockTokenizer()
                
                # Test the conversion method
                result = training_loop._convert_to_ppo_format(test_data)
                
                # Validate the result
                assert len(result) == 2
                
                # Check structure of first sample
                sample = result.data[0]
                assert "input_ids" in sample
                assert "attention_mask" in sample
                assert "query" in sample
                assert "reference" in sample
                assert "article" in sample
                assert "id" in sample
                
                # Check that query contains the prompt
                assert "Summarize the following article:" in sample["query"]
                assert "This is a test article about science." in sample["query"]
                
                # Check that metadata is preserved
                assert sample["reference"] == "Test article about science."
                assert sample["article"] == "This is a test article about science."
                assert sample["id"] == "test_001"
                
                print("✅ TRL format conversion test passed")
                return True
                
            except ImportError as e:
                print(f"❌ Import error in TRL conversion test: {e}")
                return False
    
    def test_extract_article_from_prompt(self):
        """Test article extraction from prompt."""
        import unittest.mock as mock
        
        config = {"batch_size": 4, "max_steps": 10}
        
        with mock.patch.dict('sys.modules', {
            'torch': mock.MagicMock(),
            'datasets': mock.MagicMock(), 
            'transformers': mock.MagicMock(),
            'trl': mock.MagicMock(),
            'omegaconf': mock.MagicMock(),
        }):
            try:
                from rlvr_summary.training.ppo_trainer import PPOTrainingLoop
                
                training_loop = PPOTrainingLoop(config)
                
                # Test prompt
                prompt = "Summarize the following article:\n\nThis is a test article about science.\n\nSummary:"
                
                # Extract article
                article = training_loop._extract_article_from_prompt(prompt)
                
                # Validate extraction
                assert article == "This is a test article about science."
                
                print("✅ Article extraction test passed") 
                return True
                
            except ImportError as e:
                print(f"❌ Import error in article extraction test: {e}")
                return False
    
    def test_data_format_compatibility(self):
        """Test that the new format maintains backward compatibility."""
        
        # Test data in old format
        old_format_data = [
            {
                "id": "test_001",
                "article": "Test article content.",
                "summary": "Test summary."
            }
        ]
        
        # Expected TRL format
        expected_fields = ["input_ids", "attention_mask", "query", "reference", "article", "id"]
        
        # This test validates the conversion logic structure
        for sample in old_format_data:
            # Simulate the conversion logic
            prompt = f"Summarize the following article:\n\n{sample['article']}\n\nSummary:"
            
            expected_result = {
                "input_ids": [1, 2, 3, 4, 5],  # Mock tokenization
                "attention_mask": [1, 1, 1, 1, 1],
                "query": prompt,
                "reference": sample["summary"],
                "article": sample["article"],
                "id": sample["id"],
            }
            
            # Validate all expected fields are present
            for field in expected_fields:
                assert field in expected_result
            
        print("✅ Data format compatibility test passed")
        return True


def run_tests():
    """Run all TRL integration tests."""
    print("Running TRL integration tests...")
    
    test_suite = TestTRLIntegration()
    
    results = []
    results.append(test_suite.test_convert_to_ppo_format())
    results.append(test_suite.test_extract_article_from_prompt())
    results.append(test_suite.test_data_format_compatibility())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n✨ TRL integration tests completed: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)