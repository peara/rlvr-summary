"""Tests for data processing pipeline components."""

import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.data.loaders import CNNDMLoader
from rlvr_summary.data.preprocessors import TextPreprocessor
from rlvr_summary.data.validators import DataValidator
from rlvr_summary.data.annotations import JSONAnnotationHandler
from rlvr_summary.data.batch_processor import BatchProcessor, create_data_pipeline


class TestCNNDMLoader:
    """Test CNN-DM data loader functionality."""
    
    def test_init(self):
        """Test loader initialization."""
        loader = CNNDMLoader(split="train", max_samples=10)
        assert loader.split == "train"
        assert loader.max_samples == 10
        
    def test_load_from_local(self):
        """Test loading from local JSONL files."""
        # Create temporary test data
        test_data = [
            {
                "id": "test_1",
                "article": "This is a test article about something interesting.",
                "highlights": "Test article summary."
            },
            {
                "id": "test_2", 
                "article": "Another test article with different content here.",
                "highlights": "Another summary for testing."
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "train.jsonl"
            
            # Write test data
            with open(test_file, 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')
                    
            # Test loader
            loader = CNNDMLoader(data_path=temp_path, split="train")
            results = list(loader.load_from_local())
            
            assert len(results) == 2
            assert results[0]["id"] == "test_1"
            assert "article" in results[0]
            assert "highlights" in results[0]
            
    def test_load_with_max_samples(self):
        """Test max_samples limitation."""
        test_data = [{"id": f"test_{i}", "article": f"Article {i}", "highlights": f"Summary {i}"} 
                    for i in range(10)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "train.jsonl"
            
            with open(test_file, 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')
                    
            loader = CNNDMLoader(data_path=temp_path, split="train", max_samples=3)
            results = list(loader.load_from_local())
            
            assert len(results) == 3


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor(use_spacy=False)
        assert not preprocessor.use_spacy
        
    def test_clean_text(self):
        """Test basic text cleaning."""
        preprocessor = TextPreprocessor(use_spacy=False)
        
        # Test URL removal
        text_with_url = "Check this out: https://example.com for more info."
        cleaned = preprocessor.clean_text(text_with_url)
        assert "https://example.com" not in cleaned
        
        # Test whitespace normalization
        text_with_spaces = "This   has    multiple   spaces."
        cleaned = preprocessor.clean_text(text_with_spaces)
        assert "   " not in cleaned
        
    def test_preprocess_sample(self):
        """Test sample preprocessing."""
        preprocessor = TextPreprocessor(use_spacy=False)
        
        sample = {
            "id": "test_1",
            "article": "This is a test article with some content.",
            "highlights": "Test summary."
        }
        
        result = preprocessor.preprocess_sample(sample)
        
        assert "article_clean" in result
        assert "highlights_clean" in result
        assert "article_tokens" in result
        assert "highlights_tokens" in result
        
    def test_text_stats(self):
        """Test text statistics calculation."""
        preprocessor = TextPreprocessor(use_spacy=False)
        
        text = "This is a test. It has multiple sentences."
        stats = preprocessor.get_text_stats(text)
        
        assert "char_count" in stats
        assert "word_count" in stats
        assert "sentence_count" in stats
        # Check that word count is reasonable (actual count is 8 words)
        assert stats["word_count"] == 8


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_init(self):
        """Test validator initialization."""
        validator = DataValidator(min_article_length=50)
        assert validator.min_article_length == 50
        
    def test_validate_sample_success(self):
        """Test successful sample validation."""
        validator = DataValidator(
            min_article_length=10,
            min_summary_length=5
        )
        
        sample = {
            "id": "test_1",
            "article": "This is a valid test article with sufficient length for validation.",
            "highlights": "Valid test summary."
        }
        
        result = validator.validate_sample(sample)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        
    def test_validate_sample_failure(self):
        """Test sample validation with failures."""
        validator = DataValidator(
            min_article_length=100,
            min_summary_length=20
        )
        
        sample = {
            "id": "test_1",
            "article": "Short article.",  # Too short
            "highlights": "Short."  # Too short
        }
        
        result = validator.validate_sample(sample)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        
    def test_batch_validate(self):
        """Test batch validation."""
        validator = DataValidator(min_article_length=10, min_summary_length=5)
        
        samples = [
            {
                "id": "test_1",
                "article": "This is a valid test article with sufficient length for validation.",
                "highlights": "Valid summary for testing."
            },
            {
                "id": "test_2", 
                "article": "Short.",  # Invalid - too short
                "highlights": "Also short summary."
            }
        ]
        
        result = validator.batch_validate(samples)
        print(f"Debug: Batch validation result = {result}")
        assert result["total_samples"] == 2
        # The first sample should be valid, second should fail on article length
        assert result["valid_samples"] >= 1
        assert result["failed_samples"] >= 1


class TestJSONAnnotationHandler:
    """Test JSON annotation handling."""
    
    def test_init(self):
        """Test annotation handler initialization."""
        handler = JSONAnnotationHandler()
        assert handler.annotation_schema is not None
        
    def test_create_annotation(self):
        """Test annotation creation."""
        handler = JSONAnnotationHandler(auto_timestamp=False)
        
        annotation = handler.create_annotation(
            sample_id="test_1",
            quality_score=0.8,
            factual_errors=[{"text": "error text", "error_type": "factual"}]
        )
        
        assert annotation["id"] == "test_1"
        assert annotation["annotations"]["quality_score"] == 0.8
        assert len(annotation["annotations"]["factual_errors"]) == 1
        
    def test_validate_annotation(self):
        """Test annotation validation."""
        handler = JSONAnnotationHandler()
        
        valid_annotation = {
            "id": "test_1",
            "annotations": {
                "quality_score": 0.8,
                "factual_errors": [{"text": "error", "error_type": "factual"}]
            }
        }
        
        result = handler.validate_annotation(valid_annotation)
        assert result["is_valid"] is True
        
        # Test invalid annotation
        invalid_annotation = {
            "id": "test_1",
            "annotations": {
                "quality_score": 1.5  # Invalid range
            }
        }
        
        result = handler.validate_annotation(invalid_annotation)
        assert result["is_valid"] is False
        
    def test_save_load_annotations(self):
        """Test saving and loading annotations."""
        handler = JSONAnnotationHandler()
        
        annotations = [
            {
                "id": "test_1",
                "annotations": {"quality_score": 0.8}
            },
            {
                "id": "test_2", 
                "annotations": {"quality_score": 0.9}
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "annotations.json"
            
            # Save annotations
            handler.save_annotations(annotations, temp_path)
            
            # Load annotations
            loaded = handler.load_annotations(temp_path)
            
            assert len(loaded) == 2
            assert loaded[0]["id"] == "test_1"


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    def test_init(self):
        """Test batch processor initialization."""
        processor = BatchProcessor(batch_size=50)
        assert processor.batch_size == 50
        
    def test_process_list(self):
        """Test processing a list of items."""
        processor = BatchProcessor(batch_size=2)
        
        data = [{"id": i, "value": f"item_{i}"} for i in range(5)]
        
        def simple_process(item):
            return {"processed_id": item["id"], "processed_value": item["value"].upper()}
            
        results = processor.process_list(data, simple_process)
        
        assert len(results) == 5
        assert results[0]["processed_value"] == "ITEM_0"
        
    def test_batch_size_handling(self):
        """Test proper batch size handling."""
        processor = BatchProcessor(batch_size=3)
        
        data = list(range(10))  # 10 items, should create 4 batches (3,3,3,1)
        
        def identity_process(item):
            return item
            
        results = processor.process_list(data, identity_process)
        
        assert len(results) == 10
        assert processor.stats["batches_processed"] == 4


class TestDataPipeline:
    """Test complete data pipeline integration."""
    
    def test_create_data_pipeline(self):
        """Test creating and running a complete pipeline."""
        # Create test data
        test_data = [
            {
                "id": "test_1",
                "article": "This is a comprehensive test article with substantial content for testing the data pipeline.",
                "highlights": "Test article for pipeline validation."
            },
            {
                "id": "test_2",
                "article": "Another test article with different content to verify batch processing capabilities.",
                "highlights": "Another test summary for verification."
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "train.jsonl"
            
            # Write test data
            with open(test_file, 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')
                    
            # Create pipeline components
            loader = CNNDMLoader(data_path=temp_path, split="train")
            preprocessor = TextPreprocessor(use_spacy=False)
            validator = DataValidator(min_article_length=20, min_summary_length=5)
            batch_processor = BatchProcessor(batch_size=1)
            
            # Run pipeline
            results = create_data_pipeline(
                loader=loader,
                preprocessor=preprocessor,
                validator=validator,
                batch_processor=batch_processor
            )
            
            assert "data" in results
            assert "statistics" in results
            assert len(results["data"]) == 2
            
            # Check that preprocessing was applied
            sample_result = results["data"][0]
            assert "processed" in sample_result
            assert "validation" in sample_result
            assert "article_clean" in sample_result["processed"]


if __name__ == "__main__":
    # Run tests manually
    print("Running data pipeline tests...")
    
    # Test loader
    print("Testing CNNDMLoader...")
    test_loader = TestCNNDMLoader()
    test_loader.test_init()
    test_loader.test_load_from_local()
    test_loader.test_load_with_max_samples()
    print("✓ CNNDMLoader tests passed")
    
    # Test preprocessor
    print("Testing TextPreprocessor...")
    test_preprocessor = TestTextPreprocessor()
    test_preprocessor.test_init()
    test_preprocessor.test_clean_text()
    test_preprocessor.test_preprocess_sample()
    test_preprocessor.test_text_stats()
    print("✓ TextPreprocessor tests passed")
    
    # Test validator
    print("Testing DataValidator...")
    test_validator = TestDataValidator()
    test_validator.test_init()
    test_validator.test_validate_sample_success()
    test_validator.test_validate_sample_failure()
    test_validator.test_batch_validate()
    print("✓ DataValidator tests passed")
    
    # Test annotation handler
    print("Testing JSONAnnotationHandler...")
    test_annotations = TestJSONAnnotationHandler()
    test_annotations.test_init()
    test_annotations.test_create_annotation()
    test_annotations.test_validate_annotation()
    test_annotations.test_save_load_annotations()
    print("✓ JSONAnnotationHandler tests passed")
    
    # Test batch processor
    print("Testing BatchProcessor...")
    test_batch = TestBatchProcessor()
    test_batch.test_init()
    test_batch.test_process_list()
    test_batch.test_batch_size_handling()
    print("✓ BatchProcessor tests passed")
    
    # Test pipeline integration
    print("Testing DataPipeline integration...")
    test_pipeline = TestDataPipeline()
    test_pipeline.test_create_data_pipeline()
    print("✓ DataPipeline tests passed")
    
    print("\n✅ All tests passed successfully!")