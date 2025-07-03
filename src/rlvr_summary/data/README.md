# Data Processing Pipeline

This module provides core data processing infrastructure for the RLVR Summary project, implementing utilities for loading, preprocessing, validating, and batch processing text data for summarization tasks.

## Components

### Data Flow

The pipeline now supports two output formats:

1. **Standard Format** (for custom training loops):
```python
{
    "id": "cnn_0001",
    "article": "Scientists have discovered...",
    "summary": "Scientists discover new butterfly species..."
}
```

2. **VERL PPOTrainer Format** (optimized for VERL integration):
```python
{
    "input_ids": [101, 2054, 2003, ...],  # Tokenized prompt only
    "attention_mask": [1, 1, 1, ...],
    "query": "Summarize: Scientists have discovered...\n\nSummary:",  # Optional
    "reference": "Scientists discover new butterfly species..."  # For reward computation
}
```

### VERL Integration

The data pipeline integrates seamlessly with VERL's PPOTrainer through format conversion:

```python
from rlvr_summary.training.ppo_trainer import PPOTrainingLoop

# Training loop automatically converts data to VERL format
training_loop = PPOTrainingLoop(config)
train_dataset, eval_dataset = training_loop.load_datasets()  # Returns VERL-compatible Dataset objects
```

**Benefits of VERL Format:**
- ✅ Highly optimized for memory and performance
- ✅ Built-in checkpointing, logging, evaluation
- ✅ Better HuggingFace ecosystem integration
- ✅ Automatic handling of padding, batching, device placement
- ✅ Enhanced reward function customization via functional_reward

## Components

### 1. Data Loaders (`loaders.py`)

- **CNNDMLoader**: Loads CNN-DailyMail dataset from HuggingFace datasets or local files
- **CustomDataLoader**: Generic loader for custom dataset formats (JSON/JSONL)

**Features:**
- Supports both HuggingFace datasets and local file loading
- Automatic fallback between loading methods
- Configurable sample limits for development
- JSONL format support for large datasets

**Example:**
```python
from rlvr_summary.data import CNNDMLoader

# Load from HuggingFace (when datasets library is available)
loader = CNNDMLoader(split="train", max_samples=1000)
for sample in loader.load():
    print(sample["id"], len(sample["article"]))

# Load from local files
loader = CNNDMLoader(data_path="./data", split="train")
samples = list(loader.load())
```

### 2. Text Preprocessing (`preprocessors.py`)

- **TextPreprocessor**: Comprehensive text cleaning and preprocessing
- Supports both basic preprocessing and advanced spaCy-based NLP

**Features:**
- URL and email removal
- Whitespace normalization
- Text length limits
- spaCy integration for tokenization, POS tagging, NER
- Sentence splitting and text statistics

**Example:**
```python
from rlvr_summary.data import TextPreprocessor

# Basic preprocessing (no spaCy required)
preprocessor = TextPreprocessor(use_spacy=False)
sample = {"article": "Text to process...", "highlights": "Summary..."}
processed = preprocessor.preprocess_sample(sample)

# Advanced preprocessing with spaCy
preprocessor = TextPreprocessor(use_spacy=True, spacy_model="en_core_web_sm")
processed = preprocessor.preprocess_sample(sample)
# Includes tokens, lemmas, POS tags, entities, etc.
```

### 3. Data Validation (`validators.py`)

- **DataValidator**: Comprehensive data quality validation
- Configurable validation rules for articles and summaries

**Features:**
- Length validation (min/max for articles and summaries)
- Required field checking
- Encoding issue detection
- Language detection (basic English validation)
- Content quality checks (repetition, duplicates)
- Batch validation with statistics

**Example:**
```python
from rlvr_summary.data import DataValidator

validator = DataValidator(
    min_article_length=100,
    max_article_length=10000,
    min_summary_length=20,
    max_summary_length=500
)

# Validate single sample
result = validator.validate_sample(sample)
print(f"Valid: {result['is_valid']}, Errors: {result['errors']}")

# Batch validation
batch_result = validator.batch_validate(samples)
print(f"Valid: {batch_result['valid_samples']}/{batch_result['total_samples']}")
```

### 4. JSON Annotations (`annotations.py`)

- **JSONAnnotationHandler**: Structured annotation management
- Schema validation for consistent annotation format

**Features:**
- Predefined schema for summarization annotations
- Quality scores, factual errors, tool calls support
- Annotation validation and merging
- JSONL and JSON format support
- Filtering and statistics

**Example:**
```python
from rlvr_summary.data import JSONAnnotationHandler

handler = JSONAnnotationHandler()

# Create annotation
annotation = handler.create_annotation(
    sample_id="cnn_001",
    quality_score=0.85,
    factual_errors=[{"text": "error", "error_type": "factual"}],
    tool_calls=[{"tool": "search", "query": "fact check"}]
)

# Validate annotation
result = handler.validate_annotation(annotation)
print(f"Valid annotation: {result['is_valid']}")

# Save/load annotations
handler.save_annotations([annotation], "annotations.jsonl", format_type="jsonl")
loaded = handler.load_annotations("annotations.jsonl")
```

### 5. Batch Processing (`batch_processor.py`)

- **BatchProcessor**: Efficient batch processing with parallel support
- **create_data_pipeline**: Complete pipeline orchestration

**Features:**
- Configurable batch sizes
- Parallel processing support
- Progress tracking and error handling
- Intermediate result saving
- Pipeline integration

**Example:**
```python
from rlvr_summary.data import BatchProcessor
from rlvr_summary.data.batch_processor import create_data_pipeline

# Simple batch processing
processor = BatchProcessor(batch_size=50, max_workers=4)
results = processor.process_list(data, processing_function, parallel=True)

# Complete pipeline
pipeline_results = create_data_pipeline(
    loader=CNNDMLoader(split="train"),
    preprocessor=TextPreprocessor(),
    validator=DataValidator(),
    batch_processor=BatchProcessor(batch_size=100),
    output_path="processed_data.jsonl"
)
```

## Integration Example

Complete pipeline usage:

```python
from rlvr_summary.data import (
    CNNDMLoader, TextPreprocessor, DataValidator, 
    BatchProcessor
)
from rlvr_summary.data.batch_processor import create_data_pipeline

# Configure components
loader = CNNDMLoader(split="train", max_samples=1000)
preprocessor = TextPreprocessor(use_spacy=True)
validator = DataValidator(min_article_length=50)
batch_processor = BatchProcessor(batch_size=32)

# Run complete pipeline
results = create_data_pipeline(
    loader=loader,
    preprocessor=preprocessor,
    validator=validator,
    batch_processor=batch_processor,
    output_path="processed_cnn_dm.jsonl"
)

print(f"Processed {results['statistics']['total_loaded']} samples")
print(f"Validation rate: {results['statistics']['validation_rate']:.1%}")
```

## Requirements

**Core dependencies:**
- Python 3.9+
- Standard library modules (json, pathlib, re, logging, etc.)

**Optional dependencies:**
- `datasets`: For HuggingFace CNN-DailyMail loading
- `spacy`: For advanced NLP preprocessing
- `pandas`: For enhanced data manipulation
- `tqdm`: For progress bars

**For spaCy support:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Testing

Run the test suite:
```bash
python tests/test_data_pipeline.py
```

Run the demo:
```bash
python demo_pipeline.py
```

## Architecture

The data pipeline follows a modular design:

1. **Loaders** handle data input from various sources
2. **Preprocessors** clean and normalize text data
3. **Validators** ensure data quality and consistency
4. **Annotations** manage structured metadata
5. **BatchProcessor** orchestrates efficient processing

This architecture supports the four-step synthetic trace generation pipeline by providing:
- Robust data loading for CNN-DailyMail
- Quality validation for training data
- Structured annotation handling for AI feedback
- Efficient batch processing for large-scale operations

## Next Steps

This infrastructure is ready to support:
- Four-step synthetic trace generation pipeline
- FENICE fact-checking integration
- Tool call annotation and validation
- Large-scale dataset processing for training