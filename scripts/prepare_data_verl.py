#!/usr/bin/env python3
"""
Data preparation script for VERL PPO training.

This script processes the CNN/DailyMail dataset and converts it to the format
expected by VERL (parquet files with specific schema).

VERL data format requirements (5 required fields):
1. data_source: Dataset name for reward function indexing
2. prompt: Chat template format (list of {"role": str, "content": str})
3. ability: Task category (e.g., "summarization", "math", etc.)
4. reward_model: {"style": str, "ground_truth": str}
5. extra_info: Additional metadata dictionary
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import datasets
import pandas as pd
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.data import (
    BatchProcessor,
    CNNDMLoader,
    DataValidator,
    TextPreprocessor,
)
from rlvr_summary.data.batch_processor import create_data_pipeline

# Import FENICE for document pre-processing
try:
    from rlvr_summary.fenice.FENICE import FENICE
    from rlvr_summary.fenice.utils.utils import split_into_sentences_batched
    FENICE_AVAILABLE = True
except ImportError:
    logger.warning("FENICE not available - document caching will be skipped")
    FENICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_documents_for_fenice(documents: List[str]) -> Dict[str, Dict]:
    """
    Pre-process documents for FENICE caching to avoid runtime computation.
    
    Args:
        documents: List of document texts
        
    Returns:
        Dictionary mapping document indices to cached FENICE data
    """
    if not FENICE_AVAILABLE:
        logger.warning("FENICE not available - returning empty cache")
        return {}
        
    logger.info(f"Pre-processing {len(documents)} documents for FENICE caching...")
    
    try:
        # Create temporary FENICE instance for processing
        fenice = FENICE(
            use_coref=False,  # Start with just sentence caching for safety
            num_sent_per_paragraph=5,
            sliding_paragraphs=True,
            sliding_stride=1,
        )
        
        # Cache sentences for all documents
        all_sentences = split_into_sentences_batched(
            documents, batch_size=32, return_offsets=True
        )
        
        document_cache = {}
        for i, sentences in enumerate(all_sentences):
            doc_id = fenice.get_id(i, documents[i])
            
            # Store the essential cached data
            document_cache[i] = {
                'doc_id': doc_id,
                'sentences': sentences,  # List of (sentence, start_offset, end_offset)
                'document_text': documents[i],
            }
        
        logger.info(f"✅ Successfully cached {len(document_cache)} documents")
        return document_cache
        
    except Exception as e:
        logger.error(f"Failed to pre-process documents for FENICE: {e}")
        return {}


def create_fenice_document_cache(documents: List[str]) -> Dict:
    """
    Create FENICE document cache for the extra_info field.
    
    Args:
        documents: List of document texts
        
    Returns:
        Cache dictionary to be stored in extra_info.fenice_document_cache
    """
    cache = preprocess_documents_for_fenice(documents)
    
    if cache:
        logger.info(f"Created FENICE document cache for {len(cache)} documents")
    else:
        logger.warning("FENICE document cache is empty")
        
    return cache


def prepare_rlvr_dataset(
    data_config: Dict, output_dir: Path, max_samples: int = 1000, split: str = "train"
) -> Path:
    """
    Prepare RLVR dataset in VERL format.

    Args:
        data_config: Data configuration
        output_dir: Output directory for parquet files
        max_samples: Maximum number of samples to process
        split: Dataset split ("train", "validation", "test")

    Returns:
        Path to the generated parquet file
    """
    logger.info(f"Preparing {split} dataset with max {max_samples} samples...")

    # Configure data pipeline components
    train_loader = CNNDMLoader(
        data_path=data_config.get("data_path"),
        split=split,
        max_samples=max_samples,
    )

    preprocessor = TextPreprocessor(
        max_length=data_config.get("max_input_length", 2048),
    )

    validator = DataValidator(
        min_article_length=data_config.get("min_input_length", 50),
        max_article_length=data_config.get("max_input_length", 2048),
        min_summary_length=data_config.get("min_target_length", 10),
        max_summary_length=data_config.get("max_target_length", 500),
    )

    batch_processor = BatchProcessor(
        batch_size=data_config.get("data_batch_size", 32),
        max_workers=data_config.get("data_workers", 2),
    )

    # Process dataset
    logger.info(f"Loading and processing {split} dataset...")
    pipeline_result = create_data_pipeline(
        loader=train_loader,
        preprocessor=preprocessor,
        validator=validator,
        batch_processor=batch_processor,
    )

    # Convert to VERL format
    verl_data = []
    
    # Pre-process documents for FENICE caching
    logger.info("🔄 Pre-processing documents for FENICE caching...")
    documents = []
    for idx, item in enumerate(pipeline_result["data"]):
        processed_item = item.get("processed", item.get("original", {}))
        article = processed_item.get("article", "")
        documents.append(article)
    
    # Create FENICE document cache
    fenice_document_cache = create_fenice_document_cache(documents)
    
    # Now create VERL data with cache
    for idx, item in enumerate(pipeline_result["data"]):
        processed_item = item.get("processed", item.get("original", {}))

        # Create standardized prompt for VERL in chat template format
        article = processed_item.get("article", "")
        prompt_content = f"Summarize the following article:\n\n{article}"

        verl_data.append(
            {
                "data_source": "cnn_dailymail",  # Used by reward function to select scoring method
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "summarization",  # Task category
                "reward_model": {
                    "style": "rule",  # Using rule-based reward
                    "ground_truth": processed_item.get(
                        "highlights", processed_item.get("summary", "")
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": processed_item.get("id", ""),
                    "fenice_document_cache": fenice_document_cache.get(idx, {}),
                },
            }
        )

    # Validate VERL format compliance
    logger.info("🔍 Validating VERL data format...")
    for i, sample in enumerate(verl_data[:5]):  # Validate first 5 samples
        try:
            validate_verl_format(sample)
            logger.info(f"✅ Sample {i+1} validation passed")
        except ValueError as e:
            logger.error(f"❌ Sample {i+1} validation failed: {e}")
            raise

    # Save as parquet file for VERL
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{split}_data.parquet"
    json_path = output_dir / f"{split}_data.json"

    df = pd.DataFrame(verl_data)
    df.to_parquet(parquet_path, index=False)

    # Also save as JSON for manual inspection
    import json

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verl_data[:5], f, indent=2, ensure_ascii=False)

    logger.info(
        f"✅ {split.title()} data saved to {parquet_path}: {len(verl_data)} samples"
    )
    logger.info(
        f"✅ {split.title()} data also saved to {json_path} for manual inspection"
    )

    # Show a preview of the data format
    if verl_data:
        logger.info(f"📖 Sample from {split} data:")
        sample = verl_data[0]
        print(f"\n{'='*60}")
        print(f"SAMPLE {split.upper()} DATA")
        print(f"{'='*60}")
        print(f"Data Source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Reward Model Style: {sample['reward_model']['style']}")
        print(
            f"Ground Truth (first 100 chars): {sample['reward_model']['ground_truth'][:100]}..."
        )
        print(f"Extra Info: {sample['extra_info']}")
        print(f"{'='*60}\n")

    return parquet_path


def validate_verl_format(data_sample: Dict) -> bool:
    """
    Validate that a data sample follows the VERL format requirements.

    Args:
        data_sample: Single data sample to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # Check all required fields exist
    for field in required_fields:
        if field not in data_sample:
            raise ValueError(f"Missing required field: {field}")

    # Validate prompt format (should be list of chat messages)
    prompt = data_sample["prompt"]
    if not isinstance(prompt, list):
        raise ValueError(f"'prompt' must be a list, got {type(prompt)}")

    for message in prompt:
        if not isinstance(message, dict):
            raise ValueError(f"Prompt messages must be dicts, got {type(message)}")
        if "role" not in message or "content" not in message:
            raise ValueError(
                "Each prompt message must have 'role' and 'content' fields"
            )

    # Validate reward_model format
    reward_model = data_sample["reward_model"]
    if not isinstance(reward_model, dict):
        raise ValueError(f"'reward_model' must be a dict, got {type(reward_model)}")
    if "style" not in reward_model or "ground_truth" not in reward_model:
        raise ValueError("'reward_model' must have 'style' and 'ground_truth' fields")

    # Validate other fields are strings/dicts as expected
    if not isinstance(data_sample["data_source"], str):
        raise ValueError(
            f"'data_source' must be a string, got {type(data_sample['data_source'])}"
        )
    if not isinstance(data_sample["ability"], str):
        raise ValueError(
            f"'ability' must be a string, got {type(data_sample['ability'])}"
        )
    if not isinstance(data_sample["extra_info"], dict):
        raise ValueError(
            f"'extra_info' must be a dict, got {type(data_sample['extra_info'])}"
        )

    return True


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for VERL PPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data_verl.py --output-dir ./data/verl
  python scripts/prepare_data_verl.py --max-samples 2000 --output-dir ./data/verl
  python scripts/prepare_data_verl.py --config configs/data/cnn_dailymail.yaml
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/data/cnn_dailymail.yaml",
        help="Path to data configuration file",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./data/verl",
        help="Output directory for parquet files",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process",
    )

    parser.add_argument(
        "--train-split", type=str, default="train", help="Training split name"
    )

    parser.add_argument(
        "--val-split", type=str, default="validation", help="Validation split name"
    )

    parser.add_argument(
        "--val-samples", type=int, default=200, help="Number of validation samples"
    )

    args = parser.parse_args()

    # Load data configuration
    try:
        data_config = OmegaConf.load(args.config)
        logger.info(f"✅ Loaded data config from {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load data config: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    try:
        # Prepare training data
        train_path = prepare_rlvr_dataset(
            data_config=data_config,
            output_dir=output_dir,
            max_samples=args.max_samples,
            split=args.train_split,
        )

        # Prepare validation data
        val_path = prepare_rlvr_dataset(
            data_config=data_config,
            output_dir=output_dir,
            max_samples=args.val_samples,
            split=args.val_split,
        )

        # Create summary file with paths
        summary_path = output_dir / "data_summary.yaml"
        summary = {
            "train_files": [str(train_path)],
            "val_files": [str(val_path)],
            "prepared_on": pd.Timestamp.now().isoformat(),
            "train_samples": args.max_samples,
            "val_samples": args.val_samples,
        }

        OmegaConf.save(summary, summary_path)
        logger.info(f"✅ Data summary saved to {summary_path}")

        print(f"\n🎉 Data preparation completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Training data: {train_path}")
        print(f"📊 Validation data: {val_path}")
        print(f"📋 Summary file: {summary_path}")
        print(f"\nVERL Data Format:")
        print(f"  ✅ data_source: 'cnn_dailymail' (for reward function indexing)")
        print(f"  ✅ prompt: Chat template format with role/content")
        print(f"  ✅ ability: 'summarization' (task category)")
        print(f"  ✅ reward_model: style='rule', ground_truth=reference_summary")
        print(f"  ✅ extra_info: split, index, id metadata")
        print(f"\nTo use with VERL, update your config:")
        print(f"  data.train_files: [{str(train_path)}]")
        print(f"  data.val_files: [{str(val_path)}]")

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
