#!/usr/bin/env python3
"""
Data preparation script for VERL PPO training.

This script processes the CNN/DailyMail dataset and converts it to the format
expected by VERL (parquet files with specific schema).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        use_spacy=data_config.get("use_spacy", False),
        max_length=data_config.get("max_article_length", 10000),
    )

    validator = DataValidator(
        min_article_length=data_config.get("min_article_length", 50),
        max_article_length=data_config.get("max_article_length", 10000),
        min_summary_length=data_config.get("min_summary_length", 10),
        max_summary_length=data_config.get("max_summary_length", 500),
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
    for item in pipeline_result["data"]:
        processed_item = item.get("processed", item.get("original", {}))

        # Create standardized prompt for VERL
        article = processed_item.get("article", "")
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"

        verl_data.append(
            {
                "prompt": prompt,
                "data_source": "cnn_dailymail",  # Used by reward function to select scoring method
                "id": processed_item.get("id", ""),
                # Include reference for potential reward computation
                "ground_truth": processed_item.get(
                    "highlights", processed_item.get("summary", "")
                ),
            }
        )

    # Save as parquet file for VERL
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{split}_data.parquet"

    df = pd.DataFrame(verl_data)
    df.to_parquet(parquet_path, index=False)

    logger.info(
        f"✅ {split.title()} data saved to {parquet_path}: {len(verl_data)} samples"
    )

    return parquet_path


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
        print(f"\nTo use with VERL, update your config:")
        print(f"  data.train_files: [{str(train_path)}]")
        print(f"  data.val_files: [{str(val_path)}]")

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
