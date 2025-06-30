#!/usr/bin/env python3
"""
Example usage of the RLVR Summary data processing pipeline.

This script demonstrates how to use the core data processing infrastructure
to load, preprocess, validate, and process CNN-DailyMail data.
"""

import json
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import data pipeline components
from rlvr_summary.data import (
    CNNDMLoader,
    TextPreprocessor,
    DataValidator,
    JSONAnnotationHandler,
    BatchProcessor
)
from rlvr_summary.data.batch_processor import create_data_pipeline


def create_sample_data(output_dir: Path):
    """Create sample CNN-DM data for demonstration."""
    sample_data = [
        {
            "id": "cnn_0001",
            "article": "Scientists have discovered a new species of butterfly in the Amazon rainforest. The butterfly, named Morpho amazonicus, has distinctive blue and silver wings. Researchers from the University of São Paulo found the species during a three-month expedition in the remote regions of Brazil. The discovery highlights the incredible biodiversity of the Amazon and the need for conservation efforts. The butterfly feeds primarily on rotting fruit and plays an important role in the ecosystem as a pollinator.",
            "highlights": "Scientists discover new butterfly species Morpho amazonicus in Amazon rainforest. The butterfly has blue and silver wings and feeds on rotting fruit. Discovery emphasizes Amazon biodiversity and conservation importance.",
            "url": "https://example.com/news/amazon-butterfly"
        },
        {
            "id": "cnn_0002", 
            "article": "Climate change is accelerating the melting of glaciers worldwide, according to a new study published in Nature Climate Change. Researchers analyzed satellite data from the past 20 years and found that glacial retreat has increased by 40% compared to previous decades. The study warns that this rapid melting could lead to significant sea level rise and impact freshwater supplies for billions of people. Arctic glaciers are melting fastest, followed by those in the Himalayas and Andes mountains.",
            "highlights": "New study shows glacier melting has accelerated 40% in past 20 years due to climate change. Rapid melting threatens sea level rise and freshwater supplies. Arctic glaciers melting fastest.",
            "url": "https://example.com/news/glacier-melting"
        },
        {
            "id": "cnn_0003",
            "article": "A breakthrough in quantum computing has been achieved by researchers at MIT. They successfully demonstrated a 100-qubit quantum processor that can solve certain optimization problems 10,000 times faster than classical computers. The system uses novel error correction techniques to maintain quantum coherence for longer periods. This advancement brings practical quantum computing applications closer to reality, potentially revolutionizing fields like drug discovery, financial modeling, and artificial intelligence.",
            "highlights": "MIT researchers achieve quantum computing breakthrough with 100-qubit processor. System solves optimization problems 10,000x faster than classical computers. Advancement brings practical quantum applications closer.",
            "url": "https://example.com/news/quantum-breakthrough"
        }
    ]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    sample_file = output_dir / "sample_cnn_dm.jsonl"
    with open(sample_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Created sample data at {sample_file}")
    return sample_file


def demonstrate_individual_components(data_file: Path):
    """Demonstrate individual pipeline components."""
    print("\n=== Individual Component Demonstration ===")
    
    # 1. Data Loading
    print("\n1. Data Loading with CNNDMLoader:")
    loader = CNNDMLoader(data_path=data_file.parent, split="sample_cnn_dm")
    
    print(f"Loading data from {data_file}")
    samples = list(loader.load())
    print(f"Loaded {len(samples)} samples")
    
    # Show first sample
    print("First sample:")
    first_sample = samples[0]
    print(f"  ID: {first_sample['id']}")
    print(f"  Article length: {len(first_sample['article'])} chars")
    print(f"  Summary length: {len(first_sample['highlights'])} chars")
    
    # 2. Text Preprocessing
    print("\n2. Text Preprocessing:")
    preprocessor = TextPreprocessor(use_spacy=False)  # spaCy not available
    
    processed_sample = preprocessor.preprocess_sample(first_sample)
    print(f"  Original article tokens: {len(processed_sample['article_tokens'])}")
    print(f"  Cleaned article preview: {processed_sample['article_clean'][:100]}...")
    print(f"  Article sentences: {len(processed_sample['article_sentences'])}")
    
    # 3. Data Validation
    print("\n3. Data Validation:")
    validator = DataValidator(
        min_article_length=50,
        min_summary_length=20,
        max_article_length=2000
    )
    
    validation_result = validator.validate_sample(first_sample)
    print(f"  Sample {first_sample['id']} valid: {validation_result['is_valid']}")
    if validation_result['errors']:
        print(f"  Errors: {validation_result['errors']}")
    if validation_result['warnings']:
        print(f"  Warnings: {validation_result['warnings']}")
        
    # Batch validation
    batch_result = validator.batch_validate(samples)
    print(f"  Batch validation: {batch_result['valid_samples']}/{batch_result['total_samples']} samples valid")
    print(f"  Validation rate: {batch_result['validation_rate']:.1%}")
    
    # 4. JSON Annotations
    print("\n4. JSON Annotations:")
    annotation_handler = JSONAnnotationHandler()
    
    # Create sample annotation
    annotation = annotation_handler.create_annotation(
        sample_id=first_sample['id'],
        quality_score=0.85,
        factual_errors=[],
        tool_calls=[
            {
                "tool": "search",
                "query": "Amazon butterfly species discovery",
                "position": 0
            }
        ]
    )
    
    print(f"  Created annotation for {annotation['id']}")
    print(f"  Quality score: {annotation['annotations']['quality_score']}")
    print(f"  Tool calls: {len(annotation['annotations']['tool_calls'])}")
    
    # Validate annotation
    validation = annotation_handler.validate_annotation(annotation)
    print(f"  Annotation valid: {validation['is_valid']}")
    
    # 5. Batch Processing
    print("\n5. Batch Processing:")
    batch_processor = BatchProcessor(batch_size=2)
    
    def simple_transform(item):
        return {
            "id": item["id"],
            "article_word_count": len(item["article"].split()),
            "summary_word_count": len(item["highlights"].split()),
            "processed": True
        }
    
    results = batch_processor.process_list(samples, simple_transform)
    print(f"  Processed {len(results)} items in {batch_processor.stats['batches_processed']} batches")
    print(f"  Processing time: {batch_processor.stats['processing_time']:.2f}s")
    
    # Show batch processing stats
    progress = batch_processor.get_progress_info()
    print(f"  Items per second: {progress['items_per_second']:.1f}")


def demonstrate_integrated_pipeline(data_file: Path, output_dir: Path):
    """Demonstrate the integrated data pipeline."""
    print("\n=== Integrated Pipeline Demonstration ===")
    
    # Create pipeline components
    loader = CNNDMLoader(data_path=data_file.parent, split="sample_cnn_dm")
    preprocessor = TextPreprocessor(use_spacy=False)
    validator = DataValidator(min_article_length=50, min_summary_length=20)
    batch_processor = BatchProcessor(batch_size=2)
    
    # Run complete pipeline
    output_file = output_dir / "pipeline_output.jsonl"
    
    print(f"Running integrated pipeline...")
    print(f"  Input: {data_file}")
    print(f"  Output: {output_file}")
    
    pipeline_results = create_data_pipeline(
        loader=loader,
        preprocessor=preprocessor,
        validator=validator,
        batch_processor=batch_processor,
        output_path=output_file
    )
    
    # Display results
    stats = pipeline_results["statistics"]
    print(f"\nPipeline Results:")
    print(f"  Total loaded: {stats['total_loaded']}")
    print(f"  Total preprocessed: {stats['total_preprocessed']}")
    print(f"  Total validated: {stats['total_validated']}")
    print(f"  Validation failures: {stats['validation_failures']}")
    print(f"  Processing time: {stats['processing_time']:.2f}s")
    print(f"  Items per second: {stats.get('items_per_second', 0):.1f}")
    
    # Show validation report
    if 'validation_report' in pipeline_results:
        val_report = pipeline_results['validation_report']
        print(f"  Validation rate: {val_report.get('validation_rate', 0):.1%}")
        if val_report.get('validation_errors'):
            print(f"  Error types: {list(val_report['validation_errors'].keys())}")
    
    # Show sample processed data
    if pipeline_results["data"]:
        sample_result = pipeline_results["data"][0]
        print(f"\nSample processed data structure:")
        print(f"  Keys: {list(sample_result.keys())}")
        if "processed" in sample_result:
            processed_keys = list(sample_result["processed"].keys())
            print(f"  Processed keys: {processed_keys[:5]}...")  # Show first 5
        if "validation" in sample_result:
            print(f"  Validation result: {sample_result['validation']['is_valid']}")


def main():
    """Main demonstration function."""
    print("RLVR Summary Data Processing Pipeline Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample data
    print("Creating sample CNN-DailyMail data...")
    sample_file = create_sample_data(output_dir)
    
    # Demonstrate individual components
    demonstrate_individual_components(sample_file)
    
    # Demonstrate integrated pipeline
    demonstrate_integrated_pipeline(sample_file, output_dir)
    
    print(f"\n✅ Demo completed! Check {output_dir} for output files.")
    print("\nThe data processing pipeline is ready for:")
    print("  • Loading CNN-DailyMail datasets")
    print("  • Text preprocessing and cleaning")
    print("  • Data validation and quality control")
    print("  • JSON annotation handling")
    print("  • Batch processing capabilities")
    print("\nNext steps: Install spaCy model for advanced NLP features:")
    print("  pip install spacy")
    print("  python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    main()