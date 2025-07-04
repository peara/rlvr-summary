#!/usr/bin/env python3
"""Training script for RLVR Summary.

This script provides automated training with configuration validation,
dependency checking, and clear error reporting.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_basic_dependencies():
    """Check if basic dependencies are available."""
    missing = []
    
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    try:
        from omegaconf import OmegaConf, DictConfig
    except ImportError:
        missing.append("omegaconf")
    
    try:
        import hydra
    except ImportError:
        missing.append("hydra-core")
    
    if missing:
        print(f"❌ Missing basic dependencies: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    return True

if not check_basic_dependencies():
    sys.exit(1)

try:
    from rlvr_summary.utils.config import load_config, validate_training_config, setup_directories
    from rlvr_summary.training import train_ppo_model
except ImportError as e:
    print(f"❌ Failed to import RLVR Summary modules: {e}")
    print("Please install the package first: pip install -e .")
    sys.exit(1)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def print_config_summary(config: Dict[str, Any], validation_results: Dict[str, Any]) -> None:
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION SUMMARY")
    print("="*60)
    
    # Project info
    if "project" in config:
        print(f"📋 Project: {config.project.get('name', 'Unknown')}")
        print(f"🔢 Version: {config.project.get('version', 'Unknown')}")
    
    # Model configuration
    if "model" in config:
        model_cfg = config.model
        print(f"🤖 Model: {model_cfg.get('model_name', 'Unknown')}")
        if "torch_dtype" in model_cfg:
            print(f"🔢 Data Type: {model_cfg.torch_dtype}")
        if "load_in_8bit" in model_cfg:
            print(f"⚡ 8-bit Loading: {model_cfg.load_in_8bit}")
    
    # Training configuration
    if "training" in config:
        training_cfg = config.training
        print(f"📚 Learning Rate: {training_cfg.get('learning_rate', 'Not set')}")
        print(f"📦 Batch Size: {training_cfg.get('batch_size', 'Not set')}")
        print(f"🔄 Max Steps: {training_cfg.get('max_steps', 'Not set')}")
        if "ppo_epochs" in training_cfg:
            print(f"🎯 PPO Epochs: {training_cfg.ppo_epochs}")
    
    # Data configuration
    if "data" in config:
        data_cfg = config.data
        print(f"📊 Dataset: {data_cfg.get('dataset_name', 'Unknown')}")
    
    # Paths
    if "paths" in config:
        paths = config.paths
        print(f"📁 Data Directory: {paths.get('data_dir', 'Not set')}")
        print(f"📁 Output Directory: {paths.get('output_dir', 'Not set')}")
        print(f"📁 Checkpoint Directory: {paths.get('checkpoint_dir', 'Not set')}")
    
    # W&B configuration
    if "wandb" in config:
        wandb_cfg = config.wandb
        print(f"📈 W&B Project: {wandb_cfg.get('project', 'Not set')}")
        print(f"🏷️  W&B Entity: {wandb_cfg.get('entity', 'Not set')}")
        print(f"✅ W&B Enabled: {wandb_cfg.get('enabled', False)}")
    
    # Validation results summary
    print("\n" + "="*60)
    print("✅ VALIDATION RESULTS")
    print("="*60)
    
    if validation_results["valid"]:
        print("🎉 Configuration is valid!")
    else:
        print("❌ Configuration has errors")
    
    if validation_results["warnings"]:
        print(f"⚠️  Warnings: {len(validation_results['warnings'])}")
        for warning in validation_results["warnings"]:
            print(f"   • {warning}")
    
    if validation_results["missing_files"]:
        print(f"📁 Missing files: {len(validation_results['missing_files'])}")
        for missing_file in validation_results["missing_files"]:
            print(f"   • {missing_file}")
    
    if validation_results["missing_dependencies"]:
        print(f"📦 Missing dependencies: {len(validation_results['missing_dependencies'])}")
        for dep in validation_results["missing_dependencies"]:
            print(f"   • {dep}")
    
    print("="*60)


def check_environment() -> Dict[str, Any]:
    """Check the training environment."""
    env_status = {
        "python_version": sys.version_info,
        "torch_available": False,
        "transformers_available": False,
        "verl_available": False,
        "wandb_available": False,
        "cuda_available": False,
    }
    
    # Check Python version
    if env_status["python_version"] < (3, 9):
        print(f"⚠️  Python 3.9+ recommended, found {env_status['python_version'].major}.{env_status['python_version'].minor}")
    
    # Check key dependencies
    try:
        import torch
        env_status["torch_available"] = True
        env_status["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import transformers
        env_status["transformers_available"] = True
    except ImportError:
        pass
    
    try:
        import verl
        env_status["verl_available"] = True
    except ImportError:
        pass
    
    try:
        import wandb
        env_status["wandb_available"] = True
    except ImportError:
        pass
    
    return env_status


def validate_and_prepare_training(
    config_path: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[list] = None,
    check_files: bool = True
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate configuration and prepare for training.
    
    Returns:
        Tuple of (config, validation_results)
    """
    print("🔍 Loading and validating configuration...")
    
    try:
        # Load configuration
        config = load_config(config_path, config_name, overrides)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    try:
        validation_results = validate_training_config(config, check_files=check_files)
        print("✅ Configuration validation completed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    
    return config, validation_results


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="RLVR Summary Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                              # Use default config
  python scripts/train.py --config configs            # Specify config directory
  python scripts/train.py --experiment my-experiment  # Set experiment name
  python scripts/train.py --dry-run                   # Validate only, don't train
  python scripts/train.py --log-level DEBUG           # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration directory or file"
    )
    
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Configuration file name (without .yaml extension)"
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Experiment name for W&B tracking"
    )
    
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=None,
        help="Configuration overrides (e.g., training.learning_rate=1e-4)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration only, don't start training"
    )
    
    parser.add_argument(
        "--no-file-check",
        action="store_true",
        help="Skip checking for file existence"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress configuration summary output"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    print("🚀 RLVR Summary Training Script")
    print("="*60)
    
    # Check environment
    print("🔍 Checking environment...")
    env_status = check_environment()
    
    if not env_status["torch_available"]:
        print("❌ PyTorch not available. Please install: pip install torch")
        sys.exit(1)
    
    if not env_status["transformers_available"]:
        print("❌ Transformers not available. Please install: pip install transformers")
        sys.exit(1)
    
    if not env_status["verl_available"]:
        print("❌ VERL not available. Please install: pip install verl")
        sys.exit(1)
    
    if env_status["cuda_available"]:
        print("✅ CUDA is available for GPU acceleration")
    else:
        print("⚠️  CUDA not available, will use CPU (slower)")
    
    # Load and validate configuration
    config, validation_results = validate_and_prepare_training(
        config_path=args.config,
        config_name=args.config_name,
        overrides=args.overrides,
        check_files=not args.no_file_check
    )
    
    # Print configuration summary unless quiet mode
    if not args.quiet:
        print_config_summary(config, validation_results)
    
    # Check validation results
    if not validation_results["valid"]:
        print("\n❌ Configuration validation failed with errors:")
        for error in validation_results["errors"]:
            print(f"   • {error}")
        print("\nPlease fix these errors before training.")
        sys.exit(1)
    
    if validation_results["warnings"] and not args.quiet:
        print(f"\n⚠️  Found {len(validation_results['warnings'])} warnings.")
        print("Training will continue, but you may want to review these.")
    
    # Create necessary directories
    try:
        setup_directories(config)
        print("📁 Created necessary directories")
    except Exception as e:
        print(f"⚠️  Failed to create directories: {e}")
    
    # Dry run mode - exit after validation
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")
        print("Configuration is valid and ready for training.")
        sys.exit(0)
    
    # Start training
    print("\n🚀 Starting training...")
    print("="*60)
    
    try:
        training_loop = train_ppo_model(
            config_path=args.config,
            experiment_name=args.experiment,
        )
        print("✅ Training completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies for training: {e}")
        print("Please install required packages: pip install torch transformers verl")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.exception("Training error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()