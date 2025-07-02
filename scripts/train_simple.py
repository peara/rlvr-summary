#!/usr/bin/env python3
"""Simple training script for RLVR Summary.

This script provides automated training with configuration validation,
dependency checking, and clear error reporting.

This version works with basic dependencies and falls back gracefully.
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def validate_basic_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Basic validation for training configuration."""
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "missing_files": [],
        "missing_dependencies": []
    }
    
    # Check required top-level sections
    required_sections = ["project", "paths"]
    for section in required_sections:
        if section not in config:
            results["errors"].append(f"Missing required config section: {section}")
            results["valid"] = False
    
    # Validate project section
    if "project" in config and "name" not in config["project"]:
        results["errors"].append("Missing project.name in config")
        results["valid"] = False
    
    # Validate paths
    if "paths" in config:
        required_paths = ["data_dir", "output_dir", "log_dir", "checkpoint_dir"]
        for path_key in required_paths:
            if path_key not in config["paths"]:
                results["warnings"].append(f"Missing recommended path: {path_key}")
    
    # Check training-specific sections
    if "training" in config:
        training_cfg = config["training"]
        
        # Check required training parameters
        required_training_params = ["learning_rate", "batch_size", "max_steps"]
        for param in required_training_params:
            if param not in training_cfg:
                results["warnings"].append(f"Missing training parameter: {param}")
        
        # Validate parameter ranges
        if "learning_rate" in training_cfg:
            lr = training_cfg["learning_rate"]
            if not (1e-7 <= lr <= 1e-2):
                results["warnings"].append(f"Learning rate {lr} may be outside typical range [1e-7, 1e-2]")
        
        if "batch_size" in training_cfg:
            bs = training_cfg["batch_size"]
            if bs <= 0:
                results["errors"].append(f"Batch size must be positive, got {bs}")
                results["valid"] = False
    
    # Check model configuration
    if "model" in config:
        model_cfg = config["model"]
        if "model_name" not in model_cfg:
            results["warnings"].append("Missing model.model_name in configuration")
    
    # Check for dependencies
    try:
        import torch
    except ImportError:
        results["missing_dependencies"].append("torch")
    
    try:
        import transformers
    except ImportError:
        results["missing_dependencies"].append("transformers")
    
    try:
        import trl
    except ImportError:
        results["missing_dependencies"].append("trl")
    
    if results["missing_dependencies"]:
        results["warnings"].append("Missing dependencies for full training functionality")
    
    return results


def print_config_summary(config: Dict[str, Any], validation_results: Dict[str, Any]) -> None:
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION SUMMARY")
    print("="*60)
    
    # Project info
    if "project" in config:
        print(f"📋 Project: {config['project'].get('name', 'Unknown')}")
        print(f"🔢 Version: {config['project'].get('version', 'Unknown')}")
    
    # Model configuration
    if "model" in config:
        model_cfg = config["model"]
        print(f"🤖 Model: {model_cfg.get('model_name', 'Unknown')}")
        if "torch_dtype" in model_cfg:
            print(f"🔢 Data Type: {model_cfg['torch_dtype']}")
        if "load_in_8bit" in model_cfg:
            print(f"⚡ 8-bit Loading: {model_cfg['load_in_8bit']}")
    
    # Training configuration
    if "training" in config:
        training_cfg = config["training"]
        print(f"📚 Learning Rate: {training_cfg.get('learning_rate', 'Not set')}")
        print(f"📦 Batch Size: {training_cfg.get('batch_size', 'Not set')}")
        print(f"🔄 Max Steps: {training_cfg.get('max_steps', 'Not set')}")
        if "ppo_epochs" in training_cfg:
            print(f"🎯 PPO Epochs: {training_cfg['ppo_epochs']}")
    
    # Data configuration
    if "data" in config:
        data_cfg = config["data"]
        print(f"📊 Dataset: {data_cfg.get('dataset_name', 'Unknown')}")
    
    # Paths
    if "paths" in config:
        paths = config["paths"]
        print(f"📁 Data Directory: {paths.get('data_dir', 'Not set')}")
        print(f"📁 Output Directory: {paths.get('output_dir', 'Not set')}")
        print(f"📁 Checkpoint Directory: {paths.get('checkpoint_dir', 'Not set')}")
    
    # W&B configuration
    if "wandb" in config:
        wandb_cfg = config["wandb"]
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
        "trl_available": False,
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
        import trl
        env_status["trl_available"] = True
    except ImportError:
        pass
    
    try:
        import wandb
        env_status["wandb_available"] = True
    except ImportError:
        pass
    
    return env_status


def create_directories_from_config(config: Dict[str, Any]) -> None:
    """Create necessary directories from config."""
    if "paths" not in config:
        return
    
    for path_key, path_value in config["paths"].items():
        if isinstance(path_value, str) and path_key.endswith("_dir"):
            os.makedirs(path_value, exist_ok=True)
            print(f"📁 Created directory: {path_value}")


def find_config_file(config_path: Optional[str] = None) -> str:
    """Find the configuration file to use."""
    if config_path:
        if os.path.isfile(config_path):
            return config_path
        elif os.path.isdir(config_path):
            # Look for config.yaml in the directory
            config_file = os.path.join(config_path, "config.yaml")
            if os.path.isfile(config_file):
                return config_file
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Default locations to check
    default_locations = [
        "configs/config.yaml",
        "config.yaml",
        "configs/config.yml",
        "config.yml"
    ]
    
    for location in default_locations:
        if os.path.isfile(location):
            return location
    
    raise FileNotFoundError("No configuration file found in default locations")


def run_training_command(config: Dict[str, Any], experiment_name: Optional[str] = None) -> None:
    """Run the appropriate training command."""
    # Try to use the full RLVR Summary training if available
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    try:
        from rlvr_summary.training import train_ppo_model
        
        print("🚀 Using RLVR Summary training pipeline...")
        training_loop = train_ppo_model(
            config_path=None,  # We already loaded the config
            experiment_name=experiment_name,
        )
        print("✅ Training completed successfully!")
        
    except ImportError as e:
        print(f"⚠️  RLVR Summary training not available: {e}")
        print("📝 Would run training with the following configuration:")
        
        # Show what training command would be run
        if "training" in config:
            training_cfg = config["training"]
            print(f"   • Learning Rate: {training_cfg.get('learning_rate', 'default')}")
            print(f"   • Batch Size: {training_cfg.get('batch_size', 'default')}")
            print(f"   • Max Steps: {training_cfg.get('max_steps', 'default')}")
        
        if "model" in config:
            model_cfg = config["model"]
            print(f"   • Model: {model_cfg.get('model_name', 'default')}")
        
        print("\n💡 To enable full training, install dependencies:")
        print("   pip install torch transformers trl")
        print("   pip install -e .")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="RLVR Summary Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                              # Use default config
  python scripts/train.py --config configs            # Specify config directory
  python scripts/train.py --config configs/config.yaml # Specify config file
  python scripts/train.py --experiment my-experiment  # Set experiment name
  python scripts/train.py --dry-run                   # Validate only, don't train
  python scripts/train.py --log-level DEBUG           # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file or directory"
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Experiment name for W&B tracking"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration only, don't start training"
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
    
    if env_status["cuda_available"]:
        print("✅ CUDA is available for GPU acceleration")
    else:
        print("⚠️  CUDA not available, will use CPU (slower)")
    
    # Find and load configuration
    print("🔍 Loading configuration...")
    try:
        config_file = find_config_file(args.config)
        print(f"📄 Using config file: {config_file}")
        
        config = load_yaml_config(config_file)
        print("✅ Configuration loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    print("🔍 Validating configuration...")
    try:
        validation_results = validate_basic_config(config)
        print("✅ Configuration validation completed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    
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
        create_directories_from_config(config)
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
        run_training_command(config, args.experiment)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.exception("Training error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()