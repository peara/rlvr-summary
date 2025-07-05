#!/usr/bin/env python3
"""
VERL PPO Training Script for RLVR Summary.

This script follows the VERL quickstart guide and uses VERL's built-in
main_ppo.py trainer with our custom configuration and reward function.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_config_with_paths(config: OmegaConf, data_summary_path: Optional[str] = None):
    """Update VERL config with data paths and reward function path."""
    
    # Update reward function path
    reward_file_path = Path(__file__).parent.parent / "src" / "rlvr_summary" / "rewards" / "verl_reward.py"
    if not reward_file_path.exists():
        raise FileNotFoundError(f"VERL reward function not found: {reward_file_path}")
    
    config.custom_reward_function.path = str(reward_file_path)
    logger.info(f"✅ Set reward function path: {reward_file_path}")
    
    # Update data paths if provided
    if data_summary_path:
        data_summary = OmegaConf.load(data_summary_path)
        config.data.train_files = data_summary.train_files
        config.data.val_files = data_summary.val_files
        logger.info(f"✅ Loaded data paths from {data_summary_path}")
        logger.info(f"  📊 Train files: {config.data.train_files}")
        logger.info(f"  📊 Val files: {config.data.val_files}")
    else:
        # Check if data files are specified directly
        if not config.data.train_files or not config.data.val_files:
            logger.warning("⚠️  No data files specified. Please run prepare_data_verl.py first or specify --data-summary")
    
    return config


def setup_wandb_config(config: OmegaConf, wandb_config: dict):
    """Setup W&B configuration."""
    if wandb_config.get("enabled", False):
        # Update experiment name with W&B run name if provided
        if "run_name" in wandb_config:
            config.trainer.experiment_name = wandb_config["run_name"]
        
        # Set project name
        if "project" in wandb_config:
            config.trainer.project_name = wandb_config["project"]
        
        logger.info(f"✅ W&B enabled: {config.trainer.project_name}/{config.trainer.experiment_name}")
    else:
        # Remove wandb from logger list if disabled
        if "wandb" in config.trainer.logger:
            config.trainer.logger = [l for l in config.trainer.logger if l != "wandb"]
        logger.info("✅ W&B disabled, using console logging only")


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="VERL PPO Training for RLVR Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default config
  python scripts/train_verl.py
  
  # Training with custom config
  python scripts/train_verl.py --config configs/verl_ppo_config.yaml
  
  # Training with prepared data
  python scripts/train_verl.py --data-summary ./data/verl/data_summary.yaml
  
  # Training with W&B logging
  python scripts/train_verl.py --wandb-project rlvr-summary --wandb-run baseline-v1
  
  # Training with overrides
  python scripts/train_verl.py trainer.total_epochs=2 data.train_batch_size=8
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/verl_ppo_config.yaml",
        help="Path to VERL configuration file"
    )
    
    parser.add_argument(
        "--data-summary",
        type=str,
        help="Path to data summary file from prepare_data_verl.py"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name"
    )
    
    parser.add_argument(
        "--wandb-run",
        type=str,
        help="W&B run name"
    )
    
    parser.add_argument(
        "--wandb-enabled",
        action="store_true",
        help="Enable W&B logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides (e.g., trainer.total_epochs=2)"
    )
    
    args = parser.parse_args()
    
    # Load VERL configuration
    try:
        config = OmegaConf.load(args.config)
        logger.info(f"✅ Loaded VERL config from {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.overrides:
        for override in args.overrides:
            try:
                key, value = override.split("=", 1)
                # Convert value to appropriate type
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)
                
                # Use OmegaConf.update to set nested keys
                nested_dict = {}
                keys = key.split(".")
                current_dict = nested_dict
                for k in keys[:-1]:
                    current_dict[k] = {}
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = value
                
                config = OmegaConf.merge(config, nested_dict)
                logger.info(f"✅ Applied override: {key} = {value}")
            except Exception as e:
                logger.error(f"❌ Invalid override '{override}': {e}")
                sys.exit(1)
    
    # Setup W&B configuration
    wandb_config = {
        "enabled": args.wandb_enabled or args.wandb_project is not None,
        "project": args.wandb_project or "rlvr-summary",
        "run_name": args.wandb_run,
    }
    setup_wandb_config(config, wandb_config)
    
    # Update config with paths
    try:
        config = update_config_with_paths(config, args.data_summary)
    except Exception as e:
        logger.error(f"❌ Failed to update config paths: {e}")
        sys.exit(1)
    
    # Validate configuration
    logger.info("🔍 Validating configuration...")
    
    # Check data files exist
    for train_file in config.data.train_files:
        if not Path(train_file).exists():
            logger.error(f"❌ Training file not found: {train_file}")
            sys.exit(1)
    
    for val_file in config.data.val_files:
        if not Path(val_file).exists():
            logger.error(f"❌ Validation file not found: {val_file}")
            sys.exit(1)
    
    # Check reward function exists
    if not Path(config.custom_reward_function.path).exists():
        logger.error(f"❌ Reward function not found: {config.custom_reward_function.path}")
        sys.exit(1)
    
    logger.info("✅ Configuration validation passed")
    
    # Print configuration summary
    print("\n" + "="*60)
    print("🔧 VERL TRAINING CONFIGURATION")
    print("="*60)
    print(f"📋 Project: {config.trainer.project_name}")
    print(f"🏷️  Experiment: {config.trainer.experiment_name}")
    print(f"🤖 Model: {config.actor_rollout_ref.model.path}")
    print(f"📊 Train files: {len(config.data.train_files)} files")
    print(f"📊 Val files: {len(config.data.val_files)} files")
    print(f"📦 Batch size: {config.data.train_batch_size}")
    print(f"🔄 PPO epochs: {config.actor_rollout_ref.actor.ppo_epochs}")
    print(f"📚 Learning rate: {config.actor_rollout_ref.actor.optim.lr}")
    print(f"🎯 Total training steps: {config.actor_rollout_ref.actor.optim.total_training_steps}")
    print(f"🏆 Reward function: {config.custom_reward_function.path}")
    print(f"📈 Logging: {', '.join(config.trainer.logger)}")
    print("="*60)
    
    if args.dry_run:
        print("✅ Dry run completed successfully. Configuration is valid.")
        return
    
    # Save final configuration for reference
    config_save_path = Path("./outputs/verl_training_config.yaml")
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_save_path)
    logger.info(f"💾 Saved final config to {config_save_path}")
    
    # Import and run VERL training
    print("\n🚀 Starting VERL PPO training...")
    print("="*60)
    
    try:
        # Import VERL's main training function
        from verl.trainer.main_ppo import run_ppo
        
        # Run VERL training with our configuration
        run_ppo(config)
        
        print("\n" + "="*60)
        print("🎉 VERL PPO training completed successfully!")
        print("="*60)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import VERL: {e}")
        logger.error("Please ensure VERL is properly installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
