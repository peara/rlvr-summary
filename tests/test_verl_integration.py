#!/usr/bin/env python3
"""Test VERL integration with corrected configuration."""

import os
import sys

sys.path.append("./src")


def test_verl_trainer():
    """Test VERL PPO trainer initialization."""
    print("Testing VERL PPO trainer...")

    try:
        from rlvr_summary.training.ppo_trainer import VERLPPOTrainingLoop

        print("✓ VERLPPOTrainingLoop imported successfully")

        # Test configuration
        config = {
            "model_name": "distilgpt2",
            "checkpoint_dir": "./test_checkpoints",
            "total_epochs": 1,
            "rollout_batch_size": 2,
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "data_path": "dummy",
            "train_size": 10,
        }

        # Test initialization
        trainer = VERLPPOTrainingLoop(config=config)
        print("✓ VERLPPOTrainingLoop initialized successfully")

        # Test tokenizer setup
        trainer.setup_tokenizer()
        print("✓ Tokenizer setup successful")

        # Test reward function setup
        trainer.setup_reward_function()
        print("✓ Reward function setup successful")

        # Check if custom reward file was created
        reward_file = trainer.checkpoint_dir / "custom_reward.py"
        if reward_file.exists():
            print("✓ Custom reward function file created")

            # Test the reward function
            import importlib.util

            spec = importlib.util.spec_from_file_location("custom_reward", reward_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Test the compute_score function
            score = module.compute_score(
                data_source="test",
                solution_str="This is a test summary of moderate length with proper punctuation.",
                ground_truth="reference text",
            )
            print(f"✓ Custom reward function works, test score: {score}")

        print("✓ All VERL integration tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_verl_trainer()
    sys.exit(0 if success else 1)
