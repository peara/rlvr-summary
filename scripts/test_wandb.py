#!/usr/bin/env python3
"""Script to test Weights & Biases integration."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.utils.wandb_logger import (
    WandbLogger,
    is_wandb_available,
    test_wandb_connection
)
from rlvr_summary.utils.common import setup_logging


def main():
    """Test W&B integration."""
    setup_logging(level="INFO")
    
    print("🔍 Testing RLVR Summary W&B Integration")
    print("=" * 50)
    
    # Test W&B availability
    print("1. Checking W&B availability...")
    available = is_wandb_available()
    print(f"   W&B available: {available}")
    
    if not available:
        print("   ⚠️  W&B not available. Install with: pip install wandb")
        print("   ⚠️  Set API key with: wandb login")
        return
    
    # Test connection
    print("\n2. Testing W&B connection...")
    connected = test_wandb_connection()
    print(f"   W&B connected: {connected}")
    
    if not connected:
        print("   ⚠️  W&B not properly configured. Run: wandb login")
        print("   ℹ️  Testing in offline mode...")
        
        # Test offline mode
        logger = WandbLogger(
            project="rlvr-summary-test",
            name="integration-test-offline",
            enabled=True,
        )
        
        # Log some test metrics
        logger.log({"test_metric": 0.5, "step": 1})
        logger.log({"test_metric": 0.7, "step": 2})
        
        logger.finish()
        print("   ✅ Offline logging test completed")
        return
    
    # Test online mode
    print("\n3. Testing W&B online logging...")
    logger = WandbLogger(
        project="rlvr-summary-test",
        name="integration-test-online",
        tags=["test", "integration"],
        notes="Testing W&B integration for RLVR Summary",
        enabled=True,
    )
    
    # Log test configuration
    test_config = {
        "model": "test-model",
        "learning_rate": 1e-4,
        "batch_size": 16,
    }
    logger.log_config(test_config)
    
    # Log test metrics
    for step in range(5):
        metrics = {
            "loss": 1.0 - (step * 0.1),
            "accuracy": step * 0.2,
            "step": step,
        }
        logger.log(metrics, step=step)
    
    logger.finish()
    print("   ✅ Online logging test completed")
    
    print("\n🎉 W&B integration test successful!")
    print("   Check your W&B dashboard for the test runs.")


if __name__ == "__main__":
    main()