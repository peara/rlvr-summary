#!/usr/bin/env python3
"""Setup script for RLVR Summary project."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def main():
    """Set up the RLVR Summary project."""
    print("🚀 RLVR Summary Project Setup")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"❌ Python 3.9+ required, found {python_version.major}.{python_version.minor}")
        sys.exit(1)
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install package in development mode
    success = run_command(
        "pip install -e .",
        "Installing RLVR Summary package in development mode"
    )
    if not success:
        print("❌ Failed to install package. Trying with requirements.txt...")
        success = run_command(
            "pip install -r requirements.txt",
            "Installing dependencies from requirements.txt"
        )
        if not success:
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Install development dependencies
    run_command(
        "pip install -r requirements-dev.txt",
        "Installing development dependencies"
    )
    
    # Set up pre-commit hooks
    run_command(
        "pre-commit install",
        "Setting up pre-commit hooks"
    )
    
    # Create necessary directories
    print("📁 Creating project directories...")
    directories = [
        "data", "outputs", "logs", "checkpoints",
        "data/raw", "data/processed", "data/cache",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created {directory}/")
    
    # Test basic imports
    print("🧪 Testing basic package imports...")
    try:
        import rlvr_summary
        print(f"   ✅ Successfully imported rlvr_summary v{rlvr_summary.__version__}")
    except ImportError as e:
        print(f"   ❌ Failed to import rlvr_summary: {e}")
    
    # Test CLI
    print("🖥️  Testing CLI...")
    result = run_command(
        "python -m rlvr_summary.cli --help",
        "Testing CLI functionality"
    )
    
    # Test configuration loading
    print("⚙️  Testing configuration...")
    result = run_command(
        "python -c \"from rlvr_summary.utils.config import load_config; cfg = load_config(); print('Config loaded successfully')\"",
        "Testing configuration loading"
    )
    
    # Test W&B integration (optional)
    print("🔗 Testing W&B integration...")
    result = run_command(
        "python scripts/test_wandb.py",
        "Testing Weights & Biases integration"
    )
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Configure W&B: wandb login")
    print("2. Run tests: pytest tests/")
    print("3. Start with Phase A implementation")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()