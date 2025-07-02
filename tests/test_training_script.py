"""Tests for the training script functionality."""

import pytest
import tempfile
import yaml
import sys
from pathlib import Path
import subprocess
import os

# Add scripts to path for testing
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

try:
    from train_simple import (
        load_yaml_config, 
        validate_basic_config, 
        find_config_file,
        check_environment,
        create_directories_from_config
    )
    SCRIPT_AVAILABLE = True
except ImportError:
    SCRIPT_AVAILABLE = False


class TestTrainingScript:
    """Test training script components."""
    
    def test_script_help(self):
        """Test that the script shows help correctly."""
        result = subprocess.run(
            [sys.executable, "scripts/train_simple.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        assert result.returncode == 0
        assert "RLVR Summary Training Script" in result.stdout
        assert "--config" in result.stdout
        assert "--experiment" in result.stdout
        assert "--dry-run" in result.stdout
    
    @pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="Training script not available")
    def test_load_yaml_config(self):
        """Test YAML configuration loading."""
        config_data = {
            "project": {"name": "test-project", "version": "1.0.0"},
            "paths": {"data_dir": "./data", "output_dir": "./outputs"},
            "training": {"learning_rate": 1e-4, "batch_size": 16}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            loaded_config = load_yaml_config(config_file)
            assert loaded_config["project"]["name"] == "test-project"
            assert loaded_config["training"]["learning_rate"] == 1e-4
        finally:
            os.unlink(config_file)
    
    @pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="Training script not available")
    def test_validate_basic_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            "project": {"name": "test-project"},
            "paths": {
                "data_dir": "./data",
                "output_dir": "./outputs",
                "log_dir": "./logs",
                "checkpoint_dir": "./checkpoints"
            },
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 16,
                "max_steps": 1000
            },
            "model": {"model_name": "gpt2"}
        }
        
        results = validate_basic_config(config)
        assert results["valid"] is True
        assert len(results["errors"]) == 0
    
    @pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="Training script not available")
    def test_validate_basic_config_missing_sections(self):
        """Test configuration validation with missing sections."""
        config = {
            "training": {"learning_rate": 1e-4}
        }
        
        results = validate_basic_config(config)
        assert results["valid"] is False
        assert any("Missing required config section: project" in error for error in results["errors"])
        assert any("Missing required config section: paths" in error for error in results["errors"])
    
    @pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="Training script not available")
    def test_validate_basic_config_invalid_params(self):
        """Test configuration validation with invalid parameters."""
        config = {
            "project": {"name": "test-project"},
            "paths": {"data_dir": "./data"},
            "training": {
                "learning_rate": 1.0,  # Too high
                "batch_size": -1,      # Invalid
                "max_steps": 1000
            }
        }
        
        results = validate_basic_config(config)
        assert results["valid"] is False
        assert any("Batch size must be positive" in error for error in results["errors"])
        assert any("Learning rate 1.0 may be outside typical range" in warning for warning in results["warnings"])
    
    @pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="Training script not available")
    def test_check_environment(self):
        """Test environment checking."""
        env_status = check_environment()
        
        assert "python_version" in env_status
        assert "torch_available" in env_status
        assert "transformers_available" in env_status
        assert "trl_available" in env_status
        assert isinstance(env_status["torch_available"], bool)
    
    @pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="Training script not available")
    def test_create_directories_from_config(self):
        """Test directory creation from config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "paths": {
                    "data_dir": os.path.join(temp_dir, "data"),
                    "output_dir": os.path.join(temp_dir, "outputs"),
                    "checkpoint_dir": os.path.join(temp_dir, "checkpoints")
                }
            }
            
            create_directories_from_config(config)
            
            assert os.path.exists(config["paths"]["data_dir"])
            assert os.path.exists(config["paths"]["output_dir"])
            assert os.path.exists(config["paths"]["checkpoint_dir"])
    
    def test_script_dry_run_with_valid_config(self):
        """Test script dry run with the actual config file."""
        result = subprocess.run(
            [sys.executable, "scripts/train_simple.py", "--dry-run", "--quiet"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Should succeed with the existing config
        assert result.returncode == 0
        assert "Dry run completed successfully" in result.stdout
    
    def test_script_with_nonexistent_config(self):
        """Test script with nonexistent config file."""
        result = subprocess.run(
            [sys.executable, "scripts/train_simple.py", "--config", "nonexistent.yaml", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Should fail with config not found
        assert result.returncode == 1
        assert "Failed to load configuration" in result.stdout


class TestConfigurationFiles:
    """Test the actual configuration files in the project."""
    
    def test_main_config_structure(self):
        """Test that the main config file has required structure."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        assert config_path.exists(), "Main config file should exist"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        assert "project" in config
        assert "paths" in config
        assert "name" in config.get("project", {})
        
        # Check paths
        paths = config.get("paths", {})
        assert "data_dir" in paths
        assert "output_dir" in paths
        assert "log_dir" in paths
        assert "checkpoint_dir" in paths
    
    def test_training_config_structure(self):
        """Test that training config files have required structure."""
        training_config_path = Path(__file__).parent.parent / "configs" / "training" / "ppo_baseline.yaml"
        assert training_config_path.exists(), "Training config file should exist"
        
        with open(training_config_path) as f:
            config = yaml.safe_load(f)
        
        # Check training parameters
        assert "learning_rate" in config
        assert "batch_size" in config
        assert "max_steps" in config


if __name__ == "__main__":
    # Run basic tests
    test_script = TestTrainingScript()
    test_config = TestConfigurationFiles()
    
    print("Running training script tests...")
    
    try:
        test_script.test_script_help()
        print("✅ Script help test passed")
    except Exception as e:
        print(f"❌ Script help test failed: {e}")
    
    try:
        test_script.test_script_dry_run_with_valid_config()
        print("✅ Dry run test passed")
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
    
    try:
        test_script.test_script_with_nonexistent_config()
        print("✅ Nonexistent config test passed")
    except Exception as e:
        print(f"❌ Nonexistent config test failed: {e}")
    
    try:
        test_config.test_main_config_structure()
        print("✅ Main config structure test passed")
    except Exception as e:
        print(f"❌ Main config structure test failed: {e}")
    
    try:
        test_config.test_training_config_structure()
        print("✅ Training config structure test passed")
    except Exception as e:
        print(f"❌ Training config structure test failed: {e}")
    
    print("✨ Basic tests completed!")