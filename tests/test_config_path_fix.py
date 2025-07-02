"""Test for the config_path fix in load_config function."""
import os
import tempfile
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.utils.config import load_config


class TestConfigPathFix:
    """Test that the config_path fix resolves the Hydra path issue."""
    
    def test_load_config_with_default_path(self):
        """Test that load_config works with default config_path (None)."""
        # Change to the project root directory for this test
        original_cwd = os.getcwd()
        try:
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            
            # This should work without throwing HydraException about relative paths
            config = load_config()
            assert config is not None
            assert "project" in config
            assert config.project.name == "rlvr-summary"
            
        finally:
            os.chdir(original_cwd)
    
    def test_load_config_with_relative_path(self):
        """Test that load_config works with relative config_path."""
        original_cwd = os.getcwd()
        try:
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            
            # Test with explicit relative path
            config = load_config(config_path="configs")
            assert config is not None
            assert "project" in config
            assert config.project.name == "rlvr-summary"
            
        finally:
            os.chdir(original_cwd)
    
    def test_load_config_with_absolute_path(self):
        """Test that load_config works with absolute config_path."""
        original_cwd = os.getcwd()
        try:
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            
            # Test with absolute path
            abs_config_path = project_root / "configs"
            config = load_config(config_path=str(abs_config_path))
            assert config is not None
            assert "project" in config
            assert config.project.name == "rlvr-summary"
            
        finally:
            os.chdir(original_cwd)
    
    def test_no_hydra_exception_raised(self):
        """Test that the specific HydraException about relative paths is not raised."""
        from hydra.errors import HydraException
        
        original_cwd = os.getcwd()
        try:
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            
            # This should not raise "config_path in initialize() must be relative"
            try:
                config = load_config()
                # If we get here, the fix worked
                assert True
            except HydraException as e:
                if "config_path in initialize() must be relative" in str(e):
                    pytest.fail("The original HydraException was raised - fix didn't work")
                else:
                    # Some other Hydra exception, re-raise it
                    raise
                
        finally:
            os.chdir(original_cwd)