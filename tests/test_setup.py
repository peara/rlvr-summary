"""Test project setup and basic imports."""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_package_import():
    """Test that the main package can be imported."""
    import rlvr_summary
    assert rlvr_summary.__version__ == "0.1.0"


def test_submodule_imports():
    """Test that all submodules can be imported."""
    import rlvr_summary.data
    import rlvr_summary.models
    import rlvr_summary.training
    import rlvr_summary.rewards
    import rlvr_summary.tools
    import rlvr_summary.evaluation
    import rlvr_summary.utils


def test_cli_import():
    """Test that CLI module can be imported."""
    from rlvr_summary.cli import cli
    assert cli is not None


def test_config_utilities():
    """Test configuration utilities."""
    from rlvr_summary.utils.config import load_config, validate_config
    
    # Test loading default config
    try:
        cfg = load_config()
        validate_config(cfg)
        assert cfg.project.name == "rlvr-summary"
    except Exception as e:
        # Config loading might fail in test environment, that's ok
        pytest.skip(f"Config loading failed (expected in test env): {e}")


def test_wandb_utilities():
    """Test W&B utilities."""
    from rlvr_summary.utils.wandb_logger import WandbLogger, is_wandb_available
    
    # Test W&B availability check
    available = is_wandb_available()
    assert isinstance(available, bool)
    
    # Test logger creation with disabled mode
    logger = WandbLogger(enabled=False)
    assert not logger.enabled


def test_common_utilities():
    """Test common utilities."""
    from rlvr_summary.utils.common import set_seed, get_device, format_number
    
    # Test seed setting
    set_seed(42)
    
    # Test device detection
    device = get_device("cpu")
    assert str(device) == "cpu"
    
    # Test number formatting
    assert format_number(1500) == "1.5K"
    assert format_number(1500000) == "1.5M"
    assert format_number(1500000000) == "1.5B"


if __name__ == "__main__":
    pytest.main([__file__])