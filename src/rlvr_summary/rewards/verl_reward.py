"""VERL-compatible reward function using our existing reward system."""

import sys
from pathlib import Path
from typing import Optional

# Add the project root to path to import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .integration import create_reward_function

# Create the reward function once at module level for efficiency
_config_path = project_root / "configs" / "rewards" / "combined_fenice.yaml"
_reward_fn = None

def _get_reward_function():
    """Get the reward function, creating it if necessary."""
    global _reward_fn
    if _reward_fn is None:
        # Use enhanced rule-based system with FENICE included as a weighted rule
        _reward_fn = create_reward_function(config_path=str(_config_path))
    return _reward_fn


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """VERL-compatible reward function using our enhanced reward system.

    This function follows VERL's expected signature while using our unified
    rule-based reward system with FENICE factual consistency scoring included
    as a weighted component.

    Args:
        data_source: Name of the dataset (e.g., "cnn_dailymail")
        solution_str: Generated summary text
        ground_truth: Reference summary or original article text
        extra_info: Additional information (optional, currently unused)

    Returns:
        float: Reward score between 0 and 1
    """
    # Basic validation
    if not solution_str or not isinstance(solution_str, str):
        return 0.0

    if len(solution_str.strip()) == 0:
        return 0.0

    # Determine the source text
    # In our case, ground_truth might be the reference summary or the original article
    # For summarization tasks, we typically want to use the original article as source
    source_text = ground_truth if ground_truth else ""

    # Use the unified reward function
    reward_fn = _get_reward_function()
    score = reward_fn(source_text, solution_str)

    # Ensure score is in valid range
    return float(max(0.0, min(1.0, score)))


def configure_reward_system(config_path: Optional[str] = None) -> None:
    """Configure the reward system globally.
    
    Args:
        config_path: Path to configuration file (default: combined_fenice.yaml)
    """
    global _reward_fn, _config_path
    
    if config_path:
        _config_path = Path(config_path)
    
    _reward_fn = None  # Reset cached function to pick up new config
    
    print(f"Reward system configured to use: {_config_path}")


def get_reward_system_info() -> dict:
    """Get information about the current reward system configuration.
    
    Returns:
        Dictionary with current configuration
    """
    return {
        "config_path": str(_config_path),
        "function_cached": _reward_fn is not None,
        "system_type": "unified_rule_based_with_fenice"
    }
