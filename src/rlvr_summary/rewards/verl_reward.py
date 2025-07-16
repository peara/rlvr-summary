"""VERL-compatible reward function using our existing reward system."""

import sys
from pathlib import Path
from typing import Optional

# Add the project root to path to import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rlvr_summary.rewards.integration import create_reward_function
from rlvr_summary.rewards.combined import create_combined_reward_system

# Create the reward function once at module level for efficiency
_config_path = project_root / "configs" / "rewards" / "rule_bundle.yaml"
_reward_fn = None
_use_combined_system = True  # Enable FENICE + rules by default

def _get_reward_function():
    """Get the reward function, creating it if necessary."""
    global _reward_fn
    if _reward_fn is None:
        if _use_combined_system:
            # Use combined FENICE + rule-based system
            combined_system = create_combined_reward_system(
                fenice_weight=0.7,
                rule_weight=0.3
            )
            
            def combined_reward_fn(source: str, summary: str) -> float:
                result = combined_system.evaluate(source, summary)
                return result.total_score
            
            _reward_fn = combined_reward_fn
        else:
            # Fall back to rule-based only
            _reward_fn = create_reward_function(config_path=str(_config_path))
    return _reward_fn


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """VERL-compatible reward function using our sophisticated reward system.

    This function follows VERL's expected signature while using our combined
    FENICE + rule-based reward system internally.

    Args:
        data_source: Name of the dataset (e.g., "cnn_dailymail")
        solution_str: Generated summary text
        ground_truth: Reference summary or original article text
        extra_info: Additional information (optional, can contain reward config)

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

    # Check for reward system configuration in extra_info
    global _use_combined_system
    if extra_info:
        # Allow dynamic configuration
        use_combined = extra_info.get("use_combined_system", _use_combined_system)
        fenice_weight = extra_info.get("fenice_weight", 0.7)
        rule_weight = extra_info.get("rule_weight", 0.3)
        
        if use_combined:
            # Create temporary combined system with custom config
            from rlvr_summary.rewards.combined import create_combined_reward_system
            temp_system = create_combined_reward_system(
                fenice_weight=fenice_weight,
                rule_weight=rule_weight
            )
            result = temp_system.evaluate(source_text, solution_str)
            score = result.total_score
        else:
            # Use rule-based only
            reward_fn = create_reward_function(config_path=str(_config_path))
            score = reward_fn(source_text, solution_str)
    else:
        # Use default cached reward function
        reward_fn = _get_reward_function()
        score = reward_fn(source_text, solution_str)

    # Ensure score is in valid range
    return float(max(0.0, min(1.0, score)))


def configure_reward_system(
    use_combined: bool = True,
    fenice_weight: float = 0.7,
    rule_weight: float = 0.3
) -> None:
    """Configure the reward system globally.
    
    Args:
        use_combined: Whether to use combined FENICE + rules system
        fenice_weight: Weight for FENICE score (default 0.7)
        rule_weight: Weight for rule-based score (default 0.3)
    """
    global _use_combined_system, _reward_fn
    
    _use_combined_system = use_combined
    _reward_fn = None  # Reset cached function to pick up new config
    
    print(f"Reward system configured: combined={use_combined}, "
          f"weights=({fenice_weight:.1f}, {rule_weight:.1f})")


def get_reward_system_info() -> dict:
    """Get information about the current reward system configuration.
    
    Returns:
        Dictionary with current configuration
    """
    return {
        "use_combined_system": _use_combined_system,
        "config_path": str(_config_path),
        "function_cached": _reward_fn is not None
    }
