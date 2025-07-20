"""VERL-compatible reward function using our existing reward system."""

import sys
from pathlib import Path
from typing import Optional

# Add the project root to path to import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rlvr_summary.rewards.integration import create_reward_function
from rlvr_summary.rewards.fenice import set_fenice_document_cache

# Create the reward function once at module level for efficiency
_config_path = project_root / "configs" / "rewards" / "rule_bundle.yaml"
_reward_fn = None


def _get_reward_function():
    """Get the reward function, creating it if necessary."""
    global _reward_fn
    if _reward_fn is None:
        # Use rule-based system with FENICE included as a weighted rule
        _reward_fn = create_reward_function(config_path=str(_config_path))
    return _reward_fn


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """VERL-compatible reward function using our rule-based reward system.

    This function follows VERL's expected signature while using our rule-based
    reward system with FENICE factual consistency scoring included as a weighted
    component alongside other rules.

    Args:
        data_source: Name of the dataset (e.g., "cnn_dailymail")
        solution_str: Generated summary text
        ground_truth: Reference summary or original article text
        extra_info: Additional information including potential FENICE document cache

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

    # Check for FENICE document cache in extra_info and set it if available
    if extra_info and isinstance(extra_info, dict):
        fenice_cache = extra_info.get('fenice_document_cache')
        if fenice_cache:
            set_fenice_document_cache(fenice_cache)
    
    # Use the unified reward function
    reward_fn = _get_reward_function()
    score = reward_fn(source_text, solution_str)
    
    # Clear the cache after use to avoid memory leaks
    if extra_info and isinstance(extra_info, dict) and extra_info.get('fenice_document_cache'):
        set_fenice_document_cache(None)

    # Ensure score is in valid range
    return float(max(0.0, min(1.0, score)))
