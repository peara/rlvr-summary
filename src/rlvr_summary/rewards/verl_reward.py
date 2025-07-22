"""VERL-compatible reward function using our existing reward system."""

import sys
from pathlib import Path
from typing import Optional

# Add the project root to path to import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rlvr_summary.rewards.integration import create_reward_integrator
from rlvr_summary.rewards.fenice import _validate_cache_for_document, _get_document_key

# Create the reward integrator once at module level for efficiency
_config_path = project_root / "configs" / "rewards" / "rule_bundle.yaml"
_reward_integrator = None


def _get_reward_integrator():
    """Get the reward integrator, creating it if necessary."""
    global _reward_integrator
    if _reward_integrator is None:
        # Use rule-based system with FENICE included as a weighted rule
        _reward_integrator = create_reward_integrator(config_path=str(_config_path))
    return _reward_integrator


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

    # Extract and validate FENICE cache if available
    context = None
    if extra_info and isinstance(extra_info, dict):
        cache_data = extra_info.get('fenice_document_cache')
        if cache_data and _validate_cache_for_document(cache_data, source_text):
            context = {'fenice_cache': cache_data}
        elif cache_data:
            # Cache validation failed - log but continue with runtime computation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("FENICE cache validation failed, falling back to runtime computation")
    
    # Use the reward integrator with cache context
    reward_integrator = _get_reward_integrator()
    score = reward_integrator.compute_reward(source_text, solution_str, context=context)

    # Ensure score is in valid range
    return float(max(0.0, min(1.0, score)))
