"""VERL-compatible reward function using our existing reward system."""

import sys
from pathlib import Path
from typing import Optional

# Add the project root to path to import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rlvr_summary.rewards.integration import create_reward_integrator

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

    # Extract the original document and FENICE cache from extra_info
    source_text = ""
    context = None

    if extra_info and isinstance(extra_info, dict):
        cache_data = extra_info.get("fenice_document_cache")

        # Handle JSON-serialized cache data from parquet files
        if isinstance(cache_data, str):
            try:
                import json

                cache_data = json.loads(cache_data)
            except (json.JSONDecodeError, ValueError) as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to deserialize FENICE cache: {e}")
                cache_data = None

        # Use cached document text as source (pre-computed cache is always correct)
        if cache_data and isinstance(cache_data, dict):
            source_text = cache_data.get("document_text", "")
            context = {"fenice_cache": cache_data}
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Using pre-computed FENICE document cache")

    # If no cached document text, fall back to ground_truth
    if not source_text:
        source_text = ground_truth if ground_truth else ""
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "No FENICE cache available, using runtime computation"
        )  # Use the reward integrator with cache context
    reward_integrator = _get_reward_integrator()
    score = reward_integrator.compute_reward(source_text, solution_str, context=context)

    # Ensure score is in valid range
    return float(max(0.0, min(1.0, score)))
