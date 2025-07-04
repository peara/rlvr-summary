"""VERL-compatible reward function using our existing reward system."""

import sys
from pathlib import Path
from typing import Optional

# Add the project root to path to import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rlvr_summary.rewards.integration import create_reward_function


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """VERL-compatible reward function using our existing reward system.
    
    This function follows VERL's expected signature while using our sophisticated
    rule-based reward system internally.
    
    Args:
        data_source: Name of the dataset (e.g., "cnn_dailymail")
        solution_str: Generated summary text
        ground_truth: Reference summary or original article text
        extra_info: Additional information (optional)
        
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
    
    # Get the config path for our reward system
    config_path = project_root / "configs" / "rewards" / "rule_bundle.yaml"
    
    try:
        # Create our reward function with the configured rules
        reward_fn = create_reward_function(config_path=str(config_path))
        
        # Compute the reward using our sophisticated rule system
        score = reward_fn(source_text, solution_str)
        
        # Ensure score is in valid range
        return float(max(0.0, min(1.0, score)))
        
    except Exception as e:
        # Fallback to basic scoring if our system fails
        print(f"Warning: Reward system failed ({e})")
