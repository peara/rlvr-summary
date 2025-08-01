"""Reward systems and scoring."""

from .base import BaseRule, RuleEvaluationResult, TextProcessor
from .bertscore import BertScoreConsistencyRule, create_bertscore_scorer
from .fenice import FENICEScorer, create_fenice_scorer
from .integration import (
    RewardSystemIntegrator,
    create_reward_function,
    create_reward_integrator,
)
from .rule_bundle import (
    RuleBundleRewardSystem,
    create_default_rule_bundle,
    load_rule_bundle_from_config,
)
from .rules import (
    EntityOverlapRule,
    FluencyRule,
    LengthConstraintRule,
    NumberConsistencyRule,
    ProfanityDetectionRule,
)

__all__ = [
    "BaseRule",
    "RuleEvaluationResult",
    "TextProcessor",
    "LengthConstraintRule",
    "EntityOverlapRule",
    "NumberConsistencyRule",
    "ProfanityDetectionRule",
    "FluencyRule",
    "FENICEScorer",
    "create_fenice_scorer",
    "BertScoreConsistencyRule",
    "create_bertscore_scorer",
    "RuleBundleRewardSystem",
    "load_rule_bundle_from_config",
    "create_default_rule_bundle",
    "RewardSystemIntegrator",
    "create_reward_integrator",
    "create_reward_function",
]
