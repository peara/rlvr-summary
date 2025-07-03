# Rule-Based Reward System

This document describes the rule-based reward system implemented for the baseline RL-VR training loop.

## Overview

The rule-based reward system provides a configurable foundation for evaluating summary quality using multiple scoring criteria. It supports the milestone M0 target of achieving ≥20% rule-pass rate and provides detailed metrics for training monitoring.

## Architecture

### Core Components

- **BaseRule**: Abstract interface for all reward rules
- **TextProcessor**: Utility class for text processing (words, numbers, entities)
- **Individual Rules**: Specific scoring implementations
- **RuleBundleRewardSystem**: Main orchestrator with configurable weights
- **RewardSystemIntegrator**: Training loop integration utilities

### Rule Implementations

#### 1. LengthConstraintRule
Evaluates summary length against configurable constraints.

**Configuration:**
```yaml
length:
  min_words: 20           # Minimum acceptable word count
  max_words: 100          # Maximum acceptable word count  
  optimal_range: [30, 80] # Optimal word count range
  penalty_factor: 0.5     # Penalty factor for sub-optimal length
```

**Scoring:**
- Perfect score (1.0) for optimal range
- Partial score with penalty for acceptable range
- Very low score (0.1) for unacceptable range

#### 2. EntityOverlapRule
Scores entity overlap between source and summary using Jaccard similarity.

**Configuration:**
```yaml
entity:
  min_overlap: 0.3        # Minimum acceptable overlap
  optimal_overlap: 0.7    # Target overlap for full score
```

**Features:**
- Regex-based named entity extraction
- Falls back to word overlap when no entities found
- Jaccard similarity scoring

#### 3. NumberConsistencyRule
Checks consistency of numbers between source and summary.

**Configuration:**
```yaml
numbers:
  exact_match_bonus: 1.0    # Bonus for exact matches
  partial_match_bonus: 0.5  # Bonus for partial matches
  mismatch_penalty: -0.5    # Penalty for mismatches
```

**Features:**
- Extracts integers, decimals, and percentages
- Exact and partial matching with different scores
- Penalty for numbers in summary not found in source

#### 4. ProfanityDetectionRule
Detects and penalizes profanity in summaries.

**Configuration:**
```yaml
profanity:
  enabled: true          # Enable/disable profanity detection
  penalty: -1.0          # Penalty per profane word
  wordlist_path: null    # Custom wordlist path (optional)
```

**Features:**
- Built-in profanity word list
- Support for custom word lists
- Case-insensitive matching

#### 5. FluencyRule
Basic fluency evaluation using heuristics.

**Configuration:**
```yaml
fluency:
  enabled: true          # Enable/disable fluency checking
  min_score: 0.5         # Minimum score threshold
```

**Features:**
- Average word and sentence length analysis
- Penalties for very short/long sentences
- Penalties for very short/long words on average

## Usage

### Basic Usage

```python
from rlvr_summary.rewards import create_default_rule_bundle

# Create reward system with default configuration
system = create_default_rule_bundle()

# Evaluate a single summary
result = system.evaluate(source_text, summary_text)
print(f"Score: {result.total_score:.3f}")
print(f"Pass rate: {result.pass_rate:.3f}")

# Get detailed breakdown
for rule_name, score in result.rule_scores.items():
    passed = "✓" if result.rule_passed[rule_name] else "✗"
    print(f"{rule_name}: {score:.3f} {passed}")
```

### Batch Evaluation

```python
# Evaluate multiple summaries
sources = ["Source text 1", "Source text 2"]
summaries = ["Summary 1", "Summary 2"]

results = system.evaluate_batch(sources, summaries, log_details=True)
for i, result in enumerate(results):
    print(f"Sample {i}: score={result.total_score:.3f}")
```

### Configuration Loading

```python
from rlvr_summary.rewards import load_rule_bundle_from_config

# Load from YAML configuration file
system = load_rule_bundle_from_config("configs/rewards/rule_bundle.yaml")
```

### Training Loop Integration

```python
from rlvr_summary.rewards import create_reward_integrator

# Create integrator with W&B logging
integrator = create_reward_integrator(wandb_logger=logger)

# Use in training loop
for batch in dataloader:
    sources, summaries = batch
    rewards = integrator.compute_reward_batch(sources, summaries, step=step)
    
    # Rewards are automatically logged to W&B
    # Statistics are tracked automatically

# Check milestone progress
milestone = integrator.evaluate_milestone_criteria(target_pass_rate=0.2)
print(milestone["message"])
```

### Simple Function Interface

```python
from rlvr_summary.rewards import create_reward_function

# Create simple reward function for easy integration
reward_fn = create_reward_function()

# Use directly
score = reward_fn(source_text, summary_text)
```

## Configuration

### Default Configuration

The system uses the configuration from `configs/rewards/rule_bundle.yaml`:

```yaml
# Rule weights (should sum to 1.0)
weights:
  length_constraint: 0.3
  entity_overlap: 0.3
  number_consistency: 0.2
  profanity_penalty: 0.1
  fluency: 0.1

# Rule-specific configurations
length:
  min_words: 20
  max_words: 100
  optimal_range: [30, 80]
  penalty_factor: 0.5

entity:
  min_overlap: 0.3
  optimal_overlap: 0.7

numbers:
  exact_match_bonus: 1.0
  partial_match_bonus: 0.5
  mismatch_penalty: -0.5

profanity:
  enabled: true
  penalty: -1.0

fluency:
  enabled: true
  min_score: 0.5
```

### Updating Weights

```python
# Update rule weights dynamically
new_weights = {
    "length_constraint": 0.4,
    "entity_overlap": 0.3,
}
system.update_rule_weights(new_weights)
```

## VERL Integration

The reward system now integrates seamlessly with VERL's PPOTrainer for optimized performance:

### VERL Training Loop Integration

```python
from rlvr_summary.training.ppo_trainer import PPOTrainingLoop

# VERL integration with automatic reward computation
training_loop = PPOTrainingLoop(config)
training_loop.setup()  # Automatically sets up reward function

# During VERL training, rewards are computed automatically
# The training loop handles the conversion between VERL format and reward computation
```

### Reward Computation Cycle

The VERL integration includes specialized methods for the VERL training cycle:

```python
# Used internally by VERL training loop
rewards = training_loop.compute_batch_rewards(prompts, summaries)

# Article extraction from VERL prompt format
article = training_loop._extract_article_from_prompt(prompt)
```

**Benefits of VERL Integration:**
- ✅ Automatic integration with VERL's generation cycle
- ✅ Optimized memory usage and performance
- ✅ Built-in logging and metrics tracking
- ✅ Seamless prompt format conversion
- ✅ Easy customization via functional_reward interface

## Metrics and Logging

### Rule Evaluation Result

Each evaluation returns a `RuleEvaluationResult` with:

- `total_score`: Weighted aggregate score (0.0 to 1.0)
- `rule_scores`: Individual rule scores
- `rule_details`: Detailed results for each rule
- `rule_passed`: Whether each rule passed its threshold
- `pass_rate`: Fraction of rules that passed

### Metrics for W&B

The system generates metrics suitable for W&B tracking:

```python
metrics = result.get_metrics()
# Returns:
# {
#   "reward/total_score": 0.75,
#   "reward/pass_rate": 0.8,
#   "reward/length_constraint_score": 0.9,
#   "reward/length_constraint_passed": 1.0,
#   ...
# }
```

### Cumulative Statistics

The integrator tracks cumulative statistics:

```python
stats = integrator.get_cumulative_statistics()
# Returns average scores and pass rates across all evaluations
```

## Milestone Tracking

The system supports milestone evaluation for training progress:

```python
# M0 milestone: ≥20% rule-pass rate
milestone = integrator.evaluate_milestone_criteria(target_pass_rate=0.2)

if milestone["milestone_met"]:
    print("🎉 M0 milestone achieved!")
    print(f"Pass rate: {milestone['current_pass_rate']:.3f}")
else:
    print(f"Progress: {milestone['current_pass_rate']:.3f} / {milestone['target_pass_rate']:.3f}")
```

## Testing

Comprehensive test suite with 95+ test cases:

```bash
# Run all tests
python tests/test_rewards.py

# Run specific test classes
python -c "from tests.test_rewards import TestLengthConstraintRule; TestLengthConstraintRule().test_optimal_length()"
```

## Demo

Run the demo script to see the system in action:

```bash
python demo_rewards.py
```

The demo showcases:
- All rule components with real examples
- Batch evaluation and statistics
- Configuration updates
- Milestone tracking

## Performance

The reward system is designed for efficiency:

- Built using Python built-ins (no external NLP dependencies)
- Regex-based text processing for speed
- Batch processing support
- Minimal memory footprint
- Fast evaluation (< 1ms per summary)

## Future Enhancements

Planned improvements for later phases:

1. **Enhanced Entity Recognition**: Integration with spaCy/BERT-based NER
2. **Advanced Fluency Scoring**: Language model-based fluency evaluation
3. **Semantic Similarity**: Sentence embedding-based similarity scoring
4. **Dynamic Weight Adjustment**: Adaptive weight tuning during training
5. **Custom Rule Development**: Framework for adding domain-specific rules

## Integration with Training Loops

The reward system is designed to integrate seamlessly with RL training frameworks:

### HuggingFace VERL Integration

```python
from verl import PPOTrainer
from rlvr_summary.rewards import create_reward_function

# Create reward function
reward_fn = create_reward_function()

# Use in PPO training with VERL's functional_reward
from verl.trainer.ppo import functional_reward

# Create VERL-compatible reward function
@functional_reward
def verl_reward_fn(prompts, responses):
    rewards = []
    for prompt, response in zip(prompts, responses):
        # Extract article from prompt format
        article = extract_article_from_prompt(prompt)
        reward = reward_fn(article, response)
        rewards.append(reward)
    return rewards

# Use in VERL training
trainer = PPOTrainer(config=config, policy_model=model, ...)
trainer.set_reward_function(verl_reward_fn)
```

### Custom Training Loop

```python
from rlvr_summary.rewards import create_reward_integrator

integrator = create_reward_integrator(wandb_logger=wandb_logger)

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(batch)
        
        # Compute rewards
        rewards = integrator.compute_reward_batch(
            batch["sources"], 
            outputs["summaries"],
            step=step
        )
        
        # Compute loss with reward
        loss = compute_loss(outputs, rewards)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log milestone progress
        if step % 100 == 0:
            milestone = integrator.evaluate_milestone_criteria()
            print(milestone["message"])
```

## Troubleshooting

### Common Issues

1. **Low Scores**: Check individual rule breakdowns to identify issues
2. **Weight Warnings**: Ensure rule weights sum to 1.0
3. **Import Errors**: Some dependencies are optional (config, wandb)
4. **Performance**: Use batch evaluation for multiple samples

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

result = system.evaluate(source, summary, log_details=True)
```

### Validation

Validate configuration before use:

```python
# Check rule configuration
rule_info = system.get_rule_info()
print(f"Configured rules: {list(rule_info.keys())}")

# Check weight normalization
total_weight = sum(rule.weight for rule in system.rules.values())
print(f"Total weight: {total_weight}")
```