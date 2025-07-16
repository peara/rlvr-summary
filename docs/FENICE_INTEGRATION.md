# FENICE Integration Documentation

## Overview

This document describes the integration of the distilled FENICE Factual Consistency Scorer into the RLVR-Summary reward system. FENICE provides factual consistency evaluation through claim extraction and Natural Language Inference (NLI) scoring.

## Architecture

### Combined Reward System

The new reward system implements a weighted combination:

```
R = fenice_weight × FENICE_score + rule_weight × Rules_score
```

**Default Configuration:**
- FENICE weight: 0.7 (70%)
- Rules weight: 0.3 (30%)

### Components

1. **FENICE Scorer** (`src/rlvr_summary/rewards/fenice.py`)
   - Claim extraction from summaries
   - NLI-based factual consistency scoring
   - Graceful fallback when models unavailable

2. **Combined Reward System** (`src/rlvr_summary/rewards/combined.py`)
   - Weighted combination of FENICE and rule-based scores
   - Rich metrics for training integration
   - Configurable weights and thresholds

3. **Enhanced Integration** (`src/rlvr_summary/rewards/integration.py`)
   - Support for both combined and rule-only modes
   - W&B logging with FENICE metrics

## Usage

### Basic Usage

```python
from rlvr_summary.rewards import create_combined_reward_system

# Create combined system
system = create_combined_reward_system(
    fenice_weight=0.7,
    rule_weight=0.3,
    fenice_enabled=True
)

# Evaluate a summary
result = system.evaluate(source_text, summary_text)
print(f"Total score: {result.total_score:.3f}")
print(f"FENICE score: {result.fenice_score:.3f}")
print(f"Rule score: {result.rule_score:.3f}")
```

### Training Integration

The system integrates seamlessly with existing training scripts:

```bash
# Use enhanced training script with FENICE
./scripts/train_fenice.sh

# Or use existing script (automatically uses combined system)
./scripts/train_3090.sh
```

### Configuration

#### Environment Variables

```bash
export ENABLE_FENICE=true      # Enable/disable FENICE
export FENICE_WEIGHT=0.7       # FENICE weight (0.0-1.0)
export RULE_WEIGHT=0.3         # Rule weight (0.0-1.0)
```

#### VERL Interface Configuration

```python
# Configure via extra_info in compute_score
extra_info = {
    "fenice_enabled": True,
    "fenice_weight": 0.6,
    "rule_weight": 0.4,
    "use_combined_system": True
}

score = compute_score("dataset", summary, source, extra_info)
```

#### Global Configuration

```python
from rlvr_summary.rewards.verl_reward import configure_reward_system

configure_reward_system(
    use_combined=True,
    fenice_enabled=True,
    fenice_weight=0.7,
    rule_weight=0.3
)
```

### Configuration Files

Use `configs/rewards/combined_fenice.yaml` for detailed configuration:

```yaml
use_combined_system: true
fenice_weight: 0.7
rule_weight: 0.3

fenice_config:
  enabled: true
  model_name: "Babelscape/FENICE"
  threshold: 0.5
  batch_size: 8

rule_config:
  weights:
    length_constraint: 0.25
    entity_overlap: 0.25
    # ...
```

## Metrics and Logging

### Training Metrics

The combined system provides rich metrics for tracking:

```python
metrics = result.get_metrics()
# Available metrics:
# - reward/total_score: Final combined score
# - reward/fenice_score: FENICE factual consistency score
# - reward/rule_score: Rule-based score
# - reward/fenice_weight: Applied FENICE weight
# - reward/rule_weight: Applied rule weight
# - reward/fenice_num_claims: Number of claims extracted
# - reward/fenice_enabled: Whether FENICE is active
# + All existing rule-based metrics
```

### W&B Integration

Metrics are automatically logged to Weights & Biases when available:

```python
integrator = create_reward_integrator(wandb_logger=wandb_logger)
score = integrator.compute_reward(source, summary, step=training_step)
# Metrics automatically logged to W&B
```

### Detailed Logging

Enable detailed logging for debugging:

```python
result = system.evaluate(source, summary, log_details=True)
# Logs:
# - FENICE claim extraction details
# - NLI scoring results
# - Rule evaluation details
# - Combined scoring calculation
```

## Fallback Behavior

The system is designed to be robust and gracefully handle missing dependencies:

1. **Missing Transformers**: FENICE falls back to word overlap scoring
2. **Model Loading Failure**: FENICE disables itself, system continues with rules only
3. **Empty/Invalid Input**: Returns appropriate fallback scores
4. **Configuration Errors**: Uses default configurations with warnings

## Performance Considerations

### Model Loading

- Models are loaded lazily on first use
- Models are cached after loading
- Loading can be disabled via configuration

### Batch Processing

- FENICE supports batch evaluation for efficiency
- Configurable batch sizes (default: 8)
- Memory usage controlled via max_length parameter

### GPU Usage

- Automatically detects and uses GPU when available
- Falls back to CPU if GPU unavailable
- Configurable memory utilization

## Testing

### Running Tests

```bash
# Test FENICE integration
python tests/test_fenice.py

# Test existing functionality
python tests/test_rewards.py

# Run demonstration
python scripts/demo_fenice.py
```

### Test Coverage

- FENICE scorer functionality
- Combined reward system
- Integration with existing components
- Error handling and fallbacks
- Configuration options

## Troubleshooting

### Common Issues

1. **"No module named 'transformers'"**
   - Expected when transformers not installed
   - System falls back gracefully
   - Install transformers for full FENICE functionality

2. **Model loading failures**
   - Check internet connection for model downloads
   - Verify model names in configuration
   - Check available disk space

3. **Performance issues**
   - Reduce batch_size in configuration
   - Reduce max_length parameter
   - Disable FENICE if GPU memory limited

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('rlvr_summary.rewards').setLevel(logging.DEBUG)
```

## Future Improvements

Potential enhancements for future versions:

1. **Custom FENICE Models**: Support for fine-tuned FENICE models
2. **Claim-level Metrics**: Per-claim factual consistency scores
3. **Dynamic Weighting**: Adaptive weights based on content type
4. **Caching**: Cache claim extraction and NLI results
5. **Multi-GPU Support**: Distributed FENICE evaluation

## Migration Guide

### From Rule-only System

Existing code continues to work unchanged. To enable FENICE:

```python
# Before (rule-only)
integrator = create_reward_integrator()

# After (FENICE + rules)
integrator = create_reward_integrator(use_combined_system=True)
```

### Configuration Migration

Existing rule configurations remain compatible. Add FENICE configuration:

```yaml
# Existing rule config preserved
use_combined_system: true  # Add this
fenice_weight: 0.7         # Add this
rule_weight: 0.3           # Add this
```

## References

- [FENICE Paper/Model](https://huggingface.co/Babelscape/FENICE)
- [VERL Framework Documentation](https://github.com/volcengine/verl)
- [Weights & Biases Integration](https://wandb.ai/)