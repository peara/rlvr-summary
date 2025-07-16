# FENICE Integration Documentation

## Overview

This document describes the integration of the FENICE Factual Consistency Scorer package into the RLVR-Summary reward system. FENICE provides factual consistency evaluation through claim extraction and Natural Language Inference (NLI) scoring using the official FENICE Python package from Babelscape.

## Architecture

### Rule-Based System with FENICE Integration

FENICE is integrated as a weighted rule within the existing rule-based reward system:

```
R = Σ(weight_i × rule_score_i) for all rules including FENICE
```

**Example Configuration:**
- Length constraint: 0.15 (15%)
- Entity overlap: 0.15 (15%)  
- Number consistency: 0.10 (10%)
- Profanity penalty: 0.05 (5%)
- Fluency: 0.05 (5%)
- FENICE factual consistency: 0.50 (50%)

### Components

1. **FENICE Scorer** (`src/rlvr_summary/rewards/fenice.py`)
   - Uses the official FENICE Python package (`pip install FENICE`)
   - Claim extraction and NLI-based factual consistency scoring
   - Fail-fast behavior for research environments

2. **Rule Bundle System** (`src/rlvr_summary/rewards/rule_bundle.py`)
   - Unified system that includes FENICE as a weighted rule
   - Configurable rule weights and settings
   - Configurable weights and thresholds

3. **Integration Layer** (`src/rlvr_summary/rewards/integration.py`)
   - W&B logging with FENICE metrics
   - Training loop integration

## Installation

Before using FENICE, ensure the package is installed:

```bash
pip install FENICE
```

The FENICE package includes all necessary dependencies for claim extraction and NLI-based factual consistency scoring.

## Usage

### Basic Usage

```python
from rlvr_summary.rewards import load_rule_bundle_from_config

# Load system with FENICE integration
system = load_rule_bundle_from_config("configs/rewards/rule_bundle.yaml")

# Evaluate a summary
result = system.evaluate(source_text, summary_text)
print(f"Total score: {result.total_score:.3f}")
print(f"FENICE score: {result.rule_scores['fenice_factual_consistency']:.3f}")
print(f"Pass rate: {result.pass_rate:.3f}")
```

### Training Integration

The system integrates seamlessly with existing training scripts:

```bash
# Use training script with FENICE configuration
./scripts/train_fenice.sh

# Or use existing script with rule bundle config
./scripts/train_3090.sh
```

### Configuration

Configuration is managed through YAML files that specify rule weights:

```python
# Use specific configuration 
from rlvr_summary.rewards import load_rule_bundle_from_config

system = load_rule_bundle_from_config("configs/rewards/balanced_fenice.yaml")
```

### Configuration Files

Multiple configuration files are available for different scenarios:

- `configs/rewards/rule_bundle.yaml` - Default configuration with FENICE (35% weight)
- `configs/rewards/combined_fenice.yaml` - FENICE-focused (50% weight)
- `configs/rewards/balanced_fenice.yaml` - Balanced approach (35% weight)  
- `configs/rewards/conservative_fenice.yaml` - Conservative FENICE usage (10% weight)
- `configs/rewards/fenice_focused.yaml` - Maximum FENICE emphasis (65% weight)

Example configuration structure:

```yaml
# Rule weights (sum to 1.0)
weights:
  length_constraint: 0.15
  entity_overlap: 0.15
  number_consistency: 0.10
  profanity_penalty: 0.05
  fluency: 0.05
  fenice_factual_consistency: 0.50

# FENICE configuration
fenice:
  threshold: 0.5
  batch_size: 8

# Other rule configurations...
```

## Metrics and Logging

### Training Metrics

The rule bundle system provides rich metrics for tracking:

```python
metrics = result.get_metrics()
# Available metrics:
# - reward/total_score: Final weighted rule combination score
# - reward/fenice_factual_consistency_score: FENICE score
# - reward/fenice_factual_consistency_passed: Whether FENICE threshold met
# - reward/length_constraint_score: Length rule score
# - reward/entity_overlap_score: Entity overlap score
# + All other configured rule metrics
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
# - Weighted scoring calculation
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