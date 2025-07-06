# RLVR Summary

A reinforcement learning framework for text summarization using VERL (Versatile Efficient Reinforcement Learning) with functional rewards.

## Overview

This project implements PPO-based reinforcement learning for text summarization using VERL, featuring:
- Efficient PPO training through VERL integration
- Custom rule-based reward functions for summarization quality
- CNN/DailyMail dataset support
- Clean, minimal codebase focused on VERL workflow

## Features

- **VERL Integration**: Modern PPO implementation with efficient training
- **Functional Rewards**: Rule-based rewards for length, relevance, and quality
- **CNN/DailyMail Support**: Built-in data pipeline for the standard summarization dataset
- **Minimal Architecture**: Clean, focused codebase without legacy dependencies

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rlvr-summary

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Data Preparation

```bash
# Prepare CNN/DailyMail dataset for VERL training
python scripts/prepare_data_verl.py
```

### Training

```bash
# Start PPO training with VERL, overriding default parameters
./scripts/train.sh
```

## Configuration

The project uses three main configuration files:

- **`configs/verl_ppo_config.yaml`**: VERL PPO training parameters, model settings, and training hyperparameters. Cloned from the VERL repository for reference.
- **`configs/rewards/rule_bundle.yaml`**: Reward function weights and parameters
- **`configs/data/cnn_dailymail.yaml`**: Dataset configuration and preprocessing settings

### Model Configuration

Models are specified via the `path` field in the VERL config:
```yaml
rollout:
  model:
    path: "distilgpt2"  # or any HuggingFace model path
```

## Project Structure

```
rlvr-summary/
├── configs/                      # Configuration files
│   ├── verl_ppo_config.yaml     # Main VERL training config
│   ├── rewards/rule_bundle.yaml  # Reward function parameters
│   └── data/cnn_dailymail.yaml  # Dataset configuration
├── scripts/                      # Training and data scripts
│   ├── train_verl.py            # Main VERL training script
│   └── prepare_data_verl.py     # Data preparation for VERL
├── src/rlvr_summary/            # Core package
│   ├── rewards/                 # Reward implementations
│   │   ├── verl_reward.py      # Main VERL reward function
│   │   ├── rule_bundle.py      # Rule-based reward components
│   │   └── rules.py            # Individual reward rules
│   ├── data/                   # Data processing utilities
│   ├── evaluation/             # Evaluation metrics (ROUGE, etc.)
│   └── utils/                  # Utility functions
└── tests/                      # Test suite
```

## Reward System

The reward system (`src/rlvr_summary/rewards/verl_reward.py`) evaluates summaries using:

- **Length Reward**: Encourages appropriate summary length
- **Coverage Reward**: Measures content coverage of the source
- **Redundancy Penalty**: Penalizes repetitive content
- **Fluency Reward**: Evaluates summary readability

Rewards are configurable via `configs/rewards/rule_bundle.yaml`.

## Development

### Running Tests

```bash
pytest tests/
```

### Adding Custom Rewards

1. Implement reward logic in `src/rlvr_summary/rewards/rules.py`
2. Update the reward bundle in `src/rlvr_summary/rewards/rule_bundle.py`
3. Configure weights in `configs/rewards/rule_bundle.yaml`

### Using Different Models

Update the `path` field in `configs/verl_ppo_config.yaml`:
```yaml
rollout:
  model:
    path: "microsoft/DialoGPT-medium"  # Example alternative
```

## Dependencies

- **VERL**: For PPO training implementation
- **Transformers**: For model loading and tokenization
- **Datasets**: For CNN/DailyMail data loading
- **ROUGE**: For evaluation metrics
- **PyTorch**: For deep learning framework

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

[Add your license information here]