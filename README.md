# RLVR Summary Implementation Issues

This repository contains the implementation plan for a tool-augmented summarizer using RL-VR (Reinforcement Learning from Verifier Rewards) as outlined in `plan.md`.

## Quick Start: Creating GitHub Issues

### Automated Issue Creation (Recommended)

Use the provided script to automatically create all GitHub issues:

```bash
# Make sure you have GitHub CLI installed and authenticated
gh auth login

# Run the issue creation script
./create_issues.sh
```

The script will:
- Parse all 21 issues from `quick_issues.md`
- Create appropriate labels automatically
- Create issues with proper titles, descriptions, and labels
- Respect the project structure and phases

### Manual Issue Creation

Two files have been created for manual issue creation:

#### 1. `quick_issues.md` 
**Use this for fast issue creation**
- Contains 21 ready-to-copy issue templates
- Each issue has title, labels, description, tasks, and acceptance criteria
- Simply copy each section and paste into GitHub's "New Issue" form

#### 2. `github_issues.md`
**Use this for detailed planning**
- Comprehensive document with full issue descriptions
- Includes dependencies, timeline estimates, and priority levels
- Contains detailed technical specifications for each task

## Prerequisites

For automated issue creation:
- [GitHub CLI](https://cli.github.com/) installed and authenticated
- Bash shell (Linux/macOS/WSL)
- Write access to the repository

## Project Overview

The project implements a 5-phase roadmap:

- **Phase A**: Baseline RL-VR (no tools) - 4 issues
- **Phase B**: FENICE Reward Integration - 3 issues  
- **Phase C**: Synthetic Traces + SFT - 4 issues
- **Phase D**: Tool-Aware RL-VR - 4 issues
- **Phase E**: Distillation & Deployment - 3 issues
- **Setup/Infrastructure**: 2 issues
- **Documentation**: 2 issues

**Total**: 21 issues

## Key Milestones

| ID | Milestone          | Success Threshold                               |
| -- | ------------------ | ----------------------------------------------- |
| M0 | Rule-only baseline | ≥ 20% rule pass rate                           |
| M1 | +FENICE            | Hallucinations ↓ ≥ 40% vs. M0                  |
| M2 | Tool-aware RL      | Combined pass ≥ 60%                            |
| M3 | Distilled model    | Latency ≤ 100ms & factuality within 2pp of M2  |

## Recommended Issue Labels

Create these labels in your GitHub repository:

**Phase Labels:**
- `phase-0`, `phase-a`, `phase-b`, `phase-c`, `phase-d`, `phase-e`

**Component Labels:**
- `setup`, `infrastructure`, `training`, `rl`, `sft`, `evaluation`
- `data-pipeline`, `synthetic-traces`, `rewards`, `fenice`, `tools`
- `sandbox`, `deployment`, `production`, `documentation`

**Priority Labels:**
- `high-priority`, `medium-priority`, `low-priority`

## Issue Dependencies

**Critical Path:**
1. Setup issues (1-2) must be completed first
2. Phase A (3-5) → Phase B (6-8) → Phase C (9-12) → Phase D (13-16) → Phase E (17-19)
3. Documentation issues (20-21) can run in parallel

## Estimated Timeline

- **Total Duration**: 18-26 weeks (4.5-6.5 months)
- **Setup Phase**: 2-3 weeks
- **Each Implementation Phase**: 2-5 weeks
- **Documentation**: Ongoing, 1 week final

## Getting Started

### Training Scripts

The project includes automated training scripts that handle configuration validation and dependency checking:

#### Quick Training

For immediate training with validation:

```bash
# Use default configuration
python scripts/train_simple.py --dry-run  # Validate configuration only
python scripts/train_simple.py            # Start training

# Use custom configuration
python scripts/train_simple.py --config configs/config.yaml --experiment my-experiment

# Enable debugging
python scripts/train_simple.py --log-level DEBUG --experiment debug-run
```

#### Advanced Training (requires full dependencies)

```bash
# Install full dependencies first
pip install torch transformers trl wandb
pip install -e .

# Use the enhanced training script
python scripts/train.py --config configs --experiment full-training
```

#### Training Script Features

The training scripts provide:
- **Configuration Validation**: Checks for required settings, valid parameter ranges, and missing files
- **Dependency Checking**: Verifies that required packages (torch, transformers, trl) are installed
- **Environment Detection**: Automatically detects CUDA availability and Python version
- **Directory Setup**: Creates necessary directories for data, outputs, logs, and checkpoints
- **Clear Error Messages**: Provides meaningful feedback for configuration issues
- **Configuration Summary**: Shows a detailed overview of training settings before starting
- **Dry Run Mode**: Validate configuration without starting training
- **Flexible Configuration**: Supports both file paths and directory paths for configs

### Project Setup

1. Create the GitHub labels listed above
2. Copy issues from `quick_issues.md` into GitHub
3. Assign initial priorities and team members
4. Install dependencies: `pip install -r requirements.txt`
5. Configure W&B: `wandb login`
6. Validate setup: `python scripts/train_simple.py --dry-run`
7. Start with Setup issues (infrastructure and data pipeline)
8. Follow the phase sequence for implementation

## Cost Estimates

Based on the plan:
- **Data Generation**: ~$1.5k-2k + 2 GPU-days for 50k traces
- **Training**: Multiple GPU-weeks for full training cycles
- **Total Budget**: Plan for $5k-10k in compute costs

## Technical Stack

- **Core**: PyTorch ≥2.3, HuggingFace Transformers ≥4.42, TRL
- **Infrastructure**: vLLM, W&B, spaCy, rank-bm25
- **Models**: Llama-3-1B (drafting), GPT-4 (fact-checking), DeBERTa-v3-large (FENICE)

For detailed technical specifications, see `plan.md` and `github_issues.md`.