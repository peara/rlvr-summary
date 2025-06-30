# Quick Issue Creation Template

Copy and paste each section below to create GitHub issues. Each issue is separated by `---`.

## Setup Phase Issues

### Issue: Setup project infrastructure and core dependencies
**Labels:** setup, infrastructure, phase-0, high-priority

Set up the foundational infrastructure for the RLVR Summary project according to the Library & Tooling Stack specifications.

**Tasks:**
- Initialize Python project structure with proper packaging
- Set up requirements.txt with core dependencies (PyTorch ≥2.3, HuggingFace Transformers ≥4.42, TRL, vLLM, etc.)
- Set up Weights & Biases integration for experiment tracking
- Create basic project structure (src/, tests/, configs/)
- Set up development environment and configuration files

**Acceptance Criteria:**
- All dependencies install without conflicts
- Basic project structure is established
- W&B integration is functional

---

### Issue: Implement core data processing pipeline infrastructure
**Labels:** infrastructure, data-pipeline, phase-0, high-priority

Build the foundational data processing infrastructure needed for the four-step synthetic trace generation pipeline.

**Tasks:**
- Create data loading utilities for CNN-DailyMail dataset
- Implement basic text preprocessing with spaCy
- Set up data validation and quality control framework
- Create utilities for handling structured JSON annotations
- Implement batch processing capabilities

**Acceptance Criteria:**
- Can load and preprocess CNN-DailyMail data
- Data validation framework is functional
- Pipeline can handle structured annotations

---

## Phase A Issues

### Issue: Implement hard rule-bundle reward system for baseline RL-VR
**Labels:** phase-a, rewards, rl, high-priority

Implement the rule-based reward system that will serve as the foundation for the baseline RL-VR training loop.

**Tasks:**
- Implement length constraint rewards, entity overlap scoring, number consistency checking
- Implement profanity detection and penalty
- Create configurable rule weight system
- Add rule evaluation metrics and logging

**Acceptance Criteria:**
- All rule components are implemented and tested
- System can evaluate summaries and return scores
- Target: Support evaluation of rule-pass rate metrics

---

### Issue: Implement vanilla PPO/GRPO training loop for baseline model
**Labels:** phase-a, rl, training, high-priority

Build the core reinforcement learning training loop using HuggingFace TRL for PPO/GRPO without tool integration.

**Tasks:**
- Set up base model loading and configuration
- Implement PPO training loop using HuggingFace TRL
- Integrate rule-based reward system
- Add training metrics, logging, checkpointing
- Create evaluation pipeline for ROUGE metrics

**Acceptance Criteria:**
- Training loop runs without errors
- Model checkpoints are saved correctly
- Training metrics are logged to W&B

---

### Issue: Create evaluation dashboard and metrics tracking for Phase A
**Labels:** phase-a, evaluation, monitoring, medium-priority

Establish comprehensive evaluation and monitoring infrastructure for tracking model performance during baseline training.

**Tasks:**
- Create ROUGE evaluation pipeline
- Implement rule-pass rate calculation
- Set up W&B dashboard for training metrics
- Create automated evaluation scripts and comparison tools

**Acceptance Criteria:**
- Dashboard shows real-time training progress
- Rule-pass rate is accurately calculated
- Target: Achieve ≥20% rule-pass rate milestone (M0)

---

## Phase B Issues

### Issue: Integrate distilled FENICE scorer into reward system
**Labels:** phase-b, fenice, rewards, high-priority

Implement the FENICE factual consistency scorer and integrate it with the existing rule-based reward system.

**Tasks:**
- Set up DeBERTa-v3-large for NLI scoring
- Implement distilled T5-base for claim extraction
- Create FENICE score calculation pipeline
- Implement combined reward: R = 0.7 × FENICE + 0.3 × Rules
- Add FENICE-specific logging and metrics

**Acceptance Criteria:**
- FENICE scorer produces consistent results
- Combined reward system is functional
- Validation shows improvement in factual consistency

---

### Issue: Add curriculum scheduling for FENICE threshold
**Labels:** phase-b, curriculum, training, medium-priority

Implement adaptive threshold scheduling for FENICE scoring to improve training stability and convergence.

**Tasks:**
- Implement threshold scheduler (start at 0.50, tighten to 0.65)
- Add pass-rate monitoring for threshold updates
- Create curriculum progression logic
- Add curriculum state logging and visualization

**Acceptance Criteria:**
- Threshold scheduling works automatically
- Pass-rate triggers function correctly
- Training stability is maintained during threshold updates

---

### Issue: Train model with FENICE rewards and evaluate hallucination reduction
**Labels:** phase-b, training, evaluation, high-priority

Conduct full training run with FENICE-enhanced rewards and measure hallucination reduction compared to baseline.

**Tasks:**
- Run full training with FENICE + rules reward
- Implement hallucination detection evaluation
- Compare against Phase A baseline
- Create hallucination reduction metrics

**Acceptance Criteria:**
- Training completes successfully with FENICE rewards
- Hallucination reduction ≥40% vs M0 (milestone M1)
- Results are reproducible and documented

---

## Phase C Issues

### Issue: Build four-step synthetic trace generation pipeline
**Labels:** phase-c, data-pipeline, synthetic-traces, high-priority

Implement the complete four-step pipeline for generating synthetic tool-augmented training traces.

**Tasks:**
- Step 1: Implement faulty summary generation with Llama-3-1B and adversarial prompts
- Step 2: Integrate GPT-4 fact-checking and tool call injection (<search>, <delete>)
- Step 3: Implement context editing and continuation generation
- Step 4: Add positive "search-only" trace generation

**Acceptance Criteria:**
- All four steps execute correctly
- Pipeline generates valid tool-augmented traces
- Tool call syntax is correct and consistent

---

### Issue: Add quality control filters for synthetic trace validation
**Labels:** phase-c, quality-control, filtering, high-priority

Implement the quality control filters to ensure high-quality synthetic traces according to the specifications.

**Tasks:**
- Implement result relevance filter (BM25 < 0.3 cosine threshold)
- Add FENICE gain filter (keep only if FENICE ↑ ≥ 0.05)
- Implement token budget filter (≤200 tokens for results)
- Add diversity filter using MinHash dedup (Jaccard ≥ 0.9)

**Acceptance Criteria:**
- All filters function correctly and efficiently
- Noise level ≤5% on spot-check validation
- Quality improvement is measurable

---

### Issue: Generate 5k pilot batch and scale to 50k synthetic traces
**Labels:** phase-c, data-generation, scaling, high-priority

Execute the data generation pipeline to create the full synthetic trace dataset for supervised fine-tuning.

**Tasks:**
- Generate and validate 5k pilot batch
- Analyze pilot batch quality and refine filters
- Generate full 50k trace dataset
- Implement distributed processing for scale

**Acceptance Criteria:**
- 5k pilot batch meets quality requirements
- Pipeline scales efficiently to 50k traces
- Cost estimates align with budget (≈$1.5k-2k + 2 GPU-days)

---

### Issue: Implement SFT for tool syntax learning on synthetic traces
**Labels:** phase-c, sft, training, high-priority

Supervised fine-tune the base model on synthetic traces to learn proper tool syntax and timing.

**Tasks:**
- Prepare SFT dataset from synthetic traces
- Implement supervised fine-tuning pipeline for 2-3 epochs
- Add tool syntax validation during training
- Implement syntax correctness evaluation

**Acceptance Criteria:**
- SFT training completes successfully
- Model learns correct <search>/<delete> syntax
- ≥95% of sampled generations have syntactically correct tool calls

---

## Phase D Issues

### Issue: Build tool execution sandbox for RL-VR training
**Labels:** phase-d, sandbox, tools, high-priority

Create the execution sandbox that enables real tool usage during RL-VR training rollouts.

**Tasks:**
- Implement lightweight Python subprocess pool (ReTool style)
- Add 0.3s timeout for tool execution
- Implement <search> tool for snippet retrieval and <delete> tool for context pruning
- Add context re-tokenization after deletion

**Acceptance Criteria:**
- Sandbox executes tools safely and efficiently
- <search> retrieves relevant snippets
- <delete> correctly prunes context

---

### Issue: Implement tool budget penalty in reward system
**Labels:** phase-d, rewards, budget, medium-priority

Add tool usage penalties to prevent tool spam and encourage strategic tool use.

**Tasks:**
- Implement tool usage counting
- Add budget penalty: –0.05 × (#search + #delete)
- Add hard cap of 5 deletes per document
- Create tool usage analytics

**Acceptance Criteria:**
- Tool budget penalty reduces spam behavior
- Hard caps prevent excessive tool usage
- Analytics provide insights into tool usage patterns

---

### Issue: Implement tool-aware PPO training with execution sandbox
**Labels:** phase-d, rl, training, tools, high-priority

Integrate the execution sandbox with PPO training to enable tool-aware reinforcement learning.

**Tasks:**
- Modify PPO loop to support tool execution
- Integrate FENICE + rule + budget rewards
- Implement context updates after tool usage
- Create tool-specific training metrics

**Acceptance Criteria:**
- PPO training works with live tool execution
- Tool usage improves model performance
- Training remains stable with tool integration

---

### Issue: Conduct ablation studies to measure tool contribution
**Labels:** phase-d, evaluation, ablation, medium-priority

Run comprehensive ablation studies to measure the individual contribution of each tool and budget penalty.

**Tasks:**
- Train variants: without delete, without search, without budget penalty
- Compare all variants against full system
- Measure performance on combined pass rate
- Create comprehensive comparison report

**Acceptance Criteria:**
- All ablation variants complete training
- Combined pass rate ≥60% for full system (milestone M2)
- Clear measurement of individual tool contributions

---

## Phase E Issues

### Issue: Distill 7B/13B policy into 3B fast model
**Labels:** phase-e, distillation, optimization, high-priority

Implement knowledge distillation to create a faster, smaller model while maintaining performance.

**Tasks:**
- Implement regression on thought-hidden traces
- Set up teacher-student distillation pipeline
- Configure 3B student model architecture
- Add distillation training monitoring

**Acceptance Criteria:**
- Distillation pipeline runs successfully
- 3B model maintains reasonable performance
- Knowledge transfer is effective

---

### Issue: Benchmark distilled model latency and factuality parity
**Labels:** phase-e, benchmarking, performance, high-priority

Comprehensive benchmarking of the distilled model to ensure it meets deployment requirements.

**Tasks:**
- Implement latency benchmarking pipeline
- Create factuality evaluation on hidden test sets
- Test inference optimization techniques
- Create performance comparison reports

**Acceptance Criteria:**
- Latency ≤100ms per summary (milestone M3)
- Factuality within 2 percentage points of M2
- Performance is consistent across test sets

---

### Issue: Package and deploy production inference endpoint
**Labels:** phase-e, deployment, production, high-priority

Create production-ready inference endpoint with continuous evaluation capabilities.

**Tasks:**
- Package model for production deployment
- Implement inference endpoint API
- Set up continuous evaluation with rule/FENICE checker
- Add monitoring and alerting

**Acceptance Criteria:**
- Inference endpoint is production-ready
- Continuous evaluation is functional
- Monitoring provides comprehensive visibility

---

## Documentation Issues

### Issue: Implement milestone tracking and go/no-go decision framework
**Labels:** documentation, milestones, tracking, medium-priority

Create comprehensive milestone tracking system and decision framework for project progression.

**Tasks:**
- Implement automated milestone tracking
- Create go/no-go decision framework
- Set up milestone reporting dashboard
- Create project timeline visualization

**Acceptance Criteria:**
- All milestones (M0-M3) are trackable
- Go/no-go decisions are data-driven
- Progress is visible to stakeholders

---

### Issue: Create complete project documentation and setup guides
**Labels:** documentation, setup, guides, low-priority

Create comprehensive documentation for the entire project including setup, usage, and maintenance guides.

**Tasks:**
- Create detailed setup and installation guide
- Document all configuration options
- Create user guide for each phase
- Add troubleshooting guide and contribution guidelines

**Acceptance Criteria:**
- Documentation is complete and accurate
- Setup guide enables new users to start quickly
- All APIs and interfaces are documented

---

## Milestones Summary for Reference

| ID | Milestone          | Success Threshold                               |
| -- | ------------------ | ----------------------------------------------- |
| M0 | Rule-only baseline | ≥ 20% rule pass rate                           |
| M1 | +FENICE            | Hallucinations ↓ ≥ 40% vs. M0                  |
| M2 | Tool-aware RL      | Combined pass ≥ 60%                            |
| M3 | Distilled model    | Latency ≤ 100ms & factuality within 2pp of M2  |