# GitHub Issues for RLVR Summary Implementation

This document contains all the GitHub issues that should be created to implement the roadmap outlined in `plan.md`.

## Setup and Infrastructure Issues

### Issue 1: Setup Project Infrastructure and Dependencies
**Title:** Setup project infrastructure and core dependencies  
**Labels:** setup, infrastructure, phase-0  
**Priority:** High  
**Assignee:** TBD  

**Description:**
Set up the foundational infrastructure for the RLVR Summary project according to the Library & Tooling Stack specifications.

**Tasks:**
- [ ] Initialize Python project structure with proper packaging
- [ ] Set up requirements.txt with core dependencies:
  - PyTorch ≥ 2.3
  - HuggingFace Transformers ≥ 4.42
  - HuggingFace TRL (PPO/GRPO)
  - vLLM for fast inference
  - bitsandbytes for quantization
  - spaCy 3 for NLP utilities
  - rank-bm25 for retrieval
  - DeBERTa-v3-large for FENICE NLI
  - datasets + pandas for data handling
  - Hydra for config management
- [ ] Set up Weights & Biases (wandb) integration for experiment tracking
- [ ] Create basic project structure (src/, tests/, configs/, etc.)
- [ ] Set up development environment (pre-commit hooks, linting, etc.)
- [ ] Create initial configuration files using Hydra

**Acceptance Criteria:**
- All dependencies install without conflicts
- Basic project structure is established
- Configuration management is working
- W&B integration is functional

---

### Issue 2: Implement Data Pipeline Infrastructure
**Title:** Implement core data processing pipeline infrastructure  
**Labels:** infrastructure, data-pipeline, phase-0  
**Priority:** High  
**Dependencies:** Issue 1  

**Description:**
Build the foundational data processing infrastructure needed for the four-step synthetic trace generation pipeline.

**Tasks:**
- [ ] Create data loading utilities for CNN-DailyMail dataset
- [ ] Implement basic text preprocessing with spaCy
- [ ] Set up data validation and quality control framework
- [ ] Create utilities for handling structured JSON annotations
- [ ] Implement basic logging and monitoring for data pipeline
- [ ] Create data storage and retrieval mechanisms
- [ ] Set up batch processing capabilities

**Acceptance Criteria:**
- Can load and preprocess CNN-DailyMail data
- Data validation framework is functional
- Pipeline can handle structured annotations
- Logging provides clear visibility into processing steps

---

## Phase A: Baseline RL-VR (No Tools)

### Issue 3: Implement Rule-Based Reward System
**Title:** Implement hard rule-bundle reward system for baseline RL-VR  
**Labels:** phase-a, rewards, rl  
**Priority:** High  
**Dependencies:** Issue 1, Issue 2  

**Description:**
Implement the rule-based reward system that will serve as the foundation for the baseline RL-VR training loop.

**Tasks:**
- [ ] Implement length constraint rewards
- [ ] Implement entity overlap scoring
- [ ] Implement number consistency checking
- [ ] Implement profanity detection and penalty
- [ ] Create configurable rule weight system
- [ ] Add rule evaluation metrics and logging
- [ ] Create unit tests for each rule component
- [ ] Implement rule aggregation logic

**Acceptance Criteria:**
- All rule components are implemented and tested
- Rule weights are configurable
- System can evaluate summaries and return scores
- Target: Support evaluation of rule-pass rate metrics

---

### Issue 4: Implement Vanilla PPO/GRPO Training Loop
**Title:** Implement vanilla PPO/GRPO training loop for baseline model  
**Labels:** phase-a, rl, training  
**Priority:** High  
**Dependencies:** Issue 3  

**Description:**
Build the core reinforcement learning training loop using HuggingFace TRL for PPO/GRPO without tool integration.

**Tasks:**
- [ ] Set up base model loading and configuration
- [ ] Implement PPO training loop using HuggingFace TRL
- [ ] Integrate rule-based reward system
- [ ] Add training metrics and logging
- [ ] Implement checkpointing and model saving
- [ ] Create evaluation pipeline for ROUGE metrics
- [ ] Set up distributed training capabilities
- [ ] Add gradient accumulation and memory optimization

**Acceptance Criteria:**
- Training loop runs without errors
- Model checkpoints are saved correctly
- Training metrics are logged to W&B
- Can train on small datasets successfully

---

### Issue 5: Setup Evaluation Dashboard and Metrics
**Title:** Create evaluation dashboard and metrics tracking for Phase A  
**Labels:** phase-a, evaluation, monitoring  
**Priority:** Medium  
**Dependencies:** Issue 4  

**Description:**
Establish comprehensive evaluation and monitoring infrastructure for tracking model performance during baseline training.

**Tasks:**
- [ ] Create ROUGE evaluation pipeline
- [ ] Implement rule-pass rate calculation
- [ ] Set up W&B dashboard for training metrics
- [ ] Create automated evaluation scripts
- [ ] Implement comparison tools for model versions
- [ ] Add statistical significance testing
- [ ] Create visualization tools for metrics
- [ ] Set up automated reporting

**Acceptance Criteria:**
- Dashboard shows real-time training progress
- Rule-pass rate is accurately calculated
- ROUGE scores are computed and tracked
- Target: Achieve ≥20% rule-pass rate milestone (M0)

---

## Phase B: FENICE Reward Integration

### Issue 6: Implement FENICE Scorer Integration
**Title:** Integrate distilled FENICE scorer into reward system  
**Labels:** phase-b, fenice, rewards  
**Priority:** High  
**Dependencies:** Issue 5  

**Description:**
Implement the FENICE factual consistency scorer and integrate it with the existing rule-based reward system.

**Tasks:**
- [ ] Set up DeBERTa-v3-large for NLI scoring
- [ ] Implement distilled T5-base for claim extraction
- [ ] Create FENICE score calculation pipeline
- [ ] Implement combined reward: R = 0.7 × FENICE + 0.3 × Rules
- [ ] Add FENICE-specific logging and metrics
- [ ] Create FENICE score validation tools
- [ ] Implement batch processing for efficiency
- [ ] Add error handling and fallback mechanisms

**Acceptance Criteria:**
- FENICE scorer produces consistent results
- Combined reward system is functional
- Performance is acceptable for training loop integration
- Validation shows improvement in factual consistency

---

### Issue 7: Implement Curriculum Scheduling for FENICE
**Title:** Add curriculum scheduling for FENICE threshold  
**Labels:** phase-b, curriculum, training  
**Priority:** Medium  
**Dependencies:** Issue 6  

**Description:**
Implement adaptive threshold scheduling for FENICE scoring to improve training stability and convergence.

**Tasks:**
- [ ] Implement threshold scheduler (start at 0.50, tighten to 0.65)
- [ ] Add pass-rate monitoring for threshold updates
- [ ] Create curriculum progression logic
- [ ] Implement threshold update triggers
- [ ] Add curriculum state logging
- [ ] Create manual override capabilities
- [ ] Implement curriculum evaluation metrics
- [ ] Add curriculum visualization tools

**Acceptance Criteria:**
- Threshold scheduling works automatically
- Pass-rate triggers function correctly
- Curriculum progression is logged and visible
- Training stability is maintained during threshold updates

---

### Issue 8: Train and Evaluate FENICE-Enhanced Model
**Title:** Train model with FENICE rewards and evaluate hallucination reduction  
**Labels:** phase-b, training, evaluation  
**Priority:** High  
**Dependencies:** Issue 7  

**Description:**
Conduct full training run with FENICE-enhanced rewards and measure hallucination reduction compared to baseline.

**Tasks:**
- [ ] Run full training with FENICE + rules reward
- [ ] Implement hallucination detection evaluation
- [ ] Compare against Phase A baseline
- [ ] Measure training stability and convergence
- [ ] Create hallucination reduction metrics
- [ ] Document performance improvements
- [ ] Analyze failure cases and edge conditions
- [ ] Create model comparison tools

**Acceptance Criteria:**
- Training completes successfully with FENICE rewards
- Hallucination reduction ≥40% vs M0 (milestone M1)
- Training remains stable throughout process
- Results are reproducible and well-documented

---

## Phase C: Synthetic Traces + SFT

### Issue 9: Implement Four-Step Data Pipeline
**Title:** Build four-step synthetic trace generation pipeline  
**Labels:** phase-c, data-pipeline, synthetic-traces  
**Priority:** High  
**Dependencies:** Issue 2  

**Description:**
Implement the complete four-step pipeline for generating synthetic tool-augmented training traces.

**Tasks:**
- [ ] **Step 1:** Implement faulty summary generation with Llama-3-1B
- [ ] **Step 1:** Add adversarial prompt seeding for error diversity
- [ ] **Step 2:** Integrate GPT-4 fact-checking and annotation
- [ ] **Step 2:** Implement structured JSON error detection
- [ ] **Step 2:** Build tool call injection logic (<search>, <delete>)
- [ ] **Step 3:** Implement context editing and continuation generation
- [ ] **Step 4:** Add positive "search-only" trace generation
- [ ] Create end-to-end pipeline orchestration
- [ ] Add progress tracking and resumability

**Acceptance Criteria:**
- All four steps execute correctly
- Pipeline generates valid tool-augmented traces
- Error diversity is achieved in step 1
- Tool call syntax is correct and consistent

---

### Issue 10: Implement Quality Control Filters
**Title:** Add quality control filters for synthetic trace validation  
**Labels:** phase-c, quality-control, filtering  
**Priority:** High  
**Dependencies:** Issue 9  

**Description:**
Implement the quality control filters to ensure high-quality synthetic traces according to the specifications.

**Tasks:**
- [ ] Implement result relevance filter (BM25 < 0.3 cosine threshold)
- [ ] Add FENICE gain filter (keep only if FENICE ↑ ≥ 0.05)
- [ ] Implement token budget filter (≤200 tokens for results)
- [ ] Add diversity filter using MinHash dedup (Jaccard ≥ 0.9)
- [ ] Create filter performance monitoring
- [ ] Implement spot-check validation (target ≤5% noise)
- [ ] Add filter bypass mechanisms for debugging
- [ ] Create filter effectiveness metrics

**Acceptance Criteria:**
- All filters function correctly and efficiently
- Noise level ≤5% on spot-check validation
- Filter performance is monitored and logged
- Quality improvement is measurable

---

### Issue 11: Generate Pilot and Scale to Full Dataset
**Title:** Generate 5k pilot batch and scale to 50k synthetic traces  
**Labels:** phase-c, data-generation, scaling  
**Priority:** High  
**Dependencies:** Issue 10  

**Description:**
Execute the data generation pipeline to create the full synthetic trace dataset for supervised fine-tuning.

**Tasks:**
- [ ] Generate and validate 5k pilot batch
- [ ] Analyze pilot batch quality and refine filters
- [ ] Optimize pipeline for large-scale generation
- [ ] Generate full 50k trace dataset
- [ ] Implement distributed processing for scale
- [ ] Add cost tracking and estimation
- [ ] Create data quality reports
- [ ] Implement dataset versioning and storage

**Acceptance Criteria:**
- 5k pilot batch meets quality requirements
- Pipeline scales efficiently to 50k traces
- Cost estimates align with budget (≈$1.5k-2k + 2 GPU-days)
- Dataset is properly versioned and stored

---

### Issue 12: Supervised Fine-Tuning for Tool Syntax
**Title:** Implement SFT for tool syntax learning on synthetic traces  
**Labels:** phase-c, sft, training  
**Priority:** High  
**Dependencies:** Issue 11  

**Description:**
Supervised fine-tune the base model on synthetic traces to learn proper tool syntax and timing.

**Tasks:**
- [ ] Prepare SFT dataset from synthetic traces
- [ ] Implement supervised fine-tuning pipeline
- [ ] Configure training for 2-3 epochs
- [ ] Add tool syntax validation during training
- [ ] Implement syntax correctness evaluation
- [ ] Create tool timing analysis tools
- [ ] Add training progress monitoring
- [ ] Validate 95% syntactic correctness requirement

**Acceptance Criteria:**
- SFT training completes successfully
- Model learns correct <search>/<delete> syntax
- ≥95% of sampled generations have syntactically correct tool calls
- Tool timing behavior is appropriate

---

## Phase D: Tool-Aware RL-VR

### Issue 13: Implement Execution Sandbox
**Title:** Build tool execution sandbox for RL-VR training  
**Labels:** phase-d, sandbox, tools  
**Priority:** High  
**Dependencies:** Issue 12  

**Description:**
Create the execution sandbox that enables real tool usage during RL-VR training rollouts.

**Tasks:**
- [ ] Implement lightweight Python subprocess pool (ReTool style)
- [ ] Add 0.3s timeout for tool execution
- [ ] Implement <search> tool for snippet retrieval
- [ ] Implement <delete> tool for context pruning
- [ ] Add context re-tokenization after deletion
- [ ] Implement tool result caching
- [ ] Add error handling and recovery
- [ ] Create tool execution monitoring

**Acceptance Criteria:**
- Sandbox executes tools safely and efficiently
- <search> retrieves relevant snippets
- <delete> correctly prunes context
- Timeout mechanisms work reliably

---

### Issue 14: Add Tool Budget Penalty System
**Title:** Implement tool budget penalty in reward system  
**Labels:** phase-d, rewards, budget  
**Priority:** Medium  
**Dependencies:** Issue 13  

**Description:**
Add tool usage penalties to prevent tool spam and encourage strategic tool use.

**Tasks:**
- [ ] Implement tool usage counting
- [ ] Add budget penalty: –0.05 × (#search + #delete)
- [ ] Integrate penalty into combined reward
- [ ] Add hard cap of 5 deletes per document
- [ ] Implement delete loop detection and prevention
- [ ] Create tool usage analytics
- [ ] Add configurable penalty weights
- [ ] Implement tool usage visualization

**Acceptance Criteria:**
- Tool budget penalty reduces spam behavior
- Hard caps prevent excessive tool usage
- Analytics provide insights into tool usage patterns
- Penalty weights are tunable for optimization

---

### Issue 15: Tool-Aware PPO Training Implementation
**Title:** Implement tool-aware PPO training with execution sandbox  
**Labels:** phase-d, rl, training, tools  
**Priority:** High  
**Dependencies:** Issue 14  

**Description:**
Integrate the execution sandbox with PPO training to enable tool-aware reinforcement learning.

**Tasks:**
- [ ] Modify PPO loop to support tool execution
- [ ] Integrate FENICE + rule + budget rewards
- [ ] Implement context updates after tool usage
- [ ] Add tool-aware value function estimation
- [ ] Create tool-specific training metrics
- [ ] Implement training stability monitoring
- [ ] Add tool execution logging during training
- [ ] Create tool-aware gradient computation

**Acceptance Criteria:**
- PPO training works with live tool execution
- Tool usage improves model performance
- Training remains stable with tool integration
- Tool execution doesn't significantly slow training

---

### Issue 16: Ablation Studies for Tool Contribution
**Title:** Conduct ablation studies to measure tool contribution  
**Labels:** phase-d, evaluation, ablation  
**Priority:** Medium  
**Dependencies:** Issue 15  

**Description:**
Run comprehensive ablation studies to measure the individual contribution of each tool and budget penalty.

**Tasks:**
- [ ] Train variant without delete tool
- [ ] Train variant without search tool  
- [ ] Train variant without budget penalty
- [ ] Train variant with tools but no budget penalty
- [ ] Compare all variants against full system
- [ ] Measure performance on combined pass rate
- [ ] Analyze tool usage patterns across variants
- [ ] Create comprehensive comparison report

**Acceptance Criteria:**
- All ablation variants complete training
- Combined pass rate ≥60% for full system (milestone M2)
- Clear measurement of individual tool contributions
- Insights guide future tool development

---

## Phase E: Distillation & Deployment

### Issue 17: Implement Model Distillation Pipeline
**Title:** Distill 7B/13B policy into 3B fast model  
**Labels:** phase-e, distillation, optimization  
**Priority:** High  
**Dependencies:** Issue 16  

**Description:**
Implement knowledge distillation to create a faster, smaller model while maintaining performance.

**Tasks:**
- [ ] Implement regression on thought-hidden traces
- [ ] Set up teacher-student distillation pipeline
- [ ] Configure 3B student model architecture
- [ ] Implement distillation loss functions
- [ ] Add distillation training monitoring
- [ ] Create student model evaluation pipeline
- [ ] Implement iterative distillation improvement
- [ ] Add distillation quality metrics

**Acceptance Criteria:**
- Distillation pipeline runs successfully
- 3B model maintains reasonable performance
- Distillation process is monitored and optimized
- Knowledge transfer is effective

---

### Issue 18: Benchmark Latency and Factuality
**Title:** Benchmark distilled model latency and factuality parity  
**Labels:** phase-e, benchmarking, performance  
**Priority:** High  
**Dependencies:** Issue 17  

**Description:**
Comprehensive benchmarking of the distilled model to ensure it meets deployment requirements.

**Tasks:**
- [ ] Implement latency benchmarking pipeline
- [ ] Create factuality evaluation on hidden test sets
- [ ] Measure performance parity with teacher model
- [ ] Test inference optimization techniques
- [ ] Benchmark on various hardware configurations
- [ ] Create performance comparison reports
- [ ] Validate <100ms/summary latency requirement
- [ ] Ensure factuality within 2pp of M2

**Acceptance Criteria:**
- Latency ≤100ms per summary (milestone M3)
- Factuality within 2 percentage points of M2
- Performance is consistent across test sets
- Benchmarking is reproducible and documented

---

### Issue 19: Package and Deploy Inference Endpoint
**Title:** Package and deploy production inference endpoint  
**Labels:** phase-e, deployment, production  
**Priority:** High  
**Dependencies:** Issue 18  

**Description:**
Create production-ready inference endpoint with continuous evaluation capabilities.

**Tasks:**
- [ ] Package model for production deployment
- [ ] Implement inference endpoint API
- [ ] Set up continuous evaluation with rule/FENICE checker
- [ ] Add monitoring and alerting
- [ ] Implement load balancing and scaling
- [ ] Create deployment documentation
- [ ] Set up automated testing pipeline
- [ ] Implement gradual rollout mechanisms

**Acceptance Criteria:**
- Inference endpoint is production-ready
- Continuous evaluation is functional
- Monitoring provides comprehensive visibility
- Deployment is scalable and maintainable

---

## Documentation and Milestone Tracking

### Issue 20: Create Milestone Tracking and Reporting
**Title:** Implement milestone tracking and go/no-go decision framework  
**Labels:** documentation, milestones, tracking  
**Priority:** Medium  
**Dependencies:** All phase issues  

**Description:**
Create comprehensive milestone tracking system and decision framework for project progression.

**Tasks:**
- [ ] Implement automated milestone tracking
- [ ] Create go/no-go decision framework
- [ ] Set up milestone reporting dashboard
- [ ] Create milestone evaluation criteria
- [ ] Implement milestone-based alerts
- [ ] Create project timeline visualization
- [ ] Add milestone achievement validation
- [ ] Create executive summary reports

**Acceptance Criteria:**
- All milestones (M0-M3) are trackable
- Go/no-go decisions are data-driven
- Progress is visible to stakeholders
- Milestone achievement is validated

---

### Issue 21: Create Comprehensive Project Documentation
**Title:** Create complete project documentation and setup guides  
**Labels:** documentation, setup, guides  
**Priority:** Low  
**Dependencies:** All implementation issues  

**Description:**
Create comprehensive documentation for the entire project including setup, usage, and maintenance guides.

**Tasks:**
- [ ] Create detailed setup and installation guide
- [ ] Document all configuration options
- [ ] Create user guide for each phase
- [ ] Document API endpoints and interfaces
- [ ] Create troubleshooting guide
- [ ] Add contribution guidelines
- [ ] Create architectural documentation
- [ ] Add performance tuning guide

**Acceptance Criteria:**
- Documentation is complete and accurate
- Setup guide enables new users to start quickly
- All APIs and interfaces are documented
- Troubleshooting guide covers common issues

---

## Issue Dependencies Summary

**Setup Phase (Required for all):**
1. Issue 1: Project Infrastructure
2. Issue 2: Data Pipeline Infrastructure

**Phase A Dependencies:**
- Issue 3 → Issue 4 → Issue 5

**Phase B Dependencies:**
- Issue 5 → Issue 6 → Issue 7 → Issue 8

**Phase C Dependencies:**
- Issue 2 → Issue 9 → Issue 10 → Issue 11 → Issue 12

**Phase D Dependencies:**
- Issue 12 → Issue 13 → Issue 14 → Issue 15 → Issue 16

**Phase E Dependencies:**
- Issue 16 → Issue 17 → Issue 18 → Issue 19

**Documentation:**
- Issue 20: Parallel to all phases
- Issue 21: After all implementation

## Recommended Issue Labels

- **phase-0, phase-a, phase-b, phase-c, phase-d, phase-e**: Phase identification
- **setup, infrastructure**: Foundational work
- **training, rl, sft**: Training-related tasks
- **evaluation, benchmarking**: Testing and validation
- **data-pipeline, synthetic-traces**: Data generation
- **rewards, fenice**: Reward system components
- **tools, sandbox**: Tool integration
- **deployment, production**: Production readiness
- **documentation**: Documentation tasks
- **high, medium, low**: Priority levels

## Estimated Timeline

- **Setup Phase**: 2-3 weeks
- **Phase A**: 3-4 weeks
- **Phase B**: 2-3 weeks  
- **Phase C**: 4-5 weeks
- **Phase D**: 3-4 weeks
- **Phase E**: 3-4 weeks
- **Documentation**: Ongoing, 1 week final

**Total Estimated Timeline**: 18-26 weeks (4.5-6.5 months)