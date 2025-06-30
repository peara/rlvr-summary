# Issue Creation Order and Dependencies

Create GitHub issues in this order to respect dependencies and enable parallel work.

## Wave 1: Foundation (Create these first)
**No dependencies - can start immediately**

1. **Setup project infrastructure and core dependencies**
   - Priority: High
   - Labels: setup, infrastructure, phase-0, high-priority
   - Estimated: 2-3 weeks

2. **Implement core data processing pipeline infrastructure** 
   - Priority: High
   - Labels: infrastructure, data-pipeline, phase-0, high-priority
   - Estimated: 2-3 weeks
   - Can start parallel to issue 1

## Wave 2: Phase A Implementation
**Dependencies: Wave 1 complete**

3. **Implement hard rule-bundle reward system for baseline RL-VR**
   - Priority: High
   - Dependencies: Issues 1, 2
   - Labels: phase-a, rewards, rl, high-priority
   - Estimated: 1-2 weeks

4. **Implement vanilla PPO/GRPO training loop for baseline model**
   - Priority: High  
   - Dependencies: Issue 3
   - Labels: phase-a, rl, training, high-priority
   - Estimated: 2-3 weeks

5. **Create evaluation dashboard and metrics tracking for Phase A**
   - Priority: Medium
   - Dependencies: Issue 4
   - Labels: phase-a, evaluation, monitoring, medium-priority
   - Estimated: 1 week

## Wave 3: Phase B Implementation  
**Dependencies: Wave 2 complete (specifically Issue 5)**

6. **Integrate distilled FENICE scorer into reward system**
   - Priority: High
   - Dependencies: Issue 5
   - Labels: phase-b, fenice, rewards, high-priority
   - Estimated: 2-3 weeks

7. **Add curriculum scheduling for FENICE threshold**
   - Priority: Medium
   - Dependencies: Issue 6
   - Labels: phase-b, curriculum, training, medium-priority
   - Estimated: 1 week

8. **Train model with FENICE rewards and evaluate hallucination reduction**
   - Priority: High
   - Dependencies: Issue 7
   - Labels: phase-b, training, evaluation, high-priority
   - Estimated: 1-2 weeks

## Wave 4: Phase C Implementation (Can start parallel to Wave 3)
**Dependencies: Issue 2 (data pipeline) only**

9. **Build four-step synthetic trace generation pipeline**
   - Priority: High
   - Dependencies: Issue 2
   - Labels: phase-c, data-pipeline, synthetic-traces, high-priority
   - Estimated: 3-4 weeks
   - Can start after Issue 2 completes

10. **Add quality control filters for synthetic trace validation**
    - Priority: High
    - Dependencies: Issue 9
    - Labels: phase-c, quality-control, filtering, high-priority
    - Estimated: 1-2 weeks

11. **Generate 5k pilot batch and scale to 50k synthetic traces**
    - Priority: High
    - Dependencies: Issue 10
    - Labels: phase-c, data-generation, scaling, high-priority
    - Estimated: 2-3 weeks

12. **Implement SFT for tool syntax learning on synthetic traces**
    - Priority: High
    - Dependencies: Issue 11
    - Labels: phase-c, sft, training, high-priority
    - Estimated: 1-2 weeks

## Wave 5: Phase D Implementation
**Dependencies: Issue 12 (SFT) complete**

13. **Build tool execution sandbox for RL-VR training**
    - Priority: High
    - Dependencies: Issue 12
    - Labels: phase-d, sandbox, tools, high-priority
    - Estimated: 2-3 weeks

14. **Implement tool budget penalty in reward system**
    - Priority: Medium
    - Dependencies: Issue 13
    - Labels: phase-d, rewards, budget, medium-priority
    - Estimated: 1 week

15. **Implement tool-aware PPO training with execution sandbox**
    - Priority: High
    - Dependencies: Issue 14
    - Labels: phase-d, rl, training, tools, high-priority
    - Estimated: 2-3 weeks

16. **Conduct ablation studies to measure tool contribution**
    - Priority: Medium
    - Dependencies: Issue 15
    - Labels: phase-d, evaluation, ablation, medium-priority
    - Estimated: 1-2 weeks

## Wave 6: Phase E Implementation
**Dependencies: Issue 16 (ablation studies) complete**

17. **Distill 7B/13B policy into 3B fast model**
    - Priority: High
    - Dependencies: Issue 16
    - Labels: phase-e, distillation, optimization, high-priority
    - Estimated: 2-3 weeks

18. **Benchmark distilled model latency and factuality parity**
    - Priority: High
    - Dependencies: Issue 17
    - Labels: phase-e, benchmarking, performance, high-priority
    - Estimated: 1-2 weeks

19. **Package and deploy production inference endpoint**
    - Priority: High
    - Dependencies: Issue 18
    - Labels: phase-e, deployment, production, high-priority
    - Estimated: 2-3 weeks

## Wave 7: Documentation (Can run parallel to implementation)
**Dependencies: Various, but can start early**

20. **Implement milestone tracking and go/no-go decision framework**
    - Priority: Medium
    - Dependencies: Can start after any phase issues begin
    - Labels: documentation, milestones, tracking, medium-priority
    - Estimated: 1 week

21. **Create complete project documentation and setup guides**
    - Priority: Low
    - Dependencies: Best to wait until most implementation is complete
    - Labels: documentation, setup, guides, low-priority
    - Estimated: 1 week

## Parallel Work Opportunities

**High Parallelism:**
- Issues 1 & 2 (setup) can run fully parallel
- Issues 9-12 (Phase C) can start as soon as Issue 2 is done, parallel to Phase B
- Issue 20 (milestone tracking) can start early and run parallel to implementation
- Multiple team members can work on different phases simultaneously

**Resource Planning:**
- **Week 1-3**: Issues 1, 2 (2 developers)
- **Week 4-7**: Issues 3, 4, 5, 9 (3-4 developers)  
- **Week 8-12**: Issues 6, 7, 8, 10, 11 (3-4 developers)
- **Week 13-18**: Issues 12, 13, 14, 15 (2-3 developers)
- **Week 19-24**: Issues 16, 17, 18, 19 (2-3 developers)
- **Ongoing**: Issues 20, 21 (1 developer)

## Critical Path Analysis

**Longest Critical Path** (determines minimum timeline):
Issues 1 → 3 → 4 → 5 → 6 → 7 → 8 (Phase B completion)

**Parallel Critical Path:**
Issues 2 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19

The project can complete in ~18-24 weeks if both paths are executed in parallel with adequate resources.

## Go/No-Go Decision Points

- **After Issue 5**: Check if M0 milestone (≥20% rule pass rate) is achieved
- **After Issue 8**: Check if M1 milestone (≥40% hallucination reduction) is achieved  
- **After Issue 16**: Check if M2 milestone (≥60% combined pass rate) is achieved
- **After Issue 18**: Check if M3 milestone (≤100ms latency + factuality parity) is achieved

Each decision point should trigger a formal review before proceeding to the next phase.