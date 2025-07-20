# FENICE Optimization Plan for PPO Training

## Problem Statement

FENICE factual consistency scorer is causing significant performance bottlenecks in VERL PPO training:
- Current evaluation time: 12-30 seconds per summary
- Model reloading overhead: 10 seconds per evaluation, has been cached
- PPO constraint: Summaries are generated dynamically, cannot be pre-cached
- Training bottleneck: Reward computation dominates training time

## Current Architecture Analysis

### FENICE Pipeline (3 phases)
1. **Claim Extraction**: T5-based model extracts claims from summary (~40% of compute time)
2. **Coreference Resolution**: FCoref resolves pronouns and entities (~30% of compute time)  
3. **NLI Alignment**: DeBERTa-based alignment scoring (~30% of compute time)

### VERL PPO Integration Points
- Training script: `scripts/train.sh` calls `verl.trainer.main_ppo`
- Reward function: `custom_reward_function.path=./src/rlvr_summary/rewards/verl_reward.py`
- Batch sizes: `train_batch_size=32`, `ppo_mini_batch_size=8`
- Current interface: `compute_score(data_source, solution_str, ground_truth, extra_info)`

## Optimization Strategies

### Phase 1: Document Pre-caching (10x speedup)
**Target**: Pre-compute document-dependent operations during data preparation

**Implementation**:
- Modify `scripts/prepare_data_verl.py` to pre-process documents
- Cache document sentences, coreference clusters, and paragraph structures
- Store in `extra_info.fenice_document_cache` field of VERL data format

**Expected Impact**: 
- Document processing: ~3-5 seconds → 0 seconds (runtime)
- Overall speedup: 2-3x improvement

**Files to modify**:
- `scripts/prepare_data_verl.py`: Add document pre-processing
- `src/rlvr_summary/fenice/FENICE.py`: Use cached document data

### Phase 2: Fast Claim Extraction (20-5000x speedup)
**Target**: Replace T5-based claim extraction with lightweight alternatives

**Approach 2A: Sentence-based Extraction (Ultra-fast)**
- Split summary into sentences using spaCy/NLTK
- Performance: ~0.001 seconds (5000x faster than T5)
- Trade-off: Lower precision, may miss complex claim boundaries

**Approach 2B: Lightweight BERT Classification (Fast)**
- Use DistilBERT or similar for sentence-level claim detection
- Performance: ~0.2 seconds (20x faster than T5)
- Trade-off: Better precision than sentence splitting, still much faster

**Approach 2C: Hybrid Approach (Balanced)**
- Sentence splitting for simple cases
- Lightweight BERT for complex summaries
- Adaptive selection based on summary characteristics

**Files to modify**:
- `src/rlvr_summary/fenice/claim_extractor.py`: Add fast extraction methods
- `src/rlvr_summary/fenice/FENICE.py`: Integrate fast claim extraction
- `configs/rewards/fenice_config.yaml`: Add configuration for extraction method

### Phase 3: Smart Batching (2-4x speedup)
**Target**: Leverage VERL's batch processing capabilities

**Implementation**:
- Enhance `src/rlvr_summary/rewards/verl_reward.py` for batch processing
- Process multiple summaries simultaneously through FENICE pipeline
- Utilize GPU batching for NLI alignment (DeBERTa)

**VERL Integration Points**:
- `ppo_mini_batch_size=8`: Process 8 summaries per batch
- `train_batch_size=32`: Overall training batch size
- Batch reward computation instead of individual calls

**Files to modify**:
- `src/rlvr_summary/rewards/verl_reward.py`: Add batch processing interface
- `src/rlvr_summary/fenice/FENICE.py`: Support batch evaluation
- `src/rlvr_summary/fenice/nli_aligner.py`: GPU batch processing for DeBERTa

### Phase 4: Model Optimization (2-3x speedup)
**Target**: Optimize individual model components

**Approach 4A: Model Quantization**
- Quantize DeBERTa NLI model to INT8/FP16
- Reduce memory usage and increase throughput
- Expected speedup: 2-3x for NLI alignment

**Approach 4B: Model Distillation**
- Create smaller specialized models for each FENICE component
- Train lightweight models on FENICE outputs
- Trade-off: Accuracy vs. speed

**Approach 4C: Efficient Model Loading**
- Use model sharding for large models
- Implement warm-up strategies
- Optimize memory allocation patterns

## Performance Projections

### Combined Optimization Impact
```
Current: 12-30 seconds per evaluation
Phase 1 (Document caching): 8-20 seconds per evaluation (2-3x speedup)
Phase 2A (Sentence extraction): 4-12 seconds per evaluation (additional 2x speedup)
Phase 2B (Lightweight BERT): 5-15 seconds per evaluation (additional 1.5x speedup)
Phase 3 (Batching): 1-4 seconds per evaluation (additional 3-5x speedup)
Phase 4 (Model optimization): 0.5-2 seconds per evaluation (additional 2x speedup)

Total potential speedup: 30-300x improvement
Target: <1 second per evaluation for PPO training feasibility
```

### PPO Training Impact
```
Current training time per epoch: ~8-12 hours (dominated by reward computation)
Optimized training time per epoch: ~1-2 hours (compute-bound on model training)
Training efficiency improvement: 4-6x faster overall training
```

## Issues Ordered by Implementation Difficulty (Easiest → Hardest)

### Issue 1: Document Pre-caching (EASY) ⭐
**Certainty**: 100% will work, no risks
**Effort**: 2-3 hours
**Impact**: 2-3x speedup (document processing: 3-5s → 0s)
**Why Easy**: 
- Simple data preprocessing during `prepare_data_verl.py`
- No model changes required
- No accuracy trade-offs
- Clear implementation path

**Files to modify**: 
- `scripts/prepare_data_verl.py` (add document caching)
- `src/rlvr_summary/fenice/FENICE.py` (use cached data)

### Issue 2: Sentence-based Claim Extraction (EASY) ⭐
**Certainty**: 95% will work, minor accuracy trade-off acceptable
**Effort**: 3-4 hours  
**Impact**: 5000x speedup for claim extraction (4s → 0.001s)
**Why Easy**:
- Use existing spaCy/NLTK sentence splitting
- Simple replacement in claim extraction logic
- Fast to implement and test
- Add option to fall back to T5 if needed

**Files to modify**:
- `src/rlvr_summary/fenice/claim_extractor.py` (add sentence splitting method)
- `src/rlvr_summary/fenice/FENICE.py` (integrate fast extraction)

### Issue 3: Performance Monitoring and Metrics (EASY-MEDIUM) ⭐
**Certainty**: 100% will work, essential for optimization
**Effort**: 4-6 hours
**Impact**: Enables measuring all other optimizations
**Why Easy-Medium**:
- Straightforward timing and memory tracking
- No complex model integration
- Foundation for validating other optimizations

**Files to modify**:
- `src/rlvr_summary/rewards/verl_reward.py` (add timing)
- `src/rlvr_summary/fenice/FENICE.py` (add metrics)
- `src/rlvr_summary/utils/metrics.py` (new file)

### Issue 4: Smart Batching in VERL Integration (MEDIUM) ⚡
**Certainty**: 80% will work, depends on VERL internals
**Effort**: 1-2 days
**Impact**: 2-4x speedup from batch processing
**Why Medium**:
- Need to understand VERL's reward function calling pattern
- Requires modifying reward interface carefully
- GPU memory management complexity
- Testing with actual PPO training required

**Files to modify**:
- `src/rlvr_summary/rewards/verl_reward.py` (batch interface)
- `src/rlvr_summary/fenice/FENICE.py` (batch evaluation)
- `src/rlvr_summary/fenice/nli_aligner.py` (GPU batching)

### Issue 5: Lightweight BERT Claim Extraction (MEDIUM-HARD) ⚡
**Certainty**: 70% will work well, accuracy trade-off unclear
**Effort**: 3-5 days
**Impact**: 20x speedup for claim extraction (4s → 0.2s)
**Why Medium-Hard**:
- Need to find/train suitable lightweight model
- Model evaluation and accuracy validation required
- Integration with existing pipeline
- Potential accuracy regression needs careful testing

**Files to modify**:
- `src/rlvr_summary/fenice/claim_extractor.py` (new BERT-based method)
- `configs/rewards/fenice_config.yaml` (model configuration)
- Model training/selection scripts (new)

### Issue 6: Model Quantization for DeBERTa NLI (MEDIUM-HARD) ⚡
**Certainty**: 60% will work without accuracy loss
**Effort**: 3-7 days
**Impact**: 2-3x speedup for NLI alignment
**Why Medium-Hard**:
- Quantization can be tricky with accuracy preservation
- Need to test INT8/FP16 variants
- Memory optimization complexity
- May require model fine-tuning after quantization

**Files to modify**:
- `src/rlvr_summary/fenice/nli_aligner.py` (quantized model loading)
- `src/rlvr_summary/fenice/model_manager.py` (quantized model support)
- Model quantization scripts (new)

### Issue 7: Hybrid Claim Extraction Strategy (HARD) 🔥
**Certainty**: 50% will work optimally, complex trade-offs
**Effort**: 1-2 weeks
**Impact**: Variable speedup based on summary complexity
**Why Hard**:
- Requires developing classification logic for method selection
- Multiple model integration complexity
- Extensive testing on diverse summary types
- Performance tuning for optimal switching logic

**Files to modify**:
- `src/rlvr_summary/fenice/claim_extractor.py` (hybrid logic)
- `src/rlvr_summary/fenice/FENICE.py` (method selection)
- `configs/rewards/fenice_config.yaml` (hybrid configuration)
- Extensive testing infrastructure

### Issue 8: Model Distillation for FENICE Components (HARD) 🔥
**Certainty**: 40% will maintain accuracy, high risk
**Effort**: 2-4 weeks
**Impact**: 5-10x speedup but uncertain accuracy
**Why Hard**:
- Requires training new models from scratch
- Need large datasets for distillation
- Accuracy validation across multiple domains
- Significant computational resources for training
- Integration with existing pipeline complex

**Files to create/modify**:
- `scripts/train_distilled_models.py` (new training pipeline)
- `src/rlvr_summary/fenice/distilled_models/` (new module)
- Extensive model evaluation infrastructure
- Multiple config files for different model variants

## Implementation Strategy

### Phase 1: Quick Wins (Week 1)
- **Issue 1**: Document Pre-caching
- **Issue 2**: Sentence-based Claim Extraction  
- **Issue 3**: Performance Monitoring
- **Expected Result**: 10-50x speedup, solid foundation

### Phase 2: Medium Effort (Week 2-3)
- **Issue 4**: Smart Batching
- **Issue 5**: Lightweight BERT (if Issue 2 accuracy insufficient)
- **Expected Result**: Additional 2-5x speedup

### Phase 3: Advanced Optimization (Month 2+)
- **Issue 6**: Model Quantization
- **Issue 7**: Hybrid Strategy (if needed)
- **Issue 8**: Model Distillation (research project)

### Risk Mitigation Strategy
1. **Start with certain wins** (Issues 1-3)
2. **Validate each optimization independently**
3. **Maintain fallback to original implementation**
4. **Measure actual performance gains before proceeding**
5. **Stop when target performance achieved** (<1s per evaluation)

## Technical Considerations

### VERL Compatibility
- Must maintain `compute_score()` interface signature
- Leverage `extra_info` parameter for caching
- Respect VERL's batching and data flow patterns

### Accuracy Trade-offs
- Sentence-based extraction: ~10-15% accuracy reduction
- Lightweight BERT: ~5% accuracy reduction
- Document caching: No accuracy impact
- Batching: No accuracy impact

### Resource Requirements
- GPU memory: Batch processing increases memory usage
- Storage: Document cache increases data size by ~20-30%
- CPU: Lightweight models reduce CPU requirements

## Next Steps

1. **Validation**: Test individual approaches on representative dataset
2. **Benchmarking**: Measure actual performance improvements
3. **Integration**: Ensure compatibility with VERL PPO training loop
4. **Monitoring**: Add performance metrics and accuracy tracking

## Risk Mitigation

### Accuracy Validation
- A/B testing against current FENICE implementation
- Correlation analysis for different extraction methods
- Human evaluation on sample summaries

### Performance Monitoring
- Add timing metrics to reward computation
- Track memory usage during batch processing
- Monitor training convergence with optimized rewards

### Fallback Strategy
- Configurable extraction methods (T5 vs. lightweight)
- Progressive optimization (implement phases incrementally)
- Rollback capability if accuracy degrades significantly
