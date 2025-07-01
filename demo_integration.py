#!/usr/bin/env python3
"""Integration demo for the rule-based reward system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rlvr_summary.rewards import (
    create_reward_integrator,
    create_reward_function,
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """Demonstrate reward system integration capabilities."""
    print("🔗 Reward System Integration Demo")
    print("=" * 50)
    
    print("\n1. Simple reward function creation...")
    
    # Create simple reward function
    reward_fn = create_reward_function()
    
    source = "Microsoft Corporation reported revenue of 168 billion dollars in 2021 with 15% growth."
    summary = "Microsoft reported 168 billion revenue in 2021 with 15% growth."
    
    score = reward_fn(source, summary)
    print(f"Simple reward function score: {score:.3f}")
    
    print("\n2. Advanced integrator with statistics tracking...")
    
    # Create integrator for advanced features
    integrator = create_reward_integrator()
    
    # Test data simulating training batches
    training_data = [
        {
            "sources": [
                "Apple Inc. reported strong iPhone sales with 50 billion revenue.",
                "Google announced new AI features with 20% performance improvement.",
                "Tesla delivered 500,000 electric vehicles with 30% growth.",
            ],
            "summaries": [
                "Apple had strong iPhone sales with 50 billion revenue.",
                "Google launched AI features with 20% performance boost.",
                "Tesla delivered 500,000 EVs with 30% growth.",
            ]
        },
        {
            "sources": [
                "Amazon Web Services grew by 25% with 80 billion in revenue.",
                "Netflix added 15 million subscribers with 95% retention rate.",
                "Meta platforms reached 3 billion users with 18% growth.",
            ],
            "summaries": [
                "AWS grew 25% with 80 billion revenue.",
                "Netflix gained 15 million subscribers with 95% retention.",
                "Meta reached 3 billion users with 18% growth.",
            ]
        },
    ]
    
    # Simulate training batches
    for batch_idx, batch in enumerate(training_data):
        print(f"\nProcessing training batch {batch_idx + 1}...")
        
        scores = integrator.compute_reward_batch(
            batch["sources"], 
            batch["summaries"],
            log_details=True,
            step=batch_idx,
        )
        
        print(f"Batch scores: {[f'{s:.3f}' for s in scores]}")
        print(f"Batch average: {sum(scores) / len(scores):.3f}")
    
    print("\n3. Cumulative statistics...")
    
    stats = integrator.get_cumulative_statistics()
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Average total score: {stats['average_scores']['total_score']:.3f}")
    print(f"Average pass rate: {stats['average_pass_rates']['total_pass_rate']:.3f}")
    
    print("\nPer-rule statistics:")
    for rule_name in stats['average_scores']:
        if rule_name != 'total_score':
            avg_score = stats['average_scores'][rule_name]
            avg_pass_rate = stats['average_pass_rates'][rule_name]
            print(f"  {rule_name}: score={avg_score:.3f}, pass_rate={avg_pass_rate:.3f}")
    
    print("\n4. Milestone evaluation...")
    
    # Evaluate M0 milestone (≥20% rule-pass rate)
    milestone = integrator.evaluate_milestone_criteria(target_pass_rate=0.2)
    print(f"M0 Milestone Status: {milestone['message']}")
    print(f"Current pass rate: {milestone['current_pass_rate']:.3f}")
    print(f"Target pass rate: {milestone['target_pass_rate']:.3f}")
    print(f"Evaluations: {milestone['evaluations']}")
    
    # Also check higher thresholds
    for threshold in [0.4, 0.6, 0.8]:
        milestone_high = integrator.evaluate_milestone_criteria(target_pass_rate=threshold)
        status = "✅" if milestone_high['milestone_met'] else "❌"
        print(f"{threshold*100:.0f}% threshold: {status} ({milestone_high['current_pass_rate']:.3f})")
    
    print("\n5. Rule weight adjustment demo...")
    
    # Show current rule configuration
    rule_info = integrator.get_rule_info()
    print("Current rule weights:")
    for rule_name, info in rule_info.items():
        print(f"  {rule_name}: {info['weight']:.2f}")
    
    # Update weights to emphasize different aspects
    print("\nUpdating weights to emphasize length and entity overlap...")
    new_weights = {
        "length_constraint": 0.4,
        "entity_overlap": 0.4,
        "number_consistency": 0.1,
        "profanity_penalty": 0.05,
        "fluency": 0.05,
    }
    integrator.update_rule_weights(new_weights)
    
    # Test with new weights
    test_source = "John Smith, CEO of Microsoft, announced 100 million revenue."
    test_summary = "Microsoft CEO John Smith reported 100 million in revenue."
    
    new_score = integrator.compute_reward(test_source, test_summary, log_details=True)
    print(f"Score with updated weights: {new_score:.3f}")
    
    print("\n6. Error handling and edge cases...")
    
    # Test edge cases
    edge_cases = [
        ("", "Empty source"),
        ("Source text", ""),  # Empty summary
        ("Short", "x"),  # Very short texts
        ("This is profanity test", "This is a damn bad summary"),  # Profanity
        ("Numbers: 42, 3.14", "Wrong numbers: 50, 2.71"),  # Number mismatches
    ]
    
    print("Testing edge cases:")
    for source, summary in edge_cases:
        try:
            score = integrator.compute_reward(source, summary)
            print(f"  '{summary[:20]}...': {score:.3f}")
        except Exception as e:
            print(f"  '{summary[:20]}...': ERROR - {e}")
    
    print("\n7. Performance simulation...")
    
    # Reset statistics for clean measurement
    integrator.reset_statistics()
    
    # Simulate larger batch for performance testing
    large_sources = ["Test source text for performance evaluation."] * 50
    large_summaries = ["Test summary for performance."] * 50
    
    import time
    start_time = time.time()
    large_scores = integrator.compute_reward_batch(large_sources, large_summaries)
    end_time = time.time()
    
    processing_time = end_time - start_time
    per_sample_time = processing_time / len(large_sources) * 1000  # ms
    
    print(f"Processed {len(large_sources)} samples in {processing_time:.3f}s")
    print(f"Average time per sample: {per_sample_time:.2f}ms")
    print(f"Throughput: {len(large_sources) / processing_time:.1f} samples/second")
    
    # Final statistics
    final_stats = integrator.get_cumulative_statistics()
    final_milestone = integrator.evaluate_milestone_criteria(target_pass_rate=0.2)
    
    print(f"\nFinal Statistics:")
    print(f"Total evaluations: {final_stats['total_evaluations']}")
    print(f"Final milestone status: {final_milestone['message']}")
    
    print("\n🎉 Integration demo completed!")
    print("\nKey integration features demonstrated:")
    print("  • Simple reward function interface")
    print("  • Advanced integrator with statistics tracking")
    print("  • Milestone evaluation for training progress")
    print("  • Dynamic weight adjustment")
    print("  • Batch processing capabilities")
    print("  • Error handling and edge cases")
    print("  • Performance measurement")
    print("\nThe reward system is ready for integration with RL training loops!")


if __name__ == "__main__":
    main()