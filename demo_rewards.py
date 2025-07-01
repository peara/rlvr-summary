#!/usr/bin/env python3
"""Demo script for the rule-based reward system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rlvr_summary.rewards import create_default_rule_bundle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """Demonstrate the rule-based reward system."""
    print("🎯 Rule-Based Reward System Demo")
    print("=" * 50)
    
    # Create the default reward system
    print("\n1. Creating rule bundle system...")
    system = create_default_rule_bundle()
    
    # Show rule information
    print("\n2. Configured rules:")
    rule_info = system.get_rule_info()
    for rule_name, info in rule_info.items():
        print(f"   • {rule_name}: weight={info['weight']:.2f}, threshold={info['threshold']:.2f}")
    
    print("\n3. Testing with example summaries...")
    
    # Test cases
    test_cases = [
        {
            "name": "Good Summary",
            "source": "Microsoft Corporation, founded by Bill Gates in 1975, reported revenue of 168 billion dollars in 2021. The company employs over 180,000 people worldwide and has grown by 15% in the last year.",
            "summary": "Microsoft, founded by Bill Gates in 1975, reported 168 billion dollars revenue in 2021 with 15% growth.",
        },
        {
            "name": "Poor Length Summary",
            "source": "Microsoft Corporation, founded by Bill Gates in 1975, reported revenue of 168 billion dollars in 2021. The company employs over 180,000 people worldwide and has grown by 15% in the last year.",
            "summary": "Microsoft made money.",  # Too short
        },
        {
            "name": "Number Mismatch Summary",
            "source": "Microsoft Corporation reported revenue of 168 billion dollars in 2021 with 15% growth.",
            "summary": "Microsoft reported revenue of 200 billion dollars in 2021 with 20% growth.",  # Wrong numbers
        },
        {
            "name": "Profanity Summary",
            "source": "The company had a challenging year with some difficult decisions.",
            "summary": "The company had a damn bad year with some stupid decisions.",  # Contains profanity
        },
        {
            "name": "No Entity Overlap",
            "source": "Apple Inc. reported strong iPhone sales in Q4 2023.",
            "summary": "The technology company had good smartphone revenue in the fourth quarter.",  # No entity overlap
        },
    ]
    
    # Evaluate each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        print(f"Source: {test_case['source'][:80]}...")
        print(f"Summary: {test_case['summary']}")
        
        # Evaluate
        result = system.evaluate(
            test_case['source'], 
            test_case['summary'], 
            log_details=False
        )
        
        print(f"\nResults:")
        print(f"  • Total Score: {result.total_score:.3f}")
        print(f"  • Pass Rate: {result.pass_rate:.3f} ({sum(result.rule_passed.values())}/{len(result.rule_passed)} rules passed)")
        
        print("  • Rule Breakdown:")
        for rule_name, score in result.rule_scores.items():
            passed = "✓" if result.rule_passed[rule_name] else "✗"
            print(f"    - {rule_name}: {score:.3f} {passed}")
        
        # Show specific issues
        issues = []
        if result.rule_scores.get("length_constraint", 1.0) < 0.7:
            details = result.rule_details["length_constraint"]
            issues.append(f"Length issue: {details['word_count']} words (optimal: {details['optimal_range']})")
        
        if result.rule_scores.get("number_consistency", 1.0) < 0.7:
            details = result.rule_details["number_consistency"]
            if details.get("mismatches"):
                issues.append(f"Number mismatches: {details['mismatches']}")
        
        if result.rule_scores.get("profanity_penalty", 1.0) < 1.0:
            details = result.rule_details["profanity_penalty"]
            if details.get("profanity_found"):
                issues.append(f"Profanity detected: {details['profanity_found']}")
        
        if result.rule_scores.get("entity_overlap", 1.0) < 0.5:
            details = result.rule_details["entity_overlap"]
            issues.append(f"Low entity overlap: {details['jaccard_score']:.3f}")
        
        if issues:
            print("  • Issues found:")
            for issue in issues:
                print(f"    - {issue}")
    
    print("\n4. Batch evaluation demo...")
    
    # Batch evaluation
    sources = [case["source"] for case in test_cases]
    summaries = [case["summary"] for case in test_cases]
    
    batch_results = system.evaluate_batch(sources, summaries, log_details=True)
    
    # Calculate statistics
    avg_score = sum(r.total_score for r in batch_results) / len(batch_results)
    avg_pass_rate = sum(r.pass_rate for r in batch_results) / len(batch_results)
    
    print(f"\nBatch Statistics:")
    print(f"  • Average Total Score: {avg_score:.3f}")
    print(f"  • Average Pass Rate: {avg_pass_rate:.3f}")
    
    print("\n5. Configuration demo...")
    
    # Show how to update weights
    print("Updating rule weights...")
    new_weights = {
        "length_constraint": 0.4,
        "entity_overlap": 0.3,
    }
    system.update_rule_weights(new_weights)
    
    # Re-evaluate first test case with new weights
    result_new = system.evaluate(test_cases[0]["source"], test_cases[0]["summary"])
    print(f"Score with updated weights: {result_new.total_score:.3f}")
    
    print("\n🎉 Demo completed!")
    print("\nThe rule-based reward system is ready for integration with RL training loops.")
    print("Key features demonstrated:")
    print("  • Length constraint evaluation")
    print("  • Entity overlap scoring")
    print("  • Number consistency checking")
    print("  • Profanity detection")
    print("  • Configurable rule weights")
    print("  • Detailed metrics and logging")
    print("  • Batch processing capabilities")


if __name__ == "__main__":
    main()