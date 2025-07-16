#!/usr/bin/env python3
"""
Demonstration script for FENICE factual consistency scorer integration.

This script shows how the combined FENICE + rule-based reward system works
and demonstrates the factual consistency improvements.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.rewards import (
    create_combined_reward_system,
    create_default_rule_bundle,
    FENICEScorer,
)


def demonstrate_fenice_integration():
    """Demonstrate FENICE integration with example summaries."""
    
    print("🚀 FENICE Factual Consistency Scorer Integration Demo")
    print("=" * 60)
    
    # Example article and summaries
    article = """
    Microsoft Corporation announced its quarterly earnings today, reporting revenue of $65.4 billion, 
    which represents a 16% increase from the same quarter last year. The company's cloud computing 
    division, Azure, saw particularly strong growth with 31% year-over-year increase. CEO Satya Nadella 
    stated that the company now employs over 221,000 people worldwide. Microsoft's stock price rose 
    3.2% in after-hours trading following the announcement.
    """
    
    # Different quality summaries to test
    summaries = {
        "Factually Accurate": 
            "Microsoft reported $65.4 billion in quarterly revenue, up 16% year-over-year. "
            "Azure cloud services grew 31% and the company employs over 221,000 people.",
        
        "Factually Inaccurate": 
            "Microsoft reported $75.2 billion in quarterly revenue, up 25% year-over-year. "
            "Azure cloud services grew 45% and the company employs over 300,000 people.",
        
        "Partially Accurate": 
            "Microsoft announced strong quarterly earnings with significant growth. "
            "Their cloud division Azure performed well and the company continues to grow.",
        
        "Very Short": 
            "Microsoft announced earnings.",
        
        "Too Long": 
            "Microsoft Corporation, the technology giant based in Redmond, Washington, "
            "announced its quarterly financial results today in a comprehensive report that "
            "detailed various aspects of their business performance across multiple segments "
            "including productivity software, cloud computing, gaming, and enterprise solutions, "
            "showing strong performance across the board with particularly notable growth in "
            "their cloud computing division."
    }
    
    print("\n📊 Evaluating Different Summary Types")
    print("-" * 60)
    
    # Create different reward systems for comparison
    print("\n1. Setting up reward systems...")
    
    # Combined system (FENICE + Rules)
    combined_system = create_combined_reward_system(
        fenice_weight=0.7,
        rule_weight=0.3,
        fenice_enabled=True  # Will fall back gracefully if transformers not available
    )
    
    # Rule-only system for comparison
    rule_only_system = create_default_rule_bundle()
    
    print("   ✓ Combined FENICE + Rules system created")
    print("   ✓ Rule-only system created for comparison")
    
    # Evaluate each summary
    print("\n2. Evaluating summaries...")
    print(f"{'Summary Type':<20} {'Combined':<10} {'FENICE':<10} {'Rules':<10} {'Rule-Only':<10}")
    print("-" * 60)
    
    for summary_type, summary_text in summaries.items():
        # Combined system evaluation
        combined_result = combined_system.evaluate(article, summary_text)
        
        # Rule-only system evaluation  
        rule_result = rule_only_system.evaluate(article, summary_text)
        
        print(f"{summary_type:<20} "
              f"{combined_result.total_score:<10.3f} "
              f"{combined_result.fenice_score:<10.3f} "
              f"{combined_result.rule_score:<10.3f} "
              f"{rule_result.total_score:<10.3f}")
    
    print("\n3. Detailed analysis of best summary...")
    best_summary = summaries["Factually Accurate"]
    detailed_result = combined_system.evaluate(article, best_summary, log_details=True)
    
    print(f"\nDetailed Results for 'Factually Accurate' Summary:")
    print(f"Total Score: {detailed_result.total_score:.3f}")
    print(f"FENICE Score: {detailed_result.fenice_score:.3f} (weight: {detailed_result.fenice_weight:.1f})")
    print(f"Rule Score: {detailed_result.rule_score:.3f} (weight: {detailed_result.rule_weight:.1f})")
    print(f"Passed: {detailed_result.passed}")
    
    # Show FENICE details if available
    if detailed_result.fenice_details:
        fenice_details = detailed_result.fenice_details
        print(f"\nFENICE Details:")
        print(f"  Enabled: {fenice_details.get('enabled', False)}")
        if fenice_details.get('enabled'):
            print(f"  Claims extracted: {fenice_details.get('num_claims', 0)}")
            print(f"  Average claim score: {fenice_details.get('avg_score', 0):.3f}")
        else:
            print(f"  Note: FENICE using fallback mode (transformers not available)")
    
    # Show rule details
    rule_result = detailed_result.rule_result
    print(f"\nRule-based Details:")
    print(f"  Rule pass rate: {rule_result.pass_rate:.3f}")
    print(f"  Individual rule scores:")
    for rule_name, score in rule_result.rule_scores.items():
        passed = rule_result.rule_passed[rule_name]
        print(f"    {rule_name}: {score:.3f} ({'✓' if passed else '✗'})")
    
    print("\n4. Metrics for training integration...")
    metrics = detailed_result.get_metrics()
    print(f"Available metrics for W&B logging:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Demo completed! Key observations:")
    print("   • Factually accurate summaries receive higher FENICE scores")
    print("   • Combined system balances factual accuracy with other quality metrics")
    print("   • System gracefully falls back when transformers models unavailable")
    print("   • Rich metrics available for training and evaluation")
    print("   • Configurable weights allow tuning for different use cases")


def test_configuration_options():
    """Test different configuration options."""
    
    print("\n🔧 Testing Configuration Options")
    print("=" * 60)
    
    article = "Microsoft reported $65.4 billion revenue with 16% growth."
    summary = "Microsoft announced $65.4 billion in revenue, up 16%."
    
    configurations = [
        {"name": "Default (70% FENICE, 30% Rules)", "fenice_weight": 0.7, "rule_weight": 0.3},
        {"name": "Balanced (50% FENICE, 50% Rules)", "fenice_weight": 0.5, "rule_weight": 0.5},
        {"name": "Rule-focused (30% FENICE, 70% Rules)", "fenice_weight": 0.3, "rule_weight": 0.7},
        {"name": "FENICE-only (100% FENICE)", "fenice_weight": 1.0, "rule_weight": 0.0},
    ]
    
    print(f"{'Configuration':<35} {'Total':<8} {'FENICE':<8} {'Rules':<8}")
    print("-" * 60)
    
    for config in configurations:
        system = create_combined_reward_system(
            fenice_weight=config["fenice_weight"],
            rule_weight=config["rule_weight"],
            fenice_enabled=True
        )
        
        result = system.evaluate(article, summary)
        
        print(f"{config['name']:<35} "
              f"{result.total_score:<8.3f} "
              f"{result.fenice_score:<8.3f} "
              f"{result.rule_score:<8.3f}")
    
    print("\n✅ Configuration testing completed!")


if __name__ == "__main__":
    try:
        demonstrate_fenice_integration()
        test_configuration_options()
        
        print(f"\n🎉 FENICE integration demonstration completed successfully!")
        print(f"Ready for training with combined FENICE + rule-based rewards!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)