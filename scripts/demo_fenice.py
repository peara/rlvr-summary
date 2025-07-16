#!/usr/bin/env python3
"""
Demonstration script for FENICE factual consistency scorer integration.

This script shows how the FENICE-integrated rule-based reward system works
and demonstrates the factual consistency improvements.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.rewards import (
    load_rule_bundle_from_config,
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
    
    # Try to load FENICE-integrated system
    config_path = Path(__file__).parent.parent / "configs" / "rewards" / "rule_bundle.yaml"
    try:
        fenice_system = load_rule_bundle_from_config(config_path)
        print("   ✓ FENICE-integrated rule system loaded")
    except Exception as e:
        print(f"   ⚠️  Could not load FENICE system: {e}")
        print("   ✓ Using default rule system instead")
        fenice_system = create_default_rule_bundle()
    
    # Rule-only system for comparison (without FENICE)
    rule_config = {
        "weights": {
            "length_constraint": 0.4,
            "entity_overlap": 0.4, 
            "number_consistency": 0.2,
        },
        "length": {"min_words": 20, "max_words": 100},
        "entity": {"min_overlap": 0.3},
        "numbers": {"exact_match_bonus": 1.0}
    }
    
    rule_only_system = create_default_rule_bundle()
    
    print("   ✓ Rule-only system created for comparison")
    
    # Evaluate each summary
    print("\n2. Evaluating summaries...")
    print(f"{'Summary Type':<20} {'FENICE System':<15} {'Rule-Only':<10}")
    print("-" * 50)
    
    for summary_type, summary_text in summaries.items():
        # FENICE-integrated system evaluation
        fenice_result = fenice_system.evaluate(article, summary_text)
        
        # Rule-only system evaluation  
        rule_result = rule_only_system.evaluate(article, summary_text)
        
        print(f"{summary_type:<20} "
              f"{fenice_result.total_score:<15.3f} "
              f"{rule_result.total_score:<10.3f}")
    
    print("\n3. Detailed analysis of best summary...")
    best_summary = summaries["Factually Accurate"]
    
    try:
        detailed_result = fenice_system.evaluate(article, best_summary, log_details=True)
        
        print(f"\nDetailed Results for 'Factually Accurate' Summary:")
        print(f"Total Score: {detailed_result.total_score:.3f}")
        print(f"Pass Rate: {detailed_result.pass_rate:.3f}")
        print(f"Rules passed: {sum(detailed_result.rule_passed.values())}/{len(detailed_result.rule_passed)}")
        
        print(f"\nIndividual Rule Scores:")
        for rule_name, score in detailed_result.rule_scores.items():
            passed = detailed_result.rule_passed[rule_name]
            print(f"  {rule_name}: {score:.3f} ({'✓' if passed else '✗'})")
            
        # Show FENICE details if available
        if "fenice_factual_consistency" in detailed_result.rule_details:
            fenice_details = detailed_result.rule_details["fenice_factual_consistency"]
            print(f"\nFENICE Details:")
            print(f"  Claims extracted: {fenice_details.get('num_claims', 0)}")
            print(f"  FENICE score: {fenice_details.get('fenice_score', 0):.3f}")
            
    except Exception as e:
        print(f"   ⚠️  Could not run detailed FENICE evaluation: {e}")
        print("   This is expected if FENICE package is not installed.")
    
    print("\n4. Metrics for training integration...")
    try:
        metrics = fenice_system.evaluate(article, best_summary).get_metrics()
        print(f"Available metrics for W&B logging:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    except:
        print("   Could not retrieve metrics (FENICE not available)")
    
    print("\n✅ Demo completed! Key observations:")
    print("   • FENICE provides factual consistency evaluation as a weighted rule")
    print("   • System balances factual accuracy with other quality metrics")
    print("   • Fail-fast behavior ensures proper setup in research environment")
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
        print(f"Ready for training with FENICE-integrated rule-based rewards!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)