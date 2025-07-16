#!/usr/bin/env python3
"""
Demonstration of the simplified FENICE integration using weight-based configuration.

This demo shows how FENICE is now integrated as a weighted rule in the unified 
rule-based system, eliminating the need for complex combination logic.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from rlvr_summary.rewards.rule_bundle import RuleBundleRewardSystem
from rlvr_summary.rewards.verl_reward import configure_reward_system, get_reward_system_info
import yaml


def demo_weight_configuration():
    """Demonstrate different weight configurations for FENICE integration."""
    
    print("=" * 60)
    print("SIMPLIFIED FENICE INTEGRATION DEMO")
    print("=" * 60)
    
    # Sample texts for demonstration
    source_text = """
    OpenAI announced a major breakthrough in artificial intelligence research.
    The company reported that their new model achieved 92% accuracy on complex
    reasoning tasks, a significant improvement from the previous 78% benchmark.
    The research team, led by Dr. Sarah Chen, spent 18 months developing 
    this innovative approach.
    """
    
    summary_text = """
    OpenAI achieved 92% accuracy on AI reasoning tasks, improving from 78%.
    Dr. Sarah Chen's team worked 18 months on this breakthrough.
    """
    
    # Test different configurations
    configs = [
        ("rule_bundle.yaml", "Moderate FENICE emphasis (35% weight)"),
        ("combined_fenice.yaml", "High FENICE emphasis (50% weight)")
    ]
    
    for config_file, description in configs:
        print(f"\n{'-' * 50}")
        print(f"Testing: {description}")
        print(f"Config: {config_file}")
        print(f"{'-' * 50}")
        
        # Load and display configuration
        config_path = f"configs/rewards/{config_file}"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\nRule weights:")
        for rule_name, weight in config['weights'].items():
            indicator = " ★" if rule_name == "fenice_factual_consistency" else ""
            print(f"  {rule_name}: {weight:.1%}{indicator}")
        
        # Create system
        try:
            system = RuleBundleRewardSystem(config)
            print(f"\n✓ System created with {len(system.rules)} rules")
            
            # Note: We don't actually evaluate because FENICE package may not be available
            # In real usage, this would work:
            # result = system.evaluate(source_text, summary_text)
            # print(f"Total score: {result.total_score:.3f}")
            
            print("✓ Ready for evaluation (FENICE will load on first use)")
            
        except Exception as e:
            print(f"✗ Error: {e}")


def demo_verl_integration():
    """Demonstrate VERL interface with simplified configuration."""
    
    print(f"\n{'=' * 60}")
    print("VERL INTEGRATION DEMO")
    print(f"{'=' * 60}")
    
    # Show current system info
    print("\n1. Current system configuration:")
    info = get_reward_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test configuration switching
    print("\n2. Switching configuration:")
    print("   Configuring to use high FENICE emphasis...")
    configure_reward_system('configs/rewards/combined_fenice.yaml')
    
    info = get_reward_system_info()
    print(f"   ✓ Now using: {Path(info['config_path']).name}")
    
    print("\n3. VERL interface ready:")
    print("   ✓ compute_score() function available")
    print("   ✓ No extra_info complexity needed")
    print("   ✓ Weights configured via YAML files")


def show_weight_examples():
    """Show different weight configuration examples."""
    
    print(f"\n{'=' * 60}")
    print("WEIGHT CONFIGURATION EXAMPLES")
    print(f"{'=' * 60}")
    
    examples = {
        "Conservative (Traditional focus)": {
            "length_constraint": 0.25,
            "entity_overlap": 0.25,
            "number_consistency": 0.20,
            "profanity_penalty": 0.10,
            "fluency": 0.10,
            "fenice_factual_consistency": 0.10
        },
        "Balanced (Equal emphasis)": {
            "length_constraint": 0.20,
            "entity_overlap": 0.20,
            "number_consistency": 0.15,
            "profanity_penalty": 0.05,
            "fluency": 0.05,
            "fenice_factual_consistency": 0.35
        },
        "Factuality-focused (FENICE emphasis)": {
            "length_constraint": 0.15,
            "entity_overlap": 0.15,
            "number_consistency": 0.10,
            "profanity_penalty": 0.05,
            "fluency": 0.05,
            "fenice_factual_consistency": 0.50
        },
        "Research mode (FENICE only)": {
            "length_constraint": 0.0,
            "entity_overlap": 0.0,
            "number_consistency": 0.0,
            "profanity_penalty": 0.0,
            "fluency": 0.0,
            "fenice_factual_consistency": 1.0
        }
    }
    
    for scenario, weights in examples.items():
        print(f"\n{scenario}:")
        total = sum(weights.values())
        for rule, weight in weights.items():
            percentage = weight / total * 100
            if rule == "fenice_factual_consistency":
                print(f"  ★ {rule}: {percentage:.1f}%")
            else:
                print(f"    {rule}: {percentage:.1f}%")
        print(f"    (Total: {total:.1f})")


def main():
    """Run the complete demonstration."""
    
    print("RLVR-Summary: Simplified FENICE Integration")
    print("Shows how FENICE is integrated as a weighted rule")
    
    try:
        demo_weight_configuration()
        demo_verl_integration()
        show_weight_examples()
        
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print("✓ FENICE integrated as weighted rule (not separate system)")
        print("✓ No complex combination logic needed")
        print("✓ Simple YAML configuration for weights")
        print("✓ VERL interface simplified (no extra_info needed)")
        print("✓ All functionality in single RuleBundleRewardSystem")
        print("✓ Easy to adjust FENICE emphasis via weight parameter")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()