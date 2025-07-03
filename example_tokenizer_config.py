#!/usr/bin/env python3
"""
Example demonstrating tokenizer configuration usage.

This example shows how to use the new tokenizer configuration feature
to set padding direction and other tokenizer options.
"""

import sys
from pathlib import Path

# Add src to path for running example
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rlvr_summary.models.base import ModelLoader


def demo_tokenizer_configuration():
    """Demonstrate tokenizer configuration options."""
    print("🔧 Tokenizer Configuration Demo")
    print("=" * 50)
    
    # Example 1: Default configuration (backward compatible)
    print("\n1. Default Configuration (no tokenizer section):")
    config_default = {
        "model_name": "gpt2",
        "torch_dtype": "float32"
    }
    
    loader_default = ModelLoader(config_default)
    tokenizer_config = loader_default.get_tokenizer_config()
    print(f"   Padding side: {tokenizer_config['padding_side']}")
    print(f"   Trust remote code: {tokenizer_config['trust_remote_code']}")
    
    # Example 2: Left padding for causal models
    print("\n2. Left Padding Configuration (for causal models like Llama):")
    config_left = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "tokenizer": {
            "padding_side": "left",
            "trust_remote_code": True
        }
    }
    
    loader_left = ModelLoader(config_left)
    tokenizer_config = loader_left.get_tokenizer_config()
    print(f"   Padding side: {tokenizer_config['padding_side']}")
    print(f"   Trust remote code: {tokenizer_config['trust_remote_code']}")
    
    # Example 3: Right padding for standard models
    print("\n3. Right Padding Configuration (for standard models):")
    config_right = {
        "model_name": "distilgpt2",
        "tokenizer": {
            "padding_side": "right"
        }
    }
    
    loader_right = ModelLoader(config_right)
    tokenizer_config = loader_right.get_tokenizer_config()
    print(f"   Padding side: {tokenizer_config['padding_side']}")
    print(f"   Trust remote code: {tokenizer_config['trust_remote_code']} (default)")
    
    # Example 4: No padding side change
    print("\n4. No Padding Side Change (preserve tokenizer default):")
    config_none = {
        "model_name": "gpt2",
        "tokenizer": {
            "padding_side": None
        }
    }
    
    loader_none = ModelLoader(config_none)
    tokenizer_config = loader_none.get_tokenizer_config()
    print(f"   Padding side: {tokenizer_config['padding_side']} (no change)")
    
    print("\n📋 Configuration Options Summary:")
    print("   - padding_side: 'left', 'right', or None")
    print("   - trust_remote_code: true/false (default: true)")
    print("\n💡 Use Cases:")
    print("   - Left padding: Causal models (Llama, GPT) for better attention")
    print("   - Right padding: Standard models (BERT-style, DistilGPT2)")
    print("   - None: Keep tokenizer's default padding behavior")


def demo_yaml_configuration():
    """Demonstrate loading from YAML configuration files."""
    print("\n\n📄 YAML Configuration Demo")
    print("=" * 50)
    
    import yaml
    
    # Show example YAML configs
    yaml_configs = [
        "configs/model/llama_3_1b.yaml",
        "configs/model/distilgpt2.yaml"
    ]
    
    for config_path in yaml_configs:
        print(f"\n📁 {config_path}:")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            loader = ModelLoader(config)
            tokenizer_config = loader.get_tokenizer_config()
            
            print(f"   Model: {config.get('model_name', 'N/A')}")
            print(f"   Padding: {tokenizer_config['padding_side']}")
            print(f"   Use case: {'Causal models' if tokenizer_config['padding_side'] == 'left' else 'Standard models'}")
            
        except FileNotFoundError:
            print(f"   ⚠️  Config file not found: {config_path}")
        except Exception as e:
            print(f"   ❌ Error loading config: {e}")


if __name__ == "__main__":
    demo_tokenizer_configuration()
    demo_yaml_configuration()
    
    print("\n✨ Tokenizer configuration demo completed!")
    print("\nFor more information, see the updated model configuration files in configs/model/")