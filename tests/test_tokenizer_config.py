"""Tests for tokenizer configuration functionality."""

import sys
import yaml
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlvr_summary.models.base import ModelLoader


class TestTokenizerConfiguration:
    """Test tokenizer configuration in ModelLoader."""
    
    def test_default_tokenizer_config(self):
        """Test default tokenizer configuration."""
        config = {"model_name": "gpt2"}
        loader = ModelLoader(config)
        
        tokenizer_config = loader.get_tokenizer_config()
        
        # Check defaults
        assert tokenizer_config["padding_side"] == "right"
        assert tokenizer_config["trust_remote_code"] == True
    
    def test_custom_tokenizer_config(self):
        """Test custom tokenizer configuration."""
        config = {
            "model_name": "gpt2",
            "tokenizer": {
                "padding_side": "left",
                "trust_remote_code": False
            }
        }
        loader = ModelLoader(config)
        
        tokenizer_config = loader.get_tokenizer_config()
        
        # Check custom values
        assert tokenizer_config["padding_side"] == "left"
        assert tokenizer_config["trust_remote_code"] == False
    
    def test_partial_tokenizer_config(self):
        """Test partial tokenizer configuration with defaults."""
        config = {
            "model_name": "gpt2",
            "tokenizer": {
                "padding_side": "left"
                # trust_remote_code should use default
            }
        }
        loader = ModelLoader(config)
        
        tokenizer_config = loader.get_tokenizer_config()
        
        # Check mixed values
        assert tokenizer_config["padding_side"] == "left"
        assert tokenizer_config["trust_remote_code"] == True  # Default
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing configs."""
        # Old style config without tokenizer section
        old_config = {
            "model_name": "gpt2",
            "trust_remote_code": True,
            "torch_dtype": "float32"
        }
        loader = ModelLoader(old_config)
        
        # Should still work and use defaults
        tokenizer_config = loader.get_tokenizer_config()
        assert tokenizer_config["padding_side"] == "right"
        assert tokenizer_config["trust_remote_code"] == True
        
        # Generation config should still work
        gen_config = loader.get_generation_config()
        assert "max_new_tokens" in gen_config
        assert "temperature" in gen_config
    
    def test_padding_side_validation(self):
        """Test different padding side options."""
        valid_sides = ["left", "right"]
        
        for side in valid_sides:
            config = {
                "model_name": "gpt2",
                "tokenizer": {"padding_side": side}
            }
            loader = ModelLoader(config)
            tokenizer_config = loader.get_tokenizer_config()
            assert tokenizer_config["padding_side"] == side
    
    def test_none_padding_side(self):
        """Test None padding side (no change to tokenizer)."""
        config = {
            "model_name": "gpt2",
            "tokenizer": {"padding_side": None}
        }
        loader = ModelLoader(config)
        tokenizer_config = loader.get_tokenizer_config()
        assert tokenizer_config["padding_side"] is None
    
    def test_yaml_config_loading_llama(self):
        """Test loading tokenizer config from YAML file - Llama with left padding."""
        config_path = Path(__file__).parent.parent / "configs" / "model" / "llama_3_1b.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            loader = ModelLoader(config)
            tokenizer_config = loader.get_tokenizer_config()
            
            # Llama config should use left padding for causal models
            assert tokenizer_config["padding_side"] == "left"
            assert tokenizer_config["trust_remote_code"] == True
            assert config["model_name"] == "meta-llama/Llama-3.2-1B"
        else:
            print(f"Warning: {config_path} not found, skipping YAML test")
    
    def test_yaml_config_loading_distilgpt2(self):
        """Test loading tokenizer config from YAML file - DistilGPT2 with right padding."""
        config_path = Path(__file__).parent.parent / "configs" / "model" / "distilgpt2.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            loader = ModelLoader(config)
            tokenizer_config = loader.get_tokenizer_config()
            
            # DistilGPT2 config should use right padding for standard models
            assert tokenizer_config["padding_side"] == "right"
            assert tokenizer_config["trust_remote_code"] == True
            assert config["model_name"] == "distilgpt2"
        else:
            print(f"Warning: {config_path} not found, skipping YAML test")
    
    def test_usage_examples_comprehensive(self):
        """Comprehensive test demonstrating all usage patterns clearly."""
        print("\n🔧 Tokenizer Configuration Usage Examples")
        print("=" * 60)
        
        # Example 1: Default behavior (backward compatible)
        print("\n1. DEFAULT: No tokenizer section (backward compatible)")
        config_default = {"model_name": "gpt2"}
        loader = ModelLoader(config_default)
        result = loader.get_tokenizer_config()
        print(f"   Result: padding_side='{result['padding_side']}', trust_remote_code={result['trust_remote_code']}")
        assert result["padding_side"] == "right"
        assert result["trust_remote_code"] == True
        
        # Example 2: Left padding for causal models (Llama, GPT)
        print("\n2. LEFT PADDING: For causal models (Llama, GPT)")
        config_left = {
            "model_name": "meta-llama/Llama-3.2-1B",
            "tokenizer": {"padding_side": "left", "trust_remote_code": True}
        }
        loader = ModelLoader(config_left)
        result = loader.get_tokenizer_config()
        print(f"   Result: padding_side='{result['padding_side']}', trust_remote_code={result['trust_remote_code']}")
        print("   Use case: Better attention computation for causal models")
        assert result["padding_side"] == "left"
        assert result["trust_remote_code"] == True
        
        # Example 3: Right padding for standard models
        print("\n3. RIGHT PADDING: For standard models (BERT-style, DistilGPT2)")
        config_right = {
            "model_name": "distilgpt2",
            "tokenizer": {"padding_side": "right"}
        }
        loader = ModelLoader(config_right)
        result = loader.get_tokenizer_config()
        print(f"   Result: padding_side='{result['padding_side']}', trust_remote_code={result['trust_remote_code']} (default)")
        print("   Use case: Standard padding for most transformer models")
        assert result["padding_side"] == "right"
        assert result["trust_remote_code"] == True  # Default
        
        # Example 4: No padding change (preserve tokenizer default)
        print("\n4. NO CHANGE: Preserve tokenizer's default behavior")
        config_none = {
            "model_name": "gpt2",
            "tokenizer": {"padding_side": None}
        }
        loader = ModelLoader(config_none)
        result = loader.get_tokenizer_config()
        print(f"   Result: padding_side={result['padding_side']} (unchanged)")
        print("   Use case: Keep original tokenizer padding behavior")
        assert result["padding_side"] is None
        
        print("\n✅ All usage examples verified successfully!")
        print("\n📋 Summary of Options:")
        print("   - padding_side: 'left' | 'right' | None")
        print("   - trust_remote_code: true | false (default: true)")
        print("\n💡 Quick Reference:")
        print("   - Causal models (Llama, GPT): Use 'left' padding")
        print("   - Standard models (BERT, DistilGPT2): Use 'right' padding")
        print("   - Preserve defaults: Use None")


if __name__ == "__main__":
    # Run basic tests
    test = TestTokenizerConfiguration()
    
    print("Running comprehensive tokenizer configuration tests...")
    print("=" * 70)
    
    tests = [
        ("Default tokenizer config", test.test_default_tokenizer_config),
        ("Custom tokenizer config", test.test_custom_tokenizer_config), 
        ("Partial tokenizer config", test.test_partial_tokenizer_config),
        ("Backward compatibility", test.test_backward_compatibility),
        ("Padding side validation", test.test_padding_side_validation),
        ("None padding side", test.test_none_padding_side),
        ("YAML config loading (Llama)", test.test_yaml_config_loading_llama),
        ("YAML config loading (DistilGPT2)", test.test_yaml_config_loading_distilgpt2),
        ("Usage examples comprehensive", test.test_usage_examples_comprehensive),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed += 1
    
    print("=" * 70)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    print("✨ Tokenizer configuration tests completed!")
    
    if failed == 0:
        print("\n🚀 All tests passed! The tokenizer configuration feature is working correctly.")
        print("📚 See the test output above for usage examples and configuration patterns.")