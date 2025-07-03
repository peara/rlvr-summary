"""Tests for tokenizer configuration functionality."""

import sys
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


if __name__ == "__main__":
    # Run basic tests
    test = TestTokenizerConfiguration()
    
    print("Running tokenizer configuration tests...")
    
    try:
        test.test_default_tokenizer_config()
        print("✅ Default tokenizer config test passed")
    except Exception as e:
        print(f"❌ Default tokenizer config test failed: {e}")
    
    try:
        test.test_custom_tokenizer_config()
        print("✅ Custom tokenizer config test passed")
    except Exception as e:
        print(f"❌ Custom tokenizer config test failed: {e}")
    
    try:
        test.test_partial_tokenizer_config()
        print("✅ Partial tokenizer config test passed")
    except Exception as e:
        print(f"❌ Partial tokenizer config test failed: {e}")
    
    try:
        test.test_backward_compatibility()
        print("✅ Backward compatibility test passed")
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
    
    try:
        test.test_padding_side_validation()
        print("✅ Padding side validation test passed")
    except Exception as e:
        print(f"❌ Padding side validation test failed: {e}")
    
    try:
        test.test_none_padding_side()
        print("✅ None padding side test passed")
    except Exception as e:
        print(f"❌ None padding side test failed: {e}")
    
    print("✨ Tokenizer configuration tests completed!")