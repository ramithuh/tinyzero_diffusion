
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from tzd.models.diffusion_pretrained import from_pretrained as diffusion_from_pretrained
from tzd.models.autoregressive_pretrained import from_pretrained as ar_from_pretrained
from litgpt.config import Config as LitGPTConfig
from litgpt.lora import Config as LoRAConfig
from litgpt.lora import GPT as LoRAGPT
from litgpt.model import GPT as BaseGPT

# Mock dependencies to avoid actual GPU/Checkpoint requirement
@pytest.fixture
def mock_dependencies():
    # Patch canonical locations since imports are local/lazy
    with patch("tzd.models.diffusion_pretrained.torch.load") as mock_load, \
         patch("tzd.models.diffusion_pretrained.Path.exists") as mock_exists, \
         patch("lightning.Fabric") as mock_fabric, \
         patch("lightning.fabric.plugins.BitsandbytesPrecision") as mock_bnb, \
         patch("litgpt.lora.mark_only_lora_as_trainable") as mock_freeze, \
         patch("tzd.models.autoregressive_pretrained.torch.load") as mock_ar_load, \
         patch("tzd.models.autoregressive_pretrained.Path.exists") as mock_ar_exists, \
         patch("litgpt.config.Config.from_file") as mock_config_from_file:
        
        # Setup mocks
        mock_exists.return_value = True
        mock_ar_exists.return_value = True
        
        mock_state_dict = {"model": {}}
        mock_load.return_value = mock_state_dict
        mock_ar_load.return_value = mock_state_dict
        
        # Mock Config return
        mock_config = LitGPTConfig(n_layer=2, n_head=2, n_embd=4, block_size=8)
        mock_config_from_file.return_value = mock_config
        
        # Mock Fabric context manager
        mock_fabric_instance = MagicMock()
        mock_fabric.return_value = mock_fabric_instance
        mock_fabric_instance.init_module.return_value.__enter__.return_value = None

        yield {
            "fabric": mock_fabric,
            "bnb": mock_bnb,
            "freeze": mock_freeze,
            "ar_fabric": mock_fabric,
            "ar_bnb": mock_bnb,
            "ar_freeze": mock_freeze
        }

def test_diffusion_qlora_loading(mock_dependencies):
    """Test that diffusion model loads with QLoRA when configured."""
    litgpt_config = LitGPTConfig(n_layer=2, n_head=2, n_embd=4, block_size=8)
    kwargs = {
        "quantize": "bnb.nf4",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05
    }
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50257
    tokenizer.convert_tokens_to_ids.return_value = 0
    
    # Run
    model = diffusion_from_pretrained(
        checkpoint_dir="/tmp/dummy",
        pretrained_model_name="dummy/model",
        model_alias="test_model",
        litgpt_config=litgpt_config,
        tokenizer=tokenizer,
        **kwargs
    )
    
    # Verify Fabric Quantization
    mock_dependencies["bnb"].assert_called_with(mode="nf4", dtype=torch.bfloat16)
    mock_dependencies["fabric"].assert_called_once()
    
    # Verify LoRA
    mock_dependencies["freeze"].assert_called_once()
    
    assert model is not None

def test_ar_qlora_loading(mock_dependencies):
    """Test that AR model loads with QLoRA when configured."""
    kwargs = {
        "quantize": "bnb.nf4",
        "lora_r": 8
    }
    
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50257
    tokenizer.convert_tokens_to_ids.return_value = 0
    
    # We rely on mock_config_from_file to provide config, no need to inject litgpt_config
    model = ar_from_pretrained(
        checkpoint_dir="/tmp/dummy",
        pretrained_model_name="dummy/ar_model",
        model_alias="test_ar_model",
        tokenizer=tokenizer,
        **kwargs
    )
    
    mock_dependencies["ar_bnb"].assert_called_with(mode="nf4", dtype=torch.bfloat16)
    mock_dependencies["ar_fabric"].assert_called_once()
    mock_dependencies["ar_freeze"].assert_called() # Called once or more
    assert model is not None

def test_diffusion_standard_loading(mock_dependencies):
    """Test fallback to standard loading when no quantization/LoRA."""
    litgpt_config = LitGPTConfig(n_layer=2, n_head=2, n_embd=4, block_size=8)
    
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50257
    tokenizer.convert_tokens_to_ids.return_value = 0
    
    model = diffusion_from_pretrained(
        checkpoint_dir=None, # Skip weight loading to avoid strict load error with mock
        pretrained_model_name="dummy/std_model",
        model_alias="test_std",
        litgpt_config=litgpt_config,
        tokenizer=tokenizer
    )
    
    mock_dependencies["bnb"].assert_not_called()
    mock_dependencies["fabric"].assert_not_called()
    mock_dependencies["freeze"].assert_not_called()
