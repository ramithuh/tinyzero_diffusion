"""
Tests for RL components: AutoregressiveModel and RLDiffusionModule.

These tests verify the integration of the classes, ensuring they call the underlying
methods correctly and handle data shapes as expected.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from tzd.models.autoregressive import AutoregressiveModel
from tzd.rl.module import RLDiffusionModule

# Mock LitGPT components
class MockGPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(10, 10) # Dummy layer
        
    def forward(self, x):
        # Return dummy logits [batch, seq_len, vocab_size]
        batch_size, seq_len = x.shape
        vocab_size = 100
        return torch.randn(batch_size, seq_len, vocab_size)
        
    def set_kv_cache(self, batch_size, device=None, dtype=None):
        pass
        
    def clear_kv_cache(self):
        pass

class MockBatchEncoding(dict):
    def to(self, device):
        return self
    
    @property
    def input_ids(self):
        return self["input_ids"]
        
    @property
    def attention_mask(self):
        return self["attention_mask"]

class MockTokenizer:
    def __init__(self):
        self.padding_side = "right" # Default to right to test auto-fix
        self.pad_token = None
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 1
        self.pad_token_id = 0
        
    def __call__(self, texts, **kwargs):
        # Dummy encoding
        data = {
            "input_ids": torch.randint(0, 100, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        return MockBatchEncoding(data)
        
    def decode(self, ids, **kwargs):
        return "Dummy decoded text"

@pytest.fixture
def mock_ar_model():
    model = MockGPT()
    tokenizer = MockTokenizer()
    return AutoregressiveModel(
        model=model,
        tokenizer=tokenizer,
        model_alias="test_ar",
        lr=1e-4,
        block_size=128,
        generation_block_size=128
    )

class TestAutoregressiveModel:
    """Tests for AutoregressiveModel wrapper."""
    
    def test_init_forces_left_padding(self, mock_ar_model):
        """Verify that initialization forces left padding for tokenizer."""
        # The fixture creates a tokenizer with "right" padding
        # AutoregressiveModel __init__ should change it to "left"
        assert mock_ar_model.tokenizer.padding_side == "left"
        assert mock_ar_model.tokenizer.pad_token == mock_ar_model.tokenizer.eos_token
        
    def test_sample_calls_batched_generate(self, mock_ar_model):
        """Verify sample method calls batched_generate_fn correctly."""
        batch_size = 2
        prompt_len = 5
        prompts = torch.randint(0, 100, (batch_size, prompt_len))
        
        # Mock batched_generate_fn to return an iterator of chunks
        with patch("tzd.models.autoregressive.batched_generate_fn") as mock_generate:
            # Simulate generator yielding one chunk of tokens
            mock_generate.return_value = iter([
                [torch.tensor([10, 11]), torch.tensor([20, 21])] # Chunk 1: 2 tokens for each batch item
            ])
            
            # Call sample
            max_returned_tokens = 10
            samples = mock_ar_model.sample(prompts, max_returned_tokens=max_returned_tokens)
            
            # Verify batched_generate_fn was called
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["model"] == mock_ar_model.model
            assert torch.equal(call_kwargs["prompts"], prompts)
            assert call_kwargs["max_returned_tokens"] == max_returned_tokens
            assert call_kwargs["include_prompt"] == True
            
            # Verify output shape
            # We mocked 1 chunk of 2 tokens. 
            # Logic: prompts (5) + generated (2) = 7 tokens.
            # Then padded to max_returned_tokens (10).
            assert samples.shape == (batch_size, max_returned_tokens)
            
    def test_compute_elbo_shape(self, mock_ar_model):
        """Verify compute_elbo returns correct shape and values."""
        batch_size = 3
        seq_len = 10
        samples = torch.randint(0, 100, (batch_size, seq_len))
        prompt_len = 4
        
        # Call compute_elbo
        log_probs = mock_ar_model.compute_elbo(samples, prompt_len)
        
        # Verify shape
        assert log_probs.shape == (batch_size,)
        
        # Note: We don't check gradients here because inputs are integer indices
        # and the mock model returns detached random tensors.
        # The logic test in test_grpo_dataflow.py covers gradient flow.

class TestRLDiffusionModule:
    """Tests for RLDiffusionModule integration."""
    
    @pytest.fixture
    def rl_module(self, mock_ar_model):
        # Create RLDiffusionModule with the mock AR model
        return RLDiffusionModule(
            model=mock_ar_model,
            lr=1e-4,
            num_generations=2, # Small for testing
            beta=0.01
        )
        
    def test_training_step_dataflow(self, rl_module):
        """
        Verify training_step calls sample, compute_elbo, and computes loss.
        This is a high-level integration test.
        """
        batch_size = 2
        prompt_len = 5
        
        # Create a dummy batch
        batch = {
            "input_ids": torch.randint(0, 100, (batch_size, prompt_len)),
            "attention_mask": torch.ones(batch_size, prompt_len),
            "prompts": ["3 5 2", "1 2 3"], # Added prompts key
            "numbers": [[1, 2], [3, 4]],
            "targets": [3, 7]
        }
        
        # Mock sample to return fixed sequences
        # We need to mock the underlying model.sample because RLDiffusionModule calls self.model.sample
        # But self.model is the AutoregressiveModel wrapper.
        
        # Let's patch AutoregressiveModel.sample and compute_elbo
        with patch.object(rl_module.model, "sample") as mock_sample, \
             patch.object(rl_module.model, "compute_elbo") as mock_elbo, \
             patch("tzd.rl.module.countdown_reward_batch") as mock_rewards:
             
            # Setup mocks
            seq_len = 10
            mock_sample.return_value = torch.randint(0, 100, (batch_size * rl_module.num_generations, seq_len))
            # Ensure elbo returns a tensor with gradients enabled for loss computation
            mock_elbo.return_value = torch.randn(batch_size * rl_module.num_generations, requires_grad=True) 
            mock_rewards.return_value = [1.0] * (batch_size * rl_module.num_generations) # Dummy rewards
            
            # Run training step
            loss = rl_module.training_step(batch, batch_idx=0)
            
            # Verify calls
            assert mock_sample.called
            assert mock_elbo.called
            assert mock_rewards.called
            
            # Verify loss is a scalar tensor
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0
            
    def test_validation_step_logging(self, rl_module):
        """Verify validation step logs rewards."""
        # Attach mock trainer for logging
        rl_module.trainer = MagicMock()
        rl_module.trainer.global_step = 0
        
        batch_size = 2
        prompt_len = 5
        
        batch = {
            "input_ids": torch.randint(0, 100, (batch_size, prompt_len)),
            "attention_mask": torch.ones(batch_size, prompt_len),
            "prompts": ["3 5 2", "1 2 3"], # Added prompts key
            "numbers": [[1, 2], [3, 4]],
            "targets": [3, 7]
        }
        
        # Mock dependencies
        with patch.object(rl_module.model, "sample") as mock_sample, \
             patch("tzd.rl.module.countdown_reward_batch") as mock_rewards, \
             patch.object(rl_module, "log_dict") as mock_log_dict: # Mock log_dict instead of log
             
            mock_sample.return_value = torch.randint(0, 100, (batch_size * rl_module.num_generations, 10))
            mock_rewards.return_value = [1.0] * (batch_size * rl_module.num_generations)
            
            # Run validation step
            rl_module.validation_step(batch, batch_idx=0)
            
            # Verify logging
            mock_log_dict.assert_called()
            # Check if 'val/reward_mean' was logged
            logged_keys = [call[0][0] for call in mock_log_dict.call_args_list]
            # Flatten keys if they are dicts
            all_keys = []
            for k in logged_keys:
                if isinstance(k, dict):
                    all_keys.extend(k.keys())
                else:
                    all_keys.append(k)
            
            assert "val/reward_mean" in all_keys
