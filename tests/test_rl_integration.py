"""
Integration tests for RL pipeline.
Verifies critical dataflow that unit tests missed, specifically:
1. Prompt stripping before reward computation.
2. Prompt masking during ELBO computation.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer
from tzd.rl.module import RLDiffusionModule
from tzd.models.autoregressive import AutoregressiveModel

# Use a small real tokenizer (Qwen)
@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("checkpoints/Qwen/Qwen2.5-3B")

@pytest.fixture
def mock_ar_model(tokenizer):
    """
    Mock AutoregressiveModel that behaves like the real one but with controlled output.
    """
    model = MagicMock()
    model.tokenizer = tokenizer
    model.generation_block_size = 128
    model.model_alias = "test_ar"
    
    # Mock sample to return concatenated [prompt, completion]
    # We need to control this per call, so we'll patch it in the test
    return model

@pytest.fixture
def rl_module(mock_ar_model):
    module = RLDiffusionModule(
        model=mock_ar_model,
        lr=1e-4,
        num_generations=1,
        beta=0.01
    )
    module.trainer = MagicMock()
    module.log = MagicMock()
    module.log_dict = MagicMock()
    return module

def test_reward_input_strips_prompt(rl_module, tokenizer):
    """
    CRITICAL TEST: Verify that the string passed to reward function 
    does NOT contain the prompt.
    """
    # 1. Setup Data
    prompt_text = "User: Solve 1+1.\nAssistant:"
    completion_text = "<answer> 2 </answer>"
    
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    completion_ids = tokenizer(completion_text, return_tensors="pt").input_ids
    
    # Combine (simulate model outputting prompt + completion)
    # Note: Qwen tokenizer might add special tokens, let's be careful.
    # We'll just concatenate the IDs.
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    # 2. Mock Model Behavior
    # rl_module.model.sample should return full_ids
    rl_module.model.sample.return_value = full_ids
    
    # Mock compute_elbo to return dummy loss
    rl_module.model.compute_elbo.return_value = torch.tensor(0.0, requires_grad=True)
    
    # 3. Mock Reward Function
    with patch("tzd.rl.module.countdown_reward_batch") as mock_rewards:
        mock_rewards.return_value = [1.0]
        
        # 4. Run Training Step
        batch = {
            "prompts": [prompt_text],
            "targets": [2],
            "numbers": [[1, 1]]
        }
        
        rl_module.training_step(batch, batch_idx=0)
        
        # 5. Verify Input to Reward Function
        assert mock_rewards.called
        call_args = mock_rewards.call_args
        decoded_inputs = call_args[0][0] # First arg is list of strings
        
        print(f"\nPrompt: {repr(prompt_text)}")
        print(f"Completion: {repr(completion_text)}")
        print(f"Decoded Input to Reward: {repr(decoded_inputs[0])}")
        
        # ASSERTION: The decoded input should NOT contain the prompt text
        # It should match the completion text (approximately, ignoring special tokens)
        assert prompt_text not in decoded_inputs[0], "Prompt leakage detected! Reward function received prompt."
        assert "<answer> 2 </answer>" in decoded_inputs[0], "Completion missing from reward input."

def test_elbo_masking_strips_prompt(rl_module, tokenizer):
    """
    Verify that compute_elbo is called with the correct prompt_len,
    implying that masking will happen correctly (assuming AutoregressiveModel is correct).
    """
    prompt_text = "User: Solve 1+1.\nAssistant:"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    
    full_ids = torch.cat([prompt_ids, torch.tensor([[1, 2, 3]])], dim=1) # Dummy completion
    
    rl_module.model.sample.return_value = full_ids
    rl_module.model.compute_elbo.return_value = torch.tensor(0.0, requires_grad=True)
    
    with patch("tzd.rl.module.countdown_reward_batch") as mock_rewards:
        mock_rewards.return_value = [1.0]
        
        batch = {
            "prompts": [prompt_text],
            "targets": [2],
            "numbers": [[1, 1]]
        }
        
        rl_module.training_step(batch, batch_idx=0)
        
        # Verify compute_elbo call
        rl_module.model.compute_elbo.assert_called()
        call_kwargs = rl_module.model.compute_elbo.call_args[1]
        
        assert call_kwargs["prompt_len"] == prompt_len
        print(f"\nPassed prompt_len: {call_kwargs['prompt_len']}")
        print(f"Actual prompt_len: {prompt_len}")

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
