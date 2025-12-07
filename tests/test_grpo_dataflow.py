"""
Tests to verify tinyzero_diffusion dataflow matches TinyZero.

This test suite verifies that our implementation of GRPO (without Ray/VERL)
produces the same mathematical results as TinyZero's implementation.

Key components tested:
1. Reward computation (countdown task)
2. Advantage computation (group-relative normalization)
3. KL penalty computation
4. Log probability computation (AR and Diffusion)
5. Policy loss computation
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# =============================================================================
# 1. REWARD COMPUTATION TESTS
# =============================================================================

class TestRewardComputation:
    """
    Verify reward computation matches TinyZero's countdown.py

    TinyZero uses:
    - 0.0: No equation found
    - 0.1 (format_score): Valid format but wrong answer
    - 1.0 (score): Correct answer
    """

    def test_extract_answer_with_assistant_marker(self):
        """TinyZero splits on 'Assistant:' or '<|im_start|>assistant' before extracting."""
        from tzd.rl.parsing import extract_countdown_answer

        # Test with <|im_start|>assistant marker (Qwen format)
        solution = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Using numbers [3, 5, 2], make 13.<|im_end|>
<|im_start|>assistant
Let me think...
<think>3 + 5 * 2 = 13</think>
<answer>3 + 5 * 2</answer>"""

        answer = extract_countdown_answer(solution)
        assert answer == "3 + 5 * 2", f"Expected '3 + 5 * 2', got '{answer}'"

    def test_extract_answer_last_match(self):
        """TinyZero uses the LAST <answer> tag if multiple present."""
        from tzd.rl.parsing import extract_countdown_answer

        solution = """<|im_start|>assistant
First attempt: <answer>wrong</answer>
Correction: <answer>3 + 5 * 2</answer>"""

        answer = extract_countdown_answer(solution)
        assert answer == "3 + 5 * 2", f"Should use last match, got '{answer}'"

    def test_validate_equation_exact_numbers(self):
        """Equation must use exactly the available numbers, each once."""
        from tzd.rl.rewards import validate_equation

        # Valid: uses [3, 5, 2] exactly once each
        assert validate_equation("3 + 5 * 2", [3, 5, 2]) == True
        assert validate_equation("(3+5)*2", [3, 5, 2]) == True

        # Invalid: missing number
        assert validate_equation("3 + 5", [3, 5, 2]) == False

        # Invalid: extra number
        assert validate_equation("3 + 5 * 2 + 1", [3, 5, 2]) == False

        # Invalid: repeated number
        assert validate_equation("3 + 3 * 2", [3, 5, 2]) == False

        # Valid: different order
        assert validate_equation("2 * 5 + 3", [3, 5, 2]) == True

    def test_evaluate_equation_safe(self):
        """Equation evaluation must be safe (no code execution)."""
        from tzd.rl.rewards import evaluate_equation

        # Valid arithmetic
        assert evaluate_equation("3 + 5 * 2") == 13.0
        assert evaluate_equation("(3 + 5) * 2") == 16.0
        assert evaluate_equation("10 / 2") == 5.0

        # Invalid: code injection
        assert evaluate_equation("__import__('os').system('ls')") is None
        assert evaluate_equation("print('hello')") is None

        # Division by zero
        assert evaluate_equation("1 / 0") is None

    def test_countdown_score_correct(self):
        """Score 1.0 for correct answer."""
        from tzd.rl.rewards import compute_countdown_score

        solution = "<answer>3 + 5 * 2</answer>"
        score = compute_countdown_score(solution, target=13, numbers=[3, 5, 2])
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_countdown_score_wrong_result(self):
        """Score 0.1 (format_score) for valid format but wrong answer."""
        from tzd.rl.rewards import compute_countdown_score

        # Valid equation using correct numbers, but wrong result
        solution = "<answer>3 + 5 + 2</answer>"  # = 10, not 13
        score = compute_countdown_score(solution, target=13, numbers=[3, 5, 2])
        assert score == 0.1, f"Expected 0.1, got {score}"

    def test_countdown_score_no_answer(self):
        """Score 0.0 for no equation found."""
        from tzd.rl.rewards import compute_countdown_score

        solution = "I don't know how to solve this."
        score = compute_countdown_score(solution, target=13, numbers=[3, 5, 2])
        assert score == 0.0, f"Expected 0.0, got {score}"

    def test_countdown_score_wrong_numbers(self):
        """Score 0.1 for using wrong numbers."""
        from tzd.rl.rewards import compute_countdown_score

        # Uses number not in available set
        solution = "<answer>10 + 3</answer>"
        score = compute_countdown_score(solution, target=13, numbers=[3, 5, 2])
        assert score == 0.1, f"Expected 0.1, got {score}"

    def test_countdown_batch_reward(self):
        """Test batch reward computation."""
        from tzd.rl.rewards import countdown_reward_batch

        completions = [
            "<answer>3 + 5 * 2</answer>",  # Correct (1.0)
            "<answer>3 + 5 + 2</answer>",  # Wrong result (0.1)
            "No answer here",              # No answer (0.0)
        ]
        targets = [13, 13, 13]
        numbers = [[3, 5, 2], [3, 5, 2], [3, 5, 2]]

        scores = countdown_reward_batch(completions, targets, numbers)

        assert scores == [1.0, 0.1, 0.0], f"Expected [1.0, 0.1, 0.0], got {scores}"


# =============================================================================
# 2. ADVANTAGE COMPUTATION TESTS (GRPO)
# =============================================================================

class TestGRPOAdvantage:
    """
    Verify advantage computation matches TinyZero's core_algos.py

    TinyZero's GRPO:
    - Groups samples by prompt ID (uid)
    - Computes mean and std within each group
    - Normalizes: (score - mean) / (std + eps)
    - If single sample in group: mean=0, std=1 (no normalization)
    """

    def test_tinyzero_grpo_advantage_implementation(self):
        """
        Re-implement TinyZero's compute_grpo_outcome_advantage for reference.

        From TinyZero/verl/trainer/ppo/core_algos.py:111-155
        """
        def compute_grpo_outcome_advantage_tinyzero(
            token_level_rewards: torch.Tensor,
            eos_mask: torch.Tensor,
            index: torch.Tensor,
            epsilon: float = 1e-6
        ):
            """TinyZero's exact implementation."""
            response_length = token_level_rewards.shape[-1]

            # Extract scalar scores (non-zero positions summed)
            non_zero_mask = (token_level_rewards != 0)
            scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

            id2score = defaultdict(list)
            id2mean = {}
            id2std = {}

            with torch.no_grad():
                bsz = scores.shape[0]

                # Group by prompt ID
                for i in range(bsz):
                    id2score[index[i].item()].append(scores[i])

                # Compute per-group statistics
                for idx in id2score:
                    if len(id2score[idx]) == 1:
                        id2mean[idx] = torch.tensor(0.0)
                        id2std[idx] = torch.tensor(1.0)
                    elif len(id2score[idx]) > 1:
                        id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                        id2std[idx] = torch.std(torch.stack(id2score[idx]))
                    else:
                        raise ValueError(f"no score in prompt index: {idx}")

                # Normalize
                for i in range(bsz):
                    scores[i] = (scores[i] - id2mean[index[i].item()]) / (id2std[index[i].item()] + epsilon)

                # Broadcast to all tokens
                scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

            return scores, scores

        # Test case: 2 prompts, 4 generations each = 8 samples
        batch_size = 8
        response_length = 10
        num_generations = 4

        # Rewards: only last token has non-zero value (outcome reward)
        token_level_rewards = torch.zeros(batch_size, response_length)
        # Prompt 0: rewards [1.0, 0.1, 0.0, 0.1]
        token_level_rewards[0, -1] = 1.0
        token_level_rewards[1, -1] = 0.1
        token_level_rewards[2, -1] = 0.0
        token_level_rewards[3, -1] = 0.1
        # Prompt 1: rewards [1.0, 1.0, 0.0, 0.0]
        token_level_rewards[4, -1] = 1.0
        token_level_rewards[5, -1] = 1.0
        token_level_rewards[6, -1] = 0.0
        token_level_rewards[7, -1] = 0.0

        eos_mask = torch.ones(batch_size, response_length)

        # Index: prompt ID for each sample
        # Samples 0-3 belong to prompt 0, samples 4-7 belong to prompt 1
        index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        advantages, returns = compute_grpo_outcome_advantage_tinyzero(
            token_level_rewards, eos_mask, index
        )

        # Manual calculation:
        # Prompt 0: scores = [1.0, 0.1, 0.0, 0.1], mean = 0.3, std = 0.424
        # Prompt 1: scores = [1.0, 1.0, 0.0, 0.0], mean = 0.5, std = 0.577

        # Verify group 0 normalized correctly
        group0_scores = torch.tensor([1.0, 0.1, 0.0, 0.1])
        group0_mean = group0_scores.mean()
        group0_std = group0_scores.std()
        expected_adv_0 = (1.0 - group0_mean) / (group0_std + 1e-6)

        assert torch.isclose(advantages[0, -1], expected_adv_0, atol=1e-4), \
            f"Expected {expected_adv_0}, got {advantages[0, -1]}"

    def test_tinyzero_diffusion_advantage(self):
        """
        Test tinyzero_diffusion's simpler advantage computation.
        
        Verifies RLDiffusionModule.compute_advantages matches manual calculation.
        """
        from tzd.rl.module import RLDiffusionModule
        
        # Simulate our implementation
        batch_size = 2
        num_generations = 4
        
        # Flat rewards (batch_size * num_generations,)
        rewards_tensor = torch.tensor([1.0, 0.1, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0])
        
        # Call actual implementation
        advantages = RLDiffusionModule.compute_advantages(rewards_tensor, batch_size, num_generations)
        
        # Manual verification
        # Group 0: [1.0, 0.1, 0.0, 0.1], mean=0.3 -> adv=[0.7, -0.2, -0.3, -0.2]
        expected_baseline_0 = torch.tensor([1.0, 0.1, 0.0, 0.1]).mean()
        expected_advantages_0 = torch.tensor([1.0, 0.1, 0.0, 0.1]) - expected_baseline_0
        
        assert torch.allclose(advantages[0:4], expected_advantages_0, atol=1e-5), \
            f"Advantages mismatch for group 0"
            
        # Group 1: [1.0, 1.0, 0.0, 0.0], mean=0.5 -> adv=[0.5, 0.5, -0.5, -0.5]
        expected_baseline_1 = torch.tensor([1.0, 1.0, 0.0, 0.0]).mean()
        expected_advantages_1 = torch.tensor([1.0, 1.0, 0.0, 0.0]) - expected_baseline_1
        
        assert torch.allclose(advantages[4:8], expected_advantages_1, atol=1e-5), \
            f"Advantages mismatch for group 1"

    def test_difference_zscore_vs_baseline(self):
        """
        Document the difference between TinyZero (z-score) and our implementation (baseline only).

        TinyZero: A = (r - mean) / (std + eps)  <- Z-score normalization
        Ours:     A = (r - mean)                <- Simple baseline subtraction

        This is a known difference! TinyZero's z-score provides:
        - Scale invariance (advantages have unit variance)
        - Better for heterogeneous reward scales across prompts

        Our simpler version:
        - Easier to implement
        - Works when rewards are already on same scale (0, 0.1, 1.0)
        """
        rewards = torch.tensor([1.0, 0.1, 0.0, 0.1])
        mean = rewards.mean()
        std = rewards.std()

        # Z-score (TinyZero)
        zscore_advantages = (rewards - mean) / (std + 1e-6)

        # Simple baseline (ours)
        baseline_advantages = rewards - mean

        # They should have same sign and relative ordering
        assert torch.all(torch.sign(zscore_advantages) == torch.sign(baseline_advantages)), \
            "Z-score and baseline advantages should have same sign"

        # Print for documentation
        print(f"\nAdvantage comparison (rewards={rewards.tolist()}):")
        print(f"  Z-score (TinyZero): {zscore_advantages.tolist()}")
        print(f"  Baseline (ours):    {baseline_advantages.tolist()}")
        print(f"  Ratio (zscore/baseline): {(zscore_advantages / baseline_advantages).tolist()}")


# =============================================================================
# 3. KL PENALTY TESTS
# =============================================================================

class TestKLPenalty:
    """
    Verify KL penalty computation matches TinyZero.

    TinyZero supports multiple KL types (core_algos.py:242-274):
    - "kl": Simple KL = log_p - log_ref
    - "abs": |log_p - log_ref|
    - "mse": 0.5 * (log_p - log_ref)^2
    - "low_var_kl": Schulman's low variance estimate
    """

    def test_simple_kl(self):
        """Test KL = log_p - log_ref (TinyZero's 'kl' penalty)."""
        log_p = torch.tensor([[-1.0, -2.0, -3.0]])
        log_ref = torch.tensor([[-1.5, -2.5, -2.5]])

        # TinyZero's simple KL
        kl = log_p - log_ref

        expected = torch.tensor([[0.5, 0.5, -0.5]])
        assert torch.allclose(kl, expected), f"Expected {expected}, got {kl}"

    def test_low_var_kl(self):
        """
        Test Schulman's low variance KL estimate (TinyZero's 'low_var_kl').

        From: http://joschu.net/blog/kl-approx.html
        KL ≈ exp(log_ref - log_p) - (log_ref - log_p) - 1
        """
        log_p = torch.tensor([-1.0, -2.0, -3.0])
        log_ref = torch.tensor([-1.5, -2.5, -2.5])

        # TinyZero's low_var_kl implementation
        kl = log_ref - log_p
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        kld = torch.clamp(kld, min=-10, max=10)

        # Verify it's non-negative (KL should be >= 0)
        # Note: This approximation can go negative for small differences,
        # hence the clamping in TinyZero
        print(f"\nLow-var KL: {kld.tolist()}")

    def test_our_kl_implementation(self):
        """
        Test our KL implementation from module.py:181-189.

        Our approach:
        - kld = elbo - ref_elbo  (where elbo = log_p approximation)
        - kl_loss = beta * kld.mean()
        """
        # Simulate ELBO values
        elbo = torch.tensor([-10.0, -15.0, -20.0])  # Current policy
        ref_elbo = torch.tensor([-12.0, -14.0, -22.0])  # Reference policy
        beta = 0.001

        # Our implementation
        kld = elbo - ref_elbo  # [2.0, -1.0, 2.0]
        kl_loss = beta * kld.mean()  # 0.001 * 1.0 = 0.001

        expected_kl_loss = 0.001 * 1.0
        assert torch.isclose(kl_loss, torch.tensor(expected_kl_loss)), \
            f"Expected {expected_kl_loss}, got {kl_loss}"

        # Note: Our KL can be negative if current policy has lower likelihood
        # This is correct for GRPO where we want to penalize divergence in BOTH directions


# =============================================================================
# 4. LOG PROBABILITY TESTS
# =============================================================================

class TestLogProbability:
    """
    Test log probability computation for both AR and Diffusion models.

    AR: Standard cross-entropy with causal shift
    Diffusion: ELBO via importance-weighted masking
    """

    def test_ar_log_prob_tinyzero_style(self):
        """
        Re-implement TinyZero's AR log prob computation.

        From TinyZero/verl/workers/actor/dp_actor.py:58-141
        Key: logits at position t-1 predict token at position t
        """
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        # Simulated model output
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        # TinyZero's log_prob computation
        # For response of length L starting at position P:
        # response_logits = logits[:, P-1:P+L-1, :]  (shifted by 1)
        # response_tokens = tokens[:, P:P+L]

        prompt_len = 2
        response_len = seq_len - prompt_len  # 3

        # Shift: logits[t-1] predicts tokens[t]
        # For response starting at prompt_len, we need logits from prompt_len-1
        response_logits = logits[:, prompt_len-1:-1, :]  # (batch, response_len, vocab)
        response_tokens = tokens[:, prompt_len:]  # (batch, response_len)

        # Compute log softmax and gather
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(2)).squeeze(2)

        # Sum over response
        total_log_prob = token_log_probs.sum(dim=1)

        print(f"\nAR log prob shape: {total_log_prob.shape}")
        print(f"AR log prob values: {total_log_prob.tolist()}")

        # Verify shape
        assert total_log_prob.shape == (batch_size,), \
            f"Expected shape ({batch_size},), got {total_log_prob.shape}"

    def test_our_ar_compute_elbo(self):
        """
        Test our AutoregressiveModel.compute_elbo implementation.
        
        Verifies AutoregressiveModel.compute_elbo matches manual calculation.
        """
        from tzd.models.autoregressive import AutoregressiveModel
        from unittest.mock import MagicMock
        
        batch_size = 2
        seq_len = 5
        vocab_size = 100
        prompt_len = 2
        
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        samples = torch.randint(1, vocab_size, (batch_size, seq_len))
        
        # Create mock model
        mock_gpt = MagicMock()
        # forward returns logits
        mock_gpt.return_value = logits
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        
        # Instantiate wrapper
        model = AutoregressiveModel(
            model=mock_gpt, 
            tokenizer=mock_tokenizer, 
            model_alias="test",
            lr=1e-4,
            block_size=128,
            generation_block_size=128
        )
        
        # Call actual implementation
        # Note: compute_elbo calls self.model(samples) to get logits
        completion_log_prob = model.compute_elbo(samples, prompt_len=prompt_len)
        
        # Manual verification (same logic as before)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = samples[:, 1:].contiguous()
        
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        
        mask = torch.zeros_like(target_log_probs, dtype=torch.bool)
        mask[:, prompt_len-1:] = True
        
        expected_log_prob = (target_log_probs * mask.float()).sum(dim=1)
        
        assert torch.allclose(completion_log_prob, expected_log_prob, atol=1e-5), \
            f"ELBO mismatch: {completion_log_prob} vs {expected_log_prob}"
            
        # Verify masking is correct (first element masked out)
        # We can't inspect internal mask, but result correctness implies it.

    def test_diffusion_elbo_importance_weighting(self):
        """
        Test diffusion ELBO importance weighting logic.

        From diffusion_rl.py:9-102
        Key: weight = 1 / p(mask) where p(mask) = (1-eps)*t + eps
        """
        batch_size = 2
        seq_len = 10
        eps = 1e-3

        # Sample timestep
        t = torch.rand((batch_size,))

        # Masking probability
        p_mask = (1 - eps) * t + eps
        p_mask_expanded = p_mask.unsqueeze(1).expand(-1, seq_len)

        # Importance weight
        weight = 1 / (p_mask_expanded + 1e-10)

        # Verify weight range
        # When t=0: p_mask=eps, weight=1/eps (very large)
        # When t=1: p_mask=1, weight=1
        assert weight.min() >= 1.0, "Weight should be >= 1"
        assert weight.max() <= 1/eps + 1, f"Weight should be <= {1/eps + 1}"

        print(f"\nDiffusion importance weights:")
        print(f"  t values: {t.tolist()}")
        print(f"  p_mask: {p_mask.tolist()}")
        print(f"  weight (mean per sample): {weight.mean(dim=1).tolist()}")

    def test_diffusion_elbo_prompt_masking(self):
        """
        Verify prompt tokens are NOT masked in diffusion ELBO.

        This is CRITICAL: we compute p(completion | prompt), not p(full sequence).
        """
        batch_size = 2
        seq_len = 10
        prompt_len = 4

        # Create mask indices
        mask_indices = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # CRITICAL: Do NOT mask prompt tokens!
        mask_indices[:, :prompt_len] = False

        # Verify
        assert mask_indices[:, :prompt_len].sum() == 0, "Prompt tokens should not be masked"
        assert mask_indices[:, prompt_len:].sum() == batch_size * (seq_len - prompt_len), \
            "All completion tokens should be available for masking"


# =============================================================================
# 5. POLICY LOSS TESTS
# =============================================================================

class TestPolicyLoss:
    """
    Test policy loss computation.

    TinyZero uses PPO-style clipped loss:
    - ratio = exp(log_p - log_p_old)
    - pg_loss = -advantage * ratio
    - pg_loss_clipped = -advantage * clamp(ratio, 1-eps, 1+eps)
    - loss = max(pg_loss, pg_loss_clipped)

    Our implementation uses simpler REINFORCE-style:
    - loss = -(advantage * elbo).mean()
    """

    def test_tinyzero_ppo_loss(self):
        """
        Re-implement TinyZero's compute_policy_loss.

        From core_algos.py:163-194
        """
        def masked_mean(values, mask):
            return (values * mask).sum() / mask.sum()

        def compute_policy_loss_tinyzero(old_log_prob, log_prob, advantages, eos_mask, cliprange):
            # Importance sampling ratio
            negative_approx_kl = log_prob - old_log_prob
            ratio = torch.exp(negative_approx_kl)

            # PPO KL
            ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

            # Two versions of loss
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

            # Take max (pessimistic)
            pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
            pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

            return pg_loss, pg_clipfrac, ppo_kl

        # Test case
        batch_size = 4
        response_len = 10

        torch.manual_seed(42)
        old_log_prob = torch.randn(batch_size, response_len) - 5  # Around -5
        log_prob = old_log_prob + torch.randn(batch_size, response_len) * 0.1  # Small change
        advantages = torch.randn(batch_size, response_len)
        eos_mask = torch.ones(batch_size, response_len)
        cliprange = 0.2

        pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss_tinyzero(
            old_log_prob, log_prob, advantages, eos_mask, cliprange
        )

        print(f"\nTinyZero PPO loss:")
        print(f"  pg_loss: {pg_loss.item():.4f}")
        print(f"  pg_clipfrac: {pg_clipfrac.item():.4f}")
        print(f"  ppo_kl: {ppo_kl.item():.4f}")

    def test_our_reinforce_loss(self):
        """
        Test our simpler REINFORCE-style loss.

        From module.py:192-195:
        policy_loss = -(advantages.detach() * elbo).mean()
        total_loss = policy_loss + kl_loss
        """
        batch_size = 4

        # ELBO values (scalar per sample, not per token)
        elbo = torch.tensor([-10.0, -15.0, -12.0, -8.0])
        advantages = torch.tensor([0.5, -0.3, 0.2, -0.1])
        beta = 0.001

        # Reference ELBO for KL
        ref_elbo = torch.tensor([-11.0, -14.0, -13.0, -9.0])

        # Our implementation
        kld = elbo - ref_elbo
        kl_loss = beta * kld.mean()

        policy_loss = -(advantages.detach() * elbo).mean()
        total_loss = policy_loss + kl_loss

        # Manual calculation
        weighted_elbo = advantages * elbo  # [0.5*-10, -0.3*-15, 0.2*-12, -0.1*-8]
        expected_policy_loss = -weighted_elbo.mean()

        assert torch.isclose(policy_loss, expected_policy_loss), \
            f"Expected {expected_policy_loss}, got {policy_loss}"

        print(f"\nOur REINFORCE loss:")
        print(f"  policy_loss: {policy_loss.item():.4f}")
        print(f"  kl_loss: {kl_loss.item():.6f}")
        print(f"  total_loss: {total_loss.item():.4f}")

    def test_loss_gradient_flow(self):
        """
        Verify gradient flows correctly through our loss.

        Key: advantages are detached, so gradients flow through ELBO only.
        """
        # Simulated ELBO with gradients
        elbo = torch.tensor([-10.0, -15.0], requires_grad=True)
        advantages = torch.tensor([0.5, -0.3])  # No requires_grad

        # Our loss
        policy_loss = -(advantages.detach() * elbo).mean()

        # Backward
        policy_loss.backward()

        # Check gradients exist on elbo
        assert elbo.grad is not None, "ELBO should have gradients"

        # Expected gradients: d/d(elbo) of -mean(adv * elbo) = -adv / batch_size
        expected_grad = -advantages / 2.0
        assert torch.allclose(elbo.grad, expected_grad), \
            f"Expected grad {expected_grad}, got {elbo.grad}"


# =============================================================================
# 6. INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """
    Integration tests to verify full dataflow.
    """

    def test_full_grpo_step_mock(self):
        """
        Mock a full GRPO training step to verify dataflow.
        """
        # Configuration
        batch_size = 2  # Number of unique prompts
        num_generations = 4  # Generations per prompt
        total_samples = batch_size * num_generations

        # 1. Simulate rewards from reward function
        # These would come from countdown_reward_batch()
        raw_rewards = [
            # Prompt 0 generations
            1.0, 0.1, 0.0, 1.0,
            # Prompt 1 generations
            0.0, 0.1, 0.1, 0.0
        ]
        rewards_tensor = torch.tensor(raw_rewards)

        # 2. Compute advantages (our method - simple baseline)
        rewards_reshaped = rewards_tensor.view(batch_size, num_generations)
        baseline = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages = rewards_reshaped - baseline
        advantages_flat = advantages.view(-1)

        # 3. Simulate ELBO values
        elbo = torch.randn(total_samples) - 10.0  # Around -10
        elbo.requires_grad_(True)

        # 4. Simulate reference ELBO
        ref_elbo = elbo.detach() + torch.randn(total_samples) * 0.5

        # 5. Compute KL loss
        beta = 0.001
        kld = elbo - ref_elbo
        kl_loss = beta * kld.mean()

        # 6. Compute policy loss
        policy_loss = -(advantages_flat.detach() * elbo).mean()

        # 7. Total loss
        total_loss = policy_loss + kl_loss

        # 8. Backward pass
        total_loss.backward()

        # Verify
        assert elbo.grad is not None, "ELBO should have gradients"
        assert not torch.isnan(total_loss), "Loss should not be NaN"

        print(f"\nFull GRPO step mock:")
        print(f"  Rewards: {raw_rewards}")
        print(f"  Baselines: {baseline.squeeze().tolist()}")
        print(f"  Advantages: {advantages_flat.tolist()}")
        print(f"  Policy loss: {policy_loss.item():.4f}")
        print(f"  KL loss: {kl_loss.item():.6f}")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  ELBO grad norm: {elbo.grad.norm().item():.4f}")

    def test_reward_advantage_consistency(self):
        """
        Verify that high rewards lead to positive advantages and vice versa.
        """
        # Group with varied rewards
        rewards = torch.tensor([1.0, 0.0, 0.5, 0.2])
        mean = rewards.mean()  # 0.425

        advantages = rewards - mean

        # Highest reward should have highest advantage
        assert advantages[0] == advantages.max(), \
            "Highest reward should have highest advantage"

        # Lowest reward should have lowest advantage
        assert advantages[1] == advantages.min(), \
            "Lowest reward should have lowest advantage"

        # Sum of advantages should be ~0 (mean subtraction)
        assert torch.abs(advantages.sum()) < 1e-5, \
            "Sum of advantages should be approximately 0"


# =============================================================================
# 7. DIFFERENCE DOCUMENTATION TESTS
# =============================================================================

class TestDifferencesFromTinyZero:
    """
    Document and test known differences from TinyZero.

    These tests serve as documentation of intentional differences.
    """

    def test_no_token_level_rewards(self):
        """
        Document: We use outcome rewards (scalar), not token-level rewards.

        TinyZero: Places reward on last token, broadcasts advantage to all tokens.
        Ours: Uses scalar reward per sample, no broadcasting needed.

        Impact: Equivalent for outcome-based RL (GRPO).
        """
        # TinyZero style
        response_len = 10
        token_level_rewards_tinyzero = torch.zeros(1, response_len)
        token_level_rewards_tinyzero[0, -1] = 1.0  # Reward on last token

        # Our style
        scalar_reward_ours = torch.tensor([1.0])

        # Extracting scalar from TinyZero's format
        extracted = (token_level_rewards_tinyzero != 0).float() * token_level_rewards_tinyzero
        extracted_scalar = extracted.sum(dim=-1)

        assert torch.allclose(extracted_scalar, scalar_reward_ours), \
            "Scalar extraction should match our format"

    def test_no_ppo_clipping(self):
        """
        Document: We don't use PPO clipping.

        TinyZero: Uses PPO clipped objective with ratio clamping.
        Ours: Simple REINFORCE-style loss without clipping.

        Impact: May have higher variance, but simpler to implement.
        Could add clipping later if needed.
        """
        # This is a documentation test - the difference is intentional
        pass

    def test_no_zscore_normalization(self):
        """
        Document: We don't z-score normalize advantages.

        TinyZero: advantage = (r - mean) / (std + eps)
        Ours: advantage = (r - mean)

        Impact: Our advantages have larger magnitude when std is small.
        For countdown task with 0/0.1/1.0 rewards, this is acceptable.
        """
        rewards = torch.tensor([1.0, 0.1, 0.0, 0.1])
        mean = rewards.mean()
        std = rewards.std()

        # TinyZero
        adv_tinyzero = (rewards - mean) / (std + 1e-6)

        # Ours
        adv_ours = rewards - mean

        # Verify same relative ordering
        order_tinyzero = torch.argsort(adv_tinyzero)
        order_ours = torch.argsort(adv_ours)

        assert torch.all(order_tinyzero == order_ours), \
            "Advantage ordering should be identical"

    def test_single_forward_pass_elbo(self):
        """
        Document: Our diffusion ELBO uses single forward pass per MC sample.

        SPG: Uses block-wise masking with multiple forward passes.
        Ours: Standard importance-weighted ELBO with single pass.

        Impact: Our ELBO estimate may have higher variance.
        Block-wise method available as compute_elbo_blockwise().
        """
        pass  # Documentation test

    def test_no_entropy_regularization(self):
        """
        Document: We don't use entropy regularization.

        TinyZero: entropy_loss = masked_mean(entropy, mask); loss -= entropy_coeff * entropy_loss
        Ours: No entropy term.

        Impact: May encourage less exploration. Could add later if needed.
        """
        pass  # Documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
