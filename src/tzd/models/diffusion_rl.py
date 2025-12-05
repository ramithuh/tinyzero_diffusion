"""RL-specific methods for DiffusionModel."""
import torch
import torch.nn.functional as F


class DiffusionRLMixin:
    """Mixin to add RL likelihood estimation to DiffusionModel."""

    def compute_elbo(
        self,
        x: torch.Tensor,
        num_samples: int = 10,
        eps: float = 1e-3,
        prompt_len: int = 0
    ) -> torch.Tensor:
        """
        Compute ELBO (lower bound) for sequence x.

        ELBO = E_{t,z_t}[w(t) · 1(z_t,i = mask) · log π_θ(x_i | z_t)]

        Args:
            x: Token sequence (batch_size, seq_len) - FULL sequence (prompt + completion)
            num_samples: Number of Monte Carlo samples (K)
            eps: Minimum masking probability
            prompt_len: Length of prompt tokens to exclude from loss (default: 0, meaning use full sequence)

        Returns:
            ELBO estimate (batch_size,) computed ONLY on completion tokens
        """
        if not hasattr(self, 'mask_token_id'):
            raise AttributeError("DiffusionRLMixin requires mask_token_id attribute")

        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor (batch, seq_len), got {x.dim()}D")

        batch_size, seq_len = x.shape
        device = x.device

        # Collect log-probs from multiple samples
        logprobs_list = []

        for _ in range(num_samples):
            # Sample timestep t ~ Uniform[0, 1]
            t = torch.rand((batch_size,), device=device)

            # Masking probability: p(t) = (1-ε)t + ε
            p_mask = (1 - eps) * t + eps  # (batch_size,)
            p_mask = p_mask.unsqueeze(1).expand(-1, seq_len)  # (batch_size, seq_len)

            # Randomly mask tokens
            mask_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
            
            # CRITICAL: Do NOT mask prompt tokens! We want p(completion | prompt).
            if prompt_len > 0:
                mask_indices[:, :prompt_len] = False

            # Apply masking
            noisy_x = torch.where(mask_indices, self.mask_token_id, x)

            # Forward pass
            logits = self.forward(noisy_x)  # (batch_size, seq_len, vocab_size)

            # Compute log p(x_i | noisy_x) using cross_entropy
            # cross_entropy = -log(p(x)), so log(p(x)) = -cross_entropy
            # We flatten to (N, C) for cross_entropy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = x.view(-1)
            
            # reduction='none' gives loss per token
            nll = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            token_logprobs = -nll.view(batch_size, seq_len)

            # Weight by 1/p(mask) (importance weighting)
            # Add small constant for numerical stability
            weighted_logprobs = token_logprobs / (p_mask + 1e-10)

            # Only count masked positions
            masked_logprobs = weighted_logprobs * mask_indices.float()

            # CRITICAL FIX: Only sum over COMPLETION tokens (exclude prompt)
            # If prompt_len > 0, we only compute loss on tokens after the prompt
            if prompt_len > 0:
                completion_logprobs = masked_logprobs[:, prompt_len:]  # (batch_size, completion_len)
                seq_logprob = completion_logprobs.sum(dim=1)  # (batch_size,)
            else:
                # Full sequence (for backward compatibility if prompt_len not provided)
                seq_logprob = masked_logprobs.sum(dim=1)  # (batch_size,)

            logprobs_list.append(seq_logprob)

        # Average over samples
        elbo = torch.stack(logprobs_list).mean(dim=0)  # (batch_size,)

        # Memory cleanup
        del logprobs_list
        if 'logits' in locals():
            del logits
        if 'noisy_x' in locals():
            del noisy_x
        torch.cuda.empty_cache()

        return elbo


    def compute_elbo_blockwise(
        self,
        x: torch.Tensor,
        num_blocks: int = 4,
        num_samples: int = 10,
        eps: float = 1e-3,
        prompt_len: int = 0
    ) -> torch.Tensor:
        """
        Compute ELBO with block-wise masking (SPG's approach).

        Block i strategy:
        - Blocks 0..i-1: Clean (no masking)
        - Block i: Partially masked with p(t)
        - Blocks i+1..end: Fully masked

        Args:
            x: Token sequence (batch_size, seq_len) - FULL sequence (prompt + completion)
            num_blocks: Number of blocks to divide sequence into
            num_samples: Monte Carlo samples per block
            eps: Minimum masking probability
            prompt_len: Length of prompt tokens to exclude from loss (default: 0, meaning use full sequence)

        Returns:
            ELBO estimate (batch_size,) computed ONLY on completion tokens
        """
        if not hasattr(self, 'mask_token_id'):
            raise AttributeError("DiffusionRLMixin requires mask_token_id attribute")

        batch_size, seq_len = x.shape
        device = x.device

        # CRITICAL FIX: Only divide COMPLETION into blocks (exclude prompt)
        completion_start = prompt_len
        completion_len = seq_len - prompt_len

        # Better block division to handle uneven blocks (only on completion)
        block_boundaries = torch.linspace(0, completion_len, num_blocks + 1, dtype=torch.long)
        # Shift boundaries to start after prompt
        block_boundaries = block_boundaries + completion_start

        total_logprob = torch.zeros(batch_size, device=device)

        for block_idx in range(num_blocks):
            block_start = block_boundaries[block_idx].item()
            block_end = block_boundaries[block_idx + 1].item()

            block_logprobs = []

            for _ in range(num_samples):
                # Sample timestep
                t = torch.rand((batch_size,), device=device)
                p_mask = (1 - eps) * t + eps

                # Create block-wise mask
                mask_pattern = torch.zeros_like(x, dtype=torch.bool)

                # Fully mask future blocks
                if block_end < seq_len:
                    mask_pattern[:, block_end:] = True

                # Partially mask current block
                current_block_mask = torch.rand((batch_size, block_end - block_start), device=device)
                current_block_mask = current_block_mask < p_mask.unsqueeze(1)
                
                # CRITICAL: Do NOT mask prompt tokens!
                # block_start and block_end are already shifted by prompt_len, so this mask applies 
                # ONLY to the completion part. No extra masking needed here because 
                # we only insert this mask into mask_pattern[:, block_start:block_end].
                # And block_start >= prompt_len.
                # So we are safe!
                
                mask_pattern[:, block_start:block_end] = current_block_mask

                # Apply masking
                noisy_x = torch.where(mask_pattern, self.mask_token_id, x)

                # Forward pass
                # Forward pass
                logits = self.forward(noisy_x)
                
                # Compute log p(x) using cross_entropy
                # Only need gradients for the current block? 
                # Actually, forward computes all logits.
                # We can slice logits before cross_entropy to save memory if cross_entropy supports it.
                # Yes, we only care about block_start:block_end.
                
                block_logits = logits[:, block_start:block_end, :]
                block_tokens = x[:, block_start:block_end]
                
                flat_logits = block_logits.reshape(-1, block_logits.size(-1))
                flat_targets = block_tokens.reshape(-1)
                
                nll = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                token_logprobs = -nll.view(batch_size, block_end - block_start)

                # Weight and mask
                p_mask_expanded = p_mask.unsqueeze(1).expand(-1, block_end - block_start)
                weighted_logprobs = token_logprobs / (p_mask_expanded + 1e-10)
                masked_logprobs = weighted_logprobs * current_block_mask.float()

                block_logprob = masked_logprobs.sum(dim=1)
                block_logprobs.append(block_logprob)

            # Average over samples for this block
            total_logprob += torch.stack(block_logprobs).mean(dim=0)

        # Memory cleanup
        if 'logits' in locals():
            del logits
        if 'noisy_x' in locals():
            del noisy_x
        torch.cuda.empty_cache()

        return total_logprob
