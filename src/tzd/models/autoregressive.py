import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from tzd.models.base import BaseModel
from litgpt.model import GPT
from litgpt.generate.base import batched_generate_fn

class AutoregressiveModel(BaseModel):
    """
    Wrapper for LitGPT autoregressive models to be compatible with RLDiffusionModule.
    """
    def __init__(self, model: GPT, tokenizer: Any, model_alias: str, lr: float, block_size: int = 1024, generation_block_size: int = 1024):
        super().__init__(model_alias, lr)
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.generation_block_size = generation_block_size
        
        # CRITICAL: For batched autoregressive generation, we need LEFT padding
        # so that the last token in the prompt is at the same position relative to the end
        # and generation appends correctly.
        if self.tokenizer.padding_side != "left":
            print(f"Warning: Tokenizer padding_side is {self.tokenizer.padding_side}. Forcing 'left' padding for AR generation.")
            self.tokenizer.padding_side = "left"
            
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def forward(self, x):
        return self.model(x)
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
        
    def sample(
        self,
        prompts: torch.Tensor,
        max_returned_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using the autoregressive model.
        
        Args:
            prompts: Input token IDs [batch_size, prompt_len]
            max_returned_tokens: Maximum total length of generated sequences (including prompt)
            temperature: Sampling temperature
            **kwargs: Additional sampling arguments (top_k, top_p, etc.) and compatibility args (seq_len)
            
        Returns:
            Generated sequences [batch_size, max_returned_tokens]
        """
        # Handle compatibility with DiffusionModel interface which uses seq_len
        if max_returned_tokens is None:
            max_returned_tokens = kwargs.get("seq_len")
            
        if max_returned_tokens is None:
            # Fallback to generation_block_size if available, else block_size
            max_returned_tokens = getattr(self, "generation_block_size", getattr(self, "block_size", 1024))
            
        batch_size = prompts.shape[0]
        device = prompts.device
        
        # Prepare sample args for batched_generate_fn
        sample_args = {
            "temperature": temperature,
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        # Initialize KV cache for generation
        # litgpt requires set_kv_cache to be called if input_pos is used (which batched_generate_fn does)
        # CRITICAL: Must pass device, otherwise it defaults to CPU and causes RuntimeError
        self.model.set_kv_cache(batch_size=batch_size, device=device)
        
        try:
            # Use litgpt's batched generation
            # Returns iterator of lists of tokens
            iterator = batched_generate_fn(
                model=self.model,
                prompts=prompts,
                max_returned_tokens=max_returned_tokens,
                sample_args=sample_args,
                include_prompt=True,
                include_eos=True, # We want to stop at EOS if generated
                stop_tokens=([self.tokenizer.eos_token_id],) if self.tokenizer.eos_token_id is not None else (),
            )
            
            # Consume iterator to accumulate tokens
            # batched_generate_fn yields lists of tokens (one per batch item)
            # We need to accumulate them for each batch element
            batch_accum = [[] for _ in range(batch_size)]
            
            for batch_chunk in iterator:
                for i, token_chunk in enumerate(batch_chunk):
                    if token_chunk is not None:
                        batch_accum[i].append(token_chunk)
        finally:
            # Always clear KV cache to free memory
            self.model.clear_kv_cache()
            
        # Concatenate and pad sequences
        padded_batch = []
        for i, chunks in enumerate(batch_accum):
            if not chunks:
                # Should not happen if include_prompt=True
                continue
                
            # Concatenate all chunks (prompt + generated)
            full_seq = torch.cat(chunks)
            
            current_len = full_seq.shape[0]
            if current_len < max_returned_tokens:
                # Pad with EOS or PAD token
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
                padding = torch.full((max_returned_tokens - current_len,), pad_id, device=device, dtype=full_seq.dtype)
                padded_seq = torch.cat([full_seq, padding])
            else:
                padded_seq = full_seq[:max_returned_tokens]
            padded_batch.append(padded_seq)
            
        return torch.stack(padded_batch)
        
    def compute_elbo(
        self,
        samples: torch.Tensor,
        prompt_len: int,
        num_samples: int = 1,
        eps: float = 1e-3,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of the completion given the prompt.
        For AR models, this is equivalent to the negative ELBO (or just log prob).
        
        Args:
            samples: Full sequences [batch_size, seq_len]
            prompt_len: Length of the prompt (to mask out)
            num_samples: Number of MC samples (ignored for AR as it's deterministic)
            eps: Epsilon for numerical stability (ignored for AR)
            
        Returns:
            Log probability of the completion [batch_size]
        """
        # 1. Forward pass to get logits
        logits = self.model(samples)
        
        # 2. Shift logits and labels for causal loss
        # logits[i] predicts samples[i+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = samples[:, 1:].contiguous()
        
        # 3. Compute log probabilities
        # Use cross_entropy with reduction='none' to get per-token loss, then negate for log_prob
        # Or manually: log_softmax + gather
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs_all.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        
        # 4. Masking
        # We only care about the completion, i.e., tokens from prompt_len onwards.
        # Note: shift_labels[i] corresponds to samples[i+1].
        # We want to score samples[prompt_len], samples[prompt_len+1], ...
        # These correspond to indices prompt_len-1, prompt_len, ... in shift_labels.
        mask = torch.zeros_like(target_log_probs, dtype=torch.bool)
        mask[:, prompt_len-1:] = True
        
        # Also mask out padding tokens
        if self.tokenizer.pad_token_id is not None:
            is_pad = (shift_labels == self.tokenizer.pad_token_id)
            mask = mask & (~is_pad)
            
        # 5. Sum log probs over the sequence
        completion_log_prob = (target_log_probs * mask.float()).sum(dim=1)
        
        return completion_log_prob
