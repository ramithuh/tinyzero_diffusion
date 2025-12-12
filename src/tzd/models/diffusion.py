"""Diffusion model for sequence data using SMDM approach.

Currently utilizes lit_gpt variant for initializing a diffusion model.
"""
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any, Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from tzd.models.base import BaseModel
from tzd.utils.generation import log_generations
from tzd.models.llada.inference import generate as llada_sample
from tzd.models.diffusion_rl import DiffusionRLMixin

from litgpt.model import GPT, Block, Config as GPTConfig
from litgpt.config import Config as LitGPTConfig


class DiffusionModel(BaseModel, DiffusionRLMixin):
    """
    Diffusion model wrapper supporting both SMDM and LitGPT backends.

    Supports two model types:
    - 'smdm': Original SMDM TransEncoder with non-causal attention
    - 'litgpt': New LitGPT GPT with causal=False for diffusion
    """
    def __init__(
        self,
        model_alias: str,
        lr: float,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        tokenizer: callable,
        bias: bool = False,
        model_type: str = "gpt2",
        litgpt_config: Optional[LitGPTConfig] = None,
        gpt_model: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "gpt_model"]) # Ignore non-serializable args

        self.model_alias = model_alias
        self.lr = lr
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.generation_block_size = kwargs.get("generation_block_size", block_size)
        self.val_temperatures       = kwargs.get("val_temperatures", [1.0])
        self.generation_num_steps  = kwargs.get("generation_num_steps", block_size // 2)
        self.sampling_repo         = kwargs.get("sampling_repo", "LLaDA")  # Default to LLaDA (SMDM requires rotary_emb)

        self.unk_token_id = tokenizer.unk_token_id  # unknown token id that avoid loss computation
        
        # Initialize mask_token_id
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        elif hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
             self.mask_token_id = tokenizer.unk_token_id
        elif hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            self.mask_token_id = tokenizer.pad_token_id
        else:
             # Fallback for some tokenizers
             self.mask_token_id = tokenizer.eos_token_id
             
        # Ensure mask_token_id is valid
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a valid mask_token_id, unk_token_id, or pad_token_id for diffusion.")

        self.validation_samples_table = wandb.Table(
            columns=["Epoch", "Step", "Temperature", "Sample 1", "Sample 2", "Sample 3"],
            log_mode="MUTABLE"
        )

        # Initialize model based on model_type
        if gpt_model is not None:
            self.model = gpt_model
        elif model_type == "smdm":
            # Original SMDM path
            from tzd.models.smdm.diffmodel import TransEncoder
            from tzd.models.smdm.config import Config

            config = Config(
                vocab_size=self.vocab_size,
                padded_vocab_size=self.vocab_size, # otherwise default padding multiple 512 takes over
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                block_size=block_size,
                bias=bias,
            )
            self.model = TransEncoder(config)

        elif model_type == "litgpt":
            # New LitGPT path with non-causal attention
            # If a pre-built litgpt_config is provided (from pretrained), use it
            if "litgpt_config" in kwargs:
                config = kwargs["litgpt_config"]
                # Ensure causal=False for diffusion
                config.causal = False
            else:
                # Build config from scratch
                config = LitGPTConfig(
                    vocab_size=self.vocab_size,
                    padded_vocab_size=self.vocab_size,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    block_size=block_size,
                    bias=bias,
                    causal=False,  # Non-causal attention for diffusion
                    norm_class_name=kwargs.get("norm_class_name", "LayerNorm"),
                    mlp_class_name=kwargs.get("mlp_class_name", "GptNeoxMLP"),
                    intermediate_size=kwargs.get("intermediate_size", 4 * n_embd),
                    rotary_percentage=kwargs.get("rotary_percentage", 1.0),
                )
            self.model = GPT(config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: 'smdm', 'litgpt'")

    def forward(self, x):
        if self.model_type == "smdm":
            # TransEncoder expects (batch_size, seq_len) and returns (batch_size, seq_len, vocab_size)
            logits = self.model(x).logits
        elif self.model_type == "litgpt":
            # GPT expects (batch_size, seq_len) and returns (batch_size, seq_len, vocab_size)
            # For diffusion models, we don't need KV cache (input_pos=None)
            # This makes it behave like TransEncoder - process full sequence at once
            logits = self.model(x, input_pos=None)
        elif self.model_type == "huggingface":
             logits = self.model(x)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return logits

    def forward_process(self, batch, eps=1e-3, loss_mask=None):
        """
        Forward diffusion process: randomly mask tokens with the mask token.

        :param batch: Input tokens of shape (batch_size, seq_len)
        :param eps: Minimum masking probability to avoid fully unmasked sequences
        :param loss_mask: Optional mask (1 for learnable, 0 for fixed/prompt). 
                          If provided, fixed tokens (0) will NOT be masked.

        :returns noisy_batch: Tokens with some positions masked
        :returns mask_indices: Boolean tensor indicating which positions were masked
        :returns p_mask: Masking probabilities used
        """
        b, l = batch.shape
        t = torch.rand((b,), device=batch.device)

        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        
        # If loss_mask is provided, ensure fixed tokens (where mask is 0) are NEVER masked
        if loss_mask is not None:
            # loss_mask is 1 for completion (can be masked), 0 for prompt (keep fixed)
            mask_indices = mask_indices & (loss_mask.bool())

        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, mask_indices, p_mask

    def _compute_loss(self, batch):
        """Implement SMDM's forward pass + loss with proper re-weighting."""
        input_ids = batch['input_ids']

        if(input_ids.size(1) > self.model.max_seq_length):
            warnings.warn(f"Input sequence length {input_ids.size(1)} exceeds model's max_seq_length {self.model.max_seq_length}. "
                "Truncating input to fit the model.")

        max_length = min(self.model.max_seq_length, batch["input_ids"].size(1) - 1)
        input_ids = batch["input_ids"][:, 0: max_length].contiguous().long()

        # If batch provides a loss_mask (e.g. for masking prompts), use it
        loss_mask_input = None
        if "loss_mask" in batch:
            # Extract loss mask matching the input_ids shape (truncated if needed)
            loss_mask_input = batch["loss_mask"][:, 0: max_length].contiguous()

        noisy_tokens, mask_indices, p_mask = self.forward_process(input_ids, loss_mask=loss_mask_input)
        logits = self.model(noisy_tokens)

        valid_tokens = input_ids != self.unk_token_id
        loss_mask    = mask_indices & valid_tokens

        # If batch provides a loss_mask, apply it to the loss computation as well
        if loss_mask_input is not None:
            loss_mask = loss_mask & loss_mask_input.bool()

        # implement the re-weighted loss from SMDM
        loss = F.cross_entropy(logits[loss_mask], input_ids[loss_mask], reduction='none')
        loss = loss / p_mask[loss_mask]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        return loss

    def on_validation_epoch_end(self):
        """Generate samples and log to wandb at end of validation epoch."""        
        # Only generate after training has started
        if self.val_generation_freq and self.trainer.current_epoch % self.val_generation_freq == 0:
            try:
                log_generations(
                    trainer=self.trainer,
                    model=self,
                    datamodule=self.trainer.datamodule,
                    epoch=self.trainer.current_epoch,
                    step=self.trainer.global_step,
                    temperatures=self.val_temperatures,
                    num_samples=3,
                    seq_len=self.generation_block_size,
                    num_steps=self.generation_num_steps,
                    wandb_table=self.validation_samples_table,
                    stage="validation"
                )
            except Exception as e:
                print(f"[DEBUG] ERROR in log_generations: {e}")
                import traceback
                traceback.print_exc()

    def sample(self, batch_size=1, seq_len=None, num_steps=None, temperature=1.0, repo="LLaDA", prompts=None, block_length=None):
        """Call inference functions borrowed from SMDM's & LLaDA's inference.py

        :param batch_size: Number of sequences to generate
        :param seq_len: Length of sequences to generate (defaults to model's block_size)
        :param num_steps: Number of denoising steps (defaults to seq_len // 2)
        :param temperature: Sampling temperature (0.0 for greedy)
        :param repo: Sampling method to use ('LLaDA' or 'SMDM'). Defaults to 'LLaDA' (SMDM requires rotary_emb)
        :param prompts: Optional tensor of shape (batch_size, prompt_len) containing prompt tokens
        :param block_length: Optional block length for semi-autoregressive generation (defaults to seq_len)
        :return tokens: Generated sequences of shape (batch_size, seq_len)
        """
        device = next(self.parameters()).device

        if repo == "SMDM":
            from tzd.models.smdm.inference import diff_sample as smdm_sample
            
            output_tokens = smdm_sample(
                model=self.model,
                tokenizer=None,
                prompt=prompts, # SMDM might support prompts, passing it through
                batch_size=batch_size,
                alg='origin',
                steps=num_steps or self.generation_num_steps,
                temperature=temperature,
                cfg_scale=2.0,
                context_length=seq_len or self.block_size,
                eps=1e-5,
                dim=self.mask_token_id,  # Use mask token as dim
                device=device
            )
        elif repo == "LLaDA":
            # LLaDA's generate() only supports batch_size=1, so we loop
            if prompts is not None:
                # Conditional generation
                assert prompts.shape[0] == batch_size, "Prompts batch size must match requested batch_size"
                samples = []
                for i in range(batch_size):
                    # Get single prompt: (1, L)
                    prompt = prompts[i].unsqueeze(0)
                    
                    # Calculate max possible generation length
                    prompt_len = prompt.shape[1]
                    max_gen_len = self.block_size - prompt_len
                    requested_gen_len = seq_len or self.generation_block_size
                    
                    if requested_gen_len > max_gen_len:
                        # print(f"Warning: Requested gen_len {requested_gen_len} + prompt_len {prompt_len} > block_size {self.block_size}. Truncating gen_len to {max_gen_len}.")
                        actual_gen_len = max_gen_len
                    else:
                        actual_gen_len = requested_gen_len

                    sample = llada_sample(
                        model=self.model,
                        prompt=prompt,
                        steps=num_steps or self.generation_num_steps,
                        gen_length=actual_gen_len,
                        block_length=block_length or actual_gen_len, # Use provided block_length or default to full gen_length
                        temperature=temperature,
                        cfg_scale=0.0,
                        remasking='low_confidence',
                        mask_id=self.mask_token_id,
                        device=device
                    )
                    samples.append(sample)
                output_tokens = torch.cat(samples, dim=0)
            else:
                # Unconditional generation
                empty_prompt = torch.empty((1, 0), dtype=torch.long).to(device)
                
                # Calculate max possible generation length (same logic as conditional)
                requested_gen_len = seq_len or self.generation_block_size
                max_gen_len = self.block_size  # No prompt, so full block_size available
                
                if requested_gen_len > max_gen_len:
                    actual_gen_len = max_gen_len
                else:
                    actual_gen_len = requested_gen_len
                
                samples = []
                for _ in range(batch_size):
                    sample = llada_sample(
                        model=self.model,
                        prompt=empty_prompt,
                        steps=num_steps or self.generation_num_steps,
                        gen_length=actual_gen_len,
                        block_length=block_length or actual_gen_len, # Use provided block_length or default to full gen_length
                        temperature=temperature,
                        cfg_scale=0.0,
                        remasking='low_confidence',
                        mask_id=self.mask_token_id,
                        device=device
                    )
                    samples.append(sample)
                output_tokens = torch.cat(samples, dim=0)  # Stack to (batch_size, seq_len)
        else:
            raise ValueError(f"Unknown repo: {repo}. Supported repos are 'SMDM' and 'LLaDA'")

        return output_tokens

    def test_step(self, batch, batch_idx):
        """Test step - same as validation for now."""
        return self.validation_step(batch, batch_idx)
