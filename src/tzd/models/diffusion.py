"""Diffusion model for sequence data using SMDM approach.

Currently utilizes lit_gpt variant for initializing a diffusion model.
"""
import warnings

import torch
import torch.nn.functional as F
import wandb
from tzd.models.base import BaseModel
from tzd.utils.generation import log_generations
from tzd.models.llada.inference import generate as llada_sample

from litgpt.model import GPT
from litgpt.config import Config as LitGPTConfig


class DiffusionModel(BaseModel):
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
        bias: bool = True,
        model_type: str = "smdm",  # 'smdm' or 'litgpt'
        **kwargs
    ):
        """
        Initialize the diffusion model.
        """
        super().__init__(model_alias=model_alias, lr=lr)

        self.block_size = block_size
        self.vocab_size = tokenizer.vocab_size
        self.model_type = model_type
        self.mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # inference parameters
        self.val_generation_freq   = kwargs.get("val_generation_freq", 5)
        self.generation_block_size = kwargs.get("generation_block_size", block_size)
        self.val_temperatures       = kwargs.get("val_temperatures", [1.0])
        self.generation_num_steps  = kwargs.get("generation_num_steps", block_size // 2)
        self.sampling_repo         = kwargs.get("sampling_repo", "SMDM")  # Default to SMDM

        self.unk_token_id = tokenizer.unk_token_id  # unknown token id that avoid loss computation

        self.validation_samples_table = wandb.Table(
            columns=["Epoch", "Step", "Temperature", "Sample 1", "Sample 2", "Sample 3"],
            log_mode="MUTABLE"
        )

        # Initialize model based on model_type
        if model_type == "smdm":
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
            return self.model(x)
        elif self.model_type == "litgpt":
            # GPT expects (batch_size, seq_len) and returns (batch_size, seq_len, vocab_size)
            # For diffusion models, we don't need KV cache (input_pos=None)
            # This makes it behave like TransEncoder - process full sequence at once
            return self.model(x, input_pos=None)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def forward_process(self, batch, eps=1e-3):
        """
        Forward diffusion process: randomly mask tokens with the mask token.

        :param batch: Input tokens of shape (batch_size, seq_len)
        :param eps: Minimum masking probability to avoid fully unmasked sequences

        :returns noisy_batch: Tokens with some positions masked
        :returns mask_indices: Boolean tensor indicating which positions were masked
        :returns p_mask: Masking probabilities used
        """
        b, l = batch.shape
        t = torch.rand((b,), device=batch.device)

        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
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

        noisy_tokens, mask_indices, p_mask = self.forward_process(input_ids)
        logits = self.model(noisy_tokens)

        valid_tokens = input_ids != self.unk_token_id
        loss_mask    = mask_indices & valid_tokens

        # implement the re-weighted loss from SMDM
        loss = F.cross_entropy(logits[loss_mask], input_ids[loss_mask], reduction='none')
        loss = loss / p_mask[loss_mask]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        return loss

    def on_validation_epoch_end(self):
        """Generate samples and log to wandb at end of validation epoch."""
        # Only generate after training has started (skip epoch 0)
        if self.val_generation_freq and self.trainer.current_epoch > 0 and self.trainer.current_epoch % self.val_generation_freq == 0:
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
                import traceback
                traceback.print_exc()

    def sample(self, batch_size=1, seq_len=None, num_steps=None, temperature=1.0, repo="SMDM"):
        """Call inference functions borrowed from SMDM's & LLaDA's inference.py

        :param batch_size: Number of sequences to generate
        :param seq_len: Length of sequences to generate (defaults to model's block_size)
        :param num_steps: Number of denoising steps (defaults to seq_len // 2)
        :param temperature: Sampling temperature (0.0 for greedy)
        :param repo: Sampling method to use ('SMDM' or 'LLaDA')
        :return tokens: Generated sequences of shape (batch_size, seq_len)
        """
        device = next(self.parameters()).device

        if repo == "SMDM":
            from tzd.models.smdm.inference import diff_sample as smdm_sample
            
            output_tokens = smdm_sample(
                model=self.model,
                tokenizer=None,
                prompt=None,
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
            # TODO: we currently focus only on unconditional generation
            # LLaDA's generate() only supports batch_size=1, so we loop
            empty_prompt = torch.empty((1, 0), dtype=torch.long).to(device)

            samples = []
            for _ in range(batch_size):
                sample = llada_sample(
                    model=self.model,
                    prompt=empty_prompt,
                    steps=num_steps or self.generation_num_steps,
                    gen_length=seq_len or self.generation_block_size,
                    block_length=seq_len or self.generation_block_size, # TODO: we do sampling without using semi-autoregressive decoding
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
