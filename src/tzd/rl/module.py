"""
RL Training Module for Autoregressive and Diffusion Models.
Supports:
1. GRPO (Group Relative Policy Optimization) for AR models (TinyZero style).
2. SPG (Sandwiched Policy Gradient) / Diffusion-GRPO for Diffusion models.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tzd.models.base import BaseModel
from tzd.rl.rewards import countdown_reward_batch
from tzd.utils.reward_logging import compute_reward_metrics, log_reward_metrics
from tzd.data.countdown import CountdownDataset


class RLModule(L.LightningModule):
    """
    LightningModule for RL training of AR and Diffusion Models.
    
    Modes:
    - 'grpo': For AR models. Uses exact log-likelihoods, PPO clipping, and advantage normalization.
    - 'spg': For Diffusion models. Uses ELBO estimates, PPO clipping, and advantage normalization.
    """

    def __init__(
        self,
        model: BaseModel,
        lr: float = 1e-6,
        num_generations: int = 4,  # G in GRPO paper
        beta: float = 0.01,  # KL penalty coefficient
        use_ref_model: bool = True,
        generation_kwargs: Optional[Dict] = None,
        train_dataset: Optional[Any] = None, # Passed from config/hydra
        elbo_samples: int = 1, # Number of MC samples for ELBO (Diffusion only)
        rl_method: str = "grpo", # 'grpo' or 'spg'
        clip_eps: float = 0.2, # PPO clipping epsilon
        compile_model: bool = False, # Whether to use torch.compile
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_dataset"])
        
        self.model = model
        self.train_dataset = train_dataset

        if compile_model:
            print("Compiling model with torch.compile...")
            # Compile the inner neural network (LitGPT or TransEncoder)
            # This ensures both forward() and compute_elbo() use the compiled model
            self.model.model = torch.compile(self.model.model)
        
        # Reference model for KL penalty (Critical for stability)
        self.ref_model = None
        if use_ref_model:
            print("Initializing Reference Model...")
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
        else:
            print("WARNING: No reference model used. Training may be unstable!")

        self.lr = lr
        self.num_generations = num_generations
        self.beta = beta
        self.elbo_samples = elbo_samples
        self.rl_method = rl_method.lower()
        self.clip_eps = clip_eps
        self.generation_kwargs = generation_kwargs or {}
        
        # Default generation settings
        self.gen_cfg = {
            "num_steps": 64,
            "temperature": 1.0,
            "cfg_scale": 1.0,
        }
        self.gen_cfg.update(self.generation_kwargs)
        
        print(f"RL Module initialized with method: {self.rl_method}")

    @property
    def model_alias(self):
        return self.model.model_alias

    def get_num_params(self):
        return self.model.get_num_params()

    def setup(self, stage: str):
        """Setup steps (e.g. moving ref_model to device)."""
        if self.ref_model is not None:
            self.ref_model.to(self.device)

    def forward(self, x):
        """Forward pass just delegates to underlying model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        RL Training Step (GRPO/SPG).
        """
        # 1. Unpack batch
        prompts_text = batch["prompts"]
        targets = batch["targets"]
        numbers = batch["numbers"]
        
        batch_size = len(prompts_text)
        
        # 2. Rollout (Generation)
        # Repeat prompts G times
        repeated_prompts_text = []
        repeated_targets = []
        repeated_numbers = []
        
        for i in range(batch_size):
            repeated_prompts_text.extend([prompts_text[i]] * self.num_generations)
            repeated_targets.extend([targets[i]] * self.num_generations)
            repeated_numbers.extend([numbers[i]] * self.num_generations)
            
        # Tokenize prompts
        tokenizer = self.model.tokenizer
        encoded_prompts = tokenizer(
            repeated_prompts_text,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        
        prompt_ids = encoded_prompts.input_ids
        prompt_len = prompt_ids.shape[1]
        
        # Generate samples (No Gradients)
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(
                prompts=prompt_ids,
                batch_size=len(repeated_prompts_text),
                seq_len=self.model.generation_block_size,
                num_steps=self.gen_cfg["num_steps"],
                temperature=self.gen_cfg["temperature"]
            )
            
            # Compute OLD log probs / ELBO
            # Return full sequence of per-token log probs
            if self.rl_method == "grpo":
                old_log_probs = self.model.compute_elbo(
                    samples,
                    prompt_len=prompt_len,
                    return_per_token=True
                )
            else:
                # SPG: Use blockwise (Sandwich) estimator if available, else random masking
                if hasattr(self.model, "compute_elbo_blockwise"):
                    old_log_probs = self.model.compute_elbo_blockwise(
                        samples,
                        num_samples=self.elbo_samples,
                        eps=1e-3,
                        prompt_len=prompt_len
                    )
                else:
                    old_log_probs = self.model.compute_elbo(
                        samples,
                        num_samples=self.elbo_samples,
                        eps=1e-3,
                        prompt_len=prompt_len
                    )
            
        self.model.train()
        
        # 3. Compute Rewards
        completion_ids = samples[:, prompt_len:]
        decoded_samples = [
            tokenizer.decode(s, skip_special_tokens=True) 
            for s in completion_ids
        ]
        
        rewards = countdown_reward_batch(decoded_samples, repeated_targets, repeated_numbers)
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        
        metrics = compute_reward_metrics(decoded_samples, repeated_targets, repeated_numbers)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=True, on_step=True, on_epoch=True)
        
        # 4. Compute Advantages
        advantages = self.compute_advantages(rewards_tensor, batch_size, self.num_generations)
        self.log("train/advantage_mean", advantages.mean(), on_step=True, on_epoch=True)
        self.log("train/advantage_std", advantages.std(), on_step=True, on_epoch=True)

        # 5. Compute Policy Loss (New Log Probs)
        if self.rl_method == "grpo":
             new_log_probs = self.model.compute_elbo(
                samples,
                prompt_len=prompt_len,
                return_per_token=True
            )
        else:
            if hasattr(self.model, "compute_elbo_blockwise"):
                new_log_probs = self.model.compute_elbo_blockwise(
                    samples,
                    num_samples=self.elbo_samples,
                    eps=1e-3,
                    prompt_len=prompt_len
                )
            else:
                new_log_probs = self.model.compute_elbo(
                    samples,
                    num_samples=self.elbo_samples,
                    eps=1e-3,
                    prompt_len=prompt_len
                )
        
        # 6. KL Penalty
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None and self.beta > 0:
            with torch.no_grad():
                if self.rl_method == "grpo":
                    ref_log_probs = self.ref_model.compute_elbo(
                        samples, 
                        prompt_len=prompt_len,
                        return_per_token=True
                    )
                else:
                    # Use same estimator for ref model
                    if hasattr(self.ref_model, "compute_elbo_blockwise"):
                        ref_log_probs = self.ref_model.compute_elbo_blockwise(
                            samples,
                            num_samples=self.elbo_samples,
                            prompt_len=prompt_len
                        )
                    else:
                        ref_log_probs = self.ref_model.compute_elbo(
                            samples,
                            num_samples=self.elbo_samples,
                            prompt_len=prompt_len
                        )
            
            # PPO KL: Per-token difference
            kld = new_log_probs - ref_log_probs # Shape: [B, Seq]
            
            if self.rl_method == "grpo":
                # Mask out padding/prompt is handled by compute_elbo returning 0 for prompt
                # But we valid mask is needed for averaging
                # For AR, compute_elbo should return valid log probs for completions
                # Just take mean over valid tokens
                # kld is [B, Completion_Len]
                kl_loss = self.beta * kld.sum(dim=1).mean() # Sum over seq, mean over batch? No.
                # TinyZero does: kld per token. masked_mean(kld).
                # We will trust our compute_elbo to return aligned shapes
                kl_loss = self.beta * kld.mean()
            else:
                 kl_loss = self.beta * kld.mean()

            self.log("train/kl_divergence", kld.mean(), on_step=True, on_epoch=True)
            
        # 7. PPO Loss Calculation
        if self.rl_method == "grpo": # Token-Level PPO (TinyZero Style)
            # new_log_probs: [B, Completion_Len]
            # old_log_probs: [B, Completion_Len]
            # advantages: [B] -> Broadcast to [B, Completion_Len]
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Broadcast advantage [B] -> [B, 1]
            adv_expanded = advantages.view(-1, 1).expand_as(ratio)
            adv_detached = adv_expanded.detach()
            
            surr1 = ratio * adv_detached
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_detached
            
            # Per-token min
            clipped_loss = torch.min(surr1, surr2)
            
            # Masking is implicit if log_probs are only returned for completion
            # Assuming compute_elbo returns only completion part or handles masking
            policy_loss = -clipped_loss.mean()
            
        else: # Sequence-Level PPO (Diffusion)
            ratio = torch.exp(new_log_probs - old_log_probs)
            adv_detached = advantages.detach()
            surr1 = ratio * adv_detached
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_detached
            policy_loss = -torch.min(surr1, surr2).mean()
        
        total_loss = policy_loss + kl_loss
        
        self.log("train/policy_loss", policy_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
        
        return total_loss

    @staticmethod
    def compute_advantages(rewards: torch.Tensor, batch_size: int, num_generations: int) -> torch.Tensor:
        """
        Compute group-relative advantages with Z-score normalization.

        Args:
            rewards: Tensor of shape (batch_size * num_generations,)
            batch_size: Number of unique prompts
            num_generations: Number of generations per prompt

        Returns:
            advantages: Tensor of shape (batch_size * num_generations,)
        """
        # Reshape to (batch_size, num_generations)
        rewards_reshaped = rewards.view(batch_size, num_generations)

        # Baseline = Mean within group
        baseline = rewards_reshaped.mean(dim=1, keepdim=True)
        
        # Standard Deviation within group
        std = rewards_reshaped.std(dim=1, keepdim=True)
        
        # Advantage = (Reward - Baseline) / (Std + eps)
        # Add small epsilon to avoid division by zero
        advantages = (rewards_reshaped - baseline) / (std + 1e-8)
        
        return advantages.view(-1)  # Flatten

    def configure_optimizers(self):
        """Configure optimizer."""
        # Use lower LR for RL
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer

    def train_dataloader(self):
        """
        Return dataloader for RL training.
        If train_dataset is provided (via config), use it.
        Otherwise, we expect the main trainer to handle it (but RL usually needs custom collation).
        """
        if self.train_dataset:
            return DataLoader(
                self.train_dataset,
                batch_size=1, # We handle batching via gradient accumulation or config
                shuffle=True,
                collate_fn=lambda batch: CountdownDataset.collate_fn(batch, tokenizer=self.model.tokenizer)
            )
        return None

    def validation_step(self, batch, batch_idx):
        """
        Validation step: Compute rewards on validation data.
        """
        # 1. Unpack batch
        prompts_text = batch["prompts"]
        targets = batch["targets"]
        numbers = batch["numbers"]

        # 2. Rollout (Generation)
        repeated_prompts_text = []
        repeated_targets = []
        repeated_numbers = []

        for i in range(len(prompts_text)):
            repeated_prompts_text.extend([prompts_text[i]] * self.num_generations)
            repeated_targets.extend([targets[i]] * self.num_generations)
            repeated_numbers.extend([numbers[i]] * self.num_generations)

        tokenizer = self.model.tokenizer
        encoded_prompts = tokenizer(
            repeated_prompts_text,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        prompt_ids = encoded_prompts.input_ids

        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(
                prompts=prompt_ids,
                batch_size=len(repeated_prompts_text),
                seq_len=self.model.generation_block_size,
                num_steps=self.gen_cfg["num_steps"],
                temperature=self.gen_cfg["temperature"]
            )

        # 3. Compute Rewards
        # Decode only the completion
        prompt_len = prompt_ids.shape[1]
        completion_ids = samples[:, prompt_len:]
        
        decoded_samples = [
            tokenizer.decode(s, skip_special_tokens=True) 
            for s in completion_ids
        ]

        # Log metrics
        metrics = compute_reward_metrics(decoded_samples, repeated_targets, repeated_numbers)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)

        # 4. Log sample generations (3 to WandB, 1 to console)
        # Only log on first batch to avoid too much logging
        if batch_idx == 0:
            # Print 1 example to console (matching TinyZero behavior)
            print(f"\n{'='*80}")
            print(f"Validation Sample (Step {self.trainer.global_step}):")
            print(f"Prompt: {repeated_prompts_text[0]}")
            print(f"Generated: {decoded_samples[0]}")
            print(f"Target: {repeated_targets[0]}")
            print(f"{'='*80}\n")

            # Log 3 examples to WandB table
            if self.logger:
                import wandb
                table_data = []
                for i in range(min(3, len(decoded_samples))):
                    table_data.append([
                        self.trainer.global_step,
                        repeated_prompts_text[i],
                        decoded_samples[i],
                        repeated_targets[i],
                        repeated_numbers[i]
                    ])

                table = wandb.Table(
                    columns=["step", "prompt", "generated", "target", "numbers"],
                    data=table_data
                )
                self.logger.experiment.log({f"val/samples": table})

        return metrics
