"""
RL Training Module for Diffusion Models using GRPO.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tzd.models.diffusion import DiffusionModel
from tzd.rl.rewards import countdown_reward_batch
from tzd.utils.reward_logging import compute_reward_metrics, log_reward_metrics
from tzd.data.countdown import CountdownDataset


class RLDiffusionModule(L.LightningModule):
    """
    LightningModule for RL training of Diffusion Models using GRPO.
    
    Implements:
    1. Rollout: Generate samples from prompts
    2. Reward: Score samples using task-specific reward function
    3. Advantage: Compute group-relative advantages (GRPO)
    4. Loss: Policy gradient using ELBO as log-likelihood + KL Penalty
    """

    def __init__(
        self,
        model: DiffusionModel,
        lr: float = 1e-6,
        num_generations: int = 4,  # G in GRPO paper
        beta: float = 0.01,  # KL penalty coefficient
        use_ref_model: bool = True,
        generation_kwargs: Optional[Dict] = None,
        train_dataset: Optional[Any] = None, # Passed from config/hydra
        elbo_samples: int = 1, # Number of MC samples for ELBO
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_dataset"])
        
        self.model = model
        self.train_dataset = train_dataset
        
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
        self.generation_kwargs = generation_kwargs or {}
        
        # Default generation settings
        self.gen_cfg = {
            "num_steps": 64,
            "temperature": 1.0,
            "cfg_scale": 1.0,
        }
        self.gen_cfg.update(self.generation_kwargs)

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
        GRPO Training Step.
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
            
        self.model.train()
        
        # 3. Compute Rewards
        decoded_samples = [
            tokenizer.decode(s, skip_special_tokens=True) 
            for s in samples
        ]
        
        rewards = countdown_reward_batch(decoded_samples, repeated_targets, repeated_numbers)
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        
        # Log metrics
        metrics = compute_reward_metrics(decoded_samples, repeated_targets, repeated_numbers)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=True)
        
        # 4. Compute Advantages (Group-Relative)
        # Reshape to (batch_size, num_generations)
        rewards_reshaped = rewards_tensor.view(batch_size, self.num_generations)
        
        # Baseline = Mean within group
        baseline = rewards_reshaped.mean(dim=1, keepdim=True)
        
        # Advantage = Reward - Baseline (NO Z-score normalization!)
        advantages = rewards_reshaped - baseline
        advantages = advantages.view(-1) # Flatten
        
        self.log("train/advantage_mean", advantages.mean())

        # 5. Compute Policy Loss (ELBO)
        # We compute ELBO of the *generated samples*
        # CRITICAL: samples contains PROMPT + COMPLETION, but we only optimize COMPLETION
        prompt_len = prompt_ids.shape[1]  # Length of prompt tokens

        elbo = self.model.compute_elbo(
            samples,
            num_samples=self.elbo_samples,
            eps=1e-3,
            prompt_len=prompt_len  # Only compute loss on completion tokens!
        )
        
        # 6. KL Penalty (Critical)
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None and self.beta > 0:
            with torch.no_grad():
                ref_elbo = self.ref_model.compute_elbo(
                    samples,
                    num_samples=self.elbo_samples,
                    prompt_len=prompt_len  # Same prompt_len for reference model!
                )
            
            # KL approx = log_p - log_ref = elbo - ref_elbo
            # We want to minimize KL, so we add beta * KL to loss
            # Note: Since we maximize ELBO (log_p), we maximize (elbo - beta * KL)
            # Or minimize -(elbo - beta * KL)
            # Let's stick to loss minimization:
            # Loss = - (Advantage * ELBO) + beta * KL
            
            kld = elbo - ref_elbo
            kl_loss = self.beta * kld.mean()
            self.log("train/kl_divergence", kld.mean())
            
        # Total Loss
        # We detach advantages to treat them as weights
        policy_loss = -(advantages.detach() * elbo).mean()
        total_loss = policy_loss + kl_loss
        
        self.log("train/policy_loss", policy_loss)
        self.log("train/total_loss", total_loss)
        
        return total_loss

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
        
        # 2. Rollout (Generation) - Just 1 generation per prompt for validation to save time?
        # Or use same num_generations to be consistent. Let's use 1 for speed, or num_generations.
        # Let's use num_generations to get better estimate of policy performance.
        
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
        decoded_samples = [
            tokenizer.decode(s, skip_special_tokens=True) 
            for s in samples
        ]
        
        # Log metrics
        metrics = compute_reward_metrics(decoded_samples, repeated_targets, repeated_numbers)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)
        
        return metrics
