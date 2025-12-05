"""Generation utilities for sampling and logging model outputs."""

from typing import List, Optional

import torch
import lightning as L
import wandb
from tzd.utils.reward_logging import compute_reward_metrics, log_reward_metrics


def generate_samples(
    model: L.LightningModule,
    batch_size: int,
    seq_len: int,
    num_steps: int = None,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    prompts: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate samples from the model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        # Use the model's configured sampling method (LLaDA or SMDM)
        repo = getattr(model, 'sampling_repo', 'LLaDA')  # Default to LLaDA (SMDM requires rotary_emb)
        samples = model.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            temperature=temperature,
            repo=repo,
            prompts=prompts
        )

    return samples


def decode_samples(samples: torch.Tensor, datamodule) -> List[str]:
    """Decode token samples to text using the datamodule."""
    sample_texts = []
    for sample in samples:
        sample_text = datamodule.decode(sample)
        sample_texts.append(sample_text)
    return sample_texts

def log_generations(
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule,
    epoch: int,
    step: int,
    temperatures: Optional[List[float]] = None,
    num_samples: int = 3,
    seq_len: int = None,
    num_steps: int = None,
    wandb_table=None,
    stage="unset"
) -> None:
    """
    Generate samples during validation and log them to WandB.
    
    Args:
        trainer: Lightning trainer instance
        model: The trained model
        datamodule: Data module for decoding
        epoch: Current epoch number (for logging)
        step: Current training step
        temperatures: List of temperatures to test
        num_samples: Number of samples per temperature
        seq_len: Length of sequences to generate
        num_steps: Number of diffusion steps for generation
        wandb_table: WandB table to add rows to
        stage: Stage name for logging (e.g., "validation", "test")
    """
    if temperatures is None:
        temperatures = [1.0]  # Default to single temperature for validation

    if trainer.logger is None:
        print("Warning: Trainer logger is None, skipping logging.")
        return

    print(f"DEBUG: Starting log_generations for epoch {epoch}, step {step}")

    # Get prompts from datamodule if available
    prompts = None
    prompt_texts = [""] * num_samples
    sample_indices = []
    if hasattr(datamodule, "get_sample_prompt"):
        # Get random prompts from validation set
        import random
        val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
        
        # Tokenize prompts
        # We need to manually tokenize because datamodule.tokenizer is the object, not a function
        tokenizer = datamodule.tokenizer
        encoded_prompts = []
        
        # Max allowed prompt length (leaving room for generation)
        # Assuming generation_block_size is 1024, let's limit prompt to ~256 tokens
        MAX_PROMPT_LEN = 256
        
        for i in range(num_samples):
            # Try up to 10 times to find a short enough prompt
            for attempt in range(10):
                if val_size > 0:
                    idx = random.randint(0, val_size - 1)
                    p = datamodule.get_sample_prompt(split="val", idx=idx)
                else:
                    idx = -1
                    p = datamodule.get_sample_prompt()
                
                tokens = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids
                
                if tokens.size(1) <= MAX_PROMPT_LEN:
                    prompt_texts[i] = p
                    encoded_prompts.append(tokens)
                    sample_indices.append(idx)
                    break
            else:
                # If we failed to find a short prompt after 10 tries, just truncate the last one
                # This is a fallback to prevent crashing
                print(f"Warning: Could not find short prompt after 10 attempts. Truncating.")
                tokens = tokens[:, :MAX_PROMPT_LEN]
                encoded_prompts.append(tokens)
                # Update text to match truncated tokens
                prompt_texts[i] = tokenizer.decode(tokens[0], skip_special_tokens=True)

        # Pad prompts to same length if necessary
        max_len = max(t.size(1) for t in encoded_prompts)
        padded_prompts = torch.full((num_samples, max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
        for i, t in enumerate(encoded_prompts):
            padded_prompts[i, :t.size(1)] = t
        
        prompts = padded_prompts.to(next(model.parameters()).device)

    all_sample_texts = []
    # Log samples for each temperature
    for temp in temperatures:
        # Generate samples
        samples = generate_samples(
            model,
            batch_size=num_samples,
            seq_len=seq_len,
            num_steps=num_steps,
            temperature=temp,
            device=next(model.parameters()).device,
            prompts=prompts
        )

        # Convert samples to text for logging
        sample_texts = decode_samples(samples, datamodule)
        all_sample_texts.extend(sample_texts)

        # Add row to the shared table
        # Ensure we have exactly 3 samples, pad with empty strings if needed
        padded_samples = (sample_texts + [""] * 3)[:3]
        
        # If we have prompts, prepend them to the log or include them
        # The current table structure is (epoch, step, temp, sample1, sample2, sample3)
        # We can modify the sample text to include the prompt: "Prompt: ... \n Answer: ..."
        if prompts is not None:
            padded_samples = [f"Prompt: {p}\n\nGen: {s}" for p, s in zip(prompt_texts, padded_samples)]

        wandb_table.add_data(epoch, step, temp, padded_samples[0], padded_samples[1], padded_samples[2])

    # Compute and log rewards if possible
    if hasattr(datamodule, 'get_validation_ground_truth') and sample_indices:
        try:
            # Get ground truth for the samples we generated
            targets, numbers = datamodule.get_validation_ground_truth(sample_indices)
            
            # Repeat for each temperature (since all_sample_texts contains samples for all temps)
            all_targets = targets * len(temperatures)
            all_numbers = numbers * len(temperatures)
            
            # Compute metrics
            metrics = compute_reward_metrics(
                all_sample_texts,
                all_targets,
                all_numbers
            )

            # Log to WandB
            log_reward_metrics(
                metrics,
                epoch,
                step,
                stage,
                trainer.logger
            )

        except Exception as e:
            print(f"Warning: Could not compute rewards: {e}")

    print(f"DEBUG: Logging table to WandB with {len(all_sample_texts)} samples")
    # Log the table ONCE after all temperatures are processed
    avg_length = sum(len(text.split()) for text in all_sample_texts) / len(all_sample_texts)
    trainer.logger.experiment.log({
        f"{stage}_samples": wandb_table,
        f"{stage}_avg_length": avg_length,
        "epoch": epoch,
        "step": step,
    })
    print("DEBUG: log_generations completed successfully")