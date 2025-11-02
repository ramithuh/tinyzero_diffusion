"""Generation utilities for sampling and logging model outputs."""

from typing import List, Optional

import torch
import lightning as L
import wandb


def generate_samples(
    model: L.LightningModule,
    batch_size: int,
    seq_len: int,
    num_steps: int = None,
    temperature: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Generate samples from the model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        # Use the model's configured sampling method (LLaDA or SMDM)
        repo = getattr(model, 'sampling_repo', 'SMDM')
        samples = model.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            temperature=temperature,
            repo=repo
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
        return

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
            device=next(model.parameters()).device
        )

        # Convert samples to text for logging
        sample_texts = decode_samples(samples, datamodule)
        all_sample_texts.extend(sample_texts)

        # Add row to the shared table
        # Ensure we have exactly 3 samples, pad with empty strings if needed
        padded_samples = (sample_texts + [""] * 3)[:3]
        wandb_table.add_data(epoch, step, temp, padded_samples[0], padded_samples[1], padded_samples[2])

    # Log the table ONCE after all temperatures are processed
    avg_length = sum(len(text.split()) for text in all_sample_texts) / len(all_sample_texts)
    trainer.logger.experiment.log({
        f"{stage}_samples": wandb_table,
        f"{stage}_avg_length": avg_length,
        "epoch": epoch,
        "step": step,
    })