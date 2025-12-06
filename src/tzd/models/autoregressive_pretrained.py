import torch
from pathlib import Path
from typing import Optional

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import load_checkpoint
import lightning as L

from tzd.models.autoregressive import AutoregressiveModel

def from_pretrained(
    pretrained_model_name: str,
    checkpoint_dir: str,
    tokenizer,
    model_alias: str = "ar_model",
    lr: float = 1e-4,
    block_size: int = 1024,
    generation_block_size: int = 1024,
) -> AutoregressiveModel:
    """
    Load a pretrained LitGPT model and wrap it in AutoregressiveModel.
    
    Args:
        pretrained_model_name: Name of the model (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
        checkpoint_dir: Path to the checkpoint directory
        tokenizer: Initialized tokenizer
        model_alias: Alias for the model
        lr: Learning rate
        block_size: Max sequence length for training
        generation_block_size: Max sequence length for generation
        
    Returns:
        Initialized AutoregressiveModel
    """
    print(f"Creating AutoregressiveModel from {pretrained_model_name}")
    
    checkpoint_dir = Path(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    
    # CRITICAL: Set causal=True for AR models
    config.causal = True
    print(f"  Causal attention: {config.causal}")
    
    # Update block size if needed
    config.block_size = max(block_size, generation_block_size)
    
    # Initialize model
    # with torch.device("meta"):
    model = GPT(config)
        
    # Load weights
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading pretrained weights from: {checkpoint_path}")
    
    # Use standard torch load instead of Fabric to avoid conflict with Trainer
    state_dict = torch.load(checkpoint_path, mmap=True, map_location="cpu")
    model.load_state_dict(state_dict)
    
    print("✓ Weights loaded")
    
    # Wrap in AutoregressiveModel
    ar_model = AutoregressiveModel(
        model=model,
        tokenizer=tokenizer,
        model_alias=model_alias,
        lr=lr,
        block_size=block_size,
        generation_block_size=generation_block_size
    )
    
    return ar_model
