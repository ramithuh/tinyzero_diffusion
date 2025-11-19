"""Helper functions for loading pretrained checkpoints into DiffusionModel.

This module provides utilities to:
1. Load pretrained model configs from LitGPT's registry
2. Convert HuggingFace weights to LitGPT format
3. Initialize DiffusionModel with pretrained weights and causal=False
"""

import torch
from pathlib import Path
from typing import Optional
from litgpt.config import Config as LitGPTConfig
from tzd.models.diffusion import DiffusionModel


def from_pretrained(
    pretrained_model_name: str,
    tokenizer: callable,
    model_alias: str = "diffusion_from_pretrained",
    lr: float = 1e-5,
    checkpoint_dir: Optional[Path] = None,
    **kwargs
) -> DiffusionModel:
    """
    Load a pretrained model and convert to a diffusion model.

    This function uses LitGPT's config registry to automatically populate
    all architecture parameters from well-known model names like "Qwen2.5-3B".

    Args:
        pretrained_model_name: Model name recognized by LitGPT
                              (e.g., "Qwen2.5-3B", "Llama-3-8B", "phi-2")
                              Run `litgpt download list` to see all supported models
        tokenizer: Tokenizer instance
        model_alias: Name for this model instance
        lr: Learning rate
        checkpoint_dir: Optional path to pre-downloaded LitGPT checkpoint
                       If None, will load from HuggingFace directly
        **kwargs: Override any config parameters (e.g., block_size=256)

    Returns:
        DiffusionModel with pretrained architecture (and weights if checkpoint_dir provided)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>>
        >>> # Option 1: Load config only (random weights)
        >>> model = from_pretrained(
        ...     "Qwen2.5-3B",
        ...     tokenizer=tokenizer,
        ...     block_size=256,  # Override for shorter sequences
        ... )
        >>>
        >>> # Option 2: Load config + weights from downloaded checkpoint
        >>> model = from_pretrained(
        ...     "Qwen2.5-3B",
        ...     tokenizer=tokenizer,
        ...     checkpoint_dir=Path("checkpoints/Qwen/Qwen2.5-3B"),
        ... )
    """
    # Load LitGPT config by name
    try:
        litgpt_config = LitGPTConfig.from_name(pretrained_model_name)
        print(f"✓ Loaded LitGPT config for: {pretrained_model_name}")
    except (ValueError, StopIteration) as e:
        raise ValueError(
            f"Could not find LitGPT config for '{pretrained_model_name}'. "
            f"Available configs can be listed with: litgpt download list"
        ) from e

    # Apply user overrides
    for key, value in kwargs.items():
        if hasattr(litgpt_config, key):
            setattr(litgpt_config, key, value)

    # CRITICAL: Set causal=False for diffusion models
    litgpt_config.causal = False

    print(f"\n{'='*60}")
    print(f"Creating DiffusionModel from {pretrained_model_name}")
    print(f"{'='*60}")
    print(f"Architecture:")
    print(f"  Layers: {litgpt_config.n_layer}")
    print(f"  Attention heads: {litgpt_config.n_head}")
    print(f"  Embedding dim: {litgpt_config.n_embd}")
    print(f"  Vocab size: {litgpt_config.vocab_size}")
    print(f"  Block size: {litgpt_config.block_size}")
    print(f"  Normalization: {litgpt_config.norm_class_name}")
    print(f"  MLP: {litgpt_config.mlp_class_name}")
    print(f"  Intermediate size: {litgpt_config.intermediate_size}")
    print(f"Diffusion settings:")
    print(f"  Causal attention: {litgpt_config.causal} (non-causal for diffusion)")
    print(f"{'='*60}\n")

    # Create DiffusionModel with LitGPT architecture
    # We pass the full litgpt_config in kwargs, which contains all architecture details
    # The individual parameters are needed to satisfy DiffusionModel's __init__ signature
    model = DiffusionModel(
        model_alias=model_alias,
        lr=lr,
        n_layer=litgpt_config.n_layer,
        n_head=litgpt_config.n_head,
        n_embd=litgpt_config.n_embd,
        block_size=litgpt_config.block_size,
        tokenizer=tokenizer,
        bias=litgpt_config.bias,
        model_type="litgpt",
        litgpt_config=litgpt_config,  # This is what actually gets used for the GPT model
    )

    # Load pretrained weights if checkpoint directory provided
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / "lit_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download it first using:\n"
                f"  litgpt download {pretrained_model_name}"
            )

        print(f"Loading pretrained weights from: {checkpoint_path}")
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

        # Handle both raw state dict and checkpoint dict with 'model' key
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"⚠ Missing keys: {len(missing)} (first 3: {', '.join(missing[:3])})")
        if unexpected:
            print(f"⚠ Unexpected keys: {len(unexpected)} (first 3: {', '.join(unexpected[:3])})")

        loaded = len(model.model.state_dict()) - len(missing)
        print(f"✓ Loaded {loaded}/{len(model.model.state_dict())} weight tensors\n")
    else:
        print(f"⚠ No checkpoint_dir provided - initialized with random weights")
        print(f"   To download: litgpt download {pretrained_model_name}\n")

    return model
