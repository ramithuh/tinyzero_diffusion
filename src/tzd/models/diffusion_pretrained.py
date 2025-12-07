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
    # DEBUG: Check what tokenizer actually is
    print(f"\n{'='*60}")
    print(f"DEBUG: from_pretrained() tokenizer analysis")
    print(f"{'='*60}")
    print(f"Type: {type(tokenizer)}")
    print(f"Value: {tokenizer}")
    if hasattr(tokenizer, 'vocab_size'):
        print(f"Has vocab_size: {tokenizer.vocab_size}")
    else:
        print(f"WARNING: tokenizer does not have vocab_size attribute!")
    print(f"{'='*60}\n")
    
    # 1. Set the Anchor (BOS)
    # Qwen doesn't set this by default, so we force it.
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>") # Should be 151644

    # 2. Set the Padding (PAD)
    # Your output shows this is already set, but being explicit never hurts.
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>") # Should be 151643

    tokenizer.mask_token = "<|endoftext|>"
    tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")    
    # # 3. Add the Noise Token (MASK)
    # # We add a new special token to avoid conflict with PAD/EOS
    # if "<|MASK|>" not in tokenizer.get_vocab():
    #     tokenizer.add_special_tokens({"additional_special_tokens": ["<|MASK|>"]})
        
    # # Force the attribute so we can find it easily later
    # tokenizer.mask_token = "<|MASK|>"
    # tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")

    print(f"[CONFIG] BOS ID: {tokenizer.bos_token_id} (<|im_start|>)")
    print(f"[CONFIG] PAD ID: {tokenizer.pad_token_id} (<|endoftext|>)")
    print(f"[CONFIG] MASK ID: {tokenizer.mask_token_id} {tokenizer.mask_token}")

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

    # Update vocab size in config if tokenizer was extended (e.g., added MASK token)
    if tokenizer.vocab_size != litgpt_config.vocab_size:
        print(f"[INFO] Tokenizer vocab size ({tokenizer.vocab_size}) differs from config ({litgpt_config.vocab_size})")
        print(f"[INFO] Updating config to match tokenizer (likely due to added special tokens)")
        litgpt_config.vocab_size = tokenizer.vocab_size
        litgpt_config.padded_vocab_size = tokenizer.vocab_size

    # Create DiffusionModel with LitGPT architecture
    # We pass the full litgpt_config in kwargs, which contains all architecture details
    # The individual parameters are needed to satisfy DiffusionModel's __init__ signature
    
    # Filter out kwargs that we're already passing explicitly to avoid "multiple values" errors
    explicit_params = {'model_alias', 'lr', 'n_layer', 'n_head', 'n_embd', 'block_size', 
                       'tokenizer', 'bias', 'model_type', 'litgpt_config'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_params}
    
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
        **filtered_kwargs
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
