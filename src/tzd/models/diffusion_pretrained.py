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
from litgpt.model import GPT
from tzd.models.diffusion import DiffusionModel


def from_pretrained(
    pretrained_model_name: str,
    tokenizer: callable,
    model_alias: str = "diffusion_from_pretrained",
    lr: float = 1e-5,
    checkpoint_dir: Optional[Path] = None,
    litgpt_config: Optional[LitGPTConfig] = None,
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
    # (Leaving debug prints as they are helpful)
    # 1. Set the Anchor (BOS)
    if not tokenizer.bos_token:
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # 2. Set the Padding (PAD)
    if not tokenizer.pad_token:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    if not tokenizer.mask_token:
        tokenizer.mask_token = "<|endoftext|>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print(f"[CONFIG] BOS ID: {tokenizer.bos_token_id}")
    print(f"[CONFIG] PAD ID: {tokenizer.pad_token_id}")
    print(f"[CONFIG] MASK ID: {tokenizer.mask_token_id}")

    # Load LitGPT config by name if not provided
    if litgpt_config is None:
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

    # Update vocab size in config if tokenizer was extended
    if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size != litgpt_config.vocab_size:
        print(f"[INFO] Tokenizer vocab size ({tokenizer.vocab_size}) differs from config ({litgpt_config.vocab_size})")
        litgpt_config.vocab_size = tokenizer.vocab_size
        litgpt_config.padded_vocab_size = tokenizer.vocab_size

    # Filter kwargs for DiffusionModel
    explicit_params = {'model_alias', 'lr', 'n_layer', 'n_head', 'n_embd', 'block_size', 
                       'tokenizer', 'bias', 'model_type', 'litgpt_config'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_params}
    
    # Determine if using LoRA
    lora_r = kwargs.get("lora_r", 0)
    use_lora = lora_r > 0
    quantize = kwargs.get("quantize", None)
    
    gpt_model = None
    ModelClass = GPT
    model_config_to_use = litgpt_config

    if use_lora:
        print(f"✓ LoRA enabled (r={lora_r})")
        from litgpt.lora import GPT as LoRAGPT
        from litgpt.lora import Config as LoRAConfig
        from litgpt.lora import mark_only_lora_as_trainable
        
        # Convert base config to LoRA config manually
        config_dict = litgpt_config.__dict__.copy()
        
        # Add LoRA params from kwargs
        lora_keys = [k for k in kwargs if k.startswith("lora_")]
        for k in lora_keys:
            config_dict[k] = kwargs[k]
            
        if "lora_r" not in config_dict: config_dict["lora_r"] = lora_r
        if "lora_alpha" not in config_dict: config_dict["lora_alpha"] = kwargs.get("lora_alpha", 1)
        if "lora_dropout" not in config_dict: config_dict["lora_dropout"] = kwargs.get("lora_dropout", 0.0)
        
        # Instantiate LoRAConfig directly from dict
        valid_keys = LoRAConfig.__dataclass_fields__.keys()
        lora_config_args = {k:v for k,v in config_dict.items() if k in valid_keys}
        lora_config = LoRAConfig(**lora_config_args)
        
        lora_config.causal = False
        ModelClass = LoRAGPT
        model_config_to_use = lora_config

    # Handle Quantization with Fabric
    if quantize:
        print(f"✓ Quantization enabled: {quantize}")
        import lightning as L
        from lightning.fabric.plugins import BitsandbytesPrecision
        
        if quantize.startswith("bnb."):
            mode = quantize[4:]
            dtype = torch.bfloat16 
            plugin = BitsandbytesPrecision(mode=mode, dtype=dtype)
            fabric = L.Fabric(plugins=plugin, accelerator="cuda", devices=1)
        else:
            raise ValueError(f"Unsupported quantization mode: {quantize}")
            
        with fabric.init_module():
            gpt_model = ModelClass(model_config_to_use)
    else:
        # Standard initialization
        gpt_model = ModelClass(model_config_to_use)

    # Load weights
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / "lit_model.pth"
        if not checkpoint_path.exists():
             raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading pretrained weights from: {checkpoint_path}")
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
        
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
            
        strict_load = not use_lora
        gpt_model.load_state_dict(state_dict, strict=strict_load)
        
        if use_lora:
            print(f"✓ LoRA weights initialized")
            mark_only_lora_as_trainable(gpt_model)
            print("✓ Base model frozen")
        else:
            print(f"✓ Weights loaded (strict={strict_load})")
    else:
        print("⚠ No checkpoint loaded (Random weights)")
    
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
        litgpt_config=litgpt_config, 
        gpt_model=gpt_model,
        **filtered_kwargs
    )
    
    return model
