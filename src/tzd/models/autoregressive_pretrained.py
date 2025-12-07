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
    **kwargs
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
    
    # Check for injected config
    if "litgpt_config" in kwargs:
        config = kwargs["litgpt_config"]
    else:
        checkpoint_dir = Path(checkpoint_dir)
        config = Config.from_file(checkpoint_dir / "model_config.yaml")
    
    # CRITICAL: Set causal=True for AR models
    config.causal = True
    print(f"  Causal attention: {config.causal}")
    
    # Update block size if needed
    config.block_size = max(block_size, generation_block_size)
    
    # Determine if using LoRA
    lora_r = kwargs.get("lora_r", 0)
    use_lora = lora_r > 0
    quantize = kwargs.get("quantize", None)
    
    ModelClass = GPT
    
    # LoRA Logic
    if use_lora: 
        print(f"✓ LoRA enabled (r={lora_r})")
        from litgpt.lora import GPT as LoRAGPT
        from litgpt.lora import Config as LoRAConfig
        from litgpt.lora import mark_only_lora_as_trainable
        
        config_dict = config.__dict__.copy()
        lora_keys = [k for k in kwargs if k.startswith("lora_")]
        for k in lora_keys:
            config_dict[k] = kwargs[k]
        
        if "lora_r" not in config_dict: config_dict["lora_r"] = lora_r
        if "lora_alpha" not in config_dict: config_dict["lora_alpha"] = kwargs.get("lora_alpha", 1)
        if "lora_dropout" not in config_dict: config_dict["lora_dropout"] = kwargs.get("lora_dropout", 0.0)
        
        # Re-create config as LoRA config
        # Use direct instantiation to avoid file lookups during tests
        valid_keys = LoRAConfig.__dataclass_fields__.keys()
        lora_config_args = {k:v for k,v in config_dict.items() if k in valid_keys}
        config = LoRAConfig(**lora_config_args)
        
        # Reset overrides
        config.causal = True
        config.block_size = max(block_size, generation_block_size)
    
        ModelClass = LoRAGPT

    # Quantization Logic
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
             raise ValueError(f"Unsupported quantization: {quantize}")
             
        with fabric.init_module():
            model = ModelClass(config)
    else:
        # Initialize model
        # with torch.device("meta"):
        model = ModelClass(config)
        
    # Load weights
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading pretrained weights from: {checkpoint_path}")
    
    # Use standard torch load
    state_dict = torch.load(checkpoint_path, mmap=True, map_location="cpu")
    
    strict_load = not use_lora
    model.load_state_dict(state_dict, strict=strict_load)
    
    if use_lora:
         print("✓ LoRA initialized, base weights loaded")
         mark_only_lora_as_trainable(model)
         print("✓ Base model frozen")
    
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
