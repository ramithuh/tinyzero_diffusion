# TZD - Tiny Zero Diffusion [WIP]

A minimal diffusion language model framework built with Hydra + PyTorch Lightning. This repo provides a clean implementation of discrete diffusion models for text generation.

## Features

- **Discrete Diffusion Models**: Uses SMDM (Simple Masked Diffusion Model) and LitGPT-based diffusion (https://github.com/ramithuh/litgpt_diffusion/tree/diffusion)
- **Flexible Backends**: Switch between SMDM and LitGPT model architectures
- **Hydra Configuration**: Easy experiment management with composable configs
- **PyTorch Lightning**: Clean training loop with distributed training support
- **WandB Integration**: Automatic logging of metrics and generated samples

## Installation

```bash
# Clone the repo with submodules
git clone --recurse-submodules https://github.com/ramithuh/tinyzero_diffusion.git
cd tinyzero_diffusion

# Install litgpt_diffusion submodule (custom fork with diffusion modifications)
pip install -e litgpt_diffusion/

# Install main package
pip install -e .
```

**Note:** The `litgpt_diffusion` submodule is automatically checked out to the `diffusion` branch, which includes custom modifications for bidirectional attention (`causal=False`) and fused RoPE optimization.

## Quick Start

Train a diffusion model on TinyShakespeare:

```bash
# Run training
python train.py

# Or use the training script
bash train_diffusion.sh
```

## Project Structure

```
tinyzero_diffusion/
├── src/tzd/                    # Main package
│   ├── models/                 # Model implementations
│   │   ├── diffusion.py        # Main diffusion model
│   │   ├── smdm/               # SMDM backend
│   │   ├── llada/              # LLaDA inference
│   │   └── litgpt_diffusion/   # LitGPT backend
│   ├── data/                   # Data modules
│   │   └── datamodule.py       # TinyShakespeare datamodule
│   └── utils/                  # Utilities
│       └── generation.py       # Sampling and logging
├── configs/                    # Hydra configs
│   ├── model/                  # Model configs
│   ├── data/                   # Data configs
│   ├── training/               # Training configs
│   └── tokenizer/              # Tokenizer configs
├── train.py                    # Main training script
└── pyproject.toml              # Package dependencies
```

## Configuration

The project uses Hydra for configuration management. Main config groups:

- `configs/model/diffusion.yaml` - Model architecture (layers, heads, embedding size)
- `configs/data/shakespeare.yaml` - Dataset configuration
- `configs/training/default.yaml` - Training hyperparameters
- `configs/tokenizer/llama.yaml` - Tokenizer settings

Override configs via command line:
```bash
python train.py model.n_layer=6 model.n_embd=256 training.epochs=50
```

## Model Backends

### SMDM (Simple Masked Diffusion Model)
- Non-causal Transformer encoder
- Discrete diffusion with masking

### LitGPT
- GPT architecture with `causal=False`
- More flexible MLP options (GptNeoxMLP, LLaMAMLP, etc.)
- Rotary positional embeddings

Switch backends in config:
```yaml
model:
  model_type: litgpt  # or 'smdm'
```

## Citation
