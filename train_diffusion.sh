#!/bin/bash

# Simple training script for diffusion language model
# Usage: bash train_diffusion.sh

python train.py \
  experiment_suffix=diffusion_shakespeare \
  logger.project=diffusion_lm \
  logger.name=diffusion_shakespeare
