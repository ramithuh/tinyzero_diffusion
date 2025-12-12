#!/bin/bash
#SBATCH --job-name=llada_rl
#SBATCH --partition=debug
#SBATCH --gres=gpu:gb10:1
#SBATCH --mem=100G
#SBATCH --time=12:35:00

export HYDRA_FULL_ERROR=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tzd

# Run with safe_run wrapper
cd /home/ruh/ramith/academics/10703/tinyzero_diffusion
~/safe_run.sh python train.py --config-name config_rl_llada
