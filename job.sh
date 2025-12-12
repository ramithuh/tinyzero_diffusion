#!/bin/bash
#SBATCH --job-name=limit_test
#SBATCH --partition=debug
#SBATCH --gres=gpu:gb10:1
#SBATCH --mem=100G
#SBATCH --time=12:35:00

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tzd

# Run the eater (which tries to eat 12GB+)
cd /home/ruh/ramith/academics/10703/tinyzero_diffusion
~/safe_run.sh python train.py --config-name config_rl_countdown
