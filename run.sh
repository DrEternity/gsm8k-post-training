#!/bin/bash
#SBATCH --job-name=qwen2.5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ─── Conda ───────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-alignment

# ─── Environment ─────────────────────────────────────────
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6
export TRITON_CACHE_DIR=/tmp/triton_cache

# ─── Run ─────────────────────────────────────────────────
cd ~/llm_education/project3

# Stage 0: Baseline evaluation
# PYTHONUNBUFFERED=1 python -m scripts.eval_base_gsm8k

# Stage 1: SFT data preparation
# PYTHONUNBUFFERED=1 python -m scripts.prepare_sft_data

# Stage 2: SFT training
# PYTHONUNBUFFERED=1 python -m scripts.sft_train \
#     --lora_r 32 \
#     --num_epochs 1 \
#     --batch_size 8 \
#     --grad_accum 4 \
#     --lr 2e-4 \
#     --no_wandb

# Stage 2: SFT evaluation
# PYTHONUNBUFFERED=1 python -m scripts.eval_sft \
#     --checkpoints_dir results/sft_gsm8k/r32

# Stage 3: Data filtering for GRPO
# PYTHONUNBUFFERED=1 python -m scripts.filter_grpo \
#     --adapter_path results/sft_gsm8k/r32/checkpoint-878

# Stage 4: GRPO training
# PYTHONUNBUFFERED=1 python -m scripts.grpo_train \
#     --adapter_path results/sft_gsm8k/r32/checkpoint-878 \
#     --loss_type dr_grpo \
#     --lora_r 32 \
#     --num_generations 16 \
#     --batch_size 8 \
#     --grad_accum 4 \
#     --lr 5e-6 \
#     --num_epochs 5 \
#     --no_wandb

# Stage 4: GRPO evaluation
# PYTHONUNBUFFERED=1 python -m scripts.eval_grpo --methods dapo dr_grpo

# Reference models evaluation
# PYTHONUNBUFFERED=1 python -m scripts.eval_instruct
# PYTHONUNBUFFERED=1 python -m scripts.eval_instruct --model Qwen/Qwen2.5-Math-1.5B-Instruct

# Self-consistency evaluation
# PYTHONUNBUFFERED=1 python -m scripts.eval_sc \
#     --sft_adapter results/sft_gsm8k/r32/checkpoint-878 \
#     --adapter_path results/grpo/dapo_g16/checkpoint-2760 \
#     --num_samples 8

PYTHONUNBUFFERED=1 python -m scripts.eval_grpo --methods dapo dr_grpo