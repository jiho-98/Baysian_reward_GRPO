#!/bin/bash
#
# Slurm template for running the minimal belief demo with a local HF model.
# Do not submit before confirming your lab's partition/account/GPU policy.
#
#SBATCH --job-name=emnlp-qwen-belief
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# TODO: Ask the lab/server admin which values to use here.
#SBATCH --partition=GPU_PARTITION_NAME
##SBATCH --account=ACCOUNT_NAME
##SBATCH --qos=QOS_NAME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

cd /home/kimjh/EMNLP
mkdir -p logs results hf_cache

# Keep model/cache files inside the project unless your cluster has a preferred cache path.
export HF_HOME=/home/kimjh/EMNLP/hf_cache
export TRANSFORMERS_CACHE=/home/kimjh/EMNLP/hf_cache/transformers
export TOKENIZERS_PARALLELISM=false

# Optional: uncomment and edit if your cluster uses modules.
# module purge
# module load cuda/12.1

# This venv reuses the existing torch/transformers installation via .venv/.../assignments_uv.pth.
source /home/kimjh/EMNLP/.venv/bin/activate

python scripts/minimal_belief_demo.py \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --max-new-tokens 160 \
  --output results/minimal_belief_qwen3_4b.json
