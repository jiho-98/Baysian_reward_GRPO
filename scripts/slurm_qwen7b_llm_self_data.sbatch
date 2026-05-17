#!/bin/bash
#
# Submit from the cloud GPU server:
#   ssh -p 27468 kimjh@14.38.98.203
#   cd /home/kimjh/EMNLP
#   sbatch scripts/slurm_qwen7b_llm_self_data.sbatch
#
# The SSH password is intentionally not stored in this file.

#SBATCH --job-name=emnlp-qwen7b-selfdata
#SBATCH --output=/home/kimjh/EMNLP/logs/%x_%j.out
#SBATCH --error=/home/kimjh/EMNLP/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

cd /home/kimjh/EMNLP
mkdir -p logs results hf_cache

# Use the physical GPU that appears as ID 3 on nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

export HF_HOME=/home/kimjh/EMNLP/hf_cache
export TRANSFORMERS_CACHE=/home/kimjh/EMNLP/hf_cache/transformers
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export MAX_MODEL_CALLS=${MAX_MODEL_CALLS:-200}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
export PROBLEM_GENERATOR_MAX_NEW_TOKENS=${PROBLEM_GENERATOR_MAX_NEW_TOKENS:-4096}
export ANALYZER_MAX_NEW_TOKENS=${ANALYZER_MAX_NEW_TOKENS:-1536}
export SOLVER_MAX_NEW_TOKENS=${SOLVER_MAX_NEW_TOKENS:-512}
export DIRECT_BASELINE_MAX_NEW_TOKENS=${DIRECT_BASELINE_MAX_NEW_TOKENS:-512}
export PROGRESS_LOG=${PROGRESS_LOG:-1}

source /home/kimjh/EMNLP/.venv/bin/activate

echo "=== START ==="
date
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "QWEN_MODEL=${QWEN_MODEL}"
nvidia-smi
nvidia-smi -i 3
python - <<'EOF'
import torch
print("torch import OK")
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible cuda device 0:", torch.cuda.get_device_name(0))
EOF

srun python llm_self_data_bayesian_experiment.py
