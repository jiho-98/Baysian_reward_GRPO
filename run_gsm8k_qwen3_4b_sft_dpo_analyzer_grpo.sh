#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Qwen3-4B learned analyzer pipeline matched to the prompted Bayesian GRPO
# baseline config, changing only the analyzer source to SFT+DPO.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export RUN_BASE=0
export RUN_ANSWER_ONLY=0
export RUN_BAYESIAN_PROMPTED=0
export RUN_ANALYZER_SFT=1
export RUN_SFT_ANALYZER_GRPO=0
export RUN_ANALYZER_DPO=1
export RUN_DPO_ANALYZER_GRPO=1
export RUN_COLLECT_RESULTS="${RUN_COLLECT_RESULTS:-0}"

export MODEL_NAME="Qwen/Qwen3-4B"
export ANALYZER_MODEL_NAME="Qwen/Qwen3-4B"
export BACKBONE_TAG="qwen3_4b"
export DATASET_KEY="gsm8k"
export GRPO_DATASET_NAME="fixed_metadata"
export METADATA_DIR="outputs/gsm8k_full_train_seed42"
export EXPERIMENT_ROOT="outputs/gsm8k_full_qwen3_4b"
export PROMPTED_OUTPUT_DIR="${EXPERIMENT_ROOT}/grpo_bayesian_prompted"

export DPO_GRPO_OUTPUT_DIR="${EXPERIMENT_ROOT}/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1"

export TRAIN_SIZE=7473
export EVAL_SIZE=0
export NUM_GENERATIONS=8
export MAX_STEPS=1500
export TRAIN_MAX_PROMPT_LENGTH=1024
export MAX_COMPLETION_LENGTH=1024
export TEMPERATURE=0.7
export TOP_P=0.95
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export LEARNING_RATE=5e-6
export SOLVER_USE_LORA=1
export SOLVER_GRADIENT_CHECKPOINTING=1
export SOLVER_LORA_R=16
export SOLVER_LORA_ALPHA=32
export SOLVER_LORA_DROPOUT=0.05
export LOGGING_STEPS=5
export SAVE_STEPS=250
export PROGRESS_INTERVAL_PERCENT=10
export PRIOR_LAMBDA=1.0
export PRIOR_SOFTMAX_TEMPERATURE=1.0
export JUDGE_MAX_NEW_TOKENS=768
export BF16=1
export SEED=42

export BASELINE_EVAL_BATCH_SIZE=32
export BASELINE_EVAL_MAX_NEW_TOKENS=1024
export BASELINE_EVAL_MAX_PROMPT_LENGTH=2048
export LEARNED_EVAL_BATCH_SIZE=32
export LEARNED_EVAL_MAX_NEW_TOKENS=1024
export LEARNED_EVAL_MAX_PROMPT_LENGTH=2048

export TRAIN_DATA_LABEL="GSM8K official train full"

bash run_full_baseline_experiments.sh \
  --dataset_key "${DATASET_KEY}" \
  --model_name "${MODEL_NAME}" \
  --analyzer_model_name "${ANALYZER_MODEL_NAME}" \
  --backbone_tag "${BACKBONE_TAG}" \
  --experiment_root "${EXPERIMENT_ROOT}" \
  "$@"
