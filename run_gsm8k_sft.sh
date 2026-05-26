#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
METADATA_DIR="${METADATA_DIR:-outputs/gsm8k_3000_500_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm8k_experiments/sft_qwen3b_train3000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"

SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1.0}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-4}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-200}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-200}"
USE_4BIT="${USE_4BIT:-0}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required file/path: ${path}" >&2
    exit 1
  fi
}

require_file "train_unified_analyzer_sft.py"
require_file "eval_solver_checkpoint.py"
require_file "${METADATA_DIR}/sft_train_messages.jsonl"
require_file "${METADATA_DIR}/sft_valid_messages.jsonl"
require_file "${METADATA_DIR}/selected_valid_metadata.jsonl"
require_file "${METADATA_DIR}/selected_test_metadata.jsonl"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/valid" "${OUTPUT_DIR}/test"

train_cmd=(
  python3 train_unified_analyzer_sft.py
  --model_name "${MODEL_NAME}"
  --train_path "${METADATA_DIR}/sft_train_messages.jsonl"
  --val_path "${METADATA_DIR}/sft_valid_messages.jsonl"
  --output_dir "${OUTPUT_DIR}"
  --max_length "${SFT_MAX_LENGTH}"
  --num_train_epochs "${SFT_NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${SFT_PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${SFT_LEARNING_RATE}"
  --logging_steps "${SFT_LOGGING_STEPS}"
  --save_steps "${SFT_SAVE_STEPS}"
  --eval_steps "${SFT_EVAL_STEPS}"
  --seed "${SEED}"
  --bf16
)

if [[ "${USE_4BIT}" == "1" ]]; then
  train_cmd+=(--use_4bit)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 eval_solver_checkpoint.py \
  --model_name "${MODEL_NAME}" \
  --adapter_path "${OUTPUT_DIR}" \
  --eval_metadata_path "${METADATA_DIR}/selected_valid_metadata.jsonl" \
  --output_dir "${OUTPUT_DIR}/valid" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --seed "${SEED}" \
  --no_do_sample

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 eval_solver_checkpoint.py \
  --model_name "${MODEL_NAME}" \
  --adapter_path "${OUTPUT_DIR}" \
  --eval_metadata_path "${METADATA_DIR}/selected_test_metadata.jsonl" \
  --output_dir "${OUTPUT_DIR}/test" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --seed "${SEED}" \
  --no_do_sample

echo "[DONE] GSM8K SFT train/eval saved under ${OUTPUT_DIR}"
