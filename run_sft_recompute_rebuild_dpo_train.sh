#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   chmod +x run_full_sft_dpo_grpo_pipeline.sh
#   CUDA_VISIBLE_DEVICES=3 nohup bash run_full_sft_dpo_grpo_pipeline.sh \
#     --experiment_name gsm8k_sft_dpo_grpo_v1 \
#     > logs/run_full_sft_dpo_grpo_v1.out 2>&1 &
#
# Dry run:
#   CUDA_VISIBLE_DEVICES=3 bash run_full_sft_dpo_grpo_pipeline.sh \
#     --dry_run --experiment_name dryrun_full_pipeline

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k_full_sft_dpo_grpo_pipeline_${TIMESTAMP}}"
DRY_RUN=0

print_help() {
  cat <<'EOF'
Usage:
  bash run_full_sft_dpo_grpo_pipeline.sh [--dry_run] [--experiment_name NAME]

Pipeline:
  1. Train SFT analyzer
  2. Recompute posterior with SFT analyzer
  3. Rebuild DPO data with real rejected from SFT recompute
  4. Train DPO analyzer from SFT checkpoint
  5. Train/evaluate Bayesian GRPO with SFT analyzer
  6. Train/evaluate Bayesian GRPO with SFT+DPO analyzer

Options:
  --dry_run
      Print commands and resolved paths without running jobs.
  --experiment_name NAME
      Override default timestamped experiment name.
  --help
      Show this help message.

Important environment overrides:
  CUDA_VISIBLE_DEVICES
  MODEL_NAME
  ANALYZER_MODEL_NAME
  PROMPTED_RUN_DIR
  PROMPTED_REWARD_DEBUG_JSONL
  SFT_DATASET_DIR
  SFT_DATASET_VARIANT
  METADATA_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --experiment_name)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --experiment_name requires a value" >&2
        exit 1
      fi
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

# =========================
# Base paths
# =========================

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
ANALYZER_MODEL_NAME="${ANALYZER_MODEL_NAME:-${MODEL_NAME}}"

PROMPTED_RUN_DIR="${PROMPTED_RUN_DIR:-outputs/gsm8k_experiments/grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10}"
PROMPTED_REWARD_DEBUG_JSONL="${PROMPTED_REWARD_DEBUG_JSONL:-${PROMPTED_RUN_DIR}/bayesian_reward_debug.jsonl}"

SFT_DATASET_DIR="${SFT_DATASET_DIR:-outputs/gsm8k_learned_analyzer/sft_data}"
SFT_DATASET_VARIANT="${SFT_DATASET_VARIANT:-runtime}"

PIPELINE_BASE_DIR="${PIPELINE_BASE_DIR:-outputs/gsm8k_learned_analyzer/pipelines}"
GRPO_OUTPUT_BASE_DIR="${GRPO_OUTPUT_BASE_DIR:-outputs/gsm8k_experiments}"
#LOGS_BASE_DIR="${LOGS_BASE_DIR:-logs}"

PIPELINE_ROOT="${PIPELINE_BASE_DIR}/${EXPERIMENT_NAME}"
STEP_LOG_DIR="${PIPELINE_ROOT}/step_logs"

SFT_ADAPTER_DIR="${PIPELINE_ROOT}/sft_adapter"
SFT_RECOMPUTE_DIR="${PIPELINE_ROOT}/sft_recompute_on_prompted_pool"
DPO_DATA_DIR="${PIPELINE_ROOT}/dpo_data_from_sft_recompute"
DPO_ADAPTER_DIR="${PIPELINE_ROOT}/dpo_adapter"

GRPO_SFT_OUTPUT_DIR="${GRPO_OUTPUT_BASE_DIR}/grpo_bayesian_sft_analyzer_${EXPERIMENT_NAME}"
GRPO_DPO_OUTPUT_DIR="${GRPO_OUTPUT_BASE_DIR}/grpo_bayesian_sft_dpo_analyzer_${EXPERIMENT_NAME}"

PIPELINE_SUMMARY_PATH="${PIPELINE_ROOT}/pipeline_paths.txt"

mkdir -p "${PIPELINE_ROOT}" "${STEP_LOG_DIR}" "${GRPO_OUTPUT_BASE_DIR}"

# =========================
# SFT hyperparameters
# =========================

SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-4096}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1.0}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-4}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-200}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-200}"
SFT_SEED="${SFT_SEED:-42}"
SFT_BF16="${SFT_BF16:-1}"
SFT_USE_4BIT="${SFT_USE_4BIT:-0}"

# =========================
# Recompute hyperparameters
# =========================

RECOMPUTE_BATCH_SIZE="${RECOMPUTE_BATCH_SIZE:-8}"
RECOMPUTE_MAX_NEW_TOKENS="${RECOMPUTE_MAX_NEW_TOKENS:-512}"
RECOMPUTE_BF16="${RECOMPUTE_BF16:-1}"

# =========================
# DPO data + training hyperparameters
# =========================

DPO_TARGET_EVIDENCE_TRAIN_PAIRS="${DPO_TARGET_EVIDENCE_TRAIN_PAIRS:-4000}"
DPO_TARGET_PRIOR_TRAIN_PAIRS="${DPO_TARGET_PRIOR_TRAIN_PAIRS:-1000}"
DPO_TARGET_EVIDENCE_VALID_PAIRS="${DPO_TARGET_EVIDENCE_VALID_PAIRS:-400}"
DPO_TARGET_PRIOR_VALID_PAIRS="${DPO_TARGET_PRIOR_VALID_PAIRS:-100}"
DPO_SYNTHETIC_COMPANION_RATE="${DPO_SYNTHETIC_COMPANION_RATE:-0.5}"

DPO_BETA="${DPO_BETA:-0.1}"
DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-4096}"
DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-3584}"
DPO_MAX_COMPLETION_LENGTH="${DPO_MAX_COMPLETION_LENGTH:-512}"
DPO_NUM_TRAIN_EPOCHS="${DPO_NUM_TRAIN_EPOCHS:-1.0}"
DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
DPO_PER_DEVICE_EVAL_BATCH_SIZE="${DPO_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-8}"
DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-5}"
DPO_LOGGING_STEPS="${DPO_LOGGING_STEPS:-10}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-200}"
DPO_EVAL_STEPS="${DPO_EVAL_STEPS:-200}"
DPO_SEED="${DPO_SEED:-42}"
DPO_BF16="${DPO_BF16:-1}"
DPO_USE_4BIT="${DPO_USE_4BIT:-0}"

# =========================
# GRPO hyperparameters
# =========================

METADATA_DIR="${METADATA_DIR:-outputs/gsm8k_experiments/metadata}"
GRPO_TRAIN_SIZE="${GRPO_TRAIN_SIZE:-3000}"
GRPO_EVAL_SIZE="${GRPO_EVAL_SIZE:-500}"
GRPO_MIN_SOLVE_RATE="${GRPO_MIN_SOLVE_RATE:-0.0}"
GRPO_MAX_SOLVE_RATE="${GRPO_MAX_SOLVE_RATE:-1.0}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-1024}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-500}"
GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1024}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-0.7}"
GRPO_TOP_P="${GRPO_TOP_P:-0.95}"
GRPO_PER_DEVICE_TRAIN_BATCH_SIZE="${GRPO_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRPO_GRADIENT_ACCUMULATION_STEPS="${GRPO_GRADIENT_ACCUMULATION_STEPS:-8}"
GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-5e-6}"
GRPO_LOGGING_STEPS="${GRPO_LOGGING_STEPS:-10}"
GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-100}"
GRPO_SEED="${GRPO_SEED:-42}"
GRPO_PRIOR_LAMBDA="${GRPO_PRIOR_LAMBDA:-1.0}"
GRPO_PRIOR_SOFTMAX_TEMPERATURE="${GRPO_PRIOR_SOFTMAX_TEMPERATURE:-1.0}"
GRPO_JUDGE_MAX_NEW_TOKENS="${GRPO_JUDGE_MAX_NEW_TOKENS:-768}"
GRPO_PROGRESS_INTERVAL_PERCENT="${GRPO_PROGRESS_INTERVAL_PERCENT:-5}"
GRPO_BF16="${GRPO_BF16:-1}"

GRPO_EVAL_BATCH_SIZE="${GRPO_EVAL_BATCH_SIZE:-16}"
GRPO_EVAL_MAX_NEW_TOKENS="${GRPO_EVAL_MAX_NEW_TOKENS:-1024}"
GRPO_EVAL_MAX_PROMPT_LENGTH="${GRPO_EVAL_MAX_PROMPT_LENGTH:-1024}"

GRPO_PREFLIGHT_BATCH_SIZE="${GRPO_PREFLIGHT_BATCH_SIZE:-8}"
GRPO_PREFLIGHT_MAX_NEW_TOKENS="${GRPO_PREFLIGHT_MAX_NEW_TOKENS:-512}"

# =========================
# Helpers
# =========================

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing required directory: ${path}" >&2
    exit 1
  fi
}

print_cmd() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
}

run_step() {
  local step_name="$1"
  shift
  local logfile="${STEP_LOG_DIR}/${step_name}.log"

  echo "[INFO] step=${step_name}"
  echo "[INFO] logfile=${logfile}"
  print_cmd "$@"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "$@" >"${logfile}" 2>&1
  echo "[INFO] completed=${step_name}"
}

# =========================
# Preflight checks
# =========================

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[INFO] experiment_name=${EXPERIMENT_NAME}"
echo "[INFO] pipeline_root=${PIPELINE_ROOT}"
echo "[INFO] step_log_dir=${STEP_LOG_DIR}"
echo "[INFO] dry_run=${DRY_RUN}"

require_dir "${PROMPTED_RUN_DIR}"
require_dir "${SFT_DATASET_DIR}"
require_file "${PROMPTED_REWARD_DEBUG_JSONL}"
require_file "${SFT_DATASET_DIR}/${SFT_DATASET_VARIANT}_unified_train.jsonl"
require_file "${SFT_DATASET_DIR}/${SFT_DATASET_VARIANT}_unified_valid.jsonl"

# =========================
# 1. SFT analyzer train
# =========================

SFT_CMD=(
  python3
  train_analyzer_sft.py
  --dataset_dir "${SFT_DATASET_DIR}"
  --dataset_variant "${SFT_DATASET_VARIANT}"
  --model_name "${MODEL_NAME}"
  --output_dir "${SFT_ADAPTER_DIR}"
  --max_length "${SFT_MAX_LENGTH}"
  --num_train_epochs "${SFT_NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${SFT_PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${SFT_LEARNING_RATE}"
  --logging_steps "${SFT_LOGGING_STEPS}"
  --save_steps "${SFT_SAVE_STEPS}"
  --eval_steps "${SFT_EVAL_STEPS}"
  --seed "${SFT_SEED}"
)
if [[ "${SFT_BF16}" == "1" ]]; then
  SFT_CMD+=(--bf16)
fi
if [[ "${SFT_USE_4BIT}" == "1" ]]; then
  SFT_CMD+=(--use_4bit)
fi

# =========================
# 2. Recompute with SFT analyzer
# =========================

SFT_RECOMPUTE_CMD=(
  python3
  recompute_posterior_with_learned_analyzer.py
  --input_debug_jsonl "${PROMPTED_REWARD_DEBUG_JSONL}"
  --output_dir "${SFT_RECOMPUTE_DIR}"
  --model_name "${ANALYZER_MODEL_NAME}"
  --adapter_path "${SFT_ADAPTER_DIR}"
  --batch_size "${RECOMPUTE_BATCH_SIZE}"
  --max_new_tokens "${RECOMPUTE_MAX_NEW_TOKENS}"
)
if [[ "${RECOMPUTE_BF16}" == "1" ]]; then
  SFT_RECOMPUTE_CMD+=(--bf16)
fi

# =========================
# 3. Rebuild DPO with real rejected
# =========================

REBUILD_DPO_CMD=(
  python3
  build_analyzer_dpo_data_from_gsm8k_logs.py
  --log_dir "${PROMPTED_RUN_DIR}"
  --output_dir "${DPO_DATA_DIR}"
  --learned_posterior_debug_jsonl "${SFT_RECOMPUTE_DIR}/learned_posterior_debug.jsonl"
  --target_evidence_train_pairs "${DPO_TARGET_EVIDENCE_TRAIN_PAIRS}"
  --target_prior_train_pairs "${DPO_TARGET_PRIOR_TRAIN_PAIRS}"
  --target_evidence_valid_pairs "${DPO_TARGET_EVIDENCE_VALID_PAIRS}"
  --target_prior_valid_pairs "${DPO_TARGET_PRIOR_VALID_PAIRS}"
  --synthetic_companion_rate "${DPO_SYNTHETIC_COMPANION_RATE}"
  --seed "${DPO_SEED}"
)

# =========================
# 4. DPO analyzer train from SFT checkpoint
# =========================

DPO_CMD=(
  python3
  train_analyzer_dpo.py
  --dataset_dir "${DPO_DATA_DIR}"
  --dataset_variant "${SFT_DATASET_VARIANT}"
  --model_name "${MODEL_NAME}"
  --init_adapter_path "${SFT_ADAPTER_DIR}"
  --reference_adapter_path "${SFT_ADAPTER_DIR}"
  --output_dir "${DPO_ADAPTER_DIR}"
  --beta "${DPO_BETA}"
  --max_length "${DPO_MAX_LENGTH}"
  --max_prompt_length "${DPO_MAX_PROMPT_LENGTH}"
  --max_completion_length "${DPO_MAX_COMPLETION_LENGTH}"
  --num_train_epochs "${DPO_NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${DPO_PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${DPO_PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${DPO_GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${DPO_LEARNING_RATE}"
  --logging_steps "${DPO_LOGGING_STEPS}"
  --save_steps "${DPO_SAVE_STEPS}"
  --eval_steps "${DPO_EVAL_STEPS}"
  --seed "${DPO_SEED}"
)
if [[ "${DPO_BF16}" == "1" ]]; then
  DPO_CMD+=(--bf16)
fi
if [[ "${DPO_USE_4BIT}" == "1" ]]; then
  DPO_CMD+=(--use_4bit)
fi

# =========================
# 5. Bayesian GRPO with SFT analyzer
# =========================

GRPO_SFT_CMD=(
  python3
  run_grpo_bayesian_with_learned_analyzer.py
  --analyzer_adapter_path "${SFT_ADAPTER_DIR}"
  --output_dir "${GRPO_SFT_OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --analyzer_model_name "${ANALYZER_MODEL_NAME}"
  --metadata_dir "${METADATA_DIR}"
  --train_size "${GRPO_TRAIN_SIZE}"
  --eval_size "${GRPO_EVAL_SIZE}"
  --min_solve_rate "${GRPO_MIN_SOLVE_RATE}"
  --max_solve_rate "${GRPO_MAX_SOLVE_RATE}"
  --num_generations "${GRPO_NUM_GENERATIONS}"
  --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}"
  --max_steps "${GRPO_MAX_STEPS}"
  --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}"
  --temperature "${GRPO_TEMPERATURE}"
  --top_p "${GRPO_TOP_P}"
  --per_device_train_batch_size "${GRPO_PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRPO_GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${GRPO_LEARNING_RATE}"
  --logging_steps "${GRPO_LOGGING_STEPS}"
  --save_steps "${GRPO_SAVE_STEPS}"
  --seed "${GRPO_SEED}"
  --prior_lambda "${GRPO_PRIOR_LAMBDA}"
  --prior_softmax_temperature "${GRPO_PRIOR_SOFTMAX_TEMPERATURE}"
  --judge_max_new_tokens "${GRPO_JUDGE_MAX_NEW_TOKENS}"
  --progress_interval_percent "${GRPO_PROGRESS_INTERVAL_PERCENT}"
  --eval_batch_size "${GRPO_EVAL_BATCH_SIZE}"
  --eval_max_new_tokens "${GRPO_EVAL_MAX_NEW_TOKENS}"
  --eval_max_prompt_length "${GRPO_EVAL_MAX_PROMPT_LENGTH}"
  --preflight_recompute_log_dir "${PROMPTED_RUN_DIR}"
  --preflight_recompute_output_dir "${GRPO_SFT_OUTPUT_DIR}/preflight_recompute"
  --preflight_recompute_batch_size "${GRPO_PREFLIGHT_BATCH_SIZE}"
  --preflight_recompute_max_new_tokens "${GRPO_PREFLIGHT_MAX_NEW_TOKENS}"
)
if [[ "${GRPO_BF16}" == "1" ]]; then
  GRPO_SFT_CMD+=(--bf16)
else
  GRPO_SFT_CMD+=(--no_bf16)
fi

# =========================
# 6. Bayesian GRPO with SFT+DPO analyzer
# =========================

GRPO_DPO_CMD=(
  python3
  run_grpo_bayesian_with_learned_analyzer.py
  --analyzer_adapter_path "${DPO_ADAPTER_DIR}"
  --output_dir "${GRPO_DPO_OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --analyzer_model_name "${ANALYZER_MODEL_NAME}"
  --metadata_dir "${METADATA_DIR}"
  --train_size "${GRPO_TRAIN_SIZE}"
  --eval_size "${GRPO_EVAL_SIZE}"
  --min_solve_rate "${GRPO_MIN_SOLVE_RATE}"
  --max_solve_rate "${GRPO_MAX_SOLVE_RATE}"
  --num_generations "${GRPO_NUM_GENERATIONS}"
  --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}"
  --max_steps "${GRPO_MAX_STEPS}"
  --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}"
  --temperature "${GRPO_TEMPERATURE}"
  --top_p "${GRPO_TOP_P}"
  --per_device_train_batch_size "${GRPO_PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRPO_GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${GRPO_LEARNING_RATE}"
  --logging_steps "${GRPO_LOGGING_STEPS}"
  --save_steps "${GRPO_SAVE_STEPS}"
  --seed "${GRPO_SEED}"
  --prior_lambda "${GRPO_PRIOR_LAMBDA}"
  --prior_softmax_temperature "${GRPO_PRIOR_SOFTMAX_TEMPERATURE}"
  --judge_max_new_tokens "${GRPO_JUDGE_MAX_NEW_TOKENS}"
  --progress_interval_percent "${GRPO_PROGRESS_INTERVAL_PERCENT}"
  --eval_batch_size "${GRPO_EVAL_BATCH_SIZE}"
  --eval_max_new_tokens "${GRPO_EVAL_MAX_NEW_TOKENS}"
  --eval_max_prompt_length "${GRPO_EVAL_MAX_PROMPT_LENGTH}"
  --preflight_recompute_log_dir "${PROMPTED_RUN_DIR}"
  --preflight_recompute_output_dir "${GRPO_DPO_OUTPUT_DIR}/preflight_recompute"
  --preflight_recompute_batch_size "${GRPO_PREFLIGHT_BATCH_SIZE}"
  --preflight_recompute_max_new_tokens "${GRPO_PREFLIGHT_MAX_NEW_TOKENS}"
)
if [[ "${GRPO_BF16}" == "1" ]]; then
  GRPO_DPO_CMD+=(--bf16)
else
  GRPO_DPO_CMD+=(--no_bf16)
fi

# =========================
# Run all steps
# =========================

run_step "01_train_sft" "${SFT_CMD[@]}"
run_step "02_recompute_with_sft" "${SFT_RECOMPUTE_CMD[@]}"
run_step "03_rebuild_dpo_from_sft_recompute" "${REBUILD_DPO_CMD[@]}"
run_step "04_train_dpo" "${DPO_CMD[@]}"
run_step "05_grpo_with_sft_analyzer" "${GRPO_SFT_CMD[@]}"
run_step "06_grpo_with_sft_dpo_analyzer" "${GRPO_DPO_CMD[@]}"

cat >"${PIPELINE_SUMMARY_PATH}" <<EOF
EXPERIMENT_NAME=${EXPERIMENT_NAME}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

PIPELINE_ROOT=${PIPELINE_ROOT}
STEP_LOG_DIR=${STEP_LOG_DIR}

SFT_ADAPTER_DIR=${SFT_ADAPTER_DIR}
SFT_RECOMPUTE_DEBUG_JSONL=${SFT_RECOMPUTE_DIR}/learned_posterior_debug.jsonl
SFT_RECOMPUTE_SUMMARY_JSON=${SFT_RECOMPUTE_DIR}/summary.json

DPO_DATASET_DIR=${DPO_DATA_DIR}
DPO_RUNTIME_TRAIN_JSONL=${DPO_DATA_DIR}/${SFT_DATASET_VARIANT}_unified_train.jsonl
DPO_RUNTIME_VALID_JSONL=${DPO_DATA_DIR}/${SFT_DATASET_VARIANT}_unified_valid.jsonl
DPO_DATASET_SUMMARY_JSON=${DPO_DATA_DIR}/summary.json
DPO_ADAPTER_DIR=${DPO_ADAPTER_DIR}

GRPO_SFT_OUTPUT_DIR=${GRPO_SFT_OUTPUT_DIR}
GRPO_SFT_PREFLIGHT_RECOMPUTE_DIR=${GRPO_SFT_OUTPUT_DIR}/preflight_recompute

GRPO_DPO_OUTPUT_DIR=${GRPO_DPO_OUTPUT_DIR}
GRPO_DPO_PREFLIGHT_RECOMPUTE_DIR=${GRPO_DPO_OUTPUT_DIR}/preflight_recompute
EOF

echo "[INFO] pipeline summary written to ${PIPELINE_SUMMARY_PATH}"
echo "[INFO] final outputs:"
echo "SFT adapter: ${SFT_ADAPTER_DIR}"
echo "SFT recompute summary: ${SFT_RECOMPUTE_DIR}/summary.json"
echo "DPO data summary: ${DPO_DATA_DIR}/summary.json"
echo "SFT+DPO adapter: ${DPO_ADAPTER_DIR}"
echo "GRPO with SFT analyzer: ${GRPO_SFT_OUTPUT_DIR}"
echo "GRPO with SFT+DPO analyzer: ${GRPO_DPO_OUTPUT_DIR}"
echo "[INFO] DONE"