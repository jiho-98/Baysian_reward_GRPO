#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
SKIP_VALID_EVAL="${SKIP_VALID_EVAL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --skip_valid_eval|--no_valid)
      SKIP_VALID_EVAL=1
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
DATASET_NAME="${DATASET_NAME:-fixed_metadata}"
METADATA_DIR="${METADATA_DIR:-outputs/gsm8k_3000_500_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm8k_experiments/grpo_answer_only_qwen3b_train3000_n8_steps500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
TRAIN_MAX_PROMPT_LENGTH="${TRAIN_MAX_PROMPT_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

TRAIN_SIZE="${TRAIN_SIZE:-3000}"
EVAL_SIZE="${EVAL_SIZE:-500}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
USE_LORA="${USE_LORA:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
PROGRESS_INTERVAL_PERCENT="${PROGRESS_INTERVAL_PERCENT:-10}"
MIN_SOLVE_RATE="${MIN_SOLVE_RATE:-0.0}"
MAX_SOLVE_RATE="${MAX_SOLVE_RATE:-1.0}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
VLLM_EVAL_GPU_MEMORY_UTILIZATION="${VLLM_EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LENGTH="${VLLM_MAX_MODEL_LENGTH:-0}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

METHOD_NAME="${METHOD_NAME:-GRPO Answer-only}"
TRAIN_DATA_LABEL="${TRAIN_DATA_LABEL:-}"
SUMMARY_NOTES="${SUMMARY_NOTES:-}"
STEP_LOG_DIR="${STEP_LOG_DIR:-${OUTPUT_DIR}/step_logs}"

TRAIN_METADATA_PATH="${METADATA_DIR}/selected_train_metadata.jsonl"
VALID_METADATA_PATH="${METADATA_DIR}/selected_valid_metadata.jsonl"
TEST_METADATA_PATH="${METADATA_DIR}/selected_test_metadata.jsonl"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required file/path: ${path}" >&2
    exit 1
  fi
}

count_jsonl_rows() {
  local path="$1"
  awk 'NF { count += 1 } END { print count + 0 }' "${path}"
}

resolve_requested_size() {
  local requested="$1"
  local available_path="$2"
  local label="$3"
  local allow_zero="$4"
  local available_count

  require_file "${available_path}"
  available_count="$(count_jsonl_rows "${available_path}")"

  case "${requested}" in
    full|-1)
      echo "${available_count}"
      return 0
      ;;
    none|null|"")
      if [[ "${allow_zero}" == "1" ]]; then
        echo "0"
        return 0
      fi
      echo "[ERROR] ${label}_size cannot be '${requested}'." >&2
      exit 1
      ;;
  esac

  if [[ ! "${requested}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] ${label}_size must be an integer, -1/full, or none/null for eval." >&2
    exit 1
  fi
  if (( requested < 0 )); then
    echo "[ERROR] ${label}_size=${requested} is invalid. Use -1 or full for all rows." >&2
    exit 1
  fi
  if (( requested == 0 )) && [[ "${allow_zero}" != "1" ]]; then
    echo "[ERROR] ${label}_size must be positive." >&2
    exit 1
  fi
  echo "${requested}"
}

print_command() {
  printf '[CMD]'
  for token in "$@"; do
    printf ' %q' "${token}"
  done
  printf '\n'
}

run_cmd_logged() {
  local log_path="$1"
  shift
  print_command "$@"
  echo "[INFO] log_path=${log_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${log_path}")"
  "$@" 2>&1 | tee "${log_path}"
}

require_file "Answer_only_GRPO.py"
require_file "eval_solver_checkpoint.py"
require_file "summarize_gsm8k_experiment.py"
require_file "${TRAIN_METADATA_PATH}"
require_file "${TEST_METADATA_PATH}"

RESOLVED_TRAIN_SIZE="$(resolve_requested_size "${TRAIN_SIZE}" "${TRAIN_METADATA_PATH}" "train" "0")"

if [[ -e "${VALID_METADATA_PATH}" ]]; then
  RESOLVED_EVAL_SIZE="$(resolve_requested_size "${EVAL_SIZE}" "${VALID_METADATA_PATH}" "eval" "1")"
else
  case "${EVAL_SIZE}" in
    0|none|null|"")
      RESOLVED_EVAL_SIZE="0"
      ;;
    *)
      echo "[ERROR] Missing valid metadata and eval_size=${EVAL_SIZE} requires it: ${VALID_METADATA_PATH}" >&2
      exit 1
      ;;
  esac
fi

if [[ "${SKIP_VALID_EVAL}" == "1" ]]; then
  RESOLVED_EVAL_SIZE="0"
fi

if [[ -z "${TRAIN_DATA_LABEL}" ]]; then
  if (( RESOLVED_TRAIN_SIZE > 3000 )) && (( RESOLVED_EVAL_SIZE == 0 )); then
    TRAIN_DATA_LABEL="GSM8K official train full"
  else
    TRAIN_DATA_LABEL="GSM8K-3K"
  fi
fi

if [[ ! -e "${VALID_METADATA_PATH}" ]] && (( RESOLVED_EVAL_SIZE == 0 )); then
  VALID_METADATA_PATH="${OUTPUT_DIR}/empty_valid_metadata.jsonl"
fi

if (( RESOLVED_TRAIN_SIZE > 3000 )) && (( RESOLVED_EVAL_SIZE == 0 )); then
  case "${OUTPUT_DIR}" in
    *fulltrain*|*full_train*|*gsm8k_full*|*full*)
      ;;
    *)
      echo "[ERROR] Full-train runs must use an output path containing 'full'." >&2
      exit 1
      ;;
  esac
fi

RUN_VALID_EVAL=0
if (( RESOLVED_EVAL_SIZE > 0 )) && [[ "${SKIP_VALID_EVAL}" != "1" ]]; then
  RUN_VALID_EVAL=1
fi

if [[ -z "${SUMMARY_NOTES}" ]]; then
  SUMMARY_NOTES="deterministic full-test evaluation"
fi
if [[ "${RUN_VALID_EVAL}" != "1" ]]; then
  SUMMARY_NOTES="${SUMMARY_NOTES}; valid evaluation skipped"
fi

echo "[INFO] GSM8K answer-only GRPO resolved configuration"
echo "[INFO] metadata_dir=${METADATA_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] train_metadata_path=${TRAIN_METADATA_PATH}"
echo "[INFO] valid_metadata_path=${VALID_METADATA_PATH}"
echo "[INFO] test_metadata_path=${TEST_METADATA_PATH}"
echo "[INFO] resolved_train_size=${RESOLVED_TRAIN_SIZE}"
echo "[INFO] resolved_eval_size=${RESOLVED_EVAL_SIZE}"
echo "[INFO] run_valid_eval=${RUN_VALID_EVAL}"
echo "[INFO] dry_run=${DRY_RUN}"
echo "[INFO] use_vllm=${USE_VLLM}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/test" "${STEP_LOG_DIR}"
  if (( RUN_VALID_EVAL == 1 )); then
    mkdir -p "${OUTPUT_DIR}/valid"
  fi
  if [[ ! -e "${VALID_METADATA_PATH}" ]]; then
    : > "${VALID_METADATA_PATH}"
  fi
fi

TRAIN_BOOL_ARGS=()
if [[ "${USE_LORA}" == "1" ]]; then
  TRAIN_BOOL_ARGS+=(--use_lora)
else
  TRAIN_BOOL_ARGS+=(--no-use_lora)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  TRAIN_BOOL_ARGS+=(--gradient_checkpointing)
else
  TRAIN_BOOL_ARGS+=(--no-gradient_checkpointing)
fi
TRAIN_VLLM_ARGS=()
EVAL_VLLM_ARGS=()
if [[ "${USE_VLLM}" == "1" ]]; then
  TRAIN_VLLM_ARGS+=(
    --use_vllm
    --vllm_mode "${VLLM_MODE}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )
  EVAL_VLLM_ARGS+=(
    --use_vllm
    --vllm_gpu_memory_utilization "${VLLM_EVAL_GPU_MEMORY_UTILIZATION}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )
  if (( VLLM_MAX_MODEL_LENGTH > 0 )); then
    TRAIN_VLLM_ARGS+=(--vllm_max_model_length "${VLLM_MAX_MODEL_LENGTH}")
    EVAL_VLLM_ARGS+=(--vllm_max_model_length "${VLLM_MAX_MODEL_LENGTH}")
  fi
fi

TRAIN_CMD=(
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python3
  Answer_only_GRPO.py
  --model_name "${MODEL_NAME}"
  --dataset_name "${DATASET_NAME}"
  --use_fixed_metadata
  --train_metadata_path "${TRAIN_METADATA_PATH}"
  --eval_metadata_path "${VALID_METADATA_PATH}"
  --train_size "${RESOLVED_TRAIN_SIZE}"
  --eval_size "${RESOLVED_EVAL_SIZE}"
  --num_generations "${NUM_GENERATIONS}"
  --max_prompt_length "${TRAIN_MAX_PROMPT_LENGTH}"
  --max_steps "${MAX_STEPS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --min_solve_rate "${MIN_SOLVE_RATE}"
  --max_solve_rate "${MAX_SOLVE_RATE}"
  --seed "${SEED}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --progress_interval_percent "${PROGRESS_INTERVAL_PERCENT}"
  --bf16
  --output_dir "${OUTPUT_DIR}"
  "${TRAIN_VLLM_ARGS[@]}"
  "${TRAIN_BOOL_ARGS[@]}"
)

TEST_EVAL_CMD=(
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python3
  eval_solver_checkpoint.py
  --model_name "${MODEL_NAME}"
  --adapter_path "${OUTPUT_DIR}"
  --eval_metadata_path "${TEST_METADATA_PATH}"
  --output_dir "${OUTPUT_DIR}/test"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --seed "${SEED}"
  --no_do_sample
  "${EVAL_VLLM_ARGS[@]}"
)

SUMMARY_CMD=(
  python3
  summarize_gsm8k_experiment.py
  --experiment_output_dir "${OUTPUT_DIR}"
  --metadata_dir "${METADATA_DIR}"
  --method "${METHOD_NAME}"
  --train_data "${TRAIN_DATA_LABEL}"
  --reward "answer correctness"
  --analyzer_type "none"
  --notes "${SUMMARY_NOTES}"
  --checkpoint_path "${OUTPUT_DIR}"
  --test_summary_path "${OUTPUT_DIR}/test/summary.json"
)

VALID_EVAL_CMD=()
if (( RUN_VALID_EVAL == 1 )); then
  VALID_EVAL_CMD=(
    env
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    python3
    eval_solver_checkpoint.py
    --model_name "${MODEL_NAME}"
    --adapter_path "${OUTPUT_DIR}"
    --eval_metadata_path "${VALID_METADATA_PATH}"
    --output_dir "${OUTPUT_DIR}/valid"
    --batch_size "${BATCH_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --seed "${SEED}"
    --no_do_sample
    "${EVAL_VLLM_ARGS[@]}"
  )
  SUMMARY_CMD+=(
    --valid_summary_path "${OUTPUT_DIR}/valid/summary.json"
  )
fi

run_cmd_logged "${STEP_LOG_DIR}/01_train.log" "${TRAIN_CMD[@]}"
if (( RUN_VALID_EVAL == 1 )); then
  run_cmd_logged "${STEP_LOG_DIR}/02_eval_valid.log" "${VALID_EVAL_CMD[@]}"
fi
run_cmd_logged "${STEP_LOG_DIR}/03_eval_test.log" "${TEST_EVAL_CMD[@]}"
run_cmd_logged "${STEP_LOG_DIR}/04_summary.log" "${SUMMARY_CMD[@]}"

echo "[DONE] GSM8K answer-only GRPO artifacts saved under ${OUTPUT_DIR}"
