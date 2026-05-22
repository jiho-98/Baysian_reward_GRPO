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
METADATA_DIR="${METADATA_DIR:-outputs/gsm8k_3000_500_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm8k_experiments/base_qwen3b}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_EVAL_GPU_MEMORY_UTILIZATION="${VLLM_EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LENGTH="${VLLM_MAX_MODEL_LENGTH:-0}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
TRAIN_DATA_LABEL="${TRAIN_DATA_LABEL:-0}"
METHOD_NAME="${METHOD_NAME:-Base Qwen2.5-3B-Instruct}"
SUMMARY_NOTES="${SUMMARY_NOTES:-deterministic base eval on official GSM8K test}"
STEP_LOG_DIR="${STEP_LOG_DIR:-${OUTPUT_DIR}/step_logs}"

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

require_file "eval_solver_checkpoint.py"
require_file "summarize_gsm8k_experiment.py"
require_file "${TEST_METADATA_PATH}"

RUN_VALID_EVAL=0
if [[ "${SKIP_VALID_EVAL}" != "1" ]] && [[ -e "${VALID_METADATA_PATH}" ]]; then
  if (( "$(count_jsonl_rows "${VALID_METADATA_PATH}")" > 0 )); then
    RUN_VALID_EVAL=1
  fi
fi

echo "[INFO] Base GSM8K eval resolved configuration"
echo "[INFO] metadata_dir=${METADATA_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] valid_metadata_path=${VALID_METADATA_PATH}"
echo "[INFO] test_metadata_path=${TEST_METADATA_PATH}"
echo "[INFO] run_valid_eval=${RUN_VALID_EVAL}"
echo "[INFO] dry_run=${DRY_RUN}"
echo "[INFO] use_vllm=${USE_VLLM}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}/test" "${STEP_LOG_DIR}"
  if (( RUN_VALID_EVAL == 1 )); then
    mkdir -p "${OUTPUT_DIR}/valid"
  fi
fi

EVAL_VLLM_ARGS=()
if [[ "${USE_VLLM}" == "1" ]]; then
  EVAL_VLLM_ARGS+=(
    --use_vllm
    --vllm_gpu_memory_utilization "${VLLM_EVAL_GPU_MEMORY_UTILIZATION}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )
  if (( VLLM_MAX_MODEL_LENGTH > 0 )); then
    EVAL_VLLM_ARGS+=(--vllm_max_model_length "${VLLM_MAX_MODEL_LENGTH}")
  fi
fi

TEST_EVAL_CMD=(
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python3
  eval_solver_checkpoint.py
  --model_name "${MODEL_NAME}"
  --adapter_path "__base__"
  --no_load_adapter
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
  --reward "none"
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
    --adapter_path "__base__"
    --no_load_adapter
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

if (( RUN_VALID_EVAL == 1 )); then
  run_cmd_logged "${STEP_LOG_DIR}/01_eval_valid.log" "${VALID_EVAL_CMD[@]}"
fi
run_cmd_logged "${STEP_LOG_DIR}/02_eval_test.log" "${TEST_EVAL_CMD[@]}"
run_cmd_logged "${STEP_LOG_DIR}/03_summary.log" "${SUMMARY_CMD[@]}"

echo "[DONE] Base GSM8K eval saved under ${OUTPUT_DIR}"
