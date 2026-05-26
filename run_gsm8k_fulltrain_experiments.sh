#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)
      DRY_RUN=1
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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
ANALYZER_MODEL_NAME="${ANALYZER_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
SEED="${SEED:-42}"

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-outputs/gsm8k_experiments}"
PIPELINE_DIR="${PIPELINE_DIR:-${EXPERIMENTS_ROOT}/fulltrain_pipeline}"
METADATA_DIR="${METADATA_DIR:-${EXPERIMENTS_ROOT}/metadata_fulltrain_seed42}"

ANSWER_ONLY_OUTPUT_DIR="${ANSWER_ONLY_OUTPUT_DIR:-${EXPERIMENTS_ROOT}/grpo_answer_only_qwen3b_fulltrain_n8_steps500}"
BAYES_SFT_OUTPUT_DIR="${BAYES_SFT_OUTPUT_DIR:-${EXPERIMENTS_ROOT}/grpo_bayesian_sft_analyzer_qwen3b_fulltrain_n8_steps500_lambda10}"
BAYES_SFT_DPO_OUTPUT_DIR="${BAYES_SFT_DPO_OUTPUT_DIR:-${EXPERIMENTS_ROOT}/grpo_bayesian_sft_dpo_analyzer_qwen3b_fulltrain_n8_steps500_lambda10}"
PROMPTED_OUTPUT_DIR="${PROMPTED_OUTPUT_DIR:-${EXPERIMENTS_ROOT}/grpo_bayesian_prompted_qwen3b_fulltrain_n8_steps500_lambda10}"

SFT_ANALYZER_ADAPTER_PATH="${SFT_ANALYZER_ADAPTER_PATH:-outputs/unified_analyzer_sft_v1_lambda07_bootstrap_lora}"
SFT_DPO_ANALYZER_ADAPTER_PATH="${SFT_DPO_ANALYZER_ADAPTER_PATH:-outputs/unified_analyzer_dpo_optionB_sft_lora}"

RUN_ANSWER_ONLY="${RUN_ANSWER_ONLY:-1}"
RUN_SFT_ANALYZER="${RUN_SFT_ANALYZER:-1}"
RUN_SFT_DPO_ANALYZER="${RUN_SFT_DPO_ANALYZER:-1}"
RUN_PROMPTED="${RUN_PROMPTED:-0}"
RUN_COLLECT_RESULTS="${RUN_COLLECT_RESULTS:-1}"
SKIP_PREFLIGHT_RECOMPUTE="${SKIP_PREFLIGHT_RECOMPUTE:-1}"

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

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required file/path: ${path}" >&2
    exit 1
  fi
}

require_file "prepare_gsm8k_metadata.py"
require_file "run_gsm8k_grpo_answer_only.sh"
require_file "run_grpo_bayesian_with_learned_analyzer.py"
require_file "run_gsm8k_grpo_bayesian_prompted.sh"
require_file "collect_gsm8k_learned_analyzer_results.py"

echo "[INFO] GSM8K full-train/full-test pipeline"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] metadata_dir=${METADATA_DIR}"
echo "[INFO] answer_only_output_dir=${ANSWER_ONLY_OUTPUT_DIR}"
echo "[INFO] bayes_sft_output_dir=${BAYES_SFT_OUTPUT_DIR}"
echo "[INFO] bayes_sft_dpo_output_dir=${BAYES_SFT_DPO_OUTPUT_DIR}"
echo "[INFO] prompted_output_dir=${PROMPTED_OUTPUT_DIR}"
echo "[INFO] sft_analyzer_adapter_path=${SFT_ANALYZER_ADAPTER_PATH}"
echo "[INFO] sft_dpo_analyzer_adapter_path=${SFT_DPO_ANALYZER_ADAPTER_PATH}"
echo "[INFO] dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${PIPELINE_DIR}"
fi

PREPARE_CMD=(
  python3
  prepare_gsm8k_metadata.py
  --setting full_train
  --output_dir "${METADATA_DIR}"
  --seed "${SEED}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  PREPARE_CMD+=(--dry_run)
fi
run_cmd_logged "${PIPELINE_DIR}/01_prepare_metadata.log" "${PREPARE_CMD[@]}"

if [[ "${RUN_ANSWER_ONLY}" == "1" ]]; then
  ANSWER_CMD=(
    env
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    MODEL_NAME="${MODEL_NAME}"
    METADATA_DIR="${METADATA_DIR}"
    OUTPUT_DIR="${ANSWER_ONLY_OUTPUT_DIR}"
    TRAIN_SIZE="full"
    EVAL_SIZE="0"
    SKIP_VALID_EVAL="1"
    SEED="${SEED}"
    TRAIN_DATA_LABEL="GSM8K official train full"
    METHOD_NAME="GRPO Answer-only fulltrain"
    SUMMARY_NOTES="deterministic official GSM8K test evaluation"
    bash
    run_gsm8k_grpo_answer_only.sh
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    ANSWER_CMD+=(--dry_run)
  fi
  run_cmd_logged "${PIPELINE_DIR}/02_answer_only_fulltrain.log" "${ANSWER_CMD[@]}"
fi

if [[ "${RUN_SFT_ANALYZER}" == "1" ]]; then
  require_file "${SFT_ANALYZER_ADAPTER_PATH}"
  SFT_CMD=(
    env
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
    python3
    run_grpo_bayesian_with_learned_analyzer.py
    --analyzer_adapter_path "${SFT_ANALYZER_ADAPTER_PATH}"
    --analyzer_model_name "${ANALYZER_MODEL_NAME}"
    --model_name "${MODEL_NAME}"
    --metadata_dir "${METADATA_DIR}"
    --output_dir "${BAYES_SFT_OUTPUT_DIR}"
    --train_size full
    --eval_size 0
    --skip_valid_eval
    --prior_lambda 1.0
    --prior_softmax_temperature 1.0
    --judge_max_new_tokens 768
    --num_generations 8
    --max_prompt_length 1024
    --max_completion_length 1024
    --temperature 0.7
    --top_p 0.95
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 8
    --learning_rate 5e-6
    --max_steps 500
    --seed "${SEED}"
    --method_name "GRPO Bayesian SFT Analyzer fulltrain"
    --train_data_label "GSM8K official train full"
    --analyzer_type learned_sft
    --notes "deterministic official GSM8K test evaluation"
  )
  if [[ "${SKIP_PREFLIGHT_RECOMPUTE}" == "1" ]]; then
    SFT_CMD+=(--skip_preflight_recompute)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    SFT_CMD+=(--dry_run)
  fi
  run_cmd_logged "${PIPELINE_DIR}/03_bayesian_sft_fulltrain.log" "${SFT_CMD[@]}"
fi

if [[ "${RUN_SFT_DPO_ANALYZER}" == "1" ]]; then
  require_file "${SFT_DPO_ANALYZER_ADAPTER_PATH}"
  SFT_DPO_CMD=(
    env
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
    python3
    run_grpo_bayesian_with_learned_analyzer.py
    --analyzer_adapter_path "${SFT_DPO_ANALYZER_ADAPTER_PATH}"
    --analyzer_model_name "${ANALYZER_MODEL_NAME}"
    --model_name "${MODEL_NAME}"
    --metadata_dir "${METADATA_DIR}"
    --output_dir "${BAYES_SFT_DPO_OUTPUT_DIR}"
    --train_size full
    --eval_size 0
    --skip_valid_eval
    --prior_lambda 1.0
    --prior_softmax_temperature 1.0
    --judge_max_new_tokens 768
    --num_generations 8
    --max_prompt_length 1024
    --max_completion_length 1024
    --temperature 0.7
    --top_p 0.95
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 8
    --learning_rate 5e-6
    --max_steps 500
    --seed "${SEED}"
    --method_name "GRPO Bayesian SFT+DPO Analyzer fulltrain"
    --train_data_label "GSM8K official train full"
    --analyzer_type learned_sft_dpo
    --notes "deterministic official GSM8K test evaluation"
  )
  if [[ "${SKIP_PREFLIGHT_RECOMPUTE}" == "1" ]]; then
    SFT_DPO_CMD+=(--skip_preflight_recompute)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    SFT_DPO_CMD+=(--dry_run)
  fi
  run_cmd_logged "${PIPELINE_DIR}/04_bayesian_sft_dpo_fulltrain.log" "${SFT_DPO_CMD[@]}"
fi

if [[ "${RUN_PROMPTED}" == "1" ]]; then
  PROMPTED_CMD=(
    env
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    MODEL_NAME="${MODEL_NAME}"
    PRIOR_JUDGE_MODEL="${MODEL_NAME}"
    EVIDENCE_JUDGE_MODEL="${MODEL_NAME}"
    METADATA_DIR="${METADATA_DIR}"
    OUTPUT_DIR="${PROMPTED_OUTPUT_DIR}"
    TRAIN_SIZE="full"
    EVAL_SIZE="0"
    SKIP_VALID_EVAL="1"
    SEED="${SEED}"
    PRIOR_LAMBDA="1.0"
    TRAIN_DATA_LABEL="GSM8K official train full"
    METHOD_NAME="GRPO Bayesian Prompted Analyzer fulltrain"
    SUMMARY_NOTES="deterministic official GSM8K test evaluation"
    bash
    run_gsm8k_grpo_bayesian_prompted.sh
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    PROMPTED_CMD+=(--dry_run)
  fi
  run_cmd_logged "${PIPELINE_DIR}/05_bayesian_prompted_fulltrain.log" "${PROMPTED_CMD[@]}"
fi

if [[ "${RUN_COLLECT_RESULTS}" == "1" ]]; then
  COLLECT_CMD=(
    python3
    collect_gsm8k_learned_analyzer_results.py
    --setting fulltrain
    --experiments_root "${EXPERIMENTS_ROOT}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    COLLECT_CMD+=(--dry_run)
  fi
  run_cmd_logged "${PIPELINE_DIR}/06_collect_fulltrain_results.log" "${COLLECT_CMD[@]}"
fi

echo "[DONE] GSM8K full-train/full-test pipeline complete"
