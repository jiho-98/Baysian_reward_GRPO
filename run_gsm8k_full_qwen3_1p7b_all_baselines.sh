#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --force_rerun)
      FORCE_RERUN=1
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

EXPECTED_MODEL_NAME="Qwen/Qwen3-1.7B"
MODEL_NAME="${MODEL_NAME:-${EXPECTED_MODEL_NAME}}"
ANALYZER_MODEL_NAME="${ANALYZER_MODEL_NAME:-${EXPECTED_MODEL_NAME}}"

if [[ "${MODEL_NAME}" != "${EXPECTED_MODEL_NAME}" ]]; then
  echo "[ERROR] MODEL_NAME must be ${EXPECTED_MODEL_NAME}, got ${MODEL_NAME}" >&2
  exit 1
fi
if [[ "${ANALYZER_MODEL_NAME}" != "${EXPECTED_MODEL_NAME}" ]]; then
  echo "[ERROR] ANALYZER_MODEL_NAME must be ${EXPECTED_MODEL_NAME}, got ${ANALYZER_MODEL_NAME}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEED="${SEED:-42}"

METADATA_DIR="${METADATA_DIR:-outputs/gsm8k_full_train_seed42}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/gsm8k_full_qwen3_1p7b}"
STEP_LOG_ROOT="${STEP_LOG_ROOT:-${EXPERIMENT_ROOT}/step_logs}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${EXPERIMENT_ROOT}/base_prompted}"
ANSWER_ONLY_OUTPUT_DIR="${ANSWER_ONLY_OUTPUT_DIR:-${EXPERIMENT_ROOT}/grpo_answer_only}"
PROMPTED_OUTPUT_DIR="${PROMPTED_OUTPUT_DIR:-${EXPERIMENT_ROOT}/grpo_bayesian_prompted}"
ANALYZER_PIPELINE_DIR="${ANALYZER_PIPELINE_DIR:-${EXPERIMENT_ROOT}/analyzer_pipeline}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${ANALYZER_PIPELINE_DIR}/sft_data}"
SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-${ANALYZER_PIPELINE_DIR}/sft_adapter}"
SFT_RECOMPUTE_DIR="${SFT_RECOMPUTE_DIR:-${ANALYZER_PIPELINE_DIR}/sft_recompute_on_prompted_pool}"
DPO_DATA_DIR="${DPO_DATA_DIR:-${ANALYZER_PIPELINE_DIR}/dpo_data_from_sft_recompute}"
DPO_ADAPTER_DIR="${DPO_ADAPTER_DIR:-${ANALYZER_PIPELINE_DIR}/dpo_adapter}"
SFT_GRPO_OUTPUT_DIR="${SFT_GRPO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/grpo_bayesian_sft_analyzer}"
DPO_GRPO_OUTPUT_DIR="${DPO_GRPO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/grpo_bayesian_sft_dpo_analyzer}"

COMPARISON_JSON="${COMPARISON_JSON:-${EXPERIMENT_ROOT}/gsm8k_full_qwen3_1p7b_comparison.json}"
COMPARISON_CSV="${COMPARISON_CSV:-${EXPERIMENT_ROOT}/gsm8k_full_qwen3_1p7b_comparison.csv}"
PIPELINE_CONFIG_PATH="${PIPELINE_CONFIG_PATH:-${EXPERIMENT_ROOT}/pipeline_launcher_config.json}"

RUN_BASE="${RUN_BASE:-1}"
RUN_ANSWER_ONLY="${RUN_ANSWER_ONLY:-1}"
RUN_BAYESIAN_PROMPTED="${RUN_BAYESIAN_PROMPTED:-1}"
RUN_ANALYZER_SFT="${RUN_ANALYZER_SFT:-1}"
RUN_ANALYZER_DPO="${RUN_ANALYZER_DPO:-1}"
RUN_SFT_ANALYZER_GRPO="${RUN_SFT_ANALYZER_GRPO:-1}"
RUN_DPO_ANALYZER_GRPO="${RUN_DPO_ANALYZER_GRPO:-1}"
RUN_COLLECT_RESULTS="${RUN_COLLECT_RESULTS:-1}"

# Solver / GRPO hyperparameters.
TRAIN_SIZE="${TRAIN_SIZE:-full}"
EVAL_SIZE="${EVAL_SIZE:-0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
TRAIN_MAX_PROMPT_LENGTH="${TRAIN_MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SOLVER_USE_LORA="${SOLVER_USE_LORA:-1}"
SOLVER_GRADIENT_CHECKPOINTING="${SOLVER_GRADIENT_CHECKPOINTING:-1}"
SOLVER_LORA_R="${SOLVER_LORA_R:-16}"
SOLVER_LORA_ALPHA="${SOLVER_LORA_ALPHA:-32}"
SOLVER_LORA_DROPOUT="${SOLVER_LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
PROGRESS_INTERVAL_PERCENT="${PROGRESS_INTERVAL_PERCENT:-5}"
PRIOR_LAMBDA="${PRIOR_LAMBDA:-1.0}"
PRIOR_SOFTMAX_TEMPERATURE="${PRIOR_SOFTMAX_TEMPERATURE:-1.0}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-768}"
BF16="${BF16:-1}"

# Evaluation settings preserved from previous runs.
BASELINE_EVAL_BATCH_SIZE="${BASELINE_EVAL_BATCH_SIZE:-32}"
BASELINE_EVAL_MAX_NEW_TOKENS="${BASELINE_EVAL_MAX_NEW_TOKENS:-1024}"
BASELINE_EVAL_MAX_PROMPT_LENGTH="${BASELINE_EVAL_MAX_PROMPT_LENGTH:-2048}"
LEARNED_EVAL_BATCH_SIZE="${LEARNED_EVAL_BATCH_SIZE:-32}"
LEARNED_EVAL_MAX_NEW_TOKENS="${LEARNED_EVAL_MAX_NEW_TOKENS:-1024}"
LEARNED_EVAL_MAX_PROMPT_LENGTH="${LEARNED_EVAL_MAX_PROMPT_LENGTH:-1024}"

# Analyzer SFT data + training hyperparameters copied from the prior working pipeline.
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SFT_EVIDENCE_FRACTION="${SFT_EVIDENCE_FRACTION:-0.8}"
SFT_CLEAN_FRACTION="${SFT_CLEAN_FRACTION:-0.7}"
SFT_HARD_CASE_TOP_FRACTION="${SFT_HARD_CASE_TOP_FRACTION:-0.25}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-4096}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1.0}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-4}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-200}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-200}"
SFT_LORA_R="${SFT_LORA_R:-16}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-32}"
SFT_LORA_DROPOUT="${SFT_LORA_DROPOUT:-0.05}"
SFT_BF16="${SFT_BF16:-1}"
SFT_USE_4BIT="${SFT_USE_4BIT:-0}"

# Prompted-pool recompute settings.
RECOMPUTE_BATCH_SIZE="${RECOMPUTE_BATCH_SIZE:-8}"
RECOMPUTE_MAX_NEW_TOKENS="${RECOMPUTE_MAX_NEW_TOKENS:-512}"
RECOMPUTE_MAX_INPUT_TOKENS="${RECOMPUTE_MAX_INPUT_TOKENS:-4096}"
RECOMPUTE_BF16="${RECOMPUTE_BF16:-1}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-0.8}"
EVIDENCE_WEIGHT="${EVIDENCE_WEIGHT:-0.2}"

# Analyzer DPO data + training hyperparameters copied from the prior working pipeline.
DPO_VAL_RATIO="${DPO_VAL_RATIO:-0.1}"
DPO_HARD_CASE_TOP_FRACTION="${DPO_HARD_CASE_TOP_FRACTION:-0.25}"
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
DPO_LORA_R="${DPO_LORA_R:-16}"
DPO_LORA_ALPHA="${DPO_LORA_ALPHA:-32}"
DPO_LORA_DROPOUT="${DPO_LORA_DROPOUT:-0.05}"
DPO_BF16="${DPO_BF16:-1}"
DPO_USE_4BIT="${DPO_USE_4BIT:-0}"
DPO_GRADIENT_CHECKPOINTING="${DPO_GRADIENT_CHECKPOINTING:-0}"

PROMPT_FORMAT_NOTE="shared apply_chat_template uses enable_thinking=false when supported; solver output remains explicit [Strategy]/[Reasoning]/[Final Answer]"

BF16_ARGS=()
if [[ "${BF16}" == "1" ]]; then
  BF16_ARGS+=(--bf16)
fi

SFT_BF16_ARGS=()
if [[ "${SFT_BF16}" == "1" ]]; then
  SFT_BF16_ARGS+=(--bf16)
fi
SFT_4BIT_ARGS=()
if [[ "${SFT_USE_4BIT}" == "1" ]]; then
  SFT_4BIT_ARGS+=(--use_4bit)
fi

RECOMPUTE_BF16_ARGS=()
if [[ "${RECOMPUTE_BF16}" == "1" ]]; then
  RECOMPUTE_BF16_ARGS+=(--bf16)
fi

DPO_BF16_ARGS=()
if [[ "${DPO_BF16}" == "1" ]]; then
  DPO_BF16_ARGS+=(--bf16)
fi
DPO_4BIT_ARGS=()
if [[ "${DPO_USE_4BIT}" == "1" ]]; then
  DPO_4BIT_ARGS+=(--use_4bit)
fi
DPO_GRADIENT_CHECKPOINTING_ARGS=()
if [[ "${DPO_GRADIENT_CHECKPOINTING}" == "1" ]]; then
  DPO_GRADIENT_CHECKPOINTING_ARGS+=(--gradient_checkpointing)
fi

SOLVER_USE_LORA_ARGS=()
if [[ "${SOLVER_USE_LORA}" == "1" ]]; then
  SOLVER_USE_LORA_ARGS+=(--use_lora)
else
  SOLVER_USE_LORA_ARGS+=(--no-use_lora)
fi

SOLVER_GRADIENT_CHECKPOINTING_ARGS=()
if [[ "${SOLVER_GRADIENT_CHECKPOINTING}" == "1" ]]; then
  SOLVER_GRADIENT_CHECKPOINTING_ARGS+=(--gradient_checkpointing)
else
  SOLVER_GRADIENT_CHECKPOINTING_ARGS+=(--no-gradient_checkpointing)
fi

print_command() {
  printf '[CMD]'
  for token in "$@"; do
    printf ' %q' "${token}"
  done
  printf '\n'
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required file/path: ${path}" >&2
    exit 1
  fi
}

paths_exist() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      return 1
    fi
  done
  return 0
}

should_skip_step() {
  if [[ "${FORCE_RERUN}" == "1" || "${SKIP_COMPLETED}" != "1" ]]; then
    return 1
  fi
  paths_exist "$@"
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

write_step_json() {
  local output_path="$1"
  local python_body="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${output_path}")"
  python3 - <<PY
from pathlib import Path
import json

payload = ${python_body}
path = Path(${output_path@Q})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}

write_pipeline_config() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${EXPERIMENT_ROOT}" "${STEP_LOG_ROOT}" "${ANALYZER_PIPELINE_DIR}"
  python3 - <<PY
import json
from pathlib import Path

payload = {
    "model_name": ${MODEL_NAME@Q},
    "analyzer_model_name": ${ANALYZER_MODEL_NAME@Q},
    "metadata_dir": ${METADATA_DIR@Q},
    "experiment_root": ${EXPERIMENT_ROOT@Q},
    "base_output_dir": ${BASE_OUTPUT_DIR@Q},
    "answer_only_output_dir": ${ANSWER_ONLY_OUTPUT_DIR@Q},
    "prompted_output_dir": ${PROMPTED_OUTPUT_DIR@Q},
    "sft_data_dir": ${SFT_DATA_DIR@Q},
    "sft_adapter_dir": ${SFT_ADAPTER_DIR@Q},
    "sft_recompute_dir": ${SFT_RECOMPUTE_DIR@Q},
    "dpo_data_dir": ${DPO_DATA_DIR@Q},
    "dpo_adapter_dir": ${DPO_ADAPTER_DIR@Q},
    "sft_grpo_output_dir": ${SFT_GRPO_OUTPUT_DIR@Q},
    "dpo_grpo_output_dir": ${DPO_GRPO_OUTPUT_DIR@Q},
    "comparison_json": ${COMPARISON_JSON@Q},
    "comparison_csv": ${COMPARISON_CSV@Q},
    "seed": int(${SEED@Q}),
    "num_generations": int(${NUM_GENERATIONS@Q}),
    "max_steps": int(${MAX_STEPS@Q}),
    "train_max_prompt_length": int(${TRAIN_MAX_PROMPT_LENGTH@Q}),
    "max_completion_length": int(${MAX_COMPLETION_LENGTH@Q}),
    "temperature": float(${TEMPERATURE@Q}),
    "top_p": float(${TOP_P@Q}),
    "per_device_train_batch_size": int(${PER_DEVICE_TRAIN_BATCH_SIZE@Q}),
    "gradient_accumulation_steps": int(${GRADIENT_ACCUMULATION_STEPS@Q}),
    "learning_rate": float(${LEARNING_RATE@Q}),
    "solver_use_lora": ${SOLVER_USE_LORA@Q} == '1',
    "solver_gradient_checkpointing": ${SOLVER_GRADIENT_CHECKPOINTING@Q} == '1',
    "solver_lora_r": int(${SOLVER_LORA_R@Q}),
    "solver_lora_alpha": int(${SOLVER_LORA_ALPHA@Q}),
    "solver_lora_dropout": float(${SOLVER_LORA_DROPOUT@Q}),
    "prior_lambda": float(${PRIOR_LAMBDA@Q}),
    "prior_softmax_temperature": float(${PRIOR_SOFTMAX_TEMPERATURE@Q}),
    "judge_max_new_tokens": int(${JUDGE_MAX_NEW_TOKENS@Q}),
    "baseline_eval_max_prompt_length": int(${BASELINE_EVAL_MAX_PROMPT_LENGTH@Q}),
    "learned_eval_max_prompt_length": int(${LEARNED_EVAL_MAX_PROMPT_LENGTH@Q}),
    "sft_learning_rate": float(${SFT_LEARNING_RATE@Q}),
    "dpo_learning_rate": float(${DPO_LEARNING_RATE@Q}),
    "prompt_format_note": ${PROMPT_FORMAT_NOTE@Q},
}
path = Path(${PIPELINE_CONFIG_PATH@Q})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}

echo "[INFO] GSM8K full-train Qwen3-1.7B pipeline"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] ANALYZER_MODEL_NAME=${ANALYZER_MODEL_NAME}"
echo "[INFO] METADATA_DIR=${METADATA_DIR}"
echo "[INFO] EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "[INFO] STEP_LOG_ROOT=${STEP_LOG_ROOT}"
echo "[INFO] dry_run=${DRY_RUN}"
echo "[INFO] skip_completed=${SKIP_COMPLETED}"
echo "[INFO] force_rerun=${FORCE_RERUN}"
echo "[INFO] prompt_control=${PROMPT_FORMAT_NOTE}"
echo "[INFO] solver_use_lora=${SOLVER_USE_LORA}"

require_file "prepare_gsm8k_metadata.py"
require_file "run_gsm8k_base_eval.sh"
require_file "run_gsm8k_grpo_answer_only.sh"
require_file "run_gsm8k_grpo_bayesian_prompted.sh"
require_file "run_grpo_bayesian_with_learned_analyzer.py"
require_file "build_analyzer_sft_data_from_gsm8k_logs.py"
require_file "build_analyzer_dpo_data_from_gsm8k_logs.py"
require_file "train_unified_analyzer_sft.py"
require_file "train_unified_analyzer_dpo.py"
require_file "recompute_posterior_with_learned_analyzer.py"
require_file "collect_gsm8k_full_qwen3_1p7b_results.py"

write_pipeline_config

if should_skip_step "${METADATA_DIR}/metadata_summary.json" "${METADATA_DIR}/selected_train_metadata.jsonl" "${METADATA_DIR}/selected_test_metadata.jsonl"; then
  echo "[INFO] Skipping metadata preparation because outputs already exist."
else
  run_cmd_logged \
    "${STEP_LOG_ROOT}/01_prepare_metadata.log" \
    python3 prepare_gsm8k_metadata.py \
      --setting full_train \
      --output_dir "${METADATA_DIR}" \
      --seed "${SEED}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  require_file "${METADATA_DIR}/metadata_summary.json"
  require_file "${METADATA_DIR}/selected_train_metadata.jsonl"
  require_file "${METADATA_DIR}/selected_test_metadata.jsonl"

  run_cmd_logged \
    "${STEP_LOG_ROOT}/01b_metadata_summary.log" \
    python3 -c \
    "import json, pathlib; p=pathlib.Path(${METADATA_DIR@Q})/'metadata_summary.json'; print(p.read_text(encoding='utf-8'))"
else
  echo "[INFO] dry_run metadata summary will be written to ${METADATA_DIR}/metadata_summary.json"
fi

if [[ "${RUN_BASE}" == "1" ]]; then
  if should_skip_step "${BASE_OUTPUT_DIR}/summary.json" "${BASE_OUTPUT_DIR}/test/summary.json"; then
    echo "[INFO] Skipping base prompted eval because summaries already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/02_base_prompted.log" \
      env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        MODEL_NAME="${MODEL_NAME}" \
        METADATA_DIR="${METADATA_DIR}" \
        OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
        BATCH_SIZE="${BASELINE_EVAL_BATCH_SIZE}" \
        MAX_NEW_TOKENS="${BASELINE_EVAL_MAX_NEW_TOKENS}" \
        MAX_PROMPT_LENGTH="${BASELINE_EVAL_MAX_PROMPT_LENGTH}" \
        SEED="${SEED}" \
        TRAIN_DATA_LABEL="0" \
        METHOD_NAME="Base prompted evaluation" \
        SUMMARY_NOTES="deterministic official GSM8K full-test evaluation; ${PROMPT_FORMAT_NOTE}" \
        STEP_LOG_DIR="${STEP_LOG_ROOT}/02_base_prompted_steps" \
        bash run_gsm8k_base_eval.sh --skip_valid_eval
  fi
fi

if [[ "${RUN_ANSWER_ONLY}" == "1" ]]; then
  if should_skip_step "${ANSWER_ONLY_OUTPUT_DIR}/summary.json" "${ANSWER_ONLY_OUTPUT_DIR}/test/summary.json"; then
    echo "[INFO] Skipping answer-only GRPO because summaries already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/03_grpo_answer_only.log" \
      env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        MODEL_NAME="${MODEL_NAME}" \
        METADATA_DIR="${METADATA_DIR}" \
        OUTPUT_DIR="${ANSWER_ONLY_OUTPUT_DIR}" \
        TRAIN_SIZE="${TRAIN_SIZE}" \
        EVAL_SIZE="${EVAL_SIZE}" \
        SKIP_VALID_EVAL="1" \
        BATCH_SIZE="${BASELINE_EVAL_BATCH_SIZE}" \
        MAX_NEW_TOKENS="${BASELINE_EVAL_MAX_NEW_TOKENS}" \
        MAX_PROMPT_LENGTH="${BASELINE_EVAL_MAX_PROMPT_LENGTH}" \
        TRAIN_MAX_PROMPT_LENGTH="${TRAIN_MAX_PROMPT_LENGTH}" \
        TEMPERATURE="${TEMPERATURE}" \
        TOP_P="${TOP_P}" \
        NUM_GENERATIONS="${NUM_GENERATIONS}" \
        MAX_STEPS="${MAX_STEPS}" \
        MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH}" \
        PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
        LEARNING_RATE="${LEARNING_RATE}" \
        USE_LORA="${SOLVER_USE_LORA}" \
        GRADIENT_CHECKPOINTING="${SOLVER_GRADIENT_CHECKPOINTING}" \
        LORA_R="${SOLVER_LORA_R}" \
        LORA_ALPHA="${SOLVER_LORA_ALPHA}" \
        LORA_DROPOUT="${SOLVER_LORA_DROPOUT}" \
        LOGGING_STEPS="${LOGGING_STEPS}" \
        SAVE_STEPS="${SAVE_STEPS}" \
        PROGRESS_INTERVAL_PERCENT="${PROGRESS_INTERVAL_PERCENT}" \
        SEED="${SEED}" \
        METHOD_NAME="GRPO Answer-only" \
        TRAIN_DATA_LABEL="GSM8K official train full" \
        SUMMARY_NOTES="deterministic official GSM8K full-test evaluation; ${PROMPT_FORMAT_NOTE}" \
        STEP_LOG_DIR="${STEP_LOG_ROOT}/03_grpo_answer_only_steps" \
        bash run_gsm8k_grpo_answer_only.sh --skip_valid_eval
  fi
fi

if [[ "${RUN_BAYESIAN_PROMPTED}" == "1" ]]; then
  if should_skip_step "${PROMPTED_OUTPUT_DIR}/summary.json" "${PROMPTED_OUTPUT_DIR}/bayesian_reward_debug.jsonl" "${PROMPTED_OUTPUT_DIR}/bayesian_reward_diagnostics/summary.json" "${PROMPTED_OUTPUT_DIR}/test/summary.json"; then
    echo "[INFO] Skipping prompted Bayesian GRPO because outputs already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/04_grpo_bayesian_prompted.log" \
      env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        MODEL_NAME="${MODEL_NAME}" \
        PRIOR_JUDGE_MODEL="${MODEL_NAME}" \
        EVIDENCE_JUDGE_MODEL="${MODEL_NAME}" \
        METADATA_DIR="${METADATA_DIR}" \
        OUTPUT_DIR="${PROMPTED_OUTPUT_DIR}" \
        TRAIN_SIZE="${TRAIN_SIZE}" \
        EVAL_SIZE="${EVAL_SIZE}" \
        SKIP_VALID_EVAL="1" \
        BATCH_SIZE="${BASELINE_EVAL_BATCH_SIZE}" \
        MAX_NEW_TOKENS="${BASELINE_EVAL_MAX_NEW_TOKENS}" \
        MAX_PROMPT_LENGTH="${BASELINE_EVAL_MAX_PROMPT_LENGTH}" \
        TRAIN_MAX_PROMPT_LENGTH="${TRAIN_MAX_PROMPT_LENGTH}" \
        TEMPERATURE="${TEMPERATURE}" \
        TOP_P="${TOP_P}" \
        NUM_GENERATIONS="${NUM_GENERATIONS}" \
        MAX_STEPS="${MAX_STEPS}" \
        MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH}" \
        PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
        LEARNING_RATE="${LEARNING_RATE}" \
        USE_LORA="${SOLVER_USE_LORA}" \
        GRADIENT_CHECKPOINTING="${SOLVER_GRADIENT_CHECKPOINTING}" \
        LORA_R="${SOLVER_LORA_R}" \
        LORA_ALPHA="${SOLVER_LORA_ALPHA}" \
        LORA_DROPOUT="${SOLVER_LORA_DROPOUT}" \
        LOGGING_STEPS="${LOGGING_STEPS}" \
        SAVE_STEPS="${SAVE_STEPS}" \
        PROGRESS_INTERVAL_PERCENT="${PROGRESS_INTERVAL_PERCENT}" \
        PRIOR_LAMBDA="${PRIOR_LAMBDA}" \
        PRIOR_SOFTMAX_TEMPERATURE="${PRIOR_SOFTMAX_TEMPERATURE}" \
        JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS}" \
        SEED="${SEED}" \
        METHOD_NAME="GRPO Bayesian Prompted Analyzer" \
        TRAIN_DATA_LABEL="GSM8K official train full" \
        SUMMARY_NOTES="deterministic official GSM8K full-test evaluation; ${PROMPT_FORMAT_NOTE}" \
        STEP_LOG_DIR="${STEP_LOG_ROOT}/04_grpo_bayesian_prompted_steps" \
        bash run_gsm8k_grpo_bayesian_prompted.sh --skip_valid_eval
  fi
fi

if [[ "${RUN_ANALYZER_SFT}" == "1" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${PROMPTED_OUTPUT_DIR}/bayesian_reward_debug.jsonl"
  fi
  if should_skip_step "${SFT_DATA_DIR}/summary.json" "${SFT_DATA_DIR}/runtime_unified_train.jsonl" "${SFT_DATA_DIR}/runtime_unified_valid.jsonl"; then
    echo "[INFO] Skipping analyzer SFT data build because outputs already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/05_build_sft_data.log" \
      python3 build_analyzer_sft_data_from_gsm8k_logs.py \
        --log_dir "${PROMPTED_OUTPUT_DIR}" \
        --output_dir "${SFT_DATA_DIR}" \
        --val_ratio "${SFT_VAL_RATIO}" \
        --evidence_fraction "${SFT_EVIDENCE_FRACTION}" \
        --clean_fraction "${SFT_CLEAN_FRACTION}" \
        --hard_case_top_fraction "${SFT_HARD_CASE_TOP_FRACTION}" \
        --seed "${SEED}"
  fi

  if should_skip_step "${SFT_ADAPTER_DIR}/train_summary.json" "${SFT_ADAPTER_DIR}/adapter_config.json"; then
    echo "[INFO] Skipping analyzer SFT training because adapter already exists."
  else
    write_step_json "${SFT_ADAPTER_DIR}/launcher_config.json" "{
        'model_name': ${ANALYZER_MODEL_NAME@Q},
        'train_path': str(Path(${SFT_DATA_DIR@Q}) / 'runtime_unified_train.jsonl'),
        'valid_path': str(Path(${SFT_DATA_DIR@Q}) / 'runtime_unified_valid.jsonl'),
        'output_dir': ${SFT_ADAPTER_DIR@Q},
        'max_length': int(${SFT_MAX_LENGTH@Q}),
        'num_train_epochs': float(${SFT_NUM_TRAIN_EPOCHS@Q}),
        'per_device_train_batch_size': int(${SFT_PER_DEVICE_TRAIN_BATCH_SIZE@Q}),
        'per_device_eval_batch_size': int(${SFT_PER_DEVICE_EVAL_BATCH_SIZE@Q}),
        'gradient_accumulation_steps': int(${SFT_GRADIENT_ACCUMULATION_STEPS@Q}),
        'learning_rate': float(${SFT_LEARNING_RATE@Q}),
        'logging_steps': int(${SFT_LOGGING_STEPS@Q}),
        'save_steps': int(${SFT_SAVE_STEPS@Q}),
        'eval_steps': int(${SFT_EVAL_STEPS@Q}),
        'lora_r': int(${SFT_LORA_R@Q}),
        'lora_alpha': int(${SFT_LORA_ALPHA@Q}),
        'lora_dropout': float(${SFT_LORA_DROPOUT@Q}),
        'seed': int(${SEED@Q}),
        'bf16': ${SFT_BF16@Q} == '1',
        'use_4bit': ${SFT_USE_4BIT@Q} == '1',
        'prompt_format_note': ${PROMPT_FORMAT_NOTE@Q},
    }"
    run_cmd_logged \
      "${STEP_LOG_ROOT}/06_train_sft_analyzer.log" \
      env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      python3 train_unified_analyzer_sft.py \
        --model_name "${ANALYZER_MODEL_NAME}" \
        --train_path "${SFT_DATA_DIR}/runtime_unified_train.jsonl" \
        --val_path "${SFT_DATA_DIR}/runtime_unified_valid.jsonl" \
        --output_dir "${SFT_ADAPTER_DIR}" \
        --max_length "${SFT_MAX_LENGTH}" \
        --num_train_epochs "${SFT_NUM_TRAIN_EPOCHS}" \
        --per_device_train_batch_size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${SFT_PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${SFT_LEARNING_RATE}" \
        --logging_steps "${SFT_LOGGING_STEPS}" \
        --save_steps "${SFT_SAVE_STEPS}" \
        --eval_steps "${SFT_EVAL_STEPS}" \
        --lora_r "${SFT_LORA_R}" \
        --lora_alpha "${SFT_LORA_ALPHA}" \
        --lora_dropout "${SFT_LORA_DROPOUT}" \
        --seed "${SEED}" \
        "${SFT_BF16_ARGS[@]}" \
        "${SFT_4BIT_ARGS[@]}"
  fi

  if should_skip_step "${SFT_RECOMPUTE_DIR}/summary.json" "${SFT_RECOMPUTE_DIR}/learned_posterior_debug.jsonl"; then
    echo "[INFO] Skipping SFT recompute because outputs already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/07_recompute_sft_on_prompted_pool.log" \
      env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      python3 recompute_posterior_with_learned_analyzer.py \
        --input_debug_jsonl "${PROMPTED_OUTPUT_DIR}/bayesian_reward_debug.jsonl" \
        --output_dir "${SFT_RECOMPUTE_DIR}" \
        --model_name "${ANALYZER_MODEL_NAME}" \
        --adapter_path "${SFT_ADAPTER_DIR}" \
        --batch_size "${RECOMPUTE_BATCH_SIZE}" \
        --max_new_tokens "${RECOMPUTE_MAX_NEW_TOKENS}" \
        --max_input_tokens "${RECOMPUTE_MAX_INPUT_TOKENS}" \
        --answer_weight "${ANSWER_WEIGHT}" \
        --evidence_weight "${EVIDENCE_WEIGHT}" \
        --prior_lambda "${PRIOR_LAMBDA}" \
        --prior_temperature "${PRIOR_SOFTMAX_TEMPERATURE}" \
        "${RECOMPUTE_BF16_ARGS[@]}"
  fi
fi

if [[ "${RUN_SFT_ANALYZER_GRPO}" == "1" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${SFT_ADAPTER_DIR}/adapter_config.json"
  fi
  if [[ "${SFT_ADAPTER_DIR}" != "${ANALYZER_PIPELINE_DIR}/sft_adapter" ]]; then
    echo "[ERROR] SFT analyzer adapter path must point to the new full Qwen3 SFT adapter." >&2
    exit 1
  fi
  if should_skip_step "${SFT_GRPO_OUTPUT_DIR}/summary.json" "${SFT_GRPO_OUTPUT_DIR}/test/summary.json"; then
    echo "[INFO] Skipping solver GRPO with SFT analyzer because outputs already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/08_grpo_bayesian_sft_analyzer.log" \
      env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      python3 run_grpo_bayesian_with_learned_analyzer.py \
        --analyzer_adapter_path "${SFT_ADAPTER_DIR}" \
        --analyzer_model_name "${ANALYZER_MODEL_NAME}" \
        --model_name "${MODEL_NAME}" \
        --metadata_dir "${METADATA_DIR}" \
        --output_dir "${SFT_GRPO_OUTPUT_DIR}" \
        --train_size "${TRAIN_SIZE}" \
        --eval_size "${EVAL_SIZE}" \
        --skip_valid_eval \
        --num_generations "${NUM_GENERATIONS}" \
        --max_prompt_length "${TRAIN_MAX_PROMPT_LENGTH}" \
        --max_steps "${MAX_STEPS}" \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${LEARNING_RATE}" \
        --lora_r "${SOLVER_LORA_R}" \
        --lora_alpha "${SOLVER_LORA_ALPHA}" \
        --lora_dropout "${SOLVER_LORA_DROPOUT}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --seed "${SEED}" \
        --prior_lambda "${PRIOR_LAMBDA}" \
        --prior_softmax_temperature "${PRIOR_SOFTMAX_TEMPERATURE}" \
        --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
        --progress_interval_percent "${PROGRESS_INTERVAL_PERCENT}" \
        --eval_batch_size "${LEARNED_EVAL_BATCH_SIZE}" \
        --eval_max_new_tokens "${LEARNED_EVAL_MAX_NEW_TOKENS}" \
        --eval_max_prompt_length "${LEARNED_EVAL_MAX_PROMPT_LENGTH}" \
        --skip_preflight_recompute \
        --step_logs_dir "${STEP_LOG_ROOT}/08_grpo_bayesian_sft_analyzer_steps" \
        --method_name "GRPO Bayesian reward + SFT analyzer" \
        --train_data_label "GSM8K official train full" \
        --analyzer_type learned_sft \
        --notes "deterministic official GSM8K full-test evaluation; ${PROMPT_FORMAT_NOTE}" \
        "${SOLVER_USE_LORA_ARGS[@]}" \
        "${SOLVER_GRADIENT_CHECKPOINTING_ARGS[@]}" \
        "${BF16_ARGS[@]}"
  fi
fi

if [[ "${RUN_ANALYZER_DPO}" == "1" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${PROMPTED_OUTPUT_DIR}/bayesian_reward_debug.jsonl"
    require_file "${SFT_RECOMPUTE_DIR}/learned_posterior_debug.jsonl"
  fi
  if should_skip_step "${DPO_DATA_DIR}/summary.json" "${DPO_DATA_DIR}/runtime_unified_train.jsonl" "${DPO_DATA_DIR}/runtime_unified_valid.jsonl"; then
    echo "[INFO] Skipping analyzer DPO data build because outputs already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/09_build_dpo_data.log" \
      python3 build_analyzer_dpo_data_from_gsm8k_logs.py \
        --log_dir "${PROMPTED_OUTPUT_DIR}" \
        --output_dir "${DPO_DATA_DIR}" \
        --learned_posterior_debug_jsonl "${SFT_RECOMPUTE_DIR}/learned_posterior_debug.jsonl" \
        --val_ratio "${DPO_VAL_RATIO}" \
        --hard_case_top_fraction "${DPO_HARD_CASE_TOP_FRACTION}" \
        --target_evidence_train_pairs "${DPO_TARGET_EVIDENCE_TRAIN_PAIRS}" \
        --target_prior_train_pairs "${DPO_TARGET_PRIOR_TRAIN_PAIRS}" \
        --target_evidence_valid_pairs "${DPO_TARGET_EVIDENCE_VALID_PAIRS}" \
        --target_prior_valid_pairs "${DPO_TARGET_PRIOR_VALID_PAIRS}" \
        --synthetic_companion_rate "${DPO_SYNTHETIC_COMPANION_RATE}" \
        --seed "${SEED}"
  fi

  if should_skip_step "${DPO_ADAPTER_DIR}/train_summary.json" "${DPO_ADAPTER_DIR}/adapter_config.json"; then
    echo "[INFO] Skipping analyzer DPO training because adapter already exists."
  else
    write_step_json "${DPO_ADAPTER_DIR}/launcher_config.json" "{
        'model_name': ${ANALYZER_MODEL_NAME@Q},
        'train_path': str(Path(${DPO_DATA_DIR@Q}) / 'runtime_unified_train.jsonl'),
        'valid_path': str(Path(${DPO_DATA_DIR@Q}) / 'runtime_unified_valid.jsonl'),
        'output_dir': ${DPO_ADAPTER_DIR@Q},
        'init_adapter_path': ${SFT_ADAPTER_DIR@Q},
        'reference_adapter_path': ${SFT_ADAPTER_DIR@Q},
        'beta': float(${DPO_BETA@Q}),
        'max_length': int(${DPO_MAX_LENGTH@Q}),
        'max_prompt_length': int(${DPO_MAX_PROMPT_LENGTH@Q}),
        'max_completion_length': int(${DPO_MAX_COMPLETION_LENGTH@Q}),
        'num_train_epochs': float(${DPO_NUM_TRAIN_EPOCHS@Q}),
        'per_device_train_batch_size': int(${DPO_PER_DEVICE_TRAIN_BATCH_SIZE@Q}),
        'per_device_eval_batch_size': int(${DPO_PER_DEVICE_EVAL_BATCH_SIZE@Q}),
        'gradient_accumulation_steps': int(${DPO_GRADIENT_ACCUMULATION_STEPS@Q}),
        'learning_rate': float(${DPO_LEARNING_RATE@Q}),
        'logging_steps': int(${DPO_LOGGING_STEPS@Q}),
        'save_steps': int(${DPO_SAVE_STEPS@Q}),
        'eval_steps': int(${DPO_EVAL_STEPS@Q}),
        'lora_r': int(${DPO_LORA_R@Q}),
        'lora_alpha': int(${DPO_LORA_ALPHA@Q}),
        'lora_dropout': float(${DPO_LORA_DROPOUT@Q}),
        'seed': int(${SEED@Q}),
        'bf16': ${DPO_BF16@Q} == '1',
        'use_4bit': ${DPO_USE_4BIT@Q} == '1',
        'gradient_checkpointing': ${DPO_GRADIENT_CHECKPOINTING@Q} == '1',
        'prompt_format_note': ${PROMPT_FORMAT_NOTE@Q},
    }"
    run_cmd_logged \
      "${STEP_LOG_ROOT}/10_train_dpo_analyzer.log" \
      env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      python3 train_unified_analyzer_dpo.py \
        --model_name "${ANALYZER_MODEL_NAME}" \
        --train_path "${DPO_DATA_DIR}/runtime_unified_train.jsonl" \
        --val_path "${DPO_DATA_DIR}/runtime_unified_valid.jsonl" \
        --output_dir "${DPO_ADAPTER_DIR}" \
        --init_adapter_path "${SFT_ADAPTER_DIR}" \
        --reference_adapter_path "${SFT_ADAPTER_DIR}" \
        --beta "${DPO_BETA}" \
        --max_length "${DPO_MAX_LENGTH}" \
        --max_prompt_length "${DPO_MAX_PROMPT_LENGTH}" \
        --max_completion_length "${DPO_MAX_COMPLETION_LENGTH}" \
        --num_train_epochs "${DPO_NUM_TRAIN_EPOCHS}" \
        --per_device_train_batch_size "${DPO_PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${DPO_PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${DPO_GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${DPO_LEARNING_RATE}" \
        --logging_steps "${DPO_LOGGING_STEPS}" \
        --save_steps "${DPO_SAVE_STEPS}" \
        --eval_steps "${DPO_EVAL_STEPS}" \
        --lora_r "${DPO_LORA_R}" \
        --lora_alpha "${DPO_LORA_ALPHA}" \
        --lora_dropout "${DPO_LORA_DROPOUT}" \
        --seed "${SEED}" \
        "${DPO_BF16_ARGS[@]}" \
        "${DPO_4BIT_ARGS[@]}" \
        "${DPO_GRADIENT_CHECKPOINTING_ARGS[@]}"
  fi
fi

if [[ "${RUN_DPO_ANALYZER_GRPO}" == "1" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${DPO_ADAPTER_DIR}/adapter_config.json"
  fi
  if [[ "${DPO_ADAPTER_DIR}" != "${ANALYZER_PIPELINE_DIR}/dpo_adapter" ]]; then
    echo "[ERROR] DPO analyzer adapter path must point to the new full Qwen3 DPO adapter." >&2
    exit 1
  fi
  if should_skip_step "${DPO_GRPO_OUTPUT_DIR}/summary.json" "${DPO_GRPO_OUTPUT_DIR}/test/summary.json"; then
    echo "[INFO] Skipping solver GRPO with SFT+DPO analyzer because outputs already exist."
  else
    run_cmd_logged \
      "${STEP_LOG_ROOT}/11_grpo_bayesian_sft_dpo_analyzer.log" \
      env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      python3 run_grpo_bayesian_with_learned_analyzer.py \
        --analyzer_adapter_path "${DPO_ADAPTER_DIR}" \
        --analyzer_model_name "${ANALYZER_MODEL_NAME}" \
        --model_name "${MODEL_NAME}" \
        --metadata_dir "${METADATA_DIR}" \
        --output_dir "${DPO_GRPO_OUTPUT_DIR}" \
        --train_size "${TRAIN_SIZE}" \
        --eval_size "${EVAL_SIZE}" \
        --skip_valid_eval \
        --num_generations "${NUM_GENERATIONS}" \
        --max_prompt_length "${TRAIN_MAX_PROMPT_LENGTH}" \
        --max_steps "${MAX_STEPS}" \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${LEARNING_RATE}" \
        --lora_r "${SOLVER_LORA_R}" \
        --lora_alpha "${SOLVER_LORA_ALPHA}" \
        --lora_dropout "${SOLVER_LORA_DROPOUT}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --seed "${SEED}" \
        --prior_lambda "${PRIOR_LAMBDA}" \
        --prior_softmax_temperature "${PRIOR_SOFTMAX_TEMPERATURE}" \
        --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
        --progress_interval_percent "${PROGRESS_INTERVAL_PERCENT}" \
        --eval_batch_size "${LEARNED_EVAL_BATCH_SIZE}" \
        --eval_max_new_tokens "${LEARNED_EVAL_MAX_NEW_TOKENS}" \
        --eval_max_prompt_length "${LEARNED_EVAL_MAX_PROMPT_LENGTH}" \
        --skip_preflight_recompute \
        --step_logs_dir "${STEP_LOG_ROOT}/11_grpo_bayesian_sft_dpo_analyzer_steps" \
        --method_name "GRPO Bayesian reward + SFT+DPO analyzer" \
        --train_data_label "GSM8K official train full" \
        --analyzer_type learned_sft_dpo \
        --notes "deterministic official GSM8K full-test evaluation; ${PROMPT_FORMAT_NOTE}" \
        "${SOLVER_USE_LORA_ARGS[@]}" \
        "${SOLVER_GRADIENT_CHECKPOINTING_ARGS[@]}" \
        "${BF16_ARGS[@]}"
  fi
fi

if [[ "${RUN_COLLECT_RESULTS}" == "1" ]]; then
  run_cmd_logged \
    "${STEP_LOG_ROOT}/12_collect_results.log" \
    python3 collect_gsm8k_full_qwen3_1p7b_results.py \
      --root_dir "${EXPERIMENT_ROOT}" \
      --output_json "${COMPARISON_JSON}" \
      --output_csv "${COMPARISON_CSV}"
fi

echo "[DONE] GSM8K full Qwen3-1.7B pipeline resolved under ${EXPERIMENT_ROOT}"
