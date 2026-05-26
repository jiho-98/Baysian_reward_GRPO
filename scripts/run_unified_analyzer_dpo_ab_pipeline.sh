#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"

DEBUG_JSONL="${DEBUG_JSONL:-outputs/fair_bayesian_full_qwen3b_bigmath_3000_300_n8_steps500/bayesian_reward_debug.jsonl}"
EVIDENCE_VAL_PATH="${EVIDENCE_VAL_PATH:-outputs/unified_analyzer_sft_v0/evidence_clean_val_marked.jsonl}"
PRIOR_VAL_PATH="${PRIOR_VAL_PATH:-outputs/unified_analyzer_sft_v0/prior_clean_val_marked.jsonl}"
SFT_INIT_ADAPTER="${SFT_INIT_ADAPTER:-outputs/unified_analyzer_sft_v0_lora_retry1}"
BASELINE_SUMMARY_JSON="${BASELINE_SUMMARY_JSON:-outputs/learned_analyzer_posterior_recompute_v0_bs16_promptmatched/lambda_sweep/lambda_sweep_summary.json}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/analyzer_dpo_ab_runs/${RUN_TAG}}"
DPO_DATA_DIR="${DPO_DATA_DIR:-${RUN_ROOT}/dpo_data}"
OPTION_A_DIR="${OPTION_A_DIR:-${RUN_ROOT}/option_a_base_qwen_dpo}"
OPTION_A_FILTER_DIR="${OPTION_A_FILTER_DIR:-${RUN_ROOT}/option_a_base_qwen_filter}"
OPTION_B_DIR="${OPTION_B_DIR:-${RUN_ROOT}/option_b_sft_to_dpo}"
OPTION_B_FILTER_DIR="${OPTION_B_FILTER_DIR:-${RUN_ROOT}/option_b_sft_to_dpo_filter}"

EXPECTED_PRIOR_LAMBDA="${EXPECTED_PRIOR_LAMBDA:-1.0}"
LAMBDA_SWEEP="${LAMBDA_SWEEP:-0.5 0.7 1.0}"

TARGET_PRIOR_TRAIN_PAIRS="${TARGET_PRIOR_TRAIN_PAIRS:-800}"
TARGET_PRIOR_VAL_PAIRS="${TARGET_PRIOR_VAL_PAIRS:-120}"
TARGET_EVIDENCE_TRAIN_PAIRS="${TARGET_EVIDENCE_TRAIN_PAIRS:-800}"
TARGET_EVIDENCE_VAL_PAIRS="${TARGET_EVIDENCE_VAL_PAIRS:-120}"

DPO_NUM_EPOCHS="${DPO_NUM_EPOCHS:-1.0}"
DPO_LR="${DPO_LR:-5e-5}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_EVAL_BATCH_SIZE="${DPO_EVAL_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-8}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-200}"
DPO_EVAL_STEPS="${DPO_EVAL_STEPS:-200}"
DPO_LOGGING_STEPS="${DPO_LOGGING_STEPS:-10}"
DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-4096}"
DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-3584}"
DPO_MAX_COMPLETION_LENGTH="${DPO_MAX_COMPLETION_LENGTH:-512}"

FILTER_BATCH_SIZE="${FILTER_BATCH_SIZE:-8}"
FILTER_MAX_GROUPS="${FILTER_MAX_GROUPS:-0}"
FILTER_MAX_EXAMPLES_PER_TASK="${FILTER_MAX_EXAMPLES_PER_TASK:-0}"
FILTER_FALLBACK_MODE="${FILTER_FALLBACK_MODE:-neutral}"

USE_4BIT="${USE_4BIT:-1}"
USE_BF16="${USE_BF16:-1}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"

mkdir -p "$RUN_ROOT"

for required_path in \
  "$DEBUG_JSONL" \
  "$EVIDENCE_VAL_PATH" \
  "$PRIOR_VAL_PATH" \
  "$SFT_INIT_ADAPTER" \
  "$BASELINE_SUMMARY_JSON"
do
  if [[ ! -e "$required_path" ]]; then
    echo "[ERROR] missing required path: $required_path" >&2
    exit 1
  fi
done

COMMON_TRAIN_FLAGS=(
  --model_name "$MODEL_NAME"
  --train_path "${DPO_DATA_DIR}/unified_dpo_train.jsonl"
  --val_path "${DPO_DATA_DIR}/unified_dpo_val.jsonl"
  --num_train_epochs "$DPO_NUM_EPOCHS"
  --learning_rate "$DPO_LR"
  --beta "$DPO_BETA"
  --per_device_train_batch_size "$DPO_BATCH_SIZE"
  --per_device_eval_batch_size "$DPO_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$DPO_GRAD_ACCUM"
  --save_steps "$DPO_SAVE_STEPS"
  --eval_steps "$DPO_EVAL_STEPS"
  --logging_steps "$DPO_LOGGING_STEPS"
  --max_length "$DPO_MAX_LENGTH"
  --max_prompt_length "$DPO_MAX_PROMPT_LENGTH"
  --max_completion_length "$DPO_MAX_COMPLETION_LENGTH"
)

if [[ "$USE_4BIT" == "1" ]]; then
  COMMON_TRAIN_FLAGS+=(--use_4bit)
fi
if [[ "$USE_BF16" == "1" ]]; then
  COMMON_TRAIN_FLAGS+=(--bf16)
fi
if [[ "$USE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  COMMON_TRAIN_FLAGS+=(--gradient_checkpointing)
fi

COMMON_FILTER_FLAGS=(
  --model_name "$MODEL_NAME"
  --input_debug_jsonl "$DEBUG_JSONL"
  --evidence_val_path "$EVIDENCE_VAL_PATH"
  --prior_val_path "$PRIOR_VAL_PATH"
  --baseline_summary_json "$BASELINE_SUMMARY_JSON"
  --batch_size "$FILTER_BATCH_SIZE"
  --fallback_mode "$FILTER_FALLBACK_MODE"
  --lambdas ${LAMBDA_SWEEP}
)

if [[ "$FILTER_MAX_GROUPS" != "0" ]]; then
  COMMON_FILTER_FLAGS+=(--max_groups "$FILTER_MAX_GROUPS")
fi
if [[ "$FILTER_MAX_EXAMPLES_PER_TASK" != "0" ]]; then
  COMMON_FILTER_FLAGS+=(--max_examples_per_task "$FILTER_MAX_EXAMPLES_PER_TASK")
fi
if [[ "$USE_BF16" == "1" ]]; then
  COMMON_FILTER_FLAGS+=(--bf16)
fi

echo "[INFO] run_root=$RUN_ROOT"
echo "[INFO] debug_jsonl=$DEBUG_JSONL"
echo "[INFO] sft_init_adapter=$SFT_INIT_ADAPTER"
echo "[INFO] baseline_summary_json=$BASELINE_SUMMARY_JSON"

"$PYTHON_BIN" prepare_unified_analyzer_dpo.py \
  --input_debug_jsonl "$DEBUG_JSONL" \
  --expected_prior_lambda "$EXPECTED_PRIOR_LAMBDA" \
  --target_prior_train_pairs "$TARGET_PRIOR_TRAIN_PAIRS" \
  --target_prior_val_pairs "$TARGET_PRIOR_VAL_PAIRS" \
  --target_evidence_train_pairs "$TARGET_EVIDENCE_TRAIN_PAIRS" \
  --target_evidence_val_pairs "$TARGET_EVIDENCE_VAL_PAIRS" \
  --output_dir "$DPO_DATA_DIR"

"$PYTHON_BIN" train_unified_analyzer_dpo.py \
  "${COMMON_TRAIN_FLAGS[@]}" \
  --output_dir "$OPTION_A_DIR"

"$PYTHON_BIN" run_unified_analyzer_dpo_filter.py \
  "${COMMON_FILTER_FLAGS[@]}" \
  --adapter_path "$OPTION_A_DIR" \
  --output_dir "$OPTION_A_FILTER_DIR"

"$PYTHON_BIN" train_unified_analyzer_dpo.py \
  "${COMMON_TRAIN_FLAGS[@]}" \
  --init_adapter_path "$SFT_INIT_ADAPTER" \
  --output_dir "$OPTION_B_DIR"

"$PYTHON_BIN" run_unified_analyzer_dpo_filter.py \
  "${COMMON_FILTER_FLAGS[@]}" \
  --adapter_path "$OPTION_B_DIR" \
  --output_dir "$OPTION_B_FILTER_DIR"

echo "[INFO] completed Option A and Option B"
echo "[INFO] option_a_dir=$OPTION_A_DIR"
echo "[INFO] option_a_filter=$OPTION_A_FILTER_DIR/filter_summary.json"
echo "[INFO] option_b_dir=$OPTION_B_DIR"
echo "[INFO] option_b_filter=$OPTION_B_FILTER_DIR/filter_summary.json"
