#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Learned Unified Analyzer Bayesian GRPO Solver Training
# 1) Train lambda=0.7
# 2) Train lambda=1.0
# 3) Evaluate both final adapters on the same fixed eval 300 set
# ============================================================

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
ANALYZER_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
ANALYZER_ADAPTER_PATH="outputs/unified_analyzer_sft_v0_lora_retry1"

TRAIN_METADATA="outputs/fair_bigmath_3000_300_seed42/selected_train_metadata.jsonl"
EVAL_METADATA="outputs/fair_bigmath_3000_300_seed42/selected_eval_metadata.jsonl"

BASE_OUTPUT_ROOT="outputs/learned_analyzer_solver_grpo"
LOG_DIR="${BASE_OUTPUT_ROOT}/logs"
mkdir -p "${BASE_OUTPUT_ROOT}" "${LOG_DIR}"

# Fair comparison settings: keep these identical to prior Bayesian Full run.
TRAIN_SIZE=3000
EVAL_SIZE=300
NUM_GENERATIONS=8
MAX_STEPS=500
MAX_COMPLETION_LENGTH=1024
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE="5e-6"
MIN_SOLVE_RATE="0.2"
MAX_SOLVE_RATE="0.8"
SEED=42
PRIOR_SOFTMAX_TEMPERATURE="1.0"
JUDGE_MAX_NEW_TOKENS=768
PROGRESS_INTERVAL_PERCENT=10

# Eval settings. Deterministic eval means changing eval batch size should not change accuracy.
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_MAX_NEW_TOKENS=1024
EVAL_MAX_PROMPT_LENGTH=2048

require_file () {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required file/path: ${path}" >&2
    exit 1
  fi
}

require_file "Bayesian_Full_GRPO_learned.py"
require_file "eval_solver_checkpoint.py"
require_file "${TRAIN_METADATA}"
require_file "${EVAL_METADATA}"
require_file "${ANALYZER_ADAPTER_PATH}"

python3 -m py_compile Bayesian_Full_GRPO_learned.py
python3 -m py_compile eval_solver_checkpoint.py

run_train_one_lambda () {
  local LAMBDA="$1"
  local TAG="$2"

  local OUT_DIR="${BASE_OUTPUT_ROOT}/qwen3b_bigmath_3000_300_n8_steps500_lambda${TAG}"
  local LOG_FILE="${LOG_DIR}/train_lambda${TAG}.log"
  local DEBUG_PATH="${OUT_DIR}/bayesian_reward_debug.jsonl"

  echo "============================================================"
  echo "[TRAIN START] prior_lambda=${LAMBDA}"
  echo "[TRAIN OUT]   ${OUT_DIR}"
  echo "[TRAIN LOG]   ${LOG_FILE}"
  echo "============================================================"

  mkdir -p "${OUT_DIR}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  python3 Bayesian_Full_GRPO_learned.py \
    --model_name "${MODEL_NAME}" \
    --prior_mode learned_unified_analyzer \
    --analyzer_model_name "${ANALYZER_MODEL_NAME}" \
    --analyzer_adapter_path "${ANALYZER_ADAPTER_PATH}" \
    --use_fixed_metadata \
    --train_metadata_path "${TRAIN_METADATA}" \
    --eval_metadata_path "${EVAL_METADATA}" \
    --train_size "${TRAIN_SIZE}" \
    --eval_size "${EVAL_SIZE}" \
    --num_generations "${NUM_GENERATIONS}" \
    --max_steps "${MAX_STEPS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --min_solve_rate "${MIN_SOLVE_RATE}" \
    --max_solve_rate "${MAX_SOLVE_RATE}" \
    --seed "${SEED}" \
    --prior_lambda "${LAMBDA}" \
    --prior_softmax_temperature "${PRIOR_SOFTMAX_TEMPERATURE}" \
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
    --progress_interval_percent "${PROGRESS_INTERVAL_PERCENT}" \
    --output_dir "${OUT_DIR}" \
    --reward_debug_jsonl "${DEBUG_PATH}" \
    2>&1 | tee "${LOG_FILE}"

  echo "============================================================"
  echo "[TRAIN DONE] prior_lambda=${LAMBDA}"
  echo "[TRAIN OUT]  ${OUT_DIR}"
  echo "============================================================"
}

run_eval_one_lambda () {
  local LAMBDA="$1"
  local TAG="$2"

  local ADAPTER_DIR="${BASE_OUTPUT_ROOT}/qwen3b_bigmath_3000_300_n8_steps500_lambda${TAG}"
  local EVAL_OUT="${ADAPTER_DIR}/eval_eval300_bs${EVAL_BATCH_SIZE}_deterministic"
  local LOG_FILE="${LOG_DIR}/eval_lambda${TAG}_bs${EVAL_BATCH_SIZE}.log"

  require_file "${ADAPTER_DIR}"

  echo "============================================================"
  echo "[EVAL START] prior_lambda=${LAMBDA}"
  echo "[ADAPTER]    ${ADAPTER_DIR}"
  echo "[EVAL OUT]   ${EVAL_OUT}"
  echo "[EVAL LOG]   ${LOG_FILE}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  python3 eval_solver_checkpoint.py \
    --model_name "${MODEL_NAME}" \
    --adapter_path "${ADAPTER_DIR}" \
    --eval_metadata_path "${EVAL_METADATA}" \
    --output_dir "${EVAL_OUT}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
    --max_prompt_length "${EVAL_MAX_PROMPT_LENGTH}" \
    --seed "${SEED}" \
    --no_do_sample \
    2>&1 | tee "${LOG_FILE}"

  echo "============================================================"
  echo "[EVAL DONE] prior_lambda=${LAMBDA}"
  echo "============================================================"
}

# -------------------------
# Training: lambda 0.7 then 1.0
# -------------------------
run_train_one_lambda "0.7" "07"
run_train_one_lambda "1.0" "10"

# -------------------------
# Final deterministic eval on the exact same fixed eval 300 set
# -------------------------
run_eval_one_lambda "0.7" "07"
run_eval_one_lambda "1.0" "10"

# -------------------------
# Compact comparison summary
# -------------------------
python3 - <<'PY'
import json
from pathlib import Path

root = Path("outputs/learned_analyzer_solver_grpo")
items = [
    ("lambda07", root / "qwen3b_bigmath_3000_300_n8_steps500_lambda07" / "eval_eval300_bs16_deterministic" / "summary.json"),
    ("lambda10", root / "qwen3b_bigmath_3000_300_n8_steps500_lambda10" / "eval_eval300_bs16_deterministic" / "summary.json"),
]

# In case EVAL_BATCH_SIZE was not 16, find latest summary under each eval directory.
for tag, default_path in items:
    paths = sorted((root / f"qwen3b_bigmath_3000_300_n8_steps500_{tag.replace('lambda', 'lambda')}").glob("eval_eval300_bs*_deterministic/summary.json"))

print("\n================ FINAL COMPARISON ================")
for name, model_dir in [
    ("lambda=0.7", root / "qwen3b_bigmath_3000_300_n8_steps500_lambda07"),
    ("lambda=1.0", root / "qwen3b_bigmath_3000_300_n8_steps500_lambda10"),
]:
    summaries = sorted(model_dir.glob("eval_eval300_bs*_deterministic/summary.json"))
    if not summaries:
        print(f"{name}: summary not found")
        continue
    path = summaries[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"{name}: accuracy={data.get('accuracy')} correct={data.get('correct')}/{data.get('num_examples')} summary={path}")
print("==================================================")
PY

echo "[FINISHED] Learned Analyzer Solver GRPO lambda sweep completed."
