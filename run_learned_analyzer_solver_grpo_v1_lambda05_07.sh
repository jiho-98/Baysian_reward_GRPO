#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Co-evolve validation with v1 learned analyzer on solver GRPO
# - same fair split: 3000 train / 300 eval
# - same rollout count: n=8
# - train/eval two lambda settings for v1 analyzer:
#   1) lambda=0.7 for same-lambda analyzer comparison vs v0
#   2) lambda=0.5 for v1-recalibrated best candidate
# ============================================================

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
ANALYZER_MODEL_NAME="${ANALYZER_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
ANALYZER_ADAPTER_PATH="${ANALYZER_ADAPTER_PATH:-outputs/unified_analyzer_sft_v1_lambda07_bootstrap_lora}"

TRAIN_METADATA="${TRAIN_METADATA:-outputs/fair_bigmath_3000_300_seed42/selected_train_metadata.jsonl}"
EVAL_METADATA="${EVAL_METADATA:-outputs/fair_bigmath_3000_300_seed42/selected_eval_metadata.jsonl}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/learned_analyzer_solver_grpo_v1}"
LOG_DIR="${BASE_OUTPUT_ROOT}/logs"
mkdir -p "${BASE_OUTPUT_ROOT}" "${LOG_DIR}"

# Fair comparison settings: keep aligned with the existing solver GRPO runs.
TRAIN_SIZE="${TRAIN_SIZE:-3000}"
EVAL_SIZE="${EVAL_SIZE:-300}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MIN_SOLVE_RATE="${MIN_SOLVE_RATE:-0.2}"
MAX_SOLVE_RATE="${MAX_SOLVE_RATE:-0.8}"
SEED="${SEED:-42}"
PRIOR_SOFTMAX_TEMPERATURE="${PRIOR_SOFTMAX_TEMPERATURE:-1.0}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-768}"
PROGRESS_INTERVAL_PERCENT="${PROGRESS_INTERVAL_PERCENT:-10}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-2048}"

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
  local lambda_value="$1"
  local lambda_tag="$2"

  local out_dir="${BASE_OUTPUT_ROOT}/qwen3b_bigmath_3000_300_n8_steps500_v1_lambda${lambda_tag}"
  local log_file="${LOG_DIR}/train_v1_lambda${lambda_tag}.log"
  local debug_path="${out_dir}/bayesian_reward_debug.jsonl"

  echo "============================================================"
  echo "[TRAIN START] analyzer=v1 prior_lambda=${lambda_value}"
  echo "[ANALYZER]    ${ANALYZER_ADAPTER_PATH}"
  echo "[TRAIN OUT]   ${out_dir}"
  echo "[TRAIN LOG]   ${log_file}"
  echo "============================================================"

  mkdir -p "${out_dir}"

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
    --prior_lambda "${lambda_value}" \
    --prior_softmax_temperature "${PRIOR_SOFTMAX_TEMPERATURE}" \
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
    --progress_interval_percent "${PROGRESS_INTERVAL_PERCENT}" \
    --output_dir "${out_dir}" \
    --reward_debug_jsonl "${debug_path}" \
    2>&1 | tee "${log_file}"

  echo "============================================================"
  echo "[TRAIN DONE] analyzer=v1 prior_lambda=${lambda_value}"
  echo "[TRAIN OUT]  ${out_dir}"
  echo "============================================================"
}

run_eval_one_lambda () {
  local lambda_value="$1"
  local lambda_tag="$2"

  local adapter_dir="${BASE_OUTPUT_ROOT}/qwen3b_bigmath_3000_300_n8_steps500_v1_lambda${lambda_tag}"
  local eval_out="${adapter_dir}/eval_eval300_bs${EVAL_BATCH_SIZE}_deterministic"
  local log_file="${LOG_DIR}/eval_v1_lambda${lambda_tag}_bs${EVAL_BATCH_SIZE}.log"

  require_file "${adapter_dir}"

  echo "============================================================"
  echo "[EVAL START] analyzer=v1 prior_lambda=${lambda_value}"
  echo "[ADAPTER]    ${adapter_dir}"
  echo "[EVAL OUT]   ${eval_out}"
  echo "[EVAL LOG]   ${log_file}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  python3 eval_solver_checkpoint.py \
    --model_name "${MODEL_NAME}" \
    --adapter_path "${adapter_dir}" \
    --eval_metadata_path "${EVAL_METADATA}" \
    --output_dir "${eval_out}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
    --max_prompt_length "${EVAL_MAX_PROMPT_LENGTH}" \
    --seed "${SEED}" \
    --no_do_sample \
    2>&1 | tee "${log_file}"

  echo "============================================================"
  echo "[EVAL DONE] analyzer=v1 prior_lambda=${lambda_value}"
  echo "============================================================"
}

run_train_one_lambda "0.7" "07"
run_train_one_lambda "0.5" "05"

run_eval_one_lambda "0.7" "07"
run_eval_one_lambda "0.5" "05"

python3 - <<'PY'
import json
import os
from pathlib import Path

base_output_root = Path(os.environ.get("BASE_OUTPUT_ROOT", "outputs/learned_analyzer_solver_grpo_v1"))

def load_summary(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def latest_summary(model_dir: Path):
    candidates = sorted(model_dir.glob("eval_eval300_bs*_deterministic/summary.json"))
    if not candidates:
        return None
    return candidates[-1]

rows = [
    (
        "v0_lambda07_baseline",
        latest_summary(
            Path("outputs/learned_analyzer_solver_grpo/qwen3b_bigmath_3000_300_n8_steps500_lambda07")
        ),
    ),
    (
        "v1_lambda07_same_lambda",
        latest_summary(base_output_root / "qwen3b_bigmath_3000_300_n8_steps500_v1_lambda07"),
    ),
    (
        "v1_lambda05_recalibrated",
        latest_summary(base_output_root / "qwen3b_bigmath_3000_300_n8_steps500_v1_lambda05"),
    ),
]

comparison = []
print("\n================ V1 CO-EVOLVE COMPARISON ================")
for name, path in rows:
    if path is None:
        print(f"{name}: summary not found")
        continue
    data = load_summary(path)
    if data is None:
        print(f"{name}: summary not found at {path}")
        continue
    record = {
        "name": name,
        "accuracy": data.get("accuracy"),
        "correct": data.get("correct"),
        "num_examples": data.get("num_examples"),
        "summary_path": str(path),
    }
    comparison.append(record)
    print(
        f"{name}: accuracy={record['accuracy']} "
        f"correct={record['correct']}/{record['num_examples']} "
        f"summary={record['summary_path']}"
    )

out_path = base_output_root / "v0_vs_v1_lambda07_lambda05_comparison.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"comparison_json={out_path}")
print("=========================================================")
PY

echo "[FINISHED] v1 analyzer solver GRPO lambda(0.7,0.5) sweep completed."
