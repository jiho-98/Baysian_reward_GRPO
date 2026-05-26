#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kimjh/Baysian_reward_GRPO"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/Answer_only_GRPO.py"
BASE_OUT="outputs/drgrpo_answer_only_24h"
MAX_STEPS="${MAX_STEPS:-500}"
BSZ="${BSZ:-8}"
ACC="${ACC:-1}"

cd "$ROOT"
mkdir -p "$BASE_OUT"

run_one() {
  local gpu="$1"
  local model_name="$2"
  local train_path="$3"
  local eval_path="$4"
  local train_size="$5"
  local max_prompt_length="$6"
  local output_dir="$7"

  mkdir -p "$output_dir/logs"
  echo "[START] $(date '+%Y-%m-%d %H:%M:%S') gpu=$gpu model=$model_name output=$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    --model_name "$model_name" \
    --use_fixed_metadata \
    --train_metadata_path "$train_path" \
    --eval_metadata_path "$eval_path" \
    --train_size "$train_size" \
    --eval_size 0 \
    --num_generations 8 \
    --max_prompt_length "$max_prompt_length" \
    --max_completion_length 1024 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$BSZ" \
    --gradient_accumulation_steps "$ACC" \
    --learning_rate 5e-6 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --min_solve_rate 0.0 \
    --max_solve_rate 1.0 \
    --seed 42 \
    --logging_steps 10 \
    --save_steps 100 \
    --progress_interval_percent 10 \
    --bf16 \
    --use_lora \
    --gradient_checkpointing \
    --loss_type dr_grpo \
    --scale_rewards none \
    --beta 0.0 \
    --output_dir "$output_dir" \
    > "$output_dir/logs/train.nohup.log" 2>&1
  echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S') gpu=$gpu output=$output_dir"
}

queue_gpu0() {
  run_one \
    0 \
    "Qwen/Qwen3-4B" \
    "outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl" \
    "outputs/eval_benchmarks/olympiadbench_metadata.jsonl" \
    12288 \
    2048 \
    "$BASE_OUT/qwen3_4b_bigmath_steps${MAX_STEPS}_n8_bsz${BSZ}_acc${ACC}"

  run_one \
    0 \
    "Qwen/Qwen3-1.7B" \
    "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl" \
    "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl" \
    12000 \
    2048 \
    "$BASE_OUT/qwen3_1p7b_math500_steps${MAX_STEPS}_n8_bsz${BSZ}_acc${ACC}"

  run_one \
    0 \
    "Qwen/Qwen3-1.7B" \
    "outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl" \
    "outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl" \
    7473 \
    1024 \
    "$BASE_OUT/qwen3_1p7b_gsm8k_steps${MAX_STEPS}_n8_bsz${BSZ}_acc${ACC}"
}

queue_gpu1() {
  run_one \
    1 \
    "Qwen/Qwen3-4B" \
    "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl" \
    "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl" \
    12000 \
    2048 \
    "$BASE_OUT/qwen3_4b_math500_steps${MAX_STEPS}_n8_bsz${BSZ}_acc${ACC}"

  run_one \
    1 \
    "Qwen/Qwen3-4B" \
    "outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl" \
    "outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl" \
    7473 \
    1024 \
    "$BASE_OUT/qwen3_4b_gsm8k_steps${MAX_STEPS}_n8_bsz${BSZ}_acc${ACC}"

  run_one \
    1 \
    "Qwen/Qwen3-1.7B" \
    "outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl" \
    "outputs/eval_benchmarks/olympiadbench_metadata.jsonl" \
    12288 \
    2048 \
    "$BASE_OUT/qwen3_1p7b_bigmath_steps${MAX_STEPS}_n8_bsz${BSZ}_acc${ACC}"
}

queue_gpu0 > "$BASE_OUT/gpu0_queue.log" 2>&1 &
pid0=$!
queue_gpu1 > "$BASE_OUT/gpu1_queue.log" 2>&1 &
pid1=$!

echo "[QUEUES] gpu0_pid=$pid0 gpu1_pid=$pid1"
wait "$pid0" "$pid1"
echo "[ALL DONE] $(date '+%Y-%m-%d %H:%M:%S')"
