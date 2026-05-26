#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/kimjh/EMNLP"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-3}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
STACK_FILE="${STACK_FILE:-RUNNING_TRAINING_STACK.md}"
RUN_ROOT="outputs/drgrpo_answer_only_24h"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S %Z"
}

append_stack_start() {
  local job_name="$1"
  local model_name="$2"
  local output_dir="$3"
  local train_metadata="$4"
  local eval_metadata="$5"
  local train_size="$6"
  local max_prompt_length="$7"
  cat >> "$STACK_FILE" <<EOF

## Train - ${job_name}

- Added at: $(timestamp)
- Status: running
- Server: EMNLP / GPU${GPU_ID}
- Script: \`Answer_only_GRPO.py\`
- Model: \`${model_name}\`
- Method: Dr.GRPO external baseline
- Reward: answer-only correctness
- loss_type: \`dr_grpo\`
- scale_rewards: \`none\`
- beta: \`0.0\`
- output_dir: \`${output_dir}\`
- train_metadata_path: \`${train_metadata}\`
- eval_metadata_path: \`${eval_metadata}\`
- train_size: ${train_size}
- eval_size: 0
- num_generations: 8
- max_steps: 1500
- per_device_train_batch_size: 8
- gradient_accumulation_steps: 1
- learning_rate: 5e-6
- max_prompt_length: ${max_prompt_length}
- max_completion_length: 1024
- LoRA: r16 / alpha32 / dropout0.05
- bf16: true
- gradient_checkpointing: true
- seed: 42
- log: \`${output_dir}/logs/train.nohup.log\`
EOF
}

append_stack_finish() {
  local job_name="$1"
  local output_dir="$2"
  local status="$3"
  local started_at="$4"
  local finished_at checkpoints final_checkpoint
  finished_at="$(timestamp)"
  checkpoints="$(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f ' 2>/dev/null | xargs || true)"
  final_checkpoint="$(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1 || true)"
  cat >> "$STACK_FILE" <<EOF

### Finish - ${job_name}

- Finished at: ${finished_at}
- Started at: ${started_at}
- Status: ${status}
- output_dir: \`${output_dir}\`
- final_checkpoint: \`${final_checkpoint:-none}\`
- checkpoint_list: ${checkpoints:-none}
EOF
}

run_drgrpo_job() {
  local job_name="$1"
  local model_name="$2"
  local train_metadata="$3"
  local eval_metadata="$4"
  local train_size="$5"
  local max_prompt_length="$6"
  local output_dir="$7"

  mkdir -p "$output_dir/logs"
  local started_at
  started_at="$(timestamp)"
  append_stack_start "$job_name" "$model_name" "$output_dir" "$train_metadata" "$eval_metadata" "$train_size" "$max_prompt_length"
  echo "[$started_at] START ${job_name} on GPU${GPU_ID}"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" Answer_only_GRPO.py \
    --model_name "$model_name" \
    --dataset_name fixed_metadata \
    --use_fixed_metadata \
    --train_metadata_path "$train_metadata" \
    --eval_metadata_path "$eval_metadata" \
    --train_size "$train_size" \
    --eval_size 0 \
    --num_generations 8 \
    --max_prompt_length "$max_prompt_length" \
    --max_completion_length 1024 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_steps 1500 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
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
  local exit_code=$?
  set -e

  if [ "$exit_code" -eq 0 ]; then
    append_stack_finish "$job_name" "$output_dir" "completed" "$started_at"
  else
    append_stack_finish "$job_name" "$output_dir" "failed_exit_${exit_code}" "$started_at"
    return "$exit_code"
  fi
}

mkdir -p "$RUN_ROOT/logs"

run_drgrpo_job \
  "Qwen3-1.7B Dr.GRPO Answer-only BigMath BARL-style" \
  "Qwen/Qwen3-1.7B" \
  "outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl" \
  "outputs/eval_benchmarks/olympiadbench_metadata.jsonl" \
  12288 \
  2048 \
  "${RUN_ROOT}/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1"

run_drgrpo_job \
  "Qwen3-1.7B Dr.GRPO Answer-only MATH-500" \
  "Qwen/Qwen3-1.7B" \
  "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl" \
  "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl" \
  12000 \
  2048 \
  "${RUN_ROOT}/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1"

run_drgrpo_job \
  "Qwen3-1.7B Dr.GRPO Answer-only GSM8K" \
  "Qwen/Qwen3-1.7B" \
  "outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl" \
  "outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl" \
  7473 \
  1024 \
  "${RUN_ROOT}/qwen3_1p7b_gsm8k_steps1500_n8_bsz8_acc1"

run_drgrpo_job \
  "Qwen3-4B Dr.GRPO Answer-only GSM8K" \
  "Qwen/Qwen3-4B" \
  "outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl" \
  "outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl" \
  7473 \
  1024 \
  "${RUN_ROOT}/qwen3_4b_gsm8k_steps1500_n8_bsz8_acc1"
