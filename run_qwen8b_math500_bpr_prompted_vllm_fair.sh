#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

GPU="${GPU:-0}"
PY="$PWD/.venv/bin/python"

TRAIN_METADATA="outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl"
EVAL_METADATA="outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl"
OUT="outputs/math500_experiments/grpo_bayesian_prompted_qwen8b_fulltrain12k_n8_steps1500_bsz4_acc2_lambda1_vllm4096"

test -s "$TRAIN_METADATA"
test -s "$EVAL_METADATA"
mkdir -p "$OUT/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" Bayesian_Full_GRPO.py \
  --model_name Qwen/Qwen3-8B \
  --dataset_name ricdomolm/MATH-500 \
  --use_fixed_metadata \
  --train_metadata_path "$TRAIN_METADATA" \
  --eval_metadata_path "$EVAL_METADATA" \
  --train_size 12000 \
  --eval_size 0 \
  --num_generations 8 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_steps 1500 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --min_solve_rate 0.0 \
  --max_solve_rate 1.0 \
  --prior_mode llm_strategy_prior \
  --prior_lambda 1.0 \
  --prior_softmax_temperature 1.0 \
  --prior_judge_model Qwen/Qwen3-8B \
  --evidence_judge_model Qwen/Qwen3-8B \
  --prior_judge_temperature 0.0 \
  --evidence_judge_temperature 0.0 \
  --judge_max_new_tokens 768 \
  --format_bonus 0.0 \
  --seed 42 \
  --logging_steps 10 \
  --save_steps 100 \
  --progress_interval_percent 10 \
  --bf16 \
  --use_lora \
  --gradient_checkpointing \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.20 \
  --vllm_max_model_length 4096 \
  --vllm_tensor_parallel_size 1 \
  --output_dir "$OUT" \
  --reward_debug_jsonl "$OUT/bayesian_reward_debug.jsonl"
