#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
ROOT="/home/kimjh/Baysian_reward_GRPO"
OUT="outputs/math500_experiments/verifree_lora_qwen4b_fulltrain12k_n8_steps1500_bsz8_scale0p1"

cd "$ROOT"
mkdir -p "$OUT/logs"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python VeriFree_LoRA.py \
  --model_name Qwen/Qwen3-4B \
  --use_fixed_metadata \
  --train_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl \
  --train_size 12000 \
  --eval_size 0 \
  --num_generations 8 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_steps 1500 \
  --per_device_train_batch_size 8 \
  --generation_prompt_batch_size 4 \
  --mini_train_batch_size 2 \
  --learning_rate 5e-6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --seed 42 \
  --logging_steps 10 \
  --save_steps 100 \
  --progress_interval_percent 10 \
  --bf16 \
  --use_lora \
  --gradient_checkpointing \
  --reward_source p \
  --reward_scale 0.1 \
  --sft_coef_source reward \
  --advantage_type rloo \
  --output_dir "$OUT"
