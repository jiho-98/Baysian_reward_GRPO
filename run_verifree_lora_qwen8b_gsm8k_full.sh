#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
ROOT="/home/kimjh/Baysian_reward_GRPO"
OUT="outputs/gsm8k_full_qwen3_8b/verifree_lora_qwen8b_fulltrain_n8_steps1000_bsz8_scale0p1"

cd "$ROOT"
mkdir -p "$OUT/logs"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python VeriFree_LoRA.py \
  --model_name Qwen/Qwen3-8B \
  --use_fixed_metadata \
  --train_metadata_path outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl \
  --train_size 7473 \
  --eval_size 0 \
  --num_generations 8 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_steps 1000 \
  --per_device_train_batch_size 8 \
  --generation_prompt_batch_size 2 \
  --mini_train_batch_size 1 \
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
