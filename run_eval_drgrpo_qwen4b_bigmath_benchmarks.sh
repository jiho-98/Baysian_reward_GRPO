#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PY="${PY:-.venv/bin/python}"

ADAPTER="outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/checkpoint-1500"
OUT="outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok"

mkdir -p "${OUT}/logs"

echo "[INFO] start Qwen3-4B Dr.GRPO BigMath eval on GPU=${GPU}"
echo "[INFO] adapter=${ADAPTER}"
echo "[INFO] output=${OUT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path "${ADAPTER}" \
  --eval_metadata_path outputs/eval_benchmarks/aime26_metadata.jsonl \
  --output_dir "${OUT}/aime26" \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 4096 \
  --temperature 0.0 \
  --top_p 1.0 \
  --no_do_sample \
  --seed 42 \
  --no_use_vllm \
  > "${OUT}/logs/aime26.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path "${ADAPTER}" \
  --eval_metadata_path outputs/eval_benchmarks/minervamath_metadata.jsonl \
  --output_dir "${OUT}/minervamath" \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --no_do_sample \
  --seed 42 \
  --no_use_vllm \
  > "${OUT}/logs/minervamath.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path "${ADAPTER}" \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --output_dir "${OUT}/olympiadbench" \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --no_do_sample \
  --seed 42 \
  --no_use_vllm \
  > "${OUT}/logs/olympiadbench.log" 2>&1

echo "[DONE] Qwen3-4B Dr.GRPO BigMath eval finished"
