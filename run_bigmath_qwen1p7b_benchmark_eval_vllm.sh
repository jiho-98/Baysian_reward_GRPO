#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

trap 'echo "[ERROR] line=${LINENO} status=$?"' ERR

GPU="${GPU:-1}"
PY="$PWD/.venv/bin/python"

MODEL="Qwen/Qwen3-1.7B"
ADAPTER="outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500"
BASE_OUT="outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096"

mkdir -p "$BASE_OUT/logs"

run_eval() {
  local name="$1"
  local metadata="$2"
  local out_dir="$BASE_OUT/$name"
  local log_path="$BASE_OUT/logs/${name}.log"

  mkdir -p "$out_dir"
  echo "========== ${name} $(date) =========="

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval_solver_checkpoint.py \
    --model_name "$MODEL" \
    --adapter_path "$ADAPTER" \
    --eval_metadata_path "$metadata" \
    --output_dir "$out_dir" \
    --batch_size 64 \
    --max_examples 0 \
    --max_prompt_length 2048 \
    --max_new_tokens 4096 \
    --no_do_sample \
    --seed 42 \
    --use_vllm \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_length 6144 \
    2>&1 | tee "$log_path"
}

run_eval "aime26" "outputs/eval_benchmarks/aime26_metadata.jsonl"
run_eval "minervamath" "outputs/eval_benchmarks/minervamath_metadata.jsonl"
run_eval "olympiadbench" "outputs/eval_benchmarks/olympiadbench_metadata.jsonl"
