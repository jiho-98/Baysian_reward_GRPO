#!/usr/bin/env bash
set -euo pipefail

cd /home/kimjh/EMNLP
source /home/kimjh/EMNLP/.venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export VLLM_USE_FLASHINFER_SAMPLER=0

ROOT="outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm"

mkdir -p "${ROOT}/eval_benchmarks/logs" \
  "${ROOT}/eval_benchmarks/aime26_max4096_vllm_all4096_retest" \
  "${ROOT}/eval_benchmarks/minervamath_max4096_vllm_all4096_retest" \
  "${ROOT}/eval_benchmarks/olympiadbench_max4096_vllm_all4096_retest"

run_eval() {
  local name="$1"
  local metadata_path="$2"
  local output_dir="$3"
  local log_path="$4"

  echo "[START] ${name} $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  python3 eval_solver_checkpoint.py \
    --model_name Qwen/Qwen3-1.7B \
    --adapter_path "${ROOT}" \
    --eval_metadata_path "${metadata_path}" \
    --output_dir "${output_dir}" \
    --batch_size 64 \
    --max_prompt_length 2048 \
    --max_new_tokens 4096 \
    --temperature 0.0 \
    --top_p 1.0 \
    --seed 42 \
    --bf16 \
    --load_adapter \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_length 6144 \
    --no_do_sample \
    > "${log_path}" 2>&1
  echo "[DONE] ${name} $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
}

run_eval \
  "aime26_all4096" \
  "outputs/eval_benchmarks/aime26_metadata.jsonl" \
  "${ROOT}/eval_benchmarks/aime26_max4096_vllm_all4096_retest" \
  "${ROOT}/eval_benchmarks/logs/aime26_all4096.log"

run_eval \
  "minervamath_all4096" \
  "outputs/eval_benchmarks/minervamath_metadata.jsonl" \
  "${ROOT}/eval_benchmarks/minervamath_max4096_vllm_all4096_retest" \
  "${ROOT}/eval_benchmarks/logs/minervamath_all4096.log"

run_eval \
  "olympiadbench_all4096" \
  "outputs/eval_benchmarks/olympiadbench_metadata.jsonl" \
  "${ROOT}/eval_benchmarks/olympiadbench_max4096_vllm_all4096_retest" \
  "${ROOT}/eval_benchmarks/logs/olympiadbench_all4096.log"

echo "[ALL DONE] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
