#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

GPU="${GPU:-1}"
PY="$PWD/.venv/bin/python"
MODEL="Qwen/Qwen3-8B"
ANALYZER_MODEL="Qwen/Qwen3-8B"

PROMPTED_DIR="outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm"
DEBUG_JSONL="$PROMPTED_DIR/bayesian_reward_debug.jsonl"
METADATA_DIR="outputs/gsm8k_full_train_seed42"
TRAIN_METADATA="$METADATA_DIR/selected_train_metadata.jsonl"
TEST_METADATA="$METADATA_DIR/selected_test_metadata.jsonl"

PIPELINE_ROOT="outputs/gsm8k_experiments/bpr_grpo_learned_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz4_acc2_vllm_pipeline"
SFT_DATA_DIR="$PIPELINE_ROOT/analyzer_sft_data"
SFT_ADAPTER_DIR="$PIPELINE_ROOT/analyzer_sft_adapter"
SFT_RECOMPUTE_DIR="$PIPELINE_ROOT/analyzer_sft_recompute"
DPO_DATA_DIR="$PIPELINE_ROOT/analyzer_dpo_data"
DPO_ADAPTER_DIR="$PIPELINE_ROOT/analyzer_sft_dpo_adapter"
GRPO_OUTPUT_DIR="outputs/gsm8k_experiments/bpr_grpo_learned_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz4_acc2_vllm"
STEP_LOG_DIR="$PIPELINE_ROOT/logs/steps"

mkdir -p "$STEP_LOG_DIR" "$GRPO_OUTPUT_DIR"
test "$(wc -l < "$DEBUG_JSONL")" -ge 8000

run_step () {
  local name="$1"; shift
  echo; echo "========== $name $(date) =========="
  "$@" 2>&1 | tee "$STEP_LOG_DIR/${name}.log"
}

run_step 01_build_sft_data env CUDA_VISIBLE_DEVICES="$GPU" "$PY" build_analyzer_sft_data_from_gsm8k_logs.py \
  --log_dir "$PROMPTED_DIR" --output_dir "$SFT_DATA_DIR" \
  --val_ratio 0.1 --evidence_fraction 0.5 --clean_fraction 0.3 --hard_case_top_fraction 0.3 --seed 42

run_step 02_train_sft_analyzer env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_unified_analyzer_sft.py \
  --model_name "$ANALYZER_MODEL" \
  --train_path "$SFT_DATA_DIR/runtime_unified_train.jsonl" \
  --val_path "$SFT_DATA_DIR/runtime_unified_valid.jsonl" \
  --output_dir "$SFT_ADAPTER_DIR" \
  --max_length 4096 --num_train_epochs 1.0 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 --logging_steps 10 --save_steps 100 --eval_steps 100 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --seed 42 --bf16

run_step 03_recompute_sft_posterior env CUDA_VISIBLE_DEVICES="$GPU" "$PY" recompute_posterior_with_learned_analyzer.py \
  --input_debug_jsonl "$DEBUG_JSONL" \
  --output_dir "$SFT_RECOMPUTE_DIR" \
  --model_name "$ANALYZER_MODEL" --adapter_path "$SFT_ADAPTER_DIR" \
  --batch_size 4 --max_new_tokens 512 --max_input_tokens 4096 \
  --answer_weight 0.8 --evidence_weight 0.2 --prior_lambda 1.0 --prior_temperature 1.0 --bf16

run_step 04_build_dpo_data env CUDA_VISIBLE_DEVICES="$GPU" "$PY" build_analyzer_dpo_data_from_gsm8k_logs.py \
  --log_dir "$PROMPTED_DIR" --output_dir "$DPO_DATA_DIR" \
  --learned_posterior_debug_jsonl "$SFT_RECOMPUTE_DIR/learned_posterior_debug.jsonl" \
  --val_ratio 0.1 --hard_case_top_fraction 0.3 \
  --target_evidence_train_pairs 4000 --target_prior_train_pairs 1000 \
  --target_evidence_valid_pairs 400 --target_prior_valid_pairs 100 \
  --synthetic_companion_rate 0.5 --seed 42

run_step 05_train_dpo_analyzer env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_unified_analyzer_dpo.py \
  --model_name "$ANALYZER_MODEL" \
  --train_path "$DPO_DATA_DIR/runtime_unified_train.jsonl" \
  --val_path "$DPO_DATA_DIR/runtime_unified_valid.jsonl" \
  --output_dir "$DPO_ADAPTER_DIR" \
  --init_adapter_path "$SFT_ADAPTER_DIR" --reference_adapter_path "$SFT_ADAPTER_DIR" \
  --beta 0.1 --max_length 4096 --max_prompt_length 3584 --max_completion_length 512 \
  --num_train_epochs 1.0 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 --logging_steps 10 --save_steps 100 --eval_steps 100 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --seed 42 --bf16 --gradient_checkpointing

run_step 06_train_bpr_grpo_learned_analyzer_vllm env CUDA_VISIBLE_DEVICES="$GPU" "$PY" Bayesian_Full_GRPO_learned.py \
  --model_name "$MODEL" --dataset_name gsm8k \
  --use_fixed_metadata --train_metadata_path "$TRAIN_METADATA" --eval_metadata_path "$TEST_METADATA" \
  --train_size 7473 --eval_size 0 \
  --num_generations 8 --max_prompt_length 1024 --max_steps 1000 --max_completion_length 1024 \
  --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 5e-6 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --min_solve_rate 0.0 --max_solve_rate 1.0 \
  --prior_mode learned_unified_analyzer \
  --analyzer_model_name "$ANALYZER_MODEL" --analyzer_adapter_path "$DPO_ADAPTER_DIR" \
  --prior_lambda 1.0 --prior_softmax_temperature 1.0 --judge_max_new_tokens 768 \
  --format_bonus 0.0 --seed 42 --logging_steps 5 --save_steps 250 --progress_interval_percent 10 \
  --bf16 --output_dir "$GRPO_OUTPUT_DIR" --use_lora --gradient_checkpointing \
  --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0.20 \
  --vllm_tensor_parallel_size 1 --vllm_max_model_length 3072

echo "DONE TRAIN: $GRPO_OUTPUT_DIR"
