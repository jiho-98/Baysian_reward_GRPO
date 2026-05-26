cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export OUT=outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r5_native_eager

mkdir -p "$OUT"

nohup python3 run_grpo_bayesian_with_learned_analyzer.py \
  --model_name Qwen/Qwen3-1.7B \
  --dataset_name ricdomolm/MATH-500 \
  --metadata_dir outputs/math500_experiments/metadata_fulltrain12000_test500_seed42 \
  --train_size 12000 \
  --eval_size 0 \
  --skip_valid_eval \
  --skip_preflight_recompute \
  --analyzer_model_name Qwen/Qwen3-1.7B \
  --analyzer_adapter_path outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/analyzer_sft_dpo_qwen3_1p7b_from_original_debug \
  --analyzer_type learned_sft_dpo \
  --output_dir "$OUT" \
  --num_generations 8 \
  --max_steps 1500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --learning_rate 5e-6 \
  --use_lora \
  --gradient_checkpointing \
  --logging_steps 10 \
  --save_steps 100 \
  --prior_lambda 1.0 \
  --prior_softmax_temperature 1.0 \
  --judge_max_new_tokens 768 \
  --bf16 \
  --eval_batch_size 16 \
  --eval_max_new_tokens 1024 \
  --eval_max_prompt_length 2048 \
  --method_name "Qwen3-1.7B Bayesian GRPO SFT+DPO Analyzer vLLM native eager" \
  --train_data_label "MATH train 12k" \
  --notes "MATH-500 Bayesian GRPO with SFT+DPO learned analyzer, lambda=1.0, vLLM native impl, eager init patch" \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_eval_gpu_memory_utilization 0.90 \
  --vllm_max_model_length 4096 \
  --vllm_tensor_parallel_size 1 \
  > "$OUT/nohup.out" 2>&1 &