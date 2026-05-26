# Experiment Stack

This file tracks training jobs and their configs as they are created.
Append new train jobs at the bottom so the file reads chronologically from top to bottom.

## 2026-05-21 - Qwen3-8B Answer-only GRPO on MATH 12k/test500

Status: planned

Purpose:
- Train `Qwen/Qwen3-8B` with Answer-only GRPO on the MATH training split.
- Keep the direct `Answer_only_GRPO.py` invocation style used for the Qwen3-4B run.
- Use the same metadata naming convention as the requested 4B command.

Metadata:
- Output dir: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42`
- Train metadata: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl`
- Eval metadata: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Expected train rows: `12000`
- Expected test rows: `500`

Metadata command:

```bash
cd /home/kimjh/EMNLP

.venv_qwen/bin/python prepare_math500_metadata.py \
  --setting sampled_train3000_valid500 \
  --dataset_name DigitalLearningGmbH/MATH-lighteval \
  --train_split train \
  --test_split test \
  --train_size 12000 \
  --valid_size 0 \
  --output_dir outputs/math500_experiments/metadata_fulltrain12000_test500_seed42 \
  --seed 42
```

Training config:
- Script: `Answer_only_GRPO.py`
- Model: `Qwen/Qwen3-8B`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Output dir: `outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1000`
- Train size: `12000`
- Eval size: `500`
- Num generations: `8`
- Max steps: `1000`
- Per-device train batch size: `1`
- Gradient accumulation steps: `8`
- Logging steps: `10`
- Save steps: `100`
- Max prompt length: `2048`
- Max completion length: `1024`
- Temperature: `0.7`
- Top-p: `0.95`
- LoRA: enabled with script defaults
- BF16: enabled
- Gradient checkpointing: enabled

Training command:

```bash
cd /home/kimjh/EMNLP

mkdir -p outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1000

CUDA_VISIBLE_DEVICES=1 nohup .venv_qwen/bin/python Answer_only_GRPO.py \
  --model_name Qwen/Qwen3-8B \
  --use_fixed_metadata \
  --train_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl \
  --train_size 12000 \
  --eval_size 500 \
  --output_dir outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1000 \
  --num_generations 8 \
  --max_steps 1000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --save_steps 100 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --use_lora \
  --bf16 \
  --gradient_checkpointing \
  > outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1000/train.log 2>&1 &
```

Validation commands:

```bash
wc -l outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
wc -l outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
tail -f outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1000/train.log
```

## 2026-05-22 - Qwen3-4B Bayesian + SFT-DPO Analyzer GRPO on GSM8K

Status: running

Purpose:
- Train the learned analyzer pipeline for `Qwen/Qwen3-4B` on GSM8K full-train.
- Main requested stages:
  1. Build SFT analyzer data from Bayesian Prompted GRPO logs and train the SFT analyzer.
  2. Recompute posterior with the SFT analyzer, build DPO data, and train the DPO analyzer.
  3. Train the solver with the DPO analyzer attached.

Important prerequisite:
- The Qwen3-4B GSM8K Bayesian Prompted GRPO reward log is required at:
  `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted/bayesian_reward_debug.jsonl`
- Received and verified on 2026-05-22:
  `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted/bayesian_reward_debug.jsonl`
- The final launcher command keeps `RUN_BAYESIAN_PROMPTED=0` and starts directly from analyzer SFT.

Launch:
- Started on 2026-05-22 UTC.
- Launcher PID: `7129`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- Initial GPU check: GPU 3 was empty on RTX PRO 6000 Blackwell 96GB, so the prompted-baseline-matched `per_device_train_batch_size=8` / `gradient_accumulation_steps=1` setting was kept.

Shared config:
- Dataset key: `gsm8k`
- GRPO dataset name: `fixed_metadata`
- Metadata dir: `outputs/gsm8k_full_train_seed42`
- Experiment root: `outputs/gsm8k_full_qwen3_4b`
- Model: `Qwen/Qwen3-4B`
- Analyzer model: `Qwen/Qwen3-4B`
- Solver mode: LoRA GRPO
- Train size: `7473`
- Eval size during training: `0`
- Test eval: official GSM8K test, `1319` examples
- Num generations: `8`
- Solver max steps: `1500`
- Max completion length: `1024`
- Train max prompt length: `1024`
- Eval max new tokens: `1024`
- Temperature/top-p: `0.7` / `0.95`
- Per-device train batch size: `8`
- Gradient accumulation steps: `1`
- Effective batch size: `8`
- Logging steps: `5`
- Save steps: `250`
- Progress interval percent: `10`
- Solver LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- Seed: `42`

Analyzer SFT config:
- SFT data dir: `outputs/gsm8k_full_qwen3_4b/analyzer_pipeline/sft_data`
- SFT adapter dir: `outputs/gsm8k_full_qwen3_4b/analyzer_pipeline/sft_adapter`
- Val ratio: `0.1`
- Evidence fraction: `0.8`
- Clean fraction: `0.7`
- Hard-case top fraction: `0.25`
- Max length: `4096`
- Epochs: `1.0`
- Batch size: `1`
- Gradient accumulation: `8`
- Learning rate: `2e-4`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`

Recompute and DPO config:
- SFT recompute dir: `outputs/gsm8k_full_qwen3_4b/analyzer_pipeline/sft_recompute_on_prompted_pool`
- DPO data dir: `outputs/gsm8k_full_qwen3_4b/analyzer_pipeline/dpo_data_from_sft_recompute`
- DPO adapter dir: `outputs/gsm8k_full_qwen3_4b/analyzer_pipeline/dpo_adapter`
- Recompute batch size: `8`
- Recompute max new tokens: `512`
- Recompute max input tokens: `4096`
- DPO pairs: evidence train `4000`, prior train `1000`, evidence valid `400`, prior valid `100`
- DPO beta: `0.1`
- DPO max length: `4096`
- DPO max prompt length: `3584`
- DPO max completion length: `512`
- DPO learning rate: `5e-5`

DPO-analyzer solver output:
- Output dir: `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1`
- Step logs: `outputs/gsm8k_full_qwen3_4b/step_logs`

Dry-run command:

```bash
cd /home/kimjh/EMNLP

CUDA_VISIBLE_DEVICES=3 bash run_gsm8k_qwen3_4b_sft_dpo_analyzer_grpo.sh --dry_run
```

Launch command:

```bash
cd /home/kimjh/EMNLP
mkdir -p outputs/gsm8k_full_qwen3_4b/launcher_logs

CUDA_VISIBLE_DEVICES=3 nohup bash run_gsm8k_qwen3_4b_sft_dpo_analyzer_grpo.sh \
  > outputs/gsm8k_full_qwen3_4b/launcher_logs/sft_dpo_analyzer_grpo.log 2>&1 &
```

Progress commands:

```bash
tail -f outputs/gsm8k_full_qwen3_4b/launcher_logs/sft_dpo_analyzer_grpo.log
tail -f outputs/gsm8k_full_qwen3_4b/step_logs/06_train_sft_analyzer.log
tail -f outputs/gsm8k_full_qwen3_4b/step_logs/10_train_dpo_analyzer.log
tail -f outputs/gsm8k_full_qwen3_4b/step_logs/11_grpo_bayesian_sft_dpo_analyzer.log
```

## 2026-05-22 - Test Eval for Qwen3-8B Answer-only GRPO on MATH 12k/test500

Status: ready to run

Purpose:
- Evaluate the completed `Qwen/Qwen3-8B` Answer-only GRPO LoRA adapter on the MATH test metadata.

Checkpoint:
- Adapter path: `outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500`
- Model: `Qwen/Qwen3-8B`
- Train size: `12000`
- Max steps: `1500`
- Num generations during training: `8`

Eval config:
- Eval metadata: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Output dir: `outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test`
- Batch size: `16`
- Max prompt length: `2048`
- Max new tokens: `1024`
- Decoding: deterministic greedy
- `do_sample`: `false`
- Temperature/top-p: `0.0` / `1.0`
- Seed: `42`
- BF16: enabled
- Adapter loading: enabled

Launch command:

```bash
cd /home/kimjh/EMNLP

mkdir -p outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test

CUDA_VISIBLE_DEVICES=3 nohup .venv_qwen/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500 \
  --eval_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl \
  --output_dir outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test \
  --batch_size 16 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  --load_adapter \
  --bf16 \
  --no_do_sample \
  > outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test/eval.log 2>&1 &
```

Progress command:

```bash
tail -f outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test/eval.log
```

Expected outputs:
- `outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test/summary.json`
- `outputs/math500_experiments/grpo_answer_only_qwen3_8b_fulltrain12k_n8_steps1500/test/predictions.jsonl`

## 2026-05-22 - Pure-Base Raw Problem Eval Suite

Status: ready to run

Purpose:
- Evaluate base models without any system prompt, user instruction, strategy instruction, or chat template.
- Input is exactly each metadata row's `problem` text.
- Output is parsed with a format-agnostic answer extractor.

Script:
- `eval_pure_base_model.py`

Models:
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`

Datasets:
- GSM8K full test: `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl`
- MATH 500 test: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- AIME26: `outputs/eval_benchmarks/aime26_metadata.jsonl`
- MinervaMath: `outputs/eval_benchmarks/minervamath_metadata.jsonl`
- OlympiadBench: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`

Eval config:
- Prompt mode: `raw_problem_only`
- Uses system prompt: `false`
- Uses user instruction: `false`
- Uses chat template: `false`
- Decoding: deterministic greedy
- `do_sample`: `false`
- Max prompt length: `2048`
- Max new tokens: dataset default, AIME26 `4096`, all others `1024`
- Batch size: `16`
- BF16: enabled when CUDA supports it

Dry-run command:

```bash
cd /home/kimjh/EMNLP
python3 eval_pure_base_model.py \
  --model_key qwen3_1p7b \
  --dataset_key all \
  --output_root outputs/pure_base_eval/qwen3_1p7b \
  --dry_run
```

Sequential launch command for all three models:

```bash
cd /home/kimjh/EMNLP
source /home/kimjh/EMNLP/.venv_qwen/bin/activate
mkdir -p outputs/pure_base_eval/logs

CUDA_VISIBLE_DEVICES=1 nohup bash -lc '
cd /home/kimjh/EMNLP
source /home/kimjh/EMNLP/.venv_qwen/bin/activate

python3 eval_pure_base_model.py \
  --model_key qwen3_1p7b \
  --dataset_key all \
  --output_root outputs/pure_base_eval/qwen3_1p7b \
  --batch_size 16 \
  --max_prompt_length 2048 \
  --max_new_tokens 0 \
  --seed 42 \
  --bf16 \
  --no_do_sample

python3 eval_pure_base_model.py \
  --model_key qwen3_4b \
  --dataset_key all \
  --output_root outputs/pure_base_eval/qwen3_4b \
  --batch_size 16 \
  --max_prompt_length 2048 \
  --max_new_tokens 0 \
  --seed 42 \
  --bf16 \
  --no_do_sample

python3 eval_pure_base_model.py \
  --model_key qwen3_8b \
  --dataset_key all \
  --output_root outputs/pure_base_eval/qwen3_8b \
  --batch_size 16 \
  --max_prompt_length 2048 \
  --max_new_tokens 0 \
  --seed 42 \
  --bf16 \
  --no_do_sample
' > outputs/pure_base_eval/logs/all_qwen3_pure_base_eval.log 2>&1 &
```

Progress command:

```bash
tail -f outputs/pure_base_eval/logs/all_qwen3_pure_base_eval.log
```

## 2026-05-22 - Qwen3-4B BPR-GRPO Prompted Analyzer GSM8K Full-Train with vLLM

Status: superseded before launch; user corrected target dataset to BigMath

Purpose:
- Train `Qwen/Qwen3-4B` with Bayesian posterior reward GRPO using the prompted analyzer.
- Same GSM8K full-train setup as the prompted-analyzer baseline, except `max_steps=1536` and vLLM rollout/eval inference enabled.

Training config:
- Script wrapper: `run_gsm8k_grpo_bayesian_prompted.sh`
- Underlying training script: `Bayesian_Full_GRPO.py`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Model: `Qwen/Qwen3-4B`
- Prior judge model: `Qwen/Qwen3-4B`
- Evidence judge model: `Qwen/Qwen3-4B`
- Dataset mode: `fixed_metadata`
- Metadata dir: `outputs/gsm8k_full_train_seed42`
- Train size: `7473`
- Valid eval size: `0`
- Test eval: runs after training through the wrapper
- Num generations: `8`
- Max steps: `1536`
- Per-device train batch size: `8`
- Gradient accumulation steps: `1`
- Effective batch size: `8`
- Learning rate: `5e-6`
- Max prompt length: `1024`
- Max completion length: `1024`
- Temperature: `0.7`
- Top-p: `0.95`
- Reward: posterior-normalized Bayesian evidence
- Prior mode: `llm_strategy_prior`
- Prior lambda: `1.0`
- Prior softmax temperature: `1.0`
- Judge max new tokens: `768`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- BF16: enabled
- Gradient checkpointing: enabled
- vLLM: enabled, colocate mode
- vLLM GPU memory utilization: `0.3`
- vLLM max model length: `2048`
- Output dir: `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1536_bsz8_acc1_lambda1_vllm`

Launch command:

```bash
cd /home/kimjh/EMNLP
source /home/kimjh/EMNLP/.venv_qwen/bin/activate

mkdir -p outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1536_bsz8_acc1_lambda1_vllm/launcher_logs

CUDA_VISIBLE_DEVICES=0 nohup env \
  MODEL_NAME="Qwen/Qwen3-4B" \
  DATASET_NAME="fixed_metadata" \
  PRIOR_JUDGE_MODEL="Qwen/Qwen3-4B" \
  EVIDENCE_JUDGE_MODEL="Qwen/Qwen3-4B" \
  METADATA_DIR="outputs/gsm8k_full_train_seed42" \
  OUTPUT_DIR="outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1536_bsz8_acc1_lambda1_vllm" \
  TRAIN_SIZE="7473" \
  EVAL_SIZE="0" \
  NUM_GENERATIONS="8" \
  MAX_STEPS="1536" \
  TRAIN_MAX_PROMPT_LENGTH="1024" \
  MAX_COMPLETION_LENGTH="1024" \
  TEMPERATURE="0.7" \
  TOP_P="0.95" \
  PER_DEVICE_TRAIN_BATCH_SIZE="8" \
  GRADIENT_ACCUMULATION_STEPS="1" \
  LEARNING_RATE="5e-6" \
  USE_LORA="1" \
  GRADIENT_CHECKPOINTING="1" \
  LORA_R="16" \
  LORA_ALPHA="32" \
  LORA_DROPOUT="0.05" \
  LOGGING_STEPS="5" \
  SAVE_STEPS="250" \
  PROGRESS_INTERVAL_PERCENT="10" \
  MIN_SOLVE_RATE="0.0" \
  MAX_SOLVE_RATE="1.0" \
  PRIOR_LAMBDA="1.0" \
  PRIOR_SOFTMAX_TEMPERATURE="1.0" \
  JUDGE_MAX_NEW_TOKENS="768" \
  USE_VLLM="1" \
  VLLM_MODE="colocate" \
  VLLM_GPU_MEMORY_UTILIZATION="0.3" \
  VLLM_EVAL_GPU_MEMORY_UTILIZATION="0.9" \
  VLLM_MAX_MODEL_LENGTH="2048" \
  VLLM_TENSOR_PARALLEL_SIZE="1" \
  METHOD_NAME="Qwen3-4B BPR-GRPO Prompted Analyzer" \
  TRAIN_DATA_LABEL="GSM8K official train full" \
  SUMMARY_NOTES="BPR-GRPO prompted analyzer full-train with vLLM rollout/eval; max_steps=1536" \
  STEP_LOG_DIR="outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1536_bsz8_acc1_lambda1_vllm/step_logs" \
  bash run_gsm8k_grpo_bayesian_prompted.sh --skip_valid_eval \
  > outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1536_bsz8_acc1_lambda1_vllm/launcher_logs/train_eval.log 2>&1 &
```

Progress command:

```bash
tail -f outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1536_bsz8_acc1_lambda1_vllm/launcher_logs/train_eval.log
```

## 2026-05-22 - Qwen3-4B BPR-GRPO Prompted Analyzer BigMath 12x1024 with vLLM

Status: completed

Purpose:
- Train `Qwen/Qwen3-4B` with Bayesian posterior reward GRPO using the prompted analyzer.
- Use BigMath BARL-style 12 rollout batches x 1024 prompts, with no in-training eval.
- Keep prompted-analyzer baseline settings aligned with prior BPR-GRPO runs except dataset and `max_steps=1536`.

Training config:
- Script: `Bayesian_Full_GRPO.py`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Model: `Qwen/Qwen3-4B`
- Prior judge model: `Qwen/Qwen3-4B`
- Evidence judge model: `Qwen/Qwen3-4B`
- Dataset: `SynthLabsAI/Big-Math-RL-Verified`
- Dataset mode: `fixed_metadata`
- Train metadata: `outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl`
- Eval metadata: `outputs/bigmath_barl_style_12x1024_seed42/empty_eval_metadata.jsonl`
- Train size: `12288`
- Eval size: `0`
- Num generations: `8`
- Max steps: `1536`
- Per-device train batch size: `8`
- Gradient accumulation steps: `1`
- Effective batch size: `8`
- Learning rate: `5e-6`
- Max prompt length: `1024`
- Max completion length: `1024`
- Temperature: `0.7`
- Top-p: `0.95`
- Reward: posterior-normalized Bayesian evidence
- Prior mode: `llm_strategy_prior`
- Prior lambda: `1.0`
- Prior softmax temperature: `1.0`
- Judge max new tokens: `768`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- BF16: enabled
- Gradient checkpointing: enabled
- vLLM: enabled, colocate mode
- vLLM GPU memory utilization: `0.3`
- vLLM max model length: `2048`
- Output dir: `outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm`

Launch command:

```bash
cd /home/kimjh/EMNLP
source /home/kimjh/EMNLP/.venv/bin/activate

python3 -c 'import vllm; print("vllm", vllm.__version__)'

mkdir -p outputs/bigmath_barl_style_12x1024_seed42
: > outputs/bigmath_barl_style_12x1024_seed42/empty_eval_metadata.jsonl

mkdir -p outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm

CUDA_VISIBLE_DEVICES=0 nohup python3 Bayesian_Full_GRPO.py \
  --model_name Qwen/Qwen3-4B \
  --dataset_name SynthLabsAI/Big-Math-RL-Verified \
  --prior_mode llm_strategy_prior \
  --prior_judge_model Qwen/Qwen3-4B \
  --evidence_judge_model Qwen/Qwen3-4B \
  --use_fixed_metadata \
  --train_metadata_path outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/bigmath_barl_style_12x1024_seed42/empty_eval_metadata.jsonl \
  --train_size 12288 \
  --eval_size 0 \
  --num_generations 8 \
  --max_steps 1536 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --min_solve_rate 0.0 \
  --max_solve_rate 1.0 \
  --seed 42 \
  --logging_steps 5 \
  --save_steps 250 \
  --progress_interval_percent 10 \
  --prior_lambda 1.0 \
  --prior_softmax_temperature 1.0 \
  --judge_max_new_tokens 768 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --bf16 \
  --gradient_checkpointing \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_max_model_length 2048 \
  --vllm_tensor_parallel_size 1 \
  --reward_debug_jsonl outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm/bayesian_reward_debug.jsonl \
  --output_dir outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm \
  > outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm/train.log 2>&1 &
```

Progress command:

```bash
tail -f outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm/train.log
```

## 2026-05-23 - Qwen3-4B SFT-DPO Learned Analyzer GRPO GSM8K Test Max4096

Status: canceled before launch; user asked to stand by

Purpose:
- Re-evaluate the completed `Qwen3-4B Bayesian + SFT-DPO Analyzer GRPO GSM8K` solver checkpoint with a larger decode budget.
- Keep the same GSM8K test metadata and deterministic decoding, changing only `max_new_tokens` from `1024` to `4096`.

Eval config:
- Script: `eval_solver_checkpoint.py`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- Model: `Qwen/Qwen3-4B`
- Adapter path: `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1`
- Eval metadata: `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl`
- Output dir: `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/test_max4096`
- Eval examples: `1319`
- Batch size: `8`
- Max prompt length: `2048`
- Max new tokens: `4096`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Launch command:

```bash
cd /home/kimjh/EMNLP
source /home/kimjh/EMNLP/.venv/bin/activate

mkdir -p outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/test_max4096

CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1 \
  --eval_metadata_path outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl \
  --output_dir outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/test_max4096 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --max_prompt_length 2048 \
  --seed 42 \
  --no_do_sample \
  > outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/test_max4096/eval.log 2>&1 &
```

Progress command:

```bash
tail -f outputs/gsm8k_full_qwen3_4b/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/test_max4096/eval.log
```

## 2026-05-23 - Qwen3-1.7B BigMath SFT-DPO Learned Analyzer GRPO + Eval3

Status: running; currently in SFT recompute evidence-judge generation

Purpose:
- Build the learned-analyzer row that follows from the Qwen3-1.7B BigMath BPR-GRPO prompted-analyzer teacher run.
- Run the sequence SFT analyzer -> SFT recompute -> DPO analyzer -> DPO-analyzer GRPO solver.
- After training, evaluate the solver on AIME26, MinervaMath, and OlympiadBench.

Important prerequisite:
- The Qwen3-1.7B BigMath prompted-analyzer reward debug log is required.
- Received and verified on 2026-05-23:
  `outputs/bigmath_qwen3_1p7b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`
- Verified size/rows: `125M`, `12000` valid JSONL rows.

Training config:
- GPU: `CUDA_VISIBLE_DEVICES=3`
- Model/analyzer model: `Qwen/Qwen3-1.7B`
- Source train metadata: `outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl`
- Train size: `12000`
- Solver output: `outputs/bigmath_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_bigmath12k_n8_steps1500_bsz8_acc1_lambda1_vllm`
- Num generations: `8`
- Max steps: `1500`
- Batch/accumulation: `8` / `1`
- Max prompt/completion length: `1024` / `1024`
- Temperature/top-p: `0.7` / `0.95`
- Learning rate: `5e-6`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- BF16 and gradient checkpointing enabled
- vLLM rollout enabled with colocate mode, GPU memory utilization `0.3`

Eval config:
- AIME26: `outputs/eval_benchmarks/aime26_metadata.jsonl`, max new tokens `4096`
- MinervaMath: `outputs/eval_benchmarks/minervamath_metadata.jsonl`, max new tokens `1024`
- OlympiadBench: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`, max new tokens `1024`
- Eval decoding: deterministic greedy, `--no_do_sample`

Progress commands:

```bash
tail -f outputs/bigmath_qwen3_1p7b/sft_dpo_learned_analyzer_pipeline_lambda1/logs/pipeline_nohup.log
tail -f outputs/bigmath_qwen3_1p7b/sft_dpo_learned_analyzer_pipeline_lambda1/logs/03_train_unified_analyzer_sft.log
tail -f outputs/bigmath_qwen3_1p7b/sft_dpo_learned_analyzer_pipeline_lambda1/logs/06_train_unified_analyzer_dpo.log
tail -f outputs/bigmath_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_bigmath12k_n8_steps1500_bsz8_acc1_lambda1_vllm/step_logs/01_train.log
```

## 2026-05-23 - Qwen3-1.7B BPR-GRPO Prompted Analyzer GSM8K Test Max4096 vLLM

Status: completed with `VLLM_USE_FLASHINFER_SAMPLER=0`

Purpose:
- Re-evaluate the completed `Qwen3-1.7B BPR-GRPO (Prompted Analyzer) GSM8K` checkpoint with a larger decode budget.
- Compare against the existing 1024-token GSM8K test result.

Baseline reference:
- Existing 1024-token test accuracy: `1050/1319 = 79.61%`
- Existing generated length mean: `534.22 / 1024`

Completed result:
- 4096-token vLLM test accuracy: `1066/1319 = 80.82%`
- Generated length mean: `299.34 / 4096`
- Output summary: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_prompted/test_max4096_vllm/summary.json`

Eval config:
- Script: `eval_solver_checkpoint.py`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Model: `Qwen/Qwen3-1.7B`
- Adapter path: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_prompted`
- Eval metadata: `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl`
- Output dir: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_prompted/test_max4096_vllm`
- Eval examples: `1319`
- Batch size: `16`
- Max prompt length: `2048`
- Max new tokens: `4096`
- vLLM: enabled
- vLLM max model length: `6144`
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress command:

```bash
tail -f outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_prompted/test_max4096_vllm/eval.log
```

## 2026-05-23 - Qwen3-4B BPR-GRPO Prompted Analyzer BigMath Eval3

Status: command prepared

Purpose:
- Evaluate the completed `Qwen3-4B BPR-GRPO Prompted Analyzer BigMath` checkpoint on AIME26, MinervaMath, and OlympiadBench.
- Keep AIME26 at `4096` decode tokens and the other benchmarks at `1024`.

Eval config:
- Script: `eval_solver_checkpoint.py`
- Suggested GPU: `CUDA_VISIBLE_DEVICES=1`
- Model: `Qwen/Qwen3-4B`
- Adapter path: `outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm`
- AIME26 max new tokens: `4096`
- MinervaMath max new tokens: `1024`
- OlympiadBench max new tokens: `1024`
- vLLM: enabled
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress command:

```bash
tail -f outputs/bigmath_qwen3_4b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1_vllm/eval_benchmarks/logs/eval3.log
```

## 2026-05-23 - Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath Resume + MATH-500 Test

Status: command prepared

Purpose:
- Resume the interrupted BigMath learned-analyzer GRPO solver run from `checkpoint-900`.
- After training completes, evaluate the final root adapter on MATH-500 full test set with `max_new_tokens=4096` and vLLM.

Train config:
- Script: `Bayesian_Full_GRPO_learned.py`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- Model: `Qwen/Qwen3-1.7B`
- Analyzer adapter: `outputs/bigmath_qwen3_1p7b/sft_dpo_learned_analyzer_pipeline_lambda1/analyzer_sft_dpo_qwen3_1p7b`
- Output dir: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm`
- Resume checkpoint: `checkpoint-900`
- Train metadata: `outputs/bigmath_qwen3_1p7b/sft_dpo_learned_analyzer_pipeline_lambda1/bigmath_metadata_for_pipeline/selected_train_metadata.jsonl`
- Train size: `12000`
- Max steps: `1500`
- Num generations: `8`
- Max prompt length: `2048`
- Max completion length: `1024`
- Batch/accumulation: `1 x 8`
- Learning rate: `5e-6`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- vLLM: enabled, colocate, memory utilization `0.3`, max model length `4096`
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`

Eval config:
- Script: `eval_solver_checkpoint.py`
- Eval metadata: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Eval examples: `500`
- Output dir: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/math500_test500_max4096_vllm_after_resume`
- Batch size: `32`
- Max prompt length: `2048`
- Max new tokens: `4096`
- vLLM max model length: `6144`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress commands:

```bash
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/step_logs/resume_train_then_math500_eval.nohup.log
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/step_logs/01_train_resume_from_checkpoint900.log
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/math500_test500_max4096_vllm_after_resume/eval.log
```

## 2026-05-24 - Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath Eval3

Status: command prepared

Purpose:
- Evaluate the completed `Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath` checkpoint on AIME26, MinervaMath, and OlympiadBench.
- Keep AIME26 at `4096` decode tokens and MinervaMath/OlympiadBench at `1024`.

Eval config:
- Script: `eval_solver_checkpoint.py`
- Suggested GPU: `CUDA_VISIBLE_DEVICES=3`
- Model: `Qwen/Qwen3-1.7B`
- Adapter path: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm`
- AIME26 metadata: `outputs/eval_benchmarks/aime26_metadata.jsonl`
- MinervaMath metadata: `outputs/eval_benchmarks/minervamath_metadata.jsonl`
- OlympiadBench metadata: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`
- AIME26 max new tokens: `4096`
- MinervaMath max new tokens: `1024`
- OlympiadBench max new tokens: `1024`
- vLLM: enabled
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress command:

```bash
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/eval_benchmarks/logs/eval3.log
```

## 2026-05-24 - Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath AIME26 Retest

Status: command prepared

Purpose:
- Re-evaluate only AIME26 for the completed `Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath` checkpoint.
- Keep AIME26 decode budget at `4096`.

Eval config:
- Script: `eval_solver_checkpoint.py`
- Suggested GPU: `CUDA_VISIBLE_DEVICES=3`
- Model: `Qwen/Qwen3-1.7B`
- Adapter path: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm`
- Eval metadata: `outputs/eval_benchmarks/aime26_metadata.jsonl`
- Output dir: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/eval_benchmarks/aime26_max4096_vllm_rerun`
- Batch size: `64`
- Max prompt length: `2048`
- Max new tokens: `4096`
- vLLM: enabled
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress command:

```bash
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/eval_benchmarks/logs/aime26_rerun.log
```

## 2026-05-24 - Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath Eval3 Retest

Status: command prepared

Purpose:
- Re-evaluate AIME26, MinervaMath, and OlympiadBench for the completed `Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath` checkpoint.
- Keep AIME26 at `4096` decode tokens and MinervaMath/OlympiadBench at `1024`.
- Store outputs separately under `retest_all` directories.

Eval config:
- Script: `eval_solver_checkpoint.py`
- Suggested GPU: `CUDA_VISIBLE_DEVICES=3`
- Model: `Qwen/Qwen3-1.7B`
- Adapter path: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm`
- AIME26 metadata: `outputs/eval_benchmarks/aime26_metadata.jsonl`
- MinervaMath metadata: `outputs/eval_benchmarks/minervamath_metadata.jsonl`
- OlympiadBench metadata: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`
- AIME26 max new tokens: `4096`
- MinervaMath max new tokens: `1024`
- OlympiadBench max new tokens: `1024`
- vLLM: enabled
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress command:

```bash
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/eval_benchmarks/logs/eval3_retest_all.log
```

## 2026-05-24 - Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath Eval3 All4096 Retest

Status: command prepared

Purpose:
- Re-evaluate AIME26, MinervaMath, and OlympiadBench for the completed `Qwen3-1.7B BPR-GRPO SFT+DPO Learned Analyzer BigMath` checkpoint.
- Use `max_new_tokens=4096` for all three benchmarks.
- Store outputs separately under `all4096` retest directories.

Eval config:
- Script: `eval_solver_checkpoint.py`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- Model: `Qwen/Qwen3-1.7B`
- Adapter path: `outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm`
- AIME26 metadata: `outputs/eval_benchmarks/aime26_metadata.jsonl`
- MinervaMath metadata: `outputs/eval_benchmarks/minervamath_metadata.jsonl`
- OlympiadBench metadata: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`
- Max prompt length: `2048`
- Max new tokens: `4096`
- vLLM max model length: `6144`
- Batch size: `64`
- vLLM: enabled
- vLLM workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- Decoding: deterministic, `--no_do_sample`
- Seed: `42`

Progress command:

```bash
tail -f outputs/bigmath_qwen3_1p7b/bpr_grpo_sft_dpo_learned_analyzer_fulltrain12k_n8_steps1500_lambda1_vllm/eval_benchmarks/logs/eval3_all4096.log
```
