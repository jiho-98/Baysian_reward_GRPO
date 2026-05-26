# Running Training Stack

This file tracks active and recently launched training jobs.
Append new training runs here whenever a new train job is started.

Last updated: 2026-05-22 KST

## Active Jobs

### 1. MATH-500 Answer-only GRPO - Qwen3-4B

- Status: running
- Method: Answer-only GRPO
- Reward type: answer_only_correctness
- Model: Qwen/Qwen3-4B
- Train data: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
- Eval data during train: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Train size: 12000
- Eval size: 500
- Output dir: outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500
- Log file: outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/train.log
- LoRA: true
- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- BF16: true
- Gradient checkpointing: true
- Max steps: 1500
- Num generations: 8
- Per-device train batch size: 1
- Gradient accumulation steps: 8
- Effective prompt batch: 8
- Prompt exposures: 12000
- Epoch estimate: 1.00 epoch over 12000 train prompts
- Save steps: 100
- Logging steps: 10
- Max prompt length: 2048
- Max completion length: 1024
- Temperature: 0.7
- Top p: 0.95
- Last observed progress: 1111/1500 steps, about 74%
- Last observed ETA: about 3h 17m remaining

Command:

```bash
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python Answer_only_GRPO.py \
  --model_name Qwen/Qwen3-4B \
  --use_fixed_metadata \
  --train_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl \
  --train_size 12000 \
  --eval_size 500 \
  --output_dir outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500 \
  --num_generations 8 \
  --max_steps 1500 \
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
  > outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/train.log 2>&1 &
```

## Eval Result - Pure Base Chat-Template vLLM Qwen3 1.7B/4B/8B max-token split

- Added at: 2026-05-24 14:55 KST
- Status: completed
- Eval target: pure base models, problem-only with tokenizer chat template
- Models: `qwen3_1p7b`, `qwen3_4b`, `qwen3_8b`
- Benchmarks: GSM8K, MATH-500, AIME26, MinervaMath, OlympiadBench
- Output root: `outputs/pure_base_chat_template_eval_benchmarks_vllm_maxtok_math_aime4096`
- Inference backend: vLLM
- GPU: 1
- Batch size: 32
- Max prompt length: 2048
- Max new tokens:
  - MATH-500: 4096
  - AIME26: 4096
  - GSM8K: 1024
  - MinervaMath: 1024
  - OlympiadBench: 1024
- Decoding: greedy, `do_sample=false`
- BF16: true
- Chat template: true
- Seed: 42

Results:

| Model | GSM8K | MATH-500 | AIME26 | MinervaMath | OlympiadBench |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 51.48% | 50.00% | 13.33% | 4.04% | 6.23% |
| Qwen3-4B | 47.76% | 52.00% | 10.00% | 5.88% | 4.75% |
| Qwen3-8B | 47.23% | 53.40% | 16.67% | 5.88% | 4.45% |

## Eval Job - MATH-500 BPR-GRPO Prompted Analyzer Qwen3-8B checkpoint-1500 max4096

- Added at: 2026-05-24 11:55 KST
- Status: command provided to user for manual launch
- Eval target: completed MATH-500 BPR-GRPO Prompted Analyzer - Qwen3-8B
- Model: `Qwen/Qwen3-8B`
- Adapter checkpoint: `outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/checkpoint-1500`
- Eval data: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Eval size: full MATH-500 test, 500 rows
- Output dir: `outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/test_eval_checkpoint1500_maxtok4096_vllm`
- Log file: `outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/test_eval_checkpoint1500_maxtok4096_vllm/eval.log`
- GPU: 1
- Batch size: 32
- Max examples: 0
- Max prompt length: 2048
- Max new tokens: 4096
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- vLLM: true, gpu memory utilization 0.85, max model length 6144, tensor parallel size 1
- Seed: 42

Command:

```bash
cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

export OUT=outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/test_eval_checkpoint1500_maxtok4096_vllm

mkdir -p "$OUT"

nohup python3 eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/checkpoint-1500 \
  --eval_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl \
  --output_dir "$OUT" \
  --batch_size 32 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 4096 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.85 \
  --vllm_max_model_length 6144 \
  --vllm_tensor_parallel_size 1 \
  > "$OUT/eval.log" 2>&1 &
```

## Training Job - Qwen3-1.7B Bayesian GRPO SFT+DPO Analyzer vLLM Pro6000

- Status: configured; command provided for manual launch
- Purpose: run MATH-500 Bayesian Prompted GRPO using the learned SFT+DPO analyzer, with vLLM enabled for GRPO rollout/eval
- Important note: DPO training itself does not use vLLM because it is supervised preference optimization, not generation/rollout. vLLM is enabled from the GRPO solver stage.
- Model: Qwen/Qwen3-1.7B
- Dataset: ricdomolm/MATH-500
- Metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42
- Train metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
- Eval size during train: 0
- Train size: 12000
- Analyzer type: learned_sft_dpo
- Analyzer adapter: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/analyzer_sft_dpo_qwen3_1p7b_from_original_debug
- DPO train data: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/unified_analyzer_dpo_from_original_debug/unified_dpo_train.jsonl
- DPO val data: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/unified_analyzer_dpo_from_original_debug/unified_dpo_val.jsonl
- DPO rows: train 941, val 64
- GRPO output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000
- Max steps: 1500
- Num generations: 8
- Per-device train batch size: 1
- Gradient accumulation steps: 8
- Max prompt length: 2048
- Max completion length: 1024
- Judge max new tokens: 768
- Temperature: 0.7
- Top p: 0.95
- Learning rate: 5e-6
- LoRA: enabled
- BF16: enabled
- Gradient checkpointing: enabled
- vLLM: enabled
- vLLM mode: colocate
- vLLM train GPU memory utilization: 0.45
- vLLM eval GPU memory utilization: 0.90
- vLLM max model length: 4096
- vLLM tensor parallel size: 1
- Dry-run verification: passed; wrapper forwards `--use_vllm` into `Bayesian_Full_GRPO_learned.py`

Launch command:

```bash
cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OUT=outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000

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
  --method_name "Qwen3-1.7B Bayesian GRPO SFT+DPO Analyzer vLLM" \
  --train_data_label "MATH train 12k" \
  --notes "MATH-500 Bayesian GRPO with SFT+DPO learned analyzer, lambda=1.0, vLLM colocate on Pro6000" \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_eval_gpu_memory_utilization 0.90 \
  --vllm_max_model_length 4096 \
  --vllm_tensor_parallel_size 1 \
  > "$OUT/nohup.out" 2>&1 &
```

Update:
- Launch attempt started with vLLM options correctly forwarded into `Bayesian_Full_GRPO_learned.py`.
- Attempted output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000
- Result: failed before training because vLLM is not installed in `.venv`.
- Error: `ImportError: vLLM is not available and use_vllm is set to True. Please install vLLM with pip install trl[vllm] to use it.`
- Current environment check: torch 2.11.0+cu130, trl 1.4.0, transformers 5.8.1, vllm missing.
- Next action: install vLLM in `.venv`, verify import, then relaunch into a fresh r2 output directory.

Update:
- vLLM install completed and r2 launch reached vLLM initialization.
- r2 output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r2
- r2 result: failed before training during vLLM engine profile/compile.
- Environment after install: torch 2.10.0+cu128, trl 1.4.0, transformers 4.57.6, vllm 0.18.0.
- Error: `AttributeError: <function standalone_compile ...> does not have the attribute 'FakeTensorMode'`.
- Interpretation: vLLM is installed and selected correctly, but the vLLM V1 AOT/standalone torch compile path is incompatible in this environment.
- Next retry: r3 with vLLM still enabled, but `VLLM_USE_AOT_COMPILE=0`, `VLLM_USE_STANDALONE_COMPILE=0`, and `VLLM_DISABLE_COMPILE_CACHE=1`.

## Train Pipeline - MATH-500 Bayesian Prompted GRPO Qwen3-1.7B SFT+DPO Analyzer

- Status: command provided to user for manual launch
- Goal: Qwen3-1.7B Bayesian Prompted GRPO on MATH-500 using SFT+DPO learned analyzer
- Solver model: Qwen/Qwen3-1.7B
- Analyzer model: Qwen/Qwen3-1.7B
- Train metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
- Test metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Metadata dir: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42
- Train size: 12000
- Valid/eval during train: skipped, eval_size 0
- Test after train: MATH-500 test 500 through launcher unless skip_eval is set
- Solver GRPO max steps: 1500
- Solver effective prompt batch: 8
- Solver prompt exposures: 12000
- Solver epoch estimate: 1.00 epoch over 12000 MATH train prompts
- Required input: PROMPTED_DEBUG_JSONL pointing to an existing Bayesian prompted teacher/debug log
- SFT analyzer data dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline/analyzer_training_data_v1
- Unified SFT data dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline/unified_analyzer_sft
- SFT analyzer adapter dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline/analyzer_sft_qwen3_1p7b
- Recompute dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline/recompute_sft_analyzer
- DPO data dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline/unified_analyzer_dpo
- DPO analyzer adapter dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline/analyzer_sft_dpo_qwen3_1p7b
- Final solver output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500
- Notes:
  - Step 1 trains the SFT analyzer.
  - Step 2 recomputes posterior with the SFT analyzer, prepares DPO pairs, then trains the DPO analyzer.
  - Step 3 combines the DPO analyzer with Bayesian GRPO solver training.
  - Replace PROMPTED_DEBUG_JSONL before launching.

## Train Pipeline - MATH-500 Bayesian GRPO Qwen3-1.7B SFT+DPO Analyzer lambda1

- Status: command provided to user for nohup manual launch
- Goal: Run SFT analyzer -> recompute -> DPO data -> DPO analyzer -> Bayesian GRPO solver
- Solver model: Qwen/Qwen3-1.7B
- Analyzer model: Qwen/Qwen3-1.7B
- GPU: 1
- Prompted debug log: outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl
- Metadata dir: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42
- Train metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
- Test metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Pipeline root: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1
- Pipeline nohup log: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/nohup.out
- Step logs dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/logs
- Analyzer teacher-clean data: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/analyzer_training_data_v1
- Unified SFT data: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/unified_analyzer_sft
- SFT analyzer adapter: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/analyzer_sft_qwen3_1p7b
- SFT recompute output: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/recompute_sft_analyzer
- Unified DPO data: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/unified_analyzer_dpo
- SFT+DPO analyzer adapter: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/analyzer_sft_dpo_qwen3_1p7b
- Final solver output: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1
- Train size: 12000
- Eval size during train: 0
- Skip valid eval: true
- Solver max steps: 1500
- Solver num generations: 8
- Solver per-device train batch size: 1
- Solver gradient accumulation steps: 8
- Solver effective prompt batch: 8
- Solver prompt exposures: 12000
- Solver epoch estimate: 1.00 epoch over 12000 MATH train prompts
- Solver LoRA: true
- BF16: true
- Gradient checkpointing: true
- Prior lambda: 1.0
- Prior softmax temperature: 1.0
- Recompute max new tokens: 768
- Final GRPO judge max new tokens: 768
- SFT analyzer learning rate: 5e-5
- DPO analyzer learning rate: 1e-5
- DPO beta: 0.1
- SFT/DPO max length: 4096
- DPO max prompt length: 3584
- DPO max completion length: 512
- Notes: user requested this pipeline be tracked with config and future progress updates.

Progress update commands:

```bash
tail -f outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/nohup.out
ls -lh outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_pipeline_lambda1/logs/
```

## Eval Jobs - Big-Math BARL Answer-only GRPO Qwen3-8B checkpoint-1536

- Status: command provided to user for nohup manual launch
- Eval target: Big-Math BARL Answer-only GRPO - Qwen3-8B
- Model: Qwen/Qwen3-8B
- Adapter checkpoint: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536/checkpoint-1536
- Eval benchmarks: AIME26, MinervaMath, OlympiadBench
- AIME26 eval data: outputs/eval_benchmarks/aime26_metadata.jsonl
- MinervaMath eval data: outputs/eval_benchmarks/minervamath_metadata.jsonl
- OlympiadBench eval data: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Eval root dir: outputs/bigmath_qwen3_8b_answer_only_grpo_eval_benchmarks
- Nohup log: outputs/bigmath_qwen3_8b_answer_only_grpo_eval_benchmarks/nohup.out
- Run log: outputs/bigmath_qwen3_8b_answer_only_grpo_eval_benchmarks/logs/run.log
- AIME26 output dir: outputs/bigmath_qwen3_8b_answer_only_grpo_eval_benchmarks/aime26
- MinervaMath output dir: outputs/bigmath_qwen3_8b_answer_only_grpo_eval_benchmarks/minervamath
- OlympiadBench output dir: outputs/bigmath_qwen3_8b_answer_only_grpo_eval_benchmarks/olympiadbench
- GPU: 0
- Batch size: 8
- Max examples: 0
- Max prompt length: 2048
- Max new tokens:
  - AIME26: 4096
  - MinervaMath: 1024
  - OlympiadBench: 1024
- AIME26 max-new-tokens rule: use 4096 for AIME26 evals for consistency across runs
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- Seed: 42
- Expected output layout: root nohup.out, logs/run.log, and one subdirectory per benchmark with summary.json and predictions.jsonl

### 2. Big-Math BARL Answer-only GRPO - Qwen3-1.7B

- Status: running
- Method: Answer-only GRPO
- Reward type: answer_only_correctness
- Model: Qwen/Qwen3-1.7B
- Train data: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
- Eval metadata path passed to script: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Eval size: 0
- Eval note: eval data is not used during training because eval_size is 0
- Train size: 12288
- Output dir: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500
- Log file: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/train.log
- LoRA: true
- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- BF16: true
- Gradient checkpointing: true
- Max steps: 1500
- Num generations: 8
- Per-device train batch size: 1
- Gradient accumulation steps: 8
- Effective prompt batch: 8
- Prompt exposures: 12000
- Epoch estimate: 0.98 epoch over 12288 train prompts
- Save steps: 100
- Logging steps: 10
- Max prompt length: 2048
- Max completion length: 1024
- Temperature: 0.7
- Top p: 0.95
- Last observed progress: 14/1500 steps, about 1%
- Last observed ETA: about 6h 20m remaining

Command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python Answer_only_GRPO.py \
  --model_name Qwen/Qwen3-1.7B \
  --use_fixed_metadata \
  --train_metadata_path outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --train_size 12288 \
  --eval_size 0 \
  --output_dir outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500 \
  --num_generations 8 \
  --max_steps 1500 \
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
  > outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/train.log 2>&1 &
```

## Recently Completed Jobs

### MATH-500 Answer-only GRPO - Qwen3-1.7B

- Status: completed
- Method: Answer-only GRPO
- Reward type: answer_only_correctness
- Model: Qwen/Qwen3-1.7B
- Train data: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
- Eval data during train: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Train size: 12000
- Eval size: 500
- Output dir: outputs/math500_experiments/grpo_answer_only_qwen3_1p7b_fulltrain12k_n8_steps1500
- Final checkpoint: outputs/math500_experiments/grpo_answer_only_qwen3_1p7b_fulltrain12k_n8_steps1500/checkpoint-1500
- LoRA: true
- Max steps: 1500
- Num generations: 8
- Effective prompt batch: 8
- Prompt exposures: 12000
- Epoch estimate: 1.00 epoch over 12000 train prompts
- Runtime observed: 6h 16m 25s

## Append Template

### <Run name>

- Status:
- Method:
- Reward type:
- Model:
- Train data:
- Eval data during train:
- Train size:
- Eval size:
- Output dir:
- Log file:
- LoRA:
- Max steps:
- Num generations:
- Per-device train batch size:
- Gradient accumulation steps:
- Effective prompt batch:
- Prompt exposures:
- Epoch estimate:
- Save steps:
- Logging steps:
- Max prompt length:
- Max completion length:
- Temperature:
- Top p:
- Last observed progress:
- Last observed ETA:

Command:

```bash

```

## Logging Rule

- New train/job entries must be appended at the very bottom of this file.
- Keep the file in chronological order from top to bottom, so the user can read it naturally.
- Do not insert new jobs above older jobs unless explicitly requested.
- Each new job entry should include model, method, dataset, train/eval size, output dir, log path, max steps, LoRA status, effective batch, command, and latest known status.
- When a job finishes, append or update its completion information with final checkpoint, runtime, and eval result if available.

## Eval Job - MATH-500 Answer-only GRPO Qwen3-4B checkpoint-1500

- Status: command provided to user for manual launch
- Eval target: completed MATH-500 Answer-only GRPO - Qwen3-4B
- Model: Qwen/Qwen3-4B
- Adapter checkpoint: outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/checkpoint-1500
- Eval data: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Eval size: 500
- Output dir: outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/test_eval_checkpoint1500
- Log file: outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/test_eval_checkpoint1500/eval.log
- Batch size: 16
- Max examples: 0
- Max prompt length: 2048
- Max new tokens:
  - AIME26: 4096
  - MinervaMath: 1024
  - OlympiadBench: 1024
- AIME26 max-new-tokens rule: use 4096 for AIME26 evals for consistency across runs
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- Seed: 42

Command:

```bash
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl \
  --output_dir outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/test_eval_checkpoint1500 \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 4096 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  > outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/test_eval_checkpoint1500/eval.log 2>&1 &
```

- Note: direct execution by assistant was not performed; the command above was provided for manual execution by the user.

## Train Job - Big-Math BARL Answer-only GRPO Qwen3-4B steps1536

- Status: planned / ready to launch
- Method: Answer-only GRPO
- Reward type: answer_only_correctness
- Model: Qwen/Qwen3-4B
- Train data: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
- Train data source: SynthLabsAI/Big-Math-RL-Verified filtered BARL-style pool
- Train size: 12288
- Eval metadata path passed to script: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Eval size: 0
- Eval note: eval data is not used during training because eval_size is 0
- Target eval benchmarks after training: AIME26, MinervaMath, OlympiadBench
- AIME26 eval data: outputs/eval_benchmarks/aime26_metadata.jsonl
- MinervaMath eval data: outputs/eval_benchmarks/minervamath_metadata.jsonl
- OlympiadBench eval data: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Output dir: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536
- Log file: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/train.log
- LoRA: true
- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- BF16: true
- Gradient checkpointing: true
- Max steps: 1536
- Max steps rationale: exact 1 epoch over 12288 prompts with effective prompt batch 8
- Num generations: 8
- Per-device train batch size: 1
- Gradient accumulation steps: 8
- Effective prompt batch: 8
- Prompt exposures: 12288
- Epoch estimate: 1.00 epoch over 12288 train prompts
- Save steps: 128
- Logging steps: 10
- Max prompt length: 2048
- Max completion length: 1024
- Temperature: 0.7
- Top p: 0.95
- Expected runtime estimate: roughly 9-11 hours based on prior Qwen3-4B MATH-500 1500-step runtime of 9h 32m

Command:

```bash
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python Answer_only_GRPO.py \
  --model_name Qwen/Qwen3-4B \
  --use_fixed_metadata \
  --train_metadata_path outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --train_size 12288 \
  --eval_size 0 \
  --output_dir outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536 \
  --num_generations 8 \
  --max_steps 1536 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --save_steps 128 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --use_lora \
  --bf16 \
  --gradient_checkpointing \
  > outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/train.log 2>&1 &
```

## Eval Jobs - Big-Math BARL Answer-only GRPO Qwen3-1.7B checkpoint-1500

- Status: command provided to user for manual launch
- Eval target: Big-Math BARL Answer-only GRPO - Qwen3-1.7B
- Model: Qwen/Qwen3-1.7B
- Adapter checkpoint: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500
- Eval benchmarks: AIME26, MinervaMath, OlympiadBench
- AIME26 eval data: outputs/eval_benchmarks/aime26_metadata.jsonl
- MinervaMath eval data: outputs/eval_benchmarks/minervamath_metadata.jsonl
- OlympiadBench eval data: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Eval root dir: outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks
- Log dir: outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs
- AIME26 output dir: outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/aime26
- MinervaMath output dir: outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/minervamath
- OlympiadBench output dir: outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/olympiadbench
- Batch size: 16
- Max examples: 0
- Max prompt length: 2048
- Max new tokens: 1024
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- Seed: 42

AIME26 command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/eval_benchmarks/aime26_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/aime26 \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  > outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs/aime26.log 2>&1 &
```

MinervaMath command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/eval_benchmarks/minervamath_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/minervamath \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  > outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs/minervamath.log 2>&1 &
```

OlympiadBench command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/olympiadbench \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  > outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs/olympiadbench.log 2>&1 &
```

Sequential launch command:

```bash
nohup bash -c '
cd /home/kimjh/Baysian_reward_GRPO

mkdir -p \
  outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs \
  outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/aime26 \
  outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/minervamath \
  outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/olympiadbench

CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/eval_benchmarks/aime26_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/aime26 \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 4096 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  2>&1 | tee outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs/aime26.log

CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/eval_benchmarks/minervamath_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/minervamath \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  2>&1 | tee outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs/minervamath.log

CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500 \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/olympiadbench \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  2>&1 | tee outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/logs/olympiadbench.log
' > outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/nohup.out 2>&1 &
```

Expected output layout:

```text
outputs/bigmath_qwen3_1p7b_answer_only_grpo_eval_benchmarks/
├── nohup.out
├── logs/
│   ├── aime26.log
│   ├── minervamath.log
│   └── olympiadbench.log
├── aime26/
│   ├── summary.json
│   └── predictions.jsonl
├── minervamath/
│   ├── summary.json
│   └── predictions.jsonl
└── olympiadbench/
    ├── summary.json
    └── predictions.jsonl
```

## Train Job - Big-Math BARL Answer-only GRPO Qwen3-8B steps1536

- Status: command provided to user for manual launch
- Method: Answer-only GRPO
- Reward type: answer_only_correctness
- Model: Qwen/Qwen3-8B
- Train data: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
- Train data source: SynthLabsAI/Big-Math-RL-Verified filtered BARL-style pool
- Train size: 12288
- Eval metadata path passed to script: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Eval size: 0
- Eval note: eval data is not used during training because eval_size is 0
- Target eval benchmarks after training: AIME26, MinervaMath, OlympiadBench
- Output dir: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536
- Log file: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536/train.log
- LoRA: true
- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- BF16: true
- Gradient checkpointing: true
- Max steps: 1536
- Max steps rationale: exact 1 epoch over 12288 prompts with effective prompt batch 8
- Num generations: 8
- Per-device train batch size: 1
- Gradient accumulation steps: 8
- Effective prompt batch: 8
- Prompt exposures: 12288
- Epoch estimate: 1.00 epoch over 12288 train prompts
- Save steps: 128
- Logging steps: 10
- Max prompt length: 2048
- Max completion length: 1024
- Temperature: 0.7
- Top p: 0.95
- Expected runtime estimate: heavy; likely substantially longer than Qwen3-4B 1536-step run

Command:

```bash
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python Answer_only_GRPO.py \
  --model_name Qwen/Qwen3-8B \
  --use_fixed_metadata \
  --train_metadata_path outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --train_size 12288 \
  --eval_size 0 \
  --output_dir outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536 \
  --num_generations 8 \
  --max_steps 1536 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --save_steps 128 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --use_lora \
  --bf16 \
  --gradient_checkpointing \
  > outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536/train.log 2>&1 &
```

## Eval Jobs - Big-Math BARL Answer-only GRPO Qwen3-4B checkpoint-1536

- Status: command provided to user for manual launch
- Eval target: Big-Math BARL Answer-only GRPO - Qwen3-4B
- Model: Qwen/Qwen3-4B
- Adapter checkpoint: outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/checkpoint-1536
- Eval benchmarks: AIME26, MinervaMath, OlympiadBench
- AIME26 eval data: outputs/eval_benchmarks/aime26_metadata.jsonl
- MinervaMath eval data: outputs/eval_benchmarks/minervamath_metadata.jsonl
- OlympiadBench eval data: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Eval root dir: outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks
- Log dir: outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/logs
- AIME26 output dir: outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/aime26
- MinervaMath output dir: outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/minervamath
- OlympiadBench output dir: outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/olympiadbench
- Batch size: 16
- Max examples: 0
- Max prompt length: 2048
- Max new tokens:
  - AIME26: 4096
  - MinervaMath: 1024
  - OlympiadBench: 1024
- AIME26 max-new-tokens rule: use 4096 for AIME26 evals for consistency across runs
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- Seed: 42

Sequential launch command:

```bash
nohup bash -c '
cd /home/kimjh/Baysian_reward_GRPO

mkdir -p \
  outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/logs \
  outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/aime26 \
  outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/minervamath \
  outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/olympiadbench

CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/checkpoint-1536 \
  --eval_metadata_path outputs/eval_benchmarks/aime26_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/aime26 \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 4096 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  2>&1 | tee outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/logs/aime26.log

CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/checkpoint-1536 \
  --eval_metadata_path outputs/eval_benchmarks/minervamath_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/minervamath \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  2>&1 | tee outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/logs/minervamath.log

CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-4B \
  --adapter_path outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/checkpoint-1536 \
  --eval_metadata_path outputs/eval_benchmarks/olympiadbench_metadata.jsonl \
  --output_dir outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/olympiadbench \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  2>&1 | tee outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/logs/olympiadbench.log
' > outputs/bigmath_qwen3_4b_answer_only_grpo_eval_benchmarks/nohup.out 2>&1 &
```

## Training Job Update - Qwen3-1.7B Bayesian GRPO SFT+DPO vLLM r3/r4

- Status: r3 failed before training; r4 command provided
- r3 output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r3
- r3 backend: vLLM colocate, `vllm_model_impl=vllm` default
- r3 failure: vLLM Qwen3 model implementation still entered torch compile despite AOT/standalone compile env flags
- Error key: `torch._dynamo.exc.BackendCompilerFailed` with fake tensor mode mismatch
- Interpretation: r3 is not a training/data problem. It is vLLM's native Qwen3 compile path in this environment.
- r4 change: call `Bayesian_Full_GRPO_learned.py` directly and set `--vllm_model_impl transformers`.
- Reason for direct call: `run_grpo_bayesian_with_learned_analyzer.py` currently forwards `--use_vllm` but does not expose/forward `--vllm_model_impl`.
- r4 output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r4_transformers_impl
- vLLM remains enabled in r4.

Update:
- r4 reached training start with vLLM enabled, but failed during learned analyzer load.
- r4 failure: `AssertionError: Current vLLM config is not set`.
- Cause: `--vllm_model_impl transformers` patches HuggingFace Qwen3 classes in-process; later analyzer loading with `AutoModelForCausalLM` hits the patched class outside vLLM's config context.
- Fix applied in repo: `Answer_only_GRPO.py` now patches TRL colocated vLLM init to pass `enforce_eager=True` and `compilation_config=0`.
- r5 direction: use native `vllm_model_impl=vllm` again, not `transformers`, with the eager init patch active.
- r5 output dir: outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r5_native_eager

## Eval Job - Pure Base Chat-Template Problem-Only vLLM All Benchmarks

- Status: command prepared; ready to launch on GPU0
- Purpose: rerun the pure base model benchmark with only tokenizer chat template enabled.
- Difference from previous pure raw eval: problem text is still the only content, but it is wrapped as a chat-template user message with `add_generation_prompt=True`.
- No system prompt.
- No extra answer-format instruction.
- Backend: vLLM
- vLLM stability settings in `eval_pure_base_model.py`: `enforce_eager=True`, `compilation_config=0`
- vLLM sampler setting: `VLLM_USE_FLASHINFER_SAMPLER=0`
  - Reason: current vLLM 0.21 / CUDA 13 stack failed during FlashInfer top-k/top-p sampler init on GPU0.
- GPU: 0
- Models:
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen3-4B
  - Qwen/Qwen3-8B
- Datasets per model:
  - gsm8k
  - math500
  - aime26
  - minervamath
  - olympiadbench
- Total eval jobs: 15
- Output root: `outputs/pure_base_chat_template_eval_benchmarks_vllm`
- Shared config:
  - `--dataset_key all`
  - `--batch_size 32`
  - `--max_examples 0`
  - `--max_prompt_length 2048`
  - `--max_new_tokens 0`
  - `--seed 42`
  - `--no_do_sample`
  - `--bf16`
  - `--use_chat_template`
  - `--use_vllm`
  - `--vllm_gpu_memory_utilization 0.90`
  - `--vllm_tensor_parallel_size 1`
  - `--vllm_max_model_len 6144`
  - `--vllm_dtype bfloat16`

Launch command:

```bash
cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export OUT=outputs/pure_base_chat_template_eval_benchmarks_vllm

mkdir -p outputs/gsm8k_full_train_seed42
ln -sf ../gsm8k_experiments/metadata_fulltrain_seed42/selected_test_metadata.jsonl \
  outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl

mkdir -p "$OUT/logs"

nohup bash -c '
set -euo pipefail

echo "[EVAL] Qwen3-1.7B pure base chat-template problem-only, all benchmarks"
python3 eval_pure_base_model.py \
  --model_key qwen3_1p7b \
  --dataset_key all \
  --output_root "$OUT/qwen3_1p7b" \
  --batch_size 32 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 0 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --use_chat_template \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len 6144 \
  --vllm_dtype bfloat16 \
  2>&1 | tee "$OUT/logs/qwen3_1p7b.log"

echo "[EVAL] Qwen3-4B pure base chat-template problem-only, all benchmarks"
python3 eval_pure_base_model.py \
  --model_key qwen3_4b \
  --dataset_key all \
  --output_root "$OUT/qwen3_4b" \
  --batch_size 32 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 0 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --use_chat_template \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len 6144 \
  --vllm_dtype bfloat16 \
  2>&1 | tee "$OUT/logs/qwen3_4b.log"

echo "[EVAL] Qwen3-8B pure base chat-template problem-only, all benchmarks"
python3 eval_pure_base_model.py \
  --model_key qwen3_8b \
  --dataset_key all \
  --output_root "$OUT/qwen3_8b" \
  --batch_size 32 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 0 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --use_chat_template \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len 6144 \
  --vllm_dtype bfloat16 \
  2>&1 | tee "$OUT/logs/qwen3_8b.log"

echo "[DONE] pure base chat-template problem-only vLLM eval finished"
' > "$OUT/nohup.out" 2>&1 &
```

## Train Job - GSM8K BPR-GRPO Prompted Analyzer Qwen3-8B vLLM steps1000

- Status: command prepared; ready to launch on GPU0
- Method: BPR/Full Bayesian posterior GRPO with prompted analyzer
- Script: `Bayesian_Full_GRPO.py`
- Model: `Qwen/Qwen3-8B`
- Dataset: GSM8K official full train metadata
- Train metadata: `outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_train_metadata.jsonl`
- Eval metadata during train: `outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_valid_metadata.jsonl`
- Train size: 7473
- Eval size: 0
- Max steps: 1000
- GPU: 0
- vLLM: enabled, colocate mode
- vLLM stability env: `VLLM_USE_FLASHINFER_SAMPLER=0`
- vLLM max model length: 4096
- vLLM GPU memory utilization: 0.25
- Prompted analyzer mode: `--prior_mode llm_strategy_prior`
- Prior judge model: `Qwen/Qwen3-8B`
- Evidence judge model: `Qwen/Qwen3-8B`
- Prior lambda: 1.0
- Judge max new tokens: 768
- Per-device train batch size: 8
- Gradient accumulation steps: 1
- Effective prompt batch size: 8
- Note: this keeps the same effective prompt batch as `bsz=1, grad_acc=8`, but uses a larger microbatch and may require more instantaneous VRAM.
- Output dir: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm`
- Log file: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/train.log`

Launch command:

```bash
cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm

mkdir -p "$OUT"

nohup python3 Bayesian_Full_GRPO.py \
  --model_name Qwen/Qwen3-8B \
  --dataset_name gsm8k \
  --use_fixed_metadata \
  --train_metadata_path outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_valid_metadata.jsonl \
  --train_size 7473 \
  --eval_size 0 \
  --output_dir "$OUT" \
  --prior_mode llm_strategy_prior \
  --prior_judge_model Qwen/Qwen3-8B \
  --evidence_judge_model Qwen/Qwen3-8B \
  --prior_lambda 1.0 \
  --prior_softmax_temperature 1.0 \
  --prior_judge_temperature 0.0 \
  --evidence_judge_temperature 0.0 \
  --judge_max_new_tokens 768 \
  --num_generations 8 \
  --max_steps 1000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --logging_steps 10 \
  --save_steps 100 \
  --progress_interval_percent 10 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --bf16 \
  --gradient_checkpointing \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_model_impl vllm \
  --vllm_gpu_memory_utilization 0.25 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_length 4096 \
  --vllm_group_port 51218 \
  --reward_debug_jsonl "$OUT/bayesian_reward_debug.jsonl" \
  > "$OUT/train.log" 2>&1 &
```

### Resume Update - OOM-safe configuration

- Previous run stopped at step 119/1000 due to CUDA OOM during backward.
- Last durable checkpoint: `checkpoint-100`
- Resume config changes:
  - `per_device_train_batch_size`: 8 -> 1
  - `gradient_accumulation_steps`: 1 -> 8
  - effective prompt batch size remains 8
  - `vllm_gpu_memory_utilization`: 0.25 -> 0.20
  - `judge_max_new_tokens`: stays 768
- Resume checkpoint: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/checkpoint-100`
- Resume log: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/train_resume_bsz1_acc8_vllm020.log`
- Resume reward debug: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/bayesian_reward_debug_resume_bsz1_acc8_vllm020.jsonl`

## Eval Job - GSM8K BPR-GRPO Prompted Analyzer Qwen3-8B checkpoint-1000

- Added at: 2026-05-23 11:04:09 KST
- Status: command provided to user for manual launch
- Eval target: completed GSM8K BPR-GRPO Prompted Analyzer - Qwen3-8B
- Model: `Qwen/Qwen3-8B`
- Adapter checkpoint: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/checkpoint-1000`
- Eval data: `outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_test_metadata.jsonl`
- Eval size: full GSM8K test, 1319 rows
- Output dir: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/test_eval_checkpoint1000`
- Log file: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/test_eval_checkpoint1000/eval.log`
- GPU: 0
- Batch size: 16
- Max examples: 0
- Max prompt length: 2048
- Max new tokens: 1024
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- Seed: 42

Command:

```bash
cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OUT=outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/test_eval_checkpoint1000

mkdir -p "$OUT"

nohup python3 eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/checkpoint-1000 \
  --eval_metadata_path outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_test_metadata.jsonl \
  --output_dir "$OUT" \
  --batch_size 16 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  > "$OUT/eval.log" 2>&1 &
```


## Cancelled Job - Qwen3-1.7B BPR-GRPO Learned Analyzer BigMath pipeline on this server

- Updated at: 2026-05-23 15:24:59 KST
- Status: cancelled and local generated outputs deleted
- Reason: same job is running on another server, so this server does not need to continue it
- Killed process group: `416575`
- Killed active PID: `418881`
- Active stage at cancellation: `STEP 4` recompute posterior with SFT analyzer
- Last observed progress before cancellation: `964 / 1500` prompts in `recompute_posterior_with_learned_analyzer.py`
- Deleted output dir: `outputs/bigmath_barl_style_12x1024_seed42/bpr_learned_qwen3_1p7b_sft_dpo_pipeline_lambda1`
- Deleted output dir: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1`
- Preserved source metadata/debug files:
  - `outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl`
  - `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`
  - `outputs/incoming/qwen3_1p7b_bpr_learned_analyzer_math500_prompted_debug/bayesian_reward_debug.jsonl`

## Planned Train Job - Qwen3-4B BPR-GRPO Learned Analyzer BigMath SFT+DPO+GRPO

- Added at: 2026-05-23 15:35:56 KST
- Status: receive teacher debug file, then run sequential SFT -> recompute -> DPO -> learned-analyzer GRPO
- Target eval family after training: AIME26, MinervaMath, OlympiadBench
- Model: `Qwen/Qwen3-4B`
- Method: BPR-GRPO / Bayesian GRPO with learned SFT+DPO analyzer
- Source training data: BigMath BARL 12x1024 fixed metadata
- Train metadata: `outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl`
- Eval metadata placeholder during training: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`
- Required incoming teacher debug file: `outputs/incoming/qwen3_4b_bpr_prompted_bigmath_teacher_debug/bayesian_reward_debug.jsonl`
- Optional incoming adapter dir: `outputs/incoming/qwen3_4b_bpr_prompted_bigmath_adapter`
- Pipeline root: `outputs/bigmath_barl_style_12x1024_seed42/bpr_learned_qwen3_4b_sft_dpo_pipeline_lambda1`
- Final GRPO output: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1`
- SFT config: 1 epoch, lr `5e-5`, batch 1, grad accumulation 8, max length 4096, bf16
- Recompute config: batch 2, max new tokens 768, max input tokens 4096
- DPO config: 1 epoch, lr `1e-5`, beta 0.1, batch 1, grad accumulation 8, max length 4096, bf16
- GRPO config: max steps 1500, train size 12288, num generations 8, batch 8, grad accumulation 1, LoRA r16 alpha32 dropout 0.05, bf16, judge max tokens 768, max prompt 2048, max completion 1024, lr 5e-6, temperature 0.7, top_p 0.95, save_steps 100, logging_steps 10, vLLM colocate enabled
- Fairness note: GRPO solver numeric budget is matched to the Qwen3-4B BPR-GRPO Prompted Analyzer setup. vLLM is enabled as the rollout backend for speed; the sample budget/token budget remains matched.

## Eval Job - Qwen3-1.7B Bayesian GRPO SFT+DPO Learned Analyzer MATH-500 checkpoint-1500 max4096

- Added at: 2026-05-23 15:50:57 KST
- Status: command provided
- Model: `Qwen/Qwen3-1.7B`
- Method: Bayesian GRPO with SFT+DPO learned analyzer
- Adapter checkpoint: `outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r5_native_eager/checkpoint-1500`
- Eval data: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Eval size: full MATH-500 test, 500 rows
- Output dir: `outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r5_native_eager/test_eval_checkpoint1500_maxtok4096_vllm`
- Max prompt length: 2048
- Max new tokens: 4096
- vLLM max model length: 6144
- Batch size: 32
- Decoding: greedy
- BF16: true
- Load adapter: true
- Inference backend: vLLM

## Planned Train Job - MATH-500 BPR-GRPO Prompted Analyzer Qwen3-8B vLLM steps1500

- Added at: 2026-05-23 16:17:09 KST
- Status: command provided
- Model: `Qwen/Qwen3-8B`
- Method: BPR-GRPO / Bayesian GRPO Prompted Analyzer
- Dataset: MATH-500 fixed metadata, train 12k, eval disabled during training
- Train metadata: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl`
- Eval metadata path placeholder: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Output dir: `outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm`
- Reward debug: `outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/bayesian_reward_debug.jsonl`
- Config: max steps 1500, train size 12000, eval size 0, num generations 8, per-device batch 1, grad accumulation 8, effective batch 8
- Tokens: max prompt 2048, max completion 1024, judge max new tokens 768
- Optimizer: lr 5e-6, LoRA r16 alpha32 dropout 0.05, bf16, gradient checkpointing
- Prior/evidence judge: `Qwen/Qwen3-8B`
- vLLM: colocate, memory utilization 0.20, max model length 4096, tensor parallel size 1

## Planned Train Job - Qwen3-1.7B BPR-GRPO Learned Analyzer MATH-500 SFT+DPO+GRPO

- Added at: 2026-05-23 13:43:46 KST
- Status: command provided; waiting for required prompted debug file if using incoming data
- Model: `Qwen/Qwen3-1.7B`
- Method: BPR-GRPO / Bayesian GRPO with learned SFT+DPO analyzer
- Dataset: MATH-500 fixed metadata, train 12k, eval disabled during training
- Required incoming debug file: `outputs/incoming/qwen3_1p7b_bpr_learned_analyzer_math500_prompted_debug/bayesian_reward_debug.jsonl`
- Metadata dir: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42`
- Pipeline root: `outputs/math500_experiments/bpr_learned_qwen3_1p7b_sft_dpo_pipeline_lambda1`
- Final GRPO output: `outputs/math500_experiments/bpr_learned_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_vllm`
- SFT config: 1 epoch, lr `5e-5`, batch 1, grad accumulation 8, max length 4096
- DPO config: 1 epoch, lr `1e-5`, beta 0.1, batch 1, grad accumulation 8, max length 4096
- GRPO config: max steps 1500, num generations 8, batch 1, grad accumulation 8, LoRA, bf16, vLLM colocate


## Eval Job - GSM8K BPR-GRPO Prompted Analyzer Qwen3-8B checkpoint-1000 batch64

- Added at: 2026-05-23 11:24:43 KST
- Status: launched by assistant on GPU0
- Eval target: completed GSM8K BPR-GRPO Prompted Analyzer - Qwen3-8B
- Model: `Qwen/Qwen3-8B`
- Adapter checkpoint: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/checkpoint-1000`
- Eval data: `outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_test_metadata.jsonl`
- Eval size: full GSM8K test, 1319 rows
- Output dir: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/test_eval_checkpoint1000_bsz64`
- Log file: `outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/test_eval_checkpoint1000_bsz64/eval.log`
- GPU: 0
- Batch size: 64
- Max examples: 0
- Max prompt length: 2048
- Max new tokens: 1024
- Decoding: greedy
- do_sample: false
- BF16: true
- Load adapter: true
- Seed: 42

Command:

```bash
cd /home/kimjh/Baysian_reward_GRPO
source .venv/bin/activate
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/test_eval_checkpoint1000_bsz64

mkdir -p "$OUT"

nohup python3 eval_solver_checkpoint.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/checkpoint-1000 \
  --eval_metadata_path outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_test_metadata.jsonl \
  --output_dir "$OUT" \
  --batch_size 64 \
  --max_examples 0 \
  --max_prompt_length 2048 \
  --max_new_tokens 1024 \
  --seed 42 \
  --no_do_sample \
  --bf16 \
  --load_adapter \
  > "$OUT/eval.log" 2>&1 &
```

## Results Snapshot - Main Table + Server Eval/Test Config Archive

- Added at: 2026-05-24 17:15:42 KST
- Status: recorded and saved
- Detailed archive: `EXPERIMENT_RESULTS_AND_EVAL_CONFIGS.md`
- Main result table source: user-provided current canonical table
- Average definition: average over GSM8K, MATH-500, MinervaMath, and OlympiadBench; AIME26 is tracked in eval artifacts but excluded from this main average.
- Server-side eval/test artifacts archived: 77 `summary.json` eval/test rows
- Eval metadata build summaries archived: 3 rows
- Training/launcher config artifacts archived: 15 rows
- Eval config fields recorded where available: output dir, model, method, benchmark, accuracy, correct/total, batch size, max prompt length, max new tokens, inference backend, prompt mode, sampling flag, seed, average generation length, adapter/checkpoint path.
- Canonical pure-base config note: `outputs/pure_base_chat_template_eval_benchmarks_vllm_maxtok_math_aime4096` uses chat template only, no system prompt, no user instruction, vLLM backend, max new tokens 4096 for MATH-500/AIME26 and 1024 for GSM8K/Minerva/OlympiadBench.

## Completed Train Job - Qwen3-4B BPR-GRPO Learned Analyzer BigMath final GRPO

- Updated at: 2026-05-24 18:23 KST
- Status: completed normally
- Final stage completed: learned-analyzer GRPO solver training
- Model: `Qwen/Qwen3-4B`
- Method: BPR-GRPO / Bayesian GRPO with learned SFT+DPO analyzer
- Training data: BigMath BARL-style 12 x 1024 fixed metadata
- Train size: 12288 prompts
- Max steps: 1500
- Num generations: 8
- Per-device train batch size: 8
- Gradient accumulation steps: 1
- Max prompt length: 2048
- Max completion length: 1024
- Judge max new tokens: 768
- Prior lambda: 1.0
- vLLM: enabled during GRPO training
- Pipeline root: `outputs/bigmath_barl_style_12x1024_seed42/bpr_learned_qwen3_4b_sft_dpo_pipeline_lambda1`
- Final output dir: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1`
- Final checkpoint: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500`
- Final adapter also saved at output root:
  - `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/adapter_model.safetensors`
  - `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/adapter_config.json`
- Reward debug log: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`
- Completion log: `outputs/bigmath_barl_style_12x1024_seed42/bpr_learned_qwen3_4b_sft_dpo_pipeline_lambda1/nohup.out`
- Last log line: `100%|██████████| 1500/1500 [15:49:22<00:00, 37.98s/it]`
- Next recommended step: evaluate checkpoint-1500 on GSM8K, MATH-500, MinervaMath, and OlympiadBench with the same eval protocol used in the main table.

## Eval Result - Qwen3-4B BPR-GRPO Learned Analyzer BigMath checkpoint-1500 Minerva/OlympiadBench

- Added at: 2026-05-24 18:32 KST
- Status: completed
- Model: `Qwen/Qwen3-4B`
- Method: BPR-GRPO / Bayesian GRPO with learned SFT+DPO analyzer
- Adapter checkpoint: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500`
- Output root: `outputs/bigmath_qwen3_4b_bpr_learned_analyzer_eval_benchmarks`
- Eval backend: vLLM
- Batch size: 32
- Max prompt length: 2048
- Max new tokens: 1024
- Decoding: greedy (`--no_do_sample`)
- Seed: 42
- BF16: true
- MinervaMath result: 70/272 = 25.74%, format success 96.69%, avg generation length 473.79
- OlympiadBench result: 237/674 = 35.16%, format success 73.00%, avg generation length 716.16
- Main table update: Qwen3-4B BPR-GRPO (Learned Analyzer) now has Minerva 25.74%, OlympiadBench 35.16%, average 56.09%.
- Detailed archive updated: `EXPERIMENT_RESULTS_AND_EVAL_CONFIGS.md`

## Planned Eval Suite - Unified max_new_tokens=4096 5x vLLM table rerun

- Added at: 2026-05-24 18:49 KST
- Status: code and incoming adapter directories prepared; not launched yet
- Goal: re-evaluate the full 15-row main table with max new tokens fixed to 4096 for every benchmark, repeat each cell 5 times, and aggregate mean/std accuracy.
- Script: `scripts/run_unified_4096_eval_repeats.py`
- Output root: `outputs/unified_4096_eval_5x`
- Repeats: 5
- Seed schedule: 42, 43, 44, 45, 46
- Benchmarks: GSM8K, MATH-500, MinervaMath, OlympiadBench
- Max prompt length: 2048
- Max new tokens: 4096 for all benchmarks
- vLLM max model length: 6144
- Default batch size: 32
- Inference backend: vLLM
- Pure-based mode: chat-template problem-only, no system prompt, no user instruction
- Structured/base and LoRA checkpoint modes: solver structured prompt with `[Strategy]`, `[Reasoning]`, `[Final Answer]`
- Required incoming adapter manifest: `outputs/unified_4096_eval_5x/required_adapters.md`
- Required incoming adapter root: `outputs/incoming/unified_4096_eval_adapters`
- Current check-only result: 300 planned eval jobs, 23 missing adapter cells, incoming directories created.

## Eval Suite Update - Main Table Structured-Base 4096 5x

- Added at: 2026-05-24 20:48 KST
- Status: old pure-base-included GPU0/GPU1 eval stopped; script manifest updated
- Reason: main paper table now uses `Base` as unadapted Qwen3 with the structured solver prompt, not raw/problem-only or pure chat-template prompt-only eval.
- Script: `scripts/run_unified_4096_eval_repeats.py`
- New table rows per model:
  - `Base`
  - `GRPO`
  - `BPR-GRPO (Ours)`
  - `BPR-GRPO (learned analyzer)`
- Removed from main-table rerun: `Pure-based`
- Planned jobs after update: 12 rows x 4 benchmarks x 5 repeats = 240 eval jobs
- Eval config:
  - max prompt length: 2048
  - max new tokens: 4096 for all benchmarks
  - vLLM max model length: 6144
  - decoding: greedy / `--no_do_sample`
  - repeats: 5
  - seed schedule: 42, 43, 44, 45, 46
- Split recommendation:
  - GPU0: GSM8K only
  - GPU1: MATH-500, MinervaMath, OlympiadBench

## Queued Train Suite - Dr.GRPO Answer-only gap-fill on PRO6000


- Added at: 2026-05-25 06:59:50 KST
- Status: queued, waiting for GPU memory below 20000 MiB
- Queue GPU0: Qwen3-1.7B BigMath -> Qwen3-1.7B GSM8K
- Queue GPU1: Qwen3-1.7B MATH-500 -> Qwen3-4B GSM8K
- Method: Dr.GRPO external baseline, answer-only correctness reward, loss_type=dr_grpo, scale_rewards=none, beta=0.0
- Launcher: scripts/run_drgrpo_answer_only_24h_gap_fill_queue.sh
- Logs: outputs/drgrpo_answer_only_24h/logs/gpu0_queue.nohup.log, outputs/drgrpo_answer_only_24h/logs/gpu1_queue.nohup.log

## Training Job - Qwen3-1.7B Dr.GRPO Answer-only BigMath BARL-style

- Added at: 2026-05-25 07:00 KST
- Status: completed
- Server/GPU: PRO6000 GPU0
- PID: 715427
- Method: Dr.GRPO external baseline
- Reward type: answer_only_correctness
- Loss/config difference from Answer-only GRPO: loss_type=dr_grpo, scale_rewards=none, beta=0.0
- This is not BPR reward and not DRA diversity reward.
- Model: Qwen/Qwen3-1.7B
- Train metadata: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
- Eval metadata placeholder: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Train size: 12288
- Eval size during train: 0
- num_generations: 8
- max_steps: 1500
- per_device_train_batch_size: 8
- gradient_accumulation_steps: 1
- effective prompt batch: 8
- max_prompt_length: 2048
- max_completion_length: 1024
- learning_rate: 5e-6
- LoRA: enabled, r=16, alpha=32, dropout=0.05
- bf16: true
- gradient_checkpointing: true
- seed: 42
- logging_steps: 10
- save_steps: 100
- progress_interval_percent: 10
- Output dir: outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1
- Log file: outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/logs/train.nohup.log
- Start progress observed: 6/1500 steps at about 2026-05-25 07:00 KST.
- Finish time: 2026-05-25 12:23 KST
- Runtime: 5:24:12
- Final checkpoint: outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/checkpoint-1500
- Checkpoint list: checkpoint-100, checkpoint-200, checkpoint-300, checkpoint-400, checkpoint-500, checkpoint-600, checkpoint-700, checkpoint-800, checkpoint-900, checkpoint-1000, checkpoint-1100, checkpoint-1200, checkpoint-1300, checkpoint-1400, checkpoint-1500
- Completion log line: 100%|██████████| 1500/1500 [5:24:12<00:00, 12.97s/it]

## Training Job - Qwen3-1.7B Dr.GRPO Answer-only MATH-500 full train

- Added at: 2026-05-25 07:00 KST
- Status: completed
- Server/GPU: PRO6000 GPU1
- PID: 715428
- Method: Dr.GRPO external baseline
- Reward type: answer_only_correctness
- Loss/config difference from Answer-only GRPO: loss_type=dr_grpo, scale_rewards=none, beta=0.0
- This is not BPR reward and not DRA diversity reward.
- Model: Qwen/Qwen3-1.7B
- Train metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
- Eval metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Train size: 12000
- Eval size during train: 0
- num_generations: 8
- max_steps: 1500
- per_device_train_batch_size: 8
- gradient_accumulation_steps: 1
- effective prompt batch: 8
- max_prompt_length: 2048
- max_completion_length: 1024
- learning_rate: 5e-6
- LoRA: enabled, r=16, alpha=32, dropout=0.05
- bf16: true
- gradient_checkpointing: true
- seed: 42
- logging_steps: 10
- save_steps: 100
- progress_interval_percent: 10
- Output dir: outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1
- Log file: outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1/logs/train.nohup.log
- Start progress observed: 6/1500 steps at about 2026-05-25 07:00 KST.
- Finish time: 2026-05-25 12:44 KST
- Runtime: 5:45:15
- Final checkpoint: outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1/checkpoint-1500
- Checkpoint list: checkpoint-100, checkpoint-200, checkpoint-300, checkpoint-400, checkpoint-500, checkpoint-600, checkpoint-700, checkpoint-800, checkpoint-900, checkpoint-1000, checkpoint-1100, checkpoint-1200, checkpoint-1300, checkpoint-1400, checkpoint-1500
- Completion log line: 100%|██████████| 1500/1500 [5:45:15<00:00, 13.81s/it]

## Eval Job - Qwen3-1.7B Dr.GRPO Answer-only BigMath checkpoint-1500 AIME26/Minerva/OlympiadBench

- Added at: 2026-05-25 13:02 KST
- Status: completed
- Server/GPU: PRO6000 GPU0
- Launcher PID: 741201
- Current eval PID at launch: 741207
- vLLM engine PID at launch: 741572
- Model: Qwen/Qwen3-1.7B
- Method: Dr.GRPO external baseline checkpoint eval
- Adapter checkpoint: outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/checkpoint-1500
- Eval benchmarks: AIME26, MinervaMath, OlympiadBench
- Eval metadata:
  - AIME26: outputs/eval_benchmarks/aime26_metadata.jsonl
  - MinervaMath: outputs/eval_benchmarks/minervamath_metadata.jsonl
  - OlympiadBench: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
- Output root: outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_maxtok4096_vllm
- Logs:
  - outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_maxtok4096_vllm/logs/aime26.log
  - outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_maxtok4096_vllm/logs/minervamath.log
  - outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_maxtok4096_vllm/logs/olympiadbench.log
- Batch size: 32
- Max examples: 0
- Max prompt length: 2048
- Max new tokens: 4096
- Decoding: greedy, no sampling
- Seed: 42
- BF16: true
- Load adapter: true
- Backend: vLLM
- vLLM config: gpu_memory_utilization=0.85, tensor_parallel_size=1, max_model_length=6144
- Finish time: 2026-05-25 13:09 KST
- Results:
  - AIME26: 2/30 = 6.67%, format success 93.33%, avg generation length 839.03
  - MinervaMath: 41/272 = 15.07%, format success 98.53%, avg generation length 483.69
  - OlympiadBench: 185/674 = 27.45%, format success 96.29%, avg generation length 656.06

## Eval Job - Qwen3-1.7B Dr.GRPO Answer-only MATH-500 checkpoint-1500 test

- Added at: 2026-05-25 13:02 KST
- Status: completed
- Server/GPU: PRO6000 GPU1
- Launcher PID: 741204
- Eval PID at launch: 741213
- vLLM engine PID at launch: 741565
- Model: Qwen/Qwen3-1.7B
- Method: Dr.GRPO external baseline checkpoint eval
- Adapter checkpoint: outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1/checkpoint-1500
- Eval benchmark: MATH-500 test
- Eval metadata: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
- Output root: outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1/test_eval_checkpoint1500_maxtok4096_vllm
- Log: outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1/test_eval_checkpoint1500_maxtok4096_vllm/logs/math500.log
- Batch size: 32
- Max examples: 0
- Max prompt length: 2048
- Max new tokens: 4096
- Decoding: greedy, no sampling
- Seed: 42
- BF16: true
- Load adapter: true
- Backend: vLLM
- vLLM config: gpu_memory_utilization=0.85, tensor_parallel_size=1, max_model_length=6144
- Finish time: 2026-05-25 13:03 KST
- Results:
  - MATH-500: 295/500 = 59.00%, format success 97.20%, avg generation length 476.43
