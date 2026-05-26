# Experiment Run Stack

This file is the running ledger for training jobs. Add a new entry whenever a new train run is started.

Rule: append new training jobs at the bottom of this file. The file should read oldest to newest from top to bottom.

Recording rules:

- When a train starts, record the train command context, core config, output dir, log path, reward debug path, checkpoint cadence, and epoch estimate.
- When a train finishes, update the same entry with finish time, total runtime, final checkpoint, checkpoint list, and final status.
- When eval starts, add an eval subsection under the same train entry with target checkpoint, eval metadata path, eval output dir, batch size, prompt/new-token limits, sampling mode, GPU, and log path.
- When eval finishes, update that eval subsection with accuracy, correct/total, format success rate, generated length mean, summary path, and final status.
- New train entries must be appended at the bottom of the job list. Do not insert newer jobs above older jobs.

Last updated: 2026-05-25 KST

## Entry Template

Use this shape for new jobs:

### YYYY-MM-DD - `<Model> <Method> / <Dataset>`

- Status: running | completed | failed | stopped
- Script: `<script.py>`
- Output dir: `<path>`
- Log: `<path>`
- Reward debug: `<path or n/a>`

Train config:

```text
model_name:
dataset_name:
train_metadata_path:
eval_metadata_path:
train_size:
eval_size:
method / prior_mode / reward_type:
num_generations:
max_steps:
max_prompt_length:
max_completion_length:
per_device_train_batch_size:
gradient_accumulation_steps:
effective_batch_size:
learning_rate:
logging_steps:
save_steps:
progress_interval_percent:
use_lora:
bf16:
gradient_checkpointing:
seed:
```

Train completion:

- Finished at: pending
- Total runtime: pending
- Final checkpoint: pending
- Checkpoints: pending

Eval:

- Status: in-progress
- Started at: 2026-05-25 15:14 KST
- PID at launch: `1073025`
- Eval script: `run_eval_drgrpo_qwen4b_bigmath_benchmarks.sh`
- Output root: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok`
- Logs:
  - `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/logs/nohup.out`
  - `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/logs/aime26.log`
  - `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/logs/minervamath.log`
  - `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/logs/olympiadbench.log`
- Eval model: `Qwen/Qwen3-4B + LoRA adapter`
- Adapter path: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/checkpoint-1500`
- Eval data:
  - AIME26: `outputs/eval_benchmarks/aime26_metadata.jsonl`, 30 examples, `max_new_tokens=4096`
  - MinervaMath: `outputs/eval_benchmarks/minervamath_metadata.jsonl`, 272 examples, `max_new_tokens=1024`
  - OlympiadBench: `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`, 674 examples, `max_new_tokens=1024`
- Eval config: `batch_size=16`, `max_prompt_length=2048`, deterministic decoding, `do_sample=false`, `temperature=0.0`, `top_p=1.0`, `seed=42`, backend=`transformers/no_vllm`
- Partial results as of 2026-05-25 15:30 KST:
  - AIME26: `2/30 = 6.67%`, `format_success_rate=93.33%`, `max_new_tokens=4096`, summary: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/aime26/summary.json`
  - MinervaMath: `69/272 = 25.37%`, `format_success_rate=98.90%`, `generated_length_mean=789.0`, `max_new_tokens=1024`, summary: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/minervamath/summary.json`
  - OlympiadBench: in progress

### 2026-05-24 - Qwen3-8B BPR-GRPO Prompted Analyzer / BigMath BARL-style full / benchmark eval

- Status: completed
- Eval output root: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096`
- Adapter path: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/checkpoint-1536`
- Logs: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/logs/nohup.out`, `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/logs/minerva_olympiad.nohup.out`

Eval config:

```text
model_name: Qwen/Qwen3-8B
checkpoint_type: peft_adapter
batch_size: 64
do_sample: false
temperature: null
top_p: null
max_prompt_length: 2048
max_new_tokens: 4096
seed: 42
inference_backend: vllm
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.90
vllm_max_model_length: 6144
```

Results:

- AIME26: 4 / 30 = 13.33%, format_success_rate 93.33%, generated_length_mean 1303.97
- MinervaMath: 67 / 272 = 24.63%, format_success_rate 100.00%, generated_length_mean 456.26
- OlympiadBench: 303 / 674 = 44.96%, format_success_rate 93.03%, generated_length_mean 969.13

Run note:

- The first submitted command stopped after AIME26 due to a malformed continuation line; MinervaMath and OlympiadBench were rerun and completed successfully.
- Target checkpoint: pending
- Eval metadata path: pending
- Output dir: pending
- Log: pending
- GPU: pending
- batch_size: pending
- max_prompt_length: pending
- max_new_tokens: pending
- do_sample: pending
- Summary path: pending
- Accuracy: pending
- Correct / total: pending
- Format success rate: pending
- Generated length mean: pending

### 2026-05-24 - consolidated results and eval config index

- Status: saved
- Summary file: `EXPERIMENT_RESULTS_SUMMARY.md`
- Contents: current master result table, in-progress jobs, and local eval/test config catalog from completed `outputs/**/summary.json` files.

### 2026-05-25 - Qwen3-4B VeriFree controlled LoRA / BigMath BARL-style full

- Status: stopped for Dr.GRPO BigMath baseline
- Started at: 2026-05-25 06:15 KST
- Stopped at: 2026-05-25 06:52 KST
- Script: `VeriFree_LoRA.py`
- Launcher: `run_verifree_lora_qwen4b_bigmath_full.sh`
- Output dir: `outputs/bigmath_barl_style_12x1024_seed42/verifree_lora_qwen4b_bigmath12x1024_n8_steps1536_bsz8_scale0p1`
- Log: `outputs/bigmath_barl_style_12x1024_seed42/verifree_lora_qwen4b_bigmath12x1024_n8_steps1536_bsz8_scale0p1/logs/train.nohup.log`
- Reward debug: `outputs/bigmath_barl_style_12x1024_seed42/verifree_lora_qwen4b_bigmath12x1024_n8_steps1536_bsz8_scale0p1/verifree_reward_debug.jsonl`
- GPU: `0`
- PID at launch check: `994098`

Train config:

```text
model_name: Qwen/Qwen3-4B
dataset_name: fixed_metadata / BigMath BARL-style 12x1024
train_metadata_path: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
train_size: 12288
eval_size: 0
method: VeriFree controlled LoRA port
official_reference: sail-sg/VeriFree
objective: PG on sampled reasoning prefix + reward-weighted SFT on appended gold answer
advantage_type: rloo
reward_source: p
reward_scale: 0.1
sft_coef_source: reward
num_generations: 8
max_steps: 1536
max_prompt_length: 2048
max_completion_length: 1024
per_device_train_batch_size: 8
generation_prompt_batch_size: 4
mini_train_batch_size: 2
effective_prompt_batch: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 128
progress_interval_percent: 10
use_lora: true
lora_r / alpha / dropout: 16 / 32 / 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Train completion:

- Finished at: pending
- Total runtime: pending
- Final checkpoint: pending
- Checkpoints: pending

Eval:

- Status: not started

Stop note:

- Stopped early because the full VeriFree controlled run projected beyond the 24-hour deadline.
- Replacement external baseline started on the same GPU: Qwen3-4B Dr.GRPO Answer-only / BigMath 1500-step.

### 2026-05-25 - Qwen3-4B VeriFree controlled LoRA / MATH-500 full

- Status: stopped for Dr.GRPO timing probe
- Started at: 2026-05-25 06:15 KST
- Stopped at: 2026-05-25 06:45 KST
- Script: `VeriFree_LoRA.py`
- Launcher: `run_verifree_lora_qwen4b_math500_full.sh`
- Output dir: `outputs/math500_experiments/verifree_lora_qwen4b_fulltrain12k_n8_steps1500_bsz8_scale0p1`
- Log: `outputs/math500_experiments/verifree_lora_qwen4b_fulltrain12k_n8_steps1500_bsz8_scale0p1/logs/train.nohup.log`
- Reward debug: `outputs/math500_experiments/verifree_lora_qwen4b_fulltrain12k_n8_steps1500_bsz8_scale0p1/verifree_reward_debug.jsonl`
- GPU: `1`
- PID at launch check: `994246`

Train config:

```text
model_name: Qwen/Qwen3-4B
dataset_name: fixed_metadata / MATH-500 full train
train_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
train_size: 12000
eval_size: 0
method: VeriFree controlled LoRA port
official_reference: sail-sg/VeriFree
objective: PG on sampled reasoning prefix + reward-weighted SFT on appended gold answer
advantage_type: rloo
reward_source: p
reward_scale: 0.1
sft_coef_source: reward
num_generations: 8
max_steps: 1500
max_prompt_length: 2048
max_completion_length: 1024
per_device_train_batch_size: 8
generation_prompt_batch_size: 4
mini_train_batch_size: 2
effective_prompt_batch: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 10
use_lora: true
lora_r / alpha / dropout: 16 / 32 / 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Train completion:

- Finished at: pending
- Total runtime: pending
- Final checkpoint: pending
- Checkpoints: pending

Eval:

- Status: not started

Stop note:

- Stopped early because the full VeriFree controlled run projected beyond the 24-hour deadline.
- Replacement external-baseline timing probe started with Dr.GRPO on GPU 1.

### 2026-05-25 - Qwen3-4B Dr.GRPO Answer-only / MATH-500 full 1500-step timing probe

- Status: completed
- Started at: 2026-05-25 06:45 KST
- Script: `Answer_only_GRPO.py`
- Output dir: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1`
- Log: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/logs/train.nohup.log`
- Reward debug: n/a
- GPU: `1`
- PID at launch check: `1000958`

Train config:

```text
model_name: Qwen/Qwen3-4B
dataset_name: fixed_metadata / MATH-500 full train
train_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
train_size: 12000
eval_size: 0
method: Dr.GRPO external optimizer-side baseline
reward_type: answer_only_correctness
loss_type: dr_grpo
scale_rewards: none
beta: 0.0
num_generations: 8
max_steps: 1500
max_prompt_length: 2048
max_completion_length: 1024
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_prompt_batch: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 10
use_lora: true
lora_r / alpha / dropout: 16 / 32 / 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Timing observation:

- Step 10 log at 2026-05-25 06:49 KST: tqdm ETA `4:52:27`, step time around `14.51s`, total projected runtime around `5.2h`.

Train completion:

- Finished at: 2026-05-25 15:41 KST
- Total runtime: 8h 55m 10s
- Final checkpoint: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/checkpoint-1500`
- Checkpoints: `checkpoint-100` through `checkpoint-1500` every 100 steps

Eval:

- Status: completed
- Finished at: 2026-05-25 15:56 KST
- Output dir: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096`
- Log: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096/logs/run.log`
- Summary: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096/summary.json`
- Predictions: `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096/predictions.jsonl`
- Result: 357/500 = 71.40%
- Format success: 97.20%
- Generated length mean: 538.39 tokens
- Backend: vLLM
- Eval config: batch_size 64, deterministic, max_prompt_length 2048, max_new_tokens 4096, seed 42, tensor_parallel_size 1, gpu_memory_utilization 0.90, vllm_max_model_length 6144

### 2026-05-25 - Qwen3-4B Dr.GRPO Answer-only / BigMath BARL-style full

- Status: completed
- Started at: 2026-05-25 06:52 KST
- Script: `Answer_only_GRPO.py`
- Output dir: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1`
- Log: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/logs/train.nohup.log`
- Reward debug: n/a
- GPU: `0`
- PID at launch check: `1002728`

Train config:

```text
model_name: Qwen/Qwen3-4B
dataset_name: fixed_metadata / BigMath BARL-style 12x1024
train_metadata_path: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
train_size: 12288
eval_size: 0
method: Dr.GRPO external optimizer-side baseline
reward_type: answer_only_correctness
loss_type: dr_grpo
scale_rewards: none
beta: 0.0
num_generations: 8
max_steps: 1500
max_prompt_length: 2048
max_completion_length: 1024
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_prompt_batch: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 10
use_lora: true
lora_r / alpha / dropout: 16 / 32 / 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Config-matching note:

- Matched to the recorded Qwen3-4B BPR BigMath training surface: same model family, train split, train size, `n=8`, LoRA r/alpha/dropout, bf16, gradient checkpointing, `bsz8_acc1`, lr, token limits, seed, save/log cadence, and 1500-step budget.
- Intentional method differences only: BPR uses posterior Bayesian reward with standard GRPO; this external baseline uses answer-only correctness with Dr.GRPO loss and no reward scaling, as prescribed by Dr.GRPO.

Train completion:

- Finished at: 2026-05-25 15:03 KST
- Total runtime: 8h 11m 05s
- Final checkpoint: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/checkpoint-1500`
- Checkpoints: `checkpoint-100` through `checkpoint-1500` every 100 steps

Eval:

- Status: completed for AIME26, MinervaMath, and OlympiadBench
- Started at: 2026-05-25 15:14 KST
- Eval script: `run_eval_drgrpo_qwen4b_bigmath_benchmarks.sh` for initial AIME26/MinervaMath, then manual vLLM rerun for OlympiadBench
- Adapter path: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/checkpoint-1500`
- Results:
  - AIME26: `2/30 = 6.67%`, `batch_size=16`, `max_prompt_length=2048`, `max_new_tokens=4096`, backend=`transformers`, `generated_length_mean=4096.0`, summary: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/aime26/summary.json`
  - MinervaMath: `69/272 = 25.37%`, `batch_size=16`, `max_prompt_length=2048`, `max_new_tokens=1024`, backend=`transformers`, `generated_length_mean=789.0`, summary: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/minervamath/summary.json`
  - OlympiadBench: `247/674 = 36.65%`, `batch_size=64`, `max_prompt_length=2048`, `max_new_tokens=1024`, backend=`vllm`, `vllm_tensor_parallel_size=1`, `vllm_gpu_memory_utilization=0.90`, `vllm_max_model_length=3072`, `generated_length_mean=641.31`, summary: `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_olympiadbench_vllm_bs64_tok1024/summary.json`
- Notes:
  - The first transformers OlympiadBench run was stopped at `160/674` and replaced by the vLLM run above.
  - vLLM is now usable because `Answer_only_GRPO.py` adds the active venv `bin` directory to `PATH`, so `.venv/bin/ninja` is visible to FlashInfer/JIT subprocesses.

## In Progress

### 2026-05-22 - Qwen3-1.7B Bayesian Prompted GRPO / MATH full

- Status: completed; eval completed
- Script: `Bayesian_Full_GRPO.py`
- Output dir: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1`
- Log: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/logs/run.log`
- Reward debug: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`

Config:

```text
model_name: Qwen/Qwen3-1.7B
dataset_name: fixed_metadata
train_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
train_size: 12000
eval_size: 0
min_solve_rate: 0.0
max_solve_rate: 1.0
prior_mode: llm_strategy_prior
prior_judge_model: Qwen/Qwen3-1.7B
evidence_judge_model: Qwen/Qwen3-1.7B
reward_type: posterior_normalized_bayesian_evidence
prior_lambda: 1.0
prior_softmax_temperature: 1.0
num_generations: 8
max_steps: 1500
max_prompt_length: 1024
max_completion_length: 1024
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_batch_size: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 250
progress_interval_percent: 10
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Notes:

- H200-efficient micro-batch setting: `bsz8_acc1`.
- Epoch estimate with effective batch 8: `1500 * 8 / 12000 = 1.0 epoch`.

Train completion:

- Finished at: 2026-05-22 KST
- Total runtime: 13:36:52
- Final checkpoint: `checkpoint-1500`
- Checkpoints: `checkpoint-250`, `checkpoint-500`, `checkpoint-750`, `checkpoint-1000`, `checkpoint-1250`, `checkpoint-1500`

Eval:

- Status: completed
- Target checkpoint: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500`
- Eval metadata path: `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl`
- Eval examples: 500
- Output dir: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_bs64`
- Log: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_bs64/logs/run.log`
- GPU: `CUDA_VISIBLE_DEVICES=1` in provided command
- batch_size: 64
- max_prompt_length: 2048
- max_new_tokens: 1024
- do_sample: false
- bf16: true
- Summary path: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_bs64/summary.json`
- Accuracy: 0.5980
- Correct / total: 299 / 500
- Format success rate: 0.9320
- Generated length mean: 1024.0

Benchmark Eval:

- Status: command provided
- Benchmarks: `aime26`, `minervamath`, `olympiadbench`
- Target checkpoint: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500`
- Eval metadata paths: `outputs/eval_benchmarks/aime26_metadata.jsonl`, `outputs/eval_benchmarks/minervamath_metadata.jsonl`, `outputs/eval_benchmarks/olympiadbench_metadata.jsonl`
- Eval examples: `aime26=30`, `minervamath=272`, `olympiadbench=674`
- Output dir: `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks`
- Logs: `logs/aime26.log`, `logs/minervamath.log`, `logs/olympiadbench.log`
- GPU: `CUDA_VISIBLE_DEVICES=1` in provided command
- batch_size: 64
- max_prompt_length: 2048
- max_new_tokens: 1024
- do_sample: false
- bf16: true
- Summary paths: `aime26/summary.json`, `minervamath/summary.json`, `olympiadbench/summary.json`
- Accuracy: pending
- Correct / total: pending
- Format success rate: pending
- Generated length mean: pending

### 2026-05-21 - Qwen3-1.7B Bayesian GRPO + SFT+DPO Learned Analyzer / GSM8K full

- Status: completed; eval completed
- Script: `Bayesian_Full_GRPO_learned.py`
- Output dir: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1`
- Log: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/logs/run.log`
- Reward debug: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`

Config:

```text
model_name: Qwen/Qwen3-1.7B
dataset_name: fixed_metadata
train_metadata_path: outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl
train_size: 7473
eval_size: 0
min_solve_rate: 0.0
max_solve_rate: 1.0
prior_mode: learned_unified_analyzer
analyzer_model_name: Qwen/Qwen3-1.7B
analyzer_adapter_path: outputs/gsm8k_full_qwen3_1p7b/analyzer_pipeline/dpo_adapter
prior_judge_model: Qwen/Qwen3-1.7B
evidence_judge_model: Qwen/Qwen3-1.7B
reward_type: posterior_normalized_bayesian_evidence
prior_lambda: 1.0
prior_softmax_temperature: 1.0
num_generations: 8
max_steps: 1000
max_prompt_length: 1024
max_completion_length: 1024
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_batch_size: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 5
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Notes:

- Uses the SFT+DPO learned analyzer adapter at `outputs/gsm8k_full_qwen3_1p7b/analyzer_pipeline/dpo_adapter`.
- H200-efficient micro-batch setting: `bsz8_acc1`.
- Epoch estimate with effective batch 8: `1000 * 8 / 7473 = 1.07 epochs`.

Train completion:

- Finished at: 2026-05-22 KST
- Total runtime: 8:37:26
- Final checkpoint: `checkpoint-1000`
- Checkpoints: `checkpoint-100`, `checkpoint-200`, `checkpoint-300`, `checkpoint-400`, `checkpoint-500`, `checkpoint-600`, `checkpoint-700`, `checkpoint-800`, `checkpoint-900`, `checkpoint-1000`

Eval:

- Status: completed
- Target checkpoint: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/checkpoint-1000`
- Eval metadata path: `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl`
- Eval examples: 1319
- Output dir: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/eval_test_bs64`
- Log: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/eval_test_bs64/logs/run.log`
- GPU: `CUDA_VISIBLE_DEVICES=0` in provided command
- batch_size: 64
- max_prompt_length: 2048
- max_new_tokens: 1024
- do_sample: false
- bf16: true
- Summary path: `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/eval_test_bs64/summary.json`
- Accuracy: 0.7869598180439727
- Correct / total: 1038 / 1319
- Format success rate: 0.9954510993176648
- Generated length mean: 505.99469294920397

### 2026-05-22 - Qwen3-4B Bayesian Prompted GRPO / MATH full

- Status: command provided
- Script: `Bayesian_Full_GRPO.py`
- Output dir: `outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0`
- Log: `outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0/logs/run.log`
- Reward debug: `outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0/bayesian_reward_debug.jsonl`

Config:

```text
model_name: Qwen/Qwen3-4B
dataset_name: SynthLabsAI/Big-Math-RL-Verified
train_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
train_size: 12000
eval_size: 0
min_solve_rate: 0.2
max_solve_rate: 0.8
prior_mode: llm_strategy_prior
prior_judge_model: Qwen/Qwen3-4B
evidence_judge_model: Qwen/Qwen3-4B
reward_type: posterior_normalized_bayesian_evidence
prior_lambda: 1.0
prior_softmax_temperature: 1.0
prior_judge_temperature: 0.0
evidence_judge_temperature: 0.0
judge_max_new_tokens: 768
num_generations: 8
max_steps: 1500
max_prompt_length: 2048
max_completion_length: 1024
temperature: 0.7
top_p: 0.95
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_batch_size: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 10
format_bonus: 0.0
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
bf16: true
gradient_checkpointing: true
seed: 42
gpu: CUDA_VISIBLE_DEVICES=0
```

Notes:

- Command was provided for `nohup` launch on GPU 0.
- H200-efficient micro-batch setting: `bsz8_acc1`.
- Epoch estimate with effective batch 8: `1500 * 8 / 12000 = 1.0 epoch`.
- Eval is disabled during training with `eval_size: 0`.

Train completion:

- Finished at: pending
- Total runtime: pending
- Final checkpoint: pending
- Checkpoints: pending

Eval:

- Status: not started
- Target checkpoint: pending
- Eval metadata path: pending
- Eval examples: pending
- Output dir: pending
- Log: pending
- GPU: pending
- batch_size: pending
- max_prompt_length: pending
- max_new_tokens: pending
- do_sample: pending
- bf16: pending
- Summary path: pending
- Accuracy: pending
- Correct / total: pending
- Format success rate: pending
- Generated length mean: pending

### 2026-05-22 - Qwen3-1.7B Bayesian Prompted GRPO / BigMath BARL-style full

- Status: running
- Script: `Bayesian_Full_GRPO.py`
- Output dir: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1`
- Log: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/logs/run.log`
- Reward debug: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`

Config:

```text
model_name: Qwen/Qwen3-1.7B
dataset_name: SynthLabsAI/Big-Math-RL-Verified
train_metadata_path: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
train_size: 12288
eval_size: 0
min_solve_rate: 0.0
max_solve_rate: 1.0
prior_mode: llm_strategy_prior
prior_judge_model: Qwen/Qwen3-1.7B
evidence_judge_model: Qwen/Qwen3-1.7B
reward_type: posterior_normalized_bayesian_evidence
prior_lambda: 1.0
prior_softmax_temperature: 1.0
prior_judge_temperature: 0.0
evidence_judge_temperature: 0.0
judge_max_new_tokens: 768
num_generations: 8
max_steps: 1500
max_prompt_length: 2048
max_completion_length: 1024
temperature: 0.7
top_p: 0.95
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_batch_size: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 10
format_bonus: 0.0
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
bf16: true
gradient_checkpointing: true
seed: 42
```

Dataset stats:

```text
fixed_train_metadata_full_rows: 12288
fixed_eval_metadata_full_rows: 674
selected_train_source_distribution:
  aops_forum: 280
  big_math: 2300
  cn_k12: 3065
  gsm8k: 82
  harp: 180
  math: 449
  olympiads: 1659
  omnimath: 145
  openmath: 73
  orca_math: 4055
selected_train_difficulty_distribution:
  easy: 4372
  medium: 3378
  hard_but_learnable: 3095
  too_hard: 1443
solve_rate_summary: min=0.0, mean=0.46220143636067706, max=1.0
```

Notes:

- Started on GPU 1 by user.
- H200-efficient micro-batch setting: `bsz8_acc1`.
- Epoch estimate with effective batch 8: `1500 * 8 / 12288 = 0.98 epoch`.
- Eval is disabled during training with `eval_size: 0`.
- For later AIME26 eval, use `max_new_tokens: 4096`; MinervaMath/OlympiadBench can stay at `1024` unless changed.

Train completion:

- Finished at: pending
- Total runtime: pending
- Final checkpoint: pending
- Checkpoints: pending

Eval:

- Status: not started
- Target checkpoint: pending
- Eval metadata path: pending
- Eval examples: pending
- Output dir: pending
- Log: pending
- GPU: pending
- batch_size: pending
- max_prompt_length: pending
- max_new_tokens: pending
- do_sample: pending
- bf16: pending
- Summary path: pending
- Accuracy: pending
- Correct / total: pending
- Format success rate: pending
- Generated length mean: pending

## Completed

Add completed runs here or update entries above when jobs finish.

### 2026-05-23 - Qwen3-1.7B BPR-GRPO Prompted Analyzer / BigMath -> GSM8K test eval

- Status: completed
- Note: accidental extra eval; this is not the requested GSM8K-trained Prompted Analyzer checkpoint.
- Finished at: 2026-05-23 13:01:52 KST
- Eval model: `Qwen/Qwen3-1.7B + LoRA adapter`
- Adapter path: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500`
- Checkpoint type: `peft_adapter`
- Eval data: `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl`
- Output dir: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_gsm8k_test_vllm_bs64_tok4096`
- Log: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_gsm8k_test_vllm_bs64_tok4096/logs/run.log`
- Summary: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_gsm8k_test_vllm_bs64_tok4096/summary.json`
- Predictions: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_gsm8k_test_vllm_bs64_tok4096/predictions.jsonl`
- GPU: `1`
- Result: `1023 / 1319 = 77.56%`
- Format success rate: `99.85%`
- Suspicious final answer rate: `0.00%`
- Generated length mean: `270.80`

Config:

```text
backend: vllm
batch_size: 64
max_examples: 0
max_prompt_length: 2048
max_new_tokens: 4096
vllm_max_model_length: 6144
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.5
do_sample: false
temperature: null
top_p: null
seed: 42
```

### 2026-05-24 - Qwen3-8B BPR-GRPO Learned Analyzer / GSM8K test eval

- Status: failed
- Failed at: 2026-05-24 02:30 KST
- Eval model: `Qwen/Qwen3-8B + LoRA adapter`
- Adapter path: `outputs/gsm8k_experiments/bpr_grpo_learned_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz4_acc2_vllm/checkpoint-1000`
- Checkpoint type: `peft_adapter`
- Eval data: `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl`
- Output dir: `outputs/gsm8k_experiments/bpr_grpo_learned_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz4_acc2_vllm/eval_test_vllm_bs64_tok1024`
- Log: `outputs/gsm8k_experiments/bpr_grpo_learned_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz4_acc2_vllm/eval_test_vllm_bs64_tok1024/logs/run.log`
- Summary: not created
- Predictions: not created
- GPU: `1`
- Failure reason: vLLM engine initialization failed because FlashInfer JIT could not find `ninja`.

Config:

```text
backend: vllm
batch_size: 64
max_examples: 0
max_prompt_length: 2048
max_new_tokens: 1024
vllm_max_model_length: 3072
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.9
do_sample: false
temperature: 0.0
top_p: 1.0
seed: 42
```

### 2026-05-24 - Qwen3-4B BPR-GRPO Learned Analyzer / MATH500 full

- Status: completed
- Script: `Bayesian_Full_GRPO_learned.py`
- Output dir: `outputs/math500_experiments/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1`
- Pipeline log: `outputs/math500_experiments/qwen4b_math500_bpr_grpo_learned_analyzer_pipeline_n8_steps1500/logs/pipeline.nohup.log`
- Final GRPO log: `outputs/math500_experiments/qwen4b_math500_bpr_grpo_learned_analyzer_pipeline_n8_steps1500/logs/steps/06_train_bpr_grpo_learned_analyzer_vllm.log`
- Reward debug: `outputs/math500_experiments/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`

Train config:

```text
model_name: Qwen/Qwen3-4B
dataset_name: SynthLabsAI/Big-Math-RL-Verified
train_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
train_size: 12000
eval_size: 0
method / prior_mode / reward_type: learned_unified_analyzer / posterior_normalized_bayesian_evidence
analyzer_adapter_path: outputs/math500_experiments/qwen4b_math500_bpr_grpo_learned_analyzer_pipeline_n8_steps1500/analyzer_sft_dpo_adapter
num_generations: 8
max_steps: 1500
max_prompt_length: 2048
max_completion_length: 1024
judge_max_new_tokens: 768
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
effective_batch_size: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 100
progress_interval_percent: 10
use_lora: true
bf16: true
gradient_checkpointing: true
use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.25
vllm_max_model_length: 3072
seed: 42
```

Train completion:

- Finished at: 2026-05-24 03:15 KST
- Final GRPO runtime: 17:00:28
- Final checkpoint: `checkpoint-1500`
- Checkpoints: `checkpoint-100`, `checkpoint-200`, `checkpoint-300`, `checkpoint-400`, `checkpoint-500`, `checkpoint-600`, `checkpoint-700`, `checkpoint-800`, `checkpoint-900`, `checkpoint-1000`, `checkpoint-1100`, `checkpoint-1200`, `checkpoint-1300`, `checkpoint-1400`, `checkpoint-1500`

Eval:

- Status: not started

### 2026-05-24 - Qwen3-8B BPR-GRPO Prompted Analyzer / BigMath BARL-style full

- Status: running
- Script: `Bayesian_Full_GRPO.py`
- Output dir: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1`
- Log: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/logs/train.nohup.log`
- Reward debug: `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/bayesian_reward_debug.jsonl`

Train config:

```text
model_name: Qwen/Qwen3-8B
dataset_name: SynthLabsAI/Big-Math-RL-Verified
train_metadata_path: outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
eval_metadata_path: outputs/eval_benchmarks/olympiadbench_metadata.jsonl
train_size: 12288
eval_size: 0
method / prior_mode / reward_type: llm_strategy_prior / posterior_normalized_bayesian_evidence
num_generations: 8
max_steps: 1536
max_prompt_length: 2048
max_completion_length: 1024
judge_max_new_tokens: 768
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
effective_batch_size: 8
learning_rate: 5e-6
logging_steps: 10
save_steps: 256
progress_interval_percent: 10
use_lora: true
bf16: true
gradient_checkpointing: true
use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.20
vllm_max_model_length: 3072
seed: 42
```

Train progress:

- Started at: 2026-05-24 02:55 KST
- Latest observed: `1462 / 1536`
- Status at latest check: running on GPU 1

Eval:

- Status: not started
