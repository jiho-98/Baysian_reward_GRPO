# Bayesian Reward GRPO

Research code for Bayesian-style reward design in math reasoning with GRPO.

The project started from strategy-belief prototypes and grew into a set of
controlled experiments on whether richer Bayesian reward signals can improve
over answer-only RL for mathematical reasoning.

## Core Idea

Standard answer-only GRPO gives each rollout a reward based only on final answer
correctness:

- correct answer -> reward `1`
- wrong answer -> reward `0`

This is stable and low-noise, but it ignores reasoning quality.

This repository explores a Bayesian alternative. For each problem, multiple
rollouts are treated as strategy-conditioned hypotheses. Reward is built from:

- a strategy prior `P(H_i | q)`
- an evidence likelihood `P(E_i | H_i, q)`
- posterior normalization across the rollout group

The full version implemented here uses:

`score_i = P(H_i | q)^lambda * P(E_i | H_i, q)`

and then normalizes within the rollout set:

`reward_i = score_i / sum_j score_j`

## What Is Implemented

### GRPO training variants

- [Answer_only_GRPO.py](Answer_only_GRPO.py)
  Answer-only GRPO baseline.
- [Bayesian_GRPO.py](Bayesian_GRPO.py)
  Uniform-prior likelihood-style Bayesian reward.
- [Bayesian_AH_GRPO.py](Bayesian_AH_GRPO.py)
  Answer-heavy likelihood reward with stronger correctness anchoring.
- [Bayesian_Full_GRPO.py](Bayesian_Full_GRPO.py)
  Full posterior-normalized reward with strategy prior and within-group
  normalization.

### Evaluation

- [eval_answer_only_grpo.py](eval_answer_only_grpo.py)
- [eval_bayesian_ah_grpo.py](eval_bayesian_ah_grpo.py)
- [eval_fair_compare_adapters.py](eval_fair_compare_adapters.py)

### Trajectory / posterior analysis

- [run_bayesian_trajectory_experiment.py](run_bayesian_trajectory_experiment.py)
- [offline_prior_recompute.py](offline_prior_recompute.py)
- [offline_prior_recompute_fixed.py](offline_prior_recompute_fixed.py)

### Earlier belief and prototype experiments

- [scripts/minimal_belief_demo.py](scripts/minimal_belief_demo.py)
- [multi_problem_belief_experiment.py](multi_problem_belief_experiment.py)
- [multi_task_belief_experiment.py](multi_task_belief_experiment.py)
- [llm_self_data_bayesian_experiment.py](llm_self_data_bayesian_experiment.py)
- [new_qwen.py](new_qwen.py)

### Data preparation

- [prepare_fair_bigmath_metadata.py](prepare_fair_bigmath_metadata.py)

## Current Headline Result

On the fixed fair Big-Math split (`3000 train / 300 eval`, `n=8`, `500 steps`,
same base model and metadata split), the local eval summary is:

| Model | Accuracy |
| --- | ---: |
| Base Qwen2.5-3B-Instruct | 54.67% |
| Answer-only GRPO | 54.00% |
| Bayesian AH 0.80 | 58.33% |
| Bayesian Full | 60.00% |

Difficulty breakdown on the same fair eval set:

| Model | Easy | Medium |
| --- | ---: | ---: |
| Base | 65.67% | 51.50% |
| Answer-only | 70.15% | 49.36% |
| Bayesian AH 0.80 | 70.15% | 54.94% |
| Bayesian Full | 70.15% | 57.08% |

The main observed gain from Bayesian rewards appears on medium-difficulty
problems rather than easy ones.

## Repository Policy

This GitHub repository currently tracks:

- code
- notes
- lightweight documentation

It does **not** track:

- local virtual environments
- training outputs
- checkpoints
- optimizer states
- large evaluation artifacts
- local caches

Large artifacts remain in local `outputs/` and related folders.

## Setup

Python environments are local and not committed. Install the dependencies you
need for the script you want to run. Most GRPO scripts rely on:

- `torch`
- `transformers`
- `trl`
- `peft`
- `datasets`

Some earlier demos also use:

- `openai`

Use a local `.env` file or shell exports for secrets. Do not hardcode API keys
in source files.

Example:

```bash
export OPENAI_API_KEY=your_key_here
```

## Example Commands

### Minimal belief demo

```bash
python scripts/minimal_belief_demo.py
```

### Fair metadata preparation

```bash
python prepare_fair_bigmath_metadata.py
```

### Full Bayesian GRPO training

```bash
python Bayesian_Full_GRPO.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --prior_mode llm_strategy_prior \
  --prior_judge_model Qwen/Qwen2.5-3B-Instruct \
  --evidence_judge_model Qwen/Qwen2.5-3B-Instruct \
  --use_fixed_metadata \
  --train_metadata_path outputs/fair_bigmath_3000_300_seed42/selected_train_metadata.jsonl \
  --eval_metadata_path outputs/fair_bigmath_3000_300_seed42/selected_eval_metadata.jsonl \
  --train_size 3000 \
  --eval_size 300 \
  --num_generations 8 \
  --max_steps 500 \
  --output_dir outputs/fair_bayesian_full_qwen3b_bigmath_3000_300_n8_steps500
```

### Adapter comparison on the same eval split

```bash
python eval_fair_compare_adapters.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --eval_path outputs/fair_bigmath_3000_300_seed42/selected_eval_metadata.jsonl \
  --adapter "answer_only_n8:outputs/fair_answer_only_qwen3b_bigmath_3000_300_n8_steps500/checkpoint-500" \
  --adapter "bayesian_ah080_n8:outputs/fair_bayesian_ah080_qwen3b_bigmath_3000_300_n8_steps500/checkpoint-500" \
  --adapter "bayesian_full_n8:outputs/fair_bayesian_full_qwen3b_bigmath_3000_300_n8_steps500/checkpoint-500" \
  --output_dir outputs/eval_fair_answer_vs_bayesian_ah080_vs_full_300
```

### Analyzer DPO workflow

Prepare DPO pairs from solver debug traces:

```bash
python prepare_unified_analyzer_dpo.py \
  --input_debug_jsonl outputs/your_solver_run/bayesian_reward_debug.jsonl \
  --expected_prior_lambda 0.7 \
  --output_dir outputs/unified_analyzer_dpo_v0
```

Option A: Base Qwen -> Analyzer DPO

```bash
python train_unified_analyzer_dpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path outputs/unified_analyzer_dpo_v0/unified_dpo_train.jsonl \
  --val_path outputs/unified_analyzer_dpo_v0/unified_dpo_val.jsonl \
  --output_dir outputs/analyzer_dpo_base_qwen
```

Option B: v0 SFT Analyzer -> Analyzer DPO

```bash
python train_unified_analyzer_dpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path outputs/unified_analyzer_dpo_v0/unified_dpo_train.jsonl \
  --val_path outputs/unified_analyzer_dpo_v0/unified_dpo_val.jsonl \
  --init_adapter_path outputs/unified_analyzer_sft_v0/checkpoint-XXX \
  --output_dir outputs/analyzer_dpo_from_sft_v0
```

Cheap filter before solver GRPO:

```bash
python run_unified_analyzer_dpo_filter.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --adapter_path outputs/analyzer_dpo_base_qwen \
  --input_debug_jsonl outputs/your_solver_run/bayesian_reward_debug.jsonl \
  --baseline_summary_json outputs/v0_lambda07_filter/lambda_0p7/summary.json \
  --output_dir outputs/analyzer_dpo_base_qwen_filter
```

## Notes and Summaries

Current internal summaries are tracked as text files:

- [0514_GRPO전까지_최종정리.txt](0514_GRPO전까지_최종정리.txt)
- [0514_현재까지_진행상황_총정리.txt](0514_현재까지_진행상황_총정리.txt)
- [0517_BayesianGRPO_진행상황정리.txt](0517_BayesianGRPO_진행상황정리.txt)

## Repository Layout

See [REPO_STRUCTURE.md](REPO_STRUCTURE.md) for:

- the current repository layout
- what each script family is for
- what should remain local-only
- a concrete cleanup / reorganization plan

## Status

This is an active research repository, not a polished library.

The current evidence supports the claim that posterior-style Bayesian reward is
more informative than answer-only reward in the current fair setting, but the
project still needs:

- multi-seed validation
- larger held-out evaluation
- harder subsets
- analyzer training and co-evolution with the solver
