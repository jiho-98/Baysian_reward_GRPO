# Repository Structure

This document explains the current repository layout and the recommended
cleanup direction for turning the project into a clearer long-lived research
codebase.

## 1. Current Tracked Layout

The repository currently keeps a relatively flat top-level layout because the
project grew experiment-by-experiment.

### Training scripts

- `Answer_only_GRPO.py`
- `Bayesian_GRPO.py`
- `Bayesian_AH_GRPO.py`
- `Bayesian_Full_GRPO.py`

These are the main GRPO training entry points.

### Evaluation scripts

- `eval_answer_only_grpo.py`
- `eval_bayesian_ah_grpo.py`
- `eval_fair_compare_adapters.py`

These evaluate base and LoRA-adapted models on fixed Big-Math eval splits.

### Bayesian analysis / posterior experiments

- `run_bayesian_trajectory_experiment.py`
- `offline_prior_recompute.py`
- `offline_prior_recompute_fixed.py`
- `offline_prior_recompute_fixed_copy.py`

These are for rollout-level prior / likelihood / posterior analysis outside
the GRPO training loop.

### Early belief and prototype experiments

- `scripts/minimal_belief_demo.py`
- `multi_problem_belief_experiment.py`
- `multi_task_belief_experiment.py`
- `llm_self_data_bayesian_experiment.py`
- `new_qwen.py`

These are earlier validation stages before the current GRPO pipeline.

### Data preparation

- `prepare_fair_bigmath_metadata.py`

This builds the fixed metadata split for fair comparisons.

### Research notes

- `0514_GRPO전까지_최종정리.txt`
- `0514_현재까지_진행상황_총정리.txt`
- `0517_BayesianGRPO_진행상황정리.txt`
- `experiment_notes_0513.txt`
- `idea_summary.txt`
- `review_paper_list.txt`

### Cluster / batch scripts

- `scripts/slurm_qwen_belief_demo.sbatch`
- `scripts/slurm_qwen7b_llm_self_data.sbatch`

## 2. Local-Only Directories

These are intentionally not tracked in GitHub:

- `.venv/`
- `.venv_qwen/`
- `.venv_math500/`
- `outputs/`
- `logs/`
- `results/`
- `hf_cache/`

Reason:

- too large
- machine-specific
- cache-like
- often include checkpoints, optimizer states, or generated artifacts

## 3. Recommended Cleanup Target

The project should gradually move toward the following structure.

```text
Bayesian_reward_GRPO/
  README.md
  REPO_STRUCTURE.md
  src/
    grpo/
    eval/
    analysis/
    data/
    prototypes/
    utils/
  scripts/
    slurm/
    run/
  docs/
    notes/
    summaries/
  configs/
  outputs/            # local only, ignored
```

## 4. Concrete Migration Map

### `src/grpo/`

Move:

- `Answer_only_GRPO.py`
- `Bayesian_GRPO.py`
- `Bayesian_AH_GRPO.py`
- `Bayesian_Full_GRPO.py`

Purpose:

- keep all training entry points together
- make reward-variant comparison easier

### `src/eval/`

Move:

- `eval_answer_only_grpo.py`
- `eval_bayesian_ah_grpo.py`
- `eval_fair_compare_adapters.py`

Purpose:

- separate evaluation from training
- make fixed-split comparison scripts easy to find

### `src/analysis/`

Move:

- `run_bayesian_trajectory_experiment.py`
- `offline_prior_recompute.py`
- `offline_prior_recompute_fixed.py`
- `offline_prior_recompute_fixed_copy.py`

Purpose:

- collect posterior analysis and offline ablations in one place

### `src/data/`

Move:

- `prepare_fair_bigmath_metadata.py`

Purpose:

- isolate split generation and metadata preparation

### `src/prototypes/`

Move:

- `multi_problem_belief_experiment.py`
- `multi_task_belief_experiment.py`
- `llm_self_data_bayesian_experiment.py`
- `new_qwen.py`
- `scripts/minimal_belief_demo.py`

Purpose:

- preserve the historical path of the project
- keep prototype logic separate from the main GRPO pipeline

### `scripts/slurm/`

Move:

- `scripts/slurm_qwen_belief_demo.sbatch`
- `scripts/slurm_qwen7b_llm_self_data.sbatch`

Purpose:

- keep cluster launchers separate from Python code

### `docs/notes/`

Move:

- `0514_GRPO전까지_최종정리.txt`
- `0514_현재까지_진행상황_총정리.txt`
- `0517_BayesianGRPO_진행상황정리.txt`
- `experiment_notes_0513.txt`
- `idea_summary.txt`
- `review_paper_list.txt`

Purpose:

- make the root directory less crowded
- distinguish code from research notes

## 5. Recommended Near-Term Refactor Order

The safest order is:

1. Move notes and batch scripts first.
2. Move evaluation scripts second.
3. Move analysis scripts third.
4. Move training scripts last, because they have the most internal imports and
   command examples.

This minimizes breakage while improving discoverability.

## 6. What Should Stay Flat for Now

For now, it is reasonable to keep the main GRPO entry points at the top level
until:

- imports are centralized
- shared utility functions are factored out
- launch scripts are updated
- documentation is synced

In other words, the repository now has a documented structure plan, but it does
not need a risky large-scale file move immediately.

## 7. Practical Policy Going Forward

Recommended tracking policy:

- track: code, docs, notes, small config files
- do not track: checkpoints, optimizer states, caches, virtualenvs, raw logs

Recommended output policy:

- keep all generated training and eval artifacts under `outputs/`
- keep `outputs/` ignored by Git
- if a result is important, summarize it in a tracked markdown or text file

This keeps the GitHub repository lightweight while preserving the full local
experiment history on the server.
