# Repository Guidelines

## Project Structure & Module Organization

This repository is a research codebase for Bayesian-reward GRPO on math reasoning. Most source files live at the repository root as standalone Python entry points such as `Answer_only_GRPO.py`, `Bayesian_Full_GRPO.py`, `Bayesian_Full_GRPO_learned.py`, and `eval_solver_checkpoint.py`. Dataset builders and wrappers follow the same pattern: `prepare_*.py`, `run_*.sh`, `train_*.py`, `collect_*.py`.

Use `scripts/` for auxiliary launchers, `data/` for lightweight inputs, `notes/` and `literature/` for research context, and `outputs/` for local experiment artifacts. Do not treat `outputs/`, local virtual environments, or caches as source code.

## Build, Test, and Development Commands

- `python3 prepare_gsm8k_metadata.py --setting full_train --output_dir outputs/gsm8k_full_train_seed42`
  Builds fixed GSM8K metadata.
- `bash run_gsm8k_full_qwen3_1p7b_all_baselines.sh --dry_run`
  Resolves the full Qwen3 pipeline without launching training.
- `python3 -m py_compile run_grpo_bayesian_with_learned_analyzer.py`
  Fast syntax check for Python scripts.
- `bash -n run_gsm8k_grpo_answer_only.sh`
  Shell syntax check before long runs.
- `python3 collect_gsm8k_full_qwen3_1p7b_results.py --dry_run`
  Verifies result collection paths and schema.

## Coding Style & Naming Conventions

Use 4-space indentation and keep scripts ASCII unless the file already contains non-ASCII notes. Prefer `snake_case` for functions, variables, and new file names. Keep new experiment outputs isolated under a new `outputs/...` root; never overwrite prior baselines. Shell scripts should use `set -euo pipefail`.

## Testing Guidelines

There is no formal `pytest` suite yet. Minimum validation for code changes is:

- `python3 -m py_compile` on edited Python files
- `bash -n` on edited shell scripts
- `--dry_run` on new pipeline wrappers
- A small metadata or eval smoke run when the change affects execution flow

Name new verification helpers descriptively, for example `collect_*_results.py` or `analyze_*_debug_jsonl.py`.

## Commit & Pull Request Guidelines

Follow the existing commit style: short, imperative subjects such as `Refresh GitHub README and add repository structure guide`. Keep one logical change per commit. In pull requests, include the purpose, touched scripts, exact run commands, model/dataset/seed assumptions, and whether outputs are new or reused. Do not include checkpoints, caches, or large `outputs/` artifacts in a PR.

## Security & Configuration Tips

Do not hardcode API keys or tokens. Use shell exports or a local `.env`. Before running expensive jobs, verify `CUDA_VISIBLE_DEVICES`, metadata paths, and output roots so new experiments do not mix with old results.
