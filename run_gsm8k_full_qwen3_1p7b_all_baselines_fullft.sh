#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export SOLVER_USE_LORA="${SOLVER_USE_LORA:-0}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/gsm8k_full_qwen3_1p7b_fullft}"
export STEP_LOG_ROOT="${STEP_LOG_ROOT:-${EXPERIMENT_ROOT}/step_logs}"

bash run_gsm8k_full_qwen3_1p7b_all_baselines.sh "$@"
