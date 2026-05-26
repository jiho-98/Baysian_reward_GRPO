#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
BACKBONE_TAG="${BACKBONE_TAG:-}"
OUT_ROOT="${OUT_ROOT:-}"
EVAL_METADATA_DIR="${EVAL_METADATA_DIR:-outputs/eval_benchmarks}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS:-1024}"
AIME26_MAX_NEW_TOKENS="${AIME26_MAX_NEW_TOKENS:-4096}"
BF16="${BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --backbone_tag)
      BACKBONE_TAG="$2"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --cuda_visible_devices)
      CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's#[^a-z0-9]\+#_#g; s#^_##; s#_$##'
}

if [[ -z "${BACKBONE_TAG}" ]]; then
  BACKBONE_TAG="$(slugify "${MODEL_NAME}")"
fi
if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="outputs/bigmath_${BACKBONE_TAG}_base_prompted_eval_benchmarks"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p "${OUT_ROOT}/logs"

run_eval() {
  local bench="$1"
  local max_new_tokens="$2"
  local metadata_path="${EVAL_METADATA_DIR}/${bench}_metadata.jsonl"
  local output_dir="${OUT_ROOT}/${bench}"
  local log_path="${OUT_ROOT}/logs/${bench}.log"

  if [[ ! -f "${metadata_path}" ]]; then
    echo "[ERROR] Missing eval metadata: ${metadata_path}" >&2
    exit 1
  fi

  local cmd=(
    python3 eval_solver_checkpoint.py
    --model_name "${MODEL_NAME}"
    --no_load_adapter
    --eval_metadata_path "${metadata_path}"
    --output_dir "${output_dir}"
    --batch_size "${BATCH_SIZE}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_new_tokens "${max_new_tokens}"
    --no_do_sample
  )
  if [[ "${BF16}" == "1" ]]; then
    cmd+=(--bf16)
  else
    cmd+=(--no_bf16)
  fi

  printf '[RUN] bench=%s max_new_tokens=%s output=%s\n' "${bench}" "${max_new_tokens}" "${output_dir}"
  printf '[CMD] CUDA_VISIBLE_DEVICES=%q' "${CUDA_VISIBLE_DEVICES}"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}" 2>&1 | tee "${log_path}"
}

# Protocol:
# - AIME26 uses a larger completion budget because 1024 truncated every 4B output.
# - Other BigMath eval benchmarks stay at 1024 for continuity with prior runs.
run_eval "aime26" "${AIME26_MAX_NEW_TOKENS}"
run_eval "minervamath" "${DEFAULT_MAX_NEW_TOKENS}"
run_eval "olympiadbench" "${DEFAULT_MAX_NEW_TOKENS}"

echo "[DONE] BigMath base prompted eval saved under ${OUT_ROOT}"
