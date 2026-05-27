#!/usr/bin/env bash
set -euo pipefail

# Submit the missing Table 1 adapter-training jobs with vLLM enabled.
#
# Config source:
# - EMNLP Table 1 / Appendix A.1: matched benchmark-family comparisons.
# - APPENDIX_DRAFT_FROM_ARTIFACTS.md training manifest.
# - EXPERIMENT_RUN_STACK.md and transferred training_config.json files.
#
# This script intentionally submits only the 13 unique adapters that were missing
# from the transferred EMNLP artifacts for the main Table 1 reproduction.

ROOT="/home/kimjh/BPR"
SUBDISK_ROOT="/mnt/raid5/kimjh/BPR"
SUBDISK_OUTPUTS="${SUBDISK_ROOT}/outputs"
JOB_ROOT="${SUBDISK_ROOT}/slurm_main_table_missing_vllm"
JOB_DIR="${JOB_ROOT}/jobs"
LOG_DIR="${JOB_ROOT}/logs"
MANIFEST="${JOB_ROOT}/manifest.tsv"

PARTITION="${PARTITION:-gpu}"
NODELIST="${NODELIST:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-120G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
PYTHON_BIN="${PYTHON_BIN:-${SUBDISK_ROOT}/.venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
NO_PREFLIGHT="${NO_PREFLIGHT:-0}"
SUBMIT="${SUBMIT:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      SUBMIT=0
      shift
      ;;
    --no-preflight)
      NO_PREFLIGHT=1
      shift
      ;;
    --no-submit)
      SUBMIT=0
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT"
mkdir -p "$JOB_DIR" "$LOG_DIR" "$SUBDISK_OUTPUTS"

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
}

ensure_metadata_links() {
  mkdir -p outputs

  if [[ ! -e outputs/gsm8k_full_train_seed42 ]]; then
    ln -s "${SUBDISK_ROOT}/incoming_from_host4_1/EMNLP/outputs/gsm8k_full_train_seed42" \
      outputs/gsm8k_full_train_seed42
  fi

  require_file outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl
  require_file outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl
  require_file outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl
  require_file outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl
  require_file outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl
  require_file outputs/eval_benchmarks/olympiadbench_metadata.jsonl
}

ensure_output_link() {
  local rel="$1"
  local repo_path="${ROOT}/${rel}"
  local disk_path="${SUBDISK_OUTPUTS}/${rel#outputs/}"

  mkdir -p "$(dirname "$repo_path")" "$(dirname "$disk_path")"

  if [[ -L "$repo_path" ]]; then
    return 0
  fi

  if [[ -e "$repo_path" ]]; then
    if find "$repo_path" -name adapter_model.safetensors -print -quit | grep -q .; then
      echo "[SKIP-LINK] Existing adapter output remains in repo: $rel"
      return 0
    fi
    if [[ -z "$(find "$repo_path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "$repo_path"
    else
      echo "[ERROR] Refusing to replace non-empty non-symlink path: $repo_path" >&2
      echo "        Move it manually or choose a clean output path." >&2
      exit 1
    fi
  fi

  mkdir -p "$disk_path"
  ln -s "$disk_path" "$repo_path"
}

preflight_env() {
  if [[ "$NO_PREFLIGHT" == "1" ]]; then
    echo "[WARN] Skipping local Python/vLLM preflight."
    return 0
  fi

  if [[ ! -x "$PYTHON_BIN" ]]; then
    local venv_dir="${PYTHON_BIN%/bin/python}"
    echo "[ERROR] Python env not found: $PYTHON_BIN" >&2
    echo "        Expected a vLLM-ready env on sub-disk. Example:" >&2
    echo "        uv venv $venv_dir --python 3.12" >&2
    echo "        UV_PROJECT_ENVIRONMENT=$venv_dir UV_CACHE_DIR=${SUBDISK_ROOT}/cache/uv uv sync --locked" >&2
    exit 1
  fi

  "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [m for m in ("torch", "transformers", "trl", "peft", "vllm", "datasets") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing))
print("env_ok")
PY
}

write_job() {
  local name="$1"
  local script="$2"
  local model="$3"
  local train_path="$4"
  local eval_path="$5"
  local train_size="$6"
  local eval_size="$7"
  local output_rel="$8"
  local max_steps="$9"
  local per_device="${10}"
  local grad_acc="${11}"
  local max_prompt="${12}"
  local max_completion="${13}"
  local save_steps="${14}"
  local logging_steps="${15}"
  local min_solve="${16}"
  local max_solve="${17}"
  local port="${18}"
  local extra_args="${19}"
  local reward_debug_arg=""
  local nodelist_directive=""

  local sbatch_path="${JOB_DIR}/${name}.sbatch"
  ensure_output_link "$output_rel"
  if [[ "$script" == "Bayesian_Full_GRPO.py" ]]; then
    reward_debug_arg="--reward_debug_jsonl ${output_rel}/bayesian_reward_debug.jsonl"
  fi
  if [[ -n "$NODELIST" ]]; then
    nodelist_directive="#SBATCH --nodelist=${NODELIST}"
  fi

  cat > "$sbatch_path" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${name}
#SBATCH --partition=${PARTITION}
${nodelist_directive}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err

set -euo pipefail
cd ${ROOT}

export HF_HOME=${SUBDISK_ROOT}/cache/huggingface
export HF_DATASETS_CACHE=${SUBDISK_ROOT}/cache/huggingface/datasets
export TRANSFORMERS_CACHE=${SUBDISK_ROOT}/cache/huggingface/transformers
export TRITON_CACHE_DIR=${SUBDISK_ROOT}/cache/triton
export XDG_CACHE_HOME=${SUBDISK_ROOT}/cache/xdg
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RANK=\${RANK:-\${SLURM_PROCID:-0}}
export LOCAL_RANK=\${LOCAL_RANK:-\${SLURM_LOCALID:-0}}
export WORLD_SIZE=\${WORLD_SIZE:-\${SLURM_NTASKS:-1}}
export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=\${MASTER_PORT:-${port}}
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$TRANSFORMERS_CACHE" "\$TRITON_CACHE_DIR" "\$XDG_CACHE_HOME"

PYTHON_BIN="${PYTHON_BIN}"
if [[ ! -x "\$PYTHON_BIN" ]]; then
  echo "[ERROR] Missing Python env: \$PYTHON_BIN" >&2
  exit 2
fi

"\$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [m for m in ("torch", "transformers", "trl", "peft", "vllm", "datasets") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing))
PY

mkdir -p "${output_rel}/logs"
echo "[START] \$(date '+%Y-%m-%d %H:%M:%S') ${name}"
echo "[CONFIG] script=${script} model=${model} train_size=${train_size} eval_size=${eval_size} max_steps=${max_steps} n=8 bsz=${per_device} acc=${grad_acc} prompt=${max_prompt} completion=${max_completion}"

"\$PYTHON_BIN" ${script} \\
  --model_name "${model}" \\
  --dataset_name fixed_metadata \\
  --use_fixed_metadata \\
  --train_metadata_path "${train_path}" \\
  --eval_metadata_path "${eval_path}" \\
  --train_size "${train_size}" \\
  --eval_size "${eval_size}" \\
  --num_generations 8 \\
  --max_steps "${max_steps}" \\
  --per_device_train_batch_size "${per_device}" \\
  --gradient_accumulation_steps "${grad_acc}" \\
  --learning_rate 5e-6 \\
  --max_prompt_length "${max_prompt}" \\
  --max_completion_length "${max_completion}" \\
  --temperature 0.7 \\
  --top_p 0.95 \\
  --logging_steps "${logging_steps}" \\
  --save_steps "${save_steps}" \\
  --progress_interval_percent 10 \\
  --min_solve_rate "${min_solve}" \\
  --max_solve_rate "${max_solve}" \\
  --use_lora \\
  --lora_r 16 \\
  --lora_alpha 32 \\
  --lora_dropout 0.05 \\
  --bf16 \\
  --gradient_checkpointing \\
  --use_vllm \\
  --vllm_mode colocate \\
  --vllm_model_impl vllm \\
  --vllm_gpu_memory_utilization 0.30 \\
  --vllm_tensor_parallel_size 1 \\
  --vllm_max_model_length "$((max_prompt + max_completion))" \\
  --vllm_group_port "${port}" \\
  --output_dir "${output_rel}" \\
  ${reward_debug_arg} \\
  ${extra_args} \\
  > "${output_rel}/logs/train.slurm.log" 2>&1

echo "[DONE] \$(date '+%Y-%m-%d %H:%M:%S') ${name}"
EOF

  chmod +x "$sbatch_path"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$script" "$model" "$train_size" "$eval_size" "$max_steps" \
    "$per_device" "$grad_acc" "$max_prompt" "$max_completion" "$output_rel" "$sbatch_path" \
    >> "$MANIFEST"
}

submit_jobs() {
  if [[ "$SUBMIT" != "1" ]]; then
    echo "[INFO] Submission disabled. Job files are under $JOB_DIR"
    return 0
  fi

  local job
  for job in "$JOB_DIR"/*.sbatch; do
    sbatch "$job"
  done
}

ensure_metadata_links
preflight_env

: > "$MANIFEST"
printf 'name\tscript\tmodel\ttrain_size\teval_size\tmax_steps\tper_device\tgrad_acc\tmax_prompt\tmax_completion\toutput_rel\tsbatch_path\n' > "$MANIFEST"

GSM_TRAIN="outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl"
GSM_VALID="outputs/gsm8k_full_train_seed42/selected_valid_metadata.jsonl"
MATH_TRAIN="outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl"
MATH_TEST="outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl"
BIG_TRAIN="outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl"
BIG_EVAL="outputs/eval_benchmarks/olympiadbench_metadata.jsonl"

BPR_COMMON='--prior_mode llm_strategy_prior --prior_lambda 1.0 --prior_softmax_temperature 1.0 --prior_judge_temperature 0.0 --evidence_judge_temperature 0.0 --judge_max_new_tokens 768'
DRGRPO_COMMON='--loss_type dr_grpo --scale_rewards none --beta 0.0'

# Qwen3-1.7B missing Table 1 adapters.
write_job mt_q17_grpo_math500 Answer_only_GRPO.py Qwen/Qwen3-1.7B "$MATH_TRAIN" "$MATH_TEST" 12000 500 outputs/math500_experiments/grpo_answer_only_qwen3_1p7b_fulltrain12k_n8_steps1500 1500 1 8 2048 1024 100 10 0.0 1.0 52001 ''
write_job mt_q17_grpo_bigmath Answer_only_GRPO.py Qwen/Qwen3-1.7B "$BIG_TRAIN" "$BIG_EVAL" 12288 0 outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500 1500 1 8 2048 1024 100 10 0.0 1.0 52002 ''
write_job mt_q17_drgrpo_math500 Answer_only_GRPO.py Qwen/Qwen3-1.7B "$MATH_TRAIN" "$MATH_TEST" 12000 0 outputs/drgrpo_answer_only_24h/qwen3_1p7b_math500_steps1500_n8_bsz8_acc1 1500 8 1 2048 1024 100 10 0.0 1.0 52003 "$DRGRPO_COMMON"
write_job mt_q17_drgrpo_bigmath Answer_only_GRPO.py Qwen/Qwen3-1.7B "$BIG_TRAIN" "$BIG_EVAL" 12288 0 outputs/drgrpo_answer_only_24h/qwen3_1p7b_bigmath_steps1500_n8_bsz8_acc1 1500 8 1 2048 1024 100 10 0.0 1.0 52004 "$DRGRPO_COMMON"
write_job mt_q17_bpr_math500 Bayesian_Full_GRPO.py Qwen/Qwen3-1.7B "$MATH_TRAIN" "$MATH_TEST" 12000 0 outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1 1500 8 1 1024 1024 250 10 0.0 1.0 52005 "$BPR_COMMON --prior_judge_model Qwen/Qwen3-1.7B --evidence_judge_model Qwen/Qwen3-1.7B"
write_job mt_q17_bpr_bigmath Bayesian_Full_GRPO.py Qwen/Qwen3-1.7B "$BIG_TRAIN" "$BIG_EVAL" 12288 0 outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1 1500 8 1 2048 1024 100 10 0.0 1.0 52006 "$BPR_COMMON --prior_judge_model Qwen/Qwen3-1.7B --evidence_judge_model Qwen/Qwen3-1.7B"

# Qwen3-4B missing Table 1 adapters.
write_job mt_q4_grpo_gsm8k Answer_only_GRPO.py Qwen/Qwen3-4B "$GSM_TRAIN" "$GSM_VALID" 7473 0 outputs/gsm8k_full_qwen3_4b/grpo_answer_only_qwen4b_fulltrain_n8_steps1500_bsz8_acc1 1500 8 1 1024 1024 100 10 0.0 1.0 52007 ''
write_job mt_q4_grpo_math500 Answer_only_GRPO.py Qwen/Qwen3-4B "$MATH_TRAIN" "$MATH_TEST" 12000 500 outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500 1500 1 8 2048 1024 100 10 0.0 1.0 52008 ''
write_job mt_q4_grpo_bigmath Answer_only_GRPO.py Qwen/Qwen3-4B "$BIG_TRAIN" "$BIG_EVAL" 12288 0 outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536 1536 1 8 2048 1024 128 10 0.0 1.0 52009 ''
write_job mt_q4_drgrpo_math500 Answer_only_GRPO.py Qwen/Qwen3-4B "$MATH_TRAIN" "$MATH_TEST" 12000 0 outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1 1500 8 1 2048 1024 100 10 0.0 1.0 52010 "$DRGRPO_COMMON"
write_job mt_q4_drgrpo_bigmath Answer_only_GRPO.py Qwen/Qwen3-4B "$BIG_TRAIN" "$BIG_EVAL" 12288 0 outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1 1500 8 1 2048 1024 100 10 0.0 1.0 52011 "$DRGRPO_COMMON"
write_job mt_q4_bpr_gsm8k Bayesian_Full_GRPO.py Qwen/Qwen3-4B "$GSM_TRAIN" "$GSM_VALID" 7473 0 outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1 1500 8 1 1024 1024 100 10 0.0 1.0 52012 "$BPR_COMMON --prior_judge_model Qwen/Qwen3-4B --evidence_judge_model Qwen/Qwen3-4B"
write_job mt_q4_bpr_math500 Bayesian_Full_GRPO.py Qwen/Qwen3-4B "$MATH_TRAIN" "$MATH_TEST" 12000 0 outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0 1500 8 1 2048 1024 100 10 0.2 0.8 52013 "$BPR_COMMON --prior_judge_model Qwen/Qwen3-4B --evidence_judge_model Qwen/Qwen3-4B"

echo "[INFO] Wrote manifest: $MANIFEST"
column -t -s $'\t' "$MANIFEST" || cat "$MANIFEST"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Not submitting."
  exit 0
fi

submit_jobs
