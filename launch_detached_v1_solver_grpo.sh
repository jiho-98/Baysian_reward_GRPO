#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${ROOT_DIR}/run_learned_analyzer_solver_grpo_v1_lambda05_07.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/learned_analyzer_solver_grpo_v1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LAUNCH_DIR="${ROOT_DIR}/${BASE_OUTPUT_ROOT}/launcher"
mkdir -p "${LAUNCH_DIR}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_DIR}/launch_${TIMESTAMP}.log"
PID_FILE="${LAUNCH_DIR}/launch_${TIMESTAMP}.pid"
META_FILE="${LAUNCH_DIR}/launch_${TIMESTAMP}.meta"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] Missing run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

CMD=(
  env
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
)

if command -v systemd-inhibit >/dev/null 2>&1; then
  CMD+=(
    systemd-inhibit
    --why=Prevent sleep or lid-suspend during solver GRPO training
    --what=idle:sleep:handle-lid-switch
    --mode=block
  )
fi

CMD+=(bash "${RUN_SCRIPT}")

cd "${ROOT_DIR}"
nohup setsid "${CMD[@]}" > "${LAUNCH_LOG}" 2>&1 < /dev/null &
PID="$!"

cat > "${PID_FILE}" <<EOF
${PID}
EOF

cat > "${META_FILE}" <<EOF
pid=${PID}
launch_log=${LAUNCH_LOG}
run_script=${RUN_SCRIPT}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
base_output_root=${BASE_OUTPUT_ROOT}
started_at_utc=${TIMESTAMP}
cwd=${ROOT_DIR}
EOF

echo "[LAUNCHED] pid=${PID}"
echo "[LAUNCHED] launch_log=${LAUNCH_LOG}"
echo "[LAUNCHED] pid_file=${PID_FILE}"
echo "[LAUNCHED] meta_file=${META_FILE}"
echo "[CHECK] ps -fp ${PID}"
echo "[CHECK] tail -f ${LAUNCH_LOG}"
