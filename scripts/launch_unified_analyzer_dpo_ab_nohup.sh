#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-scripts/run_unified_analyzer_dpo_ab_pipeline.sh}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/analyzer_dpo_ab_runs/logs}"
mkdir -p "$LOG_DIR"

LOG_PATH="${LOG_DIR}/run_${RUN_TAG}.log"
PID_PATH="${LOG_DIR}/run_${RUN_TAG}.pid"

export RUN_TAG
nohup bash "$PIPELINE_SCRIPT" > "$LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"

echo "PID: $PID"
echo "LOG: $LOG_PATH"
echo "PID_FILE: $PID_PATH"
