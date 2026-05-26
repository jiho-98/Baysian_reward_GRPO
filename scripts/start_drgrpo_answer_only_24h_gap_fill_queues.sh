#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/kimjh/Baysian_reward_GRPO"
cd "$ROOT_DIR"

LOG_DIR="outputs/drgrpo_answer_only_24h/logs"
mkdir -p "$LOG_DIR"

nohup bash scripts/run_drgrpo_answer_only_24h_gap_fill_queue.sh \
  0 gpu0_1p7b_bigmath_then_gsm8k \
  > "$LOG_DIR/gpu0_queue.nohup.log" 2>&1 < /dev/null &
echo "$!" > "$LOG_DIR/gpu0_queue.pid"
disown || true

nohup bash scripts/run_drgrpo_answer_only_24h_gap_fill_queue.sh \
  1 gpu1_1p7b_math500_then_4b_gsm8k \
  > "$LOG_DIR/gpu1_queue.nohup.log" 2>&1 < /dev/null &
echo "$!" > "$LOG_DIR/gpu1_queue.pid"
disown || true

echo "GPU0_QUEUE_PID=$(cat "$LOG_DIR/gpu0_queue.pid")"
echo "GPU1_QUEUE_PID=$(cat "$LOG_DIR/gpu1_queue.pid")"
