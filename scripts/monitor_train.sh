#!/usr/bin/env bash
# Monitor train_long.log; kill training if it fails. Exit 0 when target steps reached.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-$ROOT/logs/train_long.log}"
TARGET_STEPS="${2:-2000000}"
POLL_SEC="${3:-90}"

fail_patterns='CUDA error|illegal memory access|Traceback (most recent call last)|probability tensor contains either|Training failed at step'

while true; do
  if [[ -f "$LOG" ]] && grep -qE "$fail_patterns" "$LOG" 2>/dev/null; then
    echo "[monitor] Failure detected in log — stopping training."
    pkill -f 'train_battler.py --preset pure_league_play' 2>/dev/null || true
    exit 1
  fi

  if ! pgrep -f 'train_battler.py --preset pure_league_play' >/dev/null 2>&1; then
    if [[ -f "$LOG" ]] && grep -q 'Total Steps:' "$LOG" 2>/dev/null; then
      steps=$(grep 'Total Steps:' "$LOG" | tail -1 | grep -oE '[0-9,]+' | head -1 | tr -d ',')
      if [[ -n "${steps:-}" ]] && [[ "$steps" -ge "$TARGET_STEPS" ]]; then
        echo "[monitor] Training finished: $steps steps."
        exit 0
      fi
    fi
    echo "[monitor] Training process exited."
    exit 1
  fi

  if [[ -f "$LOG" ]]; then
    tail -3 "$LOG" | sed 's/^/[monitor] /'
  fi
  sleep "$POLL_SEC"
done
