#!/usr/bin/env bash
# Monitor an existing train_battler.py without restarting it.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG="${LOG:-train.log}"
MONITOR_LOG="${MONITOR_LOG:-logs/long_train_monitor.log}"
TOTAL="${TOTAL_STEPS:-5000000}"
GOOD_CKPT_FILE="${GOOD_CKPT_FILE:-logs/last_good_checkpoint.txt}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MONITOR_LOG" >&2; }

train_failed() {
  [[ -f "$LOG" ]] || return 1
  grep -q 'Training failed' "$LOG" && return 0
  grep -q 'probability tensor contains' "$LOG" && return 0
  grep -q 'CUDA error: an illegal memory access' "$LOG" && return 0
  return 1
}

current_steps() {
  grep -oP 'Steps: \K[0-9,]+' "$LOG" 2>/dev/null | tail -1 | tr -d ',' || echo 0
}

latest_step_checkpoint() {
  local best="" best_n=0 d base step
  for d in checkpoints/step_*; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    step="${base#step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step > best_n )) && best_n=$step && best=$d
  done
  echo "$best"
}

pid="${1:-$(pgrep -fn 'python3? train_battler.py' | head -1 || true)}"
[[ -n "$pid" ]] || { echo "No train_battler pid" >&2; exit 1; }
log "Watching pid=$pid"

steps=0 last_steps=-1 last_ckpt_step=-1 stall_since
stall_since=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
  sleep 45
  steps=$(current_steps)
  iter=$(grep 'Iter ' "$LOG" 2>/dev/null | tail -1 || true)
  now=$(date +%s)
  log "steps=$steps | $iter"
  if train_failed; then
    log "FAIL detected in train.log"
    exit 1
  fi
  if [[ "$steps" =~ ^[0-9]+$ && "$steps" -gt "$last_steps" ]]; then
    last_steps=$steps
    stall_since=$now
  elif (( now - stall_since > 600 )); then
    log "Stalled 600s at steps=$last_steps"
    exit 1
  fi
  ckpt=$(latest_step_checkpoint)
  if [[ -n "$ckpt" ]]; then
    ckpt_step=$(basename "$ckpt"); ckpt_step=${ckpt_step#step_}
    if [[ "$ckpt_step" =~ ^[0-9]+$ && "$ckpt_step" -gt "$last_ckpt_step" && "$steps" -gt 80000 ]]; then
      echo "$ckpt" >"$GOOD_CKPT_FILE"
      log "Marked good checkpoint: $ckpt"
      last_ckpt_step=$ckpt_step
    fi
  fi
  if [[ "$steps" =~ ^[0-9]+$ && "$steps" -ge "$TOTAL" ]]; then
    log "Target steps reached"
    exit 0
  fi
done
steps=$(current_steps)
if [[ "$steps" =~ ^[0-9]+$ && "$steps" -ge "$TOTAL" ]]; then
  log "Done steps=$steps"
  exit 0
fi
train_failed && exit 1
log "Process ended early steps=$steps"
exit 1
