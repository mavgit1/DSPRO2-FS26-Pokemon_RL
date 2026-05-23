#!/usr/bin/env bash
# Long run with auto-restart from last *verified* checkpoint (not crash-adjacent).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PRESET="${PRESET:-pure_league_pool}"
TOTAL="${TOTAL_STEPS:-5000000}"
LOG="${LOG:-train.log}"
MONITOR_LOG="${MONITOR_LOG:-logs/long_train_monitor.log}"
GOOD_CKPT_FILE="${GOOD_CKPT_FILE:-logs/last_good_checkpoint.txt}"
MLFLOW_RUN_ID="${MLFLOW_RUN_ID:-}"

mkdir -p logs checkpoints

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MONITOR_LOG"; }

purge() {
  pkill -f "train_battler.py" 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  sleep 3
  uv run --active python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
}

mark_good_checkpoint() {
  local ckpt="$1"
  [[ -n "$ckpt" && -d "$ckpt" ]] || return 0
  echo "$ckpt" >"$GOOD_CKPT_FILE"
  log "Marked good checkpoint: $ckpt"
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

run_once() {
  local extra=("$@")
  purge
  if ! (echo >/dev/tcp/localhost/8000) 2>/dev/null; then
    ./scripts/spin_up_multiple_showdown.sh 6
  fi
  : >"$LOG"
  log "Launch: preset=$PRESET total=$TOTAL extra=${extra[*]}"
  env PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 \
    uv run --active train_battler.py \
      --preset "$PRESET" \
      --timesteps "$TOTAL" \
      --disable-scheduled-validation \
      "${extra[@]}" >>"$LOG" 2>&1 &
  echo $!
}

monitor_pid() {
  local pid=$1
  local last_steps=-1 last_ckpt_step=-1 stall_since
  stall_since=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 45
    local steps iter now ckpt ckpt_step
    steps=$(current_steps)
    iter=$(grep 'Iter ' "$LOG" 2>/dev/null | tail -1 || true)
    now=$(date +%s)
    log "steps=$steps | $iter"

    if train_failed; then
      log "Detected failure in train.log"
      kill "$pid" 2>/dev/null || true
      return 1
    fi

    if [[ -n "$steps" && "$steps" =~ ^[0-9]+$ && "$steps" -gt "$last_steps" ]]; then
      last_steps=$steps
      stall_since=$now
    elif (( now - stall_since > 600 )); then
      log "Stalled 600s at steps=$last_steps"
      kill "$pid" 2>/dev/null || true
      return 1
    fi

    ckpt=$(latest_step_checkpoint)
    if [[ -n "$ckpt" ]]; then
      ckpt_step=$(basename "$ckpt"); ckpt_step=${ckpt_step#step_}
      if [[ "$ckpt_step" =~ ^[0-9]+$ && "$ckpt_step" -gt "$last_ckpt_step" && "$steps" -gt 80000 ]]; then
        mark_good_checkpoint "$ckpt"
        last_ckpt_step=$ckpt_step
      fi
    fi

    if [[ -n "$steps" && "$steps" -ge "$TOTAL" ]]; then
      wait "$pid" || true
      return 0
    fi
  done
  wait "$pid" || true
  [[ -n "$steps" && "$steps" -ge "$TOTAL" ]] && return 0
  train_failed && return 1
  return 1
}

attempt=0
while (( attempt < 15 )); do
  attempt=$((attempt + 1))
  log "=== long run attempt $attempt ==="
  extra=()
  if [[ -f "$GOOD_CKPT_FILE" ]]; then
    ckpt=$(cat "$GOOD_CKPT_FILE")
    if [[ -d "$ckpt" ]]; then
      extra+=(--resume-checkpoint "$ckpt")
      [[ -n "$MLFLOW_RUN_ID" ]] && extra+=(--mlflow-run-id "$MLFLOW_RUN_ID")
      log "Resume from good ckpt: $ckpt"
    fi
  fi
  if ((${#extra[@]} == 0)); then
    rm -rf checkpoints/* 2>/dev/null || true
    log "Cold start"
  fi

  pid=$(run_once "${extra[@]}")
  if monitor_pid "$pid"; then
    log "SUCCESS total steps reached"
    exit 0
  fi
  purge
  # Drop checkpoints newer than last good on failure (crash-adjacent).
  if [[ -f "$GOOD_CKPT_FILE" ]]; then
    good=$(cat "$GOOD_CKPT_FILE")
    good_step=$(basename "$good"); good_step=${good_step#step_}
    for d in checkpoints/step_*; do
      [[ -d "$d" ]] || continue
      s=$(basename "$d"); s=${s#step_}
      if [[ "$s" =~ ^[0-9]+$ && "$good_step" =~ ^[0-9]+$ && "$s" -gt "$good_step" ]]; then
        rm -rf "$d"
        log "Removed crash-adjacent checkpoint $d"
      fi
    done
  else
  rm -rf checkpoints/* 2>/dev/null || true
  fi
done
log "FAILED after $attempt attempts"
exit 1
