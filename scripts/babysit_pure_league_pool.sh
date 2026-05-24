#!/usr/bin/env bash
# Babysit pure_league_pool until TOTAL_STEPS (default 5M). Restarts on failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TOTAL_STEPS="${TOTAL_STEPS:-5000000}"
MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS:-2000000}"
NUM_SERVERS="${NUM_SERVERS:-6}"
LOG="${LOG:-train.log}"
BABYSIT_LOG="${BABYSIT_LOG:-logs/babysit.log}"
MLFLOW_RUN_ID="${MLFLOW_RUN_ID:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
STALL_SECS="${STALL_SECS:-300}"
BAD_CKPTS_FILE="${BAD_CKPTS_FILE:-logs/bad_checkpoints.txt}"

mkdir -p logs checkpoints
touch "$BAD_CKPTS_FILE"

LOCK_FILE="${LOCK_FILE:-/tmp/pokemon_rl_babysit.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another babysit instance is already running (lock: $LOCK_FILE)" >&2
  exit 1
fi

log() { echo "[$(date -Iseconds)] $*" >>"$BABYSIT_LOG"; }

purge_gpu() {
  log "Purging Ray / training / GPU compute processes"
  pkill -f "train_battler.py" 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  sleep 3
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
      pid="${pid// /}"
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      cmd=$(ps -p "$pid" -o comm= 2>/dev/null || true)
      case "$cmd" in
        python*|*ray*) kill -9 "$pid" 2>/dev/null || true ;;
      esac
    done
  fi
  uv run --active python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
}

mark_bad_checkpoint() {
  local ckpt="$1"
  [[ -n "$ckpt" ]] || return 0
  grep -qxF "$ckpt" "$BAD_CKPTS_FILE" 2>/dev/null || echo "$ckpt" >>"$BAD_CKPTS_FILE"
}

is_bad_checkpoint() {
  local ckpt="$1"
  grep -qxF "$ckpt" "$BAD_CKPTS_FILE" 2>/dev/null
}

list_safe_checkpoints_newest_first() {
  local d base step
  for d in checkpoints/step_*; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    step="${base#step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    is_bad_checkpoint "$d" && continue
    echo "$step $d"
  done | sort -rn | awk '{print $2}'
}

ensure_showdown() {
  if (echo >/dev/tcp/localhost/8000) 2>/dev/null; then
    return 0
  fi
  log "Starting $NUM_SERVERS Showdown servers"
  "$REPO_ROOT/scripts/spin_up_multiple_showdown.sh" "$NUM_SERVERS"
}

current_steps() {
  grep -oP 'Steps: \K[0-9,]+' "$LOG" 2>/dev/null | tail -1 | tr -d ',' || echo 0
}

train_has_fatal_errors() {
  [[ -f "$LOG" ]] || return 1
  # Grace period while Ray/env workers spin up (avoid false positives on empty log).
  if ! grep -q "Starting Training" "$LOG" 2>/dev/null; then
    return 1
  fi
  local started_at
  started_at=$(grep -n "Starting Training" "$LOG" | tail -1 | cut -d: -f1)
  local lines_after
  lines_after=$(tail -n +"$started_at" "$LOG" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${lines_after:-0}" -lt 8 ]] && ! grep -q 'Iter ' "$LOG" 2>/dev/null; then
    return 1
  fi
  grep -q 'Training failed' "$LOG" && return 0
  grep -q 'probability tensor contains either `inf`, `nan` or element < 0' "$LOG" && return 0
  grep -q 'CUDA error: an illegal memory access' "$LOG" && return 0
  # Ray worker death spiral: many actor_manager ERRORs and no rollouts completing.
  local err_count no_samples
  err_count=$(grep -c 'ERROR actor_manager.py' "$LOG" 2>/dev/null || true)
  no_samples=$(grep -c 'No samples returned from remote workers' "$LOG" 2>/dev/null || true)
  err_count=${err_count:-0}
  no_samples=${no_samples:-0}
  if (( err_count >= 6 && no_samples >= 2 )); then
    return 0
  fi
  return 1
}

run_train() {
  local extra_args=("$@")
  purge_gpu
  ensure_showdown
  : >"$LOG"
  log "Starting train: ${extra_args[*]}"
  env PYTHONUNBUFFERED=1 DISABLE_DECISION_DIAGNOSTICS=1 SAVE_LEAGUE_HISTORY=1 \
    uv run --active train_battler.py \
      --preset pure_league_play \
      --timesteps "$TOTAL_STEPS" \
      --disable-scheduled-validation \
      "${extra_args[@]}" >>"$LOG" 2>&1 &
  echo $!
}

monitor_until_done() {
  local target=$1
  local last_steps=-1
  local last_progress_at
  last_progress_at=$(date +%s)

  while true; do
    sleep 30

    if train_has_fatal_errors; then
      log "FAIL: fatal errors in train.log ($(tail -3 "$LOG" | tr '\n' ' '))"
      return 1
    fi

    if ! pgrep -f "\.venv/bin/python3 train_battler" >/dev/null 2>&1; then
      if grep -q 'Total Steps:' "$LOG" 2>/dev/null; then
        local total
        total=$(grep -oP 'Total Steps: \K[0-9,]+' "$LOG" | tail -1 | tr -d ',')
        log "Process exited. Total Steps: $total"
        [[ -n "$total" && "$total" -ge "$target" ]] && return 0
      fi
      log "Process exited without reaching target"
      return 1
    fi

    local steps iter now
    steps=$(current_steps)
    iter=$(grep 'Iter ' "$LOG" 2>/dev/null | tail -1 || true)
    now=$(date +%s)

    if [[ -n "$steps" && "$steps" =~ ^[0-9]+$ && "$steps" -gt "$last_steps" ]]; then
      last_steps=$steps
      last_progress_at=$now
    fi

  if (( now - last_progress_at > STALL_SECS )); then
      log "FAIL: stalled ${STALL_SECS}s at steps=$last_steps (likely error loop — check train.log)"
      return 1
    fi

    log "progress steps=${steps:-0} | ${iter:-warming up...}"

    if [[ -n "$steps" && "$steps" -ge "$target" ]]; then
      log "Reached target $target"
      return 0
    fi
  done
}

pick_checkpoint_for_attempt() {
  local attempt=$1
  mapfile -t ckpts < <(list_safe_checkpoints_newest_first)
  if ((${#ckpts[@]} == 0)); then
    echo ""
    return 0
  fi
  # Attempt 1: newest safe. Attempt 2+: walk back to older checkpoints.
  local idx=$((attempt - 1))
  if (( idx >= ${#ckpts[@]} )); then
    echo ""
    return 0
  fi
  echo "${ckpts[$idx]}"
}

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  log "=== Attempt $attempt / $MAX_ATTEMPTS ==="

  extra=()
  safe=$(pick_checkpoint_for_attempt "$attempt")
  if [[ -n "$safe" && -d "$safe" ]]; then
    extra+=(--resume-checkpoint "$safe")
    if [[ -n "$MLFLOW_RUN_ID" ]]; then
      extra+=(--mlflow-run-id "$MLFLOW_RUN_ID")
    fi
    log "Resuming from $safe mlflow=${MLFLOW_RUN_ID:-new}"
  else
    log "Cold start (no checkpoint for attempt $attempt)"
    rm -rf checkpoints/* 2>/dev/null || true
    MLFLOW_RUN_ID=""
  fi

  train_pid=$(run_train "${extra[@]}")

  if monitor_until_done "$TOTAL_STEPS"; then
    log "SUCCESS: reached $TOTAL_STEPS"
    exit 0
  fi

  kill "$train_pid" 2>/dev/null || true
  purge_gpu

  if [[ -n "${safe:-}" ]]; then
    mark_bad_checkpoint "$safe"
    log "Marked bad checkpoint: $safe"
  fi

  if (( attempt >= 3 )) && [[ -n "${safe:-}" ]]; then
    log "Multiple resume failures — next attempt will use older checkpoint or cold start"
  fi
done

log "Gave up after $MAX_ATTEMPTS attempts"
exit 1
