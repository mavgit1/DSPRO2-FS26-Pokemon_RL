#!/usr/bin/env bash
# Sequential remote experiments with validation after each 4M+ run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
set -a && [ -f .env ] && . ./.env && set +a

TOTAL="${TOTAL_STEPS:-4000000}"
PRESET="${PRESET:-pure_league_play}"
LOG_DIR="${LOG_DIR:-logs/experiments}"
mkdir -p "$LOG_DIR" logs/validation checkpoints

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_DIR/experiment_loop.log"; }

run_train() {
  local tag="$1"
  local log="$LOG_DIR/train_${tag}.log"
  log "=== TRAIN $tag preset=$PRESET steps=$TOTAL ==="
  rm -f logs/last_good_checkpoint.txt
  ray stop --force 2>/dev/null || true
  sleep 2
  rm -rf checkpoints/*
  env PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 \
    uv run --active train_battler.py \
      --preset "$PRESET" \
      --timesteps "$TOTAL" \
      --disable-scheduled-validation \
      >"$log" 2>&1
  local ec=$?
  if grep -qE 'CUDA error|Training failed|probability tensor' "$log"; then
    log "FAIL train $tag (error in log)"
    return 1
  fi
  grep -oP 'Total Steps: \K[0-9,]+' "$log" | tail -1 | tr -d ',' >"$LOG_DIR/steps_${tag}.txt" || true
  log "OK train $tag steps=$(cat "$LOG_DIR/steps_${tag}.txt" 2>/dev/null || echo '?')"
  return $ec
}

validate_final() {
  local tag="$1"
  local out="$LOG_DIR/benchmark_${tag}.json"
  log "=== VALIDATE $tag ==="
  uv run --active python scripts/validate_checkpoint.py \
    --checkpoint checkpoints/final \
    --protocol benchmark \
    --preset pure_league_play \
    --episodes 100 \
    --num-servers 4 \
    --output-json "$out" \
    >"$LOG_DIR/validate_${tag}.log" 2>&1 || return 1
  python3 - <<PY
import json
from pathlib import Path
p=Path("$out")
if not p.exists():
    print("no report"); raise SystemExit(1)
m=json.loads(p.read_text())["metrics"]
for k in ["benchmark/win_rate_vs_heuristic","benchmark/win_rate_vs_random","benchmark/win_rate_vs_self"]:
    print(f"  {k}: {m.get(k)}")
heur=m.get("benchmark/win_rate_vs_heuristic",0)
Path("$LOG_DIR/heuristic_${tag}.txt").write_text(str(heur))
PY
}

TAG="embed5_curriculum_$(date +%Y%m%d_%H%M)"
if run_train "$TAG"; then
  validate_final "$TAG" || true
  log "Experiment complete tag=$TAG"
else
  log "Experiment failed tag=$TAG"
  exit 1
fi
