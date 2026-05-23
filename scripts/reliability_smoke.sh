#!/usr/bin/env bash
# Quick training smoke test. Usage: ./scripts/reliability_smoke.sh [preset] [timesteps]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PRESET="${1:-pure_league_pool}"
STEPS="${2:-200000}"
LOG="logs/smoke_${PRESET}_${STEPS}_$(date +%Y%m%dT%H%M%S).log"
mkdir -p logs

ray stop --force 2>/dev/null || true
pkill -f "train_battler.py" 2>/dev/null || true
sleep 2

if ! (echo >/dev/tcp/localhost/8000) 2>/dev/null; then
  ./scripts/spin_up_multiple_showdown.sh 6
fi

echo "Smoke: preset=$PRESET steps=$STEPS log=$LOG"
set +e
env PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 \
  uv run --active train_battler.py \
    --preset "$PRESET" \
    --timesteps "$STEPS" \
    --disable-scheduled-validation \
  2>&1 | tee "$LOG"
ec=$?
set -e

echo "--- summary ---"
grep -E 'Iter |Training failed|illegal|probability tensor|Total Steps' "$LOG" | tail -10 || true
if grep -q 'Training failed' "$LOG"; then
  echo "RESULT: FAIL (training failed)"
  exit 1
fi
if grep -q 'probability tensor contains' "$LOG"; then
  echo "RESULT: FAIL (nan probs)"
  exit 1
fi
if grep -q 'CUDA error: an illegal memory access' "$LOG"; then
  echo "RESULT: FAIL (cuda illegal access)"
  exit 1
fi
steps=$(grep -oP 'Steps: \K[0-9,]+' "$LOG" | tail -1 | tr -d ',' || echo 0)
if [[ "${steps:-0}" -ge $(( STEPS * 80 / 100 )) ]]; then
  echo "RESULT: PASS steps=$steps"
  exit 0
fi
echo "RESULT: INCOMPLETE steps=$steps exit=$ec"
exit 1
