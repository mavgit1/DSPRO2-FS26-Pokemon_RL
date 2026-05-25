#!/usr/bin/env bash
# Resume pure_league_play toward 25M (pool curriculum + league mix + validation).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

CKPT="${1:-latest}"
MLFLOW_RUN_ID="${MLFLOW_RUN_ID:-}"
TOTAL="${TOTAL_STEPS:-25000000}"
LOG="${LOG:-logs/experiments/train_25M_resume.log}"
MONITOR="${MONITOR:-logs/experiments/resume_25M_monitor.log}"

mkdir -p logs/experiments
set -a && [ -f .env ] && . ./.env && set +a

if ! (echo >/dev/tcp/localhost/8000) 2>/dev/null; then
  ./scripts/spin_up_multiple_showdown.sh 8
fi

extra=()
if [[ -n "$MLFLOW_RUN_ID" ]]; then
  extra+=(--mlflow-run-id "$MLFLOW_RUN_ID")
fi

echo "[$(date -Iseconds)] Resume ckpt=$CKPT total=$TOTAL mlflow=${MLFLOW_RUN_ID:-new} stage=${RESUME_CURRICULUM_STAGE:-}" | tee "$MONITOR"

env PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 \
  RESUME_CURRICULUM_STAGE="${RESUME_CURRICULUM_STAGE:-league_training}" \
  uv run --active train_battler.py \
    --preset pure_league_play \
    --timesteps "$TOTAL" \
    --resume-checkpoint "$CKPT" \
    "${extra[@]}" \
    2>&1 | tee "$LOG"
