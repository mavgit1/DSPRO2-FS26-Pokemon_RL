#!/usr/bin/env bash
# Clean 25M run: team pool curriculum 3 -> 5 -> 10 -> 20 @ 65% WR, then league.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

TOTAL="${TOTAL_STEPS:-25000000}"
LOG="${LOG:-logs/experiments/train_pool_curriculum_25M.log}"
MONITOR="${MONITOR:-logs/experiments/pool_curriculum_25M_monitor.log}"

mkdir -p logs/experiments saved_models/pool_curriculum_reference
set -a && [ -f .env ] && . ./.env && set +a

python scripts/build_team_pool_manifest.py

if ! (echo >/dev/tcp/localhost/8000) 2>/dev/null; then
  ./scripts/spin_up_multiple_showdown.sh 8
fi

echo "[$(date -Iseconds)] Clean pool-curriculum 25M preset=pure_league_play" | tee "$MONITOR"

env PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 \
  uv run --active train_battler.py \
    --preset pure_league_play \
    --timesteps "$TOTAL" \
    2>&1 | tee "$LOG"
