#!/bin/bash
# One-shot setup for a clean install.
# Installs Python deps, sets up Showdown server (deps + custom formats),
# starts 8 servers, and waits for them to be ready.
#
# Usage:  ./scripts/setup_training.sh
# Then:   uv run train_battler.py --preset standard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHOWDOWN_DIR="$PROJECT_DIR/pokemon-showdown"

echo "============================================================"
echo " Pokemon RL — Training Setup"
echo "============================================================"

# ---------- 1. Python dependencies ----------
echo ""
echo "[1/4] Installing Python dependencies (uv sync)..."
cd "$PROJECT_DIR"
uv sync

# ---------- 2. Pokemon Showdown ----------
echo ""
echo "[2/4] Setting up Pokemon Showdown server..."

if [ ! -d "$SHOWDOWN_DIR" ]; then
    echo "  Error: $SHOWDOWN_DIR not found."
    echo "  Clone it first:  git clone https://github.com/smogon/pokemon-showdown.git"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "  Error: 'node' not found. Install Node.js first."
    exit 1
fi

# Install Showdown npm dependencies
if [ ! -d "$SHOWDOWN_DIR/node_modules" ]; then
    echo "  Installing Showdown npm dependencies..."
    (cd "$SHOWDOWN_DIR" && npm install --production)
else
    echo "  Showdown node_modules already present, skipping npm install."
fi

# ---------- 3. Custom formats ----------
echo ""
echo "[3/4] Registering custom battle formats..."
"$SCRIPT_DIR/setup_custom_formats.sh"

# ---------- 4. Start servers ----------
echo ""
echo "[4/4] Starting Pokemon Showdown servers..."
"$SCRIPT_DIR/kill_all_showdown.sh" 2>/dev/null || true
"$SCRIPT_DIR/spin_up_multiple_showdown.sh"

# Wait for first server to respond
echo ""
echo -n "Waiting for servers to be ready"
for i in $(seq 1 20); do
    if (echo > /dev/tcp/localhost/8000) >/dev/null 2>&1; then
        echo " ok!"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "============================================================"
echo " Setup complete."
echo ""
echo " Available formats:"
echo "   gen8randombattlenogimmicks   (no Dynamax, no Sleep Clause)"
echo "   gen9randombattlenogimmicks   (no Terastallize, no Sleep Clause)"
echo "   gen8customgamenogimmicks     (fixed team, no Dynamax)"
echo "   gen9customgamenogimmicks     (fixed team, no Terastallize)"
echo ""
echo " Start training:"
echo "   uv run train_battler.py --preset standard"
echo "============================================================"
