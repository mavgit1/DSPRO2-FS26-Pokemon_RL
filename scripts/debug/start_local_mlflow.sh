#!/usr/bin/env bash
# Start local MLflow tracking server for debug runs.
#
# Usage:
#   ./scripts/debug/start_local_mlflow.sh
#   ./scripts/debug/start_local_mlflow.sh --port 5001
#
# Data is stored in .mlflow/local_tracking/

set -euo pipefail

PORT="${1:-5000}"
MLFLOW_DIR="$(git rev-parse --show-toplevel)/.mlflow/local_tracking"

mkdir -p "$MLFLOW_DIR"

echo "Starting local MLflow server..."
echo "  URL: http://localhost:${PORT}"
echo "  Backend: file://${MLFLOW_DIR}"
echo ""
echo "Press Ctrl+C to stop."

exec uv run mlflow server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --backend-store-uri "file://${MLFLOW_DIR}" \
    --default-artifact-root "${MLFLOW_DIR}/artifacts"
