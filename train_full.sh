#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" scripts/run_full_pipeline.py \
  --config configs/pipeline.toml \
  "$@"
