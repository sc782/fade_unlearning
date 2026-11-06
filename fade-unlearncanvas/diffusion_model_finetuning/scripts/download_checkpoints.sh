#!/bin/bash
# Thin wrapper -- see download_checkpoints.py for the actual logic.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/download_checkpoints.py "$@"
