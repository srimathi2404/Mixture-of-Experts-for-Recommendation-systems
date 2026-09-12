#!/usr/bin/env bash
# Convenience launcher for Track C training.
# Run from anywhere: bash track_c/scripts/run_train.sh [extra args forwarded to train.py]
# e.g. bash track_c/scripts/run_train.sh --run-name core
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m track_c.src.train "$@"
