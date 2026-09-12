#!/usr/bin/env bash
# Core deliverable: train the self-attention-expert MoSE model, then evaluate
# it and produce the comparison table + expert-utilization plot.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m track_c.src.train --run-name core
python -m track_c.src.evaluate --run-name core
