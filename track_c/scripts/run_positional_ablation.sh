#!/usr/bin/env bash
# Extra experiment #2: positional encoding ablation.
# Trains two otherwise-identical models (with/without positional embeddings)
# and evaluates both, so results/pos_ablation_with_pos and
# results/pos_ablation_no_pos can be diffed directly.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m track_c.src.train --run-name pos_ablation_with_pos
python -m track_c.src.train --run-name pos_ablation_no_pos --override use_positional_embedding=false

python -m track_c.src.evaluate --run-name pos_ablation_with_pos
python -m track_c.src.evaluate --run-name pos_ablation_no_pos
