#!/usr/bin/env bash
# Extra experiment #4: sequence length / context window sweep.
# Coordinate these exact lengths with Track B's owner in advance so the two
# tracks' plots overlay cleanly (see docs/extra_experiments.md #4).
set -euo pipefail
cd "$(dirname "$0")/../.."

LENGTHS=(5 10 20 50)
for L in "${LENGTHS[@]}"; do
  python -m track_c.src.train --run-name "seqlen${L}" --override "MAX_ITEM_LIST_LENGTH=${L}"
done

RUN_NAMES=("${LENGTHS[@]/#/seqlen}")
python -m track_c.src.analysis.seqlen_sweep --run-names "${RUN_NAMES[@]}"
