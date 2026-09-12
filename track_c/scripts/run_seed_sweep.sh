#!/usr/bin/env bash
# Extra experiment #6: load-balancing-loss cross-check + polarization rate.
#
# Trains N seeds x {load-balancing off, on} and reports what fraction of
# seeds show gate collapse in each condition (see
# src/analysis/polarization.py). Uses a reduced epoch budget - polarization
# tends to show up early in training and this is an add-on, not the core
# deliverable; raise EPOCHS if the results look ambiguous.
#
# NOTE: cut down from 5 seeds x 30 epochs to 2 seeds x 8 epochs in practice
# on this box - the shared server ("arjuna") was running at load average ~59
# on 16 cores while this was live, so 4 concurrent jobs (2 seeds x on/off)
# each took ~19-22 min/epoch instead of the ~2 min/epoch a single
# unconcurrent job gets. A 30-epoch budget under that contention had an
# unbounded finish time (still improving with no early stop in sight after 9
# epochs / ~3 hours), so this was capped hard at 8 epochs for a deterministic
# ~2.5h finish - well-justified here since polarization is known to show up
# early in training anyway. With n=2 seeds the "polarization rate" itself is
# only a coarse 0%/50%/100% signal per condition, and 8 epochs is a snapshot
# rather than a converged model - raise SEEDS/EPOCHS back up if you have a
# quieter machine or more time.
#
# This machine has 4 GPUs (see `nvidia-smi`); if you want to parallelize,
# run this script's body manually across shells with
# `--override gpu_id=0` / `gpu_id=1` / `gpu_id=2` / `gpu_id=3` instead of
# running it as-is - though see the note above before assuming that's faster.
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(1 2)
EPOCHS=8
STOPPING_STEP=4
PREFIX=seedsweep

for S in "${SEEDS[@]}"; do
  python -m track_c.src.train \
    --run-name "${PREFIX}_seed${S}_lboff" --seed "${S}" \
    --override "load_balancing_coef=0.0" "epochs=${EPOCHS}" "stopping_step=${STOPPING_STEP}" "show_progress=false"

  python -m track_c.src.train \
    --run-name "${PREFIX}_seed${S}_lbon" --seed "${S}" \
    --override "load_balancing_coef=0.01" "epochs=${EPOCHS}" "stopping_step=${STOPPING_STEP}" "show_progress=false"
done

python -m track_c.src.analysis.polarization --seeds "${SEEDS[@]}" --prefix "${PREFIX}"
