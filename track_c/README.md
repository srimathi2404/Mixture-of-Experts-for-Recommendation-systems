# Track C — Self-Attention Experts for MoSE

**Question:** Does swapping Track B's LSTM (`GRU4Rec`) experts for self-attention (`SASRec`-style)
experts improve a multi-task sequential recommender (MoSE), on the same kind of task?

Fully self-contained — does not depend on Track B's code or results. Only the dataset/metric
choice needs to be agreed with Track B up front so the two final numbers are comparable.

## Plan

1. **Data**: a public sequential-recommendation dataset via RecBole (default: `ml-1m`).
   RecBole's datasets are single-task (next-item) by default, so a second task label
   (e.g. rating prediction) is engineered from the same interaction data.
2. **Baseline reference point** (for comparability with Track B, not reproduced here):
   non-sequential MMoE on a flattened snapshot of history.
3. **This track's model**: MoSE with `SASRec`-style self-attention experts — several parallel
   self-attention encoders, each producing one summary vector per user sequence, mixed by an
   MMoE-style softmax gate per task (same gating mechanism Track A/B use).
4. **Evaluation**: Recall@10 / NDCG@10 for next-item, plus accuracy/RMSE for the second task.
5. **Analysis**: expert-utilization plots (per-expert accumulated gate weight over training),
   mirroring Track A's stability analysis in a sequential setting.

## What's built here vs. what comes from libraries

- From RecBole: dataset loading/splitting, the `SASRec` self-attention block, evaluation metrics.
- Built in this track (`src/`): the MoSE wrapper (parallel self-attention experts + MMoE gate),
  the second-task label engineering, and the expert-utilization logging hook.

## Layout

```
track_c/
  configs/
    dataset.yaml          # RecBole dataset/split config
    sasrec_mose.yaml       # model + training hyperparameters
  src/
    models/
      sasrec_expert.py     # wraps RecBole's SASRec encoder as a single "expert" (seq -> vector)
      mmoe_gate.py          # softmax gating mechanism, shared shape with Track A/B
      sasrec_mose.py        # full model: N self-attention experts + per-task MMoE gates
    data/
      second_task.py        # derives a second task label (e.g. rating regression) from
                             # RecBole's interaction data
    utils/
      expert_utilization.py # hook to log per-expert gate weight per batch, for the
                             # utilization plots
    train.py                # training entry point
    evaluate.py             # evaluation entry point, produces the comparison table + plots
  scripts/
    run_train.ps1           # convenience launcher (Windows/PowerShell)
  results/                  # metrics, checkpoints, plots (gitignored contents)
```

## Status

Scaffolding only — see `TODO` markers in each file for what's implemented vs. pending.

## Known environment issue

`torch` currently fails to import in this venv (`OSError: DLL initialization routine failed`
loading `c10.dll`). Nothing here has been run end-to-end yet — fix that before attempting to
train.

## Deliverables (from the project plan)

1. A trained self-attention-expert MoSE model, evaluated with the same metrics/dataset
   conventions as Track B.
2. Expert-utilization plots for this variant.
