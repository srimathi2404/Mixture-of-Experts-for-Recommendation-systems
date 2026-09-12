# Track C — Self-Attention Experts for MoSE

**Question:** Does swapping Track B's LSTM (`GRU4Rec`) experts for self-attention (`SASRec`-style)
experts improve a multi-task sequential recommender (MoSE), on the same kind of task?

Fully self-contained — does not depend on Track B's code or results. Only the dataset/metric
choice needs to be agreed with Track B up front so the two final numbers are comparable.

## Status

Implemented and wired end-to-end (dataset load → model → loss → eval), verified with a dry-run
smoke test (real ml-1m data, one forward+backward pass, no optimizer step run). **No training has
been run yet** — see "Training" below for the command to kick that off.

## Model

1. **Data**: `ml-1m` via RecBole, sequential/leave-one-out split (`configs/dataset.yaml`).
   RecBole's sequential datasets are next-item only, so a second task label (rating regression)
   is engineered from the same interactions — see "Second task" below.
2. **This track's model** (`src/models/sasrec_mose.py`): `SASRecMoSE` — `n_experts` independent
   `SASRecExpert` towers (`src/models/sasrec_expert.py`, each its own item/position embeddings +
   a RecBole `TransformerEncoder`), mixed per-task by an `MMoEGate` softmax gate
   (`src/models/mmoe_gate.py`). The gate's input is a pooled item embedding independent of any
   one expert, matching MMoE's "gate sees the same shared input the experts consume" design.
   - Task 1 head: next-item logits, scored against a dedicated item-embedding table.
   - Task 2 head: linear layer → scalar rating prediction.
3. **Evaluation**: Recall@10 / NDCG@10 for next-item (via RecBole's own evaluator — directly
   comparable to Track B's numbers), RMSE for the rating task, plus a manual ranking-metric
   pass used for the cold-start bucketing below.
4. **Analysis**: expert-utilization plots (per-expert, per-task accumulated gate weight),
   mirroring Track A's stability analysis in a sequential setting.

### Second task: no extra label engineering needed

RecBole's `SequentialDataset.data_augmentation` builds each augmented sample as
`(history items..., target item)` and keeps *every other field of the target row* attached —
including `RATING_FIELD`. So `interaction['rating']` already **is** "the rating the user gave the
item we're asking the model to predict next": the second-task label, with no manual lookup
required. See `src/data/second_task.py` for the (one-line) accessor and the reasoning, verified
against RecBole's source.

### Why subclass `SequentialRecommender`

`SASRecMoSE` implements `calculate_loss` / `predict` / `full_sort_predict` with the exact same
contract RecBole's own `SASRec` model uses. That means RecBole's stock `Trainer` (optimizer loop,
early stopping, checkpointing, tensorboard) and evaluator (Recall@K/NDCG@K) work unmodified — no
custom training loop, and the next-item metrics come from RecBole's tested implementation rather
than a hand-rolled one. `calculate_loss` returns a 3-tuple `(task1_loss, task2_loss, aux_loss)`
rather than one summed scalar: RecBole's `Trainer` sums tuple losses for backward automatically,
but logs and tensorboards each component separately — giving per-task loss curves for free.

## What's built here vs. what comes from libraries

- From RecBole: dataset loading/splitting (`create_dataset`, `data_preparation`), the
  `TransformerEncoder` building block, the `Trainer` training loop, and the ranking evaluator.
- Built in this track (`src/`): the MoSE wrapper (parallel self-attention experts + MMoE gate),
  the expert-utilization logging hook, the load-balancing auxiliary loss, and all of
  `src/analysis/`.

## Layout

```
track_c/
  configs/
    dataset.yaml           # RecBole dataset/split config
    sasrec_mose.yaml        # model + training hyperparameters
  src/
    models/
      sasrec_expert.py      # one SASRec-style expert (seq -> vector), + attention-capture hooks
      mmoe_gate.py           # softmax gating, shared shape/contract with Track A/B
      sasrec_mose.py         # full model: N self-attention experts + per-task MMoE gates
    data/
      second_task.py         # rating-regression label accessor (see above)
    utils/
      expert_utilization.py  # per-(task, expert) gate-weight tracker + plot
      load_balancing.py      # Shazeer et al. CV^2 auxiliary loss (extra experiment #6)
    analysis/
      attention_viz.py       # extra experiment #1: attention heatmaps
      seqlen_sweep.py         # extra experiment #4: aggregation/plot across seq-length runs
      polarization.py         # extra experiment #6: polarization-rate table across seeds
    train.py                 # training entry point
    evaluate.py               # evaluation entry point: metrics, cold-start table, reports
  scripts/
    prepare_dataset.py       # one-time: fetch + convert ml-1m to RecBole atomic files
    run_train.sh              # generic launcher, forwards args to train.py
    run_core.sh                # train + evaluate the core deliverable model
    run_positional_ablation.sh # extra experiment #2
    run_seqlen_sweep.sh        # extra experiment #4
    run_seed_sweep.sh          # extra experiment #6 (+ polarization table)
  results/                  # metrics, checkpoints, plots per run (gitignored contents)
```

## Setup

```bash
pip install -r requirements.txt   # see requirements.txt for the torch install caveat
python track_c/scripts/prepare_dataset.py   # one-time: builds dataset/ml-1m/*.inter etc.
```

`prepare_dataset.py` exists because RecBole's own hosted mirror of pre-converted datasets
(`recbole.s3-accelerate.amazonaws.com`) currently returns `403 AccessDenied`, so the automatic
download `create_dataset()` would normally try no longer works. This script instead builds the
RecBole atomic files directly from the official GroupLens `ml-1m.zip`, with a line-count check
against the well-known ml-1m file sizes as an integrity check.

## Training

Everything below is written and smoke-tested (real data, one forward+backward pass) but **not
yet trained**. All commands run from the repo root.

```bash
# Core deliverable
bash track_c/scripts/run_core.sh
# equivalent to:
#   python -m track_c.src.train --run-name core
#   python -m track_c.src.evaluate --run-name core
```

Each run writes to `track_c/results/<run-name>/`: a checkpoint, `metrics.json`
(valid/test metrics + the resolved config), `expert_utilization.png` +
`utilization_summary.json`, and (after `evaluate.py`) `evaluation_report.json` with the
comparison-table numbers, RMSE, and the cold-start-by-history-length table.

Arbitrary hyperparameter overrides: `--override key=value ...` (YAML-parsed, so
`true`/`false`/numbers/lists work), or `--seed N`. E.g.:

```bash
python -m track_c.src.train --run-name my_run --override n_experts=8 learning_rate=0.0005
```

### Extra experiments (docs/extra_experiments.md)

All six are wired up; pick what compute time allows.

| # | Experiment | Command | Extra training? |
|---|---|---|---|
| 1 | Attention visualization | `python -m track_c.src.analysis.attention_viz --run-name core` | No (needs a trained `core` run) |
| 2 | Positional encoding ablation | `bash track_c/scripts/run_positional_ablation.sh` | Yes (2 runs) |
| 3 | Gate specialization by task | *(automatic — `expert_utilization.png` is already grouped by task)* | No |
| 4 | Sequence length sweep | `bash track_c/scripts/run_seqlen_sweep.sh` | Yes (4 runs) |
| 5 | Cold-start robustness | *(automatic — see `evaluation_report.json`'s `cold_start_by_history_length`)* | No |
| 6 | Load-balancing loss cross-check | `bash track_c/scripts/run_seed_sweep.sh` | Yes (4 reduced-budget runs: 2 seeds x on/off) |

#4 is most useful compared directly against Track B's own sweep at the same lengths — pass
`--track-b-json path/to/track_b_seqlen.json` (`{"seq_len": metric_value}`) to
`analysis/seqlen_sweep.py` to overlay both curves on one plot.

This machine has 4x RTX 2080 Ti (`nvidia-smi`). The sweep scripts run sequentially by default;
for extra experiments #4/#6 (many independent runs), pass `--override gpu_id=0/1/2/3` to spread
runs across shells manually if you want to parallelize.

**Caution before parallelizing on a shared server:** `arjuna` is multi-user, and when we ran 4
training jobs concurrently (one per GPU) here, GPU utilization stayed at ~10% while CPU load
average sat around 59 on 16 cores — the jobs were CPU-starved, not GPU-starved, and each one ran
roughly **10x slower per epoch** than the same job running alone. 4-way "parallelism" bought
negative real speedup in that state. Check `uptime` and `nvidia-smi` before assuming concurrent
runs will actually finish faster — on a busy shared machine, running fewer jobs at once (or one
at a time) can complete sooner in aggregate.

## Known environment note (resolved)

The conda env `dlnlp`'s default `pip install torch` initially pulled a `cu130` build that failed
with `CUDA initialization: The NVIDIA driver on your system is too old` (driver 565.57.01 only
supports up to CUDA 12.7). Fixed by installing `torch==2.5.1` with the `cu121` wheel — see
`requirements.txt`. `torch.cuda.is_available()` now returns `True`.

## Deliverables (from the project plan)

1. A trained self-attention-expert MoSE model, evaluated with the same metrics/dataset
   conventions as Track B.
2. Expert-utilization plots for this variant (also satisfies extra experiment #3).
