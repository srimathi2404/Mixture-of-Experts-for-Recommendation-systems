"""Evaluation + reporting entry point for Track C.

Loads a trained run's checkpoint (by --run-name, matching the folder
train.py wrote to), then produces:

1. Next-item Recall@K / NDCG@K via RecBole's own evaluator (deliverable #1;
   directly comparable to Track B's numbers on the same dataset/split).
2. Rating-regression RMSE for the second task (deliverable #1, task 2).
3. A manual re-derivation of the same ranking metrics (extra experiment #3's
   utilization plot needs no manual metric code, but this manual pass is
   what bucketing-by-history-length in #4 requires) - printed alongside
   RecBole's numbers as a cross-check; small differences are expected from
   floating-point tie handling, not a bug if they're close.
4. Cold-start / sparse-user robustness table (extra experiment #5): the same
   ranking metrics, bucketed by each user's *raw* (pre-truncation) number of
   interactions - not `item_seq_len`, which is capped at
   MAX_ITEM_LIST_LENGTH and so is a poor activity proxy once most users
   exceed that cap (true on ml-1m: mean ~165 interactions/user, cap 50 -
   item_seq_len alone would put almost everyone in one bucket).

`load_run` is also imported by src/analysis/attention_viz.py so both tools
share one "load a trained run" code path.

Run from the repo root:
    python -m track_c.src.evaluate --run-name core
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from .models.sasrec_mose import SASRecMoSE
from .train import CONFIG_DIR, DEFAULT_CONFIG_FILES, RESULTS_DIR

COLD_START_BUCKETS = ((0, 10), (10, 50), (50, float("inf")))


def load_run(run_name, config_files=None):
    """Rebuilds the exact config a run was trained with (dataset.yaml +
    sasrec_mose.yaml + that run's saved overrides), reloads its checkpoint,
    and returns (config, model, dataset, test_data, meta) with the model in
    eval() mode and ready to use."""
    run_dir = RESULTS_DIR / run_name
    with open(run_dir / "metrics.json") as f:
        meta = json.load(f)

    config_files = config_files or DEFAULT_CONFIG_FILES
    config = Config(
        model=SASRecMoSE,
        config_file_list=[str(f) for f in config_files],
        config_dict=meta["config_overrides"],
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRecMoSE(config, train_data.dataset).to(config["device"])
    checkpoint = torch.load(meta["checkpoint"], map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()

    return config, model, dataset, test_data, meta


def load_raw_user_interaction_counts(config) -> Counter:
    """Raw (pre-truncation, pre-split) interaction count per user, read
    straight from the atomic `.inter` file `prepare_dataset.py` wrote. This
    is what extra experiment #5 actually means by "how many interactions
    they have" - `item_seq_len` from the dataloader is capped at
    MAX_ITEM_LIST_LENGTH and can't tell a 60-interaction user from a
    600-interaction one.

    Returns: Counter keyed by the *raw* user id string (as it appears in the
    .inter file), not RecBole's internal integer token id.
    """
    inter_path = Path(config["data_path"]) / f"{config['dataset']}.inter"
    uid_field = config["USER_ID_FIELD"]
    counts = Counter()
    with open(inter_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        uid_col = next(i for i, name in enumerate(header) if name.split(":")[0] == uid_field)
        for line in f:
            counts[line.rstrip("\n").split("\t")[uid_col]] += 1
    return counts


@torch.no_grad()
def evaluate_second_task_and_ranking(model, dataset, config, eval_data, device, k=10):
    """One manual pass over eval_data computing:
    - rating-regression squared errors (-> RMSE), and
    - the rank of the true next item among all items (masking the PAD item
      id 0, same as RecBole's own full-sort evaluator), per example, plus
      each example's *raw* user interaction count (for cold-start bucketing;
      see load_raw_user_interaction_counts for why not item_seq_len).

    Mirrors RecBole's own full-sort ranking convention (see
    Trainer._full_sort_batch_eval: `scores[:, 0] = -inf`, no history
    masking for sequential models) so the recall/ndcg computed here should
    closely match RecBole's evaluator output.
    """
    raw_counts = load_raw_user_interaction_counts(config)
    # RecBole's internal integer user id -> the raw id string it came from.
    id2token = dataset.field2id_token[model.USER_ID]

    squared_errors = []
    ranks = []
    user_activity = []

    for interaction, _, positive_u, positive_i in eval_data:
        interaction = interaction.to(device)
        item_seq = interaction[model.ITEM_SEQ]
        item_seq_len = interaction[model.ITEM_SEQ_LEN]

        next_item_logits, second_task_pred, _ = model.forward(item_seq, item_seq_len)
        next_item_logits[:, 0] = float("-inf")  # PAD item is never a valid target

        rating_label = interaction[model.RATING].float()
        squared_errors.append(((second_task_pred - rating_label) ** 2).cpu())

        target_i = positive_i.to(device)
        target_scores = next_item_logits.gather(1, target_i.view(-1, 1)).squeeze(1)
        rank = (next_item_logits > target_scores.unsqueeze(1)).sum(dim=1) + 1
        ranks.append(rank.cpu())

        user_ids = interaction[model.USER_ID].cpu().numpy()
        activity = [raw_counts[id2token[uid]] for uid in user_ids]
        user_activity.append(torch.tensor(activity))

    squared_errors = torch.cat(squared_errors)
    ranks = torch.cat(ranks)
    user_activity = torch.cat(user_activity)

    rmse = squared_errors.mean().sqrt().item()
    recall_at_k, ndcg_at_k = _ranking_metrics(ranks, k)

    return {
        "rmse": rmse,
        f"recall@{k}_manual": recall_at_k,
        f"ndcg@{k}_manual": ndcg_at_k,
        "ranks": ranks,
        "user_activity": user_activity,
    }


def _ranking_metrics(ranks: torch.Tensor, k: int):
    hits = ranks <= k
    recall = hits.float().mean().item()
    ndcg = torch.where(
        hits, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)
    ).mean().item()
    return recall, ndcg


def cold_start_table(ranks: torch.Tensor, user_activity: torch.Tensor, k=10, buckets=COLD_START_BUCKETS):
    """Extra experiment #5: bucket users by their *raw* interaction count,
    report ranking metrics separately per bucket instead of one aggregate
    number.

    `buckets`: either a list of (lo, hi] cutoffs (default: the literal
    <10/10-50/>50 thresholds from docs/extra_experiments.md - meaningful now
    that `user_activity` is the raw, uncapped count), or None to fall back
    to data-driven tertiles if the fixed cutoffs don't suit a given dataset
    (e.g. all-empty or all-one-bucket).
    """
    if buckets is None:
        q1, q2 = torch.quantile(user_activity.float(), torch.tensor([1 / 3, 2 / 3])).tolist()
        cutoffs = sorted(
            {int(user_activity.min().item()) - 1, int(q1), int(q2), int(user_activity.max().item())}
        )
        buckets = list(zip(cutoffs[:-1], cutoffs[1:]))

    table = {}
    for lo, hi in buckets:
        mask = (user_activity > lo) & (user_activity <= hi)
        n = int(mask.sum().item())
        if n == 0:
            continue
        recall, ndcg = _ranking_metrics(ranks[mask], k)
        label = f"({lo}, {hi if hi != float('inf') else 'inf'}]"
        table[label] = {"n_examples": n, f"recall@{k}": recall, f"ndcg@{k}": ndcg}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="core")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--cold-start-tertiles",
        action="store_true",
        help="use data-driven tertiles instead of the fixed (0,10]/(10,50]/(50,inf] cutoffs "
        "(useful if those fixed cutoffs turn out empty/degenerate on a given dataset)",
    )
    args = parser.parse_args()

    config, model, dataset, test_data, meta = load_run(args.run_name)
    device = config["device"]

    trainer = Trainer(config, model)
    ranking_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

    extra = evaluate_second_task_and_ranking(model, dataset, config, test_data, device, k=args.k)
    ranks, user_activity = extra.pop("ranks"), extra.pop("user_activity")
    cold_start = cold_start_table(
        ranks, user_activity, k=args.k, buckets=None if args.cold_start_tertiles else COLD_START_BUCKETS
    )

    report = {
        "run_name": args.run_name,
        "trained_config": meta.get("resolved", {}),
        "ranking_metrics_recbole": {str(k_): float(v_) for k_, v_ in ranking_metrics.items()},
        "second_task_rmse": extra["rmse"],
        "manual_ranking_cross_check": {k_: v_ for k_, v_ in extra.items() if k_ != "rmse"},
        "cold_start_by_history_length": cold_start,
    }

    run_dir = RESULTS_DIR / args.run_name
    with open(run_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
