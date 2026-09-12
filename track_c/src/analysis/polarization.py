"""Extra experiment #6: load-balancing-loss cross-check.

Track A measures whether an auxiliary load-balancing loss prevents gate
"polarization" (some experts ending up with near-zero usage) in the
non-sequential MMoE setting, across a sweep of seeds (polarization is
stochastic). This script applies the same polarization metric to Track C's
sequential setting: fraction of seeds where some expert's accumulated gate
weight falls under `--threshold`, with vs. without the load-balancing loss.

Expects runs named `<prefix>_seed<K>_lboff` / `<prefix>_seed<K>_lbon` for
each seed in --seeds (see scripts/run_seed_sweep.sh), each with a saved
results/<run_name>/utilization_summary.json and metrics.json.

Run from the repo root:
    python -m track_c.src.analysis.polarization --seeds 1 2 3 4 5 --prefix seedsweep
"""

import argparse
import json

from ..train import RESULTS_DIR


def is_polarized(utilization_summary, threshold):
    return any(min(fracs) < threshold for fracs in utilization_summary.values())


def load_run_stats(run_name, metric_key):
    run_dir = RESULTS_DIR / run_name
    with open(run_dir / "utilization_summary.json") as f:
        utilization = json.load(f)
    with open(run_dir / "metrics.json") as f:
        meta = json.load(f)
    return utilization, meta["test_result"].get(metric_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--prefix", default="seedsweep")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="an expert is considered 'collapsed' below this accumulated gate-weight fraction",
    )
    parser.add_argument("--metric", default="recall@10")
    args = parser.parse_args()

    rows = []
    for lb in ("off", "on"):
        polarized = 0
        metric_values = []
        for seed in args.seeds:
            run_name = f"{args.prefix}_seed{seed}_lb{lb}"
            utilization, metric_value = load_run_stats(run_name, args.metric)
            if is_polarized(utilization, args.threshold):
                polarized += 1
            if metric_value is not None:
                metric_values.append(metric_value)

        rows.append(
            {
                "load_balancing_loss": lb,
                "polarization_rate": polarized / len(args.seeds),
                f"avg_{args.metric}": sum(metric_values) / len(metric_values) if metric_values else None,
                "n_seeds": len(args.seeds),
            }
        )

    out_path = RESULTS_DIR / f"{args.prefix}_polarization_table.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    header = f"{'load_balancing':<16}{'polarization_rate':<20}{'avg_' + args.metric:<20}"
    print(header)
    for r in rows:
        avg = r[f"avg_{args.metric}"]
        avg_str = f"{avg:.4f}" if avg is not None else "n/a"
        print(f"{r['load_balancing_loss']:<16}{r['polarization_rate']:<20.3f}{avg_str:<20}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
