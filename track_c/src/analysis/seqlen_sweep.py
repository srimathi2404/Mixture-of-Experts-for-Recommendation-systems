"""Extra experiment #4: sequence length / context window sweep.

Aggregates several Track C runs trained at different MAX_ITEM_LIST_LENGTH
values (see scripts/run_seqlen_sweep.sh) into one plot: x-axis = max sequence
length, y-axis = a ranking metric (default Recall@10). Optionally overlays
Track B's own sweep (same lengths, agreed in advance) if given as a JSON
file of {"<seq_len>": <metric_value>} pairs, turning this into the
cross-architecture comparison the project plan calls the most scientifically
meaningful add-on.

Run from the repo root, after training e.g. seqlen5/seqlen10/seqlen20/seqlen50:
    python -m track_c.src.analysis.seqlen_sweep --run-names seqlen5 seqlen10 seqlen20 seqlen50
"""

import argparse
import json

from ..train import RESULTS_DIR


def load_point(run_name, metric_key):
    with open(RESULTS_DIR / run_name / "metrics.json") as f:
        meta = json.load(f)
    seq_len = meta["resolved"]["MAX_ITEM_LIST_LENGTH"]
    value = meta["test_result"].get(metric_key)
    if value is None:
        raise KeyError(
            f"'{metric_key}' not found in {run_name}/metrics.json test_result "
            f"(available: {list(meta['test_result'].keys())})"
        )
    return seq_len, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--metric", default="recall@10")
    parser.add_argument("--track-b-json", default=None, help='optional {"seq_len": value} JSON from Track B')
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted(load_point(r, args.metric) for r in args.run_names)
    seq_lens, values = zip(*points)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(seq_lens, values, marker="o", label="Track C (self-attention)")

    if args.track_b_json:
        with open(args.track_b_json) as f:
            track_b = json.load(f)
        b_points = sorted((int(k), v) for k, v in track_b.items())
        b_x, b_y = zip(*b_points)
        ax.plot(b_x, b_y, marker="s", label="Track B (LSTM)")

    ax.set_xlabel("max sequence length")
    ax.set_ylabel(args.metric)
    ax.set_title("Sequence length sweep")
    ax.legend()
    fig.tight_layout()

    out_path = args.out or (RESULTS_DIR / "seqlen_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    for seq_len, value in points:
        print(f"  seq_len={seq_len:<4} {args.metric}={value:.4f}")


if __name__ == "__main__":
    main()
