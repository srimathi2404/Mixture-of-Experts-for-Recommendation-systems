"""Extra experiment #1: attention weight visualization.

Loads a trained run, picks a handful of test users, and plots the
self-attention weight matrix (query position x attended-to position) for
one expert's last transformer layer, averaged over heads. Turns the model
from "a metric number" into something explainable in the presentation: does
attention concentrate on the most recent items (recency bias) or spike on
specific older items regardless of distance (genuine long-range dependency)?

Run from the repo root:
    python -m track_c.src.analysis.attention_viz --run-name core --n-users 5
"""

import argparse
import random
from pathlib import Path

import torch

from ..evaluate import load_run
from ..train import RESULTS_DIR


@torch.no_grad()
def plot_attention_for_users(model, eval_data, device, expert_idx, n_users, out_dir, layer=-1, seed=0):
    expert = model.experts[expert_idx]
    expert.enable_attention_capture()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    plotted = 0
    try:
        for interaction, _, _, _ in eval_data:
            if plotted >= n_users:
                break
            interaction = interaction.to(device)
            item_seq = interaction[model.ITEM_SEQ]
            item_seq_len = interaction[model.ITEM_SEQ_LEN]
            attention_mask = model.get_attention_mask(item_seq)

            expert(item_seq, item_seq_len, attention_mask)
            layer_weights = expert.pop_attention_weights()[layer]  # (B, n_heads, seq, seq)
            attn = layer_weights.mean(dim=1)  # average heads -> (B, seq, seq)

            indices = list(range(item_seq.size(0)))
            rng.shuffle(indices)
            for i in indices:
                if plotted >= n_users:
                    break
                length = int(item_seq_len[i].item())
                if length < 3:
                    continue  # too short to make an interesting heatmap
                heat = attn[i, :length, :length].cpu().numpy()

                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(heat, ax=ax, cmap="viridis", cbar_kws={"label": "attention weight"})
                ax.set_xlabel("attended-to position (history order)")
                ax.set_ylabel("query position")
                ax.set_title(f"expert {expert_idx}, layer {layer}, len {length}")
                fig.tight_layout()
                fig.savefig(out_dir / f"attention_user{plotted}.png", dpi=150)
                plt.close(fig)
                plotted += 1
    finally:
        expert.disable_attention_capture()

    return plotted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="core")
    parser.add_argument("--expert", type=int, default=0, help="which expert's attention to visualize")
    parser.add_argument("--n-users", type=int, default=5)
    parser.add_argument("--layer", type=int, default=-1, help="transformer layer index, -1 = last")
    parser.add_argument("--seed", type=int, default=0, help="which test users get sampled")
    args = parser.parse_args()

    config, model, _dataset, test_data, _meta = load_run(args.run_name)
    out_dir = RESULTS_DIR / args.run_name / "plots" / "attention"

    n = plot_attention_for_users(
        model,
        test_data,
        config["device"],
        expert_idx=args.expert,
        n_users=args.n_users,
        out_dir=out_dir,
        layer=args.layer,
        seed=args.seed,
    )
    print(f"Saved {n} attention heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
