"""Evaluation entry point for Track C.

TODO:
1. Load a trained checkpoint (from src/train.py).
2. Compute Recall@10 / NDCG@10 for the next-item task (RecBole's built-in
   sequential-recommendation metrics) and accuracy/RMSE for the second task.
3. Produce the comparison table against Track B's numbers (same dataset/split
   convention, agreed up front) — Deliverable 1.
4. Produce the expert-utilization plot via
   src/utils/expert_utilization.ExpertUtilizationTracker.plot — Deliverable 2.

Run: python -m track_c.src.evaluate --checkpoint <path>
"""


def main():
    raise NotImplementedError("Track C: implement evaluation + reporting")


if __name__ == "__main__":
    main()
