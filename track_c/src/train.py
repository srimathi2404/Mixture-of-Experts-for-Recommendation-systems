"""Training entry point for Track C (self-attention-expert MoSE).

TODO:
1. Load configs/dataset.yaml + configs/sasrec_mose.yaml via RecBole's Config.
2. Build the dataset/dataloaders via RecBole's create_dataset / data_preparation,
   then attach the second-task label with src/data/second_task.add_next_rating_label.
3. Instantiate src/models/sasrec_mose.SASRecMoSE.
4. Standard training loop (or RecBole's Trainer, subclassed to handle the
   combined multi-task loss from SASRecMoSE.calculate_loss).
5. Log per-batch gate weights into an ExpertUtilizationTracker.
6. Save checkpoints + utilization plot + metrics table to track_c/results/.

Run: python -m track_c.src.train
"""


def main():
    raise NotImplementedError("Track C: implement the training loop")


if __name__ == "__main__":
    main()
