"""Full Track C model: N parallel self-attention experts + per-task MMoE gates.

This is the "glue code" the project plan calls out as not existing in any
library — combining SASRecExpert (this track's expert) with MMoEGate (shared
gating pattern with Track A/B) into one multi-task model trainable on top of
RecBole's data pipeline.

Task 1 (head 1): next-item recommendation (softmax over item vocabulary).
Task 2 (head 2): engineered second task, e.g. rating regression
    (see src/data/second_task.py for label construction).
"""

from torch import nn

from .mmoe_gate import MMoEGate
from .sasrec_expert import SASRecExpert


class SASRecMoSE(nn.Module):
    """Multi-task sequential recommender with self-attention experts.

    Args:
        n_items: item vocabulary size.
        n_experts: number of parallel SASRecExpert instances.
        n_tasks: number of task heads (default 2: next-item + rating).
        expert_kwargs: forwarded to each SASRecExpert.
        gate_hidden_size: unused placeholder for a future non-linear gate;
            current gate is linear-softmax per MMoEGate.
    """

    def __init__(
        self,
        n_items: int,
        n_experts: int = 4,
        n_tasks: int = 2,
        expert_kwargs: dict | None = None,
        gate_hidden_size: int = 64,
    ):
        super().__init__()
        expert_kwargs = expert_kwargs or {}
        # TODO: instantiate n_experts independent SASRecExpert modules
        # TODO: instantiate MMoEGate(input_size=..., n_experts, n_tasks)
        # TODO: task-specific heads:
        #   - task 1: linear projection to item embedding space (tied weights
        #     with the item embedding table, standard SASRec prediction head)
        #   - task 2: linear projection to a scalar (rating regression)
        raise NotImplementedError("Track C: implement the SASRecMoSE model")

    def forward(self, item_seq, item_seq_len):
        """
        Returns:
            next_item_logits: (batch, n_items)
            second_task_pred: (batch,) predicted rating (or logits, if the
                second task ends up being classification instead of regression)
            gate_weights: (batch, n_tasks, n_experts), for utilization logging.
        """
        raise NotImplementedError("Track C: implement the SASRecMoSE model")

    def calculate_loss(self, interaction, task_loss_weights=(1.0, 1.0)):
        """Combined multi-task loss: next-item CE + second-task loss.

        TODO: pull item_seq / item_seq_len / labels for both tasks out of
        RecBole's `interaction` object, compute each task's loss, combine
        with task_loss_weights, and return (total_loss, per-task losses dict)
        so per-task losses can be logged separately.
        """
        raise NotImplementedError("Track C: implement the combined training loss")
