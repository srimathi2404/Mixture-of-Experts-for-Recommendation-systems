"""MMoE-style softmax gating, mixing a fixed set of expert outputs per task.

Same gating mechanism used by Track A (feedforward experts) and Track B
(LSTM experts) — only what's inside each expert differs across tracks. Kept
as a standalone module here (rather than imported from Track A/B) to keep
Track C fully self-contained per the project's independence rule; the three
implementations should stay behaviorally equivalent.
"""

import torch
from torch import nn


class MMoEGate(nn.Module):
    """Per-task softmax gate over a shared pool of experts.

    Args:
        input_size: dimension of the (shared) input fed to the gating network
            — typically the same feature the experts also consume.
        n_experts: number of experts to gate over.
        n_tasks: number of tasks; one independent gate per task.
    """

    def __init__(self, input_size: int, n_experts: int, n_tasks: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        # TODO: one linear layer (input_size -> n_experts) per task
        raise NotImplementedError("Track C: implement the MMoE gate")

    def forward(self, gate_input, expert_outputs):
        """
        Args:
            gate_input: (batch, input_size) features driving the gate.
            expert_outputs: (batch, n_experts, expert_dim) stacked expert
                summary vectors.

        Returns:
            task_outputs: list of length n_tasks, each (batch, expert_dim) —
                the gated mixture of experts for that task.
            gate_weights: (batch, n_tasks, n_experts) — softmax weights,
                needed by utils/expert_utilization.py for the utilization plots.
        """
        raise NotImplementedError("Track C: implement the MMoE gate")
