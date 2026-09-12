"""MMoE-style softmax gating, mixing a fixed set of expert outputs per task.

Same gating mechanism used by Track A (feedforward experts) and Track B
(LSTM experts) - only what's inside each expert differs across tracks. Kept
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
            - typically the same feature the experts also consume. In Track
            C's case, this is an expert-agnostic pooled item embedding (see
            SASRecMoSE._gate_input) so the gate's decision doesn't collapse
            onto whichever expert happens to compute it.
        n_experts: number of experts to gate over.
        n_tasks: number of tasks; one independent gate per task.
    """

    def __init__(self, input_size: int, n_experts: int, n_tasks: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.gate_layers = nn.ModuleList(
            [nn.Linear(input_size, n_experts) for _ in range(n_tasks)]
        )

    def forward(self, gate_input, expert_outputs):
        """
        Args:
            gate_input: (batch, input_size) features driving the gate.
            expert_outputs: (batch, n_experts, expert_dim) stacked expert
                summary vectors.

        Returns:
            task_outputs: list of length n_tasks, each (batch, expert_dim) -
                the gated mixture of experts for that task.
            gate_weights: (batch, n_tasks, n_experts) - softmax weights,
                needed by utils/expert_utilization.py for the utilization plots.
        """
        task_outputs = []
        per_task_weights = []
        for gate_layer in self.gate_layers:
            weights = torch.softmax(gate_layer(gate_input), dim=-1)  # (B, n_experts)
            per_task_weights.append(weights)
            mixed = torch.einsum("be,bed->bd", weights, expert_outputs)
            task_outputs.append(mixed)

        gate_weights = torch.stack(per_task_weights, dim=1)  # (B, n_tasks, n_experts)
        return task_outputs, gate_weights
