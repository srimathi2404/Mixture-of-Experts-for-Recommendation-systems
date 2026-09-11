"""Logs per-expert accumulated gate weight per batch, for utilization plots.

Same measurement Track A introduces for the feedforward-expert setting
("actual utilization, not in the base repo") — reused here for the
self-attention-expert setting, per the project plan's callback to Track A's
stability analysis.
"""

from collections import defaultdict

import torch


class ExpertUtilizationTracker:
    """Accumulates gate weight mass per (task, expert) across training.

    Usage: call `.update(gate_weights)` once per training batch with the
    (batch, n_tasks, n_experts) softmax weights returned by MMoEGate, then
    `.summary()` at eval time (or periodically) to get per-expert utilization
    fractions, and `.plot(...)` to produce the utilization plot.
    """

    def __init__(self, n_tasks: int, n_experts: int):
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self._sums = torch.zeros(n_tasks, n_experts)
        self._count = 0

    def update(self, gate_weights: torch.Tensor) -> None:
        """gate_weights: (batch, n_tasks, n_experts) softmax output."""
        # TODO: accumulate gate_weights.sum(dim=0) into self._sums,
        # increment self._count by batch size
        raise NotImplementedError("Track C: implement utilization accumulation")

    def summary(self) -> dict:
        """Returns {task_idx: [utilization_fraction_per_expert]}.

        TODO: normalize self._sums by self._count, return as plain dict for
        easy plotting/logging.
        """
        raise NotImplementedError("Track C: implement utilization summary")

    def plot(self, out_path: str) -> None:
        """Bar chart of per-expert utilization, one subplot per task.

        TODO: matplotlib bar chart using self.summary(), saved to out_path.
        Mirrors Track A's expert-utilization plot for direct comparison.
        """
        raise NotImplementedError("Track C: implement utilization plot")
