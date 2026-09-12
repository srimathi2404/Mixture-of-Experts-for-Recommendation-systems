"""Logs per-expert accumulated gate weight per batch, for utilization plots.

Same measurement Track A introduces for the feedforward-expert setting
("actual utilization, not in the base repo") - reused here for the
self-attention-expert setting, per the project plan's callback to Track A's
stability analysis (core deliverable #2).

Utilization is tracked *per task* from the start (shape: n_tasks x
n_experts), not averaged over tasks - this doubles as extra experiment #3
(gate specialization by task): `.plot()` already produces the grouped
bar-by-task chart that experiment asks for, no separate code path needed.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


class ExpertUtilizationTracker:
    """Accumulates gate weight mass per (task, expert) across training.

    Usage: call `.update(gate_weights)` once per training batch with the
    (batch, n_tasks, n_experts) softmax weights returned by MMoEGate, then
    `.summary()` at eval time (or periodically) to get per-expert utilization
    fractions, and `.plot(...)` to produce the utilization plot.
    """

    def __init__(self, n_tasks: int, n_experts: int, task_names=None):
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self.task_names = list(task_names) if task_names else [
            f"task{i + 1}" for i in range(n_tasks)
        ]
        self._sums = torch.zeros(n_tasks, n_experts)
        self._count = 0

    def update(self, gate_weights: torch.Tensor) -> None:
        """gate_weights: (batch, n_tasks, n_experts) softmax output."""
        self._sums += gate_weights.sum(dim=0).detach().cpu()
        self._count += gate_weights.size(0)

    def summary(self) -> "OrderedDict[str, list]":
        """Returns {task_name: [utilization_fraction_per_expert]}.

        Each task's fractions sum to 1 (every sample's softmax gate row sums
        to 1, so summing across the batch/epoch and dividing by the sample
        count preserves that).
        """
        if self._count == 0:
            raise RuntimeError(
                "ExpertUtilizationTracker has no accumulated batches yet - "
                "call .update() during training first."
            )
        fractions = (self._sums / self._count).tolist()
        return OrderedDict(
            (self.task_names[t], fractions[t]) for t in range(self.n_tasks)
        )

    def min_utilization(self) -> float:
        """Smallest per-expert utilization fraction across all tasks - the
        quantity extra experiment #6's polarization-rate metric thresholds."""
        summary = self.summary()
        return min(min(fracs) for fracs in summary.values())

    def plot(self, out_path: str) -> None:
        """Grouped bar chart: experts on the x-axis, accumulated gate weight
        on the y-axis, one bar color per task. Mirrors Track A's
        expert-utilization plot for direct comparison, and is extra
        experiment #3's deliverable as-is."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = self.summary()
        experts = [f"E{i}" for i in range(self.n_experts)]
        x = np.arange(self.n_experts)
        width = 0.8 / max(self.n_tasks, 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        for t, name in enumerate(self.task_names):
            ax.bar(x + t * width, summary[name], width, label=name)
        ax.set_xticks(x + width * (self.n_tasks - 1) / 2)
        ax.set_xticklabels(experts)
        ax.set_ylabel("Accumulated gate weight (fraction)")
        ax.set_title("Expert utilization by task")
        ax.axhline(1.0 / self.n_experts, color="gray", linestyle="--", linewidth=1, label="uniform")
        ax.legend()
        fig.tight_layout()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def save_json(self, out_path: str) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
