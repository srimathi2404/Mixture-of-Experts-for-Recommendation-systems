"""Auxiliary load-balancing loss for gated Mixture-of-Experts models.

This is the same fix Track A imports from the large-scale MoE language-model
world (Shazeer et al. 2017, "Outrageously Large Neural Networks") to counter
gate "polarization" (some experts ending up with near-zero usage). The
formula is architecture-agnostic - it only looks at the gate's softmax
outputs, never at what's inside an expert - so it drops into Track C's
sequential/self-attention setting unchanged. See extra_experiments.md #6.
"""

import torch


def load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Coefficient-of-variation-squared importance loss.

    For each task, sums the softmax gate weight each expert received across
    the batch ("importance"), then penalizes how unevenly that importance is
    spread across experts: 0 when every expert gets exactly 1/n_experts of
    the mass on average, growing as usage concentrates on fewer experts.

    Args:
        gate_weights: (batch, n_tasks, n_experts) softmax weights, as
            returned by MMoEGate.forward.

    Returns:
        Scalar tensor, averaged over tasks.
    """
    importance = gate_weights.sum(dim=0)  # (n_tasks, n_experts)
    mean = importance.mean(dim=-1)
    var = importance.var(dim=-1, unbiased=False)
    cv_squared = var / (mean.pow(2) + 1e-10)
    return cv_squared.mean()
