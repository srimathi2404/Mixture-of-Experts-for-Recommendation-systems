"""Full Track C model: N parallel self-attention experts + per-task MMoE gates.

This is the "glue code" the project plan calls out as not existing in any
library - combining SASRecExpert (this track's expert) with MMoEGate (shared
gating pattern with Track A/B) into one multi-task model trainable on top of
RecBole's data pipeline.

Task 1 (head 1): next-item recommendation (softmax over item vocabulary).
Task 2 (head 2): engineered second task, rating regression
    (see src/data/second_task.py for label construction).

Subclasses RecBole's `SequentialRecommender` so it plugs directly into
RecBole's stock `Trainer` and evaluator: `calculate_loss` / `predict` /
`full_sort_predict` follow the exact same contract RecBole's own SASRec
model uses (see recbole.model.sequential_recommender.sasrec), so Recall@K /
NDCG@K for the next-item task come from RecBole's tested evaluator rather
than a hand-rolled reimplementation.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender

from ..data.second_task import get_rating_label
from ..utils.load_balancing import load_balancing_loss
from .mmoe_gate import MMoEGate
from .sasrec_expert import SASRecExpert


class SASRecMoSE(SequentialRecommender):
    """Multi-task sequential recommender with self-attention experts.

    All hyperparameters are read from the RecBole `config` object (see
    configs/sasrec_mose.yaml) rather than passed individually, matching
    RecBole's own model constructor convention (`__init__(self, config,
    dataset)`), which is what lets this model be trained via RecBole's
    stock `Trainer`.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_experts = config["n_experts"]
        self.n_tasks = config["n_tasks"]
        self.hidden_size = config["hidden_size"]
        self.task_loss_weights = config["task_loss_weights"]
        self.load_balancing_coef = config["load_balancing_coef"]
        self.use_positional_embedding = config["use_positional_embedding"]
        self.initializer_range = config["initializer_range"]
        self.RATING = config["RATING_FIELD"]

        expert_kwargs = dict(
            n_items=self.n_items,
            hidden_size=self.hidden_size,
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            inner_size=config["inner_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attn_dropout_prob=config["attn_dropout_prob"],
            hidden_act=config["hidden_act"],
            layer_norm_eps=config["layer_norm_eps"],
            max_seq_length=self.max_seq_length,
            use_positional_embedding=self.use_positional_embedding,
        )
        self.experts = nn.ModuleList(
            SASRecExpert(**expert_kwargs) for _ in range(self.n_experts)
        )

        # Expert-agnostic feature used only to drive the gate (mean-pooled
        # item embeddings over the valid part of the sequence). Keeps the
        # gate's decision independent of any single expert's private
        # representation - matching MMoE's "gate sees the same shared input
        # the experts consume" design (Track A/B use the same convention).
        self.gate_item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gate = MMoEGate(self.hidden_size, self.n_experts, self.n_tasks)

        # Task heads. Task 1 scores items via a dedicated embedding table
        # (the gated task-1 output isn't tied to any one expert's embedding
        # space, so it needs its own - same role as SASRec's
        # `test_item_emb = self.item_embedding.weight`, just not shared with
        # any expert).
        self.item_scorer = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.rating_head = nn.Linear(self.hidden_size, 1)

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Attach an ExpertUtilizationTracker from train.py to log gate
        # weights during training; left None (no-op) otherwise.
        self.util_tracker = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Same scheme RecBole's own SASRec uses."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _gate_input(self, item_seq, item_seq_len):
        mask = (item_seq != 0).unsqueeze(-1).float()
        pooled = (self.gate_item_embedding(item_seq) * mask).sum(dim=1)
        return pooled / item_seq_len.clamp(min=1).unsqueeze(-1).float()

    def forward(self, item_seq, item_seq_len):
        """
        Returns:
            next_item_logits: (batch, n_items)
            second_task_pred: (batch,) predicted rating.
            gate_weights: (batch, n_tasks, n_experts), for utilization logging.
        """
        attention_mask = self.get_attention_mask(item_seq)
        expert_outputs = torch.stack(
            [expert(item_seq, item_seq_len, attention_mask) for expert in self.experts],
            dim=1,
        )  # (B, n_experts, H)

        gate_input = self._gate_input(item_seq, item_seq_len)
        task_outputs, gate_weights = self.gate(gate_input, expert_outputs)

        if self.util_tracker is not None and self.training:
            self.util_tracker.update(gate_weights)

        next_item_logits = torch.matmul(
            task_outputs[0], self.item_scorer.weight.transpose(0, 1)
        )
        second_task_pred = self.rating_head(task_outputs[1]).squeeze(-1)
        return next_item_logits, second_task_pred, gate_weights

    def calculate_loss(self, interaction):
        """Combined multi-task loss: next-item CE + rating MSE (+ optional
        load-balancing auxiliary loss). Returned as a 3-tuple rather than a
        single summed scalar: RecBole's Trainer sums tuple-valued losses for
        the backward pass automatically, but logs and tensorboards each
        component separately (train_loss1/2/3) - exactly the per-task loss
        curves the project plan asks for, with no custom training loop."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        rating_label = get_rating_label(interaction, self.RATING)

        next_item_logits, second_task_pred, gate_weights = self.forward(item_seq, item_seq_len)

        task1_loss = self.ce_loss(next_item_logits, pos_items) * self.task_loss_weights[0]
        task2_loss = self.mse_loss(second_task_pred, rating_label) * self.task_loss_weights[1]

        if self.load_balancing_coef > 0:
            aux_loss = load_balancing_loss(gate_weights) * self.load_balancing_coef
        else:
            aux_loss = torch.zeros((), device=next_item_logits.device)

        return task1_loss, task2_loss, aux_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        next_item_logits, _, _ = self.forward(item_seq, item_seq_len)
        return next_item_logits.gather(1, test_item.view(-1, 1)).squeeze(1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_item_logits, _, _ = self.forward(item_seq, item_seq_len)
        return next_item_logits

    def predict_second_task(self, interaction):
        """Rating-regression prediction only. Not part of RecBole's model
        contract - used directly by evaluate.py to score the second task."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        _, second_task_pred, _ = self.forward(item_seq, item_seq_len)
        return second_task_pred
