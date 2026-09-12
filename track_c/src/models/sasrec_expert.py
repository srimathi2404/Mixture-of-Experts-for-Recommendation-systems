"""Wraps a SASRec-style self-attention encoder as a single MoSE "expert".

An expert's contract (matches Track B's GRU4Rec-expert contract, so the
surrounding MMoE gate code is identical between tracks): sequence in
(item id list + sequence length), one summary vector out per user. Each
expert owns its own item/position embeddings and transformer stack - the
experts are fully independent towers, mixed only afterwards by the MMoE gate
(src/models/mmoe_gate.py), the same way Track B's parallel GRU4Rec experts
are independent LSTM towers.

Built on top of RecBole's `TransformerEncoder` (recbole.model.layers), the
same building block RecBole's own SASRec model uses - only the surrounding
plumbing (parallel experts, no prediction head here) is specific to Track C.
"""

import torch
from torch import nn

from recbole.model.layers import TransformerEncoder


class SASRecExpert(nn.Module):
    """Self-attention sequence encoder producing one summary vector per user.

    Args:
        n_items: vocabulary size for the item embedding table (RecBole
            reserves id 0 as padding).
        hidden_size: embedding / transformer hidden dimension.
        n_layers: number of transformer encoder layers.
        n_heads: number of self-attention heads.
        inner_size: feed-forward inner dimension inside each transformer layer.
        hidden_dropout_prob, attn_dropout_prob: dropout rates.
        hidden_act: activation used in the transformer's feed-forward blocks.
        layer_norm_eps: epsilon for LayerNorm.
        max_seq_length: max user history length (padding/truncation length).
        use_positional_embedding: if False, the position embedding is never
            added to the item embedding (extra experiment #2: positional
            encoding ablation). The position-embedding table is still
            created either way, so checkpoints have a stable shape
            regardless of this flag.
    """

    def __init__(
        self,
        n_items: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        inner_size: int = 256,
        hidden_dropout_prob: float = 0.5,
        attn_dropout_prob: float = 0.5,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        max_seq_length: int = 50,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        self.use_positional_embedding = use_positional_embedding

        self.item_embedding = nn.Embedding(n_items, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Attention-weight capture (extra experiment #1: attention
        # visualization). Off by default - has no effect on forward() unless
        # enable_attention_capture() was called.
        self._attention_hooks = []
        self._captured_attention = []

    def forward(self, item_seq, item_seq_len, attention_mask):
        """
        Args:
            item_seq: (batch, max_seq_length) padded item id sequences.
            item_seq_len: (batch,) true length of each sequence.
            attention_mask: extended causal attention mask, as produced by
                `SequentialRecommender.get_attention_mask` (shared across all
                experts since it only depends on the padding pattern).

        Returns:
            (batch, hidden_size) one summary vector per user, taken at the
            last valid (non-padded) position of the transformer output.
        """
        item_emb = self.item_embedding(item_seq)
        if self.use_positional_embedding:
            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=item_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            item_emb = item_emb + self.position_embedding(position_ids)

        input_emb = self.dropout(self.LayerNorm(item_emb))
        trm_output = self.trm_encoder(
            input_emb, attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]

        gather_index = (item_seq_len - 1).view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=gather_index).squeeze(1)

    # ------------------------------------------------------------------
    # Attention-weight capture (extra experiment #1).
    #
    # RecBole's MultiHeadAttention computes attention_probs internally but
    # doesn't return them. Rather than forking that class, we register a
    # forward hook on its `softmax` submodule: the hook fires with the
    # softmax's *output*, which is exactly the (batch, n_heads, seq, seq)
    # attention-probability tensor we want, for free and without touching
    # RecBole internals.
    # ------------------------------------------------------------------
    def enable_attention_capture(self) -> None:
        self.disable_attention_capture()
        for layer in self.trm_encoder.layer:
            handle = layer.multi_head_attention.softmax.register_forward_hook(
                self._capture_hook
            )
            self._attention_hooks.append(handle)

    def disable_attention_capture(self) -> None:
        for handle in self._attention_hooks:
            handle.remove()
        self._attention_hooks = []
        self._captured_attention = []

    def _capture_hook(self, module, inputs, output):
        self._captured_attention.append(output.detach())

    def pop_attention_weights(self):
        """Returns and clears the attention weights captured on the last
        forward() call: a list of length n_layers, each
        (batch, n_heads, seq, seq). Requires enable_attention_capture()
        to have been called first."""
        weights = self._captured_attention
        self._captured_attention = []
        return weights
