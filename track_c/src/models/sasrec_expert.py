"""Wraps RecBole's SASRec self-attention encoder as a single MoSE "expert".

An expert's contract (matches Track B's GRU4Rec-expert contract, so the
surrounding MMoE gate code is identical between tracks): sequence in
(item id list + sequence length), one summary vector out per user.

TODO: implement using recbole.model.sequential_recommender.sasrec building
blocks (embedding layer + TransformerEncoder), returning the pooled hidden
state at the last valid position of each sequence (RecBole's `gather_indexes`
pattern) instead of SASRec's own prediction head.
"""

from torch import nn


class SASRecExpert(nn.Module):
    """Self-attention sequence encoder producing one summary vector per user.

    Args:
        n_items: vocabulary size for the item embedding table.
        hidden_size: embedding / transformer hidden dimension.
        n_layers: number of transformer encoder layers.
        n_heads: number of self-attention heads.
        inner_size: feed-forward inner dimension inside each transformer layer.
        hidden_dropout_prob, attn_dropout_prob: dropout rates.
        max_seq_length: max user history length (padding/truncation length).
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
        max_seq_length: int = 50,
    ):
        super().__init__()
        # TODO: item embedding + positional embedding
        # TODO: RecBole TransformerEncoder stack
        # TODO: LayerNorm + dropout on the embedded sequence, per SASRec
        raise NotImplementedError("Track C: implement the self-attention expert")

    def forward(self, item_seq, item_seq_len):
        """
        Args:
            item_seq: (batch, max_seq_length) padded item id sequences.
            item_seq_len: (batch,) true length of each sequence.

        Returns:
            (batch, hidden_size) one summary vector per user, taken at the
            last valid (non-padded) position of the transformer output.
        """
        raise NotImplementedError("Track C: implement the self-attention expert")
