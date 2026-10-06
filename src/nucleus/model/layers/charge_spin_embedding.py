"""Charge and spin embeddings used by AllScAIP/UMA.

This module is adapted from fairchem at commit
``a3fa357c35fda8fd0e071ed6f823c3aa69d8d364``. The original implementation is
MIT licensed by Meta Platforms, Inc. and affiliates.
"""

from typing import Literal

import torch
import torch.nn as nn


class ChgSpinEmbedding(nn.Module):
    """Embed integer-valued system charge or spin metadata.

    This is the charge/spin embedding module used by AllScAIP and UMA. Zatom
    uses its ``rand_emb`` mode, matching AllScAIP exactly.
    """

    def __init__(
        self,
        embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"],
        embedding_target: Literal["charge", "spin"],
        embedding_size: int,
        grad: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type

        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target
        assert embedding_size % 2 == 0, f"{embedding_size=} must be even"

        if self.embedding_target == "charge":
            # Charge range: -100 to +100 → indices 0 to 200
            self._index_offset = 100
            self._num_embeddings = 201
        elif self.embedding_target == "spin":
            # Spin range: 0 to 100 → indices 0 to 100
            self._index_offset = 0
            self._num_embeddings = 101

        if self.embedding_type == "pos_emb":
            # dividing by 2 because x_proj multiplies by 2
            if not grad:
                self.W = nn.Parameter(
                    torch.randn(embedding_size // 2) * scale, requires_grad=False
                )
            else:
                self.W = nn.Parameter(
                    torch.randn(embedding_size // 2) * scale, requires_grad=True
                )
        elif self.embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(in_features=1, out_features=embedding_size)
            if not grad:
                for param in self.lin_emb.parameters():
                    param.requires_grad = False
        elif self.embedding_type == "rand_emb":
            self.rand_emb = nn.Embedding(self._num_embeddings, embedding_size)
            if not grad:
                for param in self.rand_emb.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"embedding type {self.embedding_type} not implemented")

    def forward(self, x):
        """Embed charge or spin values."""
        # null token for spin is 0
        # charge is default 0
        if self.embedding_type == "pos_emb":
            x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
            if self.embedding_target == "charge":
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            if self.embedding_target == "spin":
                zero_idxs = torch.where(x == 0)[0]
                emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
                # this sets the null spin embedding to zero
                emb[zero_idxs] = 0
                return emb
        elif self.embedding_type == "lin_emb":
            if self.embedding_target == "spin":
                x[x == 0] = -100
            return self.lin_emb(x.unsqueeze(-1).float())
        elif self.embedding_type == "rand_emb":
            # Use tensor arithmetic instead of dict lookup (compile-friendly)
            indices = x + self._index_offset
            return self.rand_emb(indices.long())
        raise ValueError(f"embedding type {self.embedding_type} not implemented")
