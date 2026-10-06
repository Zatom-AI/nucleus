"""ProteinMPNN model architecture, vendored from Dauparas et al. (2022).

Ported from `protein_mpnn_utils.py` in
https://github.com/dauparas/ProteinMPNN (MIT License, Copyright (c) 2022
Justas Dauparas). Trimmed to the layers needed for inference (forward +
autoregressive sampling); training-only losses and the JSONL dataset/loader
classes used to train the model were dropped since zevals only ever loads
frozen checkpoints.

State dicts saved by the upstream repo (`ca_model_weights/`,
`soluble_model_weights/`, or a `vanilla_model_weights/` checkpoint pulled
from Zenodo) load into `ProteinMPNN` here unmodified — the module names and
shapes are untouched from the original.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Neighbor-gathering helpers
# ---------------------------------------------------------------------------


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Features [B,N,N,C] at neighbor indices [B,N,K] -> [B,N,K,C]."""
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Features [B,N,C] at neighbor indices [B,N,K] -> [B,N,K,C]."""
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def gather_nodes_t(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Features [B,N,C] at neighbor index [B,K] -> [B,K,C]."""
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)


def cat_neighbors_nodes(
    h_nodes: torch.Tensor, h_neighbors: torch.Tensor, e_idx: torch.Tensor
) -> torch.Tensor:
    h_nodes = gather_nodes(h_nodes, e_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


# ---------------------------------------------------------------------------
# Transformer-style encoder / decoder layers
# ---------------------------------------------------------------------------


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int) -> None:
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.act(self.W_in(h_V)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        return self.linear(d_onehot.float())


class EncLayer(nn.Module):
    def __init__(
        self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        e_idx: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parallel computation of a full transformer layer."""
        h_ev = cat_neighbors_nodes(h_V, h_E, e_idx)
        h_v_expand = h_V.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_ev], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_ev = cat_neighbors_nodes(h_V, h_E, e_idx)
        h_v_expand = h_V.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_ev], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_ev)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(
        self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel computation of a full transformer layer."""
        h_v_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V


# ---------------------------------------------------------------------------
# Structure featurization (backbone graph construction)
# ---------------------------------------------------------------------------


class CAProteinFeatures(nn.Module):
    """Backbone graph features for CA-only (Cα-trace) ProteinMPNN."""

    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 3, num_positional_embeddings + num_rbf * 9 + 7
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)  # unused
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _quaternions(self, R: torch.Tensor) -> torch.Tensor:
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1
                + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
            )
        )

        def _r(i: int, j: int) -> torch.Tensor:
            return R[:, :, :, i, j]

        signs = torch.sign(
            torch.stack(
                [_r(2, 1) - _r(1, 2), _r(0, 2) - _r(2, 0), _r(1, 0) - _r(0, 1)], -1
            )
        )
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        Q = torch.cat((xyz, w), -1)
        return F.normalize(Q, dim=-1)

    def _orientations_coarse(
        self, X: torch.Tensor, e_idx: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)  # exclude CA-CA jumps
        dX = dX * dX_mask[:, :, None]
        U = F.normalize(dX, dim=-1)
        u_2, u_1, u_0 = U[:, :-2, :], U[:, 1:-1, :], U[:, 2:, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        cosA = torch.clamp(-(u_1 * u_0).sum(-1), -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        cosD = torch.clamp((n_2 * n_1).sum(-1), -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        AD_features = torch.stack(
            (torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2
        )
        AD_features = F.pad(AD_features, (0, 0, 1, 2), "constant", 0)

        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=-1)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), "constant", 0)
        O_neighbors = gather_nodes(O, e_idx)
        X_neighbors = gather_nodes(X, e_idx)

        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features

    def _dist(
        self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, e_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), e_idx)
        return D_neighbors, e_idx, mask_neighbors

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(
            [1, 1, 1, -1]
        )
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

    def _get_rbf(
        self, A: torch.Tensor, B: torch.Tensor, e_idx: torch.Tensor
    ) -> torch.Tensor:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], e_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(
        self,
        Ca: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Featurize a Cα-only backbone as an attributed k-NN graph."""
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, e_idx, _ = self._dist(Ca, mask)

        Ca_0 = torch.zeros_like(Ca)
        Ca_2 = torch.zeros_like(Ca)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]

        _, O_features = self._orientations_coarse(Ca, e_idx)

        RBF_all = [self._rbf(D_neighbors)]
        for A, B in [
            (Ca_0, Ca_0),
            (Ca_2, Ca_2),
            (Ca_0, Ca_1),
            (Ca_0, Ca_2),
            (Ca_1, Ca_0),
            (Ca_1, Ca_2),
            (Ca_2, Ca_0),
            (Ca_2, Ca_1),
        ]:
            RBF_all.append(self._get_rbf(A, B, e_idx))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], e_idx)[:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], e_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, e_idx


class ProteinFeatures(nn.Module):
    """Backbone graph features for the full-atom (N, CA, C, O) ProteinMPNN."""

    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(
        self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, e_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, e_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(
            [1, 1, 1, -1]
        )
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

    def _get_rbf(
        self, A: torch.Tensor, B: torch.Tensor, e_idx: torch.Tensor
    ) -> torch.Tensor:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], e_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Featurize a full-atom (N, CA, C, O) backbone as an attributed k-NN graph."""
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca, N, C, O = X[:, :, 1, :], X[:, :, 0, :], X[:, :, 2, :], X[:, :, 3, :]

        D_neighbors, e_idx = self._dist(Ca, mask)

        # D_neighbors is already the CA-CA RBF block; keep 24 extra atom-pair
        # blocks so positional + RBF features match ProteinMPNN checkpoints.
        atom_pairs = [
            (N, N), (C, C), (O, O), (Cb, Cb),
            (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
            (N, C), (N, O), (N, Cb), (Cb, C), (Cb, O), (O, C),
            (N, Ca), (C, Ca), (O, Ca), (Cb, Ca), (C, N), (O, N),
            (Cb, N), (C, Cb), (O, Cb), (C, O),
        ]  # fmt: skip
        RBF_all = [self._rbf(D_neighbors)] + [
            self._get_rbf(A, B, e_idx) for A, B in atom_pairs
        ]
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], e_idx)[:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], e_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, e_idx


# ---------------------------------------------------------------------------
# ProteinMPNN
# ---------------------------------------------------------------------------


class ProteinMPNN(nn.Module):
    """Structure-conditioned autoregressive sequence design model.

    Same architecture/parameter names as the upstream repo so that both the
    vanilla/soluble (`ProteinFeatures`) and CA-only (`CAProteinFeatures`)
    checkpoints load via `load_state_dict` without remapping.
    """

    def __init__(
        self,
        num_letters: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
        ca_only: bool = False,
    ) -> None:
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        if ca_only:
            self.features = CAProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
    ) -> torch.Tensor:
        """Score a fixed (structure, sequence) pair -> per-residue log-probs."""
        E, e_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), e_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, e_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, e_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, e_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, e_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
        mask_size = e_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, e_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, e_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        omit_AAs_np: np.ndarray | None = None,
        bias_AAs_np: np.ndarray | None = None,
        chain_M_pos: torch.Tensor | None = None,
        omit_AA_mask: torch.Tensor | None = None,
        bias_by_res: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressively sample a sequence conditioned on structure `X`.

        Decoding order is randomized per-call via `randn`; positions with
        `chain_mask == 0` keep their `S_true` identity (fixed positions).
        """
        device = X.device
        E, e_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), e_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, e_idx, mask, mask_attend)

        chain_mask = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))
        mask_size = e_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, e_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float32
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, e_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, e_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
            mask_gathered = torch.gather(mask, 1, t[:, None])
            bias_by_res_gathered = torch.gather(
                bias_by_res, 1, t[:, None, None].repeat(1, 1, 21)
            )[:, 0, :]
            if (mask_gathered == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                e_idx_t = torch.gather(
                    e_idx, 1, t[:, None, None].repeat(1, 1, e_idx.shape[-1])
                )
                h_E_t = torch.gather(
                    h_E,
                    1,
                    t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
                )
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, e_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, e_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                    )
                    h_ESV_t = (
                        torch.gather(
                            mask_bw,
                            1,
                            t[:, None, None, None].repeat(
                                1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                            ),
                        )
                        * h_ESV_decoder_t
                        + h_EXV_encoder_t
                    )
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )
                h_V_t = torch.gather(
                    h_V_stack[-1],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                )[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(
                    logits
                    - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res_gathered / temperature,
                    dim=-1,
                )
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(
                        omit_AA_mask,
                        1,
                        t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]),
                    )[:, 0]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 21),
                    (chain_mask_gathered[:, :, None] * probs[:, None, :]).float(),
                )
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (
                S_t * chain_mask_gathered
                + S_true_gathered * (1.0 - chain_mask_gathered)
            ).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)
        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}
