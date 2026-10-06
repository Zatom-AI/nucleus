"""LigandMPNN model architecture, vendored from Dauparas et al. (2024).

Ported from `model_utils.py` in https://github.com/dauparas/LigandMPNN (MIT
License, Copyright (c) 2024 Justas Dauparas). Only the `model_type ==
"ligand_mpnn"` path is vendored — the plain `protein_mpnn`/`soluble_mpnn`
path in that repo shares byte-identical architecture with (and loads
directly into) `oracles.inverse_folders._protein_mpnn.model.ProteinMPNN`,
so it isn't duplicated here.

Trimmed relative to upstream: the symmetric/tied-multimer decoding branch,
`single_aa_score`, and `score` were dropped — zevals only needs unconstrained
full-backbone sampling (one structure, every residue designed, no symmetry
ties), matching the trimming already applied to `_protein_mpnn`.

Shared transformer layers (`EncLayer`, `DecLayer`, `PositionWiseFeedForward`,
`PositionalEncodings`, and the `gather_*`/`cat_neighbors_nodes` helpers) are
verified byte-identical between the two upstream repos and are imported from
`_protein_mpnn.model` rather than re-vendored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zevals.oracles.inverse_folders._protein_mpnn.model import (
    DecLayer,
    EncLayer,
    PositionalEncodings,
    PositionWiseFeedForward,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)

# ---------------------------------------------------------------------------
# Joint (ligand-atom-context) decoder layer
# ---------------------------------------------------------------------------


class DecLayerJ(nn.Module):
    """Like `DecLayer`, but `h_V`/`h_E` carry an extra per-residue "atoms"
    axis (used to update per-residue ligand-context atom node embeddings
    against their pairwise edge embeddings)."""

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
        h_v_expand = h_V.unsqueeze(-2).expand(-1, -1, -1, h_E.size(-2), -1)
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
# Ligand-context structure featurization
# ---------------------------------------------------------------------------


class ProteinFeaturesLigand(nn.Module):
    """Backbone + nearest-ligand-atom-context graph features for LigandMPNN.

    Side-chain-atoms-as-context (`use_side_chains=True` upstream) is not
    supported here — zevals only uses backbone + explicit ligand HETATM
    context, matching the default LigandMPNN inference mode.
    """

    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
        atom_context_num: int = 16,
    ) -> None:
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.atom_context_num = atom_context_num

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.node_project_down = nn.Linear(
            5 * num_rbf + 64 + 4, node_features, bias=True
        )
        self.norm_nodes = nn.LayerNorm(node_features)

        self.type_linear = nn.Linear(147, 64)
        self.y_nodes = nn.Linear(147, node_features, bias=False)
        self.y_edges = nn.Linear(num_rbf, node_features, bias=False)
        self.norm_y_edges = nn.LayerNorm(node_features)
        self.norm_y_nodes = nn.LayerNorm(node_features)

        # Periodic-table group (row 1) / period (row 2) per atomic number,
        # indexed 0..118 (0 = unknown/none). Used to build a coarse
        # (group, period) one-hot on top of the raw element one-hot so the
        # model generalises across rarely-seen elements.
        self.register_buffer(
            "periodic_table_features",
            torch.tensor(_PERIODIC_TABLE_GROUP_PERIOD, dtype=torch.long),
            persistent=False,
        )

    def _make_angle_features(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        v1 = A - B
        v2 = C - B
        e1 = F.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        return torch.cat(
            [f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1
        )

    def _dist(self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(
            [1, 1, 1, -1]
        )
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

    def _get_rbf(
        self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor
    ) -> torch.Tensor:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(self, input_features: dict) -> tuple[torch.Tensor, ...]:
        Y = input_features["Y"]
        Y_m = input_features["Y_m"]
        Y_t = input_features["Y_t"]
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            Y = Y + self.augment_eps * torch.randn_like(Y)

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = [self._rbf(D_neighbors)]
        for A, B in [
            (N, N), (C, C), (O, O), (Cb, Cb),
            (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
            (N, C), (N, O), (N, Cb), (Cb, C), (Cb, O), (O, C),
            (N, Ca), (C, Ca), (O, Ca), (Cb, Ca), (C, N), (O, N),
            (Cb, N), (C, Cb), (O, Cb), (C, O),
        ]:  # fmt: skip
            RBF_all.append(self._get_rbf(A, B, E_idx))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        Y_t = Y_t.long()
        Y_t_g = self.periodic_table_features[1][Y_t]  # group; 19 categories incl. 0
        Y_t_p = self.periodic_table_features[2][Y_t]  # period; 8 categories incl. 0

        Y_t_g_1hot_ = F.one_hot(Y_t_g, 19)
        Y_t_p_1hot_ = F.one_hot(Y_t_p, 8)
        Y_t_1hot_ = F.one_hot(Y_t, 120)
        Y_t_1hot_ = torch.cat(
            [Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1
        )  # [B, L, M, 147]
        Y_t_1hot = self.type_linear(Y_t_1hot_.float())

        D_N_Y = self._rbf(torch.sqrt(torch.sum((N[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_Ca_Y = self._rbf(
            torch.sqrt(torch.sum((Ca[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )
        D_C_Y = self._rbf(torch.sqrt(torch.sum((C[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_O_Y = self._rbf(torch.sqrt(torch.sum((O[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_Cb_Y = self._rbf(
            torch.sqrt(torch.sum((Cb[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )

        f_angles = self._make_angle_features(N, Ca, C, Y)  # [B, L, M, 4]

        D_all = torch.cat(
            (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), dim=-1
        )
        V = self.node_project_down(D_all)
        V = self.norm_nodes(V)

        Y_edges = self._rbf(
            torch.sqrt(
                torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
            )
        )
        Y_edges = self.norm_y_edges(self.y_edges(Y_edges))
        Y_nodes = self.norm_y_nodes(self.y_nodes(Y_t_1hot_.float()))

        return V, E, E_idx, Y_nodes, Y_edges, Y_m


# ---------------------------------------------------------------------------
# LigandMPNN
# ---------------------------------------------------------------------------


class LigandMPNN(nn.Module):
    """Ligand-context-conditioned autoregressive sequence design model.

    Same architecture/parameter names as upstream's `ProteinMPNN` with
    `model_type="ligand_mpnn"`, so checkpoints load via `load_state_dict`
    without remapping. Only unconstrained full-backbone sampling (no
    symmetry ties, no fixed positions) is implemented — see module
    docstring.
    """

    def __init__(
        self,
        num_letters: int = 21,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 32,
        augment_eps: float = 0.0,
        dropout: float = 0.0,
        atom_context_num: int = 25,
    ) -> None:
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeaturesLigand(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
            atom_context_num=atom_context_num,
        )
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_nodes_y = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_edges_y = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_C_norm = nn.LayerNorm(hidden_dim)
        self.context_encoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(2)]
        )
        self.y_context_encoder_layers = nn.ModuleList(
            [DecLayerJ(hidden_dim, hidden_dim, dropout=dropout) for _ in range(2)]
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.dropout = nn.Dropout(dropout)

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

    def encode(
        self, feature_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = feature_dict["mask"]
        device = mask.device

        V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(feature_dict)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)
        for i in range(len(self.context_encoder_layers)):
            Y_nodes = self.y_context_encoder_layers[i](Y_nodes, Y_edges, Y_m, Y_m_edges)
            h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
            h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, mask, Y_m)

        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
        return h_V, h_E, E_idx

    @torch.no_grad()
    def sample(self, feature_dict: dict) -> dict[str, torch.Tensor]:
        """Autoregressively sample `feature_dict["batch_size"]` sequences.

        `feature_dict` keys (all batch-size-1 except `randn`/`batch_size`):
          X, S, mask, chain_mask, bias [1,L,21], R_idx, chain_labels,
          Y, Y_t, Y_m (ligand context), randn [batch_size, L], temperature,
          batch_size.
        """
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]
        chain_mask = feature_dict["chain_mask"]
        bias = feature_dict["bias"]
        randn = feature_dict["randn"]
        temperature = feature_dict["temperature"]

        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask * chain_mask
        decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))

        E_idx = E_idx.repeat(B_decoder, 1, 1)
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(L, L, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([B, L, 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        chain_mask = chain_mask.repeat(B_decoder, 1)
        mask = mask.repeat(B_decoder, 1)
        bias = bias.repeat(B_decoder, 1, 1)

        all_probs = torch.zeros((B_decoder, L, 20), device=device, dtype=torch.float32)
        all_log_probs = torch.zeros(
            (B_decoder, L, 21), device=device, dtype=torch.float32
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(L):
            t = decoding_order[:, t_]
            chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]
            mask_t = torch.gather(mask, 1, t[:, None])[:, 0]
            bias_t = torch.gather(bias, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

            E_idx_t = torch.gather(
                E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
            )
            h_E_t = torch.gather(
                h_E,
                1,
                t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
            )
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(
                h_EXV_encoder_fw,
                1,
                t[:, None, None, None].repeat(
                    1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                ),
            )
            mask_bw_t = torch.gather(
                mask_bw,
                1,
                t[:, None, None, None].repeat(
                    1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                ),
            )

            for l, layer in enumerate(self.decoder_layers):
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(
                    h_V_stack[l],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                )
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                h_V_stack[l + 1].scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=mask_t),
                )

            h_V_t = torch.gather(
                h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1])
            )[:, 0]
            logits = self.W_out(h_V_t)
            log_probs = F.log_softmax(logits, dim=-1)

            probs = F.softmax((logits + bias_t) / temperature, dim=-1)
            probs_sample = probs[:, :20] / torch.sum(
                probs[:, :20], dim=-1, keepdim=True
            )
            S_t = torch.multinomial(probs_sample, 1)[:, 0]

            all_probs.scatter_(
                1,
                t[:, None, None].repeat(1, 1, 20),
                (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
            )
            all_log_probs.scatter_(
                1,
                t[:, None, None].repeat(1, 1, 21),
                (chain_mask_t[:, None, None] * log_probs[:, None, :]).float(),
            )
            S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
            S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
            h_S.scatter_(
                1,
                t[:, None, None].repeat(1, 1, h_S.shape[-1]),
                self.W_s(S_t)[:, None, :],
            )
            S.scatter_(1, t[:, None], S_t[:, None])

        return {
            "S": S,
            "sampling_probs": all_probs,
            "log_probs": all_log_probs,
            "decoding_order": decoding_order,
        }


# Periodic-table (group, period) lookup used by `ProteinFeaturesLigand`,
# indexed by atomic number 0-118 (0 = unknown). Row 0 is the atomic number
# itself (unused directly but kept for index-parity with upstream); row 1 is
# group (1-18, 0=unknown); row 2 is period (1-7, 0=unknown).
_PERIODIC_TABLE_GROUP_PERIOD = [
    list(range(119)),
    [
        0,
        1,
        18,
        1,
        2,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ],  # fmt: skip
    [
        0,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ],  # fmt: skip
]
