"""NA-MPNN model architecture (joint protein + DNA/RNA inverse folding),
vendored from Kubaney et al. (2025).

Ported from `inference/model_utils.py` in
https://github.com/baker-laboratory/NA-MPNN (MIT License, Copyright (c) 2025
Andrew Kubaney). Only unconstrained (no symmetry ties, no pair-bias) full-
backbone sampling is vendored — `score`, `unconditional_probs`, and the
tied-decoding branch of `sample` were dropped, matching the trimming already
applied to `_protein_mpnn`/`_ligand_mpnn`.

Shared transformer layers (`EncLayer`, `DecLayer`, `PositionalEncodings`,
`gather_edges`, `gather_nodes`, `cat_neighbors_nodes`) are verified
byte-identical to `_protein_mpnn.model` and imported from there rather than
re-vendored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zevals.oracles.inverse_folders._protein_mpnn.model import (
    DecLayer,
    EncLayer,
    PositionalEncodings,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)

# ---------------------------------------------------------------------------
# Joint protein + nucleic-acid structure featurization
# ---------------------------------------------------------------------------


class ProteinFeaturesNA(nn.Module):
    """Joint protein + DNA/RNA backbone graph features.

    Every residue contributes one reference atom (Cα for protein, C1' for
    nucleic acid) for k-NN graph construction, plus two predicted virtual
    atoms used purely as extra RBF anchor points: a Cβ (from N/CA/C, protein
    residues only) and an "N_na" (from O4'/C1'/C2', nucleic-acid residues
    only) — both built with the same plane-offset construction, just fit to
    different training targets.
    """

    def __init__(
        self,
        edge_features: int,
        node_features: int,
        atom_dict: dict[str, int],
        polytype_to_int: dict[str, int],
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
        na_ref_atom: str = "C1'",
    ) -> None:
        super().__init__()
        self.N_idx = atom_dict["N"]
        self.CA_idx = atom_dict["CA"]
        self.C_idx = atom_dict["C"]
        self.O4prime_idx = atom_dict["O4'"]
        self.C1prime_idx = atom_dict["C1'"]
        self.C2prime_idx = atom_dict["C2'"]
        self.na_ref_atom_idx = atom_dict[na_ref_atom]

        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.num_polytypes = len(polytype_to_int)

        self.node_embedding = nn.Linear(self.num_polytypes, node_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)

        total_atoms = len(atom_dict) + 2  # + predicted Cb, + predicted N_na
        edge_in = num_positional_embeddings + num_rbf * total_atoms * total_atoms
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
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(
            [1, 1, 1, 1, 1, -1]
        )
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

    def _get_all_rbf(
        self, X: torch.Tensor, E_idx: torch.Tensor, X_m: torch.Tensor
    ) -> torch.Tensor:
        # [B,L,A,3] => [B,L,A*3] => [B,L,K,A*3] => [B,L,K,A,3] => [B,L,K,A,A]
        X_flat = X.reshape((X.shape[0], X.shape[1], -1))
        X_flat_g = gather_nodes(X_flat, E_idx)
        X_g = X_flat_g.reshape(list(X_flat_g.shape)[:-1] + list(X.shape[-2:]))
        D = torch.sqrt(
            torch.sum((X[:, :, None, :, None, :] - X_g[:, :, :, None, :, :]) ** 2, -1)
            + 1e-6
        )
        RBF_all = self._rbf(D)
        X_m_gathered = gather_nodes(X_m, E_idx)
        RBF_all = (
            RBF_all
            * X_m[:, :, None, :, None, None]
            * X_m_gathered[:, :, :, None, :, None]
        )
        return RBF_all.view([X.shape[0], X.shape[1], E_idx.shape[2], -1])

    def _virtual_atom(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        w_a: float,
        w_b: float,
        w_c: float,
    ) -> torch.Tensor:
        """Same plane-offset construction ProteinMPNN uses for its virtual
        Cβ, generalised to whichever three atoms/weights are passed in."""
        b = B - A
        c = C - B
        a = torch.cross(b, c, dim=-1)
        return w_a * a + w_b * b + w_c * c + B

    def forward(
        self, feature_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = feature_dict["X"]
        mask = feature_dict["mask"]
        R_idx = feature_dict["R_idx"]
        chain_labels = feature_dict["chain_labels"]
        X_m = feature_dict["X_m"]
        protein_mask = feature_dict["protein_mask"]
        dna_mask = feature_dict["dna_mask"]
        rna_mask = feature_dict["rna_mask"]
        R_polymer_type = feature_dict["R_polymer_type"]

        if self.augment_eps > 0:
            augment_eps = (protein_mask + dna_mask + rna_mask) * self.augment_eps
            X = X + X_m[:, :, :, None] * augment_eps[
                :, :, None, None
            ] * torch.randn_like(X)

        Ca = X[:, :, self.CA_idx, :]
        N = X[:, :, self.N_idx, :]
        C = X[:, :, self.C_idx, :]
        Cb = self._virtual_atom(N, Ca, C, -0.58273431, 0.56802827, -0.54067466)

        na_ref_atom = X[:, :, self.na_ref_atom_idx, :]
        O4prime = X[:, :, self.O4prime_idx, :]
        C1prime = X[:, :, self.C1prime_idx, :]
        C2prime = X[:, :, self.C2prime_idx, :]
        N_na = self._virtual_atom(
            O4prime, C1prime, C2prime, -0.56967352, 0.51055973, -0.53122153
        )

        augmented_X = torch.cat((X, Cb[:, :, None, :], N_na[:, :, None, :]), -2)
        augmented_X_m = torch.cat(
            (X_m, protein_mask[:, :, None], (rna_mask + dna_mask)[:, :, None]), -1
        )

        # Ca and na_ref_atom are disjoint per residue (each residue is
        # exactly one polymer type), so their sum gives a single reference
        # coordinate per row for k-NN graph construction.
        D_neighbors, E_idx = self._dist(Ca + na_ref_atom, mask)

        RBF_all = self._get_all_rbf(augmented_X, E_idx, augmented_X_m)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        R_polymer_type_one_hot = F.one_hot(
            R_polymer_type, num_classes=self.num_polytypes
        ).float()
        V = self.node_embedding(R_polymer_type_one_hot)
        V = self.norm_nodes(V)

        return V, E, E_idx


# ---------------------------------------------------------------------------
# NAMPNN
# ---------------------------------------------------------------------------


class NAMPNN(nn.Module):
    """Joint protein + DNA/RNA autoregressive sequence design model.

    Same architecture/parameter names as upstream's `model_type="na_mpnn"`
    so both bundled checkpoints (design / specificity) load via
    `load_state_dict` unmodified.
    """

    def __init__(
        self,
        restype_to_int: dict[str, int],
        atom_dict: dict[str, int],
        polytype_to_int: dict[str, int],
        num_letters: int = 33,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 33,
        k_neighbors: int = 32,
        augment_eps: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.restype_to_int = restype_to_int
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_letters = num_letters

        self.features = ProteinFeaturesNA(
            node_features,
            edge_features,
            atom_dict=atom_dict,
            polytype_to_int=polytype_to_int,
            top_k=k_neighbors,
            augment_eps=augment_eps,
        )
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
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
        V, E, E_idx = self.features(feature_dict)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V, h_E, E_idx

    @torch.no_grad()
    def sample(self, feature_dict: dict) -> dict[str, torch.Tensor]:
        """Autoregressively sample `feature_dict["batch_size"]` sequences.

        `feature_dict` keys (all batch-size-1 except `randn`/`batch_size`):
          X, X_m, mask, R_idx, chain_labels, protein_mask, dna_mask,
          rna_mask, R_polymer_type, S, chain_mask, bias [1,L,num_letters],
          randn [batch_size, L], temperature, batch_size.
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

        all_probs = torch.zeros(
            (B_decoder, L, self.num_letters), device=device, dtype=torch.float32
        )
        all_log_probs = torch.zeros(
            (B_decoder, L, self.num_letters), device=device, dtype=torch.float32
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = (self.num_letters - 1) * torch.ones(
            (B_decoder, L), dtype=torch.int64, device=device
        )
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
            bias_t = torch.gather(
                bias, 1, t[:, None, None].repeat(1, 1, self.num_letters)
            )[:, 0, :]

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

            # Belt-and-suspenders: `bias` already drives UNK/DX/RX/MAS/PAD
            # (and A/C/G/U/RX's own unaliased slots) to ~0 probability;
            # zero them outright so floating-point residue can't be sampled.
            probs[:, self.restype_to_int["UNK"]] = 0
            probs[:, self.restype_to_int["DX"]] = 0
            probs[:, self.restype_to_int["RX"]] = 0
            probs[:, self.restype_to_int["MAS"]] = 0
            probs[:, self.restype_to_int["PAD"]] = 0
            probs_sample = probs / torch.sum(probs, dim=-1, keepdim=True)

            S_t = torch.multinomial(probs_sample, 1)[:, 0]

            all_probs.scatter_(
                1,
                t[:, None, None].repeat(1, 1, self.num_letters - 1),
                (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
            )
            all_log_probs.scatter_(
                1,
                t[:, None, None].repeat(1, 1, self.num_letters),
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
