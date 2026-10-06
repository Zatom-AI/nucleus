import torch
import torch.nn as nn

from nucleus.model.layers.backbone import TransformerBackbone


def _make_local_indices(batch_size: int, seq_len: int, n_keys: int) -> torch.Tensor:
    """Make deterministic local attention indices with self as a valid key."""
    keys = [torch.arange(seq_len)]
    for offset in range(1, n_keys):
        keys.append(torch.arange(seq_len).roll(-offset))
    indices = torch.stack(keys, dim=-1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def test_transformer_backbone_forward_smoke_threads_attention_masks():
    """Smoke test TransformerBackbone shapes and caller-provided mask threading."""
    torch.manual_seed(0)
    B, I, L = 2, 3, 5
    c_atom, c_s = 64, 96
    n_atom_head, n_token_head = 2, 3

    backbone = TransformerBackbone(
        c_atom=c_atom,
        c_s=c_s,
        atom_encoder={
            "n_blocks": 1,
            "n_head": n_atom_head,
            "c_bottleneck": 8,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
            "drop_path": 0.0,
            "attn_backend": "manual",
        },
        atom_decoder={
            "n_blocks": 1,
            "n_head": n_atom_head,
            "c_bottleneck": 8,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
            "drop_path": 0.0,
            "attn_backend": "manual",
        },
        token_trunk={
            "n_blocks": 1,
            "n_head": n_token_head,
            "c_bottleneck": 8,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
            "drop_path": 0.0,
            "attn_backend": "manual",
        },
        downcast={"method": "mean", "n_atoms_per_tok_max": 2},
        upcast={"method": "broadcast", "n_atoms_per_tok_max": 2},
    )
    backbone.eval()

    Q_I = torch.linspace(-0.5, 0.5, steps=B * I * c_s).reshape(B, I, c_s)
    Z_II = torch.linspace(-0.25, 0.25, steps=B * I * I * n_token_head).reshape(
        B, I, I, n_token_head
    )
    S_I = torch.linspace(-1.0, 1.0, steps=B * I * c_s).reshape(B, I, c_s)
    Q_L = torch.linspace(-0.75, 0.75, steps=B * L * c_atom).reshape(B, L, c_atom)
    P_LL = torch.linspace(-0.2, 0.2, steps=B * L * L * n_atom_head).reshape(
        B, L, L, n_atom_head
    )
    C_L = torch.linspace(-1.25, 1.25, steps=B * L * c_atom).reshape(B, L, c_atom)

    G_L = _make_local_indices(B, L, n_keys=3)

    atom_pair_mask = torch.ones(B, L, G_L.shape[-1], dtype=torch.bool)
    atom_pair_mask[:, :, -1] = False
    token_attn_mask = torch.eye(I, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    f = {
        "atom_to_token_map": torch.tensor([0, 0, 1, 2, 2], dtype=torch.long),
        "atom_pair_mask": atom_pair_mask,
        "token_pair_mask": token_attn_mask,
    }

    with torch.no_grad():
        out = backbone(
            Q_I=Q_I,
            Z_II=Z_II,
            S_I=S_I,
            Q_L=Q_L,
            P_LL=P_LL,
            C_L=C_L,
            G_I=None,
            G_L=G_L,
            f=f,
        )

    assert set(out) == {"Q_I", "Q_L"}
    assert out["Q_I"].shape == Q_I.shape
    assert out["Q_L"].shape == Q_L.shape
    assert torch.isfinite(out["Q_I"]).all()
    assert torch.isfinite(out["Q_L"]).all()


def test_transformer_backbone_threads_precomputed_sparse_token_mask():
    """The backbone should pass caller-prepared token graph inputs through unchanged."""

    class PassTransformer(nn.Module):
        def forward(self, Q, **kwargs):
            return Q

    class PassDowncast(nn.Module):
        def forward(self, Q_L, A_I, **kwargs):
            return A_I

    class PassUpcast(nn.Module):
        def forward(self, Q_L, **kwargs):
            return Q_L

    class CaptureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = None
            self.attn_mask = None

        def forward(self, Q, G, attn_mask, **kwargs):
            self.graph = G
            self.attn_mask = attn_mask
            return Q

    backbone = TransformerBackbone.__new__(TransformerBackbone)
    nn.Module.__init__(backbone)
    backbone.atom_encoder = PassTransformer()
    backbone.downcast = PassDowncast()
    backbone.token_trunk = CaptureTransformer()
    backbone.upcast = PassUpcast()
    backbone.atom_decoder = PassTransformer()

    batch_size = 2
    token_graph = torch.tensor(
        [[[0, 1, 3], [1, 0, 2], [2, 3, 0], [3, 2, 1]]], dtype=torch.long
    ).expand(batch_size, -1, -1)
    token_attn_mask = torch.tensor(
        [
            [
                [True, False, False],
                [True, True, False],
                [True, True, False],
                [True, False, True],
            ],
            [
                [True, True, False],
                [True, False, False],
                [True, False, False],
                [True, True, True],
            ],
        ],
        dtype=torch.bool,
    )
    f = {
        "atom_to_token_map": torch.arange(4),
        "atom_pair_mask": torch.ones((batch_size, 4, 4), dtype=torch.bool),
        "token_pair_mask": token_attn_mask,
    }

    backbone(
        Q_I=torch.zeros((batch_size, 4, 2)),
        Z_II=torch.zeros((batch_size, 4, 4, 1)),
        S_I=torch.zeros((batch_size, 4, 2)),
        Q_L=torch.zeros((batch_size, 4, 2)),
        P_LL=torch.zeros((batch_size, 4, 4, 1)),
        C_L=torch.zeros((batch_size, 4, 2)),
        G_I=token_graph,
        G_L=None,
        f=f,
    )

    assert backbone.token_trunk.graph is token_graph
    assert backbone.token_trunk.attn_mask is token_attn_mask
