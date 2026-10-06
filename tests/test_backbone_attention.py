import pytest
import torch

from nucleus.model.layers.backbone import DiTBlock


def _make_block(attn_backend: str = "manual") -> DiTBlock:
    """Make a DiTBlock with fixed parameters for testing."""
    torch.manual_seed(0)
    block = DiTBlock(
        c_token=8,
        c_bottleneck=4,
        n_head=2,
        mlp_ratio=1.0,
        dropout=0.0,
        drop_path=0.0,
        attn_backend=attn_backend,
    )
    block.eval()
    return block


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Make inputs for testing DiTBlock attention."""
    B, L, C, H = 2, 5, 8, 2
    Q = torch.linspace(-1.0, 1.0, steps=B * L * C).reshape(B, L, C)
    P = torch.linspace(-0.5, 0.5, steps=B * L * L * H).reshape(B, L, L, H)
    return Q, P


def _make_local_indices(batch_size: int, seq_len: int) -> torch.Tensor:
    """Make local indices for testing DiTBlock attention."""
    base = torch.stack(
        [
            torch.arange(seq_len),
            torch.arange(seq_len).roll(-1),
            torch.arange(seq_len).roll(2),
        ],
        dim=-1,
    )
    return base.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def test_ditblock_sparse_attention_matches_dense_attention_on_same_keys():
    """Test that the sparse attention implementation matches the dense attention implementation when using the same keys."""
    block = _make_block()
    Q, P = _make_inputs()
    B, L, _ = Q.shape
    G = _make_local_indices(B, L)

    dense_mask = torch.zeros(B, L, L, dtype=torch.bool)
    dense_mask.scatter_(2, G, True)
    sparse_mask = dense_mask.gather(2, G)

    dense_out = block._attention(Q, P, G=None, attn_mask=dense_mask, verbose=False)
    sparse_out = block._attention(Q, P, G=G, attn_mask=sparse_mask, verbose=False)

    torch.testing.assert_close(sparse_out, dense_out, rtol=1e-5, atol=1e-6)


def test_ditblock_cueq_attention_matches_manual_outputs_and_gradients():
    """Test that the cueq attention implementation matches the manual attention implementation in outputs and gradients."""
    pytest.importorskip("cuequivariance_torch")

    manual = _make_block(attn_backend="manual")
    cueq = _make_block(attn_backend="cueq")
    cueq.load_state_dict(manual.state_dict())

    Q, P = _make_inputs()
    Q_manual = Q.clone().requires_grad_(True)
    P_manual = P.clone().requires_grad_(True)
    Q_cueq = Q.clone().requires_grad_(True)
    P_cueq = P.clone().requires_grad_(True)
    B, L, _ = Q.shape
    G = _make_local_indices(B, L)
    dense_mask = torch.zeros(B, L, L, dtype=torch.bool)
    dense_mask.scatter_(2, G, True)
    sparse_mask = dense_mask.gather(2, G)

    manual_dense = manual._attention(
        Q_manual, P_manual, G=None, attn_mask=dense_mask, verbose=False
    )
    cueq_dense = cueq._attention(
        Q_cueq, P_cueq, G=None, attn_mask=dense_mask, verbose=False
    )
    torch.testing.assert_close(cueq_dense, manual_dense, rtol=1e-5, atol=1e-6)

    manual_sparse = manual._attention(
        Q_manual, P_manual, G=G, attn_mask=sparse_mask, verbose=False
    )
    cueq_sparse = cueq._attention(
        Q_cueq, P_cueq, G=G, attn_mask=sparse_mask, verbose=False
    )
    torch.testing.assert_close(cueq_sparse, manual_sparse, rtol=1e-5, atol=1e-6)

    manual_loss = manual_dense.square().sum() + manual_sparse.square().sum()
    cueq_loss = cueq_dense.square().sum() + cueq_sparse.square().sum()
    manual_loss.backward()
    cueq_loss.backward()

    torch.testing.assert_close(Q_cueq.grad, Q_manual.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(P_cueq.grad, P_manual.grad, rtol=1e-5, atol=1e-6)
    for manual_param, cueq_param in zip(manual.parameters(), cueq.parameters()):
        if manual_param.grad is not None:
            torch.testing.assert_close(
                cueq_param.grad, manual_param.grad, rtol=1e-5, atol=1e-6
            )
