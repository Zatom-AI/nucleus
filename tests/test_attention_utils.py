import pytest
import torch

from nucleus.model.layers.attention import (
    LocalAttentionPairBias,
    indices_to_mask_with_padding,
    infer_attention_mask_from_features,
    prepare_token_attention_inputs,
    sparse_pairbias_attention,
)
from nucleus.model.layers.block_utils import (
    create_attention_indices,
    extend_index_mask_with_neighbours,
    get_sparse_attention_indices,
)
from nucleus.model.layers.blocks import LocalTokenTransformer
from nucleus.utils.batching import SparseCollate


def _make_features():
    """Create a dictionary of features for testing."""
    tok_idx = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    unindexing_pair_mask = torch.tensor(
        [
            [False, False, True],
            [False, False, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )
    return {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": unindexing_pair_mask,
        "restype": torch.zeros(3, dtype=torch.long),
    }


def _make_sparse_example(atom_coords, atom_to_token_map, example_id):
    """Create a sparse example dictionary for testing."""
    l = atom_coords.shape[0]
    i = int(atom_to_token_map.max().item()) + 1
    return {
        "example_id": example_id,
        "t": torch.tensor([0], dtype=torch.long),
        "coord_atom_lvl_to_be_noised": atom_coords.clone(),
        "noise": torch.zeros_like(atom_coords),
        "feats": {
            "atom_to_token_map": atom_to_token_map.clone(),
            "unindexing_pair_mask": torch.zeros((i, i), dtype=torch.bool),
            "restype": torch.zeros(i, dtype=torch.long),
        },
        "ground_truth": {
            "coord_atom_lvl": atom_coords.clone(),
            "mask_atom_lvl": torch.ones(l, dtype=torch.bool),
            "sequence_gt_I": torch.zeros((i, 32), dtype=torch.float32),
            "sequence_valid_mask": torch.ones(i, dtype=torch.bool),
            "is_original_unindexed_token": torch.zeros(i, dtype=torch.bool),
        },
    }


def _set_deterministic_attention_weights(module):
    """Initialize LocalAttentionPairBias weights to a simple deterministic pattern."""
    c_a = module.to_q.weight.shape[0]
    c_pair = module.to_b.weight.shape[1]
    n_head = module.n_head

    with torch.no_grad():
        eye = torch.eye(
            c_a, device=module.to_q.weight.device, dtype=module.to_q.weight.dtype
        )
        module.to_q.weight.copy_(eye)
        module.to_k.weight.copy_(eye)
        module.to_v.weight.copy_(eye)
        module.to_g[0].weight.copy_(eye)
        module.to_o.weight.copy_(eye)
        module.to_b.weight.zero_()
        module.to_b.weight[: min(n_head, c_pair), : min(n_head, c_pair)] = torch.eye(
            min(n_head, c_pair),
            device=module.to_b.weight.device,
            dtype=module.to_b.weight.dtype,
        )


def test_local_token_transformer_can_skip_sparse_index_construction(monkeypatch):
    """Dense token attention mode should not build sparse/kNN indices."""
    transformer = LocalTokenTransformer(
        c_token=4,
        c_tokenpair=2,
        c_s=4,
        n_block=1,
        diffusion_transformer_block={
            "attn_backend": "manual",
            "dropout": 0.0,
            "no_residual_connection_between_attention_and_transition": False,
            "n_head": 2,
            "kq_norm": False,
        },
        n_local_tokens=2,
        n_keys=3,
    )
    transformer.use_checkpointing = False
    a_i = torch.randn(2, 5, 4)
    s_i = torch.randn(2, 5, 4)
    z_ii = torch.randn(2, 5, 5, 2)
    x_l = torch.randn(2, 5, 3)
    f = {
        "atom_to_token_map": torch.arange(5),
        "unindexing_pair_mask": torch.tensor(
            [
                [False, False, True, True, True],
                [False, False, True, True, True],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
            ],
            dtype=torch.bool,
        ),
    }

    def fail_create_attention_indices(*args, **kwargs):
        raise AssertionError("create_attention_indices should not be called")

    monkeypatch.setattr(
        "nucleus.model.layers.blocks.create_attention_indices",
        fail_create_attention_indices,
    )

    out = transformer(
        a_i,
        s_i,
        z_ii,
        f=f,
        X_L=x_l,
        full=False,
        use_local_attention=False,
    )

    tok_idx = torch.arange(5)
    dense_indices = tok_idx.view(1, 1, -1).expand(2, 5, -1)
    expected = transformer.blocks[0](
        a_i,
        s_i,
        z_ii,
        indices=dense_indices,
        f=f,
        tok_idx=tok_idx,
        full=True,
    )
    torch.testing.assert_close(out, expected)


def test_local_token_transformer_builds_sparse_indices_by_default(monkeypatch):
    """Existing sparse/kNN token attention behavior should remain the default."""
    transformer = LocalTokenTransformer(
        c_token=4,
        c_tokenpair=2,
        c_s=4,
        n_block=0,
        diffusion_transformer_block={},
        n_local_tokens=2,
        n_keys=3,
    )
    a_i = torch.randn(2, 5, 4)
    s_i = torch.randn(2, 5, 4)
    z_ii = torch.randn(2, 5, 5, 2)
    x_l = torch.randn(2, 5, 3)
    calls = []

    def fake_create_attention_indices(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            torch.zeros((2, 5, 3), dtype=torch.long),
            torch.ones((2, 5, 3), dtype=torch.bool),
        )

    monkeypatch.setattr(
        "nucleus.model.layers.blocks.create_attention_indices",
        fake_create_attention_indices,
    )

    out = transformer(a_i, s_i, z_ii, f={}, X_L=x_l)

    assert torch.equal(out, a_i)
    assert len(calls) == 1


def test_indices_to_mask_with_padding_handles_batched_and_unbatched_indices():
    """Test that indices_to_mask_with_padding correctly converts 2D and 3D indices to attention masks with padding."""
    indices_2d = torch.tensor(
        [
            [0, 1, 1],
            [0, 2, 3],
            [1, 2, 3],
            [0, 2, 2],
        ],
        dtype=torch.long,
    )
    attn_mask_2d = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )
    expected_2d = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, True],
            [False, True, False, True],
            [False, False, True, False],
        ],
        dtype=torch.bool,
    )

    mask_2d = indices_to_mask_with_padding(indices_2d, attn_mask_2d)
    assert torch.equal(mask_2d, expected_2d)

    indices_3d = torch.tensor(
        [
            [
                [0, 1, 1],
                [0, 2, 3],
                [1, 2, 3],
                [0, 2, 2],
            ],
            [
                [1, 2, 3],
                [0, 0, 1],
                [0, 3, 3],
                [1, 2, 2],
            ],
        ],
        dtype=torch.long,
    )
    attn_mask_3d = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, True],
                [True, False, True],
                [False, True, True],
            ],
            [
                [True, False, True],
                [True, True, False],
                [True, True, True],
                [False, True, True],
            ],
        ],
        dtype=torch.bool,
    )
    mask_3d = indices_to_mask_with_padding(indices_3d, attn_mask_3d)

    expected_3d = torch.stack(
        [
            expected_2d,
            torch.tensor(
                [
                    [False, True, False, True],
                    [True, False, False, False],
                    [True, False, False, True],
                    [False, False, True, False],
                ],
                dtype=torch.bool,
            ),
        ],
        dim=0,
    )
    assert torch.equal(mask_3d, expected_3d)


def test_inter_chain_attention_backfills_missing_inter_slots():
    """Unused inter slots should take additional intra keys before becoming padding."""
    tok_idx = torch.arange(6, dtype=torch.long)
    chain_ids = torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.long)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros((6, 6), dtype=torch.bool),
        "restype": torch.zeros(6, dtype=torch.long),
        "asym_id": chain_ids,
    }
    X_L = torch.arange(6, dtype=torch.float32).view(1, 6, 1).expand(-1, -1, 3)

    indices, index_mask = create_attention_indices(
        X_L=X_L,
        f=f,
        tok_idx=tok_idx,
        n_attn_keys=6,
        n_attn_seq_neighbours=1,
    )

    assert indices.shape == index_mask.shape == (1, 6, 6)
    assert index_mask.all()
    assert torch.equal(
        infer_attention_mask_from_features(
            f, indices, tok_idx=tok_idx, index_mask=index_mask
        ),
        index_mask,
    )

    for query_idx in range(6):
        real_indices = indices[0, query_idx, index_mask[0, query_idx]]
        assert torch.unique(real_indices).numel() == real_indices.numel()
        selected_chain_ids = chain_ids[real_indices]
        assert (selected_chain_ids == chain_ids[query_idx]).any()
        assert (
            torch.count_nonzero(selected_chain_ids != chain_ids[query_idx])
            == (chain_ids != chain_ids[query_idx]).sum()
        )


def test_attention_key_count_is_capped_for_tiny_multichain_structures():
    """Requests larger than the structure should return each real key once."""
    tok_idx = torch.arange(4, dtype=torch.long)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros((4, 4), dtype=torch.bool),
        "restype": torch.zeros(4, dtype=torch.long),
        "asym_id": torch.arange(4, dtype=torch.long),
    }
    positions = torch.eye(4, 3).unsqueeze(0)

    indices, index_mask = create_attention_indices(
        X_L=positions,
        f=f,
        tok_idx=tok_idx,
        n_attn_keys=128,
        n_attn_seq_neighbours=1,
    )

    assert indices.shape == index_mask.shape == (1, 4, 4)
    assert index_mask.all()
    expected_keys = torch.arange(4).expand(4, -1)
    assert torch.equal(torch.sort(indices[0], dim=-1).values, expected_keys)


def test_inter_chain_attention_handles_queries_with_zero_inter_chain_keys():
    """A collated single-chain example should retain an intra key and mask inter slots."""
    tok_idx = torch.arange(6, dtype=torch.long)
    same_example = torch.zeros((6, 6), dtype=torch.bool)
    same_example[:2, :2] = True
    same_example[2:, 2:] = True
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": ~same_example,
        "restype": torch.zeros(6, dtype=torch.long),
        "asym_id": torch.tensor([0, 0, 1, 2, 3, 4], dtype=torch.long),
    }
    X_L = torch.arange(18, dtype=torch.float32).view(1, 6, 3)

    indices, index_mask = create_attention_indices(
        X_L=X_L,
        f=f,
        tok_idx=tok_idx,
        n_attn_keys=4,
        n_attn_seq_neighbours=0,
    )
    attn_mask = infer_attention_mask_from_features(
        f, indices, tok_idx=tok_idx, index_mask=index_mask
    )

    assert torch.equal(index_mask.sum(dim=-1), torch.tensor([[2, 2, 4, 4, 4, 4]]))
    assert torch.equal(attn_mask, index_mask)
    query_examples = torch.tensor([0, 0, 1, 1, 1, 1]).view(1, 6, 1)
    key_examples = torch.tensor([0, 0, 1, 1, 1, 1])[indices]
    assert torch.all(
        key_examples[attn_mask] == query_examples.expand_as(indices)[attn_mask]
    )


def test_inter_chain_attention_backfills_imbalanced_complex():
    """The 1rzt-like 11/5/4/236 topology should use every available graph slot."""
    chain_sizes = torch.tensor([11, 5, 4, 236], dtype=torch.long)
    chain_ids = torch.repeat_interleave(torch.arange(4), chain_sizes)
    tok_idx = torch.arange(chain_ids.numel(), dtype=torch.long)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros((256, 256), dtype=torch.bool),
        "restype": torch.zeros(256, dtype=torch.long),
        "asym_id": chain_ids,
    }
    position_index = torch.arange(256, dtype=torch.float32)
    linear_positions = torch.stack(
        [
            position_index,
            torch.zeros_like(position_index),
            torch.zeros_like(position_index),
        ],
        dim=-1,
    )
    grid_positions = torch.stack(
        [
            position_index.remainder(17),
            torch.div(position_index, 17, rounding_mode="floor"),
            position_index.remainder(5) * 0.25,
        ],
        dim=-1,
    )
    X_L = torch.stack([linear_positions, grid_positions], dim=0)

    indices, index_mask = create_attention_indices(
        X_L=X_L,
        f=f,
        tok_idx=tok_idx,
        n_attn_keys=128,
        n_attn_seq_neighbours=32,
    )

    assert indices.shape == index_mask.shape == (2, 256, 128)
    assert index_mask.all()
    assert not torch.equal(indices[0], indices[1])
    for batch_idx in range(2):
        for query_idx in range(256):
            real_indices = indices[batch_idx, query_idx]
            assert torch.unique(real_indices).numel() == 128
            assert (chain_ids[real_indices] == chain_ids[query_idx]).any()
            assert (chain_ids[real_indices] != chain_ids[query_idx]).any()


def test_create_attention_indices_does_not_call_tensor_item(monkeypatch):
    """Graph construction should not call Tensor.item in the forward path."""
    tok_idx = torch.arange(6, dtype=torch.long)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros((6, 6), dtype=torch.bool),
        "restype": torch.zeros(6, dtype=torch.long),
        "asym_id": torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.long),
    }

    def fail_item(*args, **kwargs):
        raise AssertionError("Tensor.item() must not be used during graph construction")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    indices, index_mask = create_attention_indices(
        X_L=torch.arange(18, dtype=torch.float32).view(1, 6, 3),
        f=f,
        tok_idx=tok_idx,
        n_attn_keys=6,
        n_attn_seq_neighbours=0,
    )

    assert indices.shape == index_mask.shape == (1, 6, 6)


@pytest.mark.parametrize("full", [False, True])
def test_local_attention_pair_bias_ignores_explicitly_padded_slots(full):
    """Masked sparse slots must not affect either sparse or reconstructed full attention."""
    module = LocalAttentionPairBias(
        c_a=4,
        c_s=None,
        c_pair=2,
        n_head=2,
        kq_norm=False,
        attn_backend="manual",
    )
    module.use_checkpointing = False
    _set_deterministic_attention_weights(module)

    q_l = torch.linspace(-0.3, 0.3, steps=12).view(1, 3, 4)
    p_ll = torch.linspace(-0.2, 0.2, steps=18).view(1, 3, 3, 2)
    indices = torch.tensor([[[0, 2], [1, 0], [2, 1]]], dtype=torch.long)
    index_mask = torch.tensor(
        [[[True, False], [True, True], [True, True]]], dtype=torch.bool
    )
    f = {
        "atom_to_token_map": torch.arange(3),
        "unindexing_pair_mask": torch.zeros((3, 3), dtype=torch.bool),
    }

    changed_q_l = q_l.clone()
    changed_q_l[:, 2] += 100.0
    changed_p_ll = p_ll.clone()
    changed_p_ll[:, 0, 2] += 100.0

    baseline = module(
        q_l,
        None,
        p_ll,
        indices=indices,
        index_mask=index_mask,
        f=f,
        full=full,
    )
    changed = module(
        changed_q_l,
        None,
        changed_p_ll,
        indices=indices,
        index_mask=index_mask,
        f=f,
        full=full,
    )
    unmasked = module(
        changed_q_l,
        None,
        changed_p_ll,
        indices=indices,
        f=f,
        full=full,
    )

    torch.testing.assert_close(changed[:, 0], baseline[:, 0])
    assert not torch.allclose(unmasked[:, 0], baseline[:, 0])


def test_local_attention_pair_bias_flex_attention_matches_manual_path():
    """Test that the real FlexAttention path matches the manual path on the same module weights and inputs."""
    pytest.importorskip("torch.nn.attention.flex_attention")

    batch_size = 2
    seq_len = 25
    model_dim = 8
    pair_dim = 4
    n_head = 2

    base_module = LocalAttentionPairBias(
        c_a=model_dim,
        c_s=None,
        c_pair=pair_dim,
        n_head=n_head,
        kq_norm=False,
        attn_backend="manual",
    )
    base_module.use_checkpointing = False
    _set_deterministic_attention_weights(base_module)

    flex_module = LocalAttentionPairBias(
        c_a=model_dim,
        c_s=None,
        c_pair=pair_dim,
        n_head=n_head,
        kq_norm=False,
        attn_backend="flex",
    )
    flex_module.use_checkpointing = False
    flex_module.load_state_dict(base_module.state_dict())

    q_l = torch.linspace(
        -0.4, 0.4, steps=batch_size * seq_len * model_dim, dtype=torch.float32
    ).reshape(batch_size, seq_len, model_dim)
    p_ll = torch.linspace(
        -0.2,
        0.2,
        steps=batch_size * seq_len * seq_len * pair_dim,
        dtype=torch.float32,
    ).reshape(batch_size, seq_len, seq_len, pair_dim)

    tok_idx = torch.arange(seq_len, dtype=torch.long)
    indices = tok_idx.view(1, 1, seq_len).expand(batch_size, seq_len, seq_len).clone()
    a_mask = (
        torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]
    ).abs() <= 2
    a_mask = a_mask.to(dtype=torch.bool)
    index_mask = a_mask.unsqueeze(0).expand(batch_size, -1, -1)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros_like(a_mask),
    }

    manual_out = base_module(
        q_l,
        None,
        p_ll,
        indices=indices,
        index_mask=index_mask,
        f=f,
        tok_idx=tok_idx,
        full=False,
    )
    with torch.no_grad():
        flex_out = flex_module(
            q_l,
            None,
            p_ll,
            indices=indices,
            index_mask=index_mask,
            f=f,
            tok_idx=tok_idx,
            full=False,
        )

    assert manual_out.shape == flex_out.shape == (batch_size, seq_len, model_dim)
    torch.testing.assert_close(
        flex_out.float(),
        manual_out.float(),
        atol=5e-3,
        rtol=5e-3,
    )


def test_local_attention_pair_bias_flex_attention_accepts_unbatched_pair_bias():
    """FlexAttention should handle the unbatched P_LL layout used by token initialization."""
    pytest.importorskip("torch.nn.attention.flex_attention")

    batch_size = 2
    seq_len = 25
    model_dim = 8
    pair_dim = 4
    n_head = 2

    base_module = LocalAttentionPairBias(
        c_a=model_dim,
        c_s=None,
        c_pair=pair_dim,
        n_head=n_head,
        kq_norm=False,
        attn_backend="manual",
    )
    base_module.use_checkpointing = False
    _set_deterministic_attention_weights(base_module)

    flex_module = LocalAttentionPairBias(
        c_a=model_dim,
        c_s=None,
        c_pair=pair_dim,
        n_head=n_head,
        kq_norm=False,
        attn_backend="flex",
    )
    flex_module.use_checkpointing = False
    flex_module.load_state_dict(base_module.state_dict())

    q_l = torch.linspace(
        -0.4, 0.4, steps=batch_size * seq_len * model_dim, dtype=torch.float32
    ).reshape(batch_size, seq_len, model_dim)
    p_ll = torch.linspace(
        -0.2,
        0.2,
        steps=seq_len * seq_len * pair_dim,
        dtype=torch.float32,
    ).reshape(seq_len, seq_len, pair_dim)

    tok_idx = torch.arange(seq_len, dtype=torch.long)
    indices = tok_idx.view(1, 1, seq_len).expand(batch_size, seq_len, seq_len).clone()
    a_mask = (
        torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]
    ).abs() <= 2
    a_mask = a_mask.to(dtype=torch.bool)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": ~a_mask,
    }

    manual_out = base_module(
        q_l,
        None,
        p_ll,
        indices=indices,
        f=f,
        tok_idx=tok_idx,
        full=False,
    )
    with torch.no_grad():
        flex_out = flex_module(
            q_l,
            None,
            p_ll,
            indices=indices,
            f=f,
            tok_idx=tok_idx,
            full=False,
        )

    assert manual_out.shape == flex_out.shape == (batch_size, seq_len, model_dim)
    torch.testing.assert_close(
        flex_out.float(),
        manual_out.float(),
        atol=5e-3,
        rtol=5e-3,
    )


def test_local_attention_pair_bias_reuses_precomputed_flex_block_mask(monkeypatch):
    """Test that a caller-supplied flex_block_mask is reused instead of rebuilt."""
    pytest.importorskip("torch.nn.attention.flex_attention")

    import nucleus.model.layers.attention as attention_module

    batch_size = 1
    seq_len = 25
    model_dim = 8
    pair_dim = 4

    module = LocalAttentionPairBias(
        c_a=model_dim,
        c_s=None,
        c_pair=pair_dim,
        n_head=2,
        kq_norm=False,
        attn_backend="flex",
    )
    module.use_checkpointing = False
    _set_deterministic_attention_weights(module)

    q_l = torch.linspace(
        -0.2, 0.2, steps=batch_size * seq_len * model_dim, dtype=torch.float32
    ).reshape(batch_size, seq_len, model_dim)
    p_ll = torch.linspace(
        -0.1,
        0.1,
        steps=batch_size * seq_len * seq_len * pair_dim,
        dtype=torch.float32,
    ).reshape(batch_size, seq_len, seq_len, pair_dim)

    tok_idx = torch.arange(seq_len, dtype=torch.long)
    indices = tok_idx.view(1, 1, seq_len).expand(batch_size, seq_len, seq_len).clone()
    attn_valid = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool)
    attn_valid[:, 0, :] = False
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": ~attn_valid[0],
    }

    precomputed_block_mask = object()

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("flex_block_mask should have been reused, not rebuilt")

    monkeypatch.setattr(
        attention_module,
        "create_flex_attention_block_mask",
        fail_if_recomputed,
    )
    monkeypatch.setattr(
        attention_module,
        "torch_flex_attention",
        lambda q, k, v, **kwargs: torch.zeros_like(q),
    )

    out_1 = module(
        q_l,
        None,
        p_ll,
        indices=indices,
        f=f,
        tok_idx=tok_idx,
        full=True,
        flex_block_mask=precomputed_block_mask,
    )
    out_2 = module(
        q_l,
        None,
        p_ll,
        indices=indices,
        f=f,
        tok_idx=tok_idx,
        full=True,
        flex_block_mask=precomputed_block_mask,
    )

    assert out_1.shape == out_2.shape == (batch_size, seq_len, model_dim)


def test_infer_attention_mask_from_features_supports_batch_and_single_inputs():
    """Test that infer_attention_mask_from_features correctly infers attention masks for both batch and single inputs."""
    f = _make_features()
    tok_idx = f["atom_to_token_map"]
    indices_2d = torch.tensor(
        [
            [0, 2, 3],
            [1, 3, 0],
            [2, 0, 1],
            [3, 1, 2],
        ],
        dtype=torch.long,
    )
    expected_2d = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [True, False, True],
        ],
        dtype=torch.bool,
    )
    mask_2d = infer_attention_mask_from_features(f, indices_2d, tok_idx=tok_idx)
    assert torch.equal(mask_2d, expected_2d)

    indices_3d = torch.tensor(
        [
            [
                [0, 2, 3],
                [1, 3, 0],
                [2, 0, 1],
                [3, 1, 2],
            ],
            [
                [1, 0, 2],
                [3, 2, 1],
                [0, 3, 1],
                [2, 1, 0],
            ],
        ],
        dtype=torch.long,
    )
    expected_3d = torch.stack(
        [
            expected_2d,
            torch.tensor(
                [
                    [True, True, True],
                    [False, True, True],
                    [True, True, True],
                    [True, False, False],
                ],
                dtype=torch.bool,
            ),
        ],
        dim=0,
    )
    mask_3d = infer_attention_mask_from_features(f, indices_3d, tok_idx=tok_idx)
    assert torch.equal(mask_3d, expected_3d)

    index_mask = torch.tensor(
        [
            [
                [True, True, False],
                [True, False, False],
                [True, True, True],
                [False, False, True],
            ],
            [
                [True, False, False],
                [True, True, False],
                [False, True, False],
                [True, True, True],
            ],
        ],
        dtype=torch.bool,
    )
    combined_mask = infer_attention_mask_from_features(
        f, indices_3d, tok_idx=tok_idx, index_mask=index_mask
    )
    assert torch.equal(combined_mask, expected_3d & index_mask)


def test_extend_index_mask_with_neighbours_matches_per_batch_reference():
    """Test that extend_index_mask_with_neighbours correctly extends the index mask with neighbours."""
    mask = torch.tensor(
        [
            [True, True, False, False, False],
            [False, True, True, False, False],
            [False, False, True, True, False],
            [False, False, False, True, True],
            [True, False, False, False, True],
        ],
        dtype=torch.bool,
    )
    base_mask = ~torch.eye(5, dtype=torch.bool)
    D_LL = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 0.0, 1.5, 2.5, 3.5],
                [2.0, 1.5, 0.0, 1.0, 2.0],
                [3.0, 2.5, 1.0, 0.0, 1.0],
                [4.0, 3.5, 2.0, 1.0, 0.0],
            ],
            [
                [0.0, 4.0, 3.0, 2.0, 1.0],
                [4.0, 0.0, 2.0, 1.0, 3.0],
                [3.0, 2.0, 0.0, 4.0, 1.0],
                [2.0, 1.0, 4.0, 0.0, 3.0],
                [1.0, 3.0, 1.0, 3.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    batched_indices, batched_mask = extend_index_mask_with_neighbours(
        mask, D_LL, k=4, base_mask=base_mask
    )

    refs = [
        extend_index_mask_with_neighbours(mask, D_LL[b], k=4, base_mask=base_mask)
        for b in range(D_LL.shape[0])
    ]
    ref_indices = torch.stack([item[0].squeeze(0) for item in refs], dim=0)
    ref_mask = torch.stack([item[1].squeeze(0) for item in refs], dim=0)

    assert torch.equal(batched_indices, ref_indices)
    assert torch.equal(batched_mask, ref_mask)


def test_get_sparse_attention_indices_matches_per_batch_reference():
    """Test that get_sparse_attention_indices correctly matches per-batch reference."""
    res_idx = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
    base_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ],
        dtype=torch.bool,
    )
    D_LL = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 0.0, 1.0, 2.0, 3.0],
                [2.0, 1.0, 0.0, 1.0, 2.0],
                [3.0, 2.0, 1.0, 0.0, 1.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
            ],
            [
                [0.0, 2.0, 1.0, 4.0, 3.0],
                [2.0, 0.0, 3.0, 1.0, 4.0],
                [1.0, 3.0, 0.0, 2.0, 4.0],
                [4.0, 1.0, 2.0, 0.0, 3.0],
                [3.0, 4.0, 4.0, 3.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    batched_indices, batched_mask = get_sparse_attention_indices(
        res_idx, D_LL, n_seq_neighbours=0, k_max=4, num_res=4, base_mask=base_mask
    )
    refs = [
        get_sparse_attention_indices(
            res_idx,
            D_LL[b : b + 1],
            n_seq_neighbours=0,
            k_max=4,
            num_res=4,
            base_mask=base_mask,
        )
        for b in range(D_LL.shape[0])
    ]
    ref_indices = torch.stack([item[0].squeeze(0) for item in refs], dim=0)
    ref_mask = torch.stack([item[1].squeeze(0) for item in refs], dim=0)

    assert torch.equal(batched_indices, ref_indices)
    assert torch.equal(batched_mask, ref_mask)
    assert batched_indices.shape == (2, 5, 4)


def test_create_attention_indices_matches_per_batch_reference():
    """Test that create_attention_indices correctly matches per-batch reference."""
    f = _make_features()
    tok_idx = f["atom_to_token_map"]
    X_L = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    batched_indices, batched_mask = create_attention_indices(
        f,
        n_attn_keys=3,
        n_attn_seq_neighbours=0,
        X_L=X_L,
        tok_idx=tok_idx,
    )
    refs = [
        create_attention_indices(
            f,
            n_attn_keys=3,
            n_attn_seq_neighbours=0,
            X_L=X_L[b : b + 1],
            tok_idx=tok_idx,
        )
        for b in range(X_L.shape[0])
    ]
    ref_indices = torch.stack([item[0].squeeze(0) for item in refs], dim=0)
    ref_mask = torch.stack([item[1].squeeze(0) for item in refs], dim=0)

    assert torch.equal(batched_indices, ref_indices)
    assert torch.equal(batched_mask, ref_mask)
    assert batched_indices.shape == (2, 4, 3)
    assert not torch.equal(batched_indices[0], batched_indices[1])


def test_prepare_token_attention_inputs_supports_multi_atom_tokens(monkeypatch):
    """Token graphs should reuse their index mask and use token identities."""
    monkeypatch.setattr(
        "nucleus.model.layers.attention.infer_attention_mask_from_features",
        lambda *args, **kwargs: pytest.fail(
            "create_attention_indices already returned the complete validity mask"
        ),
    )
    token_pair_mask = torch.tensor([[True, False], [False, True]])
    f = {
        "atom_to_token_map": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "unindexing_pair_mask": ~token_pair_mask,
        "token_pair_mask": token_pair_mask,
        "restype": torch.zeros(2, dtype=torch.long),
        "asym_id": torch.tensor([0, 1], dtype=torch.long),
    }
    token_positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    indices, sparse_f = prepare_token_attention_inputs(
        f=f,
        X_I=token_positions,
        use_local_attention=True,
        n_attn_keys=2,
        n_attn_seq_neighbours=0,
    )
    attn_mask = sparse_f["token_pair_mask"]

    assert indices.shape == attn_mask.shape == (2, 2, 2)
    assert torch.equal(attn_mask.sum(dim=-1), torch.ones((2, 2), dtype=torch.long))
    query_indices = torch.arange(2).view(1, 2, 1).expand_as(indices)
    assert torch.equal(indices[attn_mask], query_indices[attn_mask])

    dense_indices, dense_f = prepare_token_attention_inputs(
        f=f,
        X_I=token_positions,
        use_local_attention=False,
        n_attn_keys=2,
        n_attn_seq_neighbours=0,
    )
    assert dense_indices is None
    dense_mask = dense_f["token_pair_mask"]
    assert torch.equal(dense_mask, token_pair_mask.unsqueeze(0).expand(2, -1, -1))
    assert sparse_f is not f and dense_f is not f
    assert f["token_pair_mask"] is token_pair_mask


def test_sparse_collate_blocks_cross_example_attention_leakage():
    """Test that SparseCollate correctly blocks cross-example attention leakage."""
    example_a = _make_sparse_example(
        atom_coords=torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [10.00, 0.00, 0.00],
            ],
            dtype=torch.float32,
        ),
        atom_to_token_map=torch.tensor([0, 1], dtype=torch.long),
        example_id="a",
    )
    example_b = _make_sparse_example(
        atom_coords=torch.tensor(
            [
                [0.05, 0.00, 0.00],
                [10.05, 0.00, 0.00],
                [20.05, 0.00, 0.00],
            ],
            dtype=torch.float32,
        ),
        atom_to_token_map=torch.tensor([0, 1, 2], dtype=torch.long),
        example_id="b",
    )

    collated = SparseCollate(max_num_tokens=5)([example_a, example_b])
    feats = collated["feats"]
    assert feats["batch_num_tokens_expanded"] == 5
    torch.testing.assert_close(feats["batch_util_rate"], torch.tensor(1.0))
    batch_atom_lvl = feats["batch_atom_lvl"]
    batch_token_lvl = feats["batch_token_lvl"]
    tok_idx = feats["atom_to_token_map"]
    X_L = collated["ground_truth"]["coord_atom_lvl"]

    dense_token_indices = (
        torch.arange(tok_idx.shape[0], device=tok_idx.device)
        .view(1, -1)
        .expand(tok_idx.shape[0], -1)
    )
    token_mask = infer_attention_mask_from_features(
        feats, dense_token_indices, tok_idx=tok_idx
    )
    assert token_mask.shape == (5, 5)
    assert not token_mask[:2, 2:].any()
    assert not token_mask[2:, :2].any()

    batched_indices, batched_mask = create_attention_indices(
        feats,
        n_attn_keys=2,
        n_attn_seq_neighbours=0,
        X_L=X_L,
        tok_idx=tok_idx,
    )
    batched_indices = batched_indices.squeeze(0)
    batched_mask = batched_mask.squeeze(0)

    ref_a = create_attention_indices(
        example_a["feats"],
        n_attn_keys=2,
        n_attn_seq_neighbours=0,
        X_L=example_a["ground_truth"]["coord_atom_lvl"],
        tok_idx=example_a["feats"]["atom_to_token_map"],
    )
    ref_b = create_attention_indices(
        example_b["feats"],
        n_attn_keys=2,
        n_attn_seq_neighbours=0,
        X_L=example_b["ground_truth"]["coord_atom_lvl"],
        tok_idx=example_b["feats"]["atom_to_token_map"],
    )
    ref_indices = torch.cat(
        [
            ref_a[0].squeeze(0),
            ref_b[0].squeeze(0) + example_a["ground_truth"]["coord_atom_lvl"].shape[0],
        ],
        dim=0,
    )
    ref_mask = torch.cat([ref_a[1].squeeze(0), ref_b[1].squeeze(0)], dim=0)

    assert torch.equal(batched_indices, ref_indices)
    assert torch.equal(batched_mask, ref_mask)

    valid_mask = infer_attention_mask_from_features(
        feats, batched_indices, tok_idx=tok_idx, index_mask=batched_mask
    )
    query_batch = batch_atom_lvl[:, None].expand_as(batched_indices)
    key_batch = batch_atom_lvl[batched_indices]
    assert torch.all(key_batch[valid_mask] == query_batch[valid_mask])

    token_query_batch = batch_token_lvl[tok_idx][:, None].expand_as(batched_indices)
    token_key_batch = batch_token_lvl[tok_idx[batched_indices]]
    assert torch.all(token_key_batch[valid_mask] == token_query_batch[valid_mask])


def test_sparse_collate_shifts_indices_and_blocks_cross_example_pairs():
    """Test that SparseCollate shifts token indices and blocks cross-example pairwise features."""
    example_a = _make_sparse_example(
        atom_coords=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        atom_to_token_map=torch.tensor([0, 1], dtype=torch.long),
        example_id="a",
    )
    example_b = _make_sparse_example(
        atom_coords=torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        atom_to_token_map=torch.tensor([0, 0, 2], dtype=torch.long),
        example_id="b",
    )

    example_a["feats"]["unindexing_pair_mask"] = torch.tensor(
        [[False, True], [True, False]], dtype=torch.bool
    )
    example_b["feats"]["unindexing_pair_mask"] = torch.tensor(
        [
            [False, False, True],
            [False, False, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )
    example_a["feats"]["token_bonds"] = torch.tensor(
        [[True, False], [False, True]], dtype=torch.bool
    )
    example_b["feats"]["token_bonds"] = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, False, True],
        ],
        dtype=torch.bool,
    )

    collated = SparseCollate()([example_a, example_b])

    assert torch.equal(
        collated["feats"]["batch_atom_lvl"], torch.tensor([0, 0, 1, 1, 1])
    )
    assert torch.equal(
        collated["feats"]["batch_token_lvl"], torch.tensor([0, 0, 1, 1, 1])
    )
    assert torch.equal(
        collated["feats"]["atom_to_token_map"],
        torch.tensor([0, 1, 2, 2, 4], dtype=torch.long),
    )
    assert torch.equal(
        collated["ground_truth"]["atom_to_token_map_per_batch_element"],
        torch.tensor([0, 1, 0, 0, 2], dtype=torch.long),
    )
    assert torch.equal(
        collated["ground_truth"]["is_original_unindexed_token_batch_offset"],
        torch.tensor([0, 0, 2, 2, 2], dtype=torch.long),
    )

    expected_unindexing_pair_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [True, False, True, True, True],
            [True, True, False, False, True],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    expected_token_bonds = torch.tensor(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, True, False],
            [False, False, True, False, False],
            [False, False, False, False, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(
        collated["feats"]["unindexing_pair_mask"], expected_unindexing_pair_mask
    )
    assert torch.equal(collated["feats"]["token_bonds"], expected_token_bonds)

    assert torch.equal(
        collated["t_L"],
        torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long),
    )
    assert torch.equal(
        collated["t_I"],
        torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long),
    )


def test_sparse_collate_zero_fills_missing_optional_token_bonds():
    """Mixed-task batches retain bonds only for conditioned examples."""
    example_with_bonds = _make_sparse_example(
        atom_coords=torch.zeros((2, 3)),
        atom_to_token_map=torch.tensor([0, 1], dtype=torch.long),
        example_id="structure_prediction",
    )
    example_without_bonds = _make_sparse_example(
        atom_coords=torch.zeros((3, 3)),
        atom_to_token_map=torch.tensor([0, 0, 2], dtype=torch.long),
        example_id="unconditional",
    )
    example_with_bonds["feats"]["token_bonds"] = torch.tensor(
        [[False, True], [True, False]], dtype=torch.bool
    )

    collated = SparseCollate()([example_with_bonds, example_without_bonds])

    expected = torch.zeros((5, 5), dtype=torch.bool)
    expected[0, 1] = True
    expected[1, 0] = True
    assert torch.equal(collated["feats"]["token_bonds"], expected)


def test_sparse_pairbias_attention_matches_single_batch_reference_for_sparse_and_full_paths():
    """Test that sparse pairbias attention matches single batch reference for both sparse and full paths."""
    B, H = 2, 2
    indices = torch.tensor(
        [
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 0],
                [3, 0, 1],
            ],
            [
                [1, 0, 2],
                [2, 3, 1],
                [3, 0, 1],
                [0, 2, 3],
            ],
        ],
        dtype=torch.long,
    )
    attn_mask = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, True],
                [True, False, True],
                [False, True, True],
            ],
            [
                [True, False, True],
                [True, True, False],
                [True, True, True],
                [False, True, True],
            ],
        ],
        dtype=torch.bool,
    )
    Q = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, -1.0],
                [0.0, 1.0, -0.5, 0.5],
                [0.5, -1.0, 1.0, 0.0],
                [1.0, 0.5, 0.0, -0.5],
            ],
            [
                [-1.0, 0.5, 1.0, 0.0],
                [0.5, 1.0, 0.0, -1.0],
                [1.0, -0.5, 0.5, 0.0],
                [0.0, 1.0, -1.0, 0.5],
            ],
        ],
        dtype=torch.float32,
    )
    K = torch.tensor(
        [
            [
                [0.5, 1.0, 0.0, -0.5],
                [1.0, 0.0, -1.0, 0.5],
                [0.0, -0.5, 1.0, 1.0],
                [1.0, 0.5, 0.5, 0.0],
            ],
            [
                [1.0, 0.0, 0.5, -1.0],
                [0.0, 1.0, -0.5, 0.5],
                [0.5, -1.0, 1.0, 0.0],
                [1.0, 0.5, 0.0, -0.5],
            ],
        ],
        dtype=torch.float32,
    )
    V = torch.tensor(
        [
            [
                [0.0, 1.0, 0.5, -0.5],
                [1.0, -1.0, 0.0, 0.5],
                [0.5, 0.5, -1.0, 1.0],
                [1.0, 0.0, 1.0, -1.0],
            ],
            [
                [1.0, 0.5, 0.0, -1.0],
                [0.5, -1.0, 1.0, 0.0],
                [0.0, 1.0, -0.5, 0.5],
                [1.0, 0.0, 0.5, -0.5],
            ],
        ],
        dtype=torch.float32,
    )
    B_sparse = torch.tensor(
        [
            [
                [[0.0, 0.5], [0.1, 0.0], [0.0, -0.5]],
                [[0.0, 0.0], [0.2, -0.1], [0.1, 0.0]],
                [[-0.2, 0.1], [0.0, 0.3], [0.4, 0.0]],
                [[0.3, 0.0], [0.0, 0.2], [-0.1, 0.4]],
            ],
            [
                [[0.2, 0.0], [0.0, -0.2], [0.1, 0.1]],
                [[-0.1, 0.1], [0.0, 0.2], [0.3, 0.0]],
                [[0.0, 0.4], [0.2, 0.0], [0.0, -0.3]],
                [[0.1, 0.0], [-0.2, 0.0], [0.0, 0.2]],
            ],
        ],
        dtype=torch.float32,
    )
    B_full = torch.tensor(
        [
            [
                [[0.0, 0.5], [0.1, 0.0], [0.0, -0.5], [0.2, 0.0]],
                [[0.0, 0.0], [0.2, -0.1], [0.1, 0.0], [0.0, 0.3]],
                [[-0.2, 0.1], [0.0, 0.3], [0.4, 0.0], [0.1, -0.2]],
                [[0.3, 0.0], [0.0, 0.2], [-0.1, 0.4], [0.2, 0.0]],
            ],
            [
                [[0.2, 0.0], [0.0, -0.2], [0.1, 0.1], [0.0, 0.4]],
                [[-0.1, 0.1], [0.0, 0.2], [0.3, 0.0], [-0.2, 0.1]],
                [[0.0, 0.4], [0.2, 0.0], [0.0, -0.3], [0.1, 0.0]],
                [[0.1, 0.0], [-0.2, 0.0], [0.0, 0.2], [0.0, -0.1]],
            ],
        ],
        dtype=torch.float32,
    )
    G = torch.tensor(
        [
            [
                [1.0, 0.5, 0.5, 1.0],
                [0.5, 1.0, 1.0, 0.5],
                [1.0, 1.0, 0.5, 0.5],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.5, 1.0, 1.0, 0.5],
                [1.0, 0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0, 1.0],
                [1.0, 1.0, 0.5, 0.5],
            ],
        ],
        dtype=torch.float32,
    )

    sparse_batched = sparse_pairbias_attention(
        Q=Q,
        K=K,
        V=V,
        B=B_sparse,
        indices=indices,
        H=H,
        gather_bias=False,
        G=G,
        full=False,
        attn_mask=attn_mask,
    )
    sparse_ref = torch.stack(
        [
            sparse_pairbias_attention(
                Q=Q[b : b + 1],
                K=K[b : b + 1],
                V=V[b : b + 1],
                B=B_sparse[b : b + 1],
                indices=indices[b : b + 1],
                H=H,
                gather_bias=False,
                G=G[b : b + 1],
                full=False,
                attn_mask=attn_mask[b : b + 1],
            ).squeeze(0)
            for b in range(B)
        ],
        dim=0,
    )
    assert torch.allclose(sparse_batched, sparse_ref, atol=1e-6, rtol=1e-6)

    full_batched = sparse_pairbias_attention(
        Q=Q,
        K=K,
        V=V,
        B=B_full,
        indices=indices,
        H=H,
        gather_bias=True,
        G=G,
        full=True,
        attn_mask=attn_mask,
    )
    full_ref = torch.stack(
        [
            sparse_pairbias_attention(
                Q=Q[b : b + 1],
                K=K[b : b + 1],
                V=V[b : b + 1],
                B=B_full[b : b + 1],
                indices=indices[b : b + 1],
                H=H,
                gather_bias=True,
                G=G[b : b + 1],
                full=True,
                attn_mask=attn_mask[b : b + 1],
            ).squeeze(0)
            for b in range(B)
        ],
        dim=0,
    )
    assert torch.allclose(full_batched, full_ref, atol=1e-6, rtol=1e-6)
