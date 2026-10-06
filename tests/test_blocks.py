import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


def _install_blocks_runtime(monkeypatch):
    """Install lightweight stand-ins for the external runtime deps used by blocks.py."""

    def _register_module(name: str, module: types.ModuleType) -> None:
        monkeypatch.setitem(sys.modules, name, module)

    foundry = types.ModuleType("foundry")
    foundry.__path__ = []
    foundry.DISABLE_CHECKPOINTING = True
    foundry_common = types.ModuleType("foundry.common")
    foundry_common.exists = lambda x: x is not None
    foundry_training = types.ModuleType("foundry.training")
    foundry_training.__path__ = []
    foundry_training_checkpoint = types.ModuleType("foundry.training.checkpoint")
    foundry_training_checkpoint.activation_checkpointing = lambda *args, **kwargs: None

    rfd3 = types.ModuleType("rfd3")
    rfd3.__path__ = []
    rfd3_model = types.ModuleType("rfd3.model")
    rfd3_model.__path__ = []
    rfd3_layers = types.ModuleType("rfd3.model.layers")
    rfd3_layers.__path__ = []
    rfd3_layer_utils = types.ModuleType("rfd3.model.layers.layer_utils")
    rfd3_block_utils = types.ModuleType("rfd3.model.layers.block_utils")
    rfd3_blocks = types.ModuleType("rfd3.model.layers.blocks")

    class Transition(nn.Module):
        def __init__(self, c, n):
            super().__init__()

        def forward(self, x):
            return x

    class ConditionedTransitionBlock(nn.Module):
        def __init__(self, c_token, c_s):
            super().__init__()

        def forward(self, x, *_args, **_kwargs):
            return x

    class Downcast(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *_args, **_kwargs):
            return x

    class Upcast(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *_args, **_kwargs):
            return x

    def linearNoBias(c_in, c_out, **_kwargs):
        return nn.Linear(c_in, c_out, bias=False)

    def indices_to_mask(indices, n):
        mask = torch.zeros(indices.shape[0], n, dtype=torch.bool)
        mask.scatter_(1, indices.clamp(min=0, max=n - 1), True)
        return mask

    class RMSNorm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    class AdaLN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *_args, **_kwargs):
            return x

    class LinearBiasInit(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    rfd3_layer_utils.Transition = Transition
    rfd3_layer_utils.linearNoBias = linearNoBias
    rfd3_layer_utils.RMSNorm = RMSNorm
    rfd3_layer_utils.AdaLN = AdaLN
    rfd3_layer_utils.LinearBiasInit = LinearBiasInit
    rfd3_block_utils.indices_to_mask = indices_to_mask
    rfd3_blocks.ConditionedTransitionBlock = ConditionedTransitionBlock
    rfd3_blocks.Downcast = Downcast
    rfd3_blocks.Upcast = Upcast

    atomworks = types.ModuleType("atomworks")
    atomworks.__path__ = []
    atomworks_constants = types.ModuleType("atomworks.constants")
    atomworks_constants.ELEMENT_NAME_TO_ATOMIC_NUMBER = {"X": 0}

    _register_module("foundry", foundry)
    _register_module("foundry.common", foundry_common)
    _register_module("foundry.training", foundry_training)
    _register_module("foundry.training.checkpoint", foundry_training_checkpoint)
    _register_module("rfd3", rfd3)
    _register_module("rfd3.model", rfd3_model)
    _register_module("rfd3.model.layers", rfd3_layers)
    _register_module("rfd3.model.layers.block_utils", rfd3_block_utils)
    _register_module("rfd3.model.layers.layer_utils", rfd3_layer_utils)
    _register_module("rfd3.model.layers.blocks", rfd3_blocks)
    _register_module("atomworks", atomworks)
    _register_module("atomworks.constants", atomworks_constants)

    nucleus_pkg = types.ModuleType("nucleus.model")
    nucleus_pkg.__path__ = [str(Path("src/nucleus/model").resolve())]
    nucleus_layers_pkg = types.ModuleType("nucleus.model.layers")
    nucleus_layers_pkg.__path__ = [str(Path("src/nucleus/model/layers").resolve())]
    _register_module("nucleus.model", nucleus_pkg)
    _register_module("nucleus.model.layers", nucleus_layers_pkg)


def test_local_token_transformer_computes_flex_block_mask_once(monkeypatch):
    """Test that LocalTokenTransformer computes the flex block mask once and reuses it across its blocks."""
    pytest.importorskip("torch.nn.attention.flex_attention")
    _install_blocks_runtime(monkeypatch)

    import nucleus.model.layers.attention as attention_module
    import nucleus.model.layers.blocks as blocks_module

    call_count = {"count": 0}

    def fake_maybe_create_flex_block_mask(*args, **kwargs):
        call_count["count"] += 1
        return None, True

    def fake_flex_attention(q, k, v, **kwargs):
        return torch.zeros_like(q)

    monkeypatch.setattr(
        blocks_module,
        "maybe_create_flex_block_mask",
        fake_maybe_create_flex_block_mask,
    )
    monkeypatch.setattr(attention_module, "create_block_mask", object())
    monkeypatch.setattr(attention_module, "torch_flex_attention", fake_flex_attention)

    module = blocks_module.LocalTokenTransformer(
        c_token=8,
        c_tokenpair=4,
        c_s=None,
        n_block=2,
        diffusion_transformer_block={
            "attn_backend": "flex",
            "dropout": 0.0,
            "no_residual_connection_between_attention_and_transition": False,
            "n_head": 2,
            "kq_norm": False,
        },
        n_registers=None,
        n_local_tokens=2,
        n_keys=8,
        attn_backend="flex",
    )

    a_i = torch.randn(1, 25, 8)
    s_i = None
    z_ii = torch.randn(1, 25, 25, 4)
    f = {
        "atom_to_token_map": torch.arange(25, dtype=torch.long),
        "unindexing_pair_mask": torch.zeros((25, 25), dtype=torch.bool),
        "restype": torch.zeros(25, dtype=torch.long),
    }

    out = module(a_i, s_i, z_ii, f=f, X_L=a_i, full=True)

    assert out.shape == (1, 25, 8)
    assert call_count["count"] == 1


def test_local_atom_transformer_computes_flex_block_mask_once(monkeypatch):
    """Test that LocalAtomTransformer computes the flex block mask once and reuses it across its blocks."""
    pytest.importorskip("torch.nn.attention.flex_attention")
    _install_blocks_runtime(monkeypatch)

    import nucleus.model.layers.attention as attention_module
    import nucleus.model.layers.blocks as blocks_module

    call_count = {"count": 0}

    def fake_maybe_create_flex_block_mask(*args, **kwargs):
        call_count["count"] += 1
        return None, True

    def fake_flex_attention(q, k, v, **kwargs):
        return torch.zeros_like(q)

    monkeypatch.setattr(
        blocks_module,
        "maybe_create_flex_block_mask",
        fake_maybe_create_flex_block_mask,
    )
    monkeypatch.setattr(attention_module, "create_block_mask", object())
    monkeypatch.setattr(attention_module, "torch_flex_attention", fake_flex_attention)

    module = blocks_module.LocalAtomTransformer(
        c_atom=8,
        c_s=None,
        c_atompair=4,
        atom_transformer_block={
            "attn_backend": "flex",
            "dropout": 0.0,
            "no_residual_connection_between_attention_and_transition": False,
            "n_head": 2,
            "kq_norm": False,
        },
        n_blocks=2,
        attn_backend="flex",
    )

    q_l = torch.randn(1, 25, 8)
    c_l = None
    p_ll = torch.randn(1, 25, 25, 4)
    indices = torch.arange(25, dtype=torch.long).view(1, 1, 25).expand(1, 25, 25)
    tok_idx = torch.arange(25, dtype=torch.long)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros((25, 25), dtype=torch.bool),
    }

    out = module(q_l, c_l, p_ll, indices=indices, f=f, tok_idx=tok_idx, full=True)

    assert out.shape == (1, 25, 8)
    assert call_count["count"] == 1


def test_compact_streaming_decoder_computes_flex_block_mask_once(monkeypatch):
    """Test that CompactStreamingDecoder computes the flex block mask once and reuses it across its atom blocks."""
    pytest.importorskip("torch.nn.attention.flex_attention")
    _install_blocks_runtime(monkeypatch)

    import nucleus.model.layers.attention as attention_module
    import nucleus.model.layers.blocks as blocks_module

    call_count = {"count": 0}

    def fake_maybe_create_flex_block_mask(*args, **kwargs):
        call_count["count"] += 1
        return None, True

    def fake_flex_attention(q, k, v, **kwargs):
        return torch.zeros_like(q)

    monkeypatch.setattr(
        blocks_module,
        "maybe_create_flex_block_mask",
        fake_maybe_create_flex_block_mask,
    )
    monkeypatch.setattr(attention_module, "create_block_mask", object())
    monkeypatch.setattr(attention_module, "torch_flex_attention", fake_flex_attention)
    monkeypatch.setattr(
        attention_module.LocalAttentionPairBias,
        "forward",
        lambda self, Q_L, C_L, P_LL, **kwargs: Q_L,
    )

    module = blocks_module.CompactStreamingDecoder(
        c_atom=8,
        c_atompair=4,
        c_token=8,
        c_s=None,
        c_tokenpair=4,
        atom_transformer_block={
            "attn_backend": "flex",
            "dropout": 0.0,
            "no_residual_connection_between_attention_and_transition": False,
            "n_head": 2,
            "kq_norm": False,
        },
        upcast={},
        downcast={},
        n_blocks=2,
        attn_backend="flex",
    )

    a_i = torch.randn(1, 25, 8)
    s_i = torch.randn(1, 25, 8)
    q_l = torch.randn(1, 25, 8)
    c_l = torch.randn(1, 25, 8)
    p_ll = torch.randn(1, 25, 25, 4)
    tok_idx = torch.arange(25, dtype=torch.long)
    indices = torch.arange(25, dtype=torch.long).view(1, 1, 25).expand(1, 25, 25)
    f = {
        "atom_to_token_map": tok_idx,
        "unindexing_pair_mask": torch.zeros((25, 25), dtype=torch.bool),
    }

    out_a, out_q, o = module(
        a_i,
        s_i,
        q_l,
        c_l,
        p_ll,
        tok_idx,
        indices,
        f=f,
    )

    assert out_a.shape == (1, 25, 8)
    assert out_q.shape == (1, 25, 8)
    assert isinstance(o, dict)
    assert call_count["count"] == 1
