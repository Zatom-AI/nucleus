import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _install_fake_runtime(monkeypatch):
    """Install lightweight stand-ins for the external runtime deps."""

    def _register_module(name: str, module: types.ModuleType) -> None:
        monkeypatch.setitem(sys.modules, name, module)

    # Ensure the repo-local helper modules are re-imported under the fake runtime.
    for name in (
        "nucleus.utils",
        "nucleus.utils.attention",
        "torch.nn.attention",
        "torch.nn.attention.flex_attention",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # foundry.training.checkpoint.activation_checkpointing
    foundry = types.ModuleType("foundry")
    foundry.__path__ = []
    foundry_common = types.ModuleType("foundry.common")
    foundry_common.exists = lambda x: x is not None
    foundry_training = types.ModuleType("foundry.training")
    foundry_training.__path__ = []
    foundry_checkpoint = types.ModuleType("foundry.training.checkpoint")

    def activation_checkpointing(fn):
        return fn

    foundry_checkpoint.activation_checkpointing = activation_checkpointing
    foundry_utils = types.ModuleType("foundry.utils")
    foundry_utils.__path__ = []
    foundry_utils_ddp = types.ModuleType("foundry.utils.ddp")
    foundry_utils_torch = types.ModuleType("foundry.utils.torch")
    foundry_utils_torch.device_of = lambda _module: torch.device("cpu")

    class RankedLogger:
        """Lightweight stand-in for the RankedLogger used by the pairformer layers. Just records calls to any method for later inspection."""

        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    # rfd3.model.layers.layer_utils
    rfd3 = types.ModuleType("rfd3")
    rfd3.__path__ = []
    rfd3_model = types.ModuleType("rfd3.model")
    rfd3_model.__path__ = []
    rfd3_layers = types.ModuleType("rfd3.model.layers")
    rfd3_layers.__path__ = []
    rfd3_block_utils = types.ModuleType("rfd3.model.layers.block_utils")
    rfd3_block_utils.indices_to_mask = lambda *args, **kwargs: None
    rfd3_layer_utils = types.ModuleType("rfd3.model.layers.layer_utils")

    class MultiDimLinear(nn.Module):
        def __init__(self, c_in, out_shape, bias=True, norm=False):
            super().__init__()
            self.out_shape = tuple(out_shape)
            out_features = int(torch.tensor(out_shape).prod().item())
            self.weight = nn.Parameter(torch.zeros(out_features, c_in))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        def forward(self, x):
            out = F.linear(
                x,
                self.weight.to(dtype=x.dtype),
                self.bias.to(dtype=x.dtype) if self.bias is not None else None,
            )
            return out.reshape(*x.shape[:-1], *self.out_shape)

    class RMSNorm(nn.Module):
        def __init__(self, _shape):
            super().__init__()

        def forward(self, x):
            return x

    class AdaLN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, q, c):
            return q

    class LinearBiasInit(nn.Module):
        def __init__(self, c_in, c_out, biasinit=0.0):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(c_out, c_in))

        def forward(self, x):
            return F.linear(x, self.weight.to(dtype=x.dtype), None)

    class Transition(nn.Module):
        def __init__(self, c, n):
            super().__init__()
            self.c = c
            self.n = n

        def forward(self, x):
            return x

    def linearNoBias(c_in, c_out):
        class _LinearNoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(c_out, c_in))

            def forward(self, x):
                return F.linear(x, self.weight.to(dtype=x.dtype), None)

        return _LinearNoBias()

    rfd3_layer_utils.MultiDimLinear = MultiDimLinear
    rfd3_layer_utils.RMSNorm = RMSNorm
    rfd3_layer_utils.AdaLN = AdaLN
    rfd3_layer_utils.LinearBiasInit = LinearBiasInit
    rfd3_layer_utils.Transition = Transition
    rfd3_layer_utils.linearNoBias = linearNoBias

    # Repo-local TritonEvoformer stub.
    evo_attention = types.ModuleType("nucleus.kernels.evo_attention")
    evo_calls = []

    def TritonEvoformer(Q, K, V, res_mask, pair_bias):
        evo_calls.append((Q, K, V, res_mask, pair_bias))
        return torch.ones_like(Q)

    evo_attention.TritonEvoformer = TritonEvoformer
    evo_attention.evo_calls = evo_calls

    flex_attention_module = types.ModuleType("torch.nn.attention.flex_attention")
    flex_calls = []
    block_mask_calls = []

    def create_block_mask(
        mask_mod,
        B,
        H,
        q_len,
        kv_len,
        device=None,
        _compile=False,
        **kwargs,
    ):
        block_mask_calls.append(
            {
                "mask_mod": mask_mod,
                "B": B,
                "H": H,
                "q_len": q_len,
                "kv_len": kv_len,
                "device": device,
                "_compile": _compile,
                **kwargs,
            }
        )
        return {
            "mask_mod": mask_mod,
            "B": B,
            "H": H,
            "q_len": q_len,
            "kv_len": kv_len,
            "device": device,
            "_compile": _compile,
            **kwargs,
        }

    def flex_attention_fn(
        q, k, v, score_mod=None, block_mask=None, kernel_options=None
    ):
        score_mod_sample = None
        if score_mod is not None:
            score_mod_sample = score_mod(torch.tensor(0.0), 0, 0, 0, 1)

        scores = torch.einsum("bhld,bhkd->bhlk", q, k)
        if score_mod is not None:
            for b in range(q.shape[0]):
                for h in range(q.shape[1]):
                    for q_idx in range(q.shape[2]):
                        for kv_idx in range(q.shape[2]):
                            scores[b, h, q_idx, kv_idx] = score_mod(
                                scores[b, h, q_idx, kv_idx],
                                b,
                                h,
                                q_idx,
                                kv_idx,
                            )

        if block_mask is not None and block_mask.get("mask_mod") is not None:
            allowed = torch.empty(
                (q.shape[0], q.shape[2], q.shape[2]),
                dtype=torch.bool,
                device=q.device,
            )
            for b in range(q.shape[0]):
                for q_idx in range(q.shape[2]):
                    for kv_idx in range(q.shape[2]):
                        allowed[b, q_idx, kv_idx] = bool(
                            block_mask["mask_mod"](b, 0, q_idx, kv_idx)
                        )
            scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))

        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.einsum("bhlk,bhkd->bhld", attn, v)
        flex_calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "score_mod": score_mod,
                "block_mask": block_mask,
                "kernel_options": kernel_options,
                "score_mod_sample": score_mod_sample,
                "out": out,
            }
        )
        return out

    flex_attention_module.create_block_mask = create_block_mask
    flex_attention_module.flex_attention = flex_attention_fn
    flex_attention_module.flex_calls = flex_calls
    flex_attention_module.block_mask_calls = block_mask_calls

    nucleus = types.ModuleType("nucleus")
    nucleus.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "nucleus")]
    nucleus.DISABLE_CHECKPOINTING = False
    nucleus_model = types.ModuleType("nucleus.model")
    nucleus_model.__path__ = []
    nucleus_model_layers = types.ModuleType("nucleus.model.layers")
    nucleus_model_layers.__path__ = []
    nucleus_model_layers_layer_utils = types.ModuleType(
        "nucleus.model.layers.layer_utils"
    )
    nucleus_model_layers_layer_utils.Transition = Transition
    nucleus_utils = types.ModuleType("nucleus.utils")
    nucleus_utils.__path__ = []
    nucleus_utils_attention = types.ModuleType("nucleus.utils.attention")

    def create_flex_attention_block_mask(
        A_mask: torch.Tensor,
        q_len: int,
        kv_len: int,
        device: torch.device,
        block_size: int = 64,
        compile: bool = False,
    ):
        A_mask = A_mask.to(dtype=torch.bool, device=device)
        if torch.all(A_mask):
            return None
        invalid_queries = ~torch.any(A_mask, dim=-1)

        if A_mask.ndim == 2:

            def mask_mod(_b, _h, q_idx, kv_idx):
                allowed = A_mask[q_idx, kv_idx]
                return allowed | (invalid_queries[q_idx] & (q_idx == kv_idx))

            return create_block_mask(
                mask_mod,
                None,
                None,
                q_len,
                kv_len,
                device=device,
                BLOCK_SIZE=block_size,
                _compile=compile,
            )

        if A_mask.ndim == 3:

            def mask_mod(b, _h, q_idx, kv_idx):
                allowed = A_mask[b, q_idx, kv_idx]
                return allowed | (invalid_queries[b, q_idx] & (q_idx == kv_idx))

            return create_block_mask(
                mask_mod,
                A_mask.shape[0],
                None,
                q_len,
                kv_len,
                device=device,
                BLOCK_SIZE=block_size,
                _compile=compile,
            )

        raise ValueError("A_mask must be [L, L] or [B, L, L].")

    nucleus_utils_attention.create_flex_attention_block_mask = (
        create_flex_attention_block_mask
    )
    nucleus_kernels = types.ModuleType("nucleus.kernels")
    nucleus_kernels.__path__ = []
    torch_nn = sys.modules.get("torch.nn")
    if torch_nn is not None:
        torch_nn_attention = types.ModuleType("torch.nn.attention")
        torch_nn_attention.__path__ = []
        torch_nn_attention.flex_attention = flex_attention_module
        _register_module("torch.nn.attention", torch_nn_attention)
        _register_module("torch.nn.attention.flex_attention", flex_attention_module)

    _register_module("foundry", foundry)
    _register_module("foundry.common", foundry_common)
    _register_module("foundry.training", foundry_training)
    _register_module("foundry.training.checkpoint", foundry_checkpoint)
    _register_module("foundry.utils", foundry_utils)
    _register_module("foundry.utils.ddp", foundry_utils_ddp)
    _register_module("foundry.utils.torch", foundry_utils_torch)
    _register_module("rfd3", rfd3)
    _register_module("rfd3.model", rfd3_model)
    _register_module("rfd3.model.layers", rfd3_layers)
    _register_module("rfd3.model.layers.block_utils", rfd3_block_utils)
    _register_module("rfd3.model.layers.layer_utils", rfd3_layer_utils)
    _register_module("nucleus", nucleus)
    _register_module("nucleus.model", nucleus_model)
    _register_module("nucleus.model.layers", nucleus_model_layers)
    _register_module(
        "nucleus.model.layers.layer_utils", nucleus_model_layers_layer_utils
    )
    _register_module("nucleus.utils", nucleus_utils)
    _register_module("nucleus.utils.attention", nucleus_utils_attention)
    _register_module("nucleus.kernels", nucleus_kernels)
    _register_module("nucleus.kernels.evo_attention", evo_attention)

    foundry_utils_ddp.RankedLogger = RankedLogger

    return evo_calls


def _import_pairformer_layers(monkeypatch):
    """Dynamically import the pairformer_layers module after installing the fake runtime dependencies."""
    module_name = "test_pairformer_layers_under_test"
    sys.modules.pop(module_name, None)
    path = (
        Path(__file__).resolve().parents[1]
        / "src/nucleus/model/layers/pairformer_layers.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reference_triton_evoformer(Q, K, V, res_mask, pair_bias):
    """Reference implementation used by the comparison test to mirror the fallback attention math."""
    q = Q.squeeze(1)
    k = K.squeeze(1)
    v = V.squeeze(1)

    logits = torch.einsum("...ihd,...jhd->...ijh", q, k)
    logits = logits + pair_bias.squeeze(1).permute(0, 2, 3, 1)
    logits = logits + res_mask[:, 0, 0, 0, :].unsqueeze(1).unsqueeze(-1)

    attn = torch.softmax(logits.float(), dim=-2).to(v.dtype)
    out = torch.einsum("...ijh,...jhd->...ihd", attn, v)
    return out.unsqueeze(1)


def _set_identity_weights(module) -> None:
    """Helper function to set the weights of the module to identity matrices for testing."""
    eye = torch.eye(8, device=module.to_q.weight.device, dtype=module.to_q.weight.dtype)
    with torch.no_grad():
        module.to_q.weight.copy_(eye)
        module.to_k.weight.copy_(eye)
        module.to_v.weight.copy_(eye)
        module.to_g[0].weight.copy_(eye)
        module.to_a.weight.copy_(eye)


def test_attention_pair_bias_uses_local_triton_evoformer_branch(monkeypatch):
    """Test that AttentionPairBiasPairformerDeepspeed correctly uses the local TritonEvoformer implementation when use_deepspeed_evo is True, and that the inputs to the evoformer have the expected shapes and values based on the input tensors and attention mask."""
    evo_calls = _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    module = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=None,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    module.use_deepspeed_evo = True

    a_i = torch.randn(25, 8)
    z_ii = torch.randn(25, 25, 4)
    a_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    a_mask = torch.nn.functional.pad(a_mask, (0, 22, 0, 22), value=False)

    out = module(a_i, None, z_ii, A_mask=a_mask)

    assert out.shape == (25, 8)
    assert len(evo_calls) == 1

    q, k, v, res_mask, pair_bias = evo_calls[0]
    assert q.shape == (1, 1, 25, 2, 4)
    assert k.shape == (1, 1, 25, 2, 4)
    assert v.shape == (1, 1, 25, 2, 4)
    assert res_mask.shape == (1, 1, 1, 1, 25)
    assert pair_bias.shape == (1, 1, 2, 25, 25)

    assert torch.isneginf(pair_bias[0, 0, 0, 0, 1])
    assert torch.equal(pair_bias[0, 0, :, 24, :], torch.zeros((2, 25)))
    assert torch.isneginf(res_mask[0, 0, 0, 0, 1])
    assert torch.isneginf(res_mask[0, 0, 0, 0, 2])


def test_attention_pair_bias_uses_local_triton_evoformer_branch_with_batch_dim(
    monkeypatch,
):
    """Test that the Triton Evoformer path preserves a leading batch dimension on Q/K/V, pair bias, and the residual mask."""
    evo_calls = _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    module = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    module.use_deepspeed_evo = True

    a_i = torch.randn(2, 25, 8)
    z_ii = torch.randn(2, 25, 25, 4)
    a_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    a_mask = torch.nn.functional.pad(a_mask, (0, 22, 0, 22), value=False)

    out = module(a_i, None, z_ii, A_mask=a_mask)

    assert out.shape == (2, 25, 8)
    assert len(evo_calls) == 1

    q, k, v, res_mask, pair_bias = evo_calls[0]
    assert q.shape == (2, 1, 25, 2, 4)
    assert k.shape == (2, 1, 25, 2, 4)
    assert v.shape == (2, 1, 25, 2, 4)
    assert res_mask.shape == (2, 1, 1, 1, 25)
    assert pair_bias.shape == (2, 1, 2, 25, 25)

    assert torch.isneginf(pair_bias[0, 0, 0, 0, 1])
    assert torch.equal(pair_bias[0, 0, :, 24, :], torch.zeros((2, 25)))
    assert torch.isneginf(res_mask[0, 0, 0, 0, 1])
    assert torch.isneginf(res_mask[0, 0, 0, 0, 2])


def test_attention_pair_bias_fallback_and_triton_paths_match_on_a_i_outputs(
    monkeypatch,
):
    """Test that the fallback attention path and the TritonEvoformer path produce the same A_I output up to numerical precision."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)
    monkeypatch.setattr(
        pairformer_layers,
        "TritonEvoformer",
        _reference_triton_evoformer,
    )

    module = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    module.force_bfloat16 = False

    eye = torch.eye(8)
    module.to_q.weight.data.copy_(eye)
    module.to_k.weight.data.copy_(eye)
    module.to_v.weight.data.copy_(eye)
    module.to_g[0].weight.data.copy_(eye)
    module.to_a.weight.data.copy_(eye)

    a_i = torch.linspace(-0.3, 0.3, steps=25 * 8, dtype=torch.float32).reshape(25, 8)
    z_ii = torch.linspace(-0.2, 0.2, steps=25 * 25 * 4, dtype=torch.float32).reshape(
        25, 25, 4
    )
    a_mask = (torch.arange(25)[:, None] - torch.arange(25)[None, :]).abs() <= 2

    module.use_deepspeed_evo = False
    fallback_out = module(a_i, None, z_ii, A_mask=a_mask)

    module.use_deepspeed_evo = True
    evo_out = module(a_i, None, z_ii, A_mask=a_mask)

    assert fallback_out.shape == evo_out.shape == (25, 8)
    torch.testing.assert_close(
        evo_out.float(),
        fallback_out.float(),
        atol=5e-3,
        rtol=5e-3,
    )


def test_pairformer_block_propagates_attn_backend(monkeypatch):
    """Test that the attn_backend setting is correctly propagated from the PairformerBlock to its AttentionPairBiasPairformerDeepspeed submodule, and that both modules end up using the local TritonEvoformer implementation when MegaFold is selected."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    block = pairformer_layers.PairformerBlock(
        c_s=8,
        c_z=4,
        attention_pair_bias={"n_head": 2, "kq_norm": False},
        attn_backend="megafold",
    )

    assert block.attn_backend == "megafold"
    assert block.use_deepspeed_evo is True
    assert block.attention_pair_bias.use_deepspeed_evo is True


def test_pairformer_block_manual_backend_disables_triton(monkeypatch):
    """Test that the manual backend keeps the original attention path enabled on the submodule."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    block = pairformer_layers.PairformerBlock(
        c_s=8,
        c_z=4,
        attention_pair_bias={"n_head": 2, "kq_norm": False},
        attn_backend="manual",
    )

    assert block.attn_backend == "manual"
    assert block.use_deepspeed_evo is False
    assert block.attention_pair_bias.use_deepspeed_evo is False


def test_attention_pair_bias_uses_local_flex_attention_branch(monkeypatch):
    """Test that the FlexAttention sketch is selected and receives a block mask plus pair-bias score_mod."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    module = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    module.use_flex_attention = True

    a_i = torch.randn(25, 8)
    z_ii = torch.randn(25, 25, 4)
    a_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    a_mask = torch.nn.functional.pad(a_mask, (0, 22, 0, 22), value=False)

    out = module(a_i, None, z_ii, A_mask=a_mask)

    assert out.shape == (25, 8)

    flex_module = sys.modules["torch.nn.attention.flex_attention"]
    assert len(flex_module.flex_calls) == 1
    assert len(flex_module.block_mask_calls) == 1

    call = flex_module.flex_calls[0]
    assert call["q"].shape == (1, 2, 25, 4)
    assert call["k"].shape == (1, 2, 25, 4)
    assert call["v"].shape == (1, 2, 25, 4)
    assert call["block_mask"]["q_len"] == 25
    assert call["block_mask"]["kv_len"] == 25
    assert call["kernel_options"]["BLOCKS_ARE_CONTIGUOUS"] is False
    assert call["kernel_options"]["ROWS_GUARANTEED_SAFE"] is False
    assert torch.is_tensor(call["score_mod_sample"])


def test_flex_attention_block_mask_is_only_created_when_needed(monkeypatch):
    """Dense masks should not trigger block-mask compilation."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    dense_mask = torch.ones((25, 25), dtype=torch.bool)
    block_mask = pairformer_layers.create_flex_attention_block_mask(
        dense_mask, 25, 25, torch.device("cpu")
    )

    assert block_mask is None
    flex_module = sys.modules["torch.nn.attention.flex_attention"]
    assert len(flex_module.block_mask_calls) == 0


def test_flex_attention_block_mask_can_be_reused_across_layers(monkeypatch):
    """A single precomputed block mask should be reusable across multiple Pairformer blocks."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    block_a = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    block_b = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    for module in (block_a, block_b):
        module.use_flex_attention = True
        module.force_bfloat16 = False
        _set_identity_weights(module)

    a_i = torch.randn(25, 8)
    z_ii = torch.randn(25, 25, 4)
    a_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    a_mask = torch.nn.functional.pad(a_mask, (0, 22, 0, 22), value=False)
    flex_block_mask = pairformer_layers.create_flex_attention_block_mask(
        a_mask, 25, 25, torch.device("cpu")
    )

    block_a(a_i, None, z_ii, A_mask=a_mask, flex_block_mask=flex_block_mask)
    block_b(a_i, None, z_ii, A_mask=a_mask, flex_block_mask=flex_block_mask)

    flex_module = sys.modules["torch.nn.attention.flex_attention"]
    assert len(flex_module.block_mask_calls) == 1


def test_attention_pair_bias_flex_matches_manual_and_megafold(monkeypatch):
    """FlexAttention, manual attention, and the MegaFold path should agree numerically."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    module = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    module.force_bfloat16 = False
    _set_identity_weights(module)

    a_i = torch.linspace(-0.3, 0.3, steps=25 * 8, dtype=torch.float32).reshape(25, 8)
    z_ii = torch.linspace(-0.2, 0.2, steps=25 * 25 * 4, dtype=torch.float32).reshape(
        25, 25, 4
    )
    a_mask = torch.ones((25, 25), dtype=torch.bool)
    a_mask[0:3, 1:9] = False

    module.use_flex_attention = False
    module.use_deepspeed_evo = False
    manual_out = module(a_i, None, z_ii, A_mask=a_mask)

    module.use_deepspeed_evo = True
    megafold_out = module(a_i, None, z_ii, A_mask=a_mask)

    module.use_deepspeed_evo = False
    module.use_flex_attention = True
    with torch.no_grad():
        flex_out = module(a_i, None, z_ii, A_mask=a_mask)

    assert manual_out.shape == megafold_out.shape == flex_out.shape == (25, 8)
    torch.testing.assert_close(
        flex_out.float(), manual_out.float(), atol=5e-3, rtol=5e-3
    )


def test_attention_pair_bias_prepares_shared_projections_across_backends(monkeypatch):
    """q/k/v should be prepared once and reused consistently across backend branches."""
    evo_calls = _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    module = pairformer_layers.AttentionPairBiasPairformerDeepspeed(
        c_a=8,
        c_s=0,
        c_pair=4,
        n_head=2,
        kq_norm=False,
    )
    module.force_bfloat16 = False
    _set_identity_weights(module)

    a_i = torch.linspace(-0.3, 0.3, steps=25 * 8, dtype=torch.float32).reshape(25, 8)
    z_ii = torch.linspace(-0.2, 0.2, steps=25 * 25 * 4, dtype=torch.float32).reshape(
        25, 25, 4
    )
    a_mask = torch.ones((25, 25), dtype=torch.bool)

    manual_out = module(a_i, None, z_ii, A_mask=a_mask)

    module.use_deepspeed_evo = True
    with torch.amp.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16):
        megafold_out = module(a_i, None, z_ii, A_mask=a_mask)

    module.use_deepspeed_evo = False
    module.use_flex_attention = True
    with torch.no_grad():
        flex_out = module(a_i, None, z_ii, A_mask=a_mask)

    expected_q = a_i.reshape(25, 2, 4) / 2.0
    expected_k = a_i.reshape(25, 2, 4)
    expected_v = a_i.reshape(25, 2, 4)

    q, k, v, _, _ = evo_calls[0]
    expected_q_bf16 = expected_q.to(dtype=q.dtype)
    expected_k_bf16 = expected_k.to(dtype=k.dtype)
    expected_v_bf16 = expected_v.to(dtype=v.dtype)
    torch.testing.assert_close(
        q.squeeze(0).squeeze(0).float(),
        expected_q_bf16.float(),
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        k.squeeze(0).squeeze(0).float(),
        expected_k_bf16.float(),
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        v.squeeze(0).squeeze(0).float(),
        expected_v_bf16.float(),
        atol=1e-3,
        rtol=1e-3,
    )

    flex_module = sys.modules["torch.nn.attention.flex_attention"]
    call = flex_module.flex_calls[0]
    torch.testing.assert_close(
        call["q"].float(),
        expected_q.permute(1, 0, 2).unsqueeze(0).float(),
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        call["k"].float(),
        expected_k.permute(1, 0, 2).unsqueeze(0).float(),
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        call["v"].float(),
        expected_v.permute(1, 0, 2).unsqueeze(0).float(),
        atol=1e-3,
        rtol=1e-3,
    )

    assert manual_out.shape == megafold_out.shape == flex_out.shape == (25, 8)
    torch.testing.assert_close(
        flex_out.float(), manual_out.float(), atol=5e-3, rtol=5e-3
    )


def test_pairformer_block_propagates_flex_attn_backend(monkeypatch):
    """Test that attn_backend='flex' propagates to the attention submodule."""
    _install_fake_runtime(monkeypatch)
    pairformer_layers = _import_pairformer_layers(monkeypatch)

    block = pairformer_layers.PairformerBlock(
        c_s=8,
        c_z=4,
        attention_pair_bias={"n_head": 2, "kq_norm": False},
        attn_backend="flex",
    )

    assert block.attn_backend == "flex"
    assert block.use_flex_attention is True
    assert block.attention_pair_bias.use_flex_attention is True
