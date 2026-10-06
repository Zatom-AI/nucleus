from typing import Literal

import torch
from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import device_of
from torch import nn

try:
    from rfd3.model.layers.layer_utils import (
        MultiDimLinear,
        RMSNorm,
        linearNoBias,
    )
except ImportError:
    raise ImportError(
        "Could not import RFD3 modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )

from nucleus import DISABLE_CHECKPOINTING
from nucleus.model.layers.layer_utils import Transition
from nucleus.utils.attention import create_flex_attention_block_mask

global_logger = RankedLogger(__name__, rank_zero_only=False)

try:
    from nucleus.kernels.evo_attention import TritonEvoformer
except ImportError:
    TritonEvoformer = None
    global_logger.warning(
        "Could not import Triton Evoformer. DeepSpeed Evo attention will not be available."
        " Please ensure that you have the necessary hardware and software requirements for running Triton kernels."
    )

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
    )
    from torch.nn.attention.flex_attention import (
        flex_attention as torch_flex_attention,
    )
except ImportError:
    create_block_mask = None
    torch_flex_attention = None
    global_logger.warning(
        "Could not import torch.nn.attention.flex_attention. The flex attention path will not be available."
        " Please ensure that you have PyTorch 2.0 or later installed to use the flex attention path."
    )


def flex_pairbias_attention(
    Q_IH: torch.Tensor,
    K_IH: torch.Tensor,
    V_IH: torch.Tensor,
    pair_bias: torch.Tensor,
    A_mask: torch.Tensor | None,
    flex_block_mask=None,
    flex_block_mask_ready=False,
    block_size: int = 64,
    blocks_are_contiguous=False,
    rows_guaranteed_safe=False,
) -> torch.Tensor:
    """FlexAttention-based pair-bias attention path."""

    if torch_flex_attention is None or create_block_mask is None:
        raise RuntimeError(
            "torch.nn.attention.flex_attention is not available in this environment."
        )

    # FlexAttention uses [B, H, L, D]; normalize both batched [B, L, H, D]
    # and unbatched [L, H, D] inputs into that layout.
    if Q_IH.ndim == 3:
        q = Q_IH.permute(1, 0, 2).unsqueeze(0).contiguous()
        k = K_IH.permute(1, 0, 2).unsqueeze(0).contiguous()
        v = V_IH.permute(1, 0, 2).unsqueeze(0).contiguous()
        pair_bias = pair_bias.permute(2, 0, 1).unsqueeze(0).contiguous()
    elif Q_IH.ndim == 4:
        q = Q_IH.transpose(1, 2).contiguous()
        k = K_IH.transpose(1, 2).contiguous()
        v = V_IH.transpose(1, 2).contiguous()
        pair_bias = pair_bias.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            "Q_IH must be [L, H, D] or [B, L, H, D] for the flex attention path."
        )

    invalid_queries = None
    if A_mask is not None:
        A_mask = A_mask.to(dtype=torch.bool, device=q.device)
        invalid_queries = ~torch.any(A_mask, dim=-1)
        diag = torch.eye(A_mask.shape[-1], device=A_mask.device, dtype=torch.bool)
        if A_mask.ndim == 2:
            A_mask = A_mask | (invalid_queries[:, None] & diag)
        elif A_mask.ndim == 3:
            A_mask = A_mask | (invalid_queries[:, :, None] & diag[None, :, :])
        if flex_block_mask is None and not flex_block_mask_ready:
            flex_block_mask = create_flex_attention_block_mask(
                A_mask, q.shape[-2], k.shape[-2], q.device, block_size=block_size
            )

    def score_mod(score, b, h, q_idx, kv_idx):
        if A_mask is None:
            return score + pair_bias[b, h, q_idx, kv_idx]

        if A_mask.ndim == 2:
            allowed = A_mask[q_idx, kv_idx]
        elif A_mask.ndim == 3:
            allowed = A_mask[b, q_idx, kv_idx]
        else:
            raise ValueError(
                "A_mask must be None, [L, L], or [B, L, L] for the flex attention path."
            )

        score = score + pair_bias[b, h, q_idx, kv_idx]
        return torch.where(allowed, score, torch.full_like(score, float("-inf")))

    out = torch_flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=flex_block_mask,
        kernel_options={
            "BLOCK_M": block_size,
            "BLOCK_N": block_size,
            "BLOCKS_ARE_CONTIGUOUS": blocks_are_contiguous,
            "ROWS_GUARANTEED_SAFE": rows_guaranteed_safe,
        },
    )

    if A_mask is not None and invalid_queries is not None:
        if invalid_queries.ndim == 1:
            out = out.masked_fill(invalid_queries[None, None, :, None], 0.0)
        else:
            out = out.masked_fill(invalid_queries[:, None, :, None], 0.0)

    return out


class AttentionPairBiasPairformerDeepspeed(nn.Module):
    """
    Attention with pair bias, as used in AF3. This is a reimplementation of the attention and pair bias mechanism used in AF3, with some modifications to support efficient attention implementations like those in DeepSpeed Evo. The main differences from a standard multi-head attention are:
        - The attention bias is computed from the pair representation Z_II, rather than being a learned
            parameter. This allows the bias to be dynamic and depend on the input structure.
        - The attention output is gated by a sigmoid function, which allows the model to control how much of the attended information to use.

    Args:
        c_a: The dimensionality of the input atom features.
        c_s: The dimensionality of the input sequence features (not used in this implementation).
        c_pair: The dimensionality of the pair representation, which is used to compute the attention bias.
        n_head: The number of attention heads.
        kq_norm: Whether to apply layer normalization to the keys and queries. This can help with training stability, especially when using efficient attention implementations.
    """

    def __init__(self, c_a, c_s, c_pair, n_head, kq_norm=False):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head

        self.to_q = MultiDimLinear(c_a, (n_head, self.c))
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, self.c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        self.ln_0 = RMSNorm((c_pair,))
        self.ln_1 = RMSNorm((c_a,))
        self.use_deepspeed_evo = False
        self.use_flex_attention = False
        self.force_bfloat16 = True

    def forward(
        self,
        A_I,  # [I, C_a]
        S_I,  # [I, C_a] | None
        Z_II,  # [I, I, C_z]
        Beta_II=None,  # [I, I]
        A_mask=None,  # [I, I] | None
        flex_block_mask=None,
        flex_block_mask_ready=False,
    ) -> torch.Tensor:
        """
        Compute the attention output for the atom features A_I, using the pair representation Z_II to compute the attention bias. The sequence features S_I are not used in this implementation, but are included in the signature for compatibility with the AF3 architecture.

        Args:
            A_I: Atom features, shape [I, C_a].
            S_I: Sequence features, shape [I, C_a] or None (not used in this implementation).
            Z_II: Pair representation, shape [I, I, C_z].
            Beta_II: Optional pair bias, shape [I, I]. This can be used to add an additional bias to the attention scores, for example to encode distance information or other pairwise features.
            A_mask: Optional attention mask, shape [I, I]. This can be used to mask out certain pairs of atoms from attending to each other, for example to enforce a local attention pattern or to mask out padding tokens.
            flex_block_mask: Optional precomputed block mask for the flex attention path. This can be used to speed up the computation of the block mask for the flex attention implementation, which can be expensive to compute on the fly for large inputs. If not provided, the block mask will be computed on the fly based on A_mask.
            flex_block_mask_ready: Whether the provided flex_block_mask is ready to be used. This should be set to True if the flex_block_mask is provided and can be used directly, or False if the flex_block_mask is not provided and needs to be computed on the fly based on A_mask.

        Returns:
            A_I: Updated atom features, shape [I, C_a].
        """
        # Input projections
        assert S_I is None
        A_I = self.ln_1(A_I)

        if (self.use_deepspeed_evo or self.force_bfloat16) and A_I.device.type != "mps":
            A_I = A_I.to(torch.bfloat16)

        # Prepare the attention projections once so every backend sees the same
        # q/k/v/g tensors before backend-specific kernel dispatch.
        q_scale = torch.sqrt(A_I.new_tensor(self.c))
        Q_IH = self.to_q(A_I) / q_scale
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + (
            Beta_II[..., None] if Beta_II is not None else 0.0
        )
        G_IH = self.to_g(A_I)

        # The attention trunk is called with either unbatched tensors
        # ([I, C] / [I, I, C]) or a leading batch dimension ([B, I, C] /
        # [B, I, I, C]). Normalize both cases into the 5D layout expected by
        # TritonEvoformer: [B, N_SEQ, HEAD, SEQ_LEN, DIM].
        is_batched = Q_IH.ndim == 4
        L = B_IIH.shape[-2]

        use_flex_attention = (
            self.use_flex_attention
            and torch_flex_attention is not None
            and create_block_mask is not None
            and L > 24
            and A_I.device.type in {"cpu", "cuda"}
        )
        use_evo_attention = (
            self.use_deepspeed_evo and TritonEvoformer is not None and L > 24
        )
        if use_flex_attention and use_evo_attention:
            raise RuntimeError(
                "Cannot use both FlexAttention and EvoAttention at the same time. Please choose one or the other based on your hardware and input sizes."
            )
        if use_flex_attention:
            # FlexAttention
            A_I = flex_pairbias_attention(
                Q_IH,
                K_IH,
                V_IH,
                B_IIH,
                A_mask=None
                if flex_block_mask is not None
                else A_mask,  # NOTE: FlexAttention handles masking internally via the block mask, in which case we don't pass A_mask here.
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )
            A_I = A_I.transpose(1, 2).contiguous()  # [B, L, H, D]
            if not is_batched:
                A_I = A_I.squeeze(0)
            A_I = G_IH * A_I  # [B, L, H, D] or [L, H, D]
            A_I = A_I.flatten(start_dim=-2)  # [B, L, Ca] or [L, Ca]
        elif not use_evo_attention:
            # ManualAttention
            A_IIH = torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH

            if A_mask is not None:
                A_IIH = A_IIH.masked_fill(~A_mask[..., None], float("-inf"))

                # Bugfix: Empty queries need to have a constant value otherwise nans are in the forward graph. I don't
                # know why this causes instabilities because the invalid queries are masked out later. Oh well!
                invalid_queries = torch.logical_not(
                    torch.any(A_mask, dim=-1, keepdim=False)
                )  # [n_q,]
                A_IIH = A_IIH.masked_fill(invalid_queries[..., None, None], 0.0)

            A_IIH = torch.softmax(A_IIH, dim=-2)  # softmax over j
            ## G_IH: [I, H, C]
            ## A_IIH: [I, I, H]
            ## V_IH: [I, H, C]
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I  # [I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [I, Ca]
        else:
            # EvoAttention
            if is_batched:
                Q_IH = Q_IH.unsqueeze(1)
                K_IH = K_IH.unsqueeze(1)
                V_IH = V_IH.unsqueeze(1)
                pair_bias = B_IIH.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
                batch_size = B_IIH.shape[0]
            else:
                Q_IH = Q_IH.unsqueeze(0).unsqueeze(0)
                K_IH = K_IH.unsqueeze(0).unsqueeze(0)
                V_IH = V_IH.unsqueeze(0).unsqueeze(0)
                pair_bias = (
                    B_IIH.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).contiguous()
                )
                batch_size = 1

            res_mask = torch.zeros(
                (batch_size, 1, 1, 1, L),
                device=Z_II.device,
                dtype=pair_bias.dtype,
            )

            if A_mask is not None:
                A_mask = A_mask.to(dtype=torch.bool, device=Z_II.device)
                pair_bias = pair_bias.masked_fill(
                    ~A_mask[None, None, None], float("-inf")
                )

                invalid_queries = ~torch.any(A_mask, dim=-1)
                if invalid_queries.any():
                    pair_bias = pair_bias.clone()
                    pair_bias[:, :, :, invalid_queries, :] = 0.0

                valid_keys = torch.any(A_mask, dim=0)
                if not valid_keys.all():
                    res_mask = res_mask.masked_fill(
                        ~valid_keys[None, None, None, None, :], float("-inf")
                    )

            A_I = TritonEvoformer(Q_IH, K_IH, V_IH, res_mask, pair_bias)
            A_I = A_I.squeeze(1)
            if not is_batched:
                A_I = A_I.squeeze(0)
            A_I = G_IH * A_I  # [I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [I, Ca]

        A_I = self.to_a(A_I)

        return A_I


class PairformerBlock(nn.Module):
    """
    A replica of the AF3 architecture built from scratch.

    Args:
        c_s: The dimensionality of the sequence features S_I. If 0, the sequence features are not used and the attention bias is computed solely from the pair representation Z_II.
        c_z: The dimensionality of the pair representation Z_II.
        attention_pair_bias: A dictionary of arguments to initialize the AttentionPairBiasPairformerDeepspeed module, which computes the attention bias from the pair representation. This should include the keys 'n_head' and 'kq_norm', as well as any other arguments needed to initialize the module.
        p_drop: The dropout probability for the row and column dropout layers (not used in this implementation, but included for compatibility with the AF3 architecture).
        triangle_multiplication: A dictionary of arguments to initialize the triangle multiplication module, which updates the pair representation Z_II based on the current pair representation and the sequence features S_I. This is not implemented in this version, but can be added later for a more complete implementation of the AF3 architecture.
        triangle_attention: A dictionary of arguments to initialize the triangle attention module, which updates the pair representation Z_II based on the current pair representation and the sequence features S_I. This is not implemented in this version, but can be added later for a more complete implementation of the AF3 architecture.
        n_transition: The number of layers in the transition modules for both the sequence features S_I and the pair representation Z_II. The transition modules are simple feedforward networks that are applied to the sequence and pair features after the attention and pair bias updates, to further process the features before the next block. The number of layers in the transition modules can be adjusted to control the capacity of the model and the amount of processing applied to the features after the attention and pair bias updates.
        attn_backend: Attention backend used for the pair-bias attention. ``"manual"`` uses the original non-MegaFold implementation, ``"megafold"`` enables the Triton kernel-based MegaFold implementation when it is available, and ``"flex"`` sketches a FlexAttention-based implementation that can exploit block-diagonal sparsity.
        use_triangle_mult: Whether to include the triangle multiplication module, which updates the pair representation Z_II based on the current pair representation and the sequence features S_I. This is not implemented in this version, but can be added later for a more complete implementation of the AF3 architecture.
        use_triangle_attn: Whether to include the triangle attention module, which updates the pair representation Z_II based on the current pair representation and the sequence features S_I. This is not implemented in this version, but can be added later for a more complete implementation of the AF3 architecture.
    """

    def __init__(
        self,
        c_s,
        c_z,
        attention_pair_bias,
        p_drop=0.1,
        triangle_multiplication=None,
        triangle_attention=None,
        n_transition=4,
        attn_backend: Literal["manual", "megafold", "flex"] = "manual",
        use_triangle_mult=False,
        use_triangle_attn=False,
    ):
        super().__init__()
        self.attn_backend = attn_backend
        self.use_deepspeed_evo = attn_backend == "megafold"
        self.use_flex_attention = attn_backend == "flex"

        self.z_transition = Transition(c=c_z, n=n_transition)

        if c_s is not None and c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)

            self.attention_pair_bias = AttentionPairBiasPairformerDeepspeed(
                c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias
            )
            self.attention_pair_bias.use_deepspeed_evo = self.use_deepspeed_evo
            self.attention_pair_bias.use_flex_attention = self.use_flex_attention

        self.use_checkpointing = not DISABLE_CHECKPOINTING

    def forward_impl(
        self, S_I, Z_II, A_mask=None, flex_block_mask=None, flex_block_mask_ready=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Pairformer block."""
        _device = device_of(self)
        _use_autocast = _device.type != "mps"
        with torch.amp.autocast(
            device_type=_device.type, enabled=_use_autocast, dtype=torch.bfloat16
        ):
            Z_II = Z_II + self.z_transition(Z_II)
            if S_I is not None:
                S_I = S_I + self.attention_pair_bias(
                    S_I,
                    None,
                    Z_II,
                    Beta_II=torch.tensor([0.0], device=Z_II.device),
                    A_mask=A_mask,
                    flex_block_mask=flex_block_mask,
                    flex_block_mask_ready=flex_block_mask_ready
                    or flex_block_mask is not None,
                )
                S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II

    def forward(
        self, S_I, Z_II, A_mask=None, flex_block_mask=None, flex_block_mask_ready=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward pass of the Pairformer block, which consists of an attention and pair bias update for the sequence features S_I based on the pair representation Z_II, followed by transition layers for both S_I and Z_II. The attention and pair bias update is computed using the AttentionPairBiasPairformerDeepspeed module, which computes the attention bias from the pair representation Z_II and applies it to the attention computation for the sequence features S_I. The transition layers are simple feedforward networks that are applied to the sequence and pair features after the attention and pair bias updates, to further process the features before the next block.

        Args:
            S_I: Sequence features, shape [I, C_s]. This is the main input to the attention and pair bias update, and is updated based on the attention computation with the pair bias derived from Z_II.
            Z_II: Pair representation, shape [I, I, C_z]. This is used to compute the attention bias for the attention computation on S_I, and is also updated through a transition layer after the attention and pair bias update.
            A_mask: Optional attention mask, shape [I, I]. This can be used to mask out certain pairs of atoms from attending to each other in the attention computation for S_I, for example to enforce a local attention pattern or to mask out padding tokens.
            flex_block_mask: Optional precomputed block mask for the flex attention path. This can be used to speed up the computation of the block mask for the flex attention implementation, which can be expensive to compute on the fly for large inputs. If not provided, the block mask will be computed on the fly based on A_mask.
            flex_block_mask_ready: Whether the provided flex_block_mask is ready to be used. This should be set to True if the flex_block_mask is provided and can be used directly, or False if the flex_block_mask is not provided and needs to be computed on the fly based on A_mask

        Returns:
            S_I: Updated sequence features, shape [I, C_s]. This is the output of the attention and pair bias update, followed by a transition layer.
            Z_II: Updated pair representation, shape [I, I, C_z]. This is the output of a transition layer applied to the input pair representation Z_II, which can be further updated with triangle multiplication and triangle attention modules in a more complete implementation of the AF3 architecture.
        """
        forward = (
            activation_checkpointing(self.forward_impl)
            if self.use_checkpointing
            else self.forward_impl
        )
        return forward(
            S_I,
            Z_II,
            A_mask=A_mask,
            flex_block_mask=flex_block_mask,
            flex_block_mask_ready=flex_block_mask_ready,
        )
