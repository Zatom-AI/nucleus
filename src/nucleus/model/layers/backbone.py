from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from beartype.typing import Literal
from foundry.common import exists
from foundry.utils.ddp import RankedLogger
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias
from torch import nn

from nucleus.model.layers.blocks import Downcast, Upcast

global_logger = RankedLogger(__name__, rank_zero_only=False)

_CUEQ_MASK_INF = 1e6
_CUEQ_ATTENTION_PAIR_BIAS: Callable | None = None
_CUEQ_ATTENTION_PAIR_BIAS_IMPORT_ATTEMPTED = False


def _get_cueq_attention_pair_bias() -> Callable | None:
    """
    Lazily import cuEquivariance pair-bias attention when requested.

    Returns:
        The `attention_pair_bias` function from `cuequivariance_torch` if available, otherwise None.
    """
    global _CUEQ_ATTENTION_PAIR_BIAS
    global _CUEQ_ATTENTION_PAIR_BIAS_IMPORT_ATTEMPTED

    if not _CUEQ_ATTENTION_PAIR_BIAS_IMPORT_ATTEMPTED:
        _CUEQ_ATTENTION_PAIR_BIAS_IMPORT_ATTEMPTED = True
        try:
            from cuequivariance_torch import attention_pair_bias
        except Exception:
            _CUEQ_ATTENTION_PAIR_BIAS = None
        else:
            _CUEQ_ATTENTION_PAIR_BIAS = attention_pair_bias

    return _CUEQ_ATTENTION_PAIR_BIAS


def _mask_pair_bias(
    pair_bias: torch.Tensor, attn_mask: torch.Tensor | None
) -> torch.Tensor:
    """
    Apply cuEquivariance-style finite masking to a per-head pair bias.

    Args:
        pair_bias: Per-head bias tensor of shape [B, H, U, V].
        attn_mask: Optional valid attention mask with shape [B, U, V].

    Returns:
        Masked pair bias tensor of shape [B, H, U, V].
    """
    if attn_mask is None:
        return pair_bias
    return pair_bias.masked_fill(~attn_mask[:, None, :, :], -_CUEQ_MASK_INF)


def _attention_pair_bias_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    attn_mask: torch.Tensor | None,
    attn_scale: float,
) -> torch.Tensor:
    """
    PyTorch pair-bias attention matching cuEquivariance's torch fallback.

    Args:
        q: Query tensor with shape [B, H, U, DH].
        k: Key tensor with shape [B, H, V, DH].
        v: Value tensor with shape [B, H, V, DH].
        pair_bias: Per-head bias with shape [B, H, U, V].
        attn_mask: Optional valid attention mask with shape [B, U, V].
        attn_scale: Attention score scale.

    Returns:
        Attention output with shape [B, U, H * DH].
    """
    B, H, _, DH = q.shape
    pair_bias = _mask_pair_bias(pair_bias, attn_mask)

    attn = torch.einsum("bhid,bhjd->bhij", q, k)
    attn = attn * attn_scale
    attn = attn + pair_bias
    attn = attn.softmax(dim=-1)
    out = torch.einsum("bhij,bhjd->bihd", attn, v).reshape(B, -1, H * DH)
    return out


class SwiGLU(nn.Module):
    """
    Simple SwiGLU MLP with gating and no hidden nonlinearity.

    Args:
        c_in: Number of input channels.
        c_hidden: Number of hidden channels.
        dropout: Dropout rate (default is 0.0).
    """

    def __init__(self, c_in: int, c_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = linearNoBias(c_in, c_hidden * 2)
        self.out_proj = linearNoBias(c_hidden, c_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SwiGLU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor after applying SwiGLU activation.
        """
        a, b = self.in_proj(x).chunk(2, dim=-1)
        gate = F.silu(a)
        c = self.dropout(gate * b)
        return self.out_proj(c)


class GatedBottleneckAdaLN(nn.Module):
    """
    Adaptive layer normalization (AdaLN) with gating and bottleneck for conditioning.

    Args:
        c_cond: Number of channels for the conditioning input.
        c_bottleneck: Number of channels for the bottleneck in the MLP.
        n_params: Number of modulation parameters to output (default is 6 for shift, scale, and gate for both attention and MLP).
    """

    def __init__(self, c_cond: int, c_bottleneck: int, n_params: int = 6):
        super().__init__()

        # Instantiate parameters
        self.bottleneck_mlp = nn.Sequential(
            nn.SiLU(),
            linearNoBias(c_cond, c_bottleneck),
            nn.SiLU(),
        )
        self.cond_proj = linearNoBias(c_bottleneck, c_cond * n_params)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize certain parameters of the module."""
        nn.init.zeros_(self.cond_proj.weight)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Project `cond` to modulation parameters.

        Args:
            cond: Conditioning tensor.

        Returns:
            Tuple containing the modulation parameters.
        """
        z = self.bottleneck_mlp(cond)  # Improve weight utilization with bottleneck
        p = self.cond_proj(z)  # Zero-initialize conditioning parameters
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = p.chunk(6, dim=-1)
        gate_a = torch.sigmoid(gate_a)  # Stabilize attention residuals
        gate_m = torch.sigmoid(gate_m)  # Stabilize MLP residuals
        return scale_a, shift_a, gate_a, scale_m, shift_m, gate_m


def _drop_path(
    x: torch.Tensor, drop_prob: float, training: bool, scale_by_keep: bool = True
) -> torch.Tensor:
    """
    DropPath (Stochastic Depth) regularization.

    Adapted from PyTorch Image Models.
    Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping a path.
        training: Whether the model is in training mode.
        scale_by_keep: Whether to scale the output by the keep probability.

    Returns:
        Tensor with dropped paths during training, or the original tensor during evaluation.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DiTBlock(nn.Module):
    """
    Diffusion Transformer (DiT) block.

    Args:
        c_token: Number of channels for token features.
        c_bottleneck: Number of channels for the conditioning bottleneck in AdaLN.
        n_head: Number of attention heads.
        mlp_ratio: Ratio of hidden dimension to input dimension in the MLP.
        dropout: Dropout rate for attention outputs and MLP intermediate activations.
        drop_path: Drop path rate for stochastic depth.
        attn_backend: Attention backend to use ("manual" or "cueq").
    """

    def __init__(
        self,
        *,
        c_token: int,
        c_bottleneck: int,
        n_head: int,
        mlp_ratio: float = 8.0 / 3.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        attn_backend: Literal["manual", "cueq"] = "cueq",
    ):
        super().__init__()
        if c_token % n_head != 0:
            raise ValueError("c_token must be divisible by n_head.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        self.n_head = n_head
        self.head_dim = c_token // n_head
        self.mlp_dim = max(
            256, math.floor((c_token * mlp_ratio / 256) + 0.5) * 256
        )  # Choose the nearest multiple of 256, rounding up on ties (e.g., 384 -> 512)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout
        self.drop_path = drop_path
        self.attn_backend = attn_backend

        # NOTE: One should aim for `*_dim` to be a multiple of 32 for optimal compute performance.
        if self.head_dim % 32 != 0:
            global_logger.warning(
                f"Head dimension {self.head_dim} is not a multiple of 32. "
                "This may lead to suboptimal performance on some hardware."
            )
        if self.mlp_dim % 32 != 0:
            global_logger.warning(
                f"MLP dimension {self.mlp_dim} is not a multiple of 32. "
                "This may lead to suboptimal performance on some hardware."
            )

        self.ada_ln = GatedBottleneckAdaLN(c_token, c_bottleneck, n_params=6)
        self.norm_a = RMSNorm(c_token)

        self.qkv = linearNoBias(c_token, c_token * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = linearNoBias(c_token, c_token)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "_cueq_gate_weight",
            torch.zeros(c_token, c_token),
            persistent=False,
        )

        self.norm_m = RMSNorm(c_token)
        self.mlp = SwiGLU(c_token, self.mlp_dim, dropout=dropout)

    def _project_qkv(
        self, Q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project and normalize q/k/v tensors as [B, H, L, DH].

        Args:
            Q: Input tensor of shape [B, L, C].

        Returns:
            A tuple of (q, k, v) tensors, each of shape [B, H, L, DH].
        """
        B, L, _ = Q.shape
        q, k, v = self.qkv(Q).reshape(B, L, 3, self.n_head, self.head_dim).unbind(dim=2)
        q = self.q_norm(q).permute(0, 2, 1, 3).contiguous()
        k = self.k_norm(k).permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        return q, k, v

    def _project_attention_output(self, out: torch.Tensor) -> torch.Tensor:
        """
        Apply the DiT attention output projection and dropout.

        Args:
            out: Input tensor of shape [B, H, L, DH].

        Returns:
            Output tensor of shape [B, L, C].
        """
        return self.attn_dropout(self.proj(out))

    def _cueq_attention(
        self,
        Q: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        attn_mask: torch.Tensor | None,
        verbose: bool,
    ) -> torch.Tensor | None:
        """Run cuEquivariance attention with per-head pair bias.

        Args:
            Q: Input tensor of shape [B, L1, C].
            q: Query tensor of shape [B, H, L1, DH].
            k: Key tensor of shape [B, H, L2, DH].
            v: Value tensor of shape [B, H, L2, DH].
            pair_bias: Pair bias tensor of shape [B, H, L1, L2].
            attn_mask: Optional attention mask tensor of shape [B, L1, L2].
            verbose: Whether to print warnings.

        Returns:
            Output tensor of shape [B, L1, C] or None if cuEquivariance is unavailable.
        """
        attention_pair_bias = _get_cueq_attention_pair_bias()
        if attention_pair_bias is None:
            return None

        pair_bias = _mask_pair_bias(pair_bias, attn_mask)
        key_mask = torch.ones(
            (q.shape[0], k.shape[2]),
            dtype=torch.bool,
            device=Q.device,
        )
        proj_bias = getattr(self.proj, "bias", None)
        out, _ = attention_pair_bias(
            s=Q,
            q=q,
            k=k,
            v=v,
            z=pair_bias,
            mask=key_mask,
            num_heads=self.n_head,
            w_proj_z=None,
            w_proj_g=self._cueq_gate_weight,
            w_proj_o=self.proj.weight,
            w_ln_z=None,
            b_ln_z=None,
            b_proj_z=None,
            b_proj_g=None if proj_bias is None else 0.5 * proj_bias,
            b_proj_o=None,
            inf=_CUEQ_MASK_INF,
            attn_scale=self.attn_scale,
            return_z_proj=False,
            is_cached_z_proj=True,
        )

        # NOTE: We scale the output by 2 to match the PyTorch pair-bias attention path,
        # which does not include the `sigmoid(0) = 0.5` factor in the bias projection gating.
        return self.attn_dropout(2.0 * out)

    def _dense_attention(
        self,
        Q: torch.Tensor,
        P: torch.Tensor,
        attn_mask: torch.Tensor | None,
        verbose: bool,
    ) -> torch.Tensor:
        """Compute dense pair-bias attention over all keys.

        Args:
            Q: Input tensor of shape [B, L, C].
            P: Pair bias tensor of shape [B, L, L, H].
            attn_mask: Optional attention mask tensor of shape [B, L, L].
            verbose: Whether to print warnings.

        Returns:
            Output tensor of shape [B, L, C].
        """
        q, k, v = self._project_qkv(Q)
        pair_bias = P.permute(0, 3, 1, 2).contiguous()

        if self.attn_backend == "cueq":
            out = self._cueq_attention(
                Q=Q,
                q=q,
                k=k,
                v=v,
                pair_bias=pair_bias,
                attn_mask=attn_mask,
                verbose=verbose,
            )
            if out is not None:
                return out

        out = _attention_pair_bias_torch(
            q=q,
            k=k,
            v=v,
            pair_bias=pair_bias,
            attn_mask=attn_mask,
            attn_scale=self.attn_scale,
        )
        return self._project_attention_output(out)

    def _sparse_attention(
        self,
        Q: torch.Tensor,
        P: torch.Tensor,
        G: torch.Tensor,
        attn_mask: torch.Tensor | None,
        verbose: bool,
    ) -> torch.Tensor:
        """Compute local pair-bias attention using graph key indices.

        Args:
            Q: Input tensor of shape [B, L, C].
            P: Pair bias tensor of shape [B, L, L, H].
            G: Graph key indices tensor of shape [B, L, K].
            attn_mask: Optional attention mask tensor of shape [B, L, K].
            verbose: Whether to print warnings.

        Returns:
            Output tensor of shape [B, L, C].
        """
        B, L, _ = Q.shape
        if G.ndim == 2:
            G = G.unsqueeze(0).expand(B, -1, -1)
        if G.ndim != 3 or G.shape[:2] != (B, L):
            raise ValueError(
                f"G shape {G.shape} does not match expected shape [B, L, K] "
                f"with B={B} and L={L}."
            )
        if G.dtype != torch.long:
            G = G.to(dtype=torch.long)

        K_sparse = G.shape[-1]

        q, k, v = self.qkv(Q).reshape(B, L, 3, self.n_head, self.head_dim).unbind(dim=2)
        q = self.q_norm(q).permute(0, 2, 1, 3).contiguous()
        k = self.k_norm(k).reshape(B, L, -1)
        v = v.reshape(B, L, -1)

        batch_idx = torch.arange(B, device=Q.device).view(B, 1, 1)
        k_gathered = k[batch_idx, G].contiguous()
        v_gathered = v[batch_idx, G].contiguous()

        k_gathered = (
            k_gathered.reshape(B, L, K_sparse, self.n_head, self.head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(B * L, self.n_head, K_sparse, self.head_dim)
        )
        v_gathered = (
            v_gathered.reshape(B, L, K_sparse, self.n_head, self.head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(B * L, self.n_head, K_sparse, self.head_dim)
        )

        query_idx = (
            torch.arange(L, device=Q.device).view(1, L, 1).expand(B, L, K_sparse)
        )
        pair_bias = P[batch_idx, query_idx, G, :]
        pair_bias = (
            pair_bias.permute(0, 1, 3, 2)
            .reshape(B * L, self.n_head, 1, K_sparse)
            .contiguous()
        )

        if attn_mask is None:
            sparse_mask = None
        else:
            sparse_mask = attn_mask.reshape(B * L, 1, K_sparse)

        q = q.permute(0, 2, 1, 3).reshape(B * L, self.n_head, 1, self.head_dim)
        Q_flat = Q.reshape(B * L, 1, -1).contiguous()

        if self.attn_backend == "cueq":
            out = self._cueq_attention(
                Q=Q_flat,
                q=q,
                k=k_gathered,
                v=v_gathered,
                pair_bias=pair_bias,
                attn_mask=sparse_mask,
                verbose=verbose,
            )
            if out is not None:
                return out.reshape(B, L, -1)

        out = _attention_pair_bias_torch(
            q=q,
            k=k_gathered,
            v=v_gathered,
            pair_bias=pair_bias,
            attn_mask=sparse_mask,
            attn_scale=self.attn_scale,
        )
        out = out.reshape(B, L, -1)
        return self._project_attention_output(out)

    def _attention(
        self,
        Q: torch.Tensor,
        P: torch.Tensor,
        G: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute multi-head self-attention with per-head pair-bias conditioning.

        NOTE: If `G` is provided, local (sparse) attention is applied using
        the indices in `G`. Otherwise, global (dense) attention is applied.

        Args:
            Q: Input tensor of shape [batch, seq_len, c_token].
            P: Pairwise features tensor of shape [batch, seq_len, seq_len, n_head].
            G: Optional attention indices tensor of shape [batch, seq_len, n_attn_keys].
            attn_mask: Optional attention mask of sparse shape [batch, seq_len, n_attn_keys] or dense shape [batch, seq_len, seq_len].
            verbose: If True, log warnings about missing attention masks.

        Returns:
            Tensor after applying multi-head self-attention with per-head pair-bias conditioning.
        """
        B, L, _ = Q.shape
        if P.shape != (B, L, L, self.n_head):
            raise ValueError(
                f"P shape {P.shape} does not match expected shape "
                f"{(B, L, L, self.n_head)}."
            )

        if not exists(G) and not exists(attn_mask):
            if verbose:
                global_logger.warning(
                    "No attention mask provided for global attention. Proceeding without an attention mask."
                )
        if exists(attn_mask):
            assert (
                attn_mask.dtype == torch.bool
            ), "Attention mask must be of type torch.bool."
            if exists(G):
                assert (
                    attn_mask.shape == (Q.shape[0], Q.shape[1], G.shape[2])
                ), f"Sparse attention mask shape {attn_mask.shape} does not match expected shape {(Q.shape[0], Q.shape[1], G.shape[2])}."
            else:
                assert (
                    attn_mask.shape == (Q.shape[0], Q.shape[1], Q.shape[1])
                ), f"Dense attention mask shape {attn_mask.shape} does not match expected shape {(Q.shape[0], Q.shape[1], Q.shape[1])}."

        if exists(G):
            out = self._sparse_attention(
                Q=Q,
                P=P,
                G=G,
                attn_mask=attn_mask,
                verbose=verbose,
            )
        else:
            out = self._dense_attention(
                Q=Q,
                P=P,
                attn_mask=attn_mask,
                verbose=verbose,
            )

        return out

    def forward(
        self,
        Q: torch.Tensor,
        C: torch.Tensor,
        P: torch.Tensor,
        G: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the DiTBlock.

        Args:
            Q: Input tensor of shape [batch, seq_len, c_token].
            C: Conditioning tensor of shape [batch, seq_len, c_cond].
            P: Pairwise features tensor of shape [batch, seq_len, seq_len, n_head].
            G: Optional attention indices tensor of shape [batch, seq_len, n_attn_keys].
            attn_mask: Optional attention mask of sparse shape [batch, seq_len, n_attn_keys] or dense shape [batch, seq_len, seq_len].

        Returns:
            Tensor after applying attention and MLP with residual connections.
        """
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = self.ada_ln(C)
        Q_attn_in = self.norm_a(Q) * (1.0 + scale_a) + shift_a
        Q = Q + _drop_path(
            gate_a * self._attention(Q_attn_in, P, G=G, attn_mask=attn_mask),
            self.drop_path,
            self.training,
        )
        Q_mlp_in = self.norm_m(Q) * (1.0 + scale_m) + shift_m
        Q = Q + _drop_path(gate_m * self.mlp(Q_mlp_in), self.drop_path, self.training)
        return Q


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) composed of multiple DiTBlocks.

    Args:
        c_token: Number of channels for token features.
        c_bottleneck: Number of channels for the conditioning bottleneck in AdaLN.
        n_blocks: Number of DiTBlocks to stack.
        n_head: Number of attention heads in each DiTBlock.
        mlp_ratio: Ratio of hidden dimension to input dimension in the MLP.
        dropout: Dropout rate for attention outputs and MLP intermediate activations.
        drop_path: Drop path rate for stochastic depth.
        attn_backend: Attention backend to use ("manual" or "cueq").
    """

    def __init__(
        self,
        *,
        c_token: int,
        c_bottleneck: int,
        n_blocks: int,
        n_head: int,
        mlp_ratio: float = 8.0 / 3.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        attn_backend: Literal["manual", "cueq"] = "cueq",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    c_token=c_token,
                    c_bottleneck=c_bottleneck,
                    n_head=n_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path * l / max(n_blocks - 1, 1),
                    attn_backend=attn_backend,
                )
                for l in range(n_blocks)
            ]
        )

    def forward(
        self,
        Q: torch.Tensor,
        C: torch.Tensor,
        P: torch.Tensor,
        G: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Diffusion Transformer.

        Args:
            Q: Token features of shape [B, N, C_token].
            C: Conditioning features of shape [B, N, C_cond].
            P: Pairwise features of shape [B, N, N, n_head].
            G: Optional attention indices of shape [B, N, n_attn_keys].
            attn_mask: Optional attention mask of sparse shape [B, N, k] or dense shape [B, N, N].

        Returns:
            Updated token features of shape [B, N, C_token].
        """
        for block in self.blocks:
            Q = block(
                Q=Q,
                C=C,
                P=P,
                G=G,
                attn_mask=attn_mask,
            )
        return Q


class TransformerBackbone(nn.Module):
    """
    UNet-shaped Diffusion Transformer backbone for atom/token diffusion.

    Args:
        c_atom: Number of channels for atom features.
        c_s: Number of channels for token features.
        atom_encoder: Configuration dictionary for the atom encoder Transformer.
        atom_decoder: Configuration dictionary for the atom decoder Transformer.
        token_trunk: Configuration dictionary for the token trunk Transformer.
        downcast: Configuration dictionary for the downcast module.
        upcast: Configuration dictionary for the upcast module.
    """

    def __init__(
        self,
        *,
        c_atom: int,
        c_s: int,
        atom_encoder: dict,
        atom_decoder: dict,
        token_trunk: dict,
        downcast: dict,
        upcast: dict,
    ):
        super().__init__()
        self.atom_encoder = DiffusionTransformer(
            c_token=c_atom,
            **dict(atom_encoder),
        )
        self.downcast = Downcast(c_atom=c_atom, c_token=c_s, **downcast)
        self.token_trunk = DiffusionTransformer(c_token=c_s, **dict(token_trunk))
        self.upcast = Upcast(c_atom=c_atom, c_token=c_s, **upcast)
        self.atom_decoder = DiffusionTransformer(
            c_token=c_atom,
            **dict(atom_decoder),
        )

        # Try to import cuEquivariance attention pair-bias function
        attention_pair_bias = _get_cueq_attention_pair_bias()
        requesting_cueq = (
            atom_encoder.get("attn_backend") == "cueq"
            or token_trunk.get("attn_backend") == "cueq"
            or atom_decoder.get("attn_backend") == "cueq"
        )
        if attention_pair_bias is None and requesting_cueq:
            global_logger.warning(
                "attn_backend='cueq' was requested, but "
                "`cuequivariance_torch.attention_pair_bias` is unavailable. "
                "Falling back to the PyTorch pair-bias attention path."
            )

    def forward(
        self,
        *,
        Q_I: torch.Tensor,
        Z_II: torch.Tensor,
        S_I: torch.Tensor,
        Q_L: torch.Tensor,
        P_LL: torch.Tensor,
        C_L: torch.Tensor,
        G_I: torch.Tensor | None,
        G_L: torch.Tensor | None,
        f: dict,
    ) -> dict[str, torch.Tensor | bool | None]:
        """
        Process atom and token features through the shared Transformer backbone.

        Args:
            Q_I: Token features of shape [B, I, C_s=C_token].
            Z_II: Token pair features of shape [B, I, I, n_token_head].
            S_I: Token conditioning features of shape [B, I, C_s=C_token].
            Q_L: Atom features of shape [B, L, C_atom].
            P_LL: Atom pair features of shape [B, L, L, n_atom_head].
            C_L: Atom conditioning features of shape [B, L, C_atom].
            G_I: Optional attention indices for tokens [B, n_tokens, n_token_attn_keys]
            G_L: Optional attention indices for atoms [B, L, n_atom_attn_keys]
            f: Dictionary of additional metadata.

        Returns:
            Dictionary containing updated token and atom features.
        """
        tok_idx = f["atom_to_token_map"]
        atom_attn_mask = f["atom_pair_mask"]
        token_attn_mask = f["token_pair_mask"]
        Q_L = self.atom_encoder(
            Q=Q_L,
            C=C_L,
            P=P_LL,
            G=G_L,
            attn_mask=atom_attn_mask,
        )
        Q_I = self.downcast(Q_L=Q_L, A_I=Q_I, tok_idx=tok_idx)
        Q_I = self.token_trunk(
            Q=Q_I,
            C=S_I,
            P=Z_II,
            G=G_I,
            attn_mask=token_attn_mask,
        )
        Q_L = self.upcast(Q_L=Q_L, A_I=Q_I, tok_idx=tok_idx)
        Q_L = self.atom_decoder(Q=Q_L, C=C_L, P=P_LL, G=G_L, attn_mask=atom_attn_mask)
        return {
            "Q_I": Q_I,
            "Q_L": Q_L,
        }
