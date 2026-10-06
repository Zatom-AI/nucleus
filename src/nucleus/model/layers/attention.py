import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing
from opt_einsum import contract as einsum

from nucleus import DISABLE_CHECKPOINTING
from nucleus.model.layers.block_utils import (
    create_attention_indices,
)
from nucleus.utils.attention import create_flex_attention_block_mask

try:
    from rfd3.model.layers.block_utils import (
        indices_to_mask,
    )
    from rfd3.model.layers.layer_utils import (
        AdaLN,
        LinearBiasInit,
        RMSNorm,
        linearNoBias,
    )
except ImportError:
    raise ImportError(
        "Could not import RFD3 modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
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


######################################################################################
##########################        Network Modules           ##########################
######################################################################################


class LocalAttentionPairBias(nn.Module):
    """
    Local attention with pairwise bias, where each query attends to a fixed number of
    neighbouring keys as specified by indices. Supports optional conditioning and
    kernelized attention.

    Args:
        c_a: int, dimension of input atom features (d_model)
        c_s: int, dimension of input sequence features
        c_pair: int, dimension of pairwise features
        n_head: int, number of attention heads
        kq_norm: bool, whether to apply RMSNorm to queries and keys
        n_attn_seq_neighbours: int, number of neighbouring keys to attend to
        n_attn_keys: int, number of keys to attend to if no indices are provided (for sparse input batches)
        attn_backend: str, attention backend to use ("manual", "megafold", or "flex")
    """

    def __init__(
        self,
        c_a,
        c_s,
        c_pair,
        n_head,
        kq_norm=True,
        n_attn_seq_neighbours=2,
        n_attn_keys=128,
        attn_backend="manual",
    ):
        super().__init__()
        self.c = c_a  # d_model dim same as input features
        self.n_head = n_head

        self.to_q = linearNoBias(c_a, self.c)
        self.to_k = linearNoBias(c_a, self.c)
        self.to_v = linearNoBias(c_a, self.c)
        self.to_b = linearNoBias(c_pair, self.n_head)
        self.to_g = nn.Sequential(
            linearNoBias(c_a, self.c, bias=False),
            nn.Sigmoid(),
        )
        self.kq_norm = kq_norm
        if kq_norm:
            self.ln_q = RMSNorm(self.c)
            self.ln_k = RMSNorm(self.c)

        # Output / Input projections
        self.to_o = linearNoBias(self.c, c_a)  # from attn to Q_L

        # Conditioned
        if exists(c_s):
            self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
            self.linear_output_project = nn.Sequential(
                LinearBiasInit(c_s, c_a, biasinit=-2.0),
                nn.Sigmoid(),
            )
        else:
            self.ln_1 = RMSNorm(c_a)

        # Used if no indices are provided
        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_flex_attention = attn_backend == "flex"
        self.use_checkpointing = not DISABLE_CHECKPOINTING

    def forward(
        self,
        Q_L,
        C_L,
        P_LL,
        indices=None,
        index_mask=None,
        f=None,
        X_L=None,
        tok_idx=None,
        full=False,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        flex_block_mask=None,
        flex_block_mask_ready=False,
    ) -> torch.Tensor:
        """
        Compute local attention with pairwise bias. If indices are not provided, they will be created on the fly from input features f and X_L. Supports both full attention (for training) and sparse attention (for inference) based on the full flag.

        Args:
            Q_L: [D, L, c_a]
            C_L: [D, L, c_s]
            P_LL: [D, L, L, c_pair] or None (if using chunked mode)
            indices: [D, L, k] long
            index_mask: [D, L, k] bool indicating which sparse slots contain real keys
            f: dict of input features (used to create indices if not provided)
            X_L: [D, L, c_a] (used to create indices if not provided)
            tok_idx: [L] token indices for the query/key positions
            full: bool, whether to compute full attention matrix (for training) or sparse attention (for inference)
            chunked_pairwise_embedder: ChunkedPairwiseEmbedder for memory efficient computation
            initializer_outputs: Dict containing features for chunked computation
            flex_block_mask: optional precomputed FlexAttention block_mask (as returned by `create_flex_attention_block_mask` / `torch.nn.attention.flex_attention.create_block_mask`). When provided, it will be forwarded to the flex attention kernel.
            flex_block_mask_ready: set True once the caller has either computed `flex_block_mask` (or determined no block mask is needed / available), so downstream layers do not attempt to build it again.

        Returns:
            Q_L: [D, L, c_a]
        """

        # If no indices are provided, prepare indices from
        if not exists(indices):
            indices, index_mask = create_attention_indices(
                f,
                n_attn_keys=self.n_attn_keys,
                n_attn_seq_neighbours=self.n_attn_seq_neighbours,
                X_L=X_L,
                tok_idx=tok_idx,
            )
        attn_mask = infer_attention_mask_from_features(
            f, indices, tok_idx=tok_idx, index_mask=index_mask
        )

        # Handle chunked P_LL computation
        if chunked_pairwise_embedder is not None and P_LL is None:
            # Compute sparse P_LL only for the attention indices
            raise NotImplementedError(
                "Chunked P_LL computation not implemented yet for sparse input batches."
            )
            P_LL_sparse = chunked_pairwise_embedder.forward_chunked(
                f=f,
                indices=indices,
                C_L=initializer_outputs["C_L"],
                Z_init_II=initializer_outputs["Z_II"],
                tok_idx=f["atom_to_token_map"],
            )
            # P_LL_sparse is already in sparse format [D, L, k, c_pair]
            use_sparse_pll = True
        else:
            # Original full P_LL computation
            P_LL_sparse = None
            use_sparse_pll = False

        def do_attention_impl(Q_L, C_L, P_LL) -> torch.Tensor:
            """
            Apply attention with optional conditioning and kernelized pairwise bias.

            Args:
                Q_L: [D, L, c_a]
                C_L: [D, L, c_s] or None
                P_LL: [D, L, L, c_pair] or [D, L, k, c_pair] (if use_sparse_pll)

            Returns:
                Q_L: [D, L, c_a]
            """
            if exists(C_L):
                Q_L = self.ada_ln_1(Q_L, C_L)
            else:
                Q_L = self.ln_1(Q_L)

            use_flex_attention = (
                self.use_flex_attention
                and torch_flex_attention is not None
                and create_block_mask is not None
                and not use_sparse_pll
                and Q_L.shape[-2] > 24
                and Q_L.device.type in {"cpu", "cuda"}
            )

            q, k, v, g = (
                self.to_q(Q_L),
                self.to_k(Q_L),
                self.to_v(Q_L),
                self.to_g(Q_L),
            )
            q, k = (self.ln_q(q), self.ln_k(k)) if self.kq_norm else (q, k)

            if use_flex_attention:
                # Sparse FlexAttention path
                b = self.to_b(P_LL)
                if b.ndim == 3:
                    b = b.permute(2, 0, 1).unsqueeze(0).contiguous()
                elif b.ndim == 4:
                    b = b.permute(0, 3, 1, 2).contiguous()
                else:
                    raise ValueError(
                        "pair_bias must be [L, L, H] or [B, L, L, H] for the flex attention path."
                    )
                if b.shape[0] == 1 and Q_L.shape[0] > 1:
                    # Token initialization emits a shared [L, L, H] pair bias that should
                    # broadcast across the batch, matching the manual attention path.
                    b = b.expand(Q_L.shape[0], -1, -1, -1).contiguous()
                attn_mask_dense = indices_to_mask_with_padding(indices, attn_mask)
                attn_out = flex_local_pairbias_attention(
                    Q_L=q,
                    K_L=k,
                    V_L=v,
                    pair_bias=b,
                    A_mask=attn_mask_dense,
                    G_L=g,
                    H=self.n_head,
                    flex_block_mask=flex_block_mask,
                    flex_block_mask_ready=flex_block_mask_ready,
                )
            else:
                # ManualAttention path
                if use_sparse_pll:
                    # Sparse ManualAttention path: Use pre-computed sparse P_LL (already gathered)
                    b = self.to_b(P_LL_sparse)  # [D, L, k, H]
                    attn_out = sparse_pairbias_attention(
                        Q=q,
                        K=k,
                        V=v,
                        B=b,
                        G=g,
                        gather_bias=False,  # Already gathered!
                        indices=indices,
                        attn_mask=attn_mask,
                        H=self.n_head,
                        full=full,
                    )  # [D, L, c]
                else:
                    # Full ManualAttention path
                    b = self.to_b(P_LL)
                    attn_out = sparse_pairbias_attention(
                        Q=q,
                        K=k,
                        V=v,
                        B=b,
                        G=g,
                        gather_bias=True,
                        indices=indices,
                        attn_mask=attn_mask,
                        H=self.n_head,
                        full=full,
                    )  # [D, L, c]

            # Output projection (from adaLN-Zero)
            Q_L = self.to_o(attn_out)
            if exists(C_L):
                Q_L = self.linear_output_project(C_L) * Q_L

            return Q_L

        do_attention = (
            activation_checkpointing(do_attention_impl)
            if self.use_checkpointing
            else do_attention_impl
        )

        # Call attention with appropriate P_LL
        if use_sparse_pll:
            return do_attention(Q_L, C_L, P_LL_sparse)
        else:
            return do_attention(Q_L, C_L, P_LL)


class ResidualGatedCrossAttention(nn.Module):
    """
    Residual gated cross-attention module with optional pairwise bias.
    This module computes attention between a query tensor and a
    key-value tensor, incorporating gating mechanisms and optional
    pairwise bias.

    Inspired by Algorithm 7 of Emyx.
    Reference: https://arxiv.org/abs/2606.19377v1

    Args:
        c_query: int, dimension of the query features
        c_kv: int, dimension of the key-value features
        c_pair: int, dimension of the pairwise features (optional)
        c_model: int, dimension of the model (default: 128)
        n_head: int, number of attention heads (default: 4)
        kq_norm: bool, whether to apply RMSNorm to queries and keys (default: False)
        dropout: float, dropout rate for the output projection (default: 0.0)
    """

    def __init__(
        self,
        c_query,
        c_kv,
        c_pair=None,
        c_model=128,
        n_head=4,
        kq_norm=False,
        kv_residual_connection=False,
        dropout=0.0,
        **_,
    ):
        super().__init__()
        self.n_head = n_head
        head_dim = c_model // n_head
        self.scale = 1 / math.sqrt(head_dim)
        assert c_model % n_head == 0, "c_model must be divisible by n_heads"

        # Parameters instantiation
        self.ln_q = RMSNorm(c_query)
        self.ln_kv = RMSNorm(c_kv)

        self.to_q = linearNoBias(c_query, c_model)
        self.to_k = linearNoBias(c_kv, c_model)
        self.to_v = linearNoBias(c_kv, c_model)
        self.to_g = nn.Sequential(
            linearNoBias(head_dim, head_dim),
            nn.Sigmoid(),
        )
        self.to_out = nn.Sequential(linearNoBias(c_model, c_query), nn.Dropout(dropout))
        self.kq_norm = kq_norm
        self.kv_residual_connection = kv_residual_connection
        if self.kq_norm:
            self.k_norm = RMSNorm(c_model)
            self.q_norm = RMSNorm(c_model)
        if self.kv_residual_connection:
            self.to_kv = nn.Sequential(RMSNorm(c_kv), linearNoBias(c_kv, c_kv))

        self.c_pair = c_pair
        if c_pair is not None:
            self.to_b = nn.Sequential(RMSNorm(c_pair), linearNoBias(c_pair, n_head))

        # Parameters initialization
        self.reset_parameter()

    def reset_parameter(self):
        """Initialize parameters using Xavier uniform initialization."""
        # Query/key/value projection: Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_g[0].weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)

    def forward(self, q, kv, attn_mask=None, pair_bias=None):
        """
        Compute residual gated cross-attention with optional pairwise bias.

        Args:
            q: Query tensor of shape [B, tok, n_q, c_query]
            kv: Key-value tensor of shape [B, tok, n_kv, c_kv]
            attn_mask: Attention mask of shape [n_q, n_kv]

        Returns:
            attn_out: Output tensor of shape [B, tok, n_q, c_query]
        """
        if self.kv_residual_connection:
            kv = kv + self.to_kv(kv)

        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        q, k, v = self.to_q(q), self.to_k(kv), self.to_v(kv)

        if self.kq_norm:
            k = self.k_norm(k)
            q = self.q_norm(q)

        q, k, v = map(
            lambda t: rearrange(t, "b t n (h c) -> b h t n c", h=self.n_head),
            (q, k, v),
        )  # [B, tok, n, heads, c] ->  [B, heads, tok, n, c]

        attn = einsum("bhtqc,bhtkc->bhtqk", q, k) * self.scale

        if pair_bias is not None:
            b = self.to_b(pair_bias)
            b = rearrange(b, "b t q k (h) -> b (h) t q k", h=self.n_head)
            attn = attn + b

        # Invalid query handling:
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[None, None], float("-inf"))

            # Bugfix: Empty queries need to have a constant value otherwise nans are in the forward graph. I don't
            # know why this causes instabilities because the invalid queries are masked out later. Oh well!
            invalid_queries = torch.logical_not(
                torch.any(attn_mask, dim=-1, keepdim=False)
            )  # [n_q,]
            attn[:, :, invalid_queries, :] = 0.0

        attn = F.softmax(attn, dim=-1)
        attn_out = einsum("bhtqk,bhtkd->bhtqd", attn, v)

        g = self.to_g(attn_out)
        attn_out = attn_out * g

        attn_out = rearrange(attn_out, "b h t n c -> b t n (h c)")
        attn_out = self.to_out(attn_out)  # [B, n_tok, n_k, c]

        return attn_out


######################################################################################
##########################        Kernel Functions          ##########################
######################################################################################


def flex_local_pairbias_attention(
    Q_L: torch.Tensor,
    K_L: torch.Tensor,
    V_L: torch.Tensor,
    pair_bias: torch.Tensor,
    A_mask: torch.Tensor | None,
    G_L: torch.Tensor | None = None,
    H: int | None = None,
    block_size: int = 64,
    flex_block_mask=None,
    flex_block_mask_ready=False,
    blocks_are_contiguous=False,
    rows_guaranteed_safe=False,
) -> torch.Tensor:
    """FlexAttention-based pair-bias local attention path."""

    if torch_flex_attention is None or create_block_mask is None:
        raise RuntimeError(
            "torch.nn.attention.flex_attention is not available in this environment."
        )

    if Q_L.ndim == 3:
        if H is None:
            H = pair_bias.shape[-1]
        if Q_L.shape[-1] % H != 0:
            raise ValueError(
                f"Expected Q_L.shape[-1] ({Q_L.shape[-1]}) to be divisible by H ({H})."
            )
        head_dim = Q_L.shape[-1] // H
        q = (
            Q_L.reshape(Q_L.shape[0], Q_L.shape[1], H, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            K_L.reshape(K_L.shape[0], K_L.shape[1], H, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            V_L.reshape(V_L.shape[0], V_L.shape[1], H, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        if G_L is not None:
            G_L = (
                G_L.reshape(G_L.shape[0], G_L.shape[1], H, head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
    elif Q_L.ndim == 2:
        if H is None:
            H = pair_bias.shape[-1]
        if Q_L.shape[-1] % H != 0:
            raise ValueError(
                f"Expected Q_L.shape[-1] ({Q_L.shape[-1]}) to be divisible by H ({H})."
            )
        head_dim = Q_L.shape[-1] // H
        q = (
            Q_L.reshape(Q_L.shape[0], H, head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        k = (
            K_L.reshape(K_L.shape[0], H, head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        v = (
            V_L.reshape(V_L.shape[0], H, head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        pair_bias = pair_bias.permute(2, 0, 1).unsqueeze(0).contiguous()
        if G_L is not None:
            G_L = (
                G_L.reshape(G_L.shape[0], H, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
    else:
        raise ValueError("Q_L must be [B, L, C] or [L, C] for the flex attention path.")

    invalid_queries = None
    if A_mask is not None:
        A_mask = A_mask.to(dtype=torch.bool, device=q.device)
        if A_mask.ndim == 3 and A_mask.shape[0] == 1 and q.shape[0] > 1:
            # Local attention can share one mask across a batch when the index layout
            # is identical; broadcast the singleton mask to match the batched queries.
            A_mask = A_mask.expand(q.shape[0], -1, -1).contiguous()
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
        bias = pair_bias[b, h, q_idx, kv_idx]
        if A_mask is None:
            return score + bias

        if A_mask.ndim == 2:
            allowed = A_mask[q_idx, kv_idx]
        elif A_mask.ndim == 3:
            allowed = A_mask[b, q_idx, kv_idx]
        else:
            raise ValueError(
                "A_mask must be None, [L, L], or [B, L, L] for the flex attention path."
            )

        score = score + bias
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

    if G_L is not None:
        out = out * G_L

    if A_mask is not None and invalid_queries is not None:
        if invalid_queries.ndim == 1:
            out = out.masked_fill(invalid_queries[None, None, :, None], 0.0)
        else:
            out = out.masked_fill(invalid_queries[:, None, :, None], 0.0)

    if Q_L.ndim == 3:
        out = (
            out.permute(0, 2, 1, 3).contiguous().reshape(Q_L.shape[0], Q_L.shape[1], -1)
        )
    else:
        out = out.squeeze(0).permute(1, 0, 2).contiguous().reshape(Q_L.shape[0], -1)

    return out


def sparse_pairbias_attention(
    Q, K, V, B, indices, H, gather_bias=True, G=None, full=False, attn_mask=None
) -> torch.Tensor:
    """
    Compute attention with sparse pairwise bias, where indices specify which
    keys to attend to for each query token.

    Args:
        Q: (D, L, c)  # query vectors
        K: (D, L, c)  # key vectors
        V: (D, L, c)  # value vectors
        B: (L, L, H)  # attention bias (unbatched or pre-gathered and [D, L, k, H])
        G: (D, L, c)  # Gate (optional)
        B2: (D, L, 14, 14, H)  # attention bias (batched and within token) (optional)
        indices: (D, L, k_neigh) long  # indices of neighbours to attend to
        attn_mask: (D, L, k_neigh) bool  # True for real keys and False for padded keys

    Returns:
        attn_out: (D, L, c)  # attention output
    """
    D, L, c = Q.shape
    k = indices.shape[-1]  # k_neigh

    if full:
        # During training, compute full attention matrix to create a more optimized torch.tensor graph.
        valid_mask = indices_to_mask_with_padding(indices, attn_mask)
        if not valid_mask.any(dim=-1).all():
            raise ValueError("Each attention query must have at least one valid key.")
        return pairbias_attention_(
            Q=Q,
            K=K,
            V=V,
            B=B,
            H=H,
            valid_mask=valid_mask,
            G=G,
        )

    # Pull vectors from dimension 1 into index torch.tensor according to unique k_neigh axis
    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)  # (D,1,1)
    K_gathered = K[batch_idx, indices].contiguous()  # (D, L, k, c)
    V_gathered = V[batch_idx, indices].contiguous()  # (D, L, k, c)

    # Gather bias or assume pre-gathered
    if gather_bias:
        query_idx = torch.arange(L, device=Q.device).view(1, L, 1)  # (1,L,1)
        query_idx = query_idx.expand(D, -1, k)
        if B.ndim == 3:
            B_gathered = B[query_idx, indices, :]  # (D, L, k, H)
        elif B.ndim == 4:  # (D, L, L, H)
            B_gathered = B[batch_idx, query_idx, indices, :]  # (D, L, k, H)
    else:
        assert B.shape == (D, L, k, H), "B must be batched with shape (D, L, k, H)"
        B_gathered = B
    B_gathered = B_gathered.contiguous()

    # Split into heads
    Q = Q.reshape(D, L, H, c // H)
    K_gathered = K_gathered.reshape(D, L, k, H, c // H)
    V_gathered = V_gathered.reshape(D, L, k, H, c // H)
    B_gathered = B_gathered.reshape(D, L, k, H)
    Q = Q.permute(0, 2, 1, 3)  # [D, H, L, c // H]
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)
    B_gathered = B_gathered.permute(0, 3, 1, 2)

    # Do attention
    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)
    attn = attn / sqrt(c // H)  # scale
    attn = attn + B_gathered  # add bias
    if exists(attn_mask):
        if not attn_mask.any(dim=-1).all():
            raise ValueError("Each attention query must have at least one valid key.")
        attn = attn.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
    attn = torch.softmax(attn, dim=-1)  # softmax over keys [D, H, L, k]
    attn_out = torch.einsum(
        "...ij,...ijc->...ic", attn, V_gathered
    )  # allocates a max of 4.95 GiB.

    # Optional gating
    if G is not None:
        G = G.reshape(D, L, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L, c).contiguous()

    return attn_out  # [D, L, c]


@torch.no_grad()
def infer_attention_mask_from_features(f, indices, tok_idx=None, index_mask=None):
    """
    Infer sparse attention validity from f["unindexing_pair_mask"] for each
    sparse query/key slot.

    Args:
        f: dict of input features (must contain "unindexing_pair_mask")
        indices: [D, L, k] long tensor of attention indices
        tok_idx: [L] long tensor of token indices for the query/key positions (optional)
        index_mask: optional [D, L, k] boolean tensor marking real sparse slots

    Returns:
        attn_mask: [D, L, k] boolean tensor indicating valid attention positions
    """
    if index_mask is not None:
        index_mask = index_mask.to(device=indices.device, dtype=torch.bool)

    if f is None or "unindexing_pair_mask" not in f:
        return index_mask

    if tok_idx is None:
        tok_idx = f.get("atom_to_token_map")
    if tok_idx is None:
        return index_mask

    tok_idx = tok_idx.to(device=indices.device, dtype=torch.long)
    unindexing_pair_mask = f["unindexing_pair_mask"].to(
        device=indices.device, dtype=torch.bool
    )

    if indices.ndim == 2:
        query_token_idx = tok_idx[:, None]
        key_token_idx = tok_idx[indices]
    elif indices.ndim == 3:
        query_token_idx = tok_idx[None, :, None]
        key_token_idx = tok_idx[indices]
    else:
        raise ValueError(f"Expected ndim 2 or 3, got {indices.ndim}")

    feature_mask = ~unindexing_pair_mask[query_token_idx, key_token_idx]
    return feature_mask if index_mask is None else feature_mask & index_mask


@torch.no_grad()
def prepare_token_attention_inputs(
    f: dict,
    X_I: torch.Tensor,
    *,
    use_local_attention: bool,
    n_attn_keys: int,
    n_attn_seq_neighbours: int,
) -> tuple[torch.Tensor | None, dict]:
    """Build token graph indices and their matching attention mask.

    Args:
        f: Input feature dictionary.
        X_I: Token representative coordinates of shape [B, I, 3].
        use_local_attention: Whether to create a sparse token graph.
        n_attn_keys: Maximum number of sparse attention keys.
        n_attn_seq_neighbours: Number of sequence neighbours to force into the graph.

    Returns:
        A pair containing optional sparse indices [B, I, K] and a shallow copy
        of ``f`` whose ``token_pair_mask`` is the matching sparse [B, I, K] or
        dense [B, I, I] boolean attention mask.
    """
    batch_size, n_tokens = X_I.shape[:2]
    token_idx = torch.arange(n_tokens, device=X_I.device)

    if use_local_attention:
        indices, index_mask = create_attention_indices(
            X_L=X_I,
            f=f,
            tok_idx=token_idx,
            n_attn_keys=n_attn_keys,
            n_attn_seq_neighbours=n_attn_seq_neighbours,
        )
        attn_mask = index_mask
    else:
        indices = None
        token_pair_mask = f["token_pair_mask"].to(device=X_I.device, dtype=torch.bool)
        if token_pair_mask.shape != (n_tokens, n_tokens):
            raise ValueError(
                f"token_pair_mask shape {token_pair_mask.shape} does not match "
                f"expected shape {(n_tokens, n_tokens)}."
            )
        attn_mask = token_pair_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return indices, {**f, "token_pair_mask": attn_mask}


@torch.no_grad()
def indices_to_mask_with_padding(indices, attn_mask=None):
    """Convert sparse indices to a dense mask, ignoring padded sparse slots."""
    if attn_mask is None:
        return indices_to_mask(indices)

    indices = indices.to(dtype=torch.long)
    attn_mask = attn_mask.to(dtype=torch.bool)

    if indices.ndim == 2:
        L = indices.shape[0]
        mask_out = torch.zeros((L, L), dtype=torch.int64, device=indices.device)
        mask_out.scatter_add_(1, indices, attn_mask.to(dtype=torch.int64))
    elif indices.ndim == 3:
        B, L, _ = indices.shape
        mask_out = torch.zeros((B, L, L), dtype=torch.int64, device=indices.device)
        mask_out.scatter_add_(2, indices, attn_mask.to(dtype=torch.int64))
    else:
        raise ValueError(f"Expected ndim 2 or 3, got {indices.ndim}")

    return mask_out > 0


def pairbias_attention_(Q, K, V, B, H, valid_mask=None, G=None) -> torch.Tensor:
    """
    Fully connected variant of pairbias attention with optional gating and valid mask.
    Equivalent to sparse attention but with all keys

    Args:
        Q: [batch_size, query_length, d_model]
        K: [batch_size, key_length, d_model]
        V: [batch_size, key_length, d_model]
        B: [batch_size, query_length, key_length, H] or [query_length, key_length, H] (unbatched)
        valid_mask: [batch_size, query_length, key_length] boolean mask indicating valid attention positions (optional)
        G: [batch_size, query_length, d_model] optional gating vector

    Returns:
        attn_out: [batch_size, query_length, H * head_dim]
    """
    D, L, c = Q.shape
    k = L

    # Split into heads
    Q = Q.reshape(D, L, H, c // H)
    K = K.reshape(D, k, H, c // H)
    V = V.reshape(D, k, H, c // H)
    B = B.reshape(D, L, k, H)

    # Flip heads upwards [..., H, d_model] -> [B, H, ..., d_model]
    Q = Q.permute(0, 2, 1, 3)  # [D, H, L, c // H]
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    B = B.permute(0, 3, 1, 2)

    # Do attention
    attn = torch.einsum("...ld,...kd->...lk", Q, K)
    attn = attn / sqrt(c // H)  # scale
    attn = attn + B  # add bias
    if exists(valid_mask):
        # expand valid mask over heads [D, H, L, L]
        attn = attn.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
    attn = torch.softmax(attn, dim=-1)  # softmax over keys [D, H, L, k]
    attn_out = torch.einsum("...ij,...jc->...ic", attn, V)

    # Optional gating
    if G is not None:
        G = G.reshape(D, L, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L, c).contiguous()

    return attn_out
