"""
Conditional Equivariant Transformer (CT).

This is the `rep_model` inside `CET`: an 8-layer message-passing transformer
that takes atom-type embeddings + radial-basis-expanded edges and produces
both scalar (x) and equivariant vector (vec) per-atom features.

Lifted from `SelfConditionedDenoisingAtoms/models/ET_models/scet.py` and
re-pathed to use only relative imports. The forward path always builds
graphs in pure Python (`FastGraphGenerator`) — the upstream `legacy=True`
path that called the compiled C++ kernel is gone.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .conditioning import DropCond, DyT, JointDropPath, adaLN2
from .graph_utils import FastGraphGenerator
from .utils import (
    CosineCutoff,
    EquivariantLayerNorm,
    NeighborEmbedding,
    act_class_mapping,
    rbf_class_mapping,
    scatter,
)


class CondEquivMultiHeadAttention(nn.Module):
    """Equivariant multi-head attention layer with conditional pre-norm."""

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        vector_cutoff: bool = False,
        dtype: torch.dtype = torch.float32,
        p_droppath: float = 0.1,
        vec_prenorm: bool = False,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        # SCD additions
        self.conditional_ln = adaLN2(dim=hidden_channels)
        self.joint_droppath = JointDropPath(drop_prob=p_droppath)
        self.vec_norm = DyT(hidden_channels)
        self.vec_prenorm = vec_prenorm

        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False, dtype=dtype
        )

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3, dtype=dtype)
        self.vector_cutoff = vector_cutoff

        self.reset_parameters()

    def reset_parameters(self):
        self.vec_norm.reset_parameters()
        self.conditional_ln.reset_parameters()

        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def set_droppath(self, p: float):
        self.joint_droppath.drop_prob = p

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, cond):
        x, gate_x = self.conditional_ln(x, cond)
        if self.vec_prenorm:
            vec = self.vec_norm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            dim_size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec

        dx, dvec = self.joint_droppath(dx * gate_x, dvec)
        return dx, dvec

    def propagate(
        self,
        edge_index: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        vec: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
        d_ij: Tensor,
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        q_i = q.index_select(0, edge_index[1])
        k_j = k.index_select(0, edge_index[0])
        v_j = v.index_select(0, edge_index[0])
        vec_j = vec.index_select(0, edge_index[0])
        x, vec = self.message(q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij)
        return self.aggregate((x, vec), edge_index[1], dim_size=dim_size)

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Optional[Tensor],
        dv: Optional[Tensor],
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        cutoff = self.cutoff(r_ij).unsqueeze(1)
        attn = self.attn_activation(attn)

        # When `vector_cutoff` is set, multiply both scalar and vector pathways
        # by the cutoff so energy is continuous at the cutoff.
        if self.vector_cutoff:
            v_j = v_j * cutoff.unsqueeze(2)
        else:
            attn = attn * cutoff

        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        x = x * attn.unsqueeze(2)
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=0, dim_size=dim_size)
        vec = scatter(vec, index, dim=0, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class ConditionalET(nn.Module):
    """Conditional Equivariant Transformer (CT).

    Based on Thölke & de Fabritiis (ICLR 2022) — Equivariant Transformers — with
    the SCD additions: conditioning via adaLN2 + DropCond, drop-path on (x, v),
    optional post-norms.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = True,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_z: int = 100,
        max_num_neighbors: int = 32,
        check_errors: bool = True,
        box_vecs=None,
        vector_cutoff: bool = True,
        dtype: torch.dtype = torch.float32,
        layernorm_on_vec: bool = True,
        p_droppath: float = 0.1,
        p_dropcond: float = 0.2,
        inv_post_norm: bool = False,
        vec_post_norm: bool = False,
        vec_prenorm: bool = False,
        # NOTE: the upstream `legacy` flag is gone — that path called the
        # compiled torchmdnet C++ kernel which AFD no longer ships. The
        # parameter is still accepted (and ignored) for backwards-compatible
        # construction from the pretrained-checkpoint hparams dict.
        legacy: bool = False,
    ):
        super().__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.dtype = dtype
        self.max_num_neighbors = max_num_neighbors

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels, dtype=dtype)

        # Only graph builder we keep is the fast Python one. It handles
        # periodic + non-periodic in one pass. Accepts a `graph_batch` with
        # pos/batch/cell/pbc/natoms attached.
        self.graph_gen = FastGraphGenerator(
            cutoff=self.cutoff_upper,
            max_neighbors=self.max_num_neighbors,
            self_loops=True,
            alt_cell_key="box",
        )

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding_module = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z, dtype
            )
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()
        self.use_inv_post_norm = inv_post_norm
        self.use_vec_post_norm = vec_post_norm
        self.x_post_norm = nn.ModuleList()
        self.v_post_norm = nn.ModuleList()
        self.vec_prenorm = vec_prenorm
        for _ in range(num_layers):
            self.attention_layers.append(
                CondEquivMultiHeadAttention(
                    hidden_channels,
                    num_rbf,
                    distance_influence,
                    num_heads,
                    act_class,
                    attn_activation,
                    cutoff_lower,
                    cutoff_upper,
                    vector_cutoff,
                    dtype,
                    p_droppath=p_droppath,
                    vec_prenorm=vec_prenorm,
                )
            )
            self.x_post_norm.append(nn.LayerNorm(hidden_channels, dtype=dtype))
            self.v_post_norm.append(EquivariantLayerNorm(hidden_channels, dtype=dtype))

        self.layernorm_on_vec = layernorm_on_vec

        # SCD additions
        self.p_cond_dropout = p_dropcond
        self.drop_cond = DropCond(dim=hidden_channels, p_drop=self.p_cond_dropout)
        self.clip_value = 1e3
        self.ignore_cond = False

        self.reset_parameters()

    def ignore_conditioning(self, ignore: bool = True):
        self.ignore_cond = ignore

    def freeze_embeddings(self, freeze: bool = True):
        self.embedding.weight.requires_grad = not freeze
        self.drop_cond.mask_token.requires_grad = not freeze

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding_module is not None:
            self.neighbor_embedding_module.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()

        if self.x_post_norm is not None:
            for ln in self.x_post_norm:
                ln.reset_parameters()
        if self.v_post_norm is not None:
            for vln in self.v_post_norm:
                vln.reset_parameters()

    def init_graph(self, pos, batch, data_batch=None):
        """Resolve (edge_index, edge_distance, edge_distance_vec) for forward.

        Three sources, in priority order:
          1. A pre-computed graph attached to `data_batch` as
             (edge_index, edge_distance, edge_distance_vec). Useful when the
             same batch will be evaluated by multiple models.
          2. `data_batch` carrying `pos`/`batch`/`cell`/`pbc`/`natoms` — the
             standard path. Built via `FastGraphGenerator` (handles periodic
             and non-periodic in one pass).
          3. `data_batch is None` — bare `pos`/`batch` for a single
             non-periodic system. Wrap them into a minimal `Data` and reuse
             the same fast path.
        """
        if data_batch is not None:
            if hasattr(data_batch, "edge_index") and data_batch.edge_index is not None:
                assert (
                    hasattr(data_batch, "edge_distance")
                    and data_batch.edge_distance is not None
                ), "data_batch has edge_index but is missing edge_distance"
                assert (
                    hasattr(data_batch, "edge_distance_vec")
                    and data_batch.edge_distance_vec is not None
                ), "data_batch has edge_index but is missing edge_distance_vec"
                return (
                    data_batch.edge_index,
                    data_batch.edge_distance,
                    data_batch.edge_distance_vec,
                )

            out = self.graph_gen(data_batch)
            return out["edge_index"], out["edge_distance"], out["edge_distance_vec"]

        # data_batch is None: build a transient one with the bare minimum
        # attributes the fast builder needs (non-periodic only).
        from torch_geometric.data import Data

        n = pos.size(0)
        cell = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0) * (
            (pos.max() - pos.min()).clamp(min=1.0) + 1.0
        )
        natoms = torch.tensor([n], dtype=torch.long, device=pos.device)
        pbc = torch.zeros(1, 3, dtype=torch.bool, device=pos.device)
        bare = Data(
            pos=pos,
            batch=batch,
            cell=cell,
            natoms=natoms,
            pbc=pbc,
        )
        out = self.graph_gen(bare)
        return out["edge_index"], out["edge_distance"], out["edge_distance_vec"]

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        graph_batch=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.init_graph(
            pos, batch, data_batch=graph_batch
        )

        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        norm = torch.norm(edge_vec[mask], dim=1, keepdim=True)
        eps = 1e-8
        edge_vec[mask] = edge_vec[mask] / (norm + eps)

        if self.neighbor_embedding_module is not None:
            x = self.neighbor_embedding_module(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

        c = self.drop_cond(cond, batch)
        clip_scale = self.clip_value
        c = c.clamp(min=-clip_scale, max=clip_scale)

        if self.ignore_cond:
            c = None

        for l, attn in enumerate(self.attention_layers):
            x = x.clamp(min=-clip_scale, max=clip_scale)
            vec = vec.clamp(min=-clip_scale, max=clip_scale)

            dx, dvec = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec, cond=c
            )

            dx = dx.clamp(min=-clip_scale, max=clip_scale)
            dvec = dvec.clamp(min=-clip_scale, max=clip_scale)

            x = x + dx
            vec = vec + dvec
            if self.use_inv_post_norm:
                x = self.x_post_norm[l](x)
            if self.use_vec_post_norm:
                vec = self.v_post_norm[l](vec)

        if not self.use_inv_post_norm:
            x = self.x_post_norm[-1](x)
        if (not self.use_vec_post_norm) and self.layernorm_on_vec:
            vec = self.v_post_norm[-1](vec)

        return x, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper}), "
            f"dtype={self.dtype}"
        )


__all__ = ["ConditionalET", "CondEquivMultiHeadAttention"]
