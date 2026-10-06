"""
Top-level CET wrapper plus shared `BaseModel` and `AccumulatedNormalization`.

Lifted from `SelfConditionedDenoisingAtoms/models/ET_models/scd_model.py`,
with the unused `ET` and `CFrad` variants removed and `AccumulatedNormalization`
moved above `BaseModel` so the forward reference is now a normal one.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.autograd import grad

from . import priors
from .conditioning import ProjHead2
from .output_modules import EquivariantScalar, EquivariantVector
from .scet import ConditionalET


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor (used to z-score the per-atom noise)."""

    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super().__init__()
        self._epsilon = epsilon
        self._acc_shape = accumulator_shape
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def reset(self):
        self.acc_sum = torch.zeros(self._acc_shape)
        self.acc_squared_sum = torch.zeros(self._acc_shape)
        self.acc_count = torch.zeros((1,))
        self.num_accumulations = torch.zeros((1,))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        var = (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        return torch.sqrt(var.clamp(min=self._epsilon)).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor, update: bool = False):
        if self.training and update:
            self.update_statistics(batch)
        return (batch - self.mean) / (self.std + 1e-8)

    def inverse(self, batch: torch.Tensor):
        return batch * (self.std + 1e-8) + self.mean


class BaseModel(nn.Module):
    """Base class shared by the (single) extracted CET variant.

    The upstream code defined this for ET, CET, and CFrad alike. Here we keep
    it because it cleanly factors the heads (noise, scalar, embedding) and the
    forward pass.
    """

    def __init__(
        self,
        emb_dim: int,
        derivative: bool = False,
        activation: str = "silu",
        aggregation: str = "sum",
        emb_agg: str = "sum",
        head_agg: str = "sum",
        dtype: torch.dtype = torch.float32,
        prior_model: Optional[nn.Module] = None,
        mean=None,
        std=None,
        add_head_to_pred: bool = False,
    ):
        super().__init__()

        self.derivative = derivative
        self.add_head_to_pred = add_head_to_pred
        self.emb_dim = emb_dim

        if isinstance(prior_model, priors.BasePrior):
            prior_model = [prior_model]
        self.prior_model = (
            None
            if prior_model is None
            else torch.nn.ModuleList(prior_model).to(dtype=dtype)
        )

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.noise_head = EquivariantVector(
            hidden_channels=emb_dim,
            activation="silu",
            dtype=dtype,
            reduce_op=aggregation,
        )
        # Toggleable so downstream code can disable the denoising branch.
        self.denoise = True

        self.noise_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.embedding_head = ProjHead2(emb_dim=emb_dim, agg=emb_agg)

        self.scalar_head = EquivariantScalar(
            hidden_channels=emb_dim,
            activation=activation,
            dtype=dtype,
            reduce_op=head_agg,
        )

    # ------------------------------------------------------------------
    # Convenience setters that the upstream training script uses; kept
    # so reset_head/reset_embeddings/etc. behave the same if invoked.
    # ------------------------------------------------------------------
    def reset_head(self):
        self.scalar_head.reset_parameters()

    def reset_embeddings(self):
        self.rep_model.embedding.reset_parameters()

    def reset_norms(self, norm_type: str = "LN"):
        if isinstance(norm_type, bool):
            if norm_type:
                norm_type = "LN"
            else:
                return

        for attn in self.rep_model.attention_layers:
            attn.conditional_ln.reset_parameters(norm_type=norm_type)

        for norm in self.rep_model.x_post_norm:
            if norm is not None and isinstance(
                norm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)
            ):
                norm.reset_parameters()

        for norm in self.rep_model.v_post_norm:
            if norm is not None and isinstance(
                norm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)
            ):
                norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ) -> dict:
        """Forward pass — see `examples.ipynb` for the input contract.

        Returns dict with keys: `noise_pred`, `mol_emb`, `y` (and optionally
        `atom_embs`, `atom_outputs`, `dy`).
        """
        batch = torch.zeros_like(z) if batch is None else batch
        if self.derivative:
            pos.requires_grad_(True)

        input_args = {"z": z, "pos": pos, "batch": batch}
        if box is not None:
            input_args["box"] = box
        if q is not None:
            input_args["q"] = q
        if s is not None:
            input_args["s"] = s
        if cond is not None:
            input_args["cond"] = cond

        if kwargs.get("graph_batch", None) is not None:
            input_args["graph_batch"] = kwargs.get("graph_batch", None)

        rep_out = self.rep_model(**input_args)
        x, v, z, pos, batch = rep_out

        output_dict: dict = {}

        # ---- Noise prediction
        noise_pred = None
        if self.denoise:
            noise_pred = self.noise_head.pre_reduce(x, v, z, pos, batch)
        output_dict["noise_pred"] = noise_pred

        # ---- Molecule embedding
        if self.add_head_to_pred:
            output_dict["mol_emb"] = self.embedding_head(x.detach(), batch)
        else:
            output_dict["mol_emb"] = self.embedding_head(x, batch)

        if kwargs.get("return_atom_embs", False):
            output_dict["atom_embs"] = x

        # ---- Scalar (y) head
        x = self.scalar_head.pre_reduce(x, v, z, pos, batch)

        if kwargs.get("atom_mask", None) is not None:
            atom_mask = kwargs.get("atom_mask")
            x = x * atom_mask.unsqueeze(-1)

        if kwargs.get("return_atom_outputs", False):
            output_dict["atom_outputs"] = x

        if self.std is not None:
            x = x * self.std

        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        x = self.scalar_head.reduce(x, batch)
        if self.mean is not None:
            x = x + self.mean
        y = self.scalar_head.post_reduce(x)

        if self.add_head_to_pred:
            mol_pred = output_dict["mol_emb"].mean(-1)
            if self.std is not None:
                mol_pred = mol_pred * self.std
            if self.mean is not None:
                mol_pred = mol_pred + self.mean
            y = y + mol_pred

        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch, box, extra_args)

        output_dict["y"] = y
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            assert dy is not None, "Autograd returned None for force prediction."
            output_dict["dy"] = -dy

        return output_dict


class CET(BaseModel):
    """Conditional Equivariant Transformer wrapper (model="CET")."""

    def __init__(
        self,
        emb_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        neighbor_embedding: bool = True,
        max_num_neighbors: int = 32,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_z: int = 118,
        layernorm_on_vec: bool = False,
        check_errors: bool = True,
        box_vecs=None,
        vector_cutoff: bool = True,
        p_droppath: float = 0.1,
        p_dropcond: float = 0.2,
        inv_post_norm: bool = False,
        vec_post_norm: bool = False,
        vec_prenorm: bool = True,
        dtype: torch.dtype = torch.float32,
        derivative: bool = False,
        activation: str = "silu",
        aggregation: str = "sum",
        emb_agg: str = "sum",
        mean=None,
        std=None,
        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            derivative=derivative,
            activation=activation,
            aggregation=aggregation,
            emb_agg=emb_agg,
            dtype=dtype,
            mean=mean,
            std=std,
            **kwargs,
        )

        self.rep_model = ConditionalET(
            hidden_channels=emb_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=activation,
            neighbor_embedding=neighbor_embedding,
            num_heads=num_heads,
            distance_influence=distance_influence,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            check_errors=check_errors,
            box_vecs=box_vecs,
            vector_cutoff=vector_cutoff,
            layernorm_on_vec=layernorm_on_vec,
            dtype=dtype,
            p_droppath=p_droppath,
            p_dropcond=p_dropcond,
            inv_post_norm=inv_post_norm,
            vec_post_norm=vec_post_norm,
            vec_prenorm=vec_prenorm,
        )


__all__ = ["CET", "BaseModel", "AccumulatedNormalization"]
