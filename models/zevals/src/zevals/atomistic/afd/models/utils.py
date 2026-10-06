"""
Core utilities for the CT model: scatter, RBFs, activations, neighbour embedding,
cosine cutoff, equivariant layer norm.

Lifted from `SelfConditionedDenoisingAtoms/models/ET_models/utils.py` and trimmed
to only the symbols actually used on the CET / ConditionalET forward path. The
upstream `OptimizedDistance` (which wrapped the compiled torchmdnet C++ kernel)
is gone; graphs are always built via `models.graph_utils.FastGraphGenerator`.

Original upstream attribution preserved: Copyright Universitat Pompeu Fabra
2020-2023 https://www.compscience.org, MIT License.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter

# ---------------------------------------------------------------------------
# Atomic masses (kept because some upstream code-paths index into this; harmless
# to keep — small footprint, exact-copy of the upstream table.)
# ---------------------------------------------------------------------------
# fmt: off
atomic_masses = np.array([
    1.0, 1.008, 4.002602, 6.94, 9.0121831,
    10.81, 12.011, 14.007, 15.999, 18.998403163,
    20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971,
    79.904, 83.798, 85.4678, 87.62, 88.90584,
    91.224, 92.90637, 95.95, 97.90721, 101.07,
    102.9055, 106.42, 107.8682, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766,
    144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592, 204.38, 207.2, 208.9804, 208.98243,
    209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
    232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083,
    257.09511, 258.09843, 259.101, 262.11, 267.122,
    268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19,
    289.194, 293.204, 293.208, 294.214,
])
# fmt: on


# ---------------------------------------------------------------------------
# Scatter (drop-in replacement for torch_scatter.scatter using scatter_reduce)
# ---------------------------------------------------------------------------
def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """torch_scatter.scatter-compatible signature, backed by torch.scatter_reduce."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    operation_dict = {
        "add": "sum",
        "sum": "sum",
        "mul": "prod",
        "mean": "mean",
        "min": "amin",
        "max": "amax",
    }
    reduce_op = operation_dict[reduce]
    index = _broadcast(index, src, dim)
    size = list(src.size())
    size[dim] = (
        dim_size
        if dim_size is not None
        else (int(index.max()) + 1 if index.numel() > 0 else 0)
    )
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce(dim, index, src, reduce_op)


# ---------------------------------------------------------------------------
# Cutoff & RBFs
# ---------------------------------------------------------------------------
class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff_upper)
        return cutoffs


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf, dtype=self.dtype
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}
act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
    "mish": nn.Mish,
}


# ---------------------------------------------------------------------------
# Neighbour embedding (initial atom-type-aware feature mixing)
# ---------------------------------------------------------------------------
class NeighborEmbedding(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff_lower: float,
        cutoff_upper: float,
        max_z: int = 100,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        # strip self-loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        msg = W * x_neighbors.index_select(0, edge_index[1])
        x_neighbors = torch.zeros(
            z.shape[0], x.shape[1], dtype=x.dtype, device=x.device
        ).index_add(0, edge_index[0], msg)
        return self.combine(torch.cat([x, x_neighbors], dim=1))


# ---------------------------------------------------------------------------
# Equivariant vector layer norm — drops the mean per atom and renormalises
# the (3, d) vector features in a rotation-equivariant way.
# ---------------------------------------------------------------------------
class EquivariantLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            # No bias term to preserve equivariance!
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input: Tensor) -> Tensor:
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input: Tensor) -> Tensor:
        return (1.0 / self.normalized_shape[0]) * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix: Tensor) -> Tensor:
        _, s, v = matrix.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        s = s.clamp(min=1e-8)
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(-2, -1)

    def forward(self, input: Tensor) -> Tensor:
        input = input.to(torch.float64)  # double precision for stable inversion
        input = self.mean_center(input)
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(
            1, 1, self.normalized_shape[0]
        )

    def extra_repr(self) -> str:
        return "{normalized_shape}, elementwise_linear={elementwise_linear}".format(
            **self.__dict__
        )


__all__ = [
    "atomic_masses",
    "scatter",
    "_broadcast",
    "CosineCutoff",
    "GaussianSmearing",
    "ExpNormalSmearing",
    "ShiftedSoftplus",
    "Swish",
    "rbf_class_mapping",
    "act_class_mapping",
    "NeighborEmbedding",
    "EquivariantLayerNorm",
]
