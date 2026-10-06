"""
Output modules used by the CET model.

Lifted from `SelfConditionedDenoisingAtoms/models/ET_models/output_modules.py`
and trimmed to: MLP, GatedEquivariantBlock, OutputModel, EquivariantScalar,
EquivariantVector. The original file also defined OutputHead and Scalar which
are not used on the CET checkpoint we load.

The upstream had a CUDA-graph-capture detection path that imported
`is_current_stream_capturing` from a compiled C++ kernel. We don't use CUDA
graphs anywhere in AFD, so the detection has been removed and the reduce
op always computes `dim_size = int(batch.max().item() + 1)`.
"""

import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional
from warnings import warn

import torch
from torch import nn

from .utils import act_class_mapping, scatter


class MLP(nn.Module):
    """Linear -> act -> [Linear -> act] x num_hidden_layers -> Linear."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: str,
        num_hidden_layers: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        act_class = act_class_mapping[activation]
        self.act = act_class()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(in_channels, hidden_channels, dtype=dtype))
        self.layers.append(self.act)
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels, dtype=dtype))
            self.layers.append(self.act)
        self.layers.append(nn.Linear(hidden_channels, out_channels, dtype=dtype))

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.layers(x)


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block (Schütt et al. 2021)."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        activation: str = "silu",
        scalar_activation: bool = False,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.vec2_proj = nn.Linear(
            hidden_channels, out_channels, bias=False, dtype=dtype
        )

        act_class = act_class_mapping[activation]
        self.update_net = MLP(
            in_channels=hidden_channels * 2,
            out_channels=out_channels * 2,
            hidden_channels=intermediate_channels,
            activation=activation,
            num_hidden_layers=0,
            dtype=dtype,
        )
        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        self.update_net.reset_parameters()

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        vec1 = torch.zeros(
            vec1_buffer.size(0),
            vec1_buffer.size(2),
            device=vec1_buffer.device,
            dtype=vec1_buffer.dtype,
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        if not mask.all():
            warnings.warn(
                f"Skipping gradients for {(~mask).sum()} atoms due to vector "
                "features being zero. This usually means an atom has no "
                "neighbours inside the cutoff radius."
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)

        vec2 = self.vec2_proj(v)
        x = torch.cat([x, vec1], dim=-1)

        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model: bool, reduce_op: str):
        super().__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.dim_size = 0

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch): ...

    def reduce(self, x, batch):
        # Upstream branched here on whether the current CUDA stream was
        # being captured into a CUDA graph; in that mode the dim_size had
        # to be locked so the captured graph would have a fixed shape. We
        # don't use CUDA graphs in AFD, so always recompute dim_size
        # from the batch index.
        self.dim_size = int(batch.max().item() + 1)
        return scatter(x, batch, dim=0, dim_size=self.dim_size, reduce=self.reduce_op)

    def post_reduce(self, x):
        return x


class EquivariantScalar(OutputModel):
    def __init__(
        self,
        hidden_channels: int,
        activation: str = "silu",
        allow_prior_model: bool = True,
        reduce_op: str = "sum",
        dtype: torch.dtype = torch.float,
        **kwargs,
    ):
        super().__init__(allow_prior_model=allow_prior_model, reduce_op=reduce_op)
        if kwargs.get("num_layers", 0) > 0:
            warn("num_layers is not used in EquivariantScalar")
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    dtype=dtype,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    activation=activation,
                    dtype=dtype,
                ),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # Include v in output so every parameter sees gradient (otherwise
        # GatedEquivariantBlock.vec2_proj would be dead in graphs with no
        # vector path).
        return x + v.sum() * 0


class EquivariantVector(EquivariantScalar):
    def __init__(
        self,
        hidden_channels: int,
        activation: str = "silu",
        reduce_op: str = "sum",
        dtype: torch.dtype = torch.float,
        **kwargs,
    ):
        super().__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op="sum",
            dtype=dtype,
            **kwargs,
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze() + x.sum() * 0


__all__ = [
    "MLP",
    "GatedEquivariantBlock",
    "OutputModel",
    "EquivariantScalar",
    "EquivariantVector",
]
