"""
Data transforms — minimal subset needed to prepare QM9-like samples for the CT
forward pass. The upstream `AddStandardKeys` from
`SelfConditionedDenoisingAtoms/data/datasets/transforms.py` carried optional
hooks into `StructureCloud` and a `remove_edge_index` flag we don't need; this
copy strips both.
"""

import torch
from torch_geometric.transforms import BaseTransform


class AddStandardKeys(BaseTransform):
    """Ensure each Data has `pbc`, `natoms`, and `cell` populated.

    QM9 entries normally only carry `z` and `pos`. The CT graph generator wants
    a bounding-box `cell`, a per-graph `pbc` flag, and an `natoms` count. This
    transform supplies sensible defaults for non-periodic molecules: zero pbc,
    a diagonal bounding-box cell padded by `cell_buffer` Å, and natoms = N.

    Args:
        cell_buffer: padding added to the bounding box in each direction
            so that all atoms strictly fit inside the cell. Default 0.5 Å.
        alt_cell_key: alternative attribute to use as the cell if `cell` is
            absent (e.g. "box" used by some upstream code). Default "box".
        remove_edge_index: if True, strip any pre-existing `edge_index` on the
            Data so the CT regenerates the graph fresh. Default True.
    """

    def __init__(
        self,
        cell_buffer: float = 0.5,
        alt_cell_key: str = "box",
        remove_edge_index: bool = True,
    ):
        self.cell_buffer = cell_buffer
        self.alt_cell_key = alt_cell_key
        self.remove_edge_index = remove_edge_index

    def forward(self, data, depth: int = 0):
        if isinstance(data, tuple) and depth == 0:
            return tuple(self.forward(d, depth=depth + 1) for d in data)

        # Ensure model inputs are tensors on a common device. Generation records
        # commonly originate as NumPy arrays, including their periodic metadata.
        data.pos = torch.as_tensor(data.pos, dtype=torch.float32)
        data.z = torch.as_tensor(data.z, dtype=torch.long, device=data.pos.device)

        # pbc
        pbc = getattr(data, "pbc", None)
        if pbc is None:
            pbc = torch.zeros((1, 3), dtype=torch.bool, device=data.pos.device)
        data.pbc = torch.as_tensor(
            pbc,
            dtype=torch.bool,
            device=data.pos.device,
        ).reshape(-1, 3)

        # natoms
        natoms = getattr(data, "natoms", None)
        if natoms is None:
            natoms = [data.pos.shape[0]]
        data.natoms = torch.as_tensor(
            natoms,
            dtype=torch.long,
            device=data.pos.device,
        ).reshape(-1)

        # cell
        cell = getattr(data, "cell", None)
        if cell is None:
            alt_cell = getattr(data, self.alt_cell_key, None)
            if alt_cell is not None:
                cell = alt_cell
            else:
                min_coords = data.pos.min(dim=0).values
                max_coords = data.pos.max(dim=0).values
                lengths = (max_coords - min_coords) + self.cell_buffer
                cell = torch.diag(lengths).unsqueeze(0)
        data.cell = torch.as_tensor(
            cell,
            dtype=torch.float32,
            device=data.pos.device,
        ).reshape(-1, 3, 3)

        if hasattr(data, "edge_index") and self.remove_edge_index:
            del data.edge_index

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cell_buffer={self.cell_buffer}, "
            f"alt_cell_key={self.alt_cell_key}, "
            f"remove_edge_index={self.remove_edge_index})"
        )


__all__ = ["AddStandardKeys"]
