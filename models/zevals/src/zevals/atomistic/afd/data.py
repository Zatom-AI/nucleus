"""
Dataset adapters: load atomistic structures from disparate sources and
yield `torch_geometric.data.Data` objects compatible with the CT forward pass.

Two layers:
  - `dict_to_data(rec)` / `load_records(path)` — low-level adapters.
  - `resolve_records(spec)` — high-level dispatcher used by the CLI. Accepts
    a `.pt` file, a `qm9[:split]` shortcut, an `mp20[:split]` shortcut (periodic
    crystals), or a folder of `.pt` files. New formats can be added by
    registering a loader.

Note: for the canonical `qm9` / `mp20` references you do NOT need these dataset
loaders — `AFDScore("qm9"|"mp20")` fetches precomputed reference embeddings from
HuggingFace. The `qm9` / `mp20` loaders here are only for building a *custom*
reference from raw structures, and their heavy deps (rdkit via
`torch_geometric.datasets.QM9`; `StructureCloud` for mp20) are imported lazily.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Low-level adapters
# ---------------------------------------------------------------------------
def _np_to_long(arr) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr), dtype=torch.long)


def _np_to_float(arr) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr), dtype=torch.float32)


def dict_to_data(rec: dict) -> Data:
    """Convert a single generation record to a `torch_geometric.data.Data`.

    Records typically look like:
        {"z": ndarray (N,), "pos": ndarray (N, 3)}
    Some files use `"atom_types"` as an alias for `"z"`; we accept either.
    Optional `"cell"` and `"pbc"` are forwarded if present (useful for
    periodic materials generators).
    """
    z_key = "z" if "z" in rec else "atom_types"
    z = _np_to_long(rec[z_key]).view(-1)
    pos = _np_to_float(rec["pos"]).view(-1, 3)
    assert (
        z.shape[0] == pos.shape[0]
    ), f"z and pos have mismatched lengths: {z.shape[0]} vs {pos.shape[0]}"
    data = Data(z=z, pos=pos)
    if "cell" in rec:
        data.cell = _np_to_float(rec["cell"]).view(-1, 3, 3)
    if "pbc" in rec:
        pbc = torch.as_tensor(np.asarray(rec["pbc"]), dtype=torch.bool).view(-1, 3)
        data.pbc = pbc
    return data


def load_records_pt(path: str) -> List[Data]:
    """Load a `.pt` file holding a list of dict-style structure records."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, list):
        raise TypeError(
            f"Expected a list of records in {path}, got {type(raw).__name__}"
        )
    return [dict_to_data(rec) for rec in raw]


# Back-compat alias.
load_generations = load_records_pt


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class GenerationDataset(Dataset):
    """In-memory list of `Data` records with an optional per-item transform."""

    def __init__(self, records: Iterable[Data], transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Data:
        data = self.records[idx]
        return self.transform(data) if self.transform is not None else data


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------
# A `spec` is one of:
#
#   - A path to a `.pt` file holding a list of dicts:    "/data/gen.pt"
#   - A `qm9[:root]` shortcut (calls torch_geometric.datasets.QM9):
#                                                        "qm9", "qm9:./tmp/qm9"
#   - An `mp20[:split]` shortcut (periodic crystals via StructureCloud's
#     MP20_dataset; split is one of train/val/test/all, default test):
#                                                        "mp20", "mp20:test"
#   - A directory of `.pt` files (each loaded and concatenated):
#                                                        "/data/gens_dir/"
#
# To add a new format, register it via `register_loader(scheme, fn)`.

_LOADERS: Dict[str, Callable[[str], List[Data]]] = {}


def register_loader(scheme: str, fn: Callable[[str], List[Data]]) -> None:
    """Register a loader for a custom `scheme:` spec (e.g. `register_loader("mp20", load_mp20)`)."""
    _LOADERS[scheme] = fn


def _load_qm9(root: str = "./tmp/qm9") -> List[Data]:
    from torch_geometric.datasets import QM9

    ds = QM9(root=root)
    return [ds[i] for i in range(len(ds))]


_LOADERS["qm9"] = _load_qm9


def _load_mp20(split: str = "test") -> List[Data]:
    """Load MP20 ground-truth crystal structures via StructureCloud.

    MP20 is a periodic-materials dataset (the materials analogue of QM9). Each
    record is returned as a `Data` carrying:
        z    : long  (N,)     atomic numbers
        pos  : float (N, 3)   absolute Cartesian positions (Å)
        cell : float (1,3,3)  lattice vectors (rows are the cell vectors)
        pbc  : bool  (1, 3)   periodic-boundary flags (all True for MP20)

    Unlike the training-time wrapper in
    `SelfConditionedDenoisingAtoms/data/datasets/mp20.py`, this loader returns
    the *raw* ground-truth structures: no `RandomCellRepeats` augmentation is
    applied, so the reference set reflects the true MP20 distribution.

    Args:
        split: which MP20 split to load — one of "train", "val", "test", or
            "all". Passed straight through to `MP20_dataset`. Defaults to
            "test", matching the spec string `"mp20"`. Use e.g. `"mp20:train"`
            or `"mp20:all"` to select another split.

    Data availability:
        Structures are fetched/cached by StructureCloud under its HuggingFace
        cache (`hf_cache_location('datasets')`, typically
        `~/.cache/huggingface/datasets/mp_20_<split>.pt`). If a preprocessed
        `.pt` is absent it is built on first access from the raw
        `TGDataset("mp_20", split=...)`, which downloads as needed.
    """
    from StructureCloud.Datasets import MP20_dataset

    ds = MP20_dataset(split=split or "test", preprocess=True)
    records: List[Data] = []
    for i in range(len(ds)):
        item = ds[i]
        z = _np_to_long(item.z).view(-1)
        pos = _np_to_float(item.pos).view(-1, 3)
        cell = _np_to_float(item.box).view(-1, 3, 3)
        if hasattr(item, "pbc") and item.pbc is not None:
            pbc = torch.as_tensor(item.pbc, dtype=torch.bool).view(-1, 3)
        else:
            pbc = torch.ones((1, 3), dtype=torch.bool)
        records.append(Data(z=z, pos=pos, cell=cell, pbc=pbc))
    return records


_LOADERS["mp20"] = _load_mp20


def resolve_records(spec: str, limit: Optional[int] = None) -> List[Data]:
    """Resolve a CLI `--ref` / `--gen` spec into a list of `Data`.

    Args:
        spec: see module docstring. Supported scheme shortcuts include
            `qm9[:root]` (non-periodic molecules) and `mp20[:split]` (periodic
            crystals; split in train/val/test/all, default test).
        limit: optional cap on the number of records returned (kept from the
            front of the list). Useful for quick sanity-checks and for
            matching N across reference/generation sets.

    Returns:
        list of `Data` objects, length min(len(source), limit).
    """
    if ":" in spec and not os.path.exists(spec):
        scheme, _, arg = spec.partition(":")
        if scheme in _LOADERS:
            records = _LOADERS[scheme](arg) if arg else _LOADERS[scheme]()
        else:
            raise ValueError(f"Unknown scheme {scheme!r}; registered: {list(_LOADERS)}")
    elif spec in _LOADERS:
        records = _LOADERS[spec]()
    elif os.path.isdir(spec):
        records = []
        for fname in sorted(os.listdir(spec)):
            if fname.endswith(".pt"):
                records.extend(load_records_pt(os.path.join(spec, fname)))
    elif spec.endswith(".pt") and os.path.isfile(spec):
        records = load_records_pt(spec)
    else:
        raise FileNotFoundError(
            f"Could not resolve records spec {spec!r}: not a known scheme, "
            f"directory, or .pt file"
        )

    if limit is not None:
        records = records[:limit]
    return records


__all__ = [
    "dict_to_data",
    "load_records_pt",
    "load_generations",  # back-compat
    "GenerationDataset",
    "register_loader",
    "resolve_records",
]
