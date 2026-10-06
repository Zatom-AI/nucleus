"""Thin wrapper around Meta FAIR's UMA machine-learned interatomic potential.

One class, ``UMAModel``, does three things:

1. It loads a UMA checkpoint (from Hugging Face, or from a local ``.pt`` file).
2. It converts your structures into the container that UMA expects.
3. It runs the model and gives back energy, forces and stress in a plain dict.

UMA is *task conditioned*: one checkpoint holds several task heads
(``omol``, ``omat``, ``oc20``, ``odac``, ``omc``). Pick the head with the
``task=`` argument, and pick the head that agrees with your chemistry.
``zevals.molecules.uma_forces`` and ``zevals.materials.uma_forces`` set that
flag for you (``omol`` / ``omat``).

UMA is *conservative*: forces are the negative gradient of the energy.
Therefore the forward pass builds an autograd graph. Do not wrap a call in
``torch.no_grad()`` or ``torch.inference_mode()`` — the forces become zero or
the call fails.

``fairchem-core`` is an optional dependency; import it only through
``zevals.atomistic.uma``, which raises an informative error when it is absent.
Adapted from the ``min_uma`` reference implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

# The task heads that a public UMA checkpoint contains.
UMA_TASKS: tuple[str, ...] = ("omol", "omat", "oc20", "odac", "omc")

# The task heads that expect a periodic cell. `omol` is the only aperiodic one.
UMA_PERIODIC_TASKS: tuple[str, ...] = ("omat", "oc20", "odac", "omc")

# UMA's own defaults, copied from fairchem's validator. OMol is charge and spin
# aware, so it wants a spin multiplicity of 1 for a closed-shell molecule. The
# other heads ignore spin and want 0.
_DEFAULT_SPIN = {"omol": 1}


def load_fairchem() -> Any:
    """Import ``fairchem.core``, or raise an informative error.

    Mirrors ``zevals.utils.packages`` — UMA metrics are behind the optional
    ``uma`` extra because ``fairchem-core`` pins ``torch~=2.8`` and
    ``numpy<2.5``, which conflicts with a newer training environment.
    """
    try:
        import fairchem.core  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "UMA force metrics require fairchem-core. "
            "Install the `zevals` uma extra (pip install 'zevals[uma]'), and "
            "note it pins torch~=2.8 and numpy<2.5 — use a separate "
            "environment if your trainer needs a newer torch."
        ) from exc
    return fairchem.core


def normalize_device(device: str | None) -> str | None:
    """Reduce a device string to what fairchem accepts (``cuda`` or ``cpu``).

    fairchem rejects an indexed device such as ``"cuda:1"``. ``zevals.utils.
    devices.resolve_device`` produces exactly that, so an indexed string is
    accepted here and reduced to ``"cuda"``, with a warning: set
    ``CUDA_VISIBLE_DEVICES`` to choose a specific GPU instead.
    """
    if device is None or device == "auto":
        return None
    if device.startswith("cuda:"):
        logger.warning(
            "UMA (fairchem) cannot target %s; using 'cuda'. "
            "Set CUDA_VISIBLE_DEVICES to select a specific GPU.",
            device,
        )
        return "cuda"
    return device


def _as_int_list(value: Any, n: int, default: int) -> list[int]:
    """Broadcast a scalar, a tensor or a sequence to a list of ``n`` ints."""
    if value is None:
        return [default] * n
    if torch.is_tensor(value):
        value = value.reshape(-1).tolist()
    if isinstance(value, (int, float)):
        return [int(value)] * n
    out = [int(v) for v in value]
    if len(out) != n:
        raise ValueError(f"expected {n} values, got {len(out)}")
    return out


def pyg_batch_to_atoms(batch: Any) -> list[Any]:
    """Split a ``torch_geometric`` ``Batch`` (or one ``Data``) into ASE atoms.

    Recognised attributes: ``pos``, ``z`` or ``atomic_numbers``, ``batch``,
    ``cell``, ``pbc``, ``charge``, ``spin``. Charge and spin, when present, are
    written into ``atoms.info`` and therefore override the model defaults.
    """
    from ase import Atoms

    pos = batch.pos.detach().cpu().double().numpy()
    z = getattr(batch, "z", None)
    if z is None:
        z = getattr(batch, "atomic_numbers", None)
    if z is None:
        raise AttributeError("batch has neither `z` nor `atomic_numbers`")
    z = z.detach().cpu().long().numpy()

    node_graph = getattr(batch, "batch", None)
    if node_graph is None:
        node_graph = torch.zeros(len(z), dtype=torch.long)
    node_graph = node_graph.detach().cpu().long().numpy()
    n_graphs = int(node_graph.max()) + 1 if len(node_graph) else 0

    cell = getattr(batch, "cell", None)
    if cell is not None:
        cell = cell.detach().cpu().double().reshape(-1, 3, 3).numpy()
    pbc = getattr(batch, "pbc", None)
    if pbc is not None:
        pbc = pbc.detach().cpu().bool().reshape(-1, 3).numpy()

    charges = _as_int_list(getattr(batch, "charge", None), n_graphs, 0)
    spins = _as_int_list(getattr(batch, "spin", None), n_graphs, -1)

    atoms_list = []
    for g in range(n_graphs):
        mask = node_graph == g
        atoms = Atoms(numbers=z[mask], positions=pos[mask])
        if cell is not None:
            atoms.set_cell(cell[g])
            atoms.set_pbc(pbc[g] if pbc is not None else True)
        atoms.info["charge"] = charges[g]
        if spins[g] >= 0:
            atoms.info["spin"] = spins[g]
        atoms_list.append(atoms)
    return atoms_list


class UMAModel:
    """Load a UMA checkpoint and compute energies and forces.

    Args:
        checkpoint: A Hugging Face model id such as ``"uma-s-1p2"``,
            ``"uma-s-1p1"`` or ``"uma-m-1p1"``. A path to a local ``.pt``
            inference checkpoint also works.
        task: The UMA task head, one of ``UMA_TASKS``. Use ``"omol"`` for
            molecules, ``"omat"`` for inorganic crystals, ``"oc20"`` for
            catalyst slabs, ``"odac"`` for MOFs, ``"omc"`` for molecular
            crystals.
        device: ``"cuda"`` or ``"cpu"``. ``None`` or ``"auto"`` selects CUDA
            when available. An indexed device (``"cuda:1"``) is reduced to
            ``"cuda"`` — set ``CUDA_VISIBLE_DEVICES`` to pick a specific GPU.
        charge: Default total charge per system, used when the input carries
            none.
        spin: Default spin multiplicity per system, used when the input
            carries none. ``None`` applies UMA's own default (1 for ``omol``,
            else 0).
        precision: ``"float32"`` (default), ``"float64"`` or ``"bfloat16"``.

    UMA weights are gated on Hugging Face. Request access at
    https://huggingface.co/facebook/UMA, then run ``hf auth login`` or export
    ``HF_TOKEN``. Without this the download returns HTTP 401.
    """

    def __init__(
        self,
        checkpoint: str = "uma-s-1p2",
        task: str = "omol",
        device: str | None = None,
        charge: int = 0,
        spin: int | None = None,
        precision: str = "float32",
    ) -> None:
        load_fairchem()
        from fairchem.core.calculate.pretrained_mlip import (  # type: ignore[import]
            get_predict_unit,
            load_predict_unit,
        )
        from fairchem.core.units.mlip_unit import (  # type: ignore[import]
            InferenceSettings,
        )

        if task not in UMA_TASKS:
            raise ValueError(f"task must be one of {UMA_TASKS}, got {task!r}")
        device = normalize_device(device)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")

        self.task = task
        self.device = device
        self.default_charge = int(charge)
        self.default_spin = _DEFAULT_SPIN.get(task, 0) if spin is None else int(spin)

        settings = InferenceSettings(base_precision_dtype=precision)
        if checkpoint.endswith(".pt"):
            self.predict_unit = load_predict_unit(
                checkpoint, inference_settings=settings, device=device
            )
        else:
            self.predict_unit = get_predict_unit(
                checkpoint, inference_settings=settings, device=device
            )
        self.checkpoint = checkpoint

    # ---------------------------------------------------------------- inputs

    def atoms_to_atomic_data(self, atoms_list: Sequence[Any]) -> Any:
        """Pack a list of ASE ``Atoms`` into one batched ``AtomicData``.

        Each atoms object may set ``info["charge"]`` and ``info["spin"]``; the
        model defaults fill the gaps. Graph construction happens on the device
        inside the model, so no neighbour list is built here.
        """
        from fairchem.core.datasets.atomic_data import (  # type: ignore[import]
            AtomicData,
            atomicdata_list_to_batch,
        )

        if len(atoms_list) == 0:
            raise ValueError("atoms_list is empty")

        records = []
        for atoms in atoms_list:
            atoms = atoms.copy()
            atoms.info.setdefault("charge", self.default_charge)
            atoms.info.setdefault("spin", self.default_spin)
            records.append(
                AtomicData.from_ase(
                    atoms,
                    r_edges=False,
                    task_name=self.task,
                    r_data_keys=["charge", "spin"],
                )
            )
        return atomicdata_list_to_batch(records)

    def to_atomic_data(self, batch: Any) -> Any:
        """Accept a PyG ``Batch``/``Data``, a list of ASE ``Atoms``, or one
        ``Atoms``, and give back a batched ``AtomicData``."""
        if hasattr(batch, "pos") and hasattr(batch, "get_positions") is False:
            return self.atoms_to_atomic_data(pyg_batch_to_atoms(batch))
        if hasattr(batch, "get_positions"):  # a single ase.Atoms
            return self.atoms_to_atomic_data([batch])
        return self.atoms_to_atomic_data(list(batch))

    # --------------------------------------------------------------- predict

    def predict(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run UMA on one batch and give back a dict of tensors.

        Args:
            batch: A ``torch_geometric`` ``Batch`` or ``Data``, a list of ASE
                ``Atoms``, or a single ``Atoms``.

        Returns:
            ``energy``: ``[n_systems]``, in eV.
            ``forces``: ``[n_atoms, 3]``, in eV/A.
            ``stress``: ``[n_systems, 9]``, present for periodic task heads.
            ``batch``: ``[n_atoms]``, the system index of every atom.
            ``natoms``: ``[n_systems]``, the atom count of every system.
        """
        data = self.to_atomic_data(batch)
        out = self.predict_unit.predict(data)

        result = {k: v.detach() for k, v in out.items()}
        result["batch"] = data.batch.detach().to(result["forces"].device)
        result["natoms"] = data.natoms.detach().to(result["forces"].device)
        return result

    def forces(self, batch: Any) -> torch.Tensor:
        """Shortcut that returns the ``[n_atoms, 3]`` force tensor only."""
        return self.predict(batch)["forces"]

    @staticmethod
    def split_per_system(
        result: dict[str, torch.Tensor], key: str = "forces"
    ) -> list[torch.Tensor]:
        """Split a per-atom tensor of ``predict`` into one tensor per system."""
        return list(torch.split(result[key], result["natoms"].tolist()))


__all__ = [
    "UMAModel",
    "UMA_TASKS",
    "UMA_PERIODIC_TASKS",
    "load_fairchem",
    "normalize_device",
    "pyg_batch_to_atoms",
]
