"""Convert zevals samples into ASE ``Atoms`` for UMA, and cache the model.

zevals hands a metric function a plain Python list. The item type depends on
the modality:

* molecules  -> ``rdkit.Chem.Mol`` (a 3D conformer is present only when the
  source file was an ``.sdf``)
* materials  -> ``pymatgen.core.Structure``

The two paths set different flags, and getting them wrong silently changes the
physics:

* a molecule is aperiodic (no cell, ``pbc=False``) and carries its formal
  charge, which the charge-aware ``omol`` head reads;
* a material is periodic (lattice matrix as the cell, ``pbc=True``), which the
  ``omat`` head needs — drop the cell and the atoms stand alone in vacuum and
  report large spurious binding forces.

Every converter returns ``None`` instead of raising, so one bad sample cannot
take down a whole validation epoch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# One model per (checkpoint, task, device). A training loop calls a metric once
# per validation epoch; without this cache each call would reload the weights.
_MODEL_CACHE: dict[tuple[str, str, str | None], Any] = {}


def get_uma_model(
    checkpoint: str = "uma-s-1p2",
    task: str = "omol",
    device: str | None = None,
    **kwargs: Any,
) -> Any:
    """Return a cached ``UMAModel``, building it on first use.

    Args:
        checkpoint: UMA checkpoint id, or a path to a local ``.pt`` file.
        task: UMA task head — ``"omol"`` for molecules, ``"omat"`` for crystals.
        device: ``"cuda"``, ``"cpu"``, or ``None``/``"auto"`` to select
            automatically.
        **kwargs: Forwarded to ``UMAModel`` (``charge``, ``spin``,
            ``precision``).

    Returns:
        A shared ``UMAModel``. The same arguments always give the same object.
    """
    from zevals.atomistic.uma.model import UMAModel, normalize_device

    key = (checkpoint, task, normalize_device(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = UMAModel(
            checkpoint=checkpoint, task=task, device=device, **kwargs
        )
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Drop every cached ``UMAModel`` and free its GPU memory."""
    _MODEL_CACHE.clear()


def mol_to_atoms(mol: Any) -> Any | None:
    """RDKit ``Mol`` -> aperiodic ASE ``Atoms``. ``None`` when unusable.

    No cell is set and ``pbc`` stays False, which is what the ``omol`` head
    expects. The formal charge of the molecule becomes ``atoms.info["charge"]``.
    A molecule without a 3D conformer (parsed from bare SMILES) has no geometry
    to score, so it returns ``None``.
    """
    from ase import Atoms

    if mol is None:
        return None
    try:
        if mol.GetNumConformers() == 0 or mol.GetNumAtoms() == 0:
            return None
        conf = mol.GetConformer()
        numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
        positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    except Exception:
        logger.debug("Skipping a molecule UMA could not read", exc_info=True)
        return None
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.info["charge"] = int(charge)
    return atoms


def structure_to_atoms(structure: Any) -> Any | None:
    """pymatgen ``Structure`` -> periodic ASE ``Atoms``. ``None`` when unusable.

    The lattice matrix becomes the cell and ``pbc`` is True, which the periodic
    heads (``omat``, ``oc20``, ``odac``, ``omc``) need.
    """
    from ase import Atoms

    if structure is None:
        return None
    try:
        numbers = [site.specie.Z for site in structure]
        positions = [list(site.coords) for site in structure]
        cell = [list(v) for v in structure.lattice.matrix]
    except Exception:
        logger.debug("Skipping a structure UMA could not read", exc_info=True)
        return None
    if len(numbers) == 0:
        return None
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def sample_to_atoms(sample: Any) -> Any | None:
    """Convert one zevals sample of any supported type. ``None`` on failure.

    The type is detected by duck typing, so neither RDKit nor pymatgen has to
    be installed for the other path to work.
    """
    if sample is None:
        return None
    if hasattr(sample, "GetNumConformers"):  # rdkit.Chem.Mol
        return mol_to_atoms(sample)
    if hasattr(sample, "lattice"):  # pymatgen.core.Structure
        return structure_to_atoms(sample)
    if hasattr(sample, "get_positions"):  # already an ase.Atoms
        return sample
    return None


def samples_to_atoms(samples: Sequence[Any]) -> list[Any]:
    """Convert a zevals sample list to ASE ``Atoms``, dropping failures.

    The result is shorter than the input whenever a sample is ``None``, has no
    3D geometry, or fails to convert. The caller keeps the original count for
    the scored-rate denominator.
    """
    converted = [sample_to_atoms(sample) for sample in samples]
    return [atoms for atoms in converted if atoms is not None]


__all__ = [
    "clear_model_cache",
    "get_uma_model",
    "mol_to_atoms",
    "sample_to_atoms",
    "samples_to_atoms",
    "structure_to_atoms",
]
