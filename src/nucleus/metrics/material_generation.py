"""Material-generation metrics for validation-time logging.

This adapter keeps the metric definitions aligned with ``zevals`` while
accepting the 3D ``AtomArray`` objects produced by the nucleus validation loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from foundry.metrics.metric import Metric

try:
    from pymatgen.core import Structure
    from zevals.materials import (
        compute_diversity as compute_material_diversity,
    )
    from zevals.materials import (
        compute_novelty as compute_material_novelty,
    )
    from zevals.materials import (
        compute_uniqueness as compute_material_uniqueness,
    )
    from zevals.materials import (
        compute_validity as compute_material_validity,
    )
    from zevals.materials.crystal import Crystal, crystal_from_structure
except ImportError:  # pragma: no cover - import-time guard
    raise ImportError(
        "Material-generation metrics require zevals. "
        "Install the `zevals` extra or add `zevals` and its dependencies."
    )

from nucleus.metrics.uma_forces import (
    compute_material_uma_force_metrics as _compute_uma_force_metrics,
)
from nucleus.utils.forces import mean_force_norm_target_from_extra_info
from nucleus.utils.material_fingerprints import (
    atom_array_to_structure as atom_array_to_material_structure,
)
from nucleus.utils.material_fingerprints import (
    compute_structure_fingerprints as compute_material_structure_fingerprints,
)
from nucleus.utils.material_fingerprints import (
    load_reference_fingerprints_from_path,
)

_atom_array_to_structure = atom_array_to_material_structure

logger = logging.getLogger(__name__)


def _as_atom_array_list(atom_array_stack: Any) -> list[Any]:
    """Normalize a single AtomArray, a stack, or a list into a Python list."""
    if atom_array_stack is None:
        return []
    if isinstance(atom_array_stack, list):
        return atom_array_stack
    if isinstance(atom_array_stack, tuple):
        return list(atom_array_stack)

    stack_depth = getattr(atom_array_stack, "stack_depth", None)
    if callable(stack_depth):
        depth = int(stack_depth())
        return [atom_array_stack[i] for i in range(depth)]
    if isinstance(stack_depth, int):
        return [atom_array_stack[i] for i in range(stack_depth)]

    return [atom_array_stack]


def _invalid_crystal() -> Crystal:
    """Create a Crystal instance that is explicitly marked invalid."""
    crystal = object.__new__(Crystal)
    crystal.valid = False
    crystal.comp_valid = False
    crystal.struct_valid = False
    return crystal


def _resolve_reference_structures(
    *,
    reference_structures: Iterable[Structure] | None,
    reference_structures_path: str | Path | None,
    extra_info: dict[str, Any] | None,
    ground_truth_atom_array_stack: Any | None,
    use_ground_truth_as_reference: bool,
) -> list[Structure]:
    """Determine the list of reference structures for novelty computation."""
    if reference_structures is not None:
        return list(reference_structures)

    if reference_structures_path is not None:
        path = Path(reference_structures_path)
        resolved = _load_structures_from_path(path)
        if resolved:
            return resolved

    if extra_info:
        for key in ("novelty_reference_structures", "reference_structures"):
            if key in extra_info and extra_info[key] is not None:
                value = extra_info[key]
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [value]

        for key in ("novelty_reference_structures_path", "reference_structures_path"):
            if key in extra_info and extra_info[key]:
                resolved = _load_structures_from_path(Path(extra_info[key]))
                if resolved:
                    return resolved

    if use_ground_truth_as_reference and ground_truth_atom_array_stack is not None:
        structures: list[Structure] = []
        for atom_array in _as_atom_array_list(ground_truth_atom_array_stack):
            structure = _atom_array_to_structure(atom_array)
            if structure is not None:
                structures.append(structure)
        return structures

    return []


def _load_structures_from_path(path: Path) -> list[Structure]:
    """Load reference structures from a CIF file or a directory of CIF files."""
    if path.is_dir():
        return [
            Structure.from_file(str(file_path))
            for file_path in sorted(path.glob("*.cif"))
        ]
    if path.suffix.lower() == ".cif" and path.exists():
        return [Structure.from_file(str(path))]
    return []


def compute_material_generation_metrics(
    predicted_atom_array_stack: Any,
    ground_truth_atom_array_stack: Any | None = None,
    extra_info: dict[str, Any] | None = None,
    *,
    reference_structures: Iterable[Structure] | None = None,
    reference_structures_path: str | Path | None = None,
    reference_fingerprints: Sequence[np.ndarray] | None = None,
    use_ground_truth_as_reference: bool = True,
    novelty_use_fingerprint: bool = True,
    compute_uma_forces: bool = False,
    uma_checkpoint: str = "uma-s-1p2",
    uma_task: str = "omat",
    uma_device: str | None = None,
    uma_batch_size: int = 8,
) -> dict[str, float]:
    """Compute material-generation metrics for a predicted atom-array stack."""
    predicted_atom_arrays = _as_atom_array_list(predicted_atom_array_stack)
    if not predicted_atom_arrays:
        return {
            "valid_rate": 0.0,
            "comp_valid_rate": 0.0,
            "struct_valid_rate": 0.0,
            "diversity_rate": 0.0,
            "unique_rate": 0.0,
            "novelty_rate": 0.0,
        }

    structures: list[Structure] = []
    crystals: list[Crystal] = []
    for atom_array in predicted_atom_arrays:
        structure = _atom_array_to_structure(atom_array)
        if structure is None:
            crystals.append(_invalid_crystal())
            continue
        structures.append(structure)
        try:
            crystals.append(crystal_from_structure(structure))
        except Exception:
            crystals.append(_invalid_crystal())

    validity = (
        compute_material_validity(crystals)
        if crystals
        else {"valid_rate": 0.0, "comp_valid_rate": 0.0, "struct_valid_rate": 0.0}
    )
    valid_structures = [
        getattr(crystal, "structure", None)
        for crystal in crystals
        if getattr(crystal, "valid", False)
    ]
    valid_structures = [
        structure for structure in valid_structures if structure is not None
    ]
    total_samples = len(predicted_atom_arrays)
    num_valid = len(valid_structures)
    uniqueness = compute_material_uniqueness(
        valid_structures, return_unique_structures=True
    )
    unique_structures = list(uniqueness.get("unique_structures", []))
    num_unique = int(uniqueness["num_unique"])

    result = {
        "valid_rate": num_valid / total_samples if total_samples else 0.0,
        "comp_valid_rate": validity["comp_valid_rate"],
        "struct_valid_rate": validity["struct_valid_rate"],
        "diversity_rate": compute_material_diversity(valid_structures)["diversity_rate"]
        if valid_structures
        else 0.0,
        "unique_rate": num_unique / num_valid if num_valid else 0.0,
        "novelty_rate": 0.0,
    }

    resolved_reference_structures = _resolve_reference_structures(
        reference_structures=reference_structures,
        reference_structures_path=reference_structures_path,
        extra_info=extra_info,
        ground_truth_atom_array_stack=ground_truth_atom_array_stack,
        use_ground_truth_as_reference=use_ground_truth_as_reference,
    )
    if (
        resolved_reference_structures or reference_fingerprints is not None
    ) and unique_structures:
        result["novelty_rate"] = compute_material_novelty(
            unique_structures,
            resolved_reference_structures,
            use_fingerprint=novelty_use_fingerprint,
            reference_fingerprints=reference_fingerprints,
        )["novel_rate"]

    # UMA is opt-in and adds no key at all when off, so an existing run's logged
    # metric set is unchanged unless the config asks for it.
    if compute_uma_forces:
        result.update(
            _compute_uma_force_metrics(
                valid_structures,
                total_samples=total_samples,
                checkpoint=uma_checkpoint,
                task=uma_task,
                device=uma_device,
                batch_size=uma_batch_size,
                target_mean_force_norm=mean_force_norm_target_from_extra_info(
                    extra_info
                ),
            )
        )

    return result


class MaterialGenerationMetrics(Metric):
    """Foundry metric adapter for nucleus material generation validation."""

    def __init__(
        self,
        reference_structures: Iterable[Structure] | None = None,
        reference_structures_path: str | Path | None = None,
        reference_fingerprints_path: str | Path | None = None,
        reference_fingerprints: Sequence[np.ndarray] | None = None,
        use_ground_truth_as_reference: bool = True,
        novelty_use_fingerprint: bool = True,
        compute_uma_forces: bool = False,
        uma_checkpoint: str = "uma-s-1p2",
        uma_task: str = "omat",
        uma_device: str | None = None,
        uma_batch_size: int = 8,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reference_structures = (
            list(reference_structures) if reference_structures is not None else None
        )
        self.reference_structures_path = (
            Path(reference_structures_path)
            if reference_structures_path is not None
            else None
        )
        if self.reference_structures is None and self.reference_structures_path:
            loaded_structures = _load_structures_from_path(
                self.reference_structures_path
            )
            if loaded_structures:
                self.reference_structures = loaded_structures
        self.reference_fingerprints_path = (
            Path(reference_fingerprints_path)
            if reference_fingerprints_path is not None
            else None
        )
        self.reference_fingerprints = (
            list(reference_fingerprints) if reference_fingerprints is not None else None
        )
        if self.reference_fingerprints is None and self.reference_fingerprints_path:
            self.reference_fingerprints = load_reference_fingerprints_from_path(
                self.reference_fingerprints_path
            )
        if (
            self.reference_fingerprints is None
            and self.reference_structures is not None
            and novelty_use_fingerprint
        ):
            self.reference_fingerprints = compute_material_structure_fingerprints(
                self.reference_structures
            )
        self.use_ground_truth_as_reference = use_ground_truth_as_reference
        self.novelty_use_fingerprint = novelty_use_fingerprint
        # UMA needs the `zevals[uma]` extra (fairchem-core) and gated weights,
        # so it stays off unless a config asks for it.
        self.compute_uma_forces = compute_uma_forces
        self.uma_checkpoint = uma_checkpoint
        self.uma_task = uma_task
        self.uma_device = uma_device
        self.uma_batch_size = uma_batch_size

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Mapping from compute() argument names to compute_from_kwargs() keys."""
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "extra_info": "extra_info",
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        """Set of compute() argument names that are optional and may be missing from kwargs."""
        return frozenset({"ground_truth_atom_array_stack", "extra_info"})

    def compute(
        self,
        predicted_atom_array_stack,
        ground_truth_atom_array_stack=None,
        extra_info=None,
    ) -> dict[str, float]:
        """Compute the material generation metrics for the given atom array stack."""
        return compute_material_generation_metrics(
            predicted_atom_array_stack=predicted_atom_array_stack,
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            extra_info=extra_info,
            reference_structures=self.reference_structures,
            reference_structures_path=self.reference_structures_path,
            reference_fingerprints=self.reference_fingerprints,
            use_ground_truth_as_reference=self.use_ground_truth_as_reference,
            novelty_use_fingerprint=self.novelty_use_fingerprint,
            compute_uma_forces=self.compute_uma_forces,
            uma_checkpoint=self.uma_checkpoint,
            uma_task=self.uma_task,
            uma_device=self.uma_device,
            uma_batch_size=self.uma_batch_size,
        )


__all__ = [
    "MaterialGenerationMetrics",
    "compute_material_generation_metrics",
    "compute_material_uniqueness",
    "compute_material_structure_fingerprints",
]
