"""Molecule-generation metrics for validation-time logging.

This adapter keeps the metric definitions aligned with ``zevals`` while
accepting the 3D ``AtomArray`` objects produced by the nucleus validation loop.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from foundry.metrics.metric import Metric
from rdkit.Chem.rdForceFieldHelpers import (
    UFFGetMoleculeForceField,
    UFFHasAllMoleculeParams,
)

from nucleus.metrics.uma_forces import (
    compute_molecule_uma_force_metrics as _compute_uma_force_metrics,
)
from nucleus.metrics.uma_forces import (
    electronic_conditioning_from_extra_info as _electronic_conditioning_from_extra_info,
)
from nucleus.utils.atom_array import (
    _rdkit_smiles_from_atom_array,
    _smiles_from_rdkit_mol,
)
from nucleus.utils.forces import (
    mean_force_norm,
    mean_force_norm_target_from_extra_info,
)

try:
    from zevals.molecules.diversity import (
        compute_diversity as compute_smiles_diversity,
    )
    from zevals.molecules.novelty import (
        compute_novelty as compute_smiles_novelty,
    )
    from zevals.molecules.posebusters import (
        POSEBUSTERS_KEYS,
        compute_posebusters_metrics,
    )
    from zevals.molecules.uniqueness import (
        compute_uniqueness as compute_smiles_uniqueness,
    )
    from zevals.molecules.validity import (
        compute_validity as compute_mol_validity,
    )
    from zevals.utils.io import load_smiles
except ImportError:
    raise ImportError(
        "Molecule-generation metrics require zevals. "
        "Install the `zevals` extra or add `zevals` and its dependencies."
    )

logger = logging.getLogger(__name__)

KCAL_PER_MOL_TO_EV = 0.0433641153087705


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


def _annotation_array(
    atom_array: Any, name: str, default: Any, n_atoms: int
) -> np.ndarray:
    """Helper to retrieve an annotation array from an atom array, with fallback to a default value."""
    value = getattr(atom_array, name, None)
    if value is None:
        return np.full(n_atoms, default)

    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(n_atoms, array.item())
    if len(array) != n_atoms:
        array = np.resize(array, n_atoms)
    return array


def _atom_array_to_pdb_block(atom_array: Any) -> str:
    """Serialize an atom array into a minimal PDB block for RDKit parsing."""
    coords = np.asarray(getattr(atom_array, "coord"))
    if coords.ndim == 1:
        coords = coords[None, :]
    n_atoms = len(coords)

    atom_names = _annotation_array(atom_array, "atom_name", "C", n_atoms)
    res_names = _annotation_array(atom_array, "res_name", "LIG", n_atoms)
    chain_ids = _annotation_array(atom_array, "chain_id", "A", n_atoms)
    res_ids = _annotation_array(atom_array, "res_id", 1, n_atoms)
    elements = _annotation_array(atom_array, "element", "C", n_atoms)

    lines = []
    for idx, (coord, atom_name, res_name, chain_id, res_id, element) in enumerate(
        zip(coords, atom_names, res_names, chain_ids, res_ids, elements, strict=True),
        start=1,
    ):
        x, y, z = (float(coord[0]), float(coord[1]), float(coord[2]))
        atom_name = str(atom_name).strip()[:4]
        res_name = str(res_name).strip()[:3]
        chain_id = str(chain_id).strip()[:1] or "A"
        element = str(element).strip()[:2]
        lines.append(
            (
                f"HETATM{idx:5d} {atom_name:>4s} {res_name:>3s} {chain_id:1s}"
                f"{int(res_id):4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}  "
            )
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _mol_has_atoms_and_conformer(mol: Any) -> bool:
    """Return whether an RDKit molecule carries atoms and 3D coordinates."""
    return mol is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0


def _atom_array_to_mol(atom_array: Any) -> Any | None:
    """Best-effort conversion from a nucleus AtomArray to an RDKit Mol."""
    try:
        mol = _rdkit_smiles_from_atom_array(
            atom_array,
            example_id="generated_molecule",
            return_mol_only=True,
        )
    except Exception:
        mol = None

    if not _mol_has_atoms_and_conformer(mol):
        return None

    return mol


def _mol_uff_mean_force_norm(
    mol: Any,
    *,
    row_id: Any = "generated_molecule",
) -> float | None:
    """Compute mean per-atom UFF force norm for a generated molecule conformer."""
    try:
        if not _mol_has_atoms_and_conformer(mol):
            return None
        if not UFFHasAllMoleculeParams(mol):
            return None
        conf_id = mol.GetConformer().GetId()
        force_field = UFFGetMoleculeForceField(mol, confId=conf_id)
        gradient = np.asarray(
            force_field.CalcGrad(),
            dtype=np.float64,
        ).reshape(-1, 3)
    except Exception:
        return None

    return mean_force_norm(-gradient, row_id=row_id)


def _compute_uff_median_force_norm_metrics(
    mol_list: list[Any],
    *,
    total_samples: int,
    target_mean_force_norm: float | None = None,
) -> dict[str, float]:
    """Compute aggregate UFF force-norm metrics for valid generated molecules."""
    if target_mean_force_norm is not None:
        target_mean_force_norm = float(target_mean_force_norm)
        if not math.isfinite(target_mean_force_norm) or target_mean_force_norm < 0:
            raise ValueError(
                "target_mean_force_norm must be a finite non-negative scalar, got "
                f"{target_mean_force_norm!r}"
            )

    values = [
        value
        for idx, mol in enumerate(mol_list)
        if (
            value := _mol_uff_mean_force_norm(
                mol,
                row_id=f"generated_molecule_{idx}",
            )
        )
        is not None
    ]
    metrics = {
        "uff_median_force_norm": float(np.median(values)) if values else 0.0,
        "uff_median_force_norm_success_rate": (
            len(values) / total_samples if total_samples else 0.0
        ),
    }
    if target_mean_force_norm is not None:
        values_eV_per_angstrom = np.asarray(values) * KCAL_PER_MOL_TO_EV
        metrics["uff_mean_force_norm_mae_eV_per_angstrom"] = (
            float(np.mean(np.abs(values_eV_per_angstrom - target_mean_force_norm)))
            if values
            else 0.0
        )
    return metrics


def _rdkit_smiles_for_valid_atom_array(
    atom_array: Any, *, example_id: str
) -> str | None:
    """Return SMILES only when RDKit can derive a molecule from the atom array."""
    try:
        mol = _rdkit_smiles_from_atom_array(
            atom_array,
            example_id=example_id,
            return_mol_only=True,
        )
    except Exception:
        return None

    if not _mol_has_atoms_and_conformer(mol):
        return None

    return _smiles_from_rdkit_mol(mol)


def _resolve_reference_smiles(
    *,
    reference_smiles: Iterable[str] | None,
    reference_smiles_path: str | Path | None,
    extra_info: dict[str, Any] | None,
    ground_truth_atom_array_stack: Any | None,
    use_ground_truth_as_reference: bool,
) -> list[str]:
    """Determine the list of reference SMILES strings to use for novelty computation, based on the provided arguments and extra_info."""
    if reference_smiles is not None:
        return list(reference_smiles)

    if reference_smiles_path is not None:
        return load_smiles(Path(reference_smiles_path))

    if extra_info:
        for key in ("novelty_reference_smiles", "reference_smiles"):
            if key in extra_info and extra_info[key] is not None:
                value = extra_info[key]
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [str(value)]

        for key in ("novelty_reference_smiles_path", "reference_smiles_path"):
            if key in extra_info and extra_info[key]:
                return load_smiles(Path(extra_info[key]))

    if use_ground_truth_as_reference and ground_truth_atom_array_stack is not None:
        example_id = "ground_truth_reference"
        return [
            smiles
            for smiles in (
                _rdkit_smiles_for_valid_atom_array(atom_array, example_id=example_id)
                for atom_array in _as_atom_array_list(ground_truth_atom_array_stack)
            )
            if smiles is not None
        ]

    return []


def _resolve_reference_fingerprint_path(
    *,
    reference_fingerprint_path: str | Path | None,
    extra_info: dict[str, Any] | None,
) -> Path | None:
    """Determine the FPSim2 reference database path to use for novelty computation."""
    if reference_fingerprint_path is not None:
        return Path(reference_fingerprint_path)

    if extra_info:
        for key in (
            "novelty_reference_fingerprint_path",
            "reference_fingerprint_path",
            "reference_fp",
        ):
            if key in extra_info and extra_info[key]:
                return Path(extra_info[key])

    return None


def compute_molecule_generation_metrics(
    predicted_atom_array_stack: Any,
    ground_truth_atom_array_stack: Any | None = None,
    extra_info: dict[str, Any] | None = None,
    *,
    reference_smiles: Iterable[str] | None = None,
    reference_smiles_path: str | Path | None = None,
    reference_fingerprint_path: str | Path | None = None,
    novelty_n_threads: int = 1,
    use_ground_truth_as_reference: bool = True,
    compute_posebusters: bool = True,
    compute_uff_force_norm: bool = True,
    compute_uma_forces: bool = False,
    uma_checkpoint: str = "uma-s-1p2",
    uma_task: str = "omol",
    uma_device: str | None = None,
    uma_batch_size: int = 8,
) -> dict[str, float]:
    """Compute the molecule-generation metrics for a predicted atom-array stack."""
    predicted_atom_arrays = _as_atom_array_list(predicted_atom_array_stack)
    if not predicted_atom_arrays:
        result = {
            "valid_rate": 0.0,
            "unique_rate": 0.0,
            "diversity_rate": 0.0,
            "novel_rate": 0.0,
        }
        result.update({key: 0.0 for key in POSEBUSTERS_KEYS})
        result.update(
            {
                "uff_median_force_norm": 0.0,
                "uff_median_force_norm_success_rate": 0.0,
            }
        )
        return result

    mol_list = [_atom_array_to_mol(atom_array) for atom_array in predicted_atom_arrays]
    valid_results = compute_mol_validity(mol_list, frag_check=True)

    num_valid = valid_results["num_valid"]
    valid_mols = valid_results["valid_mols"]
    valid_smiles = valid_results["valid_smiles"]

    total_samples = len(predicted_atom_arrays)
    target_mean_force_norm = (
        mean_force_norm_target_from_extra_info(extra_info)
        if compute_uff_force_norm or compute_uma_forces
        else None
    )

    valid_rate = num_valid / total_samples if total_samples else 0.0
    uniqueness_results = compute_smiles_uniqueness(valid_smiles)
    num_unique = uniqueness_results["num_unique"]
    unique_rate = num_unique / num_valid if num_valid else 0.0
    diversity_rate = compute_smiles_diversity(valid_smiles)["diversity_rate"]

    result = {
        "valid_rate": valid_rate,
        "unique_rate": unique_rate,
        "diversity_rate": diversity_rate,
    }

    resolved_reference_fingerprint_path = _resolve_reference_fingerprint_path(
        reference_fingerprint_path=reference_fingerprint_path,
        extra_info=extra_info,
    )
    if resolved_reference_fingerprint_path is not None:
        result["novel_rate"] = compute_smiles_novelty(
            valid_smiles,
            reference_fp=resolved_reference_fingerprint_path,
            n_threads=novelty_n_threads,
        )["novel_rate"]
    else:
        resolved_reference_smiles = _resolve_reference_smiles(
            reference_smiles=reference_smiles,
            reference_smiles_path=reference_smiles_path,
            extra_info=extra_info,
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            use_ground_truth_as_reference=use_ground_truth_as_reference,
        )
        if resolved_reference_smiles:
            novelty_results = compute_smiles_novelty(
                valid_smiles, resolved_reference_smiles
            )
            result["novel_rate"] = (
                novelty_results["num_novel"] / num_unique if num_unique else 0.0
            )
        else:
            result.update({"novel_rate": 0.0})

    if compute_posebusters:
        posebusters_metrics = (
            compute_posebusters_metrics(valid_mols)
            if valid_mols
            else {key: 0.0 for key in POSEBUSTERS_KEYS}
        )
        for key in POSEBUSTERS_KEYS:
            result[key] = float(posebusters_metrics.get(key, 0.0))
    else:
        result.update({key: 0.0 for key in POSEBUSTERS_KEYS})

    if compute_uff_force_norm:
        result.update(
            _compute_uff_median_force_norm_metrics(
                valid_mols,
                total_samples=total_samples,
                target_mean_force_norm=target_mean_force_norm,
            )
        )
    else:
        result.update(
            {
                "uff_median_force_norm": 0.0,
                "uff_median_force_norm_success_rate": 0.0,
            }
        )

    # UMA is opt-in and adds no key at all when off, so an existing run's logged
    # metric set is unchanged unless the config asks for it.
    if compute_uma_forces:
        total_charge, spin_multiplicity = _electronic_conditioning_from_extra_info(
            extra_info
        )
        result.update(
            _compute_uma_force_metrics(
                valid_mols,
                total_samples=total_samples,
                checkpoint=uma_checkpoint,
                task=uma_task,
                device=uma_device,
                batch_size=uma_batch_size,
                target_mean_force_norm=target_mean_force_norm,
                total_charge=total_charge,
                spin_multiplicity=spin_multiplicity,
            )
        )

    return result


class MoleculeGenerationMetrics(Metric):
    """Foundry metric adapter for nucleus molecule generation validation."""

    def __init__(
        self,
        reference_smiles_path: str | Path | None = None,
        reference_smiles: Iterable[str] | None = None,
        reference_fingerprint_path: str | Path | None = None,
        novelty_n_threads: int = 1,
        use_ground_truth_as_reference: bool = True,
        compute_posebusters: bool = True,
        compute_uff_force_norm: bool = True,
        compute_uma_forces: bool = False,
        uma_checkpoint: str = "uma-s-1p2",
        uma_task: str = "omol",
        uma_device: str | None = None,
        uma_batch_size: int = 8,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        base_metric_kwargs = {}
        for key in ("required_tags_all", "required_tags_any", "prohibited_tags"):
            if key in kwargs:
                base_metric_kwargs[key] = kwargs.pop(key)
        if kwargs:
            unexpected_keys = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError(
                "Unexpected MoleculeGenerationMetrics kwargs: "
                f"{unexpected_keys}. AFD is computed only by "
                "LogMoleculeAFDValidationMetricsCallback."
            )

        super().__init__(*args, **base_metric_kwargs)
        self.reference_smiles_path = (
            Path(reference_smiles_path) if reference_smiles_path is not None else None
        )
        self.reference_smiles = (
            list(reference_smiles) if reference_smiles is not None else None
        )
        if self.reference_smiles is None and self.reference_smiles_path is not None:
            self.reference_smiles = load_smiles(self.reference_smiles_path)
        self.reference_fingerprint_path = (
            Path(reference_fingerprint_path)
            if reference_fingerprint_path is not None
            else None
        )
        self.novelty_n_threads = novelty_n_threads
        self.use_ground_truth_as_reference = use_ground_truth_as_reference
        self.compute_posebusters = compute_posebusters
        self.compute_uff_force_norm = compute_uff_force_norm
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
        """Compute the molecule generation metrics for the given predicted and ground truth atom array stacks, along with any extra info."""
        return compute_molecule_generation_metrics(
            predicted_atom_array_stack=predicted_atom_array_stack,
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            extra_info=extra_info,
            reference_smiles=self.reference_smiles,
            reference_smiles_path=self.reference_smiles_path,
            reference_fingerprint_path=self.reference_fingerprint_path,
            novelty_n_threads=self.novelty_n_threads,
            use_ground_truth_as_reference=self.use_ground_truth_as_reference,
            compute_posebusters=self.compute_posebusters,
            compute_uff_force_norm=self.compute_uff_force_norm,
            compute_uma_forces=self.compute_uma_forces,
            uma_checkpoint=self.uma_checkpoint,
            uma_task=self.uma_task,
            uma_device=self.uma_device,
            uma_batch_size=self.uma_batch_size,
        )


__all__ = [
    "MoleculeGenerationMetrics",
    "POSEBUSTERS_KEYS",
    "compute_molecule_generation_metrics",
]
