"""Zevals-backed validation metrics for structure prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from foundry.metrics.metric import Metric
from rdkit import Chem
from zevals.materials.structure_prediction import (
    compute_material_structure_prediction_metrics,
)
from zevals.molecules.structure_prediction import (
    compute_molecule_structure_prediction_metrics,
)

from nucleus.metrics.uma_forces import (
    compute_material_uma_force_metrics,
    compute_molecule_uma_force_metrics,
    electronic_conditioning_from_extra_info,
)
from nucleus.utils.atom_array import _atom_array_to_smiles
from nucleus.utils.forces import mean_force_norm_target_from_extra_info
from nucleus.utils.material_fingerprints import atom_array_to_structure


def _as_atom_array_list(atom_array_stack: Any) -> list[Any]:
    """Normalize an AtomArray, AtomArrayStack, tuple, or list."""
    if atom_array_stack is None:
        return []
    if isinstance(atom_array_stack, (list, tuple)):
        return list(atom_array_stack)

    stack_depth = getattr(atom_array_stack, "stack_depth", None)
    if callable(stack_depth):
        return [atom_array_stack[index] for index in range(int(stack_depth()))]
    if isinstance(stack_depth, int):
        return [atom_array_stack[index] for index in range(stack_depth)]
    return [atom_array_stack]


def _condition_name(extra_info: dict[str, Any] | None) -> str | None:
    """Normalize a collated validation condition name."""
    if not extra_info:
        return None
    value = extra_info.get("sampled_condition_name")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    return None if value is None else str(value)


def _target_id(extra_info: dict[str, Any] | None) -> str:
    """Return one stable target ID for the zevals grouped-set API."""
    value = None if not extra_info else extra_info.get("example_id")
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return str(value or "validation_target")


def _is_material(atom_array: Any) -> bool:
    """Return whether an AtomArray contains material atoms."""
    categories = getattr(atom_array, "get_annotation_categories", lambda: [])()
    return "is_material" in categories and bool(np.any(atom_array.is_material))


def _validate_reference_coordinates(
    coordinates: Any,
    *,
    target_id: str,
) -> None:
    """Reject missing, nonfinite, or collapsed multi-atom references."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError(f"{target_id}: reference coordinates must have shape [N, 3].")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{target_id}: reference coordinates must be finite.")
    if len(coordinates) > 1:
        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        rms_radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if rms_radius <= 1e-6:
            raise ValueError(
                f"{target_id}: reference coordinates are collapsed; refusing to "
                "compute a potentially misleading structure metric."
            )


def _to_rdkit_molecule(atom_array: Any, *, example_id: str) -> Any | None:
    """Apply the same Open Babel/RDKit conversion used for inference outputs."""
    try:
        return _atom_array_to_smiles(
            atom_array,
            example_id=example_id,
            return_mol_only=True,
        )
    except Exception:
        return None


def _as_unbatched_numpy(value: Any) -> np.ndarray:
    """Move a feature tensor to NumPy and remove one collation dimension."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _molecule_from_conditioned_graph(
    atom_array: Any,
    features: dict[str, Any],
) -> Any | None:
    """Build an RDKit conformer from fixed atom types and token connectivity."""
    token_bonds = features.get("token_bonds")
    atom_to_token = features.get("atom_to_token_map")
    if token_bonds is None or atom_to_token is None:
        return None

    token_bonds = _as_unbatched_numpy(token_bonds).astype(bool, copy=False)
    atom_to_token = _as_unbatched_numpy(atom_to_token).astype(np.int64, copy=False)
    elements = np.asarray(getattr(atom_array, "element", []), dtype=object)
    coordinates = np.asarray(getattr(atom_array, "coord", []), dtype=np.float64)
    n_atoms = len(elements)
    if (
        coordinates.shape != (n_atoms, 3)
        or atom_to_token.shape != (n_atoms,)
        or token_bonds.ndim != 2
        or token_bonds.shape[0] != token_bonds.shape[1]
        or np.any(atom_to_token < 0)
        or np.any(atom_to_token >= token_bonds.shape[0])
    ):
        return None

    molecule = Chem.RWMol()
    for element in elements:
        try:
            molecule.AddAtom(Chem.Atom(str(element).title()))
        except Exception:
            return None

    for atom_i in range(n_atoms):
        token_i = atom_to_token[atom_i]
        for atom_j in range(atom_i + 1, n_atoms):
            if token_bonds[token_i, atom_to_token[atom_j]]:
                molecule.AddBond(atom_i, atom_j, Chem.BondType.SINGLE)

    conformer = Chem.Conformer(n_atoms)
    for atom_index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, coordinate.tolist())
    molecule.AddConformer(conformer, assignId=True)
    return molecule.GetMol()


class ZevalsStructurePredictionMetrics(Metric):
    """Adapt trainer validation outputs to the public zevals CSP metrics.

    Molecules use symmetry-aware heavy-atom best RMSD against the single
    preserved ground-truth conformer. Materials use target-wise pymatgen
    StructureMatcher match rate and normalized RMS displacement. UMA residual
    force metrics optionally measure adherence to a supplied mean-force-norm
    conditioning target. Predicted AtomArrays have already passed through the
    trainer's normal output construction, so evaluation sees the same structures
    inference emits.
    """

    def __init__(
        self,
        molecule_top_k: Sequence[int] = (1,),
        structure_ltol: float = 0.3,
        structure_stol: float = 0.5,
        structure_angle_tol: float = 10.0,
        scale: bool = True,
        material_top_k: Sequence[int] = (1,),
        compute_uma_forces: bool = False,
        uma_checkpoint: str = "uma-s-1p2",
        uma_molecule_task: str = "omol",
        uma_material_task: str = "omat",
        uma_device: str | None = None,
        uma_batch_size: int = 8,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        molecule_budgets = tuple(int(value) for value in molecule_top_k)
        if not molecule_budgets or any(value <= 0 for value in molecule_budgets):
            raise ValueError("molecule_top_k must be positive and non-empty.")
        budgets = tuple(int(value) for value in material_top_k)
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("material_top_k must be positive and non-empty.")
        super().__init__(*args, **kwargs)
        self.molecule_top_k = tuple(sorted(set(molecule_budgets)))
        self.structure_ltol = float(structure_ltol)
        self.structure_stol = float(structure_stol)
        self.structure_angle_tol = float(structure_angle_tol)
        self.scale = bool(scale)
        self.material_top_k = tuple(sorted(set(budgets)))
        self.compute_uma_forces = bool(compute_uma_forces)
        self.uma_checkpoint = str(uma_checkpoint)
        self.uma_molecule_task = str(uma_molecule_task)
        self.uma_material_task = str(uma_material_task)
        self.uma_device = uma_device
        self.uma_batch_size = int(uma_batch_size)
        if self.uma_batch_size <= 0:
            raise ValueError("uma_batch_size must be positive.")

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map Foundry validation-loop values to the adapter inputs."""
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "network_input": "network_input",
            "extra_info": "extra_info",
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        """Allow direct use when no validation metadata is available."""
        return frozenset({"network_input", "extra_info"})

    def compute(
        self,
        predicted_atom_array_stack: Any,
        ground_truth_atom_array_stack: Any,
        network_input: dict[str, Any] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute the domain-appropriate public zevals metric."""
        condition_name = _condition_name(extra_info)
        if condition_name is not None and condition_name not in {
            "structure_prediction",
            "sequence_fixed_structure_design",
        }:
            return {}

        predicted_arrays = _as_atom_array_list(predicted_atom_array_stack)
        ground_truth_arrays = _as_atom_array_list(ground_truth_atom_array_stack)
        if not predicted_arrays or not ground_truth_arrays:
            return {}

        target_id = _target_id(extra_info)
        target_mean_force_norm = (
            mean_force_norm_target_from_extra_info(extra_info)
            if self.compute_uma_forces
            else None
        )
        reference_array = ground_truth_arrays[0]
        if _is_material(reference_array):
            reference = atom_array_to_structure(reference_array)
            if reference is None:
                raise ValueError(f"{target_id}: invalid material reference structure.")
            candidates = [
                atom_array_to_structure(atom_array) for atom_array in predicted_arrays
            ]
            metrics = compute_material_structure_prediction_metrics(
                {target_id: candidates},
                {target_id: [reference]},
                ltol=self.structure_ltol,
                stol=self.structure_stol,
                angle_tol=self.structure_angle_tol,
                scale=self.scale,
                top_k=self.material_top_k,
            )
            if self.compute_uma_forces:
                metrics.update(
                    compute_material_uma_force_metrics(
                        candidates,
                        total_samples=len(candidates),
                        checkpoint=self.uma_checkpoint,
                        task=self.uma_material_task,
                        device=self.uma_device,
                        batch_size=self.uma_batch_size,
                        target_mean_force_norm=target_mean_force_norm,
                    )
                )
            return metrics

        features = (network_input or {}).get("f", {})
        _validate_reference_coordinates(
            reference_array.coord,
            target_id=target_id,
        )
        reference = _molecule_from_conditioned_graph(reference_array, features)
        if reference is None:
            reference = _to_rdkit_molecule(
                reference_array,
                example_id=f"{target_id}:reference",
            )
        if reference is None:
            raise ValueError(f"{target_id}: invalid molecular reference conformer.")
        candidates = []
        for index, atom_array in enumerate(predicted_arrays):
            candidate = _molecule_from_conditioned_graph(atom_array, features)
            if candidate is None:
                candidate = _to_rdkit_molecule(
                    atom_array,
                    example_id=f"{target_id}:candidate-{index}",
                )
            candidates.append(candidate)
        metrics = compute_molecule_structure_prediction_metrics(
            {target_id: candidates},
            {target_id: [reference]},
            top_k=self.molecule_top_k,
        )
        if self.compute_uma_forces:
            total_charge, spin_multiplicity = electronic_conditioning_from_extra_info(
                extra_info
            )
            metrics.update(
                compute_molecule_uma_force_metrics(
                    candidates,
                    total_samples=len(candidates),
                    checkpoint=self.uma_checkpoint,
                    task=self.uma_molecule_task,
                    device=self.uma_device,
                    batch_size=self.uma_batch_size,
                    target_mean_force_norm=target_mean_force_norm,
                    total_charge=total_charge,
                    spin_multiplicity=spin_multiplicity,
                )
            )
        return metrics


__all__ = ["ZevalsStructurePredictionMetrics"]
