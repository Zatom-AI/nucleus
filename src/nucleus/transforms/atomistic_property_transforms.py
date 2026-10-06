"""Atomistic energy, force, and electronic-state training targets.

The transforms in this module deliberately split target preparation into two
stages.  :class:`AddAtomisticPropertyTargets` runs before cropping and records
the source-system labels on the atom array, so ordinary AtomWorks slicing keeps
forces aligned with atoms.  :class:`FinalizeAtomisticPropertyTargets` runs after
all atom-count-changing transforms and creates the fixed tensor contract used by
the model and sparse collator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts

from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
    UNKNOWN_PROPERTY_TASK_ID,
)
from nucleus.utils.materials import get_material_mask

OMOL25_DOMAIN_ID = 0
OMAT24_DOMAIN_ID = 1
UNKNOWN_DOMAIN_ID = -1

_PROPERTY_TASK_IDS = {
    "unconditional": UNCONDITIONAL_PROPERTY_TASK_ID,
    "unconditional_design": UNCONDITIONAL_PROPERTY_TASK_ID,
    "structure_prediction": STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    "sequence_fixed_structure_design": STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    "mlip": MLIP_PROPERTY_TASK_ID,
}

_PROPERTY_TASK_NAMES = {
    UNCONDITIONAL_PROPERTY_TASK_ID: "unconditional",
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID: "structure_prediction",
    MLIP_PROPERTY_TASK_ID: "mlip",
}

_FORCE_ANNOTATION = "nucleus_property_force"
_FORCE_VALID_ANNOTATION = "nucleus_property_force_valid"
_SOURCE_INDEX_ANNOTATION = "nucleus_property_source_index"


def _finite_scalar(value: Any) -> float | None:
    """Return a finite scalar float, or ``None`` for malformed metadata."""
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.size != 1:
        return None
    try:
        scalar = float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None
    return scalar if np.isfinite(scalar) else None


def _integer_metadata(value: Any, *, minimum: int | None = None) -> tuple[float, bool]:
    """Convert integer-valued electronic metadata without silently rounding it."""
    scalar = _finite_scalar(value)
    if scalar is None or not np.isclose(scalar, round(scalar), rtol=0.0, atol=1e-6):
        return 0.0, False
    integer = int(round(scalar))
    if minimum is not None and integer < minimum:
        return 0.0, False
    return float(integer), True


def _calculator_results(data: Mapping[str, Any], atoms: Any) -> dict[str, Any]:
    """Collect stored calculator results without triggering a calculation."""
    results: dict[str, Any] = {}
    calculator = getattr(atoms, "calc", None)
    calculator_results = getattr(calculator, "results", None)
    if isinstance(calculator_results, Mapping):
        results.update(calculator_results)
    record_results = data.get("calculator_results")
    if isinstance(record_results, Mapping):
        # The dataset record is the serialized source of truth when available.
        results.update(record_results)
    if data.get("is_inference", False):
        energy = _specification_value(data, "energy_gt_eV")
        forces = _specification_value(data, "forces_gt_eV_per_angstrom")
        if "energy" not in results and energy is not None:
            results["energy"] = energy
        if "forces" not in results and forces is not None:
            results["forces"] = forces
    return results


def _atoms_info_value(atoms_info: Mapping[str, Any], key: str) -> Any:
    """Read metadata from flat or ASE-LMDB nested ``atoms.info`` records.

    AtomWorks preserves some ASE database metadata under ``atoms.info["data"]``.
    Explicit top-level values take precedence for compatibility with ordinary
    ASE ``Atoms`` objects and callers that normalize the metadata themselves.
    """
    if key in atoms_info:
        return atoms_info[key]
    nested_info = atoms_info.get("data")
    if isinstance(nested_info, Mapping):
        return nested_info.get(key)
    return None


def _specification_value(data: Mapping[str, Any], key: str) -> Any:
    """Read an inference-only atomistic value from a design specification."""
    if not data.get("is_inference", False):
        return None
    specification = data.get("specification")
    if not isinstance(specification, Mapping):
        return None
    if key in specification:
        return specification[key]
    extra = specification.get("extra")
    return extra.get(key) if isinstance(extra, Mapping) else None


def _explicit_domain_id(data: Mapping[str, Any]) -> int | None:
    """Parse an optional caller-supplied atomistic domain identifier."""
    value = data.get("atomistic_domain", data.get("domain"))
    if value is None:
        value = _specification_value(data, "atomistic_domain")
    if isinstance(value, str):
        normalized = value.lower().replace("_", "").replace("-", "")
        if normalized in {"omol", "omol25", "molecule", "molecular"}:
            return OMOL25_DOMAIN_ID
        if normalized in {"omat", "omat24", "material", "materials"}:
            return OMAT24_DOMAIN_ID
        return UNKNOWN_DOMAIN_ID
    scalar = _finite_scalar(value)
    if scalar is None:
        return None
    domain_id = int(scalar)
    return (
        domain_id
        if domain_id in {OMOL25_DOMAIN_ID, OMAT24_DOMAIN_ID}
        else UNKNOWN_DOMAIN_ID
    )


def _infer_domain_id(data: Mapping[str, Any], atoms: Any) -> int:
    """Infer OMol25 versus OMat24 from explicit metadata or periodicity."""
    explicit = _explicit_domain_id(data)
    if explicit is not None:
        return explicit

    atom_array = data.get("atom_array")
    if atom_array is not None and np.any(get_material_mask(atom_array)):
        return OMAT24_DOMAIN_ID
    if atoms is None:
        return UNKNOWN_DOMAIN_ID
    try:
        if bool(np.any(np.asarray(atoms.pbc, dtype=bool))):
            return OMAT24_DOMAIN_ID
    except (AttributeError, TypeError, ValueError):
        return UNKNOWN_DOMAIN_ID
    return OMOL25_DOMAIN_ID


def _validate_explicit_domain(domain: str | None) -> None:
    """Validate an optional atomistic dataset domain name."""
    if domain is None:
        return
    explicit_domain_id = _explicit_domain_id({"atomistic_domain": domain})
    if explicit_domain_id == UNKNOWN_DOMAIN_ID:
        raise ValueError(
            "domain must be one of {'omol25', 'omat24'} when provided; "
            f"got {domain!r}."
        )


def _electronic_state_metadata(
    data: Mapping[str, Any],
    *,
    domain: str | None,
) -> tuple[int, float, float, bool, bool, bool, bool]:
    """Resolve charge/spin metadata independently of property supervision."""
    atoms = data.get("atoms")
    domain_data = data
    if domain is not None:
        domain_data = dict(data, atomistic_domain=domain)
    domain_id = _infer_domain_id(domain_data, atoms)

    atoms_info = getattr(atoms, "info", {}) if atoms is not None else {}
    if not isinstance(atoms_info, Mapping):
        atoms_info = {}
    charge_value = _atoms_info_value(atoms_info, "charge")
    spin_value = _atoms_info_value(atoms_info, "spin")
    if charge_value is None:
        charge_value = _specification_value(data, "total_charge")
    if spin_value is None:
        spin_value = _specification_value(data, "spin_multiplicity")

    # A molecule-only inference specification can lack an ASE object and an
    # explicit domain. Its electronic-state fields still identify the OMol
    # conditioning contract unambiguously.
    if domain_id == UNKNOWN_DOMAIN_ID and (
        charge_value is not None or spin_value is not None
    ):
        domain_id = OMOL25_DOMAIN_ID

    has_total_charge = domain_id == OMOL25_DOMAIN_ID and charge_value is not None
    has_spin_multiplicity = domain_id == OMOL25_DOMAIN_ID and spin_value is not None
    total_charge, charge_valid = _integer_metadata(charge_value)
    spin_multiplicity, spin_valid = _integer_metadata(spin_value, minimum=1)
    if domain_id != OMOL25_DOMAIN_ID:
        total_charge = 0.0
        spin_multiplicity = 0.0
        charge_valid = False
        spin_valid = False
    return (
        domain_id,
        total_charge,
        spin_multiplicity,
        has_total_charge,
        has_spin_multiplicity,
        charge_valid,
        spin_valid,
    )


def _set_electronic_state_features(
    data: dict,
    *,
    token_count: int,
    domain_id: int,
    total_charge: float,
    spin_multiplicity: float,
    charge_valid: bool,
    spin_valid: bool,
) -> None:
    """Broadcast one system's validated electronic state to token features."""
    expose_electronic_metadata = domain_id == OMOL25_DOMAIN_ID
    input_charge = total_charge if expose_electronic_metadata and charge_valid else 0.0
    input_spin = spin_multiplicity if expose_electronic_metadata and spin_valid else 0.0
    feats = data.setdefault("feats", {})
    feats["total_charge"] = torch.full(
        (token_count, 1), input_charge, dtype=torch.float32
    )
    feats["spin_multiplicity"] = torch.full(
        (token_count, 1), input_spin, dtype=torch.float32
    )
    # Invalid metadata is numerically neutral and explicitly unavailable.
    feats["has_total_charge"] = torch.full(
        (token_count, 1),
        expose_electronic_metadata and charge_valid,
        dtype=torch.float32,
    )
    feats["has_spin_multiplicity"] = torch.full(
        (token_count, 1),
        expose_electronic_metadata and spin_valid,
        dtype=torch.float32,
    )


def _bool_tensor(value: bool) -> torch.Tensor:
    """Create a one-system boolean tensor."""
    return torch.tensor([value], dtype=torch.bool)


def canonical_property_task_name(value: Any) -> str | None:
    """Return the canonical name for a recognized atomistic task alias."""
    task_id = _PROPERTY_TASK_IDS.get(str(value))
    return _PROPERTY_TASK_NAMES.get(task_id)


def infer_property_task_name(
    data: Mapping[str, Any], *, allow_mask_inference: bool = False
) -> str | None:
    """Infer a canonical atomistic task from metadata or conditioning masks.

    Training records carry ``sampled_condition_name`` while inference records
    may instead declare ``atomistic_property_task`` in their specification.
    Design inference inputs commonly declare neither, so fall back to the
    canonical whole-system masks used by the three atomistic tasks. Callers
    handling validation examples that have already lost their ``is_inference``
    marker can explicitly enable the same mask-based fallback.
    """
    condition_name = data.get("sampled_condition_name")
    if condition_name is None:
        condition_name = _specification_value(data, "atomistic_property_task")
    if condition_name is not None:
        # An explicit but unknown task should remain invalid instead of silently
        # being reinterpreted from masks (for example, after a specification typo).
        return canonical_property_task_name(condition_name)

    if not allow_mask_inference and not data.get("is_inference", False):
        return None
    atom_array = data.get("atom_array")
    if atom_array is None:
        return None
    required_annotations = {
        "is_motif_atom_with_fixed_seq",
        "is_motif_atom_with_fixed_coord",
    }
    if not required_annotations.issubset(atom_array.get_annotation_categories()):
        return None

    fixed_sequence = np.asarray(atom_array.is_motif_atom_with_fixed_seq, dtype=bool)
    fixed_coordinates = np.asarray(
        atom_array.is_motif_atom_with_fixed_coord, dtype=bool
    )
    if fixed_sequence.size == 0:
        return None
    if fixed_sequence.all() and fixed_coordinates.all():
        return "mlip"
    if fixed_sequence.all() and not fixed_coordinates.any():
        return "structure_prediction"
    if not fixed_sequence.any() and not fixed_coordinates.any():
        return "unconditional"
    return None


def _property_task_id(data: Mapping[str, Any]) -> int:
    """Map explicit or inferred conditioning to the atomistic task ID."""
    task_name = infer_property_task_name(data)
    return _PROPERTY_TASK_IDS.get(str(task_name), UNKNOWN_PROPERTY_TASK_ID)


def _set_system_metadata(
    data: dict,
    *,
    energy: float,
    energy_present: bool,
    force_present: bool,
    domain_id: int,
    source_atom_count: int,
    retained_atom_count: int,
    total_charge: float,
    spin_multiplicity: float,
    has_total_charge: bool,
    has_spin_multiplicity: bool,
    charge_valid: bool,
    spin_valid: bool,
    complete_structure: bool,
    property_task_id: int,
) -> None:
    """Populate the stable one-system metadata contract in targets and feats."""
    domain_valid = domain_id in {OMOL25_DOMAIN_ID, OMAT24_DOMAIN_ID}
    electronic_metadata_valid = domain_id != OMOL25_DOMAIN_ID or (
        charge_valid and spin_valid
    )
    property_valid = domain_valid and electronic_metadata_valid and complete_structure
    property_task_valid = property_task_id in {
        UNCONDITIONAL_PROPERTY_TASK_ID,
        STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
        MLIP_PROPERTY_TASK_ID,
    }
    energy_valid = property_valid and energy_present

    system_tensors = {
        # Preserve the raw electronic energy until elemental references have been
        # subtracted.  OMol25 totals can be large enough that float32 quantization
        # erases sub-meV/atom residuals before the training loss sees them.
        "energy_gt_B": torch.tensor([energy], dtype=torch.float64),
        "energy_valid_mask_B": _bool_tensor(energy_valid),
        "energy_present_mask_B": _bool_tensor(energy_present),
        "force_present_mask_B": _bool_tensor(force_present),
        "domain_id_B": torch.tensor([domain_id], dtype=torch.long),
        "domain_valid_mask_B": _bool_tensor(domain_valid),
        "source_atom_count_B": torch.tensor([source_atom_count], dtype=torch.long),
        "retained_atom_count_B": torch.tensor([retained_atom_count], dtype=torch.long),
        "total_charge_B": torch.tensor([total_charge], dtype=torch.float32),
        "spin_multiplicity_B": torch.tensor([spin_multiplicity], dtype=torch.float32),
        "has_total_charge_B": _bool_tensor(has_total_charge),
        "has_spin_multiplicity_B": _bool_tensor(has_spin_multiplicity),
        "charge_valid_mask_B": _bool_tensor(charge_valid),
        "spin_valid_mask_B": _bool_tensor(spin_valid),
        "property_valid_mask_B": _bool_tensor(property_valid),
        "property_task_id_B": torch.tensor([property_task_id], dtype=torch.long),
        "property_task_valid_mask_B": _bool_tensor(property_task_valid),
    }
    data.setdefault("ground_truth", {}).update(system_tensors)

    # Model code needs these system-level values without reaching into loss-only
    # targets.  Clone to prevent an in-place feature transform from mutating GT.
    feats = data.setdefault("feats", {})
    for key in (
        "domain_id_B",
        "domain_valid_mask_B",
        "source_atom_count_B",
        "retained_atom_count_B",
        "total_charge_B",
        "spin_multiplicity_B",
        "has_total_charge_B",
        "has_spin_multiplicity_B",
        "charge_valid_mask_B",
        "spin_valid_mask_B",
        "property_valid_mask_B",
        "property_task_id_B",
        "property_task_valid_mask_B",
    ):
        feats[key] = system_tensors[key].clone()


class AddAtomisticPropertyTargets(Transform):
    """Extract source-system labels before any cropping or atom filtering.

    Energies and forces are read from serialized ASE calculator results and are
    interpreted in ASE's native eV and eV/Angstrom units.  OMol25 total charge
    and spin multiplicity are read specifically from ``atoms.info``.

    Args:
        diffusion_batch_size: Number of independently augmented diffusion views.
        domain: Optional explicit ``"omol25"`` or ``"omat24"`` domain.  This is
            preferred for training datasets; periodicity is only a fallback for
            ad hoc inference inputs.
    """

    def __init__(self, diffusion_batch_size: int, domain: str | None = None):
        if diffusion_batch_size <= 0:
            raise ValueError("diffusion_batch_size must be positive.")
        self.diffusion_batch_size = diffusion_batch_size
        _validate_explicit_domain(domain)
        self.domain = domain

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atoms = data.get("atoms")
        source_atom_count = (
            len(atoms) if atoms is not None else atom_array.array_length()
        )
        current_atom_count = atom_array.array_length()
        source_alignment_valid = source_atom_count == current_atom_count

        results = _calculator_results(data, atoms)
        energy_value = _finite_scalar(results.get("energy"))
        energy_present = energy_value is not None
        energy = energy_value if energy_present else 0.0

        raw_forces = results.get("forces")
        force_values = np.zeros((current_atom_count, 3), dtype=np.float32)
        force_valid = np.zeros(current_atom_count, dtype=bool)
        force_present = False
        if raw_forces is not None and source_alignment_valid:
            try:
                source_forces = np.asarray(raw_forces, dtype=np.float32)
            except (TypeError, ValueError, OverflowError):
                source_forces = np.empty((0, 3), dtype=np.float32)
            if source_forces.shape == (source_atom_count, 3):
                finite_force_mask = np.isfinite(source_forces).all(axis=-1)
                # A non-finite force invalidates the system-level force label;
                # retaining only some atoms would change its data contract.
                force_present = bool(np.all(finite_force_mask))
                if force_present:
                    force_values = source_forces.copy()
                    force_valid[:] = True

        source_indices = (
            np.arange(source_atom_count, dtype=np.int64)
            if source_alignment_valid
            else np.full(current_atom_count, -1, dtype=np.int64)
        )
        atom_array.set_annotation(_SOURCE_INDEX_ANNOTATION, source_indices)
        atom_array.set_annotation(_FORCE_ANNOTATION, force_values)
        atom_array.set_annotation(_FORCE_VALID_ANNOTATION, force_valid)
        data["atom_array"] = atom_array

        (
            domain_id,
            total_charge,
            spin_multiplicity,
            has_total_charge,
            has_spin_multiplicity,
            charge_valid,
            spin_valid,
        ) = _electronic_state_metadata(data, domain=self.domain)

        data["_atomistic_property_energy"] = float(energy)
        data["_atomistic_property_energy_present"] = bool(energy_present)
        data["_atomistic_property_force_present"] = bool(force_present)
        _set_system_metadata(
            data,
            energy=energy,
            energy_present=energy_present,
            force_present=force_present,
            domain_id=domain_id,
            source_atom_count=source_atom_count,
            retained_atom_count=current_atom_count,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            has_total_charge=has_total_charge,
            has_spin_multiplicity=has_spin_multiplicity,
            charge_valid=charge_valid,
            spin_valid=spin_valid,
            complete_structure=source_alignment_valid,
            property_task_id=_property_task_id(data),
        )

        forces = (
            torch.from_numpy(force_values)
            .unsqueeze(0)
            .expand(self.diffusion_batch_size, -1, -1)
        )
        data["ground_truth"]["force_gt_D_L"] = forces.clone()
        data["ground_truth"]["force_valid_mask_L"] = torch.from_numpy(
            force_valid.copy()
        )
        return data


class AddElectronicStateConditioning(Transform):
    """Expose OMol charge and spin without enabling property targets.

    Charge and spin are model inputs for every OMol task: unconditional
    generation, structure prediction, and MLIP prediction. This transform is
    intentionally independent of energy/force target extraction so generative
    pretraining can retain its lightweight target contract.

    Args:
        domain: Optional explicit ``"omol25"`` or ``"omat24"`` dataset domain.
            Training datasets should provide it; inference can infer molecules
            from their requested electronic state.
    """

    def __init__(self, domain: str | None = None):
        _validate_explicit_domain(domain)
        self.domain = domain

    def forward(self, data: dict) -> dict:
        (
            domain_id,
            total_charge,
            spin_multiplicity,
            _has_total_charge,
            _has_spin_multiplicity,
            charge_valid,
            spin_valid,
        ) = _electronic_state_metadata(data, domain=self.domain)
        _set_electronic_state_features(
            data,
            token_count=len(get_token_starts(data["atom_array"])),
            domain_id=domain_id,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            charge_valid=charge_valid,
            spin_valid=spin_valid,
        )
        return data


class FinalizeAtomisticPropertyTargets(Transform):
    """Align force labels after cropping/padding and broadcast system features."""

    def __init__(self, diffusion_batch_size: int):
        if diffusion_batch_size <= 0:
            raise ValueError("diffusion_batch_size must be positive.")
        self.diffusion_batch_size = diffusion_batch_size

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        categories = atom_array.get_annotation_categories()
        n_atoms = atom_array.array_length()

        if _SOURCE_INDEX_ANNOTATION in categories:
            source_indices = np.asarray(
                atom_array.get_annotation(_SOURCE_INDEX_ANNOTATION), dtype=np.int64
            )
        else:
            source_indices = np.full(n_atoms, -1, dtype=np.int64)
        retained_source_mask = source_indices >= 0
        if "is_virtual" in categories:
            retained_source_mask &= ~np.asarray(atom_array.is_virtual, dtype=bool)
        retained_atom_count = int(np.unique(source_indices[retained_source_mask]).size)

        ground_truth = data.setdefault("ground_truth", {})
        source_atom_count = int(ground_truth["source_atom_count_B"].reshape(-1)[0])
        complete_structure = retained_atom_count == source_atom_count

        force_values = np.zeros((n_atoms, 3), dtype=np.float32)
        force_valid = np.zeros(n_atoms, dtype=bool)
        if _FORCE_ANNOTATION in categories:
            candidate_forces = np.asarray(
                atom_array.get_annotation(_FORCE_ANNOTATION), dtype=np.float32
            )
            if candidate_forces.shape == (n_atoms, 3):
                force_values = np.where(
                    np.isfinite(candidate_forces), candidate_forces, 0.0
                )
        if _FORCE_VALID_ANNOTATION in categories:
            candidate_valid = np.asarray(
                atom_array.get_annotation(_FORCE_VALID_ANNOTATION), dtype=bool
            )
            if candidate_valid.shape == (n_atoms,):
                force_valid = candidate_valid.copy()
        force_present = bool(ground_truth["force_present_mask_B"].reshape(-1)[0].item())
        force_valid &= retained_source_mask
        force_valid &= complete_structure and force_present
        force_values[~force_valid] = 0.0

        domain_id = int(ground_truth["domain_id_B"].reshape(-1)[0])
        energy = float(data.get("_atomistic_property_energy", 0.0))
        energy_present = bool(data.get("_atomistic_property_energy_present", False))
        total_charge = float(ground_truth["total_charge_B"].reshape(-1)[0])
        spin_multiplicity = float(ground_truth["spin_multiplicity_B"].reshape(-1)[0])
        has_total_charge = bool(ground_truth["has_total_charge_B"].reshape(-1)[0])
        has_spin_multiplicity = bool(
            ground_truth["has_spin_multiplicity_B"].reshape(-1)[0]
        )
        charge_valid = bool(ground_truth["charge_valid_mask_B"].reshape(-1)[0])
        spin_valid = bool(ground_truth["spin_valid_mask_B"].reshape(-1)[0])
        property_task_id = _property_task_id(data)

        _set_system_metadata(
            data,
            energy=energy,
            energy_present=energy_present,
            force_present=force_present,
            domain_id=domain_id,
            source_atom_count=source_atom_count,
            retained_atom_count=retained_atom_count,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            has_total_charge=has_total_charge,
            has_spin_multiplicity=has_spin_multiplicity,
            charge_valid=charge_valid,
            spin_valid=spin_valid,
            complete_structure=complete_structure,
            property_task_id=property_task_id,
        )

        property_valid = bool(ground_truth["property_valid_mask_B"].item())
        ground_truth["force_gt_D_L"] = (
            torch.from_numpy(force_values)
            .unsqueeze(0)
            .expand(self.diffusion_batch_size, -1, -1)
            .clone()
        )
        ground_truth["force_valid_mask_L"] = torch.from_numpy(force_valid) & (
            property_valid
        )

        _set_electronic_state_features(
            data,
            token_count=len(get_token_starts(atom_array)),
            domain_id=domain_id,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            charge_valid=charge_valid,
            spin_valid=spin_valid,
        )
        return data
