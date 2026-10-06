"""Joint RNA--protein interface geometry and target-recovery metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from zevals.rna.tasks import (
    RNAEvaluationSample,
    load_biomolecule_structure,
    rna_parent_base,
)

logger = logging.getLogger(__name__)

_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
_VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80}


def contains_rna_protein_interface(
    path: Path | str, modified_nucleotide_map: dict[str, str] | None = None
) -> bool:
    """Return whether a structure contains at least one RNA and protein residue."""
    structure = load_biomolecule_structure(Path(path))
    model = next(structure.get_models())
    has_rna = any(
        rna_parent_base(residue, modified_nucleotide_map) is not None
        for chain in model
        for residue in chain
    )
    has_protein = any(
        residue.get_resname().strip().upper() in _PROTEIN_RESIDUES
        for chain in model
        for residue in chain
    )
    return has_rna and has_protein


def compute_rna_protein_interface_detailed(
    samples: list[RNAEvaluationSample],
    *,
    contact_cutoff: float = 5.0,
    clash_overlap_tolerance: float = 0.4,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compare predicted RNA--protein interfaces with supplied targets."""
    eligible = [sample for sample in samples if sample.reference_structure_path]
    details: list[dict[str, Any]] = []
    for sample in eligible:
        assert sample.reference_structure_path is not None
        try:
            metrics = compare_rna_protein_interfaces(
                sample.reference_structure_path,
                sample.structure_path,
                modified_nucleotide_map=sample.modified_nucleotide_map,
                contact_cutoff=contact_cutoff,
                clash_overlap_tolerance=clash_overlap_tolerance,
            )
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "status": "complete",
                    "structure": str(sample.structure_path),
                    "reference_structure": str(sample.reference_structure_path),
                    **metrics,
                }
            )
        except Exception as exc:
            logger.warning("RNA--protein interface failed for %s", sample.sample_id)
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    complete = [row for row in details if row["status"] == "complete"]
    aggregate = {
        "mean_interface_contact_precision": _mean_key(
            complete, "interface_contact_precision"
        ),
        "mean_interface_contact_recall": _mean_key(
            complete, "interface_contact_recall"
        ),
        "mean_interface_contact_f1": _mean_key(complete, "interface_contact_f1"),
        "mean_protein_aligned_interface_rna_rmsd": _mean_key(
            complete, "protein_aligned_interface_rna_rmsd"
        ),
        "interface_clash_free_rate": _mean(
            [float(row["interface_clash_count"] == 0) for row in complete]
        ),
        "interface_evaluation_coverage": (
            len(complete) / len(eligible) if eligible else 0.0
        ),
        "interface_evaluated_structures": float(len(complete)),
        "interface_failed_structures": float(len(eligible) - len(complete)),
    }
    return aggregate, details


def compare_rna_protein_interfaces(
    reference_path: Path | str,
    candidate_path: Path | str,
    *,
    modified_nucleotide_map: dict[str, str] | None = None,
    contact_cutoff: float = 5.0,
    clash_overlap_tolerance: float = 0.4,
) -> dict[str, float | int]:
    """Return contact recovery, interface RMSD, and cross-polymer clashes."""
    reference = _complex_geometry(
        Path(reference_path), modified_nucleotide_map, contact_cutoff
    )
    candidate = _complex_geometry(
        Path(candidate_path), modified_nucleotide_map, contact_cutoff
    )
    if not reference["rna_atoms"] or not reference["protein_atoms"]:
        raise ValueError("Reference does not contain both RNA and protein")
    if not candidate["rna_atoms"] or not candidate["protein_atoms"]:
        raise ValueError("Candidate does not contain both RNA and protein")

    reference_contacts = reference["contacts"]
    candidate_contacts = candidate["contacts"]
    recovered = reference_contacts & candidate_contacts
    precision = len(recovered) / len(candidate_contacts) if candidate_contacts else 0.0
    recall = len(recovered) / len(reference_contacts) if reference_contacts else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    interface_rmsd, coverage = _protein_aligned_rna_interface_rmsd(reference, candidate)
    clash_count, clashing_atoms = _cross_polymer_clashes(
        candidate["rna_atoms"],
        candidate["protein_atoms"],
        clash_overlap_tolerance,
    )
    candidate_atom_count = len(candidate["rna_atoms"]) + len(candidate["protein_atoms"])
    return {
        "reference_interface_contact_count": len(reference_contacts),
        "candidate_interface_contact_count": len(candidate_contacts),
        "recovered_interface_contact_count": len(recovered),
        "interface_contact_precision": precision,
        "interface_contact_recall": recall,
        "interface_contact_f1": f1,
        "protein_aligned_interface_rna_rmsd": interface_rmsd,
        "interface_rna_atom_coverage": coverage,
        "interface_clash_count": clash_count,
        "interface_clashing_atom_fraction": (
            len(clashing_atoms) / candidate_atom_count if candidate_atom_count else 0.0
        ),
    }


def _complex_geometry(
    path: Path,
    modified_nucleotide_map: dict[str, str] | None,
    contact_cutoff: float,
) -> dict[str, Any]:
    structure = load_biomolecule_structure(path)
    model = next(structure.get_models())
    rna_atoms: list[tuple[tuple[int, int, str], np.ndarray, str]] = []
    protein_atoms: list[tuple[tuple[int, int, str], np.ndarray, str]] = []
    protein_ca: dict[tuple[int, int], np.ndarray] = {}
    rna_c1: dict[tuple[int, int], np.ndarray] = {}
    rna_chain_index = protein_chain_index = 0

    for chain in model:
        rna_residues = [
            residue
            for residue in chain
            if rna_parent_base(residue, modified_nucleotide_map) is not None
        ]
        protein_residues = [
            residue
            for residue in chain
            if residue.get_resname().strip().upper() in _PROTEIN_RESIDUES
        ]
        if rna_residues:
            for residue_index, residue in enumerate(rna_residues):
                for atom in residue:
                    if _atom_element(atom) == "H":
                        continue
                    atom_name = atom.get_name().strip().replace("*", "'")
                    coordinate = np.asarray(atom.coord, dtype=np.float64)
                    key = (rna_chain_index, residue_index, atom_name)
                    rna_atoms.append((key, coordinate, _atom_element(atom)))
                    if atom_name == "C1'":
                        rna_c1[(rna_chain_index, residue_index)] = coordinate
            rna_chain_index += 1
        if protein_residues:
            for residue_index, residue in enumerate(protein_residues):
                for atom in residue:
                    if _atom_element(atom) == "H":
                        continue
                    atom_name = atom.get_name().strip()
                    coordinate = np.asarray(atom.coord, dtype=np.float64)
                    key = (protein_chain_index, residue_index, atom_name)
                    protein_atoms.append((key, coordinate, _atom_element(atom)))
                    if atom_name == "CA":
                        protein_ca[(protein_chain_index, residue_index)] = coordinate
            protein_chain_index += 1

    contacts = _residue_contacts(rna_atoms, protein_atoms, contact_cutoff)
    return {
        "rna_atoms": rna_atoms,
        "protein_atoms": protein_atoms,
        "rna_c1": rna_c1,
        "protein_ca": protein_ca,
        "contacts": contacts,
    }


def _protein_aligned_rna_interface_rmsd(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[float, float]:
    protein_keys = sorted(set(reference["protein_ca"]) & set(candidate["protein_ca"]))
    if len(protein_keys) < 3:
        return float("nan"), 0.0
    reference_protein = np.asarray(
        [reference["protein_ca"][key] for key in protein_keys]
    )
    candidate_protein = np.asarray(
        [candidate["protein_ca"][key] for key in protein_keys]
    )
    rotation, candidate_center, reference_center = _kabsch_transform(
        reference_protein, candidate_protein
    )
    interface_rna_residues = {
        (contact[0], contact[1]) for contact in reference["contacts"]
    }
    rna_keys = sorted(
        interface_rna_residues & set(reference["rna_c1"]) & set(candidate["rna_c1"])
    )
    if not rna_keys:
        return float("nan"), 0.0
    reference_rna = np.asarray([reference["rna_c1"][key] for key in rna_keys])
    candidate_rna = np.asarray([candidate["rna_c1"][key] for key in rna_keys])
    aligned = (candidate_rna - candidate_center) @ rotation + reference_center
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference_rna) ** 2, axis=1))))
    coverage = (
        len(rna_keys) / len(interface_rna_residues) if interface_rna_residues else 1.0
    )
    return rmsd, coverage


def _kabsch_transform(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_center = reference.mean(axis=0)
    candidate_center = candidate.mean(axis=0)
    covariance = (candidate - candidate_center).T @ (reference - reference_center)
    left, _, right_t = np.linalg.svd(covariance)
    if np.linalg.det(left @ right_t) < 0:
        left[:, -1] *= -1
    return left @ right_t, candidate_center, reference_center


def _cross_polymer_clashes(
    rna_atoms: list[tuple[tuple[int, int, str], np.ndarray, str]],
    protein_atoms: list[tuple[tuple[int, int, str], np.ndarray, str]],
    overlap_tolerance: float,
) -> tuple[int, set[tuple[str, int]]]:
    if not rna_atoms or not protein_atoms:
        return 0, set()
    protein_coordinates = np.asarray([atom[1] for atom in protein_atoms])
    protein_radii = np.asarray([_VDW_RADII.get(atom[2], 1.7) for atom in protein_atoms])
    clashes = 0
    clashing_atoms: set[tuple[str, int]] = set()
    chunk_size = 256
    for start in range(0, len(rna_atoms), chunk_size):
        chunk = rna_atoms[start : start + chunk_size]
        rna_coordinates = np.asarray([atom[1] for atom in chunk])
        rna_radii = np.asarray([_VDW_RADII.get(atom[2], 1.7) for atom in chunk])
        displacement = (
            rna_coordinates[:, np.newaxis, :] - protein_coordinates[np.newaxis, :, :]
        )
        distance_squared = np.sum(displacement * displacement, axis=-1)
        cutoffs = (
            rna_radii[:, np.newaxis] + protein_radii[np.newaxis, :] - overlap_tolerance
        )
        rna_indices, protein_indices = np.nonzero(distance_squared < cutoffs * cutoffs)
        clashes += len(rna_indices)
        clashing_atoms.update(("rna", start + int(index)) for index in rna_indices)
        clashing_atoms.update(("protein", int(index)) for index in protein_indices)
    return clashes, clashing_atoms


def _residue_contacts(
    rna_atoms: list[tuple[tuple[int, int, str], np.ndarray, str]],
    protein_atoms: list[tuple[tuple[int, int, str], np.ndarray, str]],
    contact_cutoff: float,
) -> set[tuple[int, int, int, int]]:
    """Return cross-polymer residue contacts using bounded vectorized chunks."""
    if not rna_atoms or not protein_atoms:
        return set()
    protein_coordinates = np.asarray([atom[1] for atom in protein_atoms])
    contacts: set[tuple[int, int, int, int]] = set()
    cutoff_squared = contact_cutoff * contact_cutoff
    chunk_size = 256
    for start in range(0, len(rna_atoms), chunk_size):
        chunk = rna_atoms[start : start + chunk_size]
        rna_coordinates = np.asarray([atom[1] for atom in chunk])
        displacement = (
            rna_coordinates[:, np.newaxis, :] - protein_coordinates[np.newaxis, :, :]
        )
        distance_squared = np.sum(displacement * displacement, axis=-1)
        rna_indices, protein_indices = np.nonzero(distance_squared <= cutoff_squared)
        for rna_index, protein_index in zip(rna_indices, protein_indices, strict=True):
            rna_key = chunk[int(rna_index)][0]
            protein_key = protein_atoms[int(protein_index)][0]
            contacts.add((*rna_key[:2], *protein_key[:2]))
    return contacts


def _atom_element(atom: Any) -> str:
    element = str(getattr(atom, "element", "")).strip().upper()
    if element:
        return element
    return next(
        (character for character in atom.get_name().upper() if character.isalpha()),
        "C",
    )


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    return _mean([float(row[key]) for row in rows if np.isfinite(float(row[key]))])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
