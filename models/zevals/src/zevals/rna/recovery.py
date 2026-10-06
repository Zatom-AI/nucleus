"""Direct recovery metrics against an original RNA target structure."""

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

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def compute_target_recovery_detailed(
    samples: list[RNAEvaluationSample],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compare predictions/generations directly with supplied target structures.

    C1' RMSD is Kabsch-aligned. C1' and all-heavy-atom lDDT are alignment-free.
    Atoms are matched by chain order, residue order, and atom name so harmless
    chain-ID or residue-number rewrites do not destroy correspondence.
    """
    eligible = [sample for sample in samples if sample.reference_structure_path]
    details: list[dict[str, Any]] = []
    for sample in eligible:
        assert sample.reference_structure_path is not None
        try:
            metrics = compare_rna_structures(
                sample.reference_structure_path,
                sample.structure_path,
                modified_nucleotide_map=sample.modified_nucleotide_map,
            )
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "structure": str(sample.structure_path),
                    "reference_structure": str(sample.reference_structure_path),
                    "status": "complete",
                    **metrics,
                }
            )
        except Exception as exc:
            logger.warning(
                "RNA target recovery failed for %s", sample.sample_id, exc_info=True
            )
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "structure": str(sample.structure_path),
                    "reference_structure": str(sample.reference_structure_path),
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    complete = [detail for detail in details if detail["status"] == "complete"]
    aggregate = {
        "mean_c1_aligned_rmsd": _mean_key(complete, "c1_aligned_rmsd"),
        "mean_c1_lddt": _mean_key(complete, "c1_lddt"),
        "mean_all_atom_lddt": _mean_key(complete, "all_atom_lddt"),
        "mean_c1_coverage": _mean_key(complete, "c1_coverage"),
        "mean_all_atom_coverage": _mean_key(complete, "all_atom_coverage"),
        "target_recovery_coverage": (
            len(complete) / len(eligible) if eligible else 0.0
        ),
        "target_recovery_evaluated_structures": float(len(complete)),
        "target_recovery_failed_structures": float(len(eligible) - len(complete)),
    }
    return aggregate, details


def compare_rna_structures(
    reference_path: Path | str,
    candidate_path: Path | str,
    *,
    modified_nucleotide_map: dict[str, str] | None = None,
) -> dict[str, float | int]:
    """Return direct coordinate-recovery metrics for one RNA structure pair."""
    reference_atoms = _rna_atom_coordinates(
        Path(reference_path), modified_nucleotide_map
    )
    candidate_atoms = _rna_atom_coordinates(
        Path(candidate_path), modified_nucleotide_map
    )
    common_keys = sorted(set(reference_atoms) & set(candidate_atoms))
    if not common_keys:
        raise ValueError("Reference and candidate have no matching canonical RNA atoms")

    reference_all = np.asarray([reference_atoms[key] for key in common_keys])
    candidate_all = np.asarray([candidate_atoms[key] for key in common_keys])
    c1_keys = [key for key in common_keys if key[-1] == "C1'"]
    if not c1_keys:
        raise ValueError("Reference and candidate have no matching RNA C1' atoms")
    reference_c1 = np.asarray([reference_atoms[key] for key in c1_keys])
    candidate_c1 = np.asarray([candidate_atoms[key] for key in c1_keys])

    reference_c1_count = sum(key[-1] == "C1'" for key in reference_atoms)
    return {
        "c1_aligned_rmsd": _aligned_rmsd(reference_c1, candidate_c1),
        "c1_lddt": _lddt(reference_c1, candidate_c1),
        "all_atom_lddt": _lddt(reference_all, candidate_all),
        "c1_coverage": len(c1_keys) / reference_c1_count,
        "all_atom_coverage": len(common_keys) / len(reference_atoms),
        "matched_c1_atoms": len(c1_keys),
        "matched_heavy_atoms": len(common_keys),
    }


def _rna_atom_coordinates(
    path: Path, modified_nucleotide_map: dict[str, str] | None
) -> dict[tuple[int, int, str], np.ndarray]:
    structure = load_biomolecule_structure(path)
    model = next(structure.get_models())
    coordinates: dict[tuple[int, int, str], np.ndarray] = {}
    rna_chain_index = 0
    for chain in model:
        residues = [
            residue
            for residue in chain
            if rna_parent_base(residue, modified_nucleotide_map) is not None
        ]
        if not residues:
            continue
        for residue_index, residue in enumerate(residues):
            for atom in residue:
                element = str(getattr(atom, "element", "")).strip().upper()
                if element == "H":
                    continue
                atom_name = atom.get_name().strip().replace("*", "'")
                coordinates[(rna_chain_index, residue_index, atom_name)] = np.asarray(
                    atom.coord, dtype=np.float64
                )
        rna_chain_index += 1
    if not coordinates:
        raise ValueError(f"No supported RNA atoms found in {path}")
    return coordinates


def _aligned_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_centered = reference - reference.mean(axis=0)
    candidate_centered = candidate - candidate.mean(axis=0)
    covariance = candidate_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    if np.linalg.det(left @ right_t) < 0:
        left[:, -1] *= -1
    rotation = left @ right_t
    aligned = candidate_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))


def _lddt(
    reference: np.ndarray, candidate: np.ndarray, inclusion_radius: float = 15.0
) -> float:
    if len(reference) < 2:
        return float("nan")
    included_count = 0
    threshold_counts = np.zeros(len(_LDDT_THRESHOLDS), dtype=np.int64)
    chunk_size = 256
    for start in range(0, len(reference), chunk_size):
        stop = min(start + chunk_size, len(reference))
        reference_distances = np.linalg.norm(
            reference[start:stop, np.newaxis, :] - reference[np.newaxis, :, :],
            axis=-1,
        )
        candidate_distances = np.linalg.norm(
            candidate[start:stop, np.newaxis, :] - candidate[np.newaxis, :, :],
            axis=-1,
        )
        row_indices = np.arange(start, stop)[:, np.newaxis]
        column_indices = np.arange(len(reference))[np.newaxis, :]
        included = (column_indices > row_indices) & (
            reference_distances < inclusion_radius
        )
        distance_errors = np.abs(
            candidate_distances[included] - reference_distances[included]
        )
        included_count += len(distance_errors)
        for threshold_index, threshold in enumerate(_LDDT_THRESHOLDS):
            threshold_counts[threshold_index] += np.count_nonzero(
                distance_errors < threshold
            )
    if included_count == 0:
        return float("nan")
    return float(np.mean(threshold_counts / included_count))


def _mean_key(details: list[dict[str, Any]], key: str) -> float:
    values = [float(detail[key]) for detail in details]
    finite_values = [value for value in values if np.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")
