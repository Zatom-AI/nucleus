"""Intrinsic physical-geometry checks for predicted/generated RNA structures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zevals.rna.tasks import load_biomolecule_structure, rna_parent_base

logger = logging.getLogger(__name__)

_VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
}


def compute_rna_geometry_detailed(
    structure_paths: list[Path],
    *,
    modified_nucleotide_maps: Mapping[str, Mapping[str, str]] | None = None,
    clash_overlap_tolerance: float = 0.4,
    min_o3_p_distance: float = 1.2,
    max_o3_p_distance: float = 2.2,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Measure nonlocal heavy-atom clashes and phosphodiester continuity.

    Clash pairs within one residue or between adjacent residues in the same
    chain are excluded because those pairs include covalently connected atoms
    and their 1-3 neighbours. A nonlocal pair clashes when its van der Waals
    overlap exceeds ``clash_overlap_tolerance`` Angstrom.

    Chain continuity is evaluated from O3'(i)-P(i+1) distances. Missing link
    atoms reduce link coverage rather than being silently counted as breaks.
    """
    details: list[dict[str, Any]] = []
    maps = modified_nucleotide_maps or {}
    for path in structure_paths:
        try:
            detail = _compute_single_geometry(
                path,
                modified_nucleotide_map=maps.get(str(path)),
                clash_overlap_tolerance=clash_overlap_tolerance,
                min_o3_p_distance=min_o3_p_distance,
                max_o3_p_distance=max_o3_p_distance,
            )
            details.append({"structure": str(path), "status": "complete", **detail})
        except Exception as exc:
            logger.warning("RNA geometry evaluation failed for %s", path, exc_info=True)
            details.append(
                {
                    "structure": str(path),
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    complete = [detail for detail in details if detail["status"] == "complete"]
    total = len(structure_paths)
    aggregate = {
        "clash_free_rate": _mean(
            [float(detail["clash_pair_count"] == 0) for detail in complete]
        ),
        "mean_clashing_atom_fraction": _mean(
            [float(detail["clashing_atom_fraction"]) for detail in complete]
        ),
        "chain_continuity_rate": _weighted_rate(
            complete, "continuous_link_count", "evaluated_link_count"
        ),
        "chain_link_coverage": _weighted_rate(
            complete, "evaluated_link_count", "expected_link_count"
        ),
        "geometry_evaluation_coverage": len(complete) / total if total else 0.0,
        "geometry_evaluated_structures": float(len(complete)),
        "geometry_failed_structures": float(total - len(complete)),
    }
    return aggregate, details


def _compute_single_geometry(
    path: Path,
    *,
    modified_nucleotide_map: Mapping[str, str] | None,
    clash_overlap_tolerance: float,
    min_o3_p_distance: float,
    max_o3_p_distance: float,
) -> dict[str, float | int]:
    structure = load_biomolecule_structure(path)
    model = next(structure.get_models())
    atoms: list[tuple[np.ndarray, str, str, int]] = []
    chain_residues: dict[str, list[Any]] = {}

    for chain in model:
        residues = [
            residue
            for residue in chain
            if rna_parent_base(residue, modified_nucleotide_map) is not None
        ]
        if not residues:
            continue
        chain_id = str(chain.id)
        chain_residues[chain_id] = residues
        for residue_index, residue in enumerate(residues):
            for atom in residue:
                element = _atom_element(atom)
                if element == "H":
                    continue
                atoms.append(
                    (
                        np.asarray(atom.coord, dtype=np.float64),
                        element,
                        chain_id,
                        residue_index,
                    )
                )

    if not atoms:
        raise ValueError(f"No supported RNA heavy atoms found in {path}")

    clash_pairs = 0
    clashing_atoms: set[int] = set()
    for first_index in range(len(atoms)):
        coord_i, element_i, chain_i, residue_i = atoms[first_index]
        radius_i = _VDW_RADII.get(element_i, 1.70)
        for second_index in range(first_index + 1, len(atoms)):
            coord_j, element_j, chain_j, residue_j = atoms[second_index]
            if chain_i == chain_j and abs(residue_i - residue_j) <= 1:
                continue
            radius_j = _VDW_RADII.get(element_j, 1.70)
            distance = float(np.linalg.norm(coord_i - coord_j))
            if distance < radius_i + radius_j - clash_overlap_tolerance:
                clash_pairs += 1
                clashing_atoms.update((first_index, second_index))

    expected_links = 0
    evaluated_links = 0
    continuous_links = 0
    chain_breaks = 0
    for residues in chain_residues.values():
        for previous, following in zip(residues, residues[1:]):
            expected_links += 1
            if "O3'" not in previous or "P" not in following:
                continue
            evaluated_links += 1
            link_distance = float(
                np.linalg.norm(previous["O3'"].coord - following["P"].coord)
            )
            if min_o3_p_distance <= link_distance <= max_o3_p_distance:
                continuous_links += 1
            else:
                chain_breaks += 1

    return {
        "heavy_atom_count": len(atoms),
        "clash_pair_count": clash_pairs,
        "clashing_atom_count": len(clashing_atoms),
        "clashing_atom_fraction": len(clashing_atoms) / len(atoms),
        "expected_link_count": expected_links,
        "evaluated_link_count": evaluated_links,
        "continuous_link_count": continuous_links,
        "chain_break_count": chain_breaks,
        "chain_continuity": (
            continuous_links / evaluated_links if evaluated_links else float("nan")
        ),
        "chain_link_coverage": (
            evaluated_links / expected_links if expected_links else 1.0
        ),
    }


def _atom_element(atom: Any) -> str:
    element = str(getattr(atom, "element", "")).strip().upper()
    if element:
        return element
    name = atom.get_name().strip().upper()
    return next((character for character in name if character.isalpha()), "C")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _weighted_rate(
    details: list[dict[str, Any]], numerator_key: str, denominator_key: str
) -> float:
    numerator = sum(float(detail[numerator_key]) for detail in details)
    denominator = sum(float(detail[denominator_key]) for detail in details)
    return numerator / denominator if denominator else float("nan")
