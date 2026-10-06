"""RNA covalent stereochemistry and base-interaction plausibility metrics."""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zevals.rna.tasks import load_biomolecule_structure, rna_parent_base

logger = logging.getLogger(__name__)

_COMMON_BONDS = (
    ("P", "OP1"),
    ("P", "OP2"),
    ("P", "O5'"),
    ("O5'", "C5'"),
    ("C5'", "C4'"),
    ("C4'", "O4'"),
    ("C4'", "C3'"),
    ("C3'", "O3'"),
    ("C3'", "C2'"),
    ("C2'", "O2'"),
    ("C2'", "C1'"),
    ("C1'", "O4'"),
)
_BASE_BONDS = {
    "A": (
        ("C1'", "N9"),
        ("N9", "C8"),
        ("C8", "N7"),
        ("N7", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C4", "N9"),
        ("C6", "N6"),
    ),
    "G": (
        ("C1'", "N9"),
        ("N9", "C8"),
        ("C8", "N7"),
        ("N7", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C4", "N9"),
        ("C6", "O6"),
        ("C2", "N2"),
    ),
    "C": (
        ("C1'", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "N4"),
    ),
    "U": (
        ("C1'", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "O4"),
    ),
}
_BASE_RING_ATOMS = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"),
    "G": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"),
    "C": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "U": ("N1", "C2", "N3", "C4", "C5", "C6"),
}
_SUGAR_TORSIONS = (
    ("O4'", "C1'", "C2'", "C3'"),
    ("C1'", "C2'", "C3'", "C4'"),
    ("C2'", "C3'", "C4'", "O4'"),
    ("C3'", "C4'", "O4'", "C1'"),
    ("C4'", "O4'", "C1'", "C2'"),
)
_VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80}


def compute_rna_stereochemistry_detailed(
    structure_paths: list[Path],
    *,
    modified_nucleotide_maps: Mapping[str, Mapping[str, str]] | None = None,
    bond_tolerance: float = 0.20,
    angle_tolerance_degrees: float = 25.0,
    clash_overlap_tolerance: float = 0.4,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate covalent geometry, sugar pucker, pairing, and stacking.

    Coverage is reported for every topology-dependent metric. Modified
    residues use their canonical parent for the conserved atom topology;
    atoms introduced by the modification are not assigned invented bonds.
    """
    details: list[dict[str, Any]] = []
    maps = modified_nucleotide_maps or {}
    for path in structure_paths:
        try:
            metrics = _single_stereochemistry(
                path,
                modified_nucleotide_map=maps.get(str(path)),
                bond_tolerance=bond_tolerance,
                angle_tolerance_degrees=angle_tolerance_degrees,
                clash_overlap_tolerance=clash_overlap_tolerance,
            )
            details.append({"structure": str(path), "status": "complete", **metrics})
        except Exception as exc:
            logger.warning("RNA stereochemistry failed for %s", path, exc_info=True)
            details.append(
                {"structure": str(path), "status": "failed", "error": repr(exc)}
            )

    complete = [row for row in details if row["status"] == "complete"]
    total = len(structure_paths)
    aggregate = {
        "bond_length_valid_fraction": _ratio_sum(
            complete, "valid_bond_count", "evaluated_bond_count"
        ),
        "bond_length_coverage": _ratio_sum(
            complete, "evaluated_bond_count", "expected_bond_count"
        ),
        "bond_angle_valid_fraction": _ratio_sum(
            complete, "valid_angle_count", "evaluated_angle_count"
        ),
        "bond_angle_coverage": _ratio_sum(
            complete, "evaluated_angle_count", "expected_angle_count"
        ),
        "intra_residue_clash_free_rate": _mean(
            [float(row["intra_residue_clash_count"] == 0) for row in complete]
        ),
        "local_steric_clash_free_rate": _mean(
            [
                float(
                    row["intra_residue_clash_count"]
                    + row["adjacent_residue_clash_count"]
                    == 0
                )
                for row in complete
            ]
        ),
        "sugar_pucker_valid_fraction": _ratio_sum(
            complete, "valid_sugar_pucker_count", "evaluated_sugar_pucker_count"
        ),
        "sugar_pucker_coverage": _ratio_sum(
            complete, "evaluated_sugar_pucker_count", "rna_residue_count"
        ),
        "north_sugar_pucker_fraction": _ratio_sum(
            complete, "north_sugar_pucker_count", "evaluated_sugar_pucker_count"
        ),
        "south_sugar_pucker_fraction": _ratio_sum(
            complete, "south_sugar_pucker_count", "evaluated_sugar_pucker_count"
        ),
        "base_paired_residue_fraction": _ratio_sum(
            complete, "base_paired_residue_count", "base_geometry_residue_count"
        ),
        "base_stacked_residue_fraction": _ratio_sum(
            complete, "base_stacked_residue_count", "base_geometry_residue_count"
        ),
        "base_interaction_coverage": _ratio_sum(
            complete, "base_geometry_residue_count", "rna_residue_count"
        ),
        "modified_residue_fraction": _ratio_sum(
            complete, "modified_residue_count", "rna_residue_count"
        ),
        "stereochemistry_evaluation_coverage": len(complete) / total if total else 0.0,
        "stereochemistry_evaluated_structures": float(len(complete)),
        "stereochemistry_failed_structures": float(total - len(complete)),
    }
    return aggregate, details


def _single_stereochemistry(
    path: Path,
    *,
    modified_nucleotide_map: Mapping[str, str] | None,
    bond_tolerance: float,
    angle_tolerance_degrees: float,
    clash_overlap_tolerance: float,
) -> dict[str, float | int]:
    structure = load_biomolecule_structure(path)
    model = next(structure.get_models())
    residues: list[tuple[str, int, Any, str]] = []
    for chain in model:
        rna_index = 0
        for residue in chain:
            parent = rna_parent_base(residue, modified_nucleotide_map)
            if parent is None:
                continue
            residues.append((str(chain.id), rna_index, residue, parent))
            rna_index += 1
    if not residues:
        raise ValueError(f"No supported RNA residues found in {path}")

    expected_bonds = evaluated_bonds = valid_bonds = 0
    expected_angles = evaluated_angles = valid_angles = 0
    intra_clashes = 0
    adjacent_clashes = 0
    evaluated_puckers = valid_puckers = north_puckers = south_puckers = 0
    base_geometries: list[dict[str, Any]] = []
    modified_count = 0

    for chain_id, residue_index, residue, parent in residues:
        atom_coords = _atom_coordinates(residue)
        is_modified = residue.get_resname().strip().upper() != parent
        modified_count += int(is_modified)
        # The sugar/phosphate topology is conserved. A modification's base
        # connectivity is not inferred from its canonical sequence parent
        # (PSU is the classic counterexample), so only canonical residues get
        # canonical base covalent checks.
        bonds = _COMMON_BONDS if is_modified else (*_COMMON_BONDS, *_BASE_BONDS[parent])
        expected_bonds += len(bonds)
        adjacency: dict[str, set[str]] = {}
        for first, second in bonds:
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
            if first not in atom_coords or second not in atom_coords:
                continue
            evaluated_bonds += 1
            distance = float(np.linalg.norm(atom_coords[first] - atom_coords[second]))
            valid_bonds += int(
                abs(distance - _ideal_bond_length(first, second)) <= bond_tolerance
            )

        for center, neighbours in adjacency.items():
            neighbour_pairs = list(itertools.combinations(sorted(neighbours), 2))
            expected_angles += len(neighbour_pairs)
            for first, second in neighbour_pairs:
                if not {first, center, second} <= atom_coords.keys():
                    continue
                evaluated_angles += 1
                angle = _angle(
                    atom_coords[first], atom_coords[center], atom_coords[second]
                )
                ideal = _ideal_angle(center)
                valid_angles += int(abs(angle - ideal) <= angle_tolerance_degrees)

        intra_clashes += _intra_residue_clashes(
            atom_coords,
            bonds,
            overlap_tolerance=clash_overlap_tolerance,
        )

        if all(set(torsion) <= atom_coords.keys() for torsion in _SUGAR_TORSIONS):
            evaluated_puckers += 1
            torsions = [
                _dihedral(*(atom_coords[name] for name in torsion))
                for torsion in _SUGAR_TORSIONS
            ]
            phase, amplitude = _pseudorotation_from_torsions(torsions)
            north = phase <= 45.0 or phase >= 315.0
            south = 135.0 <= phase <= 225.0
            amplitude_valid = 15.0 <= amplitude <= 60.0
            north_puckers += int(north)
            south_puckers += int(south)
            valid_puckers += int(amplitude_valid and (north or south))

        ring_names = _BASE_RING_ATOMS[parent]
        if all(name in atom_coords for name in ring_names):
            ring = np.asarray([atom_coords[name] for name in ring_names])
            centroid = ring.mean(axis=0)
            _, _, right_t = np.linalg.svd(ring - centroid)
            polar_atoms = [
                coord
                for name, coord in atom_coords.items()
                if name in set(ring_names) | {"N2", "N4", "N6", "O2", "O4", "O6"}
                and name.startswith(("N", "O"))
            ]
            base_geometries.append(
                {
                    "key": (chain_id, residue_index),
                    "centroid": centroid,
                    "normal": right_t[-1],
                    "polar_atoms": polar_atoms,
                }
            )

    for previous, following in zip(residues, residues[1:]):
        if previous[0] != following[0] or following[1] != previous[1] + 1:
            continue
        previous_atoms = _atom_coordinates(previous[2])
        following_atoms = _atom_coordinates(following[2])
        adjacent_clashes += _adjacent_residue_clashes(
            previous_atoms,
            following_atoms,
            overlap_tolerance=clash_overlap_tolerance,
        )
        expected_bonds += 1
        if "O3'" in previous_atoms and "P" in following_atoms:
            evaluated_bonds += 1
            distance = float(
                np.linalg.norm(previous_atoms["O3'"] - following_atoms["P"])
            )
            valid_bonds += int(abs(distance - 1.60) <= bond_tolerance)
        for first_name, center_name, second_name, ideal in (
            ("C3'", "O3'", "P", 120.0),
            ("O3'", "P", "O5'", 109.5),
        ):
            expected_angles += 1
            first_atoms = previous_atoms
            center_atoms = previous_atoms if center_name == "O3'" else following_atoms
            second_atoms = following_atoms
            if (
                first_name not in first_atoms
                or center_name not in center_atoms
                or second_name not in second_atoms
            ):
                continue
            evaluated_angles += 1
            angle = _angle(
                first_atoms[first_name],
                center_atoms[center_name],
                second_atoms[second_name],
            )
            valid_angles += int(abs(angle - ideal) <= angle_tolerance_degrees)

    paired: set[tuple[str, int]] = set()
    stacked: set[tuple[str, int]] = set()
    pair_count = stack_count = 0
    for first, second in itertools.combinations(base_geometries, 2):
        delta = second["centroid"] - first["centroid"]
        centroid_distance = float(np.linalg.norm(delta))
        normal_cosine = abs(float(np.dot(first["normal"], second["normal"])))
        plane_separation = abs(float(np.dot(delta, first["normal"])))
        same_chain_adjacent = (
            first["key"][0] == second["key"][0]
            and abs(first["key"][1] - second["key"][1]) <= 1
        )
        polar_distance = _minimum_distance(first["polar_atoms"], second["polar_atoms"])
        if (
            not same_chain_adjacent
            and 4.0 <= centroid_distance <= 12.0
            and normal_cosine >= math.cos(math.radians(35.0))
            and plane_separation <= 2.5
            and 2.2 <= polar_distance <= 4.0
        ):
            pair_count += 1
            paired.update((first["key"], second["key"]))
        if (
            2.8 <= centroid_distance <= 6.0
            and normal_cosine >= math.cos(math.radians(35.0))
            and 2.5 <= plane_separation <= 4.5
        ):
            stack_count += 1
            stacked.update((first["key"], second["key"]))

    return {
        "rna_residue_count": len(residues),
        "modified_residue_count": modified_count,
        "expected_bond_count": expected_bonds,
        "evaluated_bond_count": evaluated_bonds,
        "valid_bond_count": valid_bonds,
        "expected_angle_count": expected_angles,
        "evaluated_angle_count": evaluated_angles,
        "valid_angle_count": valid_angles,
        "intra_residue_clash_count": intra_clashes,
        "adjacent_residue_clash_count": adjacent_clashes,
        "evaluated_sugar_pucker_count": evaluated_puckers,
        "valid_sugar_pucker_count": valid_puckers,
        "north_sugar_pucker_count": north_puckers,
        "south_sugar_pucker_count": south_puckers,
        "base_geometry_residue_count": len(base_geometries),
        "base_pair_count": pair_count,
        "base_paired_residue_count": len(paired),
        "base_stack_count": stack_count,
        "base_stacked_residue_count": len(stacked),
    }


def _atom_coordinates(residue: Any) -> dict[str, np.ndarray]:
    return {
        atom.get_name().strip().replace("*", "'"): np.asarray(
            atom.coord, dtype=np.float64
        )
        for atom in residue
        if _atom_element(atom) != "H"
    }


def _ideal_bond_length(first: str, second: str) -> float:
    names = {first, second}
    if names == {"P", "OP1"} or names == {"P", "OP2"}:
        return 1.49
    if "P" in names:
        return 1.60
    elements = sorted((first[0], second[0]))
    if elements == ["C", "O"]:
        if names & {"O2", "O4", "O6"} and not names & {"O2'", "O3'", "O4'", "O5'"}:
            return 1.24
        return 1.43
    if elements == ["C", "N"]:
        return 1.38 if not names & {"C1'"} else 1.47
    if elements == ["C", "C"]:
        return 1.40 if not any("'" in name for name in names) else 1.52
    return 1.45


def _ideal_angle(center: str) -> float:
    if center == "P":
        return 109.5
    if "'" not in center and center[0] in {"C", "N"}:
        return 120.0
    return 109.5


def _intra_residue_clashes(
    atom_coords: Mapping[str, np.ndarray],
    bonds: tuple[tuple[str, str], ...],
    *,
    overlap_tolerance: float,
) -> int:
    adjacency: dict[str, set[str]] = {}
    for first, second in bonds:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    excluded: set[frozenset[str]] = set()
    for start in adjacency:
        frontier = {start}
        visited = {start}
        for _ in range(3):
            frontier = {
                neighbour
                for atom_name in frontier
                for neighbour in adjacency.get(atom_name, set())
                if neighbour not in visited
            }
            for neighbour in frontier:
                excluded.add(frozenset((start, neighbour)))
            visited.update(frontier)
    clashes = 0
    topology_atom_names = sorted(set(atom_coords) & set(adjacency))
    for first, second in itertools.combinations(topology_atom_names, 2):
        if frozenset((first, second)) in excluded:
            continue
        radii = _VDW_RADII.get(first[0], 1.7) + _VDW_RADII.get(second[0], 1.7)
        distance = float(np.linalg.norm(atom_coords[first] - atom_coords[second]))
        clashes += int(distance < radii - overlap_tolerance)
    return clashes


def _adjacent_residue_clashes(
    previous: Mapping[str, np.ndarray],
    following: Mapping[str, np.ndarray],
    *,
    overlap_tolerance: float,
) -> int:
    # Exclude the O3'-P bond and its two 1-3 pairs. Every other cross-residue
    # overlap is nonbonded local sterics and should not disappear merely
    # because the residues are adjacent in sequence.
    excluded = {
        ("O3'", "P"),
        ("C3'", "P"),
        ("O3'", "O5'"),
    }
    clashes = 0
    for first_name, first_coord in previous.items():
        for second_name, second_coord in following.items():
            if (first_name, second_name) in excluded:
                continue
            radii = _VDW_RADII.get(first_name[0], 1.7) + _VDW_RADII.get(
                second_name[0], 1.7
            )
            distance = float(np.linalg.norm(first_coord - second_coord))
            clashes += int(distance < radii - overlap_tolerance)
    return clashes


def _pseudorotation_from_torsions(torsions: list[float]) -> tuple[float, float]:
    """Fit the five ring torsions to the Altona-Sundaralingam cosine model."""
    theta = 4.0 * math.pi * np.arange(5) / 5.0
    design = np.column_stack((np.cos(theta), np.sin(theta)))
    cosine_coefficient, sine_coefficient = np.linalg.lstsq(
        design, np.asarray(torsions), rcond=None
    )[0]
    amplitude = float(math.hypot(cosine_coefficient, sine_coefficient))
    phase = math.degrees(math.atan2(-sine_coefficient, cosine_coefficient)) % 360.0
    return phase, amplitude


def _angle(first: np.ndarray, center: np.ndarray, second: np.ndarray) -> float:
    left = first - center
    right = second - center
    cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _dihedral(
    first: np.ndarray, second: np.ndarray, third: np.ndarray, fourth: np.ndarray
) -> float:
    first_bond = second - first
    middle_bond = third - second
    last_bond = fourth - third
    middle_unit = middle_bond / np.linalg.norm(middle_bond)
    first_normal = first_bond - np.dot(first_bond, middle_unit) * middle_unit
    last_normal = last_bond - np.dot(last_bond, middle_unit) * middle_unit
    x = np.dot(first_normal, last_normal)
    y = np.dot(np.cross(middle_unit, first_normal), last_normal)
    return math.degrees(math.atan2(float(y), float(x)))


def _minimum_distance(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    if not first or not second:
        return float("inf")
    return min(
        float(np.linalg.norm(left - right)) for left in first for right in second
    )


def _atom_element(atom: Any) -> str:
    element = str(getattr(atom, "element", "")).strip().upper()
    if element:
        return element
    return next(
        (character for character in atom.get_name().upper() if character.isalpha()),
        "C",
    )


def _ratio_sum(rows: list[dict[str, Any]], numerator: str, denominator: str) -> float:
    numerator_sum = sum(float(row[numerator]) for row in rows)
    denominator_sum = sum(float(row[denominator]) for row in rows)
    return numerator_sum / denominator_sum if denominator_sum else float("nan")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
