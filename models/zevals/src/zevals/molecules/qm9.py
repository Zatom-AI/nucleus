"""Canonical QM9 molecule-generation metrics.

This module implements the distance-based bond and valence rules used by the
EDM QM9 evaluator.  It intentionally works on atomic numbers and Cartesian
coordinates instead of inferred bonds stored in an input file, so scoring is
independent of CIF/PDB readers and their version-specific bond perception.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zevals.utils.packages import load_rdkit_chem

QM9_ATOMIC_NUMBER_TO_ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
QM9_ALLOWED_VALENCES = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}

# Bond lengths are in pm.  The strict ``<`` comparisons and margins match
# ehoogeboom/e3_diffusion_for_molecules/qm9/bond_analyze.py.
_SINGLE_BOND_LENGTHS = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
_DOUBLE_BOND_LENGTHS = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
_TRIPLE_BOND_LENGTHS = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}
_BOND_MARGINS_PM = (10, 5, 3)


def qm9_bond_order(element_a: str, element_b: str, distance_angstrom: float) -> int:
    """Infer a QM9 bond order from an element pair and distance.

    Unsupported elements are not part of the QM9 vocabulary and return zero.
    """
    if (
        element_a not in _SINGLE_BOND_LENGTHS
        or element_b not in _SINGLE_BOND_LENGTHS[element_a]
        or not np.isfinite(distance_angstrom)
    ):
        return 0

    distance_pm = 100.0 * float(distance_angstrom)
    if distance_pm >= _SINGLE_BOND_LENGTHS[element_a][element_b] + _BOND_MARGINS_PM[0]:
        return 0
    if (
        element_a in _DOUBLE_BOND_LENGTHS
        and element_b in _DOUBLE_BOND_LENGTHS[element_a]
        and distance_pm
        < _DOUBLE_BOND_LENGTHS[element_a][element_b] + _BOND_MARGINS_PM[1]
    ):
        if (
            element_a in _TRIPLE_BOND_LENGTHS
            and element_b in _TRIPLE_BOND_LENGTHS[element_a]
            and distance_pm
            < _TRIPLE_BOND_LENGTHS[element_a][element_b] + _BOND_MARGINS_PM[2]
        ):
            return 3
        return 2
    return 1


def infer_qm9_bond_orders(
    atomic_numbers: Sequence[int] | np.ndarray,
    positions: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the lower-triangular EDM bond-order matrix for one molecule."""
    atomic_numbers_array = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    positions_array = np.asarray(positions, dtype=np.float64)
    if positions_array.shape != (len(atomic_numbers_array), 3):
        raise ValueError(
            "QM9 coordinates must have shape "
            f"({len(atomic_numbers_array)}, 3), got {positions_array.shape}."
        )
    if not np.isfinite(positions_array).all():
        raise ValueError("QM9 coordinates must all be finite.")

    bond_orders = np.zeros(
        (len(atomic_numbers_array), len(atomic_numbers_array)), dtype=np.int8
    )
    elements = [
        QM9_ATOMIC_NUMBER_TO_ELEMENT.get(int(atomic_number))
        for atomic_number in atomic_numbers_array
    ]
    for atom_index in range(len(atomic_numbers_array)):
        for other_index in range(atom_index):
            element_a = elements[atom_index]
            element_b = elements[other_index]
            if element_a is None or element_b is None:
                continue
            distance = float(
                np.linalg.norm(
                    positions_array[atom_index] - positions_array[other_index]
                )
            )
            bond_orders[atom_index, other_index] = qm9_bond_order(
                element_a, element_b, distance
            )
    return bond_orders


def _sample_arrays(sample: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract atomic numbers and positions from a tuple, mapping, or atoms object."""
    if isinstance(sample, tuple) and len(sample) == 2:
        atomic_numbers, positions = sample
    elif isinstance(sample, Mapping):
        atomic_numbers = next(
            (
                sample[key]
                for key in ("atomic_numbers", "z", "atom_types")
                if key in sample
            ),
            None,
        )
        positions = next(
            (
                sample[key]
                for key in ("positions", "pos", "coordinates")
                if key in sample
            ),
            None,
        )
    else:
        atomic_numbers = getattr(sample, "atomic_number", None)
        if atomic_numbers is None and hasattr(sample, "get_atomic_numbers"):
            atomic_numbers = sample.get_atomic_numbers()
        positions = getattr(sample, "coord", None)
        if positions is None and hasattr(sample, "get_positions"):
            positions = sample.get_positions()

    if atomic_numbers is None or positions is None:
        raise TypeError(
            "Each QM9 sample must expose atomic numbers and Cartesian positions."
        )
    return (
        np.asarray(atomic_numbers, dtype=np.int64).reshape(-1),
        np.asarray(positions, dtype=np.float64),
    )


def _canonical_smiles(
    molecule: Any,
    *,
    remove_hydrogens: bool,
    isomeric: bool,
    largest_fragment: bool,
) -> str | None:
    Chem = load_rdkit_chem()
    try:
        Chem.SanitizeMol(molecule)
        if largest_fragment:
            fragments = Chem.rdmolops.GetMolFrags(molecule, asMols=True)
            molecule = max(
                fragments,
                default=molecule,
                key=lambda fragment: fragment.GetNumAtoms(),
            )
        if remove_hydrogens:
            molecule = Chem.RemoveHs(molecule)
        smiles = Chem.MolToSmiles(molecule, isomericSmiles=isomeric)
        return smiles or None
    except Exception:
        return None


def _molecule_from_bond_orders(
    atomic_numbers: np.ndarray,
    bond_orders: np.ndarray,
) -> Any:
    Chem = load_rdkit_chem()
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
    }
    molecule = Chem.RWMol()
    for atomic_number in atomic_numbers:
        molecule.AddAtom(Chem.Atom(int(atomic_number)))
    for atom_index in range(len(atomic_numbers)):
        for other_index in range(atom_index):
            order = int(bond_orders[atom_index, other_index])
            if order:
                molecule.AddBond(atom_index, other_index, bond_types[order])
    return molecule


def evaluate_qm9_sample(
    sample: Any,
    *,
    smiles_remove_hydrogens: bool = False,
    isomeric_smiles: bool = True,
    smiles_largest_fragment: bool = True,
) -> dict[str, Any]:
    """Evaluate stability and canonical-SMILES validity for one QM9 sample.

    ``smiles_largest_fragment=True`` matches EDM generated-sample validity.
    EDM's training-reference builder instead canonicalizes the full inferred
    graph, so callers preparing its novelty reference should pass ``False``.
    """
    atomic_numbers, positions = _sample_arrays(sample)
    num_atoms = len(atomic_numbers)
    unsupported_count = int(
        sum(
            int(atomic_number) not in QM9_ATOMIC_NUMBER_TO_ELEMENT
            for atomic_number in atomic_numbers
        )
    )
    if num_atoms == 0 or positions.shape != (num_atoms, 3):
        return {
            "num_atoms": num_atoms,
            "num_stable_atoms": 0,
            "molecule_stable": False,
            "valid": False,
            "canonical_smiles": None,
            "unsupported_atom_count": unsupported_count,
        }

    try:
        bond_orders = infer_qm9_bond_orders(atomic_numbers, positions)
    except ValueError:
        return {
            "num_atoms": num_atoms,
            "num_stable_atoms": 0,
            "molecule_stable": False,
            "valid": False,
            "canonical_smiles": None,
            "unsupported_atom_count": unsupported_count,
        }

    valences = bond_orders.sum(axis=0, dtype=np.int64) + bond_orders.sum(
        axis=1, dtype=np.int64
    )
    stable_atoms = []
    for atomic_number, valence in zip(atomic_numbers, valences, strict=True):
        element = QM9_ATOMIC_NUMBER_TO_ELEMENT.get(int(atomic_number))
        stable_atoms.append(
            element is not None and int(valence) == QM9_ALLOWED_VALENCES[element]
        )
    num_stable_atoms = int(sum(stable_atoms))
    molecule_stable = num_atoms > 0 and num_stable_atoms == num_atoms

    canonical_smiles = None
    if unsupported_count == 0:
        molecule = _molecule_from_bond_orders(atomic_numbers, bond_orders)
        canonical_smiles = _canonical_smiles(
            molecule,
            remove_hydrogens=smiles_remove_hydrogens,
            isomeric=isomeric_smiles,
            largest_fragment=smiles_largest_fragment,
        )
    return {
        "num_atoms": num_atoms,
        "num_stable_atoms": num_stable_atoms,
        "molecule_stable": molecule_stable,
        "valid": canonical_smiles is not None,
        "canonical_smiles": canonical_smiles,
        "unsupported_atom_count": unsupported_count,
    }


def canonicalize_qm9_reference_smiles(
    smiles: Sequence[str],
    *,
    remove_hydrogens: bool = False,
    isomeric: bool = True,
) -> set[str]:
    """Canonicalize a QM9 training-reference SMILES collection with RDKit."""
    Chem = load_rdkit_chem()
    canonical = set()
    for value in smiles:
        try:
            parser_parameters = Chem.SmilesParserParams()
            parser_parameters.removeHs = remove_hydrogens
            molecule = Chem.MolFromSmiles(str(value), parser_parameters)
            if molecule is None:
                continue
            normalized = Chem.MolToSmiles(molecule, isomericSmiles=isomeric)
        except Exception:
            continue
        if normalized:
            canonical.add(normalized)
    return canonical


def _strings_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_qm9_generation_metrics(
    samples: Sequence[Any],
    reference_smiles: Sequence[str],
    *,
    smiles_remove_hydrogens: bool = False,
    isomeric_smiles: bool = True,
) -> dict[str, Any]:
    """Compute the five Table-2 QM9 metrics with explicit denominators.

    Validity is over all generated samples, uniqueness is over valid samples,
    and novelty is over unique valid samples, matching the canonical EDM
    evaluator.  Reference SMILES are canonicalized with the same hydrogen and
    stereochemistry settings as generated SMILES.
    """
    load_rdkit_chem()
    from rdkit import rdBase

    with rdBase.BlockLogs():
        sample_results = [
            evaluate_qm9_sample(
                sample,
                smiles_remove_hydrogens=smiles_remove_hydrogens,
                isomeric_smiles=isomeric_smiles,
            )
            for sample in samples
        ]
    num_generated = len(sample_results)
    num_atoms = sum(int(result["num_atoms"]) for result in sample_results)
    num_stable_atoms = sum(int(result["num_stable_atoms"]) for result in sample_results)
    num_stable_molecules = sum(
        bool(result["molecule_stable"]) for result in sample_results
    )
    valid_smiles = [
        str(result["canonical_smiles"]) for result in sample_results if result["valid"]
    ]
    unique_smiles = set(valid_smiles)
    with rdBase.BlockLogs():
        reference_set = canonicalize_qm9_reference_smiles(
            reference_smiles,
            remove_hydrogens=smiles_remove_hydrogens,
            isomeric=isomeric_smiles,
        )
    novel_smiles = unique_smiles.difference(reference_set)

    counts = {
        "generated": num_generated,
        "atoms": num_atoms,
        "stable_atoms": num_stable_atoms,
        "stable_molecules": int(num_stable_molecules),
        "valid": len(valid_smiles),
        "unique": len(unique_smiles),
        "novel": len(novel_smiles),
        "unsupported_atoms": sum(
            int(result["unsupported_atom_count"]) for result in sample_results
        ),
        "canonical_reference": len(reference_set),
    }
    denominators = {
        "atom_stable": num_atoms,
        "molecule_stable": num_generated,
        "valid": num_generated,
        "unique": len(valid_smiles),
        "novel": len(unique_smiles),
    }
    numerators = {
        "atom_stable": num_stable_atoms,
        "molecule_stable": int(num_stable_molecules),
        "valid": len(valid_smiles),
        "unique": len(unique_smiles),
        "novel": len(novel_smiles),
    }
    rates = {
        key: numerators[key] / denominator if denominator else 0.0
        for key, denominator in denominators.items()
    }
    return {
        "protocol": {
            "name": "edm-qm9-with-h",
            "bond_length_unit": "angstrom",
            "smiles_remove_hydrogens": smiles_remove_hydrogens,
            "isomeric_smiles": isomeric_smiles,
            "validity_fragment_policy": "largest_fragment",
            "novelty_reference": "qm9_training_split",
            "novelty_reference_fragment_policy": "full_inferred_graph",
        },
        "counts": counts,
        "denominators": denominators,
        "zero_denominator_metrics": [
            key for key, denominator in denominators.items() if denominator == 0
        ],
        "rates": rates,
        "percent": {key: 100.0 * value for key, value in rates.items()},
        "digests": {
            "valid_smiles_sha256": _strings_sha256(valid_smiles),
            "unique_smiles_sha256": _strings_sha256(sorted(unique_smiles)),
            "reference_smiles_sha256": _strings_sha256(sorted(reference_set)),
        },
    }


__all__ = [
    "QM9_ALLOWED_VALENCES",
    "QM9_ATOMIC_NUMBER_TO_ELEMENT",
    "canonicalize_qm9_reference_smiles",
    "compute_qm9_generation_metrics",
    "evaluate_qm9_sample",
    "infer_qm9_bond_orders",
    "qm9_bond_order",
]
