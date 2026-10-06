from __future__ import annotations

import numpy as np
import pytest

from zevals.molecules.qm9 import (
    compute_qm9_generation_metrics,
    evaluate_qm9_sample,
    qm9_bond_order,
)


def _methane() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([6, 1, 1, 1, 1]),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629],
            ]
        ),
    )


def _water() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([8, 1, 1]),
        np.array([[0.0, 0.0, 0.0], [0.958, 0.0, 0.0], [-0.240, 0.927, 0.0]]),
    )


def test_qm9_bond_order_uses_edm_thresholds() -> None:
    assert qm9_bond_order("C", "C", 1.22) == 3
    assert qm9_bond_order("C", "C", 1.38) == 2
    assert qm9_bond_order("C", "C", 1.50) == 1
    assert qm9_bond_order("C", "C", 1.65) == 0
    assert qm9_bond_order("Xe", "C", 1.0) == 0


def test_qm9_stability_retains_explicit_hydrogens() -> None:
    result = evaluate_qm9_sample(_methane())

    assert result == {
        "num_atoms": 5,
        "num_stable_atoms": 5,
        "molecule_stable": True,
        "valid": True,
        "canonical_smiles": "[H]C([H])([H])[H]",
        "unsupported_atom_count": 0,
    }


def test_qm9_table_metrics_use_canonical_denominators() -> None:
    methane_smiles = evaluate_qm9_sample(_methane())["canonical_smiles"]
    metrics = compute_qm9_generation_metrics(
        [_methane(), _methane(), _water()], [methane_smiles]
    )

    assert metrics["counts"] == {
        "generated": 3,
        "atoms": 13,
        "stable_atoms": 13,
        "stable_molecules": 3,
        "valid": 3,
        "unique": 2,
        "novel": 1,
        "unsupported_atoms": 0,
        "canonical_reference": 1,
    }
    assert metrics["denominators"] == {
        "atom_stable": 13,
        "molecule_stable": 3,
        "valid": 3,
        "unique": 3,
        "novel": 2,
    }
    assert metrics["rates"] == pytest.approx(
        {
            "atom_stable": 1.0,
            "molecule_stable": 1.0,
            "valid": 1.0,
            "unique": 2 / 3,
            "novel": 1 / 2,
        }
    )


def test_qm9_out_of_vocabulary_atoms_are_invalid() -> None:
    result = evaluate_qm9_sample((np.array([54]), np.zeros((1, 3))))

    assert result["unsupported_atom_count"] == 1
    assert result["num_stable_atoms"] == 0
    assert not result["molecule_stable"]
    assert not result["valid"]


def test_qm9_edm_reference_preserves_full_inferred_graph() -> None:
    atomic_numbers, positions = _methane()
    disconnected = (
        np.concatenate([atomic_numbers, atomic_numbers]),
        np.concatenate([positions, positions + 10.0]),
    )

    generated = evaluate_qm9_sample(disconnected)
    reference = evaluate_qm9_sample(
        disconnected,
        smiles_largest_fragment=False,
    )

    assert generated["canonical_smiles"] == "[H]C([H])([H])[H]"
    assert reference["canonical_smiles"] == ("[H]C([H])([H])[H].[H]C([H])([H])[H]")
