"""Tests for single-reference molecular structure prediction."""

from __future__ import annotations

import pytest


def test_molecule_best_rmsd_uses_explicit_top_k(monkeypatch: pytest.MonkeyPatch):
    from rdkit import Chem

    from zevals.molecules import structure_prediction

    molecule = Chem.MolFromSmiles("CC")
    distances = iter([0.8, 0.2, 0.1])
    monkeypatch.setattr(
        structure_prediction,
        "_heavy_atom_best_rmsd",
        lambda *_args: next(distances),
    )

    result = structure_prediction.compute_molecule_structure_prediction_metrics(
        {"target": [molecule, molecule, molecule]},
        {"target": [molecule]},
        top_k=(1, 2, 3),
    )

    assert result == pytest.approx(
        {
            "molecule_best_rmsd_mean_top1": 0.8,
            "molecule_best_rmsd_median_top1": 0.8,
            "molecule_best_rmsd_mean_top2": 0.2,
            "molecule_best_rmsd_median_top2": 0.2,
            "molecule_best_rmsd_mean_top3": 0.1,
            "molecule_best_rmsd_median_top3": 0.1,
        }
    )


def test_molecule_best_rmsd_macro_averages_targets(
    monkeypatch: pytest.MonkeyPatch,
):
    from rdkit import Chem

    from zevals.molecules import structure_prediction

    molecule = Chem.MolFromSmiles("CC")
    distances = iter([0.1, 0.5])
    monkeypatch.setattr(
        structure_prediction,
        "_heavy_atom_best_rmsd",
        lambda *_args: next(distances),
    )

    result = structure_prediction.compute_molecule_structure_prediction_metrics(
        {"a": [molecule], "b": [molecule]},
        {"a": [molecule], "b": [molecule]},
    )

    assert result["molecule_best_rmsd_mean_top1"] == pytest.approx(0.3)
    assert result["molecule_best_rmsd_median_top1"] == pytest.approx(0.3)


def test_molecule_best_rmsd_requires_one_reference_and_enough_candidates():
    from rdkit import Chem

    from zevals.molecules.structure_prediction import (
        compute_molecule_structure_prediction_metrics,
    )

    molecule = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError, match="exactly one reference conformer"):
        compute_molecule_structure_prediction_metrics(
            {"target": [molecule]},
            {"target": [molecule, molecule]},
        )
    with pytest.raises(ValueError, match="top-2 requires at least 2"):
        compute_molecule_structure_prediction_metrics(
            {"target": [molecule]},
            {"target": [molecule]},
            top_k=(2,),
        )


def test_molecule_best_rmsd_rejects_mismatched_graphs():
    from rdkit import Chem

    from zevals.molecules.structure_prediction import (
        compute_molecule_structure_prediction_metrics,
    )

    with pytest.raises(ValueError, match="molecular graphs must match"):
        compute_molecule_structure_prediction_metrics(
            {"target": [Chem.MolFromSmiles("CO")]},
            {"target": [Chem.MolFromSmiles("CC")]},
        )
