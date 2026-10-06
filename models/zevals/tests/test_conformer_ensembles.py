"""Tests for GEOM-style molecular conformer-ensemble metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_conformer_metrics_follow_precision_and_recall_nearest_neighbours():
    """COV/MAT directions match the GeoMol definitions on a fixed matrix."""
    from zevals.molecules.conformer_ensembles import _summarize_distance_matrix

    # Rows are generated conformers, columns are references.
    result = _summarize_distance_matrix(
        np.asarray([[0.1, 1.0], [0.4, 2.0], [3.0, 0.2]]),
        rmsd_threshold=0.3,
    )

    assert result == pytest.approx(
        {
            "cov_recall": 1.0,
            "mat_recall": 0.15,
            "cov_precision": 2 / 3,
            "mat_precision": (0.1 + 0.4 + 0.2) / 3,
        }
    )


def test_conformer_metrics_macro_average_targets(monkeypatch: pytest.MonkeyPatch):
    """Each target contributes once, independent of its ensemble size."""
    from zevals.molecules import conformer_ensembles

    matrices = iter(
        [
            np.asarray([[0.1, 2.0], [1.0, 0.2]]),
            np.asarray([[0.7]]),
        ]
    )

    def fake_rmsd(_generated, _reference):
        return next(matrix_values)

    def matrix_rmsd(generated, reference):
        nonlocal matrix_values
        try:
            return next(matrix_values)
        except StopIteration:
            matrix_values = iter(next(matrices).ravel())
            return next(matrix_values)

    matrix_values = iter(next(matrices).ravel())
    monkeypatch.setattr(conformer_ensembles, "_heavy_atom_best_rmsd", matrix_rmsd)
    monkeypatch.setattr(
        conformer_ensembles, "_validate_target_graphs", lambda *_args, **_kwargs: None
    )
    result = conformer_ensembles.compute_conformer_ensemble_metrics(
        {"a": [object(), object()], "b": [object()]},
        {"a": [object(), object()], "b": [object()]},
        rmsd_threshold=0.5,
        generated_per_reference=None,
    )

    assert result == pytest.approx(
        {
            "conformer_cov_recall_mean": 0.5,
            "conformer_cov_recall_median": 0.5,
            "conformer_mat_recall_mean": 0.425,
            "conformer_mat_recall_median": 0.425,
            "conformer_cov_precision_mean": 0.5,
            "conformer_cov_precision_median": 0.5,
            "conformer_mat_precision_mean": 0.425,
            "conformer_mat_precision_median": 0.425,
            "conformer_candidate_count_mean": 1.5,
            "conformer_evaluated_candidate_count_mean": 1.5,
            "conformer_reference_count_mean": 1.5,
        }
    )


def test_conformer_metrics_enforce_two_candidates_per_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    """GEOM benchmark scoring uses exactly the first 2R candidates."""
    from rdkit import Chem

    from zevals.molecules import conformer_ensembles

    molecule = Chem.MolFromSmiles("CC")
    distances = iter([0.1, 0.2])
    monkeypatch.setattr(
        conformer_ensembles,
        "_heavy_atom_best_rmsd",
        lambda *_args: next(distances),
    )
    result = conformer_ensembles.compute_conformer_ensemble_metrics(
        {"target": [molecule, molecule, molecule]},
        {"target": [molecule]},
        rmsd_threshold=0.5,
    )

    assert result["conformer_candidate_count_mean"] == 3.0
    assert result["conformer_evaluated_candidate_count_mean"] == 2.0
    assert result["conformer_mat_precision_mean"] == pytest.approx(0.15)


def test_conformer_metrics_reject_undersized_or_mismatched_graphs():
    """Benchmark budgets and target graph identity are explicit invariants."""
    from rdkit import Chem

    from zevals.molecules.conformer_ensembles import (
        compute_conformer_ensemble_metrics,
    )

    ethane = Chem.MolFromSmiles("CC")
    methanol = Chem.MolFromSmiles("CO")
    with pytest.raises(ValueError, match="requires 2 generated conformers"):
        compute_conformer_ensemble_metrics(
            {"target": [ethane]},
            {"target": [ethane]},
            rmsd_threshold=0.5,
        )
    with pytest.raises(ValueError, match="molecular graphs must match"):
        compute_conformer_ensemble_metrics(
            {"target": [methanol, methanol]},
            {"target": [ethane]},
            rmsd_threshold=0.5,
        )


def test_conformer_metrics_reject_mismatched_target_ids():
    """Target grouping must be explicit; accidental cross-target scoring is invalid."""
    from zevals.molecules.conformer_ensembles import compute_conformer_ensemble_metrics

    with pytest.raises(ValueError, match="target IDs must match"):
        compute_conformer_ensemble_metrics(
            {"generated": [object()]},
            {"reference": [object()]},
            rmsd_threshold=0.5,
        )


def test_load_molecules_groups_sdf_conformers_by_target(tmp_path: Path):
    """A ground-truth directory activates explicit target-based ensemble loading."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from zevals.pipeline.loaders import MoleculeConformerEnsembles, load_molecules

    def write_sdf(path: Path) -> None:
        molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
        assert AllChem.EmbedMolecule(molecule, randomSeed=0) == 0
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()

    generated_dir = tmp_path / "generated" / "target-a"
    reference_dir = tmp_path / "reference" / "target-a"
    generated_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    write_sdf(generated_dir / "sample.sdf")
    write_sdf(reference_dir / "reference.sdf")

    samples = load_molecules(generated_dir.parent, reference_dir.parent)

    assert isinstance(samples, MoleculeConformerEnsembles)
    assert list(samples.generated_by_target) == ["target-a"]
    assert list(samples.reference_by_target) == ["target-a"]
    assert len(samples.molecules) == 1
