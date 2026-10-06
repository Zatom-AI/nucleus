"""Checks for leakage exclusions, intervention location, and probe metrics."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_protein_embeddings")


@pytest.fixture
def analysis(probe):
    return importlib.import_module("analyze_protein_embeddings")


def test_holdout_excludes_parent_pdb_and_training_domain(probe):
    train = [dict(domain_id="train", pdb_id="1ABC", length="64")]
    rows = [
        dict(domain_id="other_domain", pdb_id="1abc", length="70"),
        dict(domain_id="train", pdb_id="2abc", length="70"),
        dict(domain_id="short", pdb_id="3abc", length="49"),
        dict(domain_id="heldout", pdb_id="4abc", length="128"),
    ]
    assert probe.eligible_rows(rows, train) == [(3, rows[3])]


def test_token_intervention_is_local_and_norm_controlled(probe):
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    original = values.clone()
    direction = torch.tensor([0.0, -3.0])
    changed = probe.perturb_token(values, 1, direction, 0.5)
    torch.testing.assert_close(changed[:, [0, 2]], values[:, [0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(changed[:, 1], torch.tensor([[3.0, 1.5]]))
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    torch.testing.assert_close(
        probe.perturb_token(values, 1, direction, 0), values, rtol=0, atol=0
    )


def test_response_excludes_virtual_atoms_and_sums_xyz(probe):
    reference = torch.zeros(1, 4, 3)
    changed = torch.tensor(
        [[[1.0, 2.0, 2.0], [99.0, 99.0, 99.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]]
    )
    result = probe.response_by_residue(
        changed,
        reference,
        torch.tensor([0, 0, 1, 1]),
        2,
        torch.tensor([True, False, True, True]),
    )
    np.testing.assert_array_equal(result, [9.0, 2.0])


def test_deterministic_pool_matches_mean_and_rejects_wrong_mapping(probe):
    from types import SimpleNamespace

    module = SimpleNamespace(linear=torch.nn.Identity())
    coordinates = torch.linspace(-1.0, 1.0, 84).reshape(1, 28, 3)
    mapping = torch.arange(2).repeat_interleave(14)
    expected = torch.zeros(1, 2, 3).index_reduce(
        1, mapping, coordinates, "mean", include_self=False
    )
    actual = probe.deterministic_atom14_pool(module, coordinates, mapping, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="contiguous"):
        probe.deterministic_atom14_pool(module, coordinates, mapping.flip(0), 2)


def test_neighbor_exclusion_and_residue_specific_chance(analysis):
    values = np.array([[0.0], [0.01], [1.0], [3.0]])
    structures = np.array([0, 0, 1, 2])
    labels = np.array(["A", "B", "A", "B"])
    scores, chance = analysis.neighbor_scores(values, labels, structures, k=1)
    np.testing.assert_array_equal(scores, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(chance, [0.5, 1 / 3, 1 / 3], rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="Insufficient"):
        analysis.neighbor_scores(values, labels, structures, k=3)


def test_responses_weight_proteins_equally(analysis):
    rows = [dict(structure=0, residue=0, magnitude=0.5, response=[9.0, 1.0])] * 3
    rows += [dict(structure=1, residue=1, magnitude=0.5, response=[3.0, 1.0])]
    result = analysis.summarize_responses(rows)["0.5"]
    assert result["target_rms"]["mean"] == 2.0
    assert result["target_fraction"]["mean"] == pytest.approx((0.9 + 0.25) / 2)


def test_sequence_loss_uses_observed_labels_and_natural_logs(probe):
    probabilities = torch.tensor([[[0.25, 0.75], [0.8, 0.2]]], dtype=torch.float64)
    loss = probe.sequence_loss_by_residue(probabilities.log(), torch.tensor([1, 0]))
    np.testing.assert_allclose(loss, -np.log([0.75, 0.8]), rtol=1e-6, atol=1e-7)


def test_loss_changes_are_paired_and_protein_weighted(analysis):
    def row(structure, residue, loss, baseline):
        return dict(
            structure=structure,
            residue=residue,
            magnitude=0.5,
            structure_loss=loss,
            baseline_structure_loss=baseline,
            sequence_loss=loss,
            baseline_sequence_loss=baseline,
        )

    # Unequal numbers of trials/residues must not increase a protein's weight.
    rows = [row(0, 0, [7.0, 5.0], [4.0, 6.0])] * 3
    rows += [row(1, 1, [8.0, 3.0, 11.0], [4.0, 2.0, 9.0])]
    result = analysis.summarize_losses(rows)["0.5"]
    for metric in ("structure_loss", "sequence_loss"):
        assert result[f"target_{metric}"]["mean"] == 2.0
        assert result[f"other_{metric}"]["mean"] == 1.0
        assert result[f"target_{metric}"]["baseline_mean"] == 3.0


def test_pairing_rejects_missing_directions_and_nonzero_controls(analysis, probe):
    import copy

    rows = [
        dict(
            structure=0,
            residue=0,
            direction=0,
            magnitude=magnitude,
            direction_sha256="same",
            structure_loss=[1.0 + magnitude, 2.0],
            baseline_structure_loss=[1.0, 2.0],
            sequence_loss=[1.0 + magnitude, 2.0],
            baseline_sequence_loss=[1.0, 2.0],
        )
        for magnitude in probe.MAGNITUDES
    ]
    analysis.validate_paired_trials(rows, rows)
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.validate_paired_trials(rows[:-1], rows)
    altered = copy.deepcopy(rows)
    altered[1]["direction_sha256"] = "different"
    with pytest.raises(ValueError, match="directions"):
        analysis.validate_paired_trials(rows, altered)
    altered = copy.deepcopy(rows)
    altered[0]["structure_loss"][0] += 0.1
    with pytest.raises(ValueError, match="Zero perturbation"):
        analysis.validate_paired_trials(rows, altered)
