"""Tests for the trainer's structure-prediction metric adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

import nucleus.metrics.structure_prediction as structure_prediction


def _molecule_array(label: str):
    return SimpleNamespace(
        label=label,
        coord=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        get_annotation_categories=lambda: [],
    )


def _material_array(label: str):
    return SimpleNamespace(
        label=label,
        is_material=np.asarray([True]),
        get_annotation_categories=lambda: ["is_material"],
    )


def test_molecule_structure_prediction_forwards_uma_target_and_state(monkeypatch):
    """Molecule CSP uses the omol head and its conditioned charge and spin."""
    reference_array = _molecule_array("reference")
    predicted_arrays = [_molecule_array("candidate-1"), _molecule_array("candidate-2")]
    molecules = {id(reference_array): "reference-mol"}
    molecules.update(
        {
            id(atom_array): f"candidate-mol-{index}"
            for index, atom_array in enumerate(predicted_arrays)
        }
    )
    monkeypatch.setattr(
        structure_prediction,
        "_molecule_from_conditioned_graph",
        lambda atom_array, features: molecules[id(atom_array)],
    )
    monkeypatch.setattr(
        structure_prediction,
        "compute_molecule_structure_prediction_metrics",
        lambda *args, **kwargs: {"molecule_best_rmsd_mean_top1": 0.25},
    )
    captured = {}

    def fake_uma(samples, **kwargs):
        captured["samples"] = samples
        captured.update(kwargs)
        return {"uma_mean_force_norm_mae_eV_per_angstrom": 0.125}

    monkeypatch.setattr(
        structure_prediction,
        "compute_molecule_uma_force_metrics",
        fake_uma,
    )
    metric = structure_prediction.ZevalsStructurePredictionMetrics(
        compute_uma_forces=True,
        uma_checkpoint="checkpoint",
        uma_molecule_task="omol",
        uma_device="cuda",
        uma_batch_size=4,
    )

    result = metric.compute(
        predicted_arrays,
        [reference_array],
        network_input={"f": {}},
        extra_info={
            "sampled_condition_name": "structure_prediction",
            "example_id": "molecule-target",
            "specification": {
                "has_mean_force_norm": True,
                "mean_force_norm": 0.75,
                "extra": {
                    "total_charge": -1,
                    "spin_multiplicity": 2,
                },
            },
        },
    )

    assert result["molecule_best_rmsd_mean_top1"] == pytest.approx(0.25)
    assert result["uma_mean_force_norm_mae_eV_per_angstrom"] == pytest.approx(0.125)
    assert captured == {
        "samples": ["candidate-mol-0", "candidate-mol-1"],
        "total_samples": 2,
        "checkpoint": "checkpoint",
        "task": "omol",
        "device": "cuda",
        "batch_size": 4,
        "target_mean_force_norm": pytest.approx(0.75),
        "total_charge": -1,
        "spin_multiplicity": 2,
    }


def test_material_structure_prediction_forwards_uma_target(monkeypatch):
    """Material CSP uses the periodic head and preserves invalid candidates."""
    reference_array = _material_array("reference")
    predicted_arrays = [_material_array("candidate-1"), _material_array("candidate-2")]
    structures = {
        id(reference_array): "reference-structure",
        id(predicted_arrays[0]): "candidate-structure",
        id(predicted_arrays[1]): None,
    }
    monkeypatch.setattr(
        structure_prediction,
        "atom_array_to_structure",
        lambda atom_array: structures[id(atom_array)],
    )
    monkeypatch.setattr(
        structure_prediction,
        "compute_material_structure_prediction_metrics",
        lambda *args, **kwargs: {"material_match_rate_top1": 1.0},
    )
    captured = {}

    def fake_uma(samples, **kwargs):
        captured["samples"] = samples
        captured.update(kwargs)
        return {"uma_mean_force_norm_mae_eV_per_angstrom": 0.5}

    monkeypatch.setattr(
        structure_prediction,
        "compute_material_uma_force_metrics",
        fake_uma,
    )
    metric = structure_prediction.ZevalsStructurePredictionMetrics(
        compute_uma_forces=True,
        uma_checkpoint="checkpoint",
        uma_material_task="omat",
        uma_device="cpu",
        uma_batch_size=2,
    )

    result = metric.compute(
        predicted_arrays,
        [reference_array],
        extra_info={
            "sampled_condition_name": "structure_prediction",
            "example_id": "material-target",
            "specification": {
                "has_mean_force_norm": True,
                "mean_force_norm": 1.5,
            },
        },
    )

    assert result["material_match_rate_top1"] == pytest.approx(1.0)
    assert result["uma_mean_force_norm_mae_eV_per_angstrom"] == pytest.approx(0.5)
    assert captured == {
        "samples": ["candidate-structure", None],
        "total_samples": 2,
        "checkpoint": "checkpoint",
        "task": "omat",
        "device": "cpu",
        "batch_size": 2,
        "target_mean_force_norm": pytest.approx(1.5),
    }


def test_structure_prediction_does_not_call_uma_when_disabled(monkeypatch):
    """UMA remains opt-in and adds no keys when disabled."""
    reference_array = _material_array("reference")
    predicted_array = _material_array("candidate")
    monkeypatch.setattr(
        structure_prediction,
        "atom_array_to_structure",
        lambda atom_array: atom_array.label,
    )
    monkeypatch.setattr(
        structure_prediction,
        "compute_material_structure_prediction_metrics",
        lambda *args, **kwargs: {"material_match_rate_top1": 1.0},
    )
    monkeypatch.setattr(
        structure_prediction,
        "compute_material_uma_force_metrics",
        lambda *args, **kwargs: pytest.fail("UMA must not run while disabled"),
    )

    result = structure_prediction.ZevalsStructurePredictionMetrics().compute(
        [predicted_array],
        [reference_array],
        extra_info={
            "sampled_condition_name": "structure_prediction",
            "has_mean_force_norm": True,
            "mean_force_norm": 1.5,
        },
    )

    assert result == {"material_match_rate_top1": 1.0}


def test_structure_prediction_rejects_nonpositive_uma_batch_size():
    with pytest.raises(ValueError, match="uma_batch_size must be positive"):
        structure_prediction.ZevalsStructurePredictionMetrics(uma_batch_size=0)
