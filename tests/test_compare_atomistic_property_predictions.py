import pytest

from scripts.compare_atomistic_property_predictions import (
    compare_prediction_metadata,
)


def test_compare_prediction_metadata_reports_system_balanced_metrics():
    metadata = {
        "atomistic_properties": {
            "energy_eV": -8.0,
            "forces_eV_per_angstrom": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        },
        "specification": {
            "extra": {
                "source_example_id": "example:1",
                "atomistic_domain": "omol25",
                "energy_gt_eV": -10.0,
                "forces_gt_eV_per_angstrom": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            }
        },
    }

    metrics = compare_prediction_metadata(metadata)

    assert metrics["energy_absolute_error_eV"] == pytest.approx(2.0)
    assert metrics["energy_per_atom_absolute_error_eV"] == pytest.approx(1.0)
    assert metrics["force_component_mae_eV_per_angstrom"] == pytest.approx(1 / 3)
    assert metrics["force_component_rmse_eV_per_angstrom"] == pytest.approx(
        (1 / 3) ** 0.5
    )
    assert metrics["force_vector_mae_eV_per_angstrom"] == pytest.approx(1.0)
    assert metrics["force_vector_rmse_eV_per_angstrom"] == pytest.approx(1.0)
    assert metrics["force_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["force_norm_mae_eV_per_angstrom"] == pytest.approx(1.0)


def test_compare_prediction_metadata_rejects_misaligned_force_arrays():
    metadata = {
        "atomistic_properties": {
            "energy_eV": 0.0,
            "forces_eV_per_angstrom": [[0.0, 0.0, 0.0]],
        },
        "specification": {
            "extra": {
                "energy_gt_eV": 0.0,
                "forces_gt_eV_per_angstrom": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            }
        },
    }

    with pytest.raises(ValueError, match="identical shape"):
        compare_prediction_metadata(metadata)


def test_compare_prediction_metadata_unwraps_inference_reference_batch():
    """Full inference JSON wraps specification extras in a singleton batch."""
    metadata = {
        "atomistic_properties": {
            "energy_eV": -8.0,
            "forces_eV_per_angstrom": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        },
        "specification": {
            "extra": {
                "source_example_id": ["example:1"],
                "atomistic_domain": ["omat24"],
                "energy_gt_eV": [-10.0],
                "forces_gt_eV_per_angstrom": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            }
        },
    }

    metrics = compare_prediction_metadata(metadata)

    assert metrics["source_example_id"] == "example:1"
    assert metrics["domain"] == "omat24"
    assert metrics["energy_absolute_error_eV"] == pytest.approx(2.0)
    assert metrics["force_vector_mae_eV_per_angstrom"] == pytest.approx(1.0)


def test_compare_prediction_metadata_rejects_multi_item_reference_batch():
    """One per-sample metadata file cannot select among multiple references."""
    metadata = {
        "atomistic_properties": {
            "energy_eV": 0.0,
            "forces_eV_per_angstrom": [[0.0, 0.0, 0.0]],
        },
        "specification": {
            "extra": {
                "energy_gt_eV": [0.0, 1.0],
                "forces_gt_eV_per_angstrom": [
                    [[0.0, 0.0, 0.0]],
                    [[1.0, 1.0, 1.0]],
                ],
            }
        },
    }

    with pytest.raises(ValueError, match="scalar or singleton-batched scalar"):
        compare_prediction_metadata(metadata)
