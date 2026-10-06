"""Tests for atomistic property validation metrics."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import nucleus.callbacks.metrics_logging as metrics_logging
from nucleus.metrics.atomistic_properties import compute_atomistic_property_metrics


def test_atomistic_property_metrics_report_physical_unit_errors():
    """Test that the atomistic property metrics report physical unit errors correctly."""
    network_output = {
        "energy_B": torch.tensor([[-8.0]]),
        "forces_L": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [99.0, 99.0, 99.0]]]
        ),
    }
    extra_info = {
        "energy_gt_B": torch.tensor([-10.0]),
        "energy_valid_mask_B": torch.tensor([True]),
        "source_atom_count_B": torch.tensor([2]),
        "force_gt_D_L": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        "force_valid_mask_L": torch.tensor([True, True, False]),
    }

    metrics = compute_atomistic_property_metrics(network_output, extra_info)

    assert metrics["energy_mae_eV"] == pytest.approx(2.0)
    assert metrics["energy_per_atom_mae_eV"] == pytest.approx(1.0)
    assert metrics["force_component_mae_eV_per_angstrom"] == pytest.approx(1 / 3)
    assert metrics["force_component_rmse_eV_per_angstrom"] == pytest.approx(
        (1 / 3) ** 0.5
    )
    assert metrics["force_vector_mae_eV_per_angstrom"] == pytest.approx(1.0)
    assert metrics["force_vector_rmse_eV_per_angstrom"] == pytest.approx(1.0)
    assert metrics["force_norm_mae_eV_per_angstrom"] == pytest.approx(1.0)
    assert metrics["force_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["energy_absolute_error_sum_eV"] == pytest.approx(2.0)
    assert metrics["energy_per_atom_absolute_error_sum_eV"] == pytest.approx(1.0)
    assert metrics["force_component_absolute_error_sum"] == pytest.approx(2.0)
    assert metrics["force_component_squared_error_sum"] == pytest.approx(2.0)
    assert metrics["force_vector_error_sum"] == pytest.approx(2.0)
    assert metrics["force_vector_squared_error_sum"] == pytest.approx(2.0)
    assert metrics["force_norm_absolute_error_sum"] == pytest.approx(2.0)
    assert metrics["force_cosine_similarity_sum"] == pytest.approx(1.0)
    assert metrics["force_cosine_similarity_count"] == 1.0
    assert metrics["num_energy_systems"] == 1.0
    assert metrics["num_force_components"] == 6.0
    assert metrics["num_force_atoms"] == 2.0


def test_atomistic_property_metrics_subtract_large_energies_in_float64():
    """Metric subtraction retains errors smaller than a float32 total-energy ULP."""
    baseline = -1_000_000_000.0
    metrics = compute_atomistic_property_metrics(
        {
            "energy_B": torch.tensor([[baseline + 0.25]], dtype=torch.float64),
            "forces_L": torch.zeros(1, 1, 3),
        },
        {
            "energy_gt_B": torch.tensor([baseline], dtype=torch.float64),
            "energy_valid_mask_B": torch.tensor([True]),
            "source_atom_count_B": torch.tensor([1]),
            "force_gt_D_L": torch.zeros(1, 1, 3),
            "force_valid_mask_L": torch.tensor([True]),
        },
    )

    assert metrics["energy_mae_eV"] == 0.25
    assert metrics["energy_per_atom_mae_eV"] == 0.25


def test_atomistic_property_epoch_metrics_are_globally_weighted(tmp_path, monkeypatch):
    """Test epoch force metrics weight atoms/components instead of structures."""

    def compute_metrics(
        predicted_forces: torch.Tensor,
        target_forces: torch.Tensor,
    ) -> dict[str, float]:
        num_atoms = predicted_forces.shape[0]
        return compute_atomistic_property_metrics(
            {
                "energy_B": torch.tensor([[2.0]]),
                "forces_L": predicted_forces.unsqueeze(0),
            },
            {
                "energy_gt_B": torch.tensor([0.0]),
                "energy_valid_mask_B": torch.tensor([True]),
                "source_atom_count_B": torch.tensor([num_atoms]),
                "force_gt_D_L": target_forces.unsqueeze(0),
                "force_valid_mask_L": torch.ones(num_atoms, dtype=torch.bool),
            },
        )

    first = compute_metrics(
        torch.tensor([[4.0, 3.0, 3.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    second_targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    second = compute_metrics(second_targets + 1.0, second_targets)
    prefix = "atomistic_property_metrics."
    dataframe = pd.DataFrame(
        [
            {f"{prefix}{key}": value for key, value in metrics.items()}
            for metrics in (first, second)
        ]
    )
    component_mae_column = f"{prefix}force_component_mae_eV_per_angstrom"
    naive_structure_mean = dataframe[component_mae_column].mean()
    dataframe.drop(
        columns=[
            component_mae_column,
            f"{prefix}force_component_rmse_eV_per_angstrom",
        ],
        inplace=True,
    )

    dataframe["example_id"] = ["one_atom", "three_atoms"]
    dataframe["epoch"] = 4
    dataframe["dataset"] = "mlip_validation"
    validation_results_path = tmp_path / "validation_output_all_epochs.csv"
    dataframe.to_csv(validation_results_path, index=False)

    logged_metrics = {}
    fabric = SimpleNamespace(
        is_global_zero=True,
        log_dict=lambda metrics, step: logged_metrics.update(metrics),
    )
    trainer = SimpleNamespace(
        fabric=fabric,
        validation_results_path=validation_results_path,
        state={"current_epoch": 4, "global_step": 12},
    )
    monkeypatch.setattr(metrics_logging, "print_df_as_table", lambda *args: None)

    metrics_logging.LogDesignValidationMetricsCallback().on_validation_epoch_end(
        trainer
    )
    log_prefix = "val/mlip_validation/"
    aggregated = pd.Series(
        {
            key.removeprefix(log_prefix): value
            for key, value in logged_metrics.items()
            if key.startswith(log_prefix)
        }
    )
    force_error = torch.cat((torch.full((1, 3), 3.0), torch.ones((3, 3))), dim=0)
    vector_error = torch.linalg.vector_norm(force_error, dim=-1)
    predictions = torch.cat(
        (
            torch.tensor([[4.0, 3.0, 3.0]]),
            second_targets + 1.0,
        )
    )
    targets = torch.cat((torch.tensor([[1.0, 0.0, 0.0]]), second_targets))
    cosine_valid = torch.linalg.vector_norm(targets, dim=-1) > 0
    cosine = torch.nn.functional.cosine_similarity(
        predictions[cosine_valid], targets[cosine_valid]
    )

    assert aggregated[f"{prefix}force_component_mae_eV_per_angstrom"] == pytest.approx(
        force_error.abs().mean().item()
    )
    assert aggregated[f"{prefix}force_component_rmse_eV_per_angstrom"] == pytest.approx(
        torch.sqrt(torch.square(force_error).mean()).item()
    )
    assert aggregated[f"{prefix}force_vector_mae_eV_per_angstrom"] == pytest.approx(
        vector_error.mean().item()
    )
    assert aggregated[f"{prefix}force_vector_rmse_eV_per_angstrom"] == pytest.approx(
        torch.sqrt(torch.square(vector_error).mean()).item()
    )
    assert aggregated[f"{prefix}force_norm_mae_eV_per_angstrom"] == pytest.approx(
        torch.abs(
            torch.linalg.vector_norm(predictions, dim=-1)
            - torch.linalg.vector_norm(targets, dim=-1)
        )
        .mean()
        .item()
    )
    assert aggregated[f"{prefix}force_cosine_similarity"] == pytest.approx(
        cosine.mean().item()
    )
    assert aggregated[f"{prefix}num_force_atoms"] == 4.0
    assert aggregated[f"{prefix}num_force_components"] == 12.0
    assert aggregated[f"{prefix}force_cosine_similarity_count"] == 3.0
    assert f"{prefix}force_component_absolute_error_sum" not in aggregated

    assert aggregated[f"{prefix}force_component_mae_eV_per_angstrom"] != pytest.approx(
        naive_structure_mean
    )


def test_atomistic_property_aggregation_rejects_incomplete_sufficient_statistics():
    """A positive count cannot be reduced against a missing numerator."""
    prefix = "atomistic_property_metrics."
    dataframe = pd.DataFrame(
        {
            f"{prefix}num_force_components": [3.0],
            f"{prefix}force_component_absolute_error_sum": [1.0],
        }
    )

    with pytest.raises(ValueError, match="force_component_squared_error_sum"):
        metrics_logging._aggregate_atomistic_property_metric_columns(
            dataframe,
            dataframe.mean(numeric_only=True),
        )
