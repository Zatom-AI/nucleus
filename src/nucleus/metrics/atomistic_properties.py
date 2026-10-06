"""Validation metrics for atomistic energy and force predictions."""

from __future__ import annotations

from typing import Any

import torch
from foundry.metrics.metric import Metric


def _first_prediction_view(
    value: Any,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the first diffusion view from a prediction tensor."""
    tensor = torch.as_tensor(value).detach().to(dtype=dtype)
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.shape}.")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one prediction view.")
    return tensor[0]


def compute_atomistic_property_metrics(
    network_output: dict[str, Any],
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Compute physical-unit errors for one clean-endpoint MLIP prediction."""
    predicted_energy = _first_prediction_view(
        network_output["energy_B"],
        name="energy_B",
        ndim=2,
        dtype=torch.float64,
    ).reshape(-1)
    target_energy = (
        torch.as_tensor(extra_info["energy_gt_B"]).detach().double().reshape(-1)
    )
    energy_valid = (
        torch.as_tensor(extra_info["energy_valid_mask_B"]).detach().bool().reshape(-1)
    )
    if predicted_energy.shape != target_energy.shape:
        raise ValueError(
            "Predicted and target energies must have identical system shapes; got "
            f"{predicted_energy.shape} and {target_energy.shape}."
        )
    energy_valid &= torch.isfinite(predicted_energy) & torch.isfinite(target_energy)
    if not torch.any(energy_valid):
        raise ValueError("MLIP validation example has no valid energy target.")

    energy_absolute_error = torch.abs(
        predicted_energy[energy_valid] - target_energy[energy_valid]
    )
    source_atom_count = (
        torch.as_tensor(extra_info["source_atom_count_B"])
        .detach()
        .double()
        .reshape(-1)[energy_valid]
        .clamp_min(1)
    )
    energy_per_atom_absolute_error = energy_absolute_error / source_atom_count

    predicted_forces = _first_prediction_view(
        network_output["forces_L"], name="forces_L", ndim=3
    )
    target_forces_all = torch.as_tensor(extra_info["force_gt_D_L"]).detach().float()
    if target_forces_all.ndim != 3 or target_forces_all.shape[0] < 1:
        raise ValueError(
            "force_gt_D_L must have shape [D, L, 3] with D >= 1; got "
            f"{target_forces_all.shape}."
        )
    target_forces = target_forces_all[0]
    if predicted_forces.shape != target_forces.shape:
        raise ValueError(
            "Predicted and target forces must have identical shapes; got "
            f"{predicted_forces.shape} and {target_forces.shape}."
        )

    force_valid = (
        torch.as_tensor(extra_info["force_valid_mask_L"]).detach().bool().reshape(-1)
    )
    if force_valid.shape != predicted_forces.shape[:1]:
        raise ValueError(
            "force_valid_mask_L must align with the atom dimension; got "
            f"{force_valid.shape} for forces {predicted_forces.shape}."
        )
    force_valid &= torch.isfinite(predicted_forces).all(dim=-1)
    force_valid &= torch.isfinite(target_forces).all(dim=-1)
    if not torch.any(force_valid):
        raise ValueError("MLIP validation example has no valid force targets.")

    predicted_forces = predicted_forces[force_valid]
    target_forces = target_forces[force_valid]
    force_error = predicted_forces - target_forces
    vector_error = torch.linalg.vector_norm(force_error, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted_forces, dim=-1)
    target_norm = torch.linalg.vector_norm(target_forces, dim=-1)
    cosine_denominator = predicted_norm * target_norm
    cosine_valid = cosine_denominator > 0
    force_cosine_similarity_sum = 0.0
    force_cosine_similarity = 0.0
    if torch.any(cosine_valid):
        cosine = (predicted_forces * target_forces).sum(dim=-1)[cosine_valid]
        cosine_similarity = cosine / cosine_denominator[cosine_valid]
        force_cosine_similarity_sum = float(cosine_similarity.sum())
        force_cosine_similarity = float(cosine_similarity.mean())

    return {
        "energy_mae_eV": float(energy_absolute_error.mean()),
        "energy_per_atom_mae_eV": float(energy_per_atom_absolute_error.mean()),
        "force_component_mae_eV_per_angstrom": float(force_error.abs().mean()),
        "force_component_rmse_eV_per_angstrom": float(
            torch.sqrt(torch.square(force_error).mean())
        ),
        "force_vector_mae_eV_per_angstrom": float(vector_error.mean()),
        "force_vector_rmse_eV_per_angstrom": float(
            torch.sqrt(torch.square(vector_error).mean())
        ),
        "force_norm_mae_eV_per_angstrom": float(
            torch.abs(predicted_norm - target_norm).mean()
        ),
        "force_cosine_similarity": force_cosine_similarity,
        # Additive sufficient statistics let the epoch logger aggregate systems
        # with different atom counts exactly while retaining the per-example
        # metrics above for inspection in validation CSVs.
        "energy_absolute_error_sum_eV": float(energy_absolute_error.sum()),
        "energy_per_atom_absolute_error_sum_eV": float(
            energy_per_atom_absolute_error.sum()
        ),
        "force_component_absolute_error_sum": float(force_error.abs().sum()),
        "force_component_squared_error_sum": float(torch.square(force_error).sum()),
        "force_vector_error_sum": float(vector_error.sum()),
        "force_vector_squared_error_sum": float(torch.square(vector_error).sum()),
        "force_norm_absolute_error_sum": float(
            torch.abs(predicted_norm - target_norm).sum()
        ),
        "force_cosine_similarity_sum": force_cosine_similarity_sum,
        "force_cosine_similarity_count": float(cosine_valid.sum()),
        "num_energy_systems": float(energy_valid.sum()),
        "num_force_components": float(force_valid.sum() * predicted_forces.shape[-1]),
        "num_force_atoms": float(force_valid.sum()),
    }


class AtomisticPropertyMetrics(Metric):
    """Foundry adapter for clean-endpoint energy and force validation metrics."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map trainer validation outputs to :meth:`compute` arguments."""
        return {
            "network_output": "network_output",
            "extra_info": "extra_info",
        }

    def compute(
        self,
        network_output: dict[str, Any],
        extra_info: dict[str, Any],
    ) -> dict[str, float]:
        """Compute MLIP errors in eV and eV/Angstrom."""
        return compute_atomistic_property_metrics(network_output, extra_info)


__all__ = ["AtomisticPropertyMetrics", "compute_atomistic_property_metrics"]
