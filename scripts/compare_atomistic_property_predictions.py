#!/usr/bin/env python3
"""Compare Zatom-2 inference metadata against embedded atomistic references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_scalar(value: Any, *, name: str) -> float:
    """Convert an unbatched or singleton-batched numeric value to a scalar."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if array.shape == (1,):
        array = array[0]
    if array.shape != ():
        raise ValueError(
            f"{name} must be a scalar or singleton-batched scalar; got shape "
            f"{array.shape}."
        )
    return float(array)


def _as_force_array(value: Any, *, name: str) -> np.ndarray:
    """Convert an unbatched or singleton-batched force array to ``[N, 3]``."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric force array.") from error
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(
                f"{name} has a non-singleton batch dimension {array.shape[0]}; "
                "score each prediction separately."
            )
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape [n_atoms, 3] or [1, n_atoms, 3]; got "
            f"{array.shape}."
        )
    return array


def _unwrap_optional_metadata(value: Any) -> Any:
    """Remove the singleton batch wrapper used by full inference metadata."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def compare_prediction_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Compute energy and force errors from one Zatom-2 output metadata JSON."""
    try:
        prediction = metadata["atomistic_properties"]
        reference = metadata["specification"]["extra"]
        predicted_energy_value = prediction["energy_eV"]
        reference_energy_value = reference["energy_gt_eV"]
        predicted_force_value = prediction["forces_eV_per_angstrom"]
        reference_force_value = reference["forces_gt_eV_per_angstrom"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Prediction metadata must contain atomistic_properties and embedded "
            "specification.extra energy/force references."
        ) from error

    predicted_energy = _as_scalar(
        predicted_energy_value, name="atomistic_properties.energy_eV"
    )
    reference_energy = _as_scalar(
        reference_energy_value, name="specification.extra.energy_gt_eV"
    )
    predicted_forces = _as_force_array(
        predicted_force_value,
        name="atomistic_properties.forces_eV_per_angstrom",
    )
    reference_forces = _as_force_array(
        reference_force_value,
        name="specification.extra.forces_gt_eV_per_angstrom",
    )

    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            "Predicted and reference force arrays must have identical shape; got "
            f"{predicted_forces.shape} and {reference_forces.shape}."
        )
    if not (
        np.isfinite(predicted_energy)
        and np.isfinite(reference_energy)
        and np.isfinite(predicted_forces).all()
        and np.isfinite(reference_forces).all()
    ):
        raise ValueError("Energy and force values must all be finite.")

    force_error = predicted_forces - reference_forces
    vector_error = np.linalg.norm(force_error, axis=-1)
    predicted_norm = np.linalg.norm(predicted_forces, axis=-1)
    reference_norm = np.linalg.norm(reference_forces, axis=-1)
    cosine_denominator = predicted_norm * reference_norm
    cosine_valid = cosine_denominator > 0
    mean_cosine = None
    if np.any(cosine_valid):
        cosine = np.sum(predicted_forces * reference_forces, axis=-1)[cosine_valid]
        cosine /= cosine_denominator[cosine_valid]
        mean_cosine = float(np.mean(cosine))

    n_atoms = predicted_forces.shape[0]
    energy_error = predicted_energy - reference_energy
    return {
        "source_example_id": _unwrap_optional_metadata(
            reference.get("source_example_id")
        ),
        "domain": _unwrap_optional_metadata(reference.get("atomistic_domain")),
        "n_atoms": n_atoms,
        "predicted_energy_eV": predicted_energy,
        "reference_energy_eV": reference_energy,
        "energy_error_eV": energy_error,
        "energy_absolute_error_eV": abs(energy_error),
        "energy_per_atom_absolute_error_eV": abs(energy_error) / max(n_atoms, 1),
        "force_component_mae_eV_per_angstrom": float(np.mean(np.abs(force_error))),
        "force_component_rmse_eV_per_angstrom": float(
            np.sqrt(np.mean(np.square(force_error)))
        ),
        "force_vector_mae_eV_per_angstrom": float(np.mean(vector_error)),
        "force_vector_rmse_eV_per_angstrom": float(
            np.sqrt(np.mean(np.square(vector_error)))
        ),
        "force_cosine_similarity": mean_cosine,
        "force_norm_mae_eV_per_angstrom": float(
            np.mean(np.abs(predicted_norm - reference_norm))
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse output metadata paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction_json",
        type=Path,
        nargs="+",
        help="Zatom-2 output metadata JSON file(s) produced by `zatom2 design`.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Print comparison metrics as JSON keyed by prediction path."""
    args = parse_args(argv)
    comparisons = {}
    for path in args.prediction_json:
        with path.open() as handle:
            metadata = json.load(handle)
        comparisons[str(path)] = compare_prediction_metadata(metadata)
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
