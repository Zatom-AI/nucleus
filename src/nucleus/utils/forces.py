"""Utilities for force-vector statistics."""

import math
from collections.abc import Mapping
from typing import Any

import torch

MEAN_FORCE_NORM_CONDITIONING = "mean_force_norm"
RMS_AND_P95_CONDITIONING = "rms_and_p95"
FORCE_CONDITIONING_METHODS = {
    MEAN_FORCE_NORM_CONDITIONING,
    RMS_AND_P95_CONDITIONING,
}
FORCE_LOG_EPSILON = 1e-8


def mean_force_norm_target_from_extra_info(
    extra_info: Mapping[str, Any] | None,
) -> float | None:
    """Return an enabled scalar mean-force-norm target from inference metadata.

    ``SparseCollate`` stores non-tensor metadata as one-element lists during
    validation, while direct metric calls use scalars. Accept both layouts but
    reject a multi-example target: UMA's generation metric receives one shared
    conditioning target per generated structure stack.
    """

    def _scalar(value: Any, *, name: str) -> Any:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain exactly one value.")
            return value.detach().cpu().item()
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must contain exactly one value.")
            return _scalar(value[0], name=name)
        if hasattr(value, "shape") and hasattr(value, "size"):
            if int(value.size) != 1:
                raise ValueError(f"{name} must contain exactly one value.")
            return value.item()
        return value

    if not isinstance(extra_info, Mapping):
        return None
    metadata = extra_info
    nested_specification = extra_info.get("specification")
    if (
        "has_mean_force_norm" not in metadata
        and "mean_force_norm" not in metadata
        and isinstance(nested_specification, Mapping)
    ):
        metadata = nested_specification
    nested_extra = metadata.get("extra")
    if (
        "has_mean_force_norm" not in metadata
        and "mean_force_norm" not in metadata
        and isinstance(nested_extra, Mapping)
    ):
        metadata = nested_extra
    enabled = _scalar(
        metadata.get("has_mean_force_norm", False),
        name="has_mean_force_norm",
    )
    if not bool(enabled):
        return None
    if metadata.get("mean_force_norm") is None:
        raise ValueError(
            "has_mean_force_norm=True requires mean_force_norm in extra_info."
        )
    value = _scalar(metadata["mean_force_norm"], name="mean_force_norm")
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mean_force_norm must be numeric; got {value!r}.") from exc
    if not math.isfinite(target) or target < 0:
        raise ValueError(
            "mean_force_norm must be finite and non-negative; got " f"{target!r}."
        )
    return target


def force_conditioning_method_from_features(token_1d_features: Any) -> str:
    """Infer the configured force method from the model's token features."""
    features = token_1d_features or {}
    if not hasattr(features, "get"):
        return MEAN_FORCE_NORM_CONDITIONING
    has_legacy = any(
        features.get(key) is not None
        for key in ("mean_force_norm", "has_mean_force_norm")
    )
    has_rms_p95 = any(
        features.get(key) is not None for key in ("force_rms_p95", "has_force_rms_p95")
    )
    if has_legacy and has_rms_p95:
        raise ValueError(
            "Configure either mean_force_norm or force_rms_p95 token features, "
            "not both."
        )
    if has_rms_p95:
        return RMS_AND_P95_CONDITIONING
    return MEAN_FORCE_NORM_CONDITIONING


def _validated_forces(force_array: Any, *, row_id: Any) -> torch.Tensor:
    """Return finite float64 forces after validating the atom-vector layout."""
    forces = torch.as_tensor(force_array, dtype=torch.float64)
    if forces.ndim != 2 or forces.shape[-1] != 3:
        raise ValueError(
            f"Example {row_id} has invalid forces shape {tuple(forces.shape)}; "
            "expected [N_atoms, 3]."
        )
    if forces.shape[0] == 0:
        raise ValueError(f"Example {row_id} contains zero atoms.")
    if not torch.isfinite(forces).all():
        raise ValueError(f"Example {row_id} contains non-finite forces.")
    return forces


def mean_force_norm(force_array: Any, *, row_id: Any = "unknown") -> float:
    """Compute mean per-atom force norm from an ``[N_atoms, 3]`` array."""
    forces = _validated_forces(force_array, row_id=row_id)

    force_norms = torch.linalg.norm(forces, dim=-1)
    return float(force_norms.mean().item())


def force_rms_and_p95(
    force_array: Any,
    *,
    row_id: Any = "unknown",
) -> tuple[float, float]:
    """Compute RMS and 95th-percentile per-atom force magnitudes.

    Force values are expected in eV/Angstrom. The percentile uses PyTorch's
    default linear interpolation, matching ``torch.quantile(..., 0.95)``.

    Args:
        force_array: Force vectors with shape ``[N_atoms, 3]``.
        row_id: Identifier included in validation errors.

    Returns:
        ``(F_RMS, F_95)`` in eV/Angstrom.
    """
    forces = _validated_forces(force_array, row_id=row_id)
    squared_magnitudes = torch.sum(forces.square(), dim=-1)
    force_rms = torch.sqrt(squared_magnitudes.mean())
    force_p95 = torch.quantile(torch.sqrt(squared_magnitudes), 0.95)
    return float(force_rms.item()), float(force_p95.item())


def log_force_rms_and_p95(
    force_rms: Any,
    force_p95: Any,
    *,
    epsilon: float = FORCE_LOG_EPSILON,
) -> tuple[float, float]:
    """Apply the dimensionless ``log(epsilon + F / (1 eV/Angstrom))`` map."""

    def _validated_value(value: Any, *, name: str) -> float:
        if hasattr(value, "item"):
            value = value.item()
        try:
            raw_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric; got {value!r}.") from exc
        if not math.isfinite(raw_value):
            raise ValueError(f"{name} must be finite; got {value!r}.")
        if raw_value < 0:
            raise ValueError(f"{name} must be non-negative; got {raw_value!r}.")
        return raw_value

    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive; got {epsilon}.")
    force_rms = _validated_value(force_rms, name="force_rms")
    force_p95 = _validated_value(force_p95, name="force_p95")
    return math.log(epsilon + force_rms), math.log(epsilon + force_p95)
