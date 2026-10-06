"""Shared UMA force-metric adapters for validation-time metrics."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)


def electronic_conditioning_from_extra_info(
    extra_info: Mapping[str, Any] | None,
) -> tuple[int | None, int | None]:
    """Extract one inference system's valid ground-truth charge and spin.

    Metric metadata can contain runtime tensors at the top level, while the
    original inference request is preserved below ``specification`` and/or
    ``extra``. Search those containers independently for charge and spin so a
    top-level value for one property does not hide a nested value for the other.
    """

    def scalar(value: Any, *, name: str) -> Any:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain exactly one value.")
            return value.detach().cpu().item()
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must contain exactly one value.")
            return scalar(value[0], name=name)
        if hasattr(value, "shape") and hasattr(value, "size"):
            if int(value.size) != 1:
                raise ValueError(f"{name} must contain exactly one value.")
            return value.item()
        return value

    def integer_metadata(
        metadata: Mapping[str, Any],
        *,
        value_keys: tuple[str, ...],
        validity_keys: tuple[str, ...],
        name: str,
        minimum: int | None = None,
    ) -> int | None:
        for validity_key in validity_keys:
            if validity_key in metadata and not bool(
                scalar(metadata[validity_key], name=validity_key)
            ):
                return None
        value = next(
            (metadata[key] for key in value_keys if metadata.get(key) is not None),
            None,
        )
        if value is None:
            return None
        value = scalar(value, name=name)
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be an integer scalar, got {value!r}."
            ) from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be an integer scalar, got {value!r}.")
        integer = int(numeric)
        if minimum is not None and integer < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {integer}.")
        return integer

    if not isinstance(extra_info, Mapping):
        return None, None

    # Breadth-first traversal gives shallower runtime metadata precedence over
    # nested copies while supporting the shapes produced by the trainer:
    # ``extra_info["specification"]`` and
    # ``extra_info["specification"]["extra"]``.
    metadata_candidates: list[Mapping[str, Any]] = []
    pending: list[Mapping[str, Any]] = [extra_info]
    seen: set[int] = set()
    while pending:
        metadata = pending.pop(0)
        if id(metadata) in seen:
            continue
        seen.add(id(metadata))
        metadata_candidates.append(metadata)
        for container_key in ("specification", "extra"):
            nested = metadata.get(container_key)
            if isinstance(nested, Mapping):
                pending.append(nested)

    def first_integer_metadata(**kwargs: Any) -> int | None:
        for metadata in metadata_candidates:
            value = integer_metadata(metadata, **kwargs)
            if value is not None:
                return value
        return None

    total_charge = first_integer_metadata(
        value_keys=("total_charge_B", "total_charge"),
        validity_keys=(
            "charge_valid_mask_B",
            "has_total_charge_B",
            "has_total_charge",
        ),
        name="total_charge",
    )
    spin_multiplicity = first_integer_metadata(
        value_keys=("spin_multiplicity_B", "spin_multiplicity"),
        validity_keys=(
            "spin_valid_mask_B",
            "has_spin_multiplicity_B",
            "has_spin_multiplicity",
        ),
        name="spin_multiplicity",
        minimum=1,
    )
    return total_charge, spin_multiplicity


def compute_molecule_uma_force_metrics(
    molecules: Sequence[Any],
    *,
    total_samples: int,
    checkpoint: str,
    task: str,
    device: str | None,
    batch_size: int,
    target_mean_force_norm: float | None = None,
    total_charge: int | None = None,
    spin_multiplicity: int | None = None,
) -> dict[str, float]:
    """Compute molecule UMA metrics, returning no keys when UMA is unavailable."""
    try:
        from zevals.molecules.uma_forces import (
            compute_uma_force_metrics as compute_zevals_uma_force_metrics,
        )

        # Validation normally disables gradients, while UMA obtains forces by
        # differentiating energy with respect to coordinates.
        with torch.enable_grad():
            return compute_zevals_uma_force_metrics(
                molecules,
                total_samples,
                checkpoint=checkpoint,
                task=task,
                device=device,
                batch_size=batch_size,
                target_mean_force_norm=target_mean_force_norm,
                total_charge=total_charge,
                spin_multiplicity=spin_multiplicity,
            )
    except Exception:
        logger.warning("Skipping UMA force metrics due to an error", exc_info=True)
        return {}


def compute_material_uma_force_metrics(
    structures: Sequence[Any],
    *,
    total_samples: int,
    checkpoint: str,
    task: str,
    device: str | None,
    batch_size: int,
    target_mean_force_norm: float | None = None,
) -> dict[str, float]:
    """Compute material UMA metrics, returning no keys when UMA is unavailable."""
    try:
        from zevals.materials.uma_forces import (
            compute_uma_force_metrics as compute_zevals_uma_force_metrics,
        )

        # Validation normally disables gradients, while UMA obtains forces by
        # differentiating energy with respect to coordinates.
        with torch.enable_grad():
            return compute_zevals_uma_force_metrics(
                structures,
                total_samples,
                checkpoint=checkpoint,
                task=task,
                device=device,
                batch_size=batch_size,
                target_mean_force_norm=target_mean_force_norm,
            )
    except Exception:
        logger.warning("Skipping UMA force metrics due to an error", exc_info=True)
        return {}


__all__ = [
    "compute_material_uma_force_metrics",
    "compute_molecule_uma_force_metrics",
    "electronic_conditioning_from_extra_info",
]
