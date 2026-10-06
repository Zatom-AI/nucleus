"""UMA force metrics — the shared core behind the molecule and material wrappers.

A generated 3D structure that sits near a physical energy minimum has small
residual forces on every atom. A strained or badly generated one has a large
maximum force. UMA gives that force field at near-DFT quality, so this is a
much stronger geometry check than the classical UFF force norm
(``zevals.molecules.conformer_quality``), at the cost of a GPU and a gated
model download.

Do not call this module directly from a pipeline. Use the modality wrapper,
which sets the correct UMA task head:

* ``zevals.molecules.uma_forces.compute_uma_force_metrics`` -> ``omol``
* ``zevals.materials.uma_forces.compute_uma_force_metrics`` -> ``omat``

Contract, same as every other zevals metric function: it takes a plain list, it
never raises because of one bad sample, it accepts ``total_samples`` so a rate
counts against the full generated set, and it returns a flat dict of floats.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Metric names, in the order they are reported. `uma_metric_keys` rebuilds this
# for a non-default prefix.
UMA_KEYS: tuple[str, ...] = (
    "uma_scored_rate",
    "uma_force_max_mean",
    "uma_force_max_median",
    "uma_mean_force_norm_mean",
    "uma_force_rms_mean",
    "uma_energy_per_atom_mean",
)

DEFAULT_CHECKPOINT = "uma-s-1p2"


def uma_metric_keys(prefix: str = "uma") -> tuple[str, ...]:
    """Metric key names for a given prefix (``"uma"`` by default)."""
    return tuple(f"{prefix}{key[len('uma'):]}" for key in UMA_KEYS)


def _zeros(prefix: str) -> dict[str, float]:
    """Every key at 0.0 — the result when nothing could be scored."""
    return dict.fromkeys(uma_metric_keys(prefix), 0.0)


def _with_target_error_zero(
    metrics: dict[str, float],
    *,
    prefix: str,
    target_mean_force_norm: float | None,
) -> dict[str, float]:
    """Include the conditioning-error key when a target was requested."""
    if target_mean_force_norm is not None:
        metrics[f"{prefix}_mean_force_norm_mae_eV_per_angstrom"] = 0.0
    return metrics


def _validated_integer_conditioning(
    value: Any | None,
    *,
    name: str,
    minimum: int | None = None,
) -> int | None:
    """Validate an optional system-level electronic-state conditioning value."""
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer scalar, got {value!r}") from exc
    if not scalar.is_integer():
        raise ValueError(f"{name} must be an integer scalar, got {value!r}")
    integer = int(scalar)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {integer}")
    return integer


def _apply_electronic_conditioning(
    atoms_list: Sequence[Any],
    *,
    total_charge: Any | None,
    spin_multiplicity: Any | None,
) -> None:
    """Override converted samples with one inference evaluation's ground truth."""
    charge = _validated_integer_conditioning(total_charge, name="total_charge")
    spin = _validated_integer_conditioning(
        spin_multiplicity,
        name="spin_multiplicity",
        minimum=1,
    )
    for atoms in atoms_list:
        if charge is not None:
            atoms.info["charge"] = charge
        if spin is not None:
            atoms.info["spin"] = spin


def _apply_periodic_electronic_defaults(
    atoms_list: Sequence[Any],
    *,
    task: str,
    total_charge: Any | None,
    spin_multiplicity: Any | None,
    model_kwargs: dict[str, Any],
) -> None:
    """Enforce UMA's fixed charge/spin inputs for periodic task heads."""
    supplied_values = {
        "total_charge": total_charge,
        "spin_multiplicity": spin_multiplicity,
        "charge": model_kwargs.get("charge"),
        "spin": model_kwargs.get("spin"),
    }
    for name, value in supplied_values.items():
        integer = _validated_integer_conditioning(value, name=name)
        if integer not in (None, 0):
            raise ValueError(
                f"UMA task {task!r} requires {name}=0, got {integer}. "
                "Only the 'omol' task supports varying charge or spin."
            )

    # AtomicData always carries charge and spin tensors. Override values from
    # pre-populated ASE inputs so periodic heads receive UMA's in-distribution
    # null/default state rather than molecule-style electronic conditioning.
    for atoms in atoms_list:
        atoms.info["charge"] = 0
        atoms.info["spin"] = 0


def score_uma_samples(
    samples: Sequence[Any],
    *,
    task: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str | None = None,
    batch_size: int = 8,
    model: Any | None = None,
    total_charge: int | None = None,
    spin_multiplicity: int | None = None,
    **model_kwargs: Any,
) -> list[dict[str, int | float]]:
    """Return index-aligned UMA predictions for every successfully scored sample.

    Unlike :func:`compute_uma_force_metrics`, this function retains the input
    index for each prediction.  That is required when every generated sample
    has its own conditioning target, charge, or spin.  Callers may pre-populate
    per-sample ``ASE Atoms.info`` values; shared ``total_charge`` and
    ``spin_multiplicity`` arguments override them when provided.

    A failed conversion, non-finite prediction, or failed batch is omitted from
    the result.  The retained ``sample_index`` values allow callers to align the
    remaining scores with their original metadata without shifting indices.
    """
    import torch

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    from zevals.atomistic.uma.adapters import get_uma_model, sample_to_atoms
    from zevals.atomistic.uma.model import UMA_PERIODIC_TASKS

    indexed_atoms = [
        (sample_index, atoms)
        for sample_index, sample in enumerate(samples)
        if (atoms := sample_to_atoms(sample)) is not None
    ]
    atoms_list = [atoms for _, atoms in indexed_atoms]
    if task in UMA_PERIODIC_TASKS:
        _apply_periodic_electronic_defaults(
            atoms_list,
            task=task,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            model_kwargs=model_kwargs,
        )
    else:
        _apply_electronic_conditioning(
            atoms_list,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
        )
    if not indexed_atoms:
        return []

    if model is None:
        model = get_uma_model(
            checkpoint=checkpoint, task=task, device=device, **model_kwargs
        )

    records: list[dict[str, int | float]] = []
    for start in range(0, len(indexed_atoms), batch_size):
        indexed_chunk = indexed_atoms[start : start + batch_size]
        chunk = [atoms for _, atoms in indexed_chunk]
        try:
            result = model.predict(chunk)
            per_system = model.split_per_system(result, "forces")
            energy = result["energy"].reshape(-1).float().cpu()
            if len(per_system) != len(indexed_chunk) or len(energy) != len(
                indexed_chunk
            ):
                raise ValueError(
                    "UMA returned a different number of systems than it received."
                )
        except Exception:
            # A whole chunk can fail (GPU OOM, an element the head does not
            # cover). Drop it and keep going; callers report a scored rate.
            logger.warning(
                "Skipping a UMA batch of %d samples due to an error",
                len(chunk),
                exc_info=True,
            )
            continue

        for (sample_index, _), forces, sample_energy in zip(
            indexed_chunk,
            per_system,
            energy,
            strict=True,
        ):
            force_tensor = forces.float().cpu()
            if (
                force_tensor.ndim != 2
                or force_tensor.shape[0] == 0
                or force_tensor.shape[1] != 3
                or not torch.isfinite(force_tensor).all()
                or not torch.isfinite(sample_energy)
            ):
                continue
            force_norms = torch.linalg.norm(force_tensor, dim=1)
            records.append(
                {
                    "sample_index": sample_index,
                    "force_max": float(force_norms.max()),
                    "mean_force_norm": float(force_norms.mean()),
                    "force_rms": float(torch.sqrt((force_tensor**2).sum(dim=1).mean())),
                    "energy_per_atom": float(sample_energy) / force_tensor.shape[0],
                }
            )
    return records


def compute_uma_force_metrics(
    samples: Sequence[Any],
    total_samples: int | None = None,
    *,
    task: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str | None = None,
    batch_size: int = 8,
    prefix: str = "uma",
    model: Any | None = None,
    target_mean_force_norm: float | None = None,
    total_charge: int | None = None,
    spin_multiplicity: int | None = None,
    **model_kwargs: Any,
) -> dict[str, float]:
    """Score generated structures with UMA and return a flat dict of floats.

    Args:
        samples: RDKit ``Mol`` objects, pymatgen ``Structure`` objects, or ASE
            ``Atoms``. ``None`` items and items without 3D geometry are skipped.
        total_samples: The full generated-sample count, used as the denominator
            of the scored rate. Defaults to ``len(samples)``; pass the
            pre-validity total to match the zevals convention of measuring
            success against everything generated.
        task: UMA task head. Required, because the right value depends on the
            chemistry — ``"omol"`` for molecules, ``"omat"`` for crystals.
        checkpoint: UMA checkpoint id, or a path to a local ``.pt`` file.
        device: ``"cuda"``, ``"cpu"``, or ``None``/``"auto"``. An indexed
            device such as ``"cuda:1"`` is reduced to ``"cuda"`` (fairchem
            cannot target an index — use ``CUDA_VISIBLE_DEVICES``).
        batch_size: Systems per forward pass. Lower it if you hit GPU OOM.
        prefix: Prepended to every metric key.
        model: An existing ``UMAModel`` to reuse. When omitted, a cached one is
            built from ``checkpoint``/``task``/``device``.
        target_mean_force_norm: Optional reference mean force norm in eV/A. When
            provided, report the mean absolute difference between this scalar
            and each scored structure's UMA mean per-atom force norm.
        total_charge: Optional ground-truth total charge shared by the generated
            structures for one inference evaluation. For ``omol``, it overrides
            charges inferred from generated molecular graphs. Periodic tasks
            accept only zero.
        spin_multiplicity: Optional ground-truth spin multiplicity shared by the
            generated structures for one inference evaluation. Periodic tasks
            accept only zero.
        **model_kwargs: Forwarded to ``UMAModel`` (``charge``, ``spin``,
            ``precision``).

    Returns:
        A flat dict of floats. With the default prefix:
        ``uma_scored_rate`` (scored count / ``total_samples``),
        ``uma_force_max_mean`` and ``uma_force_max_median`` (the largest force
        on any atom, in eV/A, averaged over samples),
        ``uma_mean_force_norm_mean`` (the mean per-atom force norm for each
        sample, averaged over samples), ``uma_force_rms_mean`` (the per-sample
        root-mean-square force norm, averaged over samples), and
        ``uma_energy_per_atom_mean`` (eV/atom, for reference only — UMA
        energies are absolute, so compare them only within one chemical
        system). When ``target_mean_force_norm`` is provided,
        ``uma_mean_force_norm_mae_eV_per_angstrom`` measures adherence to that
        target. Every requested key is 0.0 when nothing could be scored.
    """
    import numpy as np

    n_total = len(samples) if total_samples is None else int(total_samples)
    if target_mean_force_norm is not None:
        target_mean_force_norm = float(target_mean_force_norm)
        if not np.isfinite(target_mean_force_norm) or target_mean_force_norm < 0:
            raise ValueError(
                "target_mean_force_norm must be a finite non-negative scalar, got "
                f"{target_mean_force_norm!r}"
            )

    if n_total == 0:
        logger.info("UMA scored 0 of %d samples: no usable 3D geometry", n_total)
        return _with_target_error_zero(
            _zeros(prefix),
            prefix=prefix,
            target_mean_force_norm=target_mean_force_norm,
        )

    records = score_uma_samples(
        samples,
        task=task,
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        model=model,
        total_charge=total_charge,
        spin_multiplicity=spin_multiplicity,
        **model_kwargs,
    )

    if not records:
        logger.warning("UMA scored 0 of %d samples: every batch failed", n_total)
        return _with_target_error_zero(
            _zeros(prefix),
            prefix=prefix,
            target_mean_force_norm=target_mean_force_norm,
        )

    force_max = [float(record["force_max"]) for record in records]
    mean_force_norm = [float(record["mean_force_norm"]) for record in records]
    force_rms = [float(record["force_rms"]) for record in records]
    energy_per_atom = [float(record["energy_per_atom"]) for record in records]
    metrics = {
        f"{prefix}_scored_rate": len(records) / n_total,
        f"{prefix}_force_max_mean": float(np.mean(force_max)),
        f"{prefix}_force_max_median": float(np.median(force_max)),
        f"{prefix}_mean_force_norm_mean": float(np.mean(mean_force_norm)),
        f"{prefix}_force_rms_mean": float(np.mean(force_rms)),
        f"{prefix}_energy_per_atom_mean": float(np.mean(energy_per_atom)),
    }
    if target_mean_force_norm is not None:
        metrics[f"{prefix}_mean_force_norm_mae_eV_per_angstrom"] = float(
            np.mean([abs(value - target_mean_force_norm) for value in mean_force_norm])
        )
    return metrics


__all__ = [
    "DEFAULT_CHECKPOINT",
    "UMA_KEYS",
    "compute_uma_force_metrics",
    "score_uma_samples",
    "uma_metric_keys",
]
