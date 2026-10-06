"""UMA force metrics for small-molecule generation (task head ``omol``).

The molecule path is aperiodic: no cell, ``pbc=False``, and the molecule's
formal charge is passed through to the charge-aware ``omol`` head. Only a
molecule that carries a 3D conformer can be scored (``.sdf`` input, not bare
SMILES), same as every other conformer-level check in this package.

This is the near-DFT counterpart of ``conformer_quality.py``'s UFF force norm:
same idea (residual force = distance from a relaxed geometry), far better
force field, but it needs ``fairchem-core`` and gated UMA weights. Off by
default; switch it on with ``compute_uma_forces: true``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zevals.atomistic.uma.metrics import (
    DEFAULT_CHECKPOINT,
    UMA_KEYS,
    uma_metric_keys,
)

# Molecules are aperiodic and charge/spin aware: OMol25, wB97M-V/def2-TZVPD.
UMA_MOLECULE_TASK = "omol"

UMA_MOLECULE_KEYS: tuple[str, ...] = UMA_KEYS


def compute_uma_force_metrics(
    mol_list: Sequence[Any],
    total_samples: int | None = None,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    task: str = UMA_MOLECULE_TASK,
    device: str | None = None,
    batch_size: int = 8,
    prefix: str = "uma",
    model: Any | None = None,
    target_mean_force_norm: float | None = None,
    total_charge: int | None = None,
    spin_multiplicity: int | None = None,
    **model_kwargs: Any,
) -> dict[str, float]:
    """UMA residual-force metrics for generated molecule conformers.

    Args:
        mol_list: RDKit ``Mol`` objects. Items that are ``None`` or carry no
            3D conformer are skipped, not raised on.
        total_samples: Denominator of ``uma_scored_rate``; defaults to
            ``len(mol_list)``. Pass the pre-validity total sample count to
            measure success against everything generated.
        checkpoint: UMA checkpoint id, or a path to a local ``.pt`` file.
        task: UMA task head. Leave at ``"omol"`` unless you are scoring
            molecular crystals (``"omc"``), which are periodic and belong in
            the materials pipeline instead.
        device: ``"cuda"``, ``"cpu"``, or ``None``/``"auto"``.
        batch_size: Molecules per forward pass.
        prefix: Prepended to every metric key.
        model: An existing ``UMAModel`` to reuse instead of the shared cache.
        target_mean_force_norm: Optional conditioning target in eV/A. When
            provided, report the UMA mean-force-norm MAE against it.
        total_charge: Optional ground-truth charge for this inference
            evaluation. It overrides formal charges inferred from each generated
            molecular graph before ``UMAModel.predict`` is called.
        spin_multiplicity: Optional ground-truth spin multiplicity for this
            inference evaluation.
        **model_kwargs: Forwarded to ``UMAModel`` (``charge``, ``spin``,
            ``precision``).

    Returns:
        ``uma_scored_rate``, ``uma_force_max_mean``, ``uma_force_max_median``,
        ``uma_mean_force_norm_mean``, ``uma_force_rms_mean``,
        ``uma_energy_per_atom_mean`` — all floats, all 0.0 when nothing could
        be scored.
    """
    from zevals.atomistic.uma.metrics import (
        compute_uma_force_metrics as _compute,
    )

    return _compute(
        mol_list,
        total_samples,
        task=task,
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        prefix=prefix,
        model=model,
        target_mean_force_norm=target_mean_force_norm,
        total_charge=total_charge,
        spin_multiplicity=spin_multiplicity,
        **model_kwargs,
    )


__all__ = [
    "UMA_MOLECULE_KEYS",
    "UMA_MOLECULE_TASK",
    "compute_uma_force_metrics",
    "uma_metric_keys",
]
