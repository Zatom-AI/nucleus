"""UMA force metrics for crystal generation (task head ``omat``).

The materials path is periodic: the pymatgen lattice matrix becomes the ASE
cell and ``pbc=True``. That flag matters — drop the cell and the atoms stand
alone in vacuum, so a perfectly relaxed crystal reports large spurious forces.

Pick the head that matches the chemistry:

| ``task``  | Use it for                    |
|-----------|-------------------------------|
| ``omat``  | inorganic crystals (default)  |
| ``omc``   | molecular crystals            |
| ``odac``  | metal-organic frameworks      |
| ``oc20``  | catalyst slabs and adsorbates |

Off by default; switch it on with ``compute_uma_forces: true``. It needs
``fairchem-core`` and gated UMA weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zevals.atomistic.uma.metrics import (
    DEFAULT_CHECKPOINT,
    UMA_KEYS,
    uma_metric_keys,
)
from zevals.atomistic.uma.model import UMA_PERIODIC_TASKS

# Inorganic crystals: OMat24, DFT PBE. Periodic, charge/spin agnostic.
UMA_MATERIAL_TASK = "omat"

UMA_MATERIAL_KEYS: tuple[str, ...] = UMA_KEYS


def compute_uma_force_metrics(
    structures: Sequence[Any],
    total_samples: int | None = None,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    task: str = UMA_MATERIAL_TASK,
    device: str | None = None,
    batch_size: int = 8,
    prefix: str = "uma",
    model: Any | None = None,
    target_mean_force_norm: float | None = None,
    **model_kwargs: Any,
) -> dict[str, float]:
    """UMA residual-force metrics for generated periodic structures.

    Args:
        structures: pymatgen ``Structure`` objects. Items that are ``None`` or
            fail to convert are skipped, not raised on.
        total_samples: Denominator of ``uma_scored_rate``; defaults to
            ``len(structures)``. Pass the pre-validity total sample count to
            measure success against everything generated.
        checkpoint: UMA checkpoint id, or a path to a local ``.pt`` file.
        task: UMA task head — one of ``UMA_PERIODIC_TASKS``. A non-periodic
            head (``"omol"``) is rejected, because it ignores the cell and
            would silently report vacuum forces for a crystal.
        device: ``"cuda"``, ``"cpu"``, or ``None``/``"auto"``.
        batch_size: Structures per forward pass. Large unit cells build large
            neighbour graphs — lower it if you hit GPU OOM.
        prefix: Prepended to every metric key.
        model: An existing ``UMAModel`` to reuse instead of the shared cache.
        target_mean_force_norm: Optional conditioning target in eV/A. When
            provided, report the UMA mean-force-norm MAE against it.
        **model_kwargs: Forwarded to ``UMAModel``. ``precision`` may vary;
            ``charge`` and ``spin`` are accepted only when they are zero, as
            required by UMA's periodic task heads.

    Returns:
        ``uma_scored_rate``, ``uma_force_max_mean``, ``uma_force_max_median``,
        ``uma_mean_force_norm_mean``, ``uma_force_rms_mean``,
        ``uma_energy_per_atom_mean`` — all floats, all 0.0 when nothing could
        be scored.

    Raises:
        ValueError: If ``task`` is not a periodic UMA head.
    """
    if task not in UMA_PERIODIC_TASKS:
        raise ValueError(
            f"materials UMA metrics need a periodic task head, one of "
            f"{UMA_PERIODIC_TASKS}, got {task!r}. Use "
            f"zevals.molecules.uma_forces for aperiodic molecules."
        )

    from zevals.atomistic.uma.metrics import (
        compute_uma_force_metrics as _compute,
    )

    return _compute(
        structures,
        total_samples,
        task=task,
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        prefix=prefix,
        model=model,
        target_mean_force_norm=target_mean_force_norm,
        **model_kwargs,
    )


__all__ = [
    "UMA_MATERIAL_KEYS",
    "UMA_MATERIAL_TASK",
    "UMA_PERIODIC_TASKS",
    "compute_uma_force_metrics",
    "uma_metric_keys",
]
