"""Small-molecule conformer geometric quality (UFF force-norm stability).

Mirrors the Zatom-2 molecule-generation evaluator: fits a UFF force field to
each generated conformer and reports the median per-atom force norm — a
proxy for how far the 3D geometry sits from a local energy minimum (lower is
better; a well-relaxed conformer has near-zero net force on every atom).
Requires the molecule to carry a 3D conformer (see `pipeline/loaders.py`'s
`.sdf` handling) — conformer-less molecules (e.g. parsed from bare SMILES)
simply don't contribute a value, matching every other conformer-dependent
check in this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

KCAL_PER_MOL_TO_EV = 0.0433641153087705


def _mol_has_atoms_and_conformer(mol: Any) -> bool:
    """Return whether an RDKit molecule carries atoms and 3D coordinates."""
    return mol is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0


def _mean_force_norm(force_array: Any) -> float:
    """Mean per-atom force norm from an `[N_atoms, 3]` array."""
    forces = np.asarray(force_array, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[-1] != 3 or forces.shape[0] == 0:
        raise ValueError(f"Invalid forces shape {forces.shape}; expected [N_atoms, 3].")
    if not np.isfinite(forces).all():
        raise ValueError("Non-finite forces.")
    return float(np.linalg.norm(forces, axis=-1).mean())


def _mol_uff_mean_force_norm(mol: Any) -> float | None:
    """Mean per-atom UFF force norm for one molecule's conformer, or None if
    it can't be computed (no conformer, or incomplete UFF parameters)."""
    from rdkit.Chem.rdForceFieldHelpers import (
        UFFGetMoleculeForceField,
        UFFHasAllMoleculeParams,
    )

    try:
        if not _mol_has_atoms_and_conformer(mol):
            return None
        if not UFFHasAllMoleculeParams(mol):
            return None
        conf_id = mol.GetConformer().GetId()
        force_field = UFFGetMoleculeForceField(mol, confId=conf_id)
        gradient = np.asarray(force_field.CalcGrad(), dtype=np.float64).reshape(-1, 3)
    except Exception:
        return None
    return _mean_force_norm(-gradient)


def compute_uff_force_norm_metrics(
    mol_list: list[Any],
    total_samples: int | None = None,
    *,
    target_mean_force_norm: float | None = None,
) -> dict[str, float]:
    """Median UFF force-norm across generated conformers, and the fraction
    of samples it could be computed for at all.

    Args:
        mol_list: RDKit molecules to score — only ones with a 3D conformer
            and full UFF parameter coverage contribute a value.
        total_samples: Denominator for `uff_median_force_norm_success_rate`;
            defaults to `len(mol_list)` if not given (pass the pre-validity
            total sample count to match the Zatom-2 convention of measuring
            success rate against everything generated, not just the valid
            subset).
        target_mean_force_norm: Optional ground-truth mean force norm in eV/A.
            When provided, report the mean absolute error between this target
            and each scored conformer's mean per-atom UFF force norm, converted
            from RDKit's native kcal/mol/A to eV/A.

    Returns:
        The aggregate UFF force-norm metrics. The result additionally contains
        `uff_mean_force_norm_mae_eV_per_angstrom` when
        `target_mean_force_norm` is provided.
    """
    if target_mean_force_norm is not None:
        target_mean_force_norm = float(target_mean_force_norm)
        if not np.isfinite(target_mean_force_norm) or target_mean_force_norm < 0:
            raise ValueError(
                "target_mean_force_norm must be a finite non-negative scalar, got "
                f"{target_mean_force_norm!r}"
            )

    total = total_samples if total_samples is not None else len(mol_list)
    values = [
        value
        for mol in mol_list
        if (value := _mol_uff_mean_force_norm(mol)) is not None
    ]
    metrics = {
        "uff_median_force_norm": float(np.median(values)) if values else 0.0,
        "uff_median_force_norm_success_rate": len(values) / total if total else 0.0,
    }
    if target_mean_force_norm is not None:
        values_eV_per_angstrom = np.asarray(values) * KCAL_PER_MOL_TO_EV
        metrics["uff_mean_force_norm_mae_eV_per_angstrom"] = (
            float(np.mean(np.abs(values_eV_per_angstrom - target_mean_force_norm)))
            if values
            else 0.0
        )
    return metrics
