"""PoseBusters-based ligand pose validation.

Runs the full PoseBusters check suite (geometry, clashes, stereochemistry,
energy, interactions) on a set of generated protein–ligand complexes.

The PoseBusters oracle backend is injected at pipeline construction time
so the underlying implementation is swappable via config.
"""

from __future__ import annotations

from pathlib import Path

from zevals.oracles.base import BasePoseValidator, PoseValidationResult


def validate_poses(
    validator: BasePoseValidator,
    ligand_paths: list[Path],
    protein_paths: list[Path | None] | None = None,
) -> list[PoseValidationResult]:
    """Run pose validation for a batch of ligand (+ optional receptor) paths.

    Args:
        validator:     Injected oracle (e.g. PoseBustersValidator).
        ligand_paths:  One SDF/MOL2 file per generated pose.
        protein_paths: Corresponding receptor PDB files (optional per pose).

    Returns:
        One PoseValidationResult per input ligand.
    """
    if protein_paths is None:
        protein_paths = [None] * len(ligand_paths)

    return [
        validator.validate(lig, prot) for lig, prot in zip(ligand_paths, protein_paths)
    ]


def compute_posebusters_pass_rate(
    results: list[PoseValidationResult],
) -> dict[str, float]:
    """Fraction of poses that pass all PoseBusters checks.

    Returns:
        {"pb_valid": float in [0, 1]}
    """
    if not results:
        return {"pb_valid": 0.0}
    passing = sum(1 for r in results if r.passes)
    return {"pb_valid": passing / len(results)}
