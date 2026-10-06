"""Ligand self-consistency: redocking roundtrip.

After generating a ligand pose, we strip the coordinates, re-dock the 2-D
graph back into the pocket, and check whether the re-docked pose recovers the
original (RMSD < threshold). This tests whether the generated conformation is
consistent with the pocket geometry.

The docking oracle is intentionally left as a stub — wire in your preferred
docking engine (Vina, Glide, DiffDock, etc.) by implementing the abstract
interface below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDockingOracle(ABC):
    """Minimal interface for a docking oracle used in self-consistency checks."""

    @abstractmethod
    def dock(self, ligand_sdf: Path, protein_pdb: Path, n_poses: int = 1) -> list[Path]:
        """Return paths to docked pose SDF files (best `n_poses` poses)."""


def compute_redock_self_consistency(
    generated_sdfs: list[Path],
    protein_pdbs: list[Path],
    docking_oracle: BaseDockingOracle,
    rmsd_threshold: float = 2.0,
) -> dict[str, float]:
    """Fraction of ligands whose top-1 redocked pose recovers the generated pose.

    Args:
        generated_sdfs:   Original generated ligand SDF files.
        protein_pdbs:     Corresponding receptor PDB files.
        docking_oracle:   Docking backend (inject via config or directly).
        rmsd_threshold:   RMSD (Å) below which redocking is considered successful.

    Returns:
        {"redock_self_consistency": float in [0, 1]}
    """
    from zevals.ligands.pose_quality import compute_rmsd

    successes = 0
    for gen_sdf, prot_pdb in zip(generated_sdfs, protein_pdbs):
        redocked = docking_oracle.dock(gen_sdf, prot_pdb, n_poses=1)
        if not redocked:
            continue
        rmsd = compute_rmsd(redocked[0], gen_sdf)
        if rmsd < rmsd_threshold:
            successes += 1

    total = len(generated_sdfs)
    return {"redock_self_consistency": successes / total if total else 0.0}
