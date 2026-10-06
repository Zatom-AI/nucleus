"""Binding-pose quality metrics: RMSD and interaction fingerprints.

Requires: rdkit (for RMSD via Chem.rdMolAlign)
"""

from __future__ import annotations

from pathlib import Path

from zevals.utils.packages import load_rdkit_chem


def compute_rmsd(
    generated_sdf: Path,
    reference_sdf: Path,
    use_hungarian: bool = True,
) -> float:
    """Symmetric RMSD between a generated pose and a reference crystal pose.

    Args:
        generated_sdf:  Generated ligand SDF file.
        reference_sdf:  Reference (crystal / docked) SDF file.
        use_hungarian:  Use Hungarian atom matching (symmetry-corrected RMSD).

    Returns:
        RMSD in Angstroms. Returns float('inf') on parse failure.
    """
    Chem = load_rdkit_chem()  # Ensure rdkit is available before proceeding
    try:
        from rdkit.Chem import rdMolAlign  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "rdMolAlign requires working rdkit. Install rdkit: pip install rdkit"
        ) from exc

    gen_mol = Chem.MolFromMolFile(str(generated_sdf), removeHs=False)
    ref_mol = Chem.MolFromMolFile(str(reference_sdf), removeHs=False)

    if gen_mol is None or ref_mol is None:
        return float("inf")

    try:
        if use_hungarian:
            return rdMolAlign.CalcRMS(gen_mol, ref_mol)
        return rdMolAlign.AlignMol(gen_mol, ref_mol)
    except Exception:
        return float("inf")


def compute_rmsd_batch(
    generated_sdfs: list[Path],
    reference_sdfs: list[Path],
    threshold: float = 2.0,
) -> dict[str, float]:
    """RMSD statistics over a batch of generated vs. reference poses.

    Args:
        generated_sdfs: List of generated ligand SDF files.
        reference_sdfs: Corresponding reference SDF files.
        threshold:      RMSD threshold (Å) for counting a pose as "successful".

    Returns:
        {"rmsd_mean": float, "rmsd_median": float, "rmsd_success_rate": float}
    """
    import statistics

    rmsds = [compute_rmsd(g, r) for g, r in zip(generated_sdfs, reference_sdfs)]
    finite = [v for v in rmsds if v != float("inf")]
    if not finite:
        return {
            "rmsd_mean": float("inf"),
            "rmsd_median": float("inf"),
            "rmsd_success_rate": 0.0,
        }

    return {
        "rmsd_mean": statistics.mean(finite),
        "rmsd_median": statistics.median(finite),
        "rmsd_success_rate": sum(v < threshold for v in finite) / len(rmsds),
    }
