"""Protein designability metric (FrameDiff / La Proteina pipeline).

Designability = % of generated backbones where at least one ProteinMPNN/SolubleMPNN
sequence folds back (via ESMFold) to within scRMSD < 2 Å of the original backbone
and, when configured, passes a mean ESMFold pLDDT threshold.

Pipeline per backbone:
  1. ProteinMPNN/SolubleMPNN → N candidate sequences (inverse fold).
  2. ESMFold → predicted structure per sequence (forward fold).
  3. Compute CA-RMSD between predicted and original backbone (scRMSD).
  4. Backbone is designable if any candidate passes the scRMSD and pLDDT gates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from zevals.oracles.base import BaseInverseFolder, BaseStructurePredictor

logger = logging.getLogger(__name__)


def compute_designability(
    backbone_pdbs: list[Path],
    structure_predictor: BaseStructurePredictor,
    inverse_folder: BaseInverseFolder,
    output_dir: Path,
    sc_rmsd_threshold: float = 2.0,
    plddt_threshold: float | None = None,
    sc_atom: str = "CA",
) -> tuple[dict[str, float], list[Path]]:
    """Fraction of backbones passing the scRMSD and optional pLDDT gates.

    Args:
        backbone_pdbs:       Paths to generated backbone structure files
                             (`.cif` or `.pdb`).
        structure_predictor: Forward-folding oracle (ESMFold).
        inverse_folder:      Inverse-folding oracle (ProteinMPNN / SolubleMPNN).
        output_dir:          Root directory for oracle I/O temp files.
        sc_rmsd_threshold:   Å cutoff (default 2.0, following FrameDiff).
        plddt_threshold:     Optional mean ESMFold pLDDT cutoff.
        sc_atom:             Per-residue atom the scRMSD is measured on
                             (default ``"CA"``; e.g. ``"C1'"``/``"C4'"`` for
                             nucleic-acid backbones, or a backbone atom like
                             ``"N"``/``"C"``).

    Returns:
        Tuple of:
          - {"designability": fraction of designable backbones,
             "mean_min_scRMSD": mean of per-backbone min(scRMSD)}
          - the subset of `backbone_pdbs` that passed the threshold, in the
            same order as the input. Diversity/novelty are only meaningful
            over this designable subset (per proteins/METRICS.md) — pass it
            straight through to `diversity.py`/`novelty.py` rather than the
            full `backbone_pdbs` list.
    """
    sc_rmsds: list[float] = []
    best_plddts: list[float] = []
    designable_flags: list[bool] = []

    for pdb_path in backbone_pdbs:
        try:
            bb_out = output_dir / pdb_path.stem
            bb_out.mkdir(parents=True, exist_ok=True)

            # Step 1: inverse-fold the backbone.
            inv_result = inverse_folder.design(pdb_path, bb_out / "sequences")
            if not inv_result.sequences:
                sc_rmsds.append(float("inf"))
                best_plddts.append(float("nan"))
                designable_flags.append(False)
                continue

            # Step 2: forward-fold each designed sequence.
            fasta_path = bb_out / "designed.fasta"
            _write_fasta(fasta_path, inv_result.sequences)
            predictions = structure_predictor.predict(fasta_path, bb_out / "structures")

            # Step 3: compute scRMSD for each prediction vs the original backbone.
            rmsds = [
                _atom_rmsd(pdb_path, pred.pdb_path, sc_atom) for pred in predictions
            ]
            if not rmsds:
                sc_rmsds.append(float("inf"))
                best_plddts.append(float("nan"))
                designable_flags.append(False)
                continue

            best_idx = int(np.argmin(rmsds))
            sc_rmsds.append(rmsds[best_idx])
            best_plddts.append(predictions[best_idx].plddt)
            designable_flags.append(
                any(
                    rmsd < sc_rmsd_threshold
                    and (plddt_threshold is None or pred.plddt > plddt_threshold)
                    for rmsd, pred in zip(rmsds, predictions)
                )
            )
        except Exception:
            logger.warning(
                "Skipping %s: designability check failed", pdb_path, exc_info=True
            )
            sc_rmsds.append(float("inf"))
            best_plddts.append(float("nan"))
            designable_flags.append(False)

    n = len(sc_rmsds)
    designable_pdbs = [
        p for p, is_designable in zip(backbone_pdbs, designable_flags) if is_designable
    ]
    finite = [r for r in sc_rmsds if r != float("inf")]
    finite_plddts = [p for p in best_plddts if np.isfinite(p)]

    metrics = {
        "designability": len(designable_pdbs) / n if n else 0.0,
        "mean_min_scRMSD": sum(finite) / len(finite) if finite else float("inf"),
        "mean_best_plddt": sum(finite_plddts) / len(finite_plddts)
        if finite_plddts
        else float("nan"),
    }
    return metrics, designable_pdbs


def _write_fasta(path: Path, sequences: list[str]) -> None:
    lines = []
    for i, seq in enumerate(sequences):
        lines += [f">seq_{i}", seq]
    path.write_text("\n".join(lines) + "\n")


def _atom_coords(pdb_path: Path, atom: str = "CA") -> np.ndarray:
    """Ordered coordinates of one representative `atom` per residue in a
    (single-model) structure file — `.cif` (mmCIF) or `.pdb`, dispatched by
    suffix. Called on both the original backbone (may now be either format)
    and the refold (see `oracles/structure_predictors/*.py`), so both need to
    work through this one function.

    `atom` selects which per-residue atom defines the self-consistency RMSD
    (default ``"CA"`` for proteins; e.g. ``"C1'"`` / ``"C4'"`` for nucleic
    acids). Residues lacking that atom are skipped."""
    from zevals.utils.io import load_structure_mmcif, load_structure_pdb

    if pdb_path.suffix == ".cif":
        structure = load_structure_mmcif(pdb_path)
    else:
        structure = load_structure_pdb(pdb_path)
    coords = [
        residue[atom].get_coord()
        for residue in structure.get_residues()
        if atom in residue
    ]
    return np.asarray(coords, dtype=np.float64)


def _kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD between two corresponding point sets after optimal rigid superposition.

    Standard Kabsch algorithm: center both point sets, find the rotation
    minimizing squared distance via SVD of the cross-covariance matrix
    (with a reflection correction so the result is a proper rotation), then
    measure RMSD post-alignment.
    """
    # Invalid refold coordinates should invalidate this candidate, not abort
    # collection of the remaining sequences/backbones in a sweep.
    if (
        mobile.ndim != 2
        or target.ndim != 2
        or mobile.shape[1:] != (3,)
        or target.shape[1:] != (3,)
        or mobile.shape[0] != target.shape[0]
        or not np.isfinite(mobile).all()
        or not np.isfinite(target).all()
    ):
        return float("inf")

    p = mobile - mobile.mean(axis=0)
    q = target - target.mean(axis=0)

    H = p.T @ q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return float("inf")
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    p_aligned = p @ R.T
    return float(np.sqrt(((p_aligned - q) ** 2).sum(axis=-1).mean()))


def _atom_rmsd(ref_pdb: Path, pred_pdb: Path, atom: str = "CA") -> float:
    """Per-residue `atom` RMSD between two structure files after optimal
    (Kabsch) superposition.

    `ref_pdb` may be `.cif` or `.pdb` (the original backbone); `pred_pdb` is
    the refold (see `oracles/structure_predictors/*.py`). `_atom_coords`
    dispatches on each file's own suffix. `atom` picks the representative
    atom (default ``"CA"``).

    Assumes positional residue correspondence between the two files (true
    here: `pred_pdb` is the refold of the exact sequence designed for
    `ref_pdb`'s backbone, so residue counts and order match). Falls back to
    the shorter length if they differ slightly (e.g. missing terminal atoms).
    """
    ref_coords = _atom_coords(ref_pdb, atom)
    pred_coords = _atom_coords(pred_pdb, atom)
    n = min(len(ref_coords), len(pred_coords))
    if n == 0:
        return float("inf")
    return _kabsch_rmsd(pred_coords[:n], ref_coords[:n])
