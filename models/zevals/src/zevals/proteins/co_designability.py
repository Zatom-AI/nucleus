"""Sequence-Structure Co-designability metric.

A backbone is co-designable if, for at least one designed sequence:
  - Sequence recovery passes a threshold (designed sequence is plausible), AND
  - scRMSD < sc_rmsd_threshold (the sequence folds back to the original backbone).

Both criteria must pass simultaneously for the same sequence — this is a joint
per-backbone metric following the FrameDiff / La Proteina benchmarking convention.

Sequence recovery = fraction of residues where the designed amino acid matches
the native sequence (if a reference sequence is available).
"""

from __future__ import annotations

import logging
from pathlib import Path

from zevals.oracles.base import BaseInverseFolder, BaseStructurePredictor
from zevals.proteins.designability import _atom_rmsd, _write_fasta

logger = logging.getLogger(__name__)


def compute_co_designability(
    backbone_pdbs: list[Path],
    structure_predictor: BaseStructurePredictor,
    inverse_folder: BaseInverseFolder,
    output_dir: Path,
    sc_rmsd_threshold: float = 2.0,
    seq_recovery_threshold: float = 0.0,  # set >0 if native sequences are available
    native_sequences: list[str] | None = None,
) -> dict[str, float]:
    """Fraction of backbones that are jointly sequence- and structure-designable.

    Args:
        backbone_pdbs:          Paths to generated backbone structure files
                                (`.cif` or `.pdb`).
        structure_predictor:    Forward-folding oracle.
        inverse_folder:         Inverse-folding oracle.
        output_dir:             Root directory for oracle I/O temp files.
        sc_rmsd_threshold:      Å cutoff for structural recovery (default 2.0).
        seq_recovery_threshold: Min sequence recovery to count (default 0.0 = disabled).
        native_sequences:       Native sequences corresponding to each backbone, used
                                to compute sequence recovery. If None, only structural
                                recovery is checked (degenerates to designability).

    Returns:
        {
          "co_designability":       fraction passing both criteria,
          "struct_only_designability": fraction passing structural criterion only,
          "mean_seq_recovery":      mean sequence recovery over all designed sequences,
          "mean_plddt_co_designable": mean ESMFold pLDDT of the passing sequence,
                                       averaged over co-designable backbones only
                                       (METRICS.md section 2.2).
        }

    Note: this computes Cα co-designability only (`_atom_rmsd`, same as
    `designability.py`). All-atom co-designability — La-Proteína's headline
    metric, comparing every heavy atom rather than just Cα — needs correct
    atom37 correspondence between the generated structure and the ESMFold
    refold and is not implemented yet.
    """
    n = len(backbone_pdbs)
    co_designable = 0
    struct_only = 0
    seq_recoveries: list[float] = []
    plddt_co_designable: list[float] = []

    for i, pdb_path in enumerate(backbone_pdbs):
        try:
            native_seq = native_sequences[i] if native_sequences else None
            bb_out = output_dir / pdb_path.stem
            bb_out.mkdir(parents=True, exist_ok=True)

            inv_result = inverse_folder.design(pdb_path, bb_out / "sequences")
            if not inv_result.sequences:
                continue

            fasta_path = bb_out / "designed.fasta"
            _write_fasta(fasta_path, inv_result.sequences)
            predictions = structure_predictor.predict(fasta_path, bb_out / "structures")

            for designed_seq, pred in zip(inv_result.sequences, predictions):
                rmsd = _atom_rmsd(pdb_path, pred.pdb_path)
                struct_ok = rmsd < sc_rmsd_threshold

                if native_seq is not None:
                    recovery = _seq_recovery(designed_seq, native_seq)
                    seq_recoveries.append(recovery)
                    seq_ok = recovery >= seq_recovery_threshold
                else:
                    seq_ok = True  # no native seq → skip sequence criterion

                if struct_ok:
                    struct_only += 1
                    if seq_ok:
                        co_designable += 1
                        plddt_co_designable.append(pred.plddt)
                    break  # count the backbone once (best sequence)
        except Exception:
            logger.warning(
                "Skipping %s: co-designability check failed", pdb_path, exc_info=True
            )

    return {
        "co_designability": co_designable / n if n else 0.0,
        "struct_only_designability": struct_only / n if n else 0.0,
        "mean_seq_recovery": sum(seq_recoveries) / len(seq_recoveries)
        if seq_recoveries
        else float("nan"),
        "mean_plddt_co_designable": sum(plddt_co_designable) / len(plddt_co_designable)
        if plddt_co_designable
        else float("nan"),
    }


def _seq_recovery(designed: str, native: str) -> float:
    """Fraction of positions where designed matches native (same length assumed)."""
    if not native or len(designed) != len(native):
        return float("nan")
    matches = sum(d == n for d, n in zip(designed, native))
    return matches / len(native)
