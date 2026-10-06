"""Ligand evaluation pipeline.

Evaluates generated protein-conditioned ligand poses using PoseBusters
and RMSD-based metrics against a reference pose set.

Hydra config (ligands.yaml):
  _target_: zevals.pipeline.ligands.LigandEvalPipeline
  oracle:
    pose_validator:
      _target_: zevals.oracles.pose_validators.posebusters.PoseBustersValidator
      table: dock
      add_hydrogens: true
  rmsd_threshold: 2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zevals.ligands.pose_quality import compute_rmsd_batch
from zevals.ligands.posebusters import (
    compute_posebusters_pass_rate,
    validate_poses,
)
from zevals.oracles.base import BasePoseValidator
from zevals.pipeline.base import EvalPipeline, EvalResults, safe_update

logger = logging.getLogger(__name__)


@dataclass
class LigandSample:
    ligand_path: Path
    protein_path: Path | None = None
    reference_ligand_path: Path | None = None


class LigandEvalPipeline(EvalPipeline):
    """End-to-end evaluation for protein-conditioned ligand design."""

    def __init__(
        self,
        pose_validator: BasePoseValidator,
        rmsd_threshold: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self.pose_validator = pose_validator
        self.rmsd_threshold = rmsd_threshold

    def run(self, samples: list[LigandSample]) -> EvalResults:
        """Evaluate a batch of generated ligand poses.

        Args:
            samples: One LigandSample per generated ligand.

        Returns:
            EvalResults with PoseBusters pass rate and RMSD metrics.
        """
        ligand_paths = [s.ligand_path for s in samples]
        protein_paths = [s.protein_path for s in samples]

        agg: dict[str, float] = {}
        pb_results = None
        try:
            pb_results = validate_poses(
                self.pose_validator, ligand_paths, protein_paths
            )
            agg.update(compute_posebusters_pass_rate(pb_results))
        except Exception:
            logger.warning(
                "Skipping PoseBusters metrics due to an error", exc_info=True
            )

        ref_paths = [
            s.reference_ligand_path for s in samples if s.reference_ligand_path
        ]
        gen_paths = [s.ligand_path for s in samples if s.reference_ligand_path]
        if ref_paths:
            safe_update(
                agg,
                "RMSD",
                compute_rmsd_batch,
                gen_paths,
                ref_paths,
                threshold=self.rmsd_threshold,
            )

        if pb_results is not None:
            per_sample = [
                {"ligand": str(s.ligand_path), "pb_passes": r.passes}
                for s, r in zip(samples, pb_results)
            ]
        else:
            per_sample = [{"ligand": str(s.ligand_path)} for s in samples]

        return EvalResults(per_sample=per_sample, aggregate=agg)

    def aggregate(self, per_sample: list[dict[str, Any]]) -> dict[str, float]:
        passing = sum(1 for d in per_sample if d.get("pb_passes"))
        return {"pb_valid": passing / len(per_sample) if per_sample else 0.0}
