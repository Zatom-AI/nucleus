"""RNA evaluation pipeline.

Evaluates generated RNA structures on validity (scTM > 0.45), structural
novelty (TMalign vs a directory of PDB+RNAsolo reference structures),
structural diversity (qTMclust), and sequence diversity (MMseqs2). TMalign
and qTMclust (both part of USalign) are the RNA-native equivalents of
FoldSeek used for proteins — FoldSeek's structural alphabet doesn't
understand nucleic acids. Oracles are fully config-driven.

Hydra config (rna.yaml):
  _target_: zevals.pipeline.rna.RNAEvalPipeline
  oracle:
    structure_predictor:
      _target_: zevals.oracles.structure_predictors.boltz.BoltzPredictor
      entity_type: rna
      executable: /path/to/boltz_env/bin/boltz
      repo_path: /path/to/boltz
      cache_dir: /path/to/boltz_cache
    inverse_folder:
      _target_: zevals.oracles.inverse_folders.na_mpnn.NAMPNNFolder
      checkpoint_path: null   # falls back to zevals setup --checkpoints na_mpnn_design
      num_sequences: 8
    structure_searcher:
      _target_: zevals.oracles.structure_search.usalign.USalignSearcher
      binary_path: null
  sc_tm_threshold: 0.45
  novelty_tm_threshold: 0.45
  target_db: /path/to/dir_of_reference_pdbs   # PDB+RNAsolo; a directory, not a compiled DB
  tm_align_bin: TMalign
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zevals.oracles.base import (
    BaseInverseFolder,
    BaseStructurePredictor,
    BaseStructureSearcher,
)
from zevals.pipeline.base import EvalPipeline, EvalResults
from zevals.rna.diversity import (
    compute_sequence_diversity,
    compute_structural_diversity,
)
from zevals.rna.novelty import compute_novelty
from zevals.rna.validity import compute_validity_detailed

logger = logging.getLogger(__name__)


class RNAEvalPipeline(EvalPipeline):
    """End-to-end evaluation for RNA backbone generation."""

    def __init__(
        self,
        structure_predictor: BaseStructurePredictor,
        inverse_folder: BaseInverseFolder,
        structure_searcher: BaseStructureSearcher | None = None,
        sc_tm_threshold: float = 0.45,
        sc_atom: str = "C1'",
        novelty_tm_threshold: float = 0.45,
        target_db: Path | str | None = None,
        output_dir: Path | str = "/tmp/zevals/rna",
        mmseqs_bin: str = "mmseqs",
        tm_align_bin: str = "TMalign",
        **kwargs: Any,
    ) -> None:
        self.structure_predictor = structure_predictor
        self.inverse_folder = inverse_folder
        self.structure_searcher = structure_searcher
        self.sc_tm_threshold = sc_tm_threshold
        self.sc_atom = sc_atom
        self.novelty_tm_threshold = novelty_tm_threshold
        self.target_db = Path(target_db) if target_db else None
        self.output_dir = Path(output_dir)
        self.mmseqs_bin = mmseqs_bin
        self.tm_align_bin = tm_align_bin

    def run(self, samples: list[Path]) -> EvalResults:
        """Evaluate a list of generated RNA backbone structure files.

        Accepts `.cif` (preferred) or `.pdb`, same as `ProteinEvalPipeline`
        — NA-MPNN and USalign (TMalign/qTMclust) both dispatch on the file's
        own suffix; see `pipeline/loaders.py::load_rna`.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        agg: dict[str, float] = {}
        metadata: dict[str, Any] = {"metric_status": {}}
        validity_aggregate, per_sample, designed_seqs = compute_validity_detailed(
            samples,
            structure_predictor=self.structure_predictor,
            inverse_folder=self.inverse_folder,
            output_dir=self.output_dir / "validity",
            sc_tm_threshold=self.sc_tm_threshold,
            tm_align_bin=self.tm_align_bin,
            sc_atom=self.sc_atom,
        )
        agg.update(validity_aggregate)
        failed_backbones = int(validity_aggregate["failed_backbones"])
        metadata["metric_status"]["rna_self_consistency"] = {
            "status": (
                "complete"
                if failed_backbones == 0
                else "failed"
                if validity_aggregate["evaluated_backbones"] == 0
                else "partial"
            ),
            "evaluated_backbones": int(validity_aggregate["evaluated_backbones"]),
            "failed_backbones": failed_backbones,
        }

        if self.structure_searcher is not None and self.target_db is not None:
            _run_metric(
                agg,
                metadata,
                "rna_novelty",
                compute_novelty,
                samples,
                structure_searcher=self.structure_searcher,
                target_db=self.target_db,
                output_dir=self.output_dir / "novelty",
                tm_threshold=self.novelty_tm_threshold,
            )
        else:
            metadata["metric_status"]["rna_novelty"] = {
                "status": "unavailable",
                "reason": (
                    "target_db is not configured"
                    if self.target_db is None
                    else "RNA structure searcher is not configured"
                ),
            }

        if self.structure_searcher is not None:
            _run_metric(
                agg,
                metadata,
                "rna_structural_diversity",
                compute_structural_diversity,
                samples,
                structure_searcher=self.structure_searcher,
                output_dir=self.output_dir / "struct_diversity",
                min_tm=self.sc_tm_threshold,
            )
        else:
            metadata["metric_status"]["rna_structural_diversity"] = {
                "status": "unavailable",
                "reason": "RNA structure searcher is not configured",
            }

        if designed_seqs:
            _run_metric(
                agg,
                metadata,
                "rna_sequence_diversity",
                compute_sequence_diversity,
                designed_seqs,
                output_dir=self.output_dir / "seq_diversity",
                mmseqs_bin=self.mmseqs_bin,
            )
        else:
            metadata["metric_status"]["rna_sequence_diversity"] = {
                "status": "unavailable",
                "reason": "No inverse-designed RNA sequences were available",
            }

        return EvalResults(
            per_sample=per_sample,
            aggregate=agg,
            metadata=metadata,
        )

    def aggregate(self, per_sample: list[dict[str, Any]]) -> dict[str, float]:
        paths = [Path(d["structure"]) for d in per_sample]
        return self.run(paths).aggregate


def _run_metric(
    aggregate: dict[str, float],
    metadata: dict[str, Any],
    metric_name: str,
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Run an RNA metric while retaining explicit success/failure metadata."""
    try:
        aggregate.update(function(*args, **kwargs))
        metadata["metric_status"][metric_name] = {"status": "complete"}
    except Exception as exc:
        logger.exception("%s failed", metric_name)
        metadata["metric_status"][metric_name] = {
            "status": "failed",
            "error": repr(exc),
        }
