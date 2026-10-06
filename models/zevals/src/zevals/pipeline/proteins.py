"""Protein evaluation pipeline.

Evaluates generated protein backbones on designability, novelty, diversity,
secondary structure content, and (optionally) co-designability. Diversity
and novelty are scoped to the designable subset of samples; secondary
structure content is reported over the full generated set. Oracles are
fully config-driven.

Hydra config (proteins.yaml):
  _target_: zevals.pipeline.proteins.ProteinEvalPipeline
  oracle:
    structure_predictor:
      _target_: zevals.oracles.structure_predictors.esmfold.ESMFoldPredictor
      model_name: facebook/esmfold_v1
      num_recycles: 4
    inverse_folder:
      _target_: zevals.oracles.inverse_folders.protein_mpnn.ProteinMPNNFolder
      weights_dir: /path/to/vanilla_model_weights
      num_sequences: 8
      temperature: 0.1
  structure_searcher:
    _target_: zevals.oracles.structure_search.foldseek.FoldSeekSearcher
  sc_rmsd_threshold: 2.0
  plddt_threshold: null
  run_co_designability: false
  target_db: null
  output_dir: /tmp/zevals/proteins
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
from zevals.pipeline.base import EvalPipeline, EvalResults, safe_update
from zevals.proteins.co_designability import compute_co_designability
from zevals.proteins.designability import compute_designability
from zevals.proteins.diversity import (
    compute_sequence_diversity,
    compute_structural_diversity,
    compute_tmalign_complete_linkage_diversity,
)
from zevals.proteins.novelty import compute_novelty
from zevals.proteins.secondary_structure import compute_secondary_structure

logger = logging.getLogger(__name__)


class ProteinEvalPipeline(EvalPipeline):
    """End-to-end evaluation for protein backbone / sequence design."""

    def __init__(
        self,
        structure_predictor: BaseStructurePredictor,
        inverse_folder: BaseInverseFolder,
        structure_searcher: BaseStructureSearcher | None = None,
        sc_rmsd_threshold: float = 2.0,
        plddt_threshold: float | None = None,
        sc_atom: str = "CA",
        run_co_designability: bool = False,
        run_tmalign_diversity: bool = False,
        tmalign_bin: str = "TMalign",
        tmalign_diversity_query_set: str = "designable",
        target_db: Path | str | None = None,
        output_dir: Path | str = "/tmp/zevals/proteins",
        mmseqs_bin: str = "mmseqs",
        **kwargs: Any,
    ) -> None:
        self.structure_predictor = structure_predictor
        self.inverse_folder = inverse_folder
        self.structure_searcher = structure_searcher
        self.sc_rmsd_threshold = sc_rmsd_threshold
        self.plddt_threshold = plddt_threshold
        self.sc_atom = sc_atom
        self.run_co_designability = run_co_designability
        self.run_tmalign_diversity = run_tmalign_diversity
        if tmalign_diversity_query_set not in {"designable", "all"}:
            raise ValueError(
                "tmalign_diversity_query_set must be 'designable' or 'all'"
            )
        self.tmalign_diversity_query_set = tmalign_diversity_query_set
        self.tmalign_bin = tmalign_bin
        self.target_db = Path(target_db) if target_db else None
        self.output_dir = Path(output_dir)
        self.mmseqs_bin = mmseqs_bin

    def run(self, samples: list[Path]) -> EvalResults:
        """Evaluate a list of generated protein backbone structure files.

        Accepts `.cif` (preferred — richer chain-ID/residue-numbering than
        PDB's fixed-width fields) or `.pdb`; each oracle/metric dispatches on
        the file's own suffix rather than assuming one format, so a mixed
        list works too. See `pipeline/loaders.py::load_proteins` for how
        `ZevalSuite` discovers these files from a samples directory.

        Diversity and novelty are computed over the *designable* subset of
        `samples` only (per proteins/METRICS.md), not the full generated
        set — a backbone that can't be designed in the first place isn't a
        meaningful structure to ask "how novel/diverse is it?" about.
        Secondary structure content is the exception: it's reported over
        the full generated set, matching Proteína's own convention.

        Args:
            samples: Paths to generated backbone structure files (`.cif` or `.pdb`).

        Returns:
            EvalResults with designability, novelty, diversity, and
            secondary-structure-content metrics. Each metric group is
            computed independently — a failure in one (missing binary,
            oracle error, ...) is logged and simply omits that group's keys
            rather than aborting the whole call, since this is meant to run
            unattended inside a training loop.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        agg: dict[str, float] = {}

        designable_pdbs: list[Path] = []
        try:
            designability_metrics, designable_pdbs = compute_designability(
                samples,
                structure_predictor=self.structure_predictor,
                inverse_folder=self.inverse_folder,
                output_dir=self.output_dir / "designability",
                sc_rmsd_threshold=self.sc_rmsd_threshold,
                plddt_threshold=self.plddt_threshold,
                sc_atom=self.sc_atom,
            )
            agg.update(designability_metrics)
        except Exception:
            logger.warning(
                "Skipping designability metrics due to an error", exc_info=True
            )

        if self.run_co_designability:
            safe_update(
                agg,
                "co-designability",
                compute_co_designability,
                samples,
                structure_predictor=self.structure_predictor,
                inverse_folder=self.inverse_folder,
                output_dir=self.output_dir / "co_designability",
                sc_rmsd_threshold=self.sc_rmsd_threshold,
            )

        if self.structure_searcher is not None and designable_pdbs:
            safe_update(
                agg,
                "structural diversity",
                compute_structural_diversity,
                designable_pdbs,
                foldseek=self.structure_searcher,
                output_dir=self.output_dir / "struct_diversity",
            )
            if self.target_db is not None:
                safe_update(
                    agg,
                    "novelty",
                    compute_novelty,
                    designable_pdbs,
                    foldseek=self.structure_searcher,
                    target_db=self.target_db,
                    output_dir=self.output_dir / "novelty",
                )

        if self.run_tmalign_diversity:
            tmalign_pdbs = (
                samples
                if self.tmalign_diversity_query_set == "all"
                else designable_pdbs
            )
            if tmalign_pdbs:
                safe_update(
                    agg,
                    "TM-align complete-linkage diversity",
                    compute_tmalign_complete_linkage_diversity,
                    tmalign_pdbs,
                    output_dir=self.output_dir / "tmalign_diversity",
                    tmalign_bin=self.tmalign_bin,
                )

        # Collect designed sequences written during designability run, from
        # designable backbones only (same subset as structural diversity/novelty).
        designed_seqs = _collect_designed_sequences(
            self.output_dir / "designability", designable_pdbs
        )
        if designed_seqs:
            safe_update(
                agg,
                "sequence diversity",
                compute_sequence_diversity,
                designed_seqs,
                output_dir=self.output_dir / "seq_diversity",
                mmseqs_bin=self.mmseqs_bin,
            )

        safe_update(agg, "secondary structure", compute_secondary_structure, samples)

        per_sample = [{"structure": str(p)} for p in samples]
        return EvalResults(per_sample=per_sample, aggregate=agg)

    def aggregate(self, per_sample: list[dict[str, Any]]) -> dict[str, float]:
        paths = [Path(d["structure"]) for d in per_sample]
        return self.run(paths).aggregate


def _collect_designed_sequences(
    designability_dir: Path, designable_pdbs: list[Path]
) -> list[str]:
    """Gather designed sequences written during the designability run, for
    designable backbones only (`compute_designability` writes each
    backbone's candidates to `designability_dir / pdb_path.stem`)."""
    sequences = []
    for pdb_path in designable_pdbs:
        fasta_path = designability_dir / pdb_path.stem / "designed.fasta"
        if not fasta_path.exists():
            continue
        for line in fasta_path.read_text().splitlines():
            if not line.startswith(">") and line.strip():
                sequences.append(line.strip())
    return sequences
