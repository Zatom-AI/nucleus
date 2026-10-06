"""Molecule evaluation pipeline.

Evaluates a list of generated 3D molecular structures (RDKit `Chem.Mol`
objects — see `pipeline/loaders.py::load_molecules`, which preserves real
conformers from `.sdf` input, Zatom's native output format) against a
reference training set. Mirrors the Zatom-2/nucleus molecule-generation
evaluator: validity + uniqueness + diversity + novelty (2D, graph-level) plus
PoseBusters + UFF force-norm (3D, conformer-level).

Hydra config (molecules.yaml):
  _target_: zevals.pipeline.molecules.MoleculeEvalPipeline
  reference_fingerprint_path: /path/to/train_fpsim2.h5
  fingerprint_radius: 2
  fingerprint_bits:  2048
  compute_posebusters: true
  compute_uff_force_norm: true
  compute_uma_forces: false          # optional; needs the `uma` extra + gated weights
  uma_checkpoint: uma-s-1p2
  uma_task: omol
  uma_device: null
  uma_batch_size: 8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zevals.molecules.conformer_ensembles import compute_conformer_ensemble_metrics
from zevals.molecules.conformer_quality import compute_uff_force_norm_metrics
from zevals.molecules.diversity import compute_diversity
from zevals.molecules.novelty import compute_novelty
from zevals.molecules.posebusters import POSEBUSTERS_KEYS, compute_posebusters_metrics
from zevals.molecules.uma_forces import (
    UMA_MOLECULE_TASK,
    compute_uma_force_metrics,
)
from zevals.molecules.uniqueness import compute_uniqueness
from zevals.molecules.validity import compute_validity
from zevals.pipeline.base import EvalPipeline, EvalResults, safe_update
from zevals.pipeline.loaders import MoleculeConformerEnsembles

logger = logging.getLogger(__name__)


class MoleculeEvalPipeline(EvalPipeline):
    """End-to-end evaluation for unconditional 3D small-molecule generation."""

    def __init__(
        self,
        reference_smiles_path: Path | str | None = None,
        reference_fingerprint_path: Path | str | None = None,
        novelty_n_threads: int = 1,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        compute_posebusters: bool = True,
        compute_uff_force_norm: bool = True,
        compute_conformer_ensemble_metrics: bool = False,
        conformer_rmsd_threshold: float | None = None,
        conformer_generated_per_reference: int | None = 2,
        compute_uma_forces: bool = False,
        uma_checkpoint: str = "uma-s-1p2",
        uma_task: str = UMA_MOLECULE_TASK,
        uma_device: str | None = None,
        uma_batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.reference_smiles: list[str] = []
        if reference_smiles_path is not None:
            self.reference_smiles = Path(reference_smiles_path).read_text().splitlines()
        self.reference_fingerprint_path = (
            Path(reference_fingerprint_path)
            if reference_fingerprint_path is not None
            else None
        )
        self.novelty_n_threads = novelty_n_threads
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.compute_posebusters = compute_posebusters
        self.compute_uff_force_norm = compute_uff_force_norm
        self.compute_conformer_ensemble_metrics = compute_conformer_ensemble_metrics
        self.conformer_rmsd_threshold = conformer_rmsd_threshold
        self.conformer_generated_per_reference = conformer_generated_per_reference
        # UMA is opt-in: it needs the `uma` extra (fairchem-core) and gated
        # Hugging Face weights, so it stays off unless a config asks for it.
        self.compute_uma_forces = compute_uma_forces
        self.uma_checkpoint = uma_checkpoint
        self.uma_task = uma_task
        self.uma_device = uma_device
        self.uma_batch_size = uma_batch_size

    def run(self, samples: list[Any]) -> EvalResults:
        """Evaluate a list of generated RDKit `Chem.Mol` objects.

        Args:
            samples: Generated molecules (one RDKit `Chem.Mol` per sample,
                with or without a 3D conformer — see `pipeline/loaders.py`).

        Returns:
            EvalResults with per-sample validity flags and dataset-level metrics.
        """
        ensemble_samples = (
            samples if isinstance(samples, MoleculeConformerEnsembles) else None
        )
        molecules = ensemble_samples.molecules if ensemble_samples else samples
        per_sample = [{"mol": m} for m in molecules]
        agg: dict[str, float] = {}
        total_samples = len(molecules)

        # Validity is a prerequisite for every downstream metric (they all
        # operate on `valid_smiles`/`valid_mols`) — if it fails, there's
        # nothing meaningful left to compute, so bail out early rather than
        # trying (and failing again) on empty input.
        try:
            validity_metrics = compute_validity(molecules)
            valid_mols = validity_metrics["valid_mols"]
            valid_smiles = validity_metrics["valid_smiles"]
            # `valid_mols`/`valid_smiles` are full per-sample lists needed by
            # the downstream metric functions below, not scalar metrics —
            # keep them out of the WandB-loggable aggregate dict.
            agg.update(
                {
                    k: v
                    for k, v in validity_metrics.items()
                    if k not in ("valid_mols", "valid_smiles")
                }
            )
        except Exception:
            logger.warning("Skipping molecule metrics due to an error", exc_info=True)
            return EvalResults(per_sample=per_sample, aggregate=agg)

        safe_update(agg, "uniqueness", compute_uniqueness, valid_smiles)
        num_unique = len(set(valid_smiles))
        safe_update(
            agg,
            "diversity",
            compute_diversity,
            valid_smiles,
            self.fingerprint_radius,
            self.fingerprint_bits,
        )

        if self.reference_fingerprint_path is not None:
            safe_update(
                agg,
                "novelty",
                compute_novelty,
                valid_smiles,
                reference_fp=self.reference_fingerprint_path,
                n_threads=self.novelty_n_threads,
            )
        elif self.reference_smiles:
            try:
                novelty_results = compute_novelty(valid_smiles, self.reference_smiles)
                # Zatom-2/nucleus convention: normalize by the unique (not
                # valid) count, unlike compute_novelty's own novel_rate.
                agg["novel_rate"] = (
                    novelty_results["num_novel"] / num_unique if num_unique else 0.0
                )
            except Exception:
                logger.warning(
                    "Skipping novelty metrics due to an error", exc_info=True
                )
        else:
            agg["novel_rate"] = 0.0

        if self.compute_posebusters:
            safe_update(agg, "PoseBusters", compute_posebusters_metrics, valid_mols)
        else:
            agg.update(dict.fromkeys(POSEBUSTERS_KEYS, 0.0))

        if self.compute_uff_force_norm:
            safe_update(
                agg,
                "UFF force norm",
                compute_uff_force_norm_metrics,
                valid_mols,
                total_samples,
            )
        else:
            agg.update(
                {
                    "uff_median_force_norm": 0.0,
                    "uff_median_force_norm_success_rate": 0.0,
                }
            )

        if self.compute_conformer_ensemble_metrics and ensemble_samples is not None:
            if self.conformer_rmsd_threshold is None:
                raise ValueError(
                    "conformer_rmsd_threshold is required for conformer ensemble metrics."
                )
            safe_update(
                agg,
                "conformer ensemble",
                compute_conformer_ensemble_metrics,
                ensemble_samples.generated_by_target,
                ensemble_samples.reference_by_target,
                rmsd_threshold=self.conformer_rmsd_threshold,
                generated_per_reference=self.conformer_generated_per_reference,
            )

        if self.compute_uma_forces:
            safe_update(
                agg,
                "UMA force",
                compute_uma_force_metrics,
                valid_mols,
                total_samples,
                checkpoint=self.uma_checkpoint,
                task=self.uma_task,
                device=self.uma_device,
                batch_size=self.uma_batch_size,
            )

        return EvalResults(per_sample=per_sample, aggregate=agg)

    def aggregate(self, per_sample: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate per-sample metrics into dataset-level metrics."""
        mols = [d["mol"] for d in per_sample]
        return self.run(mols).aggregate
