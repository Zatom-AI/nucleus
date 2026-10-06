"""Materials evaluation pipeline.

Evaluates generated crystal structures using validity checks, novelty against
a reference database, internal diversity, and (if installed) the full
LeMatGenBench suite.

Hydra config (materials.yaml):
  _target_: zevals.pipeline.materials.MaterialEvalPipeline
  reference_structures: null
  run_lematgenbench: true
  lematgenbench_repo_path: /path/to/pinned/lemat-genbench
  lematgenbench_output_dir: /path/to/evaluation/artifacts
  compute_uma_forces: false          # optional; needs the `uma` extra + gated weights
  uma_checkpoint: uma-s-1p2
  uma_task: omat
  uma_device: null
  uma_batch_size: 8
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pymatgen.core import Structure  # type: ignore[import]

from zevals.materials.crystal import crystal_from_structure
from zevals.materials.diversity import compute_diversity
from zevals.materials.lematgenbench import (
    LEMAT_GENBENCH_COMMIT,
    LEMAT_GENBENCH_CONFIG,
    LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256,
    LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256,
    LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256,
    LEMAT_GENBENCH_UV_LOCK_SHA256,
    run_lematgenbench,
)
from zevals.materials.novelty import compute_novelty
from zevals.materials.structure_prediction import (
    compute_material_structure_prediction_metrics,
)
from zevals.materials.uma_forces import (
    UMA_MATERIAL_TASK,
    compute_uma_force_metrics,
)
from zevals.materials.uniqueness import compute_uniqueness
from zevals.materials.validity import compute_validity
from zevals.pipeline.base import EvalPipeline, EvalResults, safe_update
from zevals.pipeline.loaders import MaterialStructureCandidates

logger = logging.getLogger(__name__)


class MaterialEvalPipeline(EvalPipeline):
    """End-to-end evaluation for crystal and periodic material generation."""

    def __init__(
        self,
        reference_structures: list[Structure] | None = None,
        run_lematgenbench: bool = False,
        lematgenbench_repo_path: str | Path | None = None,
        lematgenbench_output_dir: str | Path | None = None,
        lematgenbench_run_name: str = "zevals-materials",
        lematgenbench_config_name: str = LEMAT_GENBENCH_CONFIG,
        lematgenbench_families: Sequence[str] | None = None,
        lematgenbench_fingerprint_method: str = "structure-matcher",
        lematgenbench_runner_command: Sequence[str] = (
            "uv",
            "run",
            "--frozen",
            "python",
        ),
        lematgenbench_expected_commit: str | None = LEMAT_GENBENCH_COMMIT,
        lematgenbench_expected_lock_sha256: str | None = (
            LEMAT_GENBENCH_UV_LOCK_SHA256
        ),
        lematgenbench_verify_mace_checkpoint: bool = True,
        lematgenbench_expected_mace_checkpoint_sha256: str | None = (
            LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256
        ),
        lematgenbench_verify_orb_checkpoint: bool = True,
        lematgenbench_expected_orb_checkpoint_sha256: str | None = (
            LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256
        ),
        lematgenbench_verify_uma_checkpoint: bool = True,
        lematgenbench_expected_uma_checkpoint_sha256: str | None = (
            LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256
        ),
        lematgenbench_enforce_reference_revisions: bool = True,
        lematgenbench_runtime_device: str = "cuda",
        lematgenbench_cuda_visible_devices: str | None = None,
        lematgenbench_parse_failures_as_invalid: bool = False,
        lematgenbench_require_complete_oracle_coverage: bool = True,
        lematgenbench_require_clean_repo: bool = True,
        lematgenbench_timeout_seconds: float | None = None,
        lematgenbench_fail_on_family_error: bool = True,
        lematgenbench_resume: bool = False,
        lematgenbench_strict: bool = True,
        compute_structure_prediction_metrics: bool = False,
        structure_matcher_ltol: float = 0.3,
        structure_matcher_stol: float = 0.5,
        structure_matcher_angle_tol: float = 10.0,
        structure_matcher_scale: bool = True,
        structure_prediction_top_k: Sequence[int] = (1,),
        compute_uma_forces: bool = False,
        uma_checkpoint: str = "uma-s-1p2",
        uma_task: str = UMA_MATERIAL_TASK,
        uma_device: str | None = None,
        uma_batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        self.reference_structures = reference_structures
        self._run_lematgenbench = run_lematgenbench
        self.lematgenbench_repo_path = lematgenbench_repo_path
        self.lematgenbench_output_dir = lematgenbench_output_dir
        self.lematgenbench_run_name = lematgenbench_run_name
        self.lematgenbench_config_name = lematgenbench_config_name
        self.lematgenbench_families = (
            tuple(lematgenbench_families)
            if lematgenbench_families is not None
            else None
        )
        self.lematgenbench_fingerprint_method = lematgenbench_fingerprint_method
        self.lematgenbench_runner_command = tuple(lematgenbench_runner_command)
        self.lematgenbench_expected_commit = lematgenbench_expected_commit
        self.lematgenbench_expected_lock_sha256 = lematgenbench_expected_lock_sha256
        self.lematgenbench_verify_mace_checkpoint = lematgenbench_verify_mace_checkpoint
        self.lematgenbench_expected_mace_checkpoint_sha256 = (
            lematgenbench_expected_mace_checkpoint_sha256
        )
        self.lematgenbench_verify_orb_checkpoint = lematgenbench_verify_orb_checkpoint
        self.lematgenbench_expected_orb_checkpoint_sha256 = (
            lematgenbench_expected_orb_checkpoint_sha256
        )
        self.lematgenbench_verify_uma_checkpoint = lematgenbench_verify_uma_checkpoint
        self.lematgenbench_expected_uma_checkpoint_sha256 = (
            lematgenbench_expected_uma_checkpoint_sha256
        )
        self.lematgenbench_enforce_reference_revisions = (
            lematgenbench_enforce_reference_revisions
        )
        self.lematgenbench_runtime_device = lematgenbench_runtime_device
        self.lematgenbench_cuda_visible_devices = lematgenbench_cuda_visible_devices
        self.lematgenbench_parse_failures_as_invalid = (
            lematgenbench_parse_failures_as_invalid
        )
        self.lematgenbench_require_complete_oracle_coverage = (
            lematgenbench_require_complete_oracle_coverage
        )
        self.lematgenbench_require_clean_repo = lematgenbench_require_clean_repo
        self.lematgenbench_timeout_seconds = lematgenbench_timeout_seconds
        self.lematgenbench_fail_on_family_error = lematgenbench_fail_on_family_error
        self.lematgenbench_resume = lematgenbench_resume
        self.lematgenbench_strict = lematgenbench_strict
        self.compute_structure_prediction_metrics = compute_structure_prediction_metrics
        self.structure_matcher_ltol = structure_matcher_ltol
        self.structure_matcher_stol = structure_matcher_stol
        self.structure_matcher_angle_tol = structure_matcher_angle_tol
        self.structure_matcher_scale = structure_matcher_scale
        self.structure_prediction_top_k = tuple(
            int(budget) for budget in structure_prediction_top_k
        )
        # UMA is opt-in: it needs the `uma` extra (fairchem-core) and gated
        # Hugging Face weights, so it stays off unless a config asks for it.
        self.compute_uma_forces = compute_uma_forces
        self.uma_checkpoint = uma_checkpoint
        self.uma_task = uma_task
        self.uma_device = uma_device
        self.uma_batch_size = uma_batch_size

    def run(
        self, samples: list[Structure] | MaterialStructureCandidates
    ) -> EvalResults:
        """Evaluate a list of generated PyMatGen structures.

        Args:
            samples: PyMatGen ``Structure`` objects.

        Returns:
            EvalResults with validity, novelty, and diversity metrics.
        """
        candidate_samples = (
            samples if isinstance(samples, MaterialStructureCandidates) else None
        )
        structures = candidate_samples.structures if candidate_samples else samples
        agg: dict[str, float] = {}
        metadata: dict[str, Any] = {}
        per_sample = [
            {"sample_idx": i, "structure": structure}
            for i, structure in enumerate(structures)
        ]

        # Validity is a prerequisite for every downstream metric (they all
        # operate on `valid_structures`) — if it fails, bail out early.
        try:
            crystals = [
                crystal_from_structure(structure, i)
                for i, structure in enumerate(structures)
            ]
            agg.update(compute_validity(crystals))
            valid_structures = [
                getattr(c, "structure", None)
                for c in crystals
                if getattr(c, "valid", False)
            ]
            valid_structures = [s for s in valid_structures if s is not None]
        except Exception:
            logger.warning("Skipping material metrics due to an error", exc_info=True)
            return EvalResults(per_sample=per_sample, aggregate=agg)

        safe_update(agg, "diversity", compute_diversity, valid_structures)

        unique_structures: list[Structure] = []
        try:
            uniqueness = compute_uniqueness(
                valid_structures, return_unique_structures=True
            )
            agg["unique_rate"] = float(uniqueness["unique_rate"])
            unique_structures = uniqueness.get("unique_structures", [])
        except Exception:
            logger.warning("Skipping uniqueness/novelty due to an error", exc_info=True)

        if self.reference_structures is not None:
            safe_update(
                agg,
                "novelty",
                compute_novelty,
                unique_structures,
                self.reference_structures,
            )

        if self._run_lematgenbench:
            try:
                if self.lematgenbench_repo_path is None:
                    raise ValueError(
                        "lematgenbench_repo_path is required when "
                        "run_lematgenbench=True"
                    )
                lematgenbench_result = run_lematgenbench(
                    structures,
                    repo_path=self.lematgenbench_repo_path,
                    run_name=self.lematgenbench_run_name,
                    output_dir=self.lematgenbench_output_dir,
                    config_name=self.lematgenbench_config_name,
                    families=self.lematgenbench_families,
                    fingerprint_method=self.lematgenbench_fingerprint_method,
                    runner_command=self.lematgenbench_runner_command,
                    expected_commit=self.lematgenbench_expected_commit,
                    expected_lock_sha256=self.lematgenbench_expected_lock_sha256,
                    verify_mace_checkpoint=self.lematgenbench_verify_mace_checkpoint,
                    expected_mace_checkpoint_sha256=(
                        self.lematgenbench_expected_mace_checkpoint_sha256
                    ),
                    verify_orb_checkpoint=self.lematgenbench_verify_orb_checkpoint,
                    expected_orb_checkpoint_sha256=(
                        self.lematgenbench_expected_orb_checkpoint_sha256
                    ),
                    verify_uma_checkpoint=self.lematgenbench_verify_uma_checkpoint,
                    expected_uma_checkpoint_sha256=(
                        self.lematgenbench_expected_uma_checkpoint_sha256
                    ),
                    enforce_reference_revisions=(
                        self.lematgenbench_enforce_reference_revisions
                    ),
                    runtime_device=self.lematgenbench_runtime_device,
                    cuda_visible_devices=self.lematgenbench_cuda_visible_devices,
                    parse_failures_as_invalid=(
                        self.lematgenbench_parse_failures_as_invalid
                    ),
                    require_complete_oracle_coverage=(
                        self.lematgenbench_require_complete_oracle_coverage
                    ),
                    require_clean_repo=self.lematgenbench_require_clean_repo,
                    timeout_seconds=self.lematgenbench_timeout_seconds,
                    fail_on_family_error=self.lematgenbench_fail_on_family_error,
                    resume=self.lematgenbench_resume,
                )
                agg.update(lematgenbench_result.metrics)
                metadata["lematgenbench"] = lematgenbench_result.provenance
            except Exception:
                logger.warning(
                    "Skipping LeMat-GenBench metrics due to an error", exc_info=True
                )
                if self.lematgenbench_strict:
                    raise

        if self.compute_structure_prediction_metrics and candidate_samples is not None:
            safe_update(
                agg,
                "material structure prediction",
                compute_material_structure_prediction_metrics,
                candidate_samples.generated_by_target,
                candidate_samples.reference_by_target,
                ltol=self.structure_matcher_ltol,
                stol=self.structure_matcher_stol,
                angle_tol=self.structure_matcher_angle_tol,
                scale=self.structure_matcher_scale,
                top_k=self.structure_prediction_top_k,
            )

        if self.compute_uma_forces:
            safe_update(
                agg,
                "UMA force",
                compute_uma_force_metrics,
                valid_structures,
                len(structures),
                checkpoint=self.uma_checkpoint,
                task=self.uma_task,
                device=self.uma_device,
                batch_size=self.uma_batch_size,
            )

        return EvalResults(
            per_sample=per_sample,
            aggregate=agg,
            metadata=metadata,
        )

    def aggregate(self, per_sample: list[dict[str, Any]]) -> dict[str, float]:
        structures = [d["structure"] for d in per_sample]
        return self.run(structures).aggregate
