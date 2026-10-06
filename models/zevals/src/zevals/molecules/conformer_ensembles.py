"""Metrics for conditional molecular conformer-ensemble generation.

The GEOM/GeoMol convention compares the generated and reference conformer
ensembles separately for each molecule.  Coverage (COV) is the fraction of
members whose nearest conformer in the other ensemble is within an RMSD
threshold, while Matching (MAT, also called AMR) is the corresponding mean
nearest-neighbour RMSD.  Recall uses reference conformers as the query set;
precision swaps the two ensembles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zevals.molecules.structure_prediction import (
    _heavy_atom_best_rmsd,
)
from zevals.molecules.structure_prediction import (
    _validate_target_graphs as _validate_single_reference_graphs,
)

CONFORMER_ENSEMBLE_KEYS = (
    "conformer_cov_recall_mean",
    "conformer_cov_recall_median",
    "conformer_mat_recall_mean",
    "conformer_mat_recall_median",
    "conformer_cov_precision_mean",
    "conformer_cov_precision_median",
    "conformer_mat_precision_mean",
    "conformer_mat_precision_median",
    "conformer_candidate_count_mean",
    "conformer_evaluated_candidate_count_mean",
    "conformer_reference_count_mean",
)


def _select_benchmark_candidates(
    generated: Sequence[Any],
    reference: Sequence[Any],
    *,
    generated_per_reference: int | None,
    target_id: str,
) -> list[Any]:
    """Select a deterministic GEOM-style candidate budget for one target."""
    if generated_per_reference is None:
        return list(generated)
    if generated_per_reference <= 0:
        raise ValueError("generated_per_reference must be positive or None.")
    required = generated_per_reference * len(reference)
    if len(generated) < required:
        raise ValueError(
            f"{target_id}: benchmark requires {required} generated conformers "
            f"({generated_per_reference} per reference), got {len(generated)}."
        )
    return list(generated[:required])


def _validate_target_graphs(
    generated: Sequence[Any], reference: Sequence[Any], *, target_id: str
) -> None:
    """Require one consistent molecular graph across a target's ensembles."""
    if not reference:
        raise ValueError(f"{target_id}: reference ensemble must be non-empty.")
    _validate_single_reference_graphs(generated, reference[0], target_id=target_id)
    _validate_single_reference_graphs(reference, reference[0], target_id=target_id)


def _summarize_distance_matrix(
    distances: np.ndarray,
    *,
    rmsd_threshold: float,
) -> dict[str, float]:
    """Compute one target's COV/MAT precision and recall from ``[G, R]`` RMSD."""
    if distances.ndim != 2 or 0 in distances.shape:
        raise ValueError("distances must be a non-empty [generated, reference] array.")

    generated_minima = distances.min(axis=1)
    reference_minima = distances.min(axis=0)
    return {
        "cov_recall": float(np.mean(reference_minima <= rmsd_threshold)),
        "mat_recall": float(np.mean(reference_minima)),
        "cov_precision": float(np.mean(generated_minima <= rmsd_threshold)),
        "mat_precision": float(np.mean(generated_minima)),
    }


def compute_conformer_ensemble_metrics(
    generated_by_target: Mapping[str, Sequence[Any]],
    reference_by_target: Mapping[str, Sequence[Any]],
    *,
    rmsd_threshold: float,
    generated_per_reference: int | None = 2,
) -> dict[str, float]:
    """Compute GEOM-style COV/MAT precision and recall across conformer ensembles.

    Args:
        generated_by_target: Target ID to generated RDKit conformers. Each target
            must contain at least one generated conformer.
        reference_by_target: The same target IDs mapped to reference RDKit
            conformers. Each target must contain at least one conformer.
        rmsd_threshold: Coverage threshold in Å. Use the benchmark's documented
            value, e.g. 0.5 for GEOM-QM9 and 1.25 for the original GEOM-Drugs
            protocol.
        generated_per_reference: Candidate budget multiplier. The GEOM/GeoDiff
            benchmark uses two generated conformers per reference conformer.
            Extra candidates are deterministically ignored; undersized target
            ensembles raise. Set to ``None`` for non-benchmark diagnostics.

    Returns:
        Macro mean and median COV/MAT precision and recall. COV values are
        fractions in ``[0, 1]``; MAT values are in Å.
    """
    if rmsd_threshold <= 0:
        raise ValueError("rmsd_threshold must be positive.")
    generated_ids = set(generated_by_target)
    reference_ids = set(reference_by_target)
    if generated_ids != reference_ids:
        raise ValueError(
            "Generated and reference target IDs must match; "
            f"missing generated={sorted(reference_ids - generated_ids)}, "
            f"missing reference={sorted(generated_ids - reference_ids)}."
        )
    if not generated_ids:
        raise ValueError("At least one target conformer ensemble is required.")

    per_target: list[dict[str, float]] = []
    candidate_counts: list[int] = []
    evaluated_candidate_counts: list[int] = []
    reference_counts: list[int] = []
    for target_id in sorted(generated_ids):
        generated = generated_by_target[target_id]
        reference = reference_by_target[target_id]
        if not generated or not reference:
            raise ValueError(
                f"{target_id}: both conformer ensembles must be non-empty."
            )
        evaluated_generated = _select_benchmark_candidates(
            generated,
            reference,
            generated_per_reference=generated_per_reference,
            target_id=target_id,
        )
        _validate_target_graphs(
            evaluated_generated,
            reference,
            target_id=target_id,
        )
        candidate_counts.append(len(generated))
        evaluated_candidate_counts.append(len(evaluated_generated))
        reference_counts.append(len(reference))
        distances = np.asarray(
            [
                [
                    _heavy_atom_best_rmsd(generated_mol, reference_mol)
                    for reference_mol in reference
                ]
                for generated_mol in evaluated_generated
            ],
            dtype=np.float64,
        )
        per_target.append(
            _summarize_distance_matrix(distances, rmsd_threshold=rmsd_threshold)
        )

    metrics = {
        f"conformer_{metric_name}_{statistic}": float(
            getattr(np, statistic)([metrics[metric_name] for metrics in per_target])
        )
        for metric_name in (
            "cov_recall",
            "mat_recall",
            "cov_precision",
            "mat_precision",
        )
        for statistic in ("mean", "median")
    }
    metrics.update(
        {
            "conformer_candidate_count_mean": float(np.mean(candidate_counts)),
            "conformer_evaluated_candidate_count_mean": float(
                np.mean(evaluated_candidate_counts)
            ),
            "conformer_reference_count_mean": float(np.mean(reference_counts)),
        }
    )
    return metrics


__all__ = ["CONFORMER_ENSEMBLE_KEYS", "compute_conformer_ensemble_metrics"]
