"""Crystal structure-prediction metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher  # type: ignore[import]
from tqdm.auto import tqdm

DEFAULT_TOP_K = (1,)

MATERIAL_STRUCTURE_PREDICTION_KEYS = (
    "material_match_rate",
    "material_normalized_rms_dist_mean",
    "material_normalized_rms_dist_median",
    "material_match_rate_top1",
    "material_normalized_rms_dist_mean_top1",
    "material_normalized_rms_dist_median_top1",
    "material_matched_target_count_top1",
    "material_matched_target_count",
    "material_unmatched_target_count",
    "material_candidate_count",
    "material_candidate_count_min",
    "material_invalid_candidate_count",
    "material_matcher_error_count",
)


def _normalize_top_k(top_k: Sequence[int] | int | None) -> tuple[int, ...]:
    """Return sorted, deduplicated, positive candidate budgets."""
    if top_k is None:
        return ()
    budgets = (top_k,) if isinstance(top_k, int) else tuple(top_k)
    for budget in budgets:
        if int(budget) <= 0:
            raise ValueError(f"top_k budgets must be positive, got {budget}.")
    return tuple(sorted({int(budget) for budget in budgets}))


def compute_material_structure_prediction_metrics(
    generated_by_target: Mapping[str, Sequence[Any]],
    reference_by_target: Mapping[str, Sequence[Any]],
    *,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    scale: bool = True,
    top_k: Sequence[int] | int | None = DEFAULT_TOP_K,
    show_progress: bool = False,
) -> dict[str, float]:
    """Compute target-wise Match Rate at the full and fixed candidate budgets.

    Each target must have exactly one ground-truth structure. A target counts
    as matched when any evaluated candidate matches that reference, and its
    RMS is the smallest matched value. The unsuffixed keys
    use every supplied candidate, which makes them sensitive to how many
    samples the caller happened to draw. Each budget in ``top_k`` additionally
    reports the same metrics over only the first k candidates, so
    ``material_match_rate_top1`` is the one-shot accuracy that CDVAE/DiffCSP
    report alongside their best-of-20 numbers. Candidate order is the caller's
    generation order; truncation is deterministic and never reorders.

    Every target must provide at least ``k`` candidates so that metrics with the
    same name are directly comparable across runs.

    All RMS values are pymatgen's normalized displacement, not Angstrom RMSD.
    Set ``show_progress=True`` to display completed target comparisons.
    """
    import numpy as np

    generated_ids = set(generated_by_target)
    reference_ids = set(reference_by_target)
    if generated_ids != reference_ids:
        raise ValueError(
            "Generated and reference target IDs must match; "
            f"missing generated={sorted(reference_ids - generated_ids)}, "
            f"missing reference={sorted(generated_ids - reference_ids)}."
        )
    if not generated_ids:
        raise ValueError("At least one material target is required.")
    budgets = _normalize_top_k(top_k)

    matcher = StructureMatcher(
        ltol=float(ltol),
        stol=float(stol),
        angle_tol=float(angle_tol),
        scale=bool(scale),
    )
    best_target_rms_distances: list[float] = []
    best_budget_rms_distances: dict[int, list[float]] = {
        budget: [] for budget in budgets
    }
    candidate_count = 0
    candidate_counts_per_target: list[int] = []
    invalid_candidate_count = 0
    matcher_error_count = 0
    for target_id in tqdm(
        sorted(generated_ids),
        desc="Matching material structures",
        unit="target",
        disable=not show_progress,
    ):
        candidates = generated_by_target[target_id]
        references = reference_by_target[target_id]
        if not candidates:
            raise ValueError(f"{target_id}: generated candidates must be non-empty.")
        if len(references) != 1 or references[0] is None:
            raise ValueError(
                f"{target_id}: exactly one valid reference structure is required."
            )
        if budgets and len(candidates) < max(budgets):
            raise ValueError(
                f"{target_id}: top-{max(budgets)} requires at least "
                f"{max(budgets)} generated candidates, got {len(candidates)}."
            )
        reference = references[0]

        target_rms_distances: list[float] = []
        # Keep one RMS per candidate so `top_k` can truncate in generation order.
        candidate_best_distances: list[float | None] = []
        for candidate in candidates:
            candidate_count += 1
            if candidate is None:
                invalid_candidate_count += 1
                candidate_best_distances.append(None)
                continue
            try:
                rms_distance = matcher.get_rms_dist(candidate, reference)
            except Exception:
                matcher_error_count += 1
                invalid_candidate_count += 1
                candidate_best_distances.append(None)
                continue
            normalized_rms = (
                float(rms_distance[0])
                if rms_distance is not None and np.isfinite(rms_distance[0])
                else None
            )
            candidate_best_distances.append(normalized_rms)
            if normalized_rms is not None:
                target_rms_distances.append(normalized_rms)

        candidate_counts_per_target.append(len(candidates))
        if target_rms_distances:
            best_target_rms_distances.append(min(target_rms_distances))
        for budget in budgets:
            budget_distances = [
                distance
                for distance in candidate_best_distances[:budget]
                if distance is not None
            ]
            if budget_distances:
                best_budget_rms_distances[budget].append(min(budget_distances))

    if best_target_rms_distances:
        rms_mean = float(np.mean(best_target_rms_distances))
        rms_median = float(np.median(best_target_rms_distances))
    else:
        rms_mean = float("nan")
        rms_median = float("nan")

    matched_target_count = len(best_target_rms_distances)
    metrics = {
        "material_match_rate": matched_target_count / len(generated_ids),
        "material_normalized_rms_dist_mean": rms_mean,
        "material_normalized_rms_dist_median": rms_median,
        "material_matched_target_count": float(matched_target_count),
        "material_unmatched_target_count": float(
            len(generated_ids) - matched_target_count
        ),
        "material_candidate_count": float(candidate_count),
        "material_candidate_count_min": float(min(candidate_counts_per_target)),
        "material_invalid_candidate_count": float(invalid_candidate_count),
        "material_matcher_error_count": float(matcher_error_count),
    }
    for budget in budgets:
        matched = best_budget_rms_distances[budget]
        metrics[f"material_match_rate_top{budget}"] = len(matched) / len(generated_ids)
        metrics[f"material_matched_target_count_top{budget}"] = float(len(matched))
        metrics[f"material_normalized_rms_dist_mean_top{budget}"] = (
            float(np.mean(matched)) if matched else float("nan")
        )
        metrics[f"material_normalized_rms_dist_median_top{budget}"] = (
            float(np.median(matched)) if matched else float("nan")
        )
    return metrics


def compute_material_match_rate(
    generated_by_target: Mapping[str, Sequence[Any]],
    reference_by_target: Mapping[str, Sequence[Any]],
    *,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    scale: bool = True,
) -> dict[str, float]:
    """Compute crystal structure-prediction Match Rate (MR) only."""
    metrics = compute_material_structure_prediction_metrics(
        generated_by_target,
        reference_by_target,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        scale=scale,
        top_k=None,
    )
    return {"material_match_rate": metrics["material_match_rate"]}


__all__ = [
    "DEFAULT_TOP_K",
    "MATERIAL_STRUCTURE_PREDICTION_KEYS",
    "compute_material_match_rate",
    "compute_material_structure_prediction_metrics",
]
