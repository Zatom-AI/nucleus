"""Single-reference molecular structure-prediction metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_TOP_K = (1,)


def _heavy_atom_best_rmsd(generated: Any, reference: Any) -> float:
    """Return symmetry-aware heavy-atom RMSD in Angstrom."""
    from rdkit.Chem import RemoveHs, rdMolAlign

    try:
        generated = RemoveHs(generated, sanitize=False)
        reference = RemoveHs(reference, sanitize=False)
        if generated.GetNumAtoms() == 0 or reference.GetNumAtoms() == 0:
            return float("inf")
        return float(rdMolAlign.GetBestRMS(generated, reference))
    except Exception:
        return float("inf")


def _heavy_atom_graph_id(molecule: Any) -> str:
    """Return a canonical heavy-atom graph identifier."""
    from rdkit.Chem import MolToSmiles, RemoveHs

    if molecule is None:
        raise ValueError("conformer is missing")
    try:
        heavy_molecule = RemoveHs(molecule, sanitize=False)
        if heavy_molecule.GetNumAtoms() == 0:
            raise ValueError("conformer has no heavy atoms")
        return str(MolToSmiles(heavy_molecule, isomericSmiles=True))
    except Exception as exc:
        raise ValueError("could not determine the conformer's molecular graph") from exc


def _validate_target_graphs(
    generated: Sequence[Any], reference: Any, *, target_id: str
) -> None:
    """Require every candidate to have the conditioned reference graph."""
    try:
        reference_graph = _heavy_atom_graph_id(reference)
        generated_graphs = {_heavy_atom_graph_id(molecule) for molecule in generated}
    except ValueError as exc:
        raise ValueError(f"{target_id}: {exc}") from exc
    if generated_graphs != {reference_graph}:
        raise ValueError(
            f"{target_id}: generated and reference molecular graphs must match."
        )


def _normalize_top_k(top_k: Sequence[int] | int) -> tuple[int, ...]:
    """Return sorted, deduplicated, positive candidate budgets."""
    budgets = (top_k,) if isinstance(top_k, int) else tuple(top_k)
    if not budgets or any(int(budget) <= 0 for budget in budgets):
        raise ValueError("top_k budgets must be positive and non-empty.")
    return tuple(sorted({int(budget) for budget in budgets}))


def compute_molecule_structure_prediction_metrics(
    generated_by_target: Mapping[str, Sequence[Any]],
    reference_by_target: Mapping[str, Sequence[Any]],
    *,
    top_k: Sequence[int] | int = DEFAULT_TOP_K,
) -> dict[str, float]:
    """Compute best symmetry-aware heavy-atom RMSD at fixed candidate budgets.

    Each target represents one fixed molecular graph and exactly one ground-truth
    conformer. For every requested budget ``k``, only the first ``k`` generated
    conformers are considered. Every target must provide at least ``k`` candidates
    so that metrics with the same name are directly comparable across runs.
    """
    generated_ids = set(generated_by_target)
    reference_ids = set(reference_by_target)
    if generated_ids != reference_ids:
        raise ValueError(
            "Generated and reference target IDs must match; "
            f"missing generated={sorted(reference_ids - generated_ids)}, "
            f"missing reference={sorted(generated_ids - reference_ids)}."
        )
    if not generated_ids:
        raise ValueError("At least one molecular target is required.")

    budgets = _normalize_top_k(top_k)
    best_rmsd_by_budget = {budget: [] for budget in budgets}
    for target_id in sorted(generated_ids):
        generated = generated_by_target[target_id]
        references = reference_by_target[target_id]
        if not generated:
            raise ValueError(f"{target_id}: generated conformers must be non-empty.")
        if len(references) != 1:
            raise ValueError(
                f"{target_id}: exactly one reference conformer is required."
            )
        if len(generated) < max(budgets):
            raise ValueError(
                f"{target_id}: top-{max(budgets)} requires at least "
                f"{max(budgets)} generated conformers, got {len(generated)}."
            )

        reference = references[0]
        _validate_target_graphs(generated, reference, target_id=target_id)
        candidate_rmsds = [
            _heavy_atom_best_rmsd(candidate, reference) for candidate in generated
        ]
        for budget in budgets:
            best_rmsd_by_budget[budget].append(min(candidate_rmsds[:budget]))

    metrics: dict[str, float] = {}
    for budget, best_rmsds in best_rmsd_by_budget.items():
        metrics[f"molecule_best_rmsd_mean_top{budget}"] = float(np.mean(best_rmsds))
        metrics[f"molecule_best_rmsd_median_top{budget}"] = float(np.median(best_rmsds))
    return metrics


__all__ = ["DEFAULT_TOP_K", "compute_molecule_structure_prediction_metrics"]
