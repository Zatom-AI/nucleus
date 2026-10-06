"""Explicit, coverage-aware constraint scoring for conditioned RNA generation."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zevals.rna.tasks import (
    RNAChainSequence,
    RNAEvaluationSample,
    load_biomolecule_structure,
    rna_parent_base,
)

_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    ">=": operator.ge,
    ">": operator.gt,
}


def compute_constraint_satisfaction_detailed(
    samples: list[RNAEvaluationSample],
    sequences: Mapping[str, list[RNAChainSequence]],
    metric_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate manifest constraints without silently accepting unknown types.

    The manifest schema is ``{"items": [{"type": ...}, ...]}``. Supported
    types are ``sequence_length``, ``chain_count``, ``gc_fraction``, ``motif``,
    ``atom_distance``, ``residue_contact``, and ``metric``. Unknown items are
    reported as unsupported and excluded from the pass-rate denominator.
    """
    details: list[dict[str, Any]] = []
    for sample in samples:
        raw_items = _constraint_items(sample.constraints)
        item_results: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            try:
                result = _evaluate_item(
                    item,
                    sample,
                    sequences.get(sample.sample_id, []),
                    metric_rows.get(sample.sample_id, {}),
                )
                item_results.append({"index": index, **item, **result})
            except Exception as exc:
                item_results.append(
                    {
                        "index": index,
                        **item,
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
        evaluated = [item for item in item_results if item["status"] == "complete"]
        passed = sum(bool(item["passed"]) for item in evaluated)
        unsupported = sum(item["status"] == "unsupported" for item in item_results)
        failed = sum(item["status"] == "failed" for item in item_results)
        details.append(
            {
                "sample_id": sample.sample_id,
                "status": (
                    "complete"
                    if evaluated and failed == 0
                    else "partial"
                    if evaluated
                    else "unavailable"
                ),
                "constraint_count": len(item_results),
                "evaluated_constraint_count": len(evaluated),
                "passed_constraint_count": passed,
                "unsupported_constraint_count": unsupported,
                "failed_constraint_count": failed,
                "all_constraints_satisfied": (
                    passed == len(evaluated) if evaluated else None
                ),
                "constraint_pass_fraction": (
                    passed / len(evaluated) if evaluated else float("nan")
                ),
                "items": item_results,
            }
        )

    with_constraints = [row for row in details if row["constraint_count"] > 0]
    evaluated_samples = [
        row for row in with_constraints if row["evaluated_constraint_count"] > 0
    ]
    total_evaluated_items = sum(
        int(row["evaluated_constraint_count"]) for row in with_constraints
    )
    total_passed_items = sum(
        int(row["passed_constraint_count"]) for row in with_constraints
    )
    aggregate = {
        "constraint_satisfaction_rate": (
            sum(bool(row["all_constraints_satisfied"]) for row in evaluated_samples)
            / len(evaluated_samples)
            if evaluated_samples
            else float("nan")
        ),
        "individual_constraint_pass_rate": (
            total_passed_items / total_evaluated_items
            if total_evaluated_items
            else float("nan")
        ),
        "constraint_sample_coverage": (
            len(evaluated_samples) / len(with_constraints) if with_constraints else 0.0
        ),
        "constraint_evaluated_samples": float(len(evaluated_samples)),
        "constraint_unsupported_items": float(
            sum(int(row["unsupported_constraint_count"]) for row in with_constraints)
        ),
        "constraint_failed_items": float(
            sum(int(row["failed_constraint_count"]) for row in with_constraints)
        ),
    }
    return aggregate, details


def _constraint_items(constraints: dict[str, Any] | None) -> list[dict[str, Any]]:
    if constraints is None:
        return []
    items = constraints.get("items")
    if items is None:
        return [
            {"type": "unsupported_legacy_key", "key": key, "value": value}
            for key, value in constraints.items()
        ]
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("constraints.items must be a list of objects")
    return [dict(item) for item in items]


def _evaluate_item(
    item: Mapping[str, Any],
    sample: RNAEvaluationSample,
    chains: list[RNAChainSequence],
    metric_row: Mapping[str, Any],
) -> dict[str, Any]:
    constraint_type = item.get("type")
    if constraint_type == "sequence_length":
        selected = _selected_chains(chains, item.get("chain"))
        value = sum(len(chain.sequence) for chain in selected)
        return _bounded_result(value, item)
    if constraint_type == "chain_count":
        return _bounded_result(len(chains), item)
    if constraint_type == "gc_fraction":
        selected = _selected_chains(chains, item.get("chain"))
        sequence = "".join(chain.sequence for chain in selected)
        if not sequence:
            raise ValueError("No sequence is available for gc_fraction")
        value = sum(base in {"G", "C"} for base in sequence) / len(sequence)
        return _bounded_result(value, item)
    if constraint_type == "motif":
        selected = _selected_chains(chains, item.get("chain"))
        motif = str(item["pattern"]).upper().replace("T", "U")
        present = any(motif in chain.sequence for chain in selected)
        expected = bool(item.get("present", True))
        return {"status": "complete", "value": present, "passed": present == expected}
    if constraint_type in {"atom_distance", "residue_contact"}:
        first = _select_atom(sample.structure_path, item["first"], sample)
        second = _select_atom(sample.structure_path, item["second"], sample)
        value = float(np.linalg.norm(first - second))
        if constraint_type == "residue_contact" and "max" not in item:
            item = {**item, "max": 5.0}
        return _bounded_result(value, item)
    if constraint_type == "metric":
        value = _nested_metric(metric_row, str(item["path"]))
        comparison = str(item.get("operator", ">="))
        if comparison not in _OPERATORS:
            raise ValueError(f"Unsupported metric operator: {comparison!r}")
        threshold = float(item["value"])
        return {
            "status": "complete",
            "observed": float(value),
            "passed": bool(_OPERATORS[comparison](float(value), threshold)),
        }
    return {
        "status": "unsupported",
        "reason": f"Unsupported constraint type: {constraint_type!r}",
    }


def _bounded_result(value: float, item: Mapping[str, Any]) -> dict[str, Any]:
    exact = item.get("value")
    minimum = item.get("min")
    maximum = item.get("max")
    if exact is None and minimum is None and maximum is None:
        raise ValueError("Bounded constraint requires value, min, or max")
    passed = True
    if exact is not None:
        passed &= value == float(exact)
    if minimum is not None:
        passed &= value >= float(minimum)
    if maximum is not None:
        passed &= value <= float(maximum)
    return {"status": "complete", "observed": value, "passed": bool(passed)}


def _selected_chains(
    chains: list[RNAChainSequence], chain_id: Any
) -> list[RNAChainSequence]:
    if chain_id is None:
        return chains
    selected = [chain for chain in chains if chain.chain_id == str(chain_id)]
    if not selected:
        raise ValueError(f"RNA chain {chain_id!r} is unavailable")
    return selected


def _select_atom(
    path: Path, selector: Mapping[str, Any], sample: RNAEvaluationSample
) -> np.ndarray:
    chain_id = str(selector["chain"])
    residue_index = int(selector["residue_index"])
    atom_name = str(selector["atom"]).replace("*", "'")
    structure = load_biomolecule_structure(path)
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} is unavailable in {path}")
    residues = [
        residue
        for residue in model[chain_id]
        if rna_parent_base(residue, sample.modified_nucleotide_map) is not None
    ]
    if residue_index < 0 or residue_index >= len(residues):
        raise ValueError(
            f"RNA residue index {residue_index} is unavailable in chain {chain_id!r}"
        )
    residue = residues[residue_index]
    atoms = {
        atom.get_name().strip().replace("*", "'"): np.asarray(
            atom.coord, dtype=np.float64
        )
        for atom in residue
    }
    if atom_name not in atoms:
        raise ValueError(
            f"Atom {atom_name!r} is unavailable at {chain_id}:{residue_index}"
        )
    return atoms[atom_name]


def _nested_metric(row: Mapping[str, Any], path: str) -> float:
    value: Any = row
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"Metric path {path!r} is unavailable")
        value = value[component]
    if not isinstance(value, int | float):
        raise ValueError(f"Metric path {path!r} does not resolve to a number")
    return float(value)
