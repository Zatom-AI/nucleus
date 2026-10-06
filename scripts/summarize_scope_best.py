#!/usr/bin/env python3
"""Strict cohort audits and length-stratified statistics for SCOPe evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def wilson_interval(successes: int, total: int) -> list[float]:
    """Supplementary binomial score interval, nondegenerate at zero/all success."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid success/total counts")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def stratified_bootstrap(
    values: np.ndarray, lengths: np.ndarray, *, seed: int = 20260919, draws: int = 10000
) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling backbones within fixed length strata.

    Refold candidates and pairwise alignments are not independent observations.
    This estimates generation uncertainty conditional on one trained checkpoint.
    """
    values, lengths = np.asarray(values, float), np.asarray(lengths)
    if values.ndim != 1 or values.shape != lengths.shape or not len(values):
        raise ValueError("Expected nonempty equally sized one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite values cannot be silently dropped")
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    for length in np.unique(lengths):
        stratum = values[lengths == length]
        indices = rng.integers(0, len(stratum), size=(draws, len(stratum)))
        totals += stratum[indices].sum(axis=1)
    low, high = np.quantile(totals / len(values), [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def audit_cohort(payload: dict, specs: dict) -> list[dict]:
    """Require every attempted backbone and all eight finite refold metrics."""
    expected = {f"{key}_0_model_0": val["length"] for key, val in specs.items()}
    rows = payload["per_backbone"]
    if Counter(row["sample"] for row in rows) != Counter(dict.fromkeys(expected, 1)):
        raise ValueError("Missing, extra, or duplicate generated samples")
    refolds: dict[str, list[dict]] = {key: [] for key in expected}
    for refold in payload["per_refold"]:
        refolds[refold["sample"]].append(refold)
    for row in rows:
        candidates = refolds[row["sample"]]
        if len(candidates) != 8 or {r["sequence_index"] for r in candidates} != set(
            range(8)
        ):
            raise ValueError(f"Incomplete best-of-eight: {row['sample']}")
        if any(r["scTM"] is None for r in candidates):
            raise ValueError(f"Missing TM-align scores: {row['sample']}")
        rmsds = np.array([r["scRMSD"] for r in candidates], float)
        tms = np.array([r["scTM"] for r in candidates], float)
        if not np.isfinite(rmsds).all() or not np.isfinite(tms).all():
            raise ValueError(f"Nonfinite refold metrics: {row['sample']}")
        if not np.allclose(
            [rmsds.min(), tms.max()], [row["min_scRMSD"], row["max_scTM"]]
        ):
            raise ValueError("Best-of-eight summaries disagree with individual refolds")
        if bool(row["designable"]) != bool(rmsds.min() < 1.5):
            raise ValueError("Designability threshold mismatch")
        row["length"] = expected[row["sample"]]
    return sorted(rows, key=lambda row: row["sample"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(args: argparse.Namespace) -> None:
    """Compute the manuscript's metrics, plus PDB novelty and P-SEA content."""
    from aggregate_scope_self_consistency_shards import aggregate
    from compute_scope_sweep_novelty_diversity import compute_foldseek_novelty
    from evaluate_scope_training_designability import write_json
    from zevals.proteins.diversity import compute_tmalign_complete_linkage_diversity

    root = args.root / args.model
    aggregate(
        root / "shards",
        root / "self_consistency_details.json",
        root / "self_consistency_refolds.csv",
    )
    payload = json.loads((root / "self_consistency_details.json").read_text())
    specs = json.loads((root / "inputs.json").read_text())
    rows = audit_cohort(payload, specs)
    if len(rows) != 1027 or Counter(row["length"] for row in rows) != Counter(
        dict.fromkeys(range(50, 129), 13)
    ):
        raise ValueError("Expected exactly 13 samples for each length 50–128")
    for name, database in [("scope2k_train", args.train_db), ("PDB", args.pdb_db)]:
        compute_foldseek_novelty(
            sc_root=root,
            target_db=database,
            target_name=name,
            output_subdir=Path("novelty") / name,
            query_set="all",
            foldseek_bin=Path(args.foldseek),
            tm_threshold=0.5,
            force=False,
        )
        hits = list(
            csv.DictReader((root / "novelty" / name / "best_foldseek_hits.csv").open())
        )
        hit_map = {row["query"]: row for row in hits}
        if set(hit_map) != {row["sample"] for row in rows}:
            raise ValueError("Novelty queries do not match cohort")
        for row in rows:
            hit = hit_map[row["sample"]]
            # A search with no retrieved hit is not evidence of a novel fold.
            row[f"{name}_hit_found"] = bool(hit["best_target"])
            row[f"{name}_nearest_tm"] = (
                float(hit["best_tm"]) if hit["best_target"] else None
            )
            row[f"{name}_nearest_target"] = hit["best_target"]

    import biotite.structure as struc
    import biotite.structure.io as strucio

    for row in rows:
        atoms = strucio.load_structure(row["backbone_pdb"])
        sse = struc.annotate_sse(atoms)
        if len(sse) != row["length"]:
            raise ValueError("Secondary-structure annotation length mismatch")
        for code, label in [("a", "alpha"), ("b", "beta"), ("c", "coil")]:
            row[f"{label}_fraction"] = float(np.mean(sse == code))
        row["unclassified_fraction"] = float(np.mean(~np.isin(sse, ["a", "b", "c"])))
    write_csv(root / "per_backbone_metrics.csv", rows)
    scalar_summary = scalar_statistics(rows)
    write_json(root / "scalar_summary.json", scalar_summary)

    diversity_path = root / "diversity" / "metrics.json"
    if diversity_path.exists():
        diversity = json.loads(diversity_path.read_text())
    else:
        diversity = compute_tmalign_complete_linkage_diversity(
            [Path(row["backbone_pdb"]) for row in rows],
            root / "diversity",
            tmalign_bin=args.tmalign,
            symmetrize="min",
            chunk_size=8,
            n_workers=args.workers,
        )
        if diversity["tmalign_n_missing_pair_scores"] != 0:
            raise ValueError(
                "Missing pairwise TM-align comparisons; cannot report diversity"
            )
        write_json(diversity_path, diversity)
    write_json(
        root / "evaluation_complete.json",
        {
            "n_backbones": len(rows),
            "n_refolds": len(payload["per_refold"]),
            "length_counts": dict(Counter(row["length"] for row in rows)),
            "all_refolds_finite": True,
            "all_pairs_present": True,
            "novelty_missing_hits": {
                name: sum(not row[f"{name}_hit_found"] for row in rows)
                for name in ("scope2k_train", "PDB")
            },
            "diversity": diversity,
            "statistics": scalar_summary,
        },
    )


def scalar_statistics(rows: list[dict]) -> dict:
    """Summarize all-sample outcomes and conditionally designable subsets."""
    keys = [
        "designable",
        "min_scRMSD",
        "max_scTM",
        "scope2k_train_nearest_tm",
        "PDB_nearest_tm",
        "alpha_fraction",
        "beta_fraction",
        "coil_fraction",
        "unclassified_fraction",
    ]
    summary = {
        "ci_method": "10000 within-length backbone bootstrap draws; percentile 95% CI",
        "uncertainty_scope": "Generation variability conditional on selected checkpoints; excludes training-seed and checkpoint-selection uncertainty.",
        "groups": {},
    }
    groups = {"all": rows}
    for lo, hi in [(50, 75), (76, 101), (102, 128)]:
        groups[f"length_{lo}_{hi}"] = [r for r in rows if lo <= r["length"] <= hi]
    for name, group in groups.items():
        result = {"n": len(group), "metrics": {}}
        lengths = np.array([row["length"] for row in group])
        for key in keys:
            if any(row[key] is None for row in group):
                result["metrics"][key] = {
                    "unresolved_missing_values": sum(row[key] is None for row in group)
                }
                continue
            values = np.array([row[key] for row in group], float)
            mean, low, high = stratified_bootstrap(values, lengths)
            result["metrics"][key] = {"mean": mean, "ci95": [low, high]}
            if key == "designable":
                result["metrics"][key]["wilson_binomial_ci95"] = wilson_interval(
                    int(values.sum()), len(values)
                )
        for database in ("scope2k_train", "PDB"):
            resolved = all(row[f"{database}_hit_found"] for row in group)
            if resolved:
                for quality_gate in (False, True):
                    values = np.array(
                        [
                            row[f"{database}_nearest_tm"] < 0.5
                            and (row["designable"] or not quality_gate)
                            for row in group
                        ]
                    )
                    mean, low, high = stratified_bootstrap(values, lengths)
                    key = f"{database}_" + (
                        "designable_and_novel" if quality_gate else "tm_below_0.5"
                    )
                    result["metrics"][key] = {"mean": mean, "ci95": [low, high]}
                    result["metrics"][key]["wilson_binomial_ci95"] = wilson_interval(
                        int(values.sum()), len(values)
                    )
        summary["groups"][name] = result
    summary["per_length"] = [
        {
            "length": length,
            "n": sum(row["length"] == length for row in rows),
            **{
                key: float(
                    np.mean([row[key] for row in rows if row["length"] == length])
                )
                for key in keys
                if all(row[key] is not None for row in rows)
            },
        }
        for length in sorted({row["length"] for row in rows})
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-db", type=Path, required=True)
    parser.add_argument("--pdb-db", type=Path, required=True)
    parser.add_argument("--foldseek", required=True)
    parser.add_argument("--tmalign", default="TMalign")
    parser.add_argument("--workers", type=int, default=16)
    metrics(parser.parse_args())


if __name__ == "__main__":
    main()
