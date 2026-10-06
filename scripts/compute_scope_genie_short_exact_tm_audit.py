#!/usr/bin/env python3
"""Exact TM-align audit of cached Genie-short-confident samples.

FoldSeek is used only to nominate the nearest SCOPe training target.  The
nominee is then rescored with the exact local TM-align binary.  This is a
conservative audit: if the exact score to the FoldSeek nominee is already
>= 0.5, the generated structure cannot be novel under a full exact search.
It is not an exhaustive all-2,000-target TM-align search.
"""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from compute_scope_sweep_genie_metrics import _read_plddt

DETAIL_FIELDS = [
    "sweep",
    "step",
    "sample",
    "scTM_selected",
    "pLDDT_selected",
    "foldseek_target",
    "foldseek_best_tm",
    "tmalign_tm_query",
    "tmalign_tm_target",
    "tmalign_tm_min",
    "tmalign_tm_max",
]

SUMMARY_FIELDS = [
    "sweep",
    "step",
    "output_root",
    "n_genie_short_confident",
    "n_exact_rescored",
    "foldseek_mean_best_tm",
    "tmalign_mean_tm_query",
    "tmalign_mean_tm_target",
    "tmalign_mean_tm_min",
    "tmalign_novel_rate_query_below_0.5",
    "tmalign_novel_rate_target_below_0.5",
    "tmalign_novel_rate_min_below_0.5",
]


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    return float(value) if value not in (None, "") else math.nan


def _tmalign_scores(binary: Path, query: Path, target: Path) -> Tuple[float, float]:
    completed = subprocess.Popen(
        [str(binary), str(query), str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = completed.communicate()
    if completed.returncode:
        raise RuntimeError(
            "TMalign failed for {} vs {}: {}".format(query, target, stderr)
        )
    scores = []
    for line in stdout.splitlines():
        if line.startswith("TM-score="):
            scores.append(float(line.split("=", 1)[1].split("(", 1)[0].strip()))
    if len(scores) < 2:
        raise RuntimeError(
            "could not parse two TM-scores for {} vs {}".format(query, target)
        )
    return scores[0], scores[1]


def _read_short_confident(root: Path) -> Dict[str, Dict[str, Any]]:
    details = json.loads((root / "self_consistency_details.json").read_text())
    refolds = {}  # type: Dict[str, List[Dict[str, Any]]]
    for row in details["per_refold"]:
        refolds.setdefault(row["sample"], []).append(row)

    confident = {}
    for backbone in details["per_backbone"]:
        sample = backbone["sample"]
        best = max(refolds[sample], key=lambda row: _float(row.get("scTM")))
        plddt = _read_plddt(Path(best["prediction_pdb"]))
        if _float(best.get("scTM")) > 0.5 and plddt > 70:
            confident[sample] = {
                "backbone_pdb": backbone["backbone_pdb"],
                "scTM": _float(best.get("scTM")),
                "pLDDT": plddt,
            }
    return confident


def _read_foldseek(root: Path) -> Dict[str, Dict[str, Any]]:
    paths = sorted((root / "novelty").glob("*/best_foldseek_hits.csv"))
    if not paths:
        raise FileNotFoundError(
            "no cached best_foldseek_hits.csv under {}".format(root)
        )
    with paths[0].open(newline="") as handle:
        return {row["query"]: row for row in csv.DictReader(handle)}


def evaluate_root(
    row: Dict[str, str],
    train_pdb_dir: Path,
    tmalign_bin: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(row["output_root"])
    confident = _read_short_confident(root)
    foldseek = _read_foldseek(root)
    detail_rows = []
    for sample, short in confident.items():
        hit = foldseek.get(sample)
        if not hit or not hit.get("best_target"):
            continue
        target = train_pdb_dir / (hit["best_target"] + ".pdb")
        if not target.is_file():
            raise FileNotFoundError(target)
        query = Path(short["backbone_pdb"])
        tm_query, tm_target = _tmalign_scores(tmalign_bin, query, target)
        detail_rows.append(
            {
                "sweep": row.get("run_tag", ""),
                "step": row.get("step", ""),
                "sample": sample,
                "scTM_selected": short["scTM"],
                "pLDDT_selected": short["pLDDT"],
                "foldseek_target": hit["best_target"],
                "foldseek_best_tm": _float(hit.get("best_tm")),
                "tmalign_tm_query": tm_query,
                "tmalign_tm_target": tm_target,
                "tmalign_tm_min": min(tm_query, tm_target),
                "tmalign_tm_max": max(tm_query, tm_target),
            }
        )

    def rate(key: str) -> float:
        values = [_float(item[key]) for item in detail_rows]
        values = [value for value in values if math.isfinite(value)]
        return (
            sum(value < 0.5 for value in values) / len(values) if values else math.nan
        )

    def mean(key: str) -> float:
        values = [_float(item[key]) for item in detail_rows]
        values = [value for value in values if math.isfinite(value)]
        return sum(values) / len(values) if values else math.nan

    summary = {
        "sweep": row.get("run_tag", ""),
        "step": row.get("step", ""),
        "output_root": str(root),
        "n_genie_short_confident": len(confident),
        "n_exact_rescored": len(detail_rows),
        "foldseek_mean_best_tm": mean("foldseek_best_tm"),
        "tmalign_mean_tm_query": mean("tmalign_tm_query"),
        "tmalign_mean_tm_target": mean("tmalign_tm_target"),
        "tmalign_mean_tm_min": mean("tmalign_tm_min"),
        "tmalign_novel_rate_query_below_0.5": rate("tmalign_tm_query"),
        "tmalign_novel_rate_target_below_0.5": rate("tmalign_tm_target"),
        "tmalign_novel_rate_min_below_0.5": rate("tmalign_tm_min"),
    }
    return detail_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", action="append", type=Path, required=True)
    parser.add_argument("--train-target-pdb-dir", type=Path, required=True)
    parser.add_argument("--tmalign-bin", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, required=True)
    args = parser.parse_args()

    detail_rows = []
    summary_rows = []
    for summary_path in args.summary_csv:
        with summary_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("complete", "1").lower() not in {"1", "true", "yes"}:
                continue
            details, summary = evaluate_root(
                row, args.train_target_pdb_dir, args.tmalign_bin
            )
            prefix = "paper_table4_"
            summary["sweep"] = (
                summary_path.parent.name[len(prefix) :]
                if summary_path.parent.name.startswith(prefix)
                else summary_path.parent.name
            )
            for item in details:
                item["sweep"] = summary["sweep"]
            detail_rows.extend(details)
            summary_rows.append(summary)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.summary_out, summary_rows, SUMMARY_FIELDS)
    _write_csv(args.details_out, detail_rows, DETAIL_FIELDS)
    print("wrote {} and {}".format(args.summary_out, args.details_out))


if __name__ == "__main__":
    main()
