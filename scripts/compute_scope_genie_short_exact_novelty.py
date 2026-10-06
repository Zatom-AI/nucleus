#!/usr/bin/env python3
"""Compute the paper-style Genie-short novelty with exhaustive TMalign.

For every cached checkpoint, select Genie-short-confident samples using the
paper rule (argmax scTM over eight refolds and selected pLDDT > 70).  Each
selected generated backbone is compared with every canonical SCOPe-2k train
domain.  Both TMalign directional scores are retained; the primary paper-style
column uses the score normalized by the generated query (Chain_1), with the
target-normalized and conservative per-hit-minimum variants also reported.

The script supports two-worker Slurm execution.  Workers write independent
CSV shards; a merge invocation combines them after the TMalign pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from compute_scope_sweep_genie_metrics import _read_plddt

DETAIL_FIELDS = [
    "sweep",
    "step",
    "sample",
    "scTM_selected",
    "pLDDT_selected",
    "n_targets",
    "max_tmalign_tm_query",
    "max_tmalign_tm_query_target",
    "max_tmalign_tm_target",
    "max_tmalign_tm_target_target",
    "max_tmalign_tm_min",
    "max_tmalign_tm_min_target",
]


def _float(value: Any) -> float:
    return float(value) if value not in (None, "") else math.nan


def _short_confident(root: Path) -> dict[str, dict[str, Any]]:
    details = json.loads((root / "self_consistency_details.json").read_text())
    refolds: dict[str, list[dict[str, Any]]] = {}
    for row in details["per_refold"]:
        refolds.setdefault(row["sample"], []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for backbone in details["per_backbone"]:
        sample = backbone["sample"]
        best = max(refolds[sample], key=lambda row: _float(row.get("scTM")))
        plddt = _read_plddt(Path(best["prediction_pdb"]))
        if _float(best.get("scTM")) > 0.5 and plddt > 70:
            selected[sample] = {
                "backbone_pdb": backbone["backbone_pdb"],
                "scTM": _float(best.get("scTM")),
                "pLDDT": plddt,
            }
    return selected


def _parse_tmalign(stdout: str) -> tuple[float, float]:
    scores: list[float] = []
    for line in stdout.splitlines():
        if line.startswith("TM-score="):
            scores.append(float(line.split("=", 1)[1].split("(", 1)[0].strip()))
    if len(scores) < 2:
        raise RuntimeError("could not parse two TMalign scores")
    return scores[0], scores[1]


def _score_pair(args: tuple[str, str, str]) -> tuple[str, float, float]:
    binary, query, target = args
    completed = subprocess.run(
        [binary, query, target],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"TMalign failed for {query} vs {target}: {completed.stderr}"
        )
    query_tm, target_tm = _parse_tmalign(completed.stdout)
    return Path(target).stem, query_tm, target_tm


def evaluate_root(
    row: dict[str, str],
    *,
    targets: list[Path],
    tmalign_bin: Path,
    pair_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(row["output_root"])
    selected = _short_confident(root)
    details: list[dict[str, Any]] = []
    for sample, item in selected.items():
        pair_args = [
            (str(tmalign_bin), str(item["backbone_pdb"]), str(target))
            for target in targets
        ]
        with ThreadPoolExecutor(max_workers=pair_workers) as executor:
            scores = list(executor.map(_score_pair, pair_args))
        best_query = max(scores, key=lambda score: score[1])
        best_target = max(scores, key=lambda score: score[2])
        best_min = max(scores, key=lambda score: min(score[1], score[2]))
        details.append(
            {
                "sweep": row.get("run_tag", ""),
                "step": row.get("step", ""),
                "sample": sample,
                "scTM_selected": item["scTM"],
                "pLDDT_selected": item["pLDDT"],
                "n_targets": len(targets),
                "max_tmalign_tm_query": best_query[1],
                "max_tmalign_tm_query_target": best_query[0],
                "max_tmalign_tm_target": best_target[2],
                "max_tmalign_tm_target_target": best_target[0],
                "max_tmalign_tm_min": min(best_min[1], best_min[2]),
                "max_tmalign_tm_min_target": best_min[0],
            }
        )

    def mean(key: str) -> float:
        values = [_float(item[key]) for item in details]
        values = [value for value in values if math.isfinite(value)]
        return sum(values) / len(values) if values else math.nan

    def rate(key: str) -> float:
        values = [_float(item[key]) for item in details]
        values = [value for value in values if math.isfinite(value)]
        return (
            sum(value < 0.5 for value in values) / len(values) if values else math.nan
        )

    summary = {
        "sweep": row.get("run_tag", ""),
        "step": row.get("step", ""),
        "output_root": str(root),
        "n_genie_short_confident": len(selected),
        "n_exact_rescored": len(details),
        "n_canonical_targets": len(targets),
        "mean_max_tmalign_tm_query": mean("max_tmalign_tm_query"),
        "mean_max_tmalign_tm_target": mean("max_tmalign_tm_target"),
        "mean_max_tmalign_tm_min": mean("max_tmalign_tm_min"),
        "genie_short_novel_count": sum(
            _float(item["max_tmalign_tm_query"]) < 0.5 for item in details
        ),
        "genie_short_novel_rate_conditional": rate("max_tmalign_tm_query"),
        "genie_short_novel_rate_target_normalized": rate("max_tmalign_tm_target"),
        "genie_short_novel_rate_min_directional": rate("max_tmalign_tm_min"),
        "genie_short_novel_rate_all": (
            sum(_float(item["max_tmalign_tm_query"]) < 0.5 for item in details) / 96
            if details
            else math.nan
        ),
        "criterion": "max exact TMalign TM-score normalized by generated backbone (Chain_1) < 0.5",
    }
    return details, summary


def read_rows(
    paths: list[Path], step_filter: int | None = None
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(
                row
                for row in csv.DictReader(handle)
                if str(row.get("complete", "")).lower() in {"1", "true", "yes"}
                and (step_filter is None or int(row["step"]) == step_filter)
            )
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", action="append", type=Path)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--canonical-target-dir", type=Path)
    parser.add_argument("--tmalign-bin", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pair-workers", type=int, default=32)
    parser.add_argument("--worker-out-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--details-out", type=Path)
    parser.add_argument("--merge-worker-dir", type=Path)
    args = parser.parse_args()

    if args.merge_worker_dir:
        summaries: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for path in sorted(args.merge_worker_dir.glob("summary_worker_*.csv")):
            with path.open(newline="") as handle:
                summaries.extend(csv.DictReader(handle))
        for path in sorted(args.merge_worker_dir.glob("details_worker_*.csv")):
            with path.open(newline="") as handle:
                details.extend(csv.DictReader(handle))
        write_csv(args.summary_out, summaries)
        write_csv(args.details_out, details, fields=DETAIL_FIELDS)
        print(f"merged {len(summaries)} summaries and {len(details)} details")
        return

    required = [
        args.summary_csv,
        args.canonical_target_dir,
        args.tmalign_bin,
        args.worker_out_dir,
    ]
    if any(value is None for value in required):
        parser.error(
            "worker mode requires --summary-csv, --canonical-target-dir, --tmalign-bin, and --worker-out-dir"
        )
    rows = read_rows(args.summary_csv or [], step_filter=args.step)
    targets = sorted((args.canonical_target_dir or Path()).glob("*.pdb"))
    if not targets:
        raise ValueError("no canonical targets")
    assigned = [
        row
        for index, row in enumerate(rows)
        if index % args.num_workers == args.worker_index
    ]
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for index, row in enumerate(assigned, start=1):
        print(
            f"worker {args.worker_index}: checkpoint {index}/{len(assigned)} step={row.get('step')}",
            flush=True,
        )
        root_details, root_summary = evaluate_root(
            row,
            targets=targets,
            tmalign_bin=args.tmalign_bin,
            pair_workers=args.pair_workers,
        )
        summaries.append(root_summary)
        details.extend(root_details)
    args.worker_out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.worker_out_dir / f"summary_worker_{args.worker_index}.csv", summaries
    )
    write_csv(
        args.worker_out_dir / f"details_worker_{args.worker_index}.csv",
        details,
        fields=DETAIL_FIELDS,
    )
    print(
        f"worker {args.worker_index}: wrote {len(summaries)} summaries and {len(details)} details"
    )


if __name__ == "__main__":
    main()
