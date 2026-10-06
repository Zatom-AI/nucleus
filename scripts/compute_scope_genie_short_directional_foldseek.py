#!/usr/bin/env python3
"""Run a Genie-short-style novelty audit with directional FoldSeek TM scores.

This is the scalable part of the audit.  For each cached Zatom checkpoint we
select the Genie-short-confident samples (argmax scTM over eight refolds and
selected pLDDT > 70), search them against a FoldSeek database, and preserve
both directional TM scores.  The old SCOPe novelty cache used ``alntmscore``;
that is retained only for comparison and is not used for the 0.5 decisions.

The result is not a claim that FoldSeek and TM-align are numerically
identical.  It provides a consistent train-versus-held-out diagnostic while
the accompanying exact-TM audit rescored FoldSeek nominees with TMalign.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from compute_scope_sweep_genie_metrics import _read_plddt

DETAIL_FIELDS = [
    "sweep",
    "step",
    "sample",
    "scTM_selected",
    "pLDDT_selected",
    "n_hits",
    "best_qtmscore",
    "best_qtmscore_target",
    "best_ttmscore",
    "best_ttmscore_target",
    "best_alntmscore",
    "best_alntmscore_target",
    "best_min_directional_tm",
    "best_min_directional_tm_target",
]

SUMMARY_FIELDS = [
    "sweep",
    "step",
    "output_root",
    "target_name",
    "n_genie_short_confident",
    "n_queries",
    "n_hits_total",
    "mean_best_qtmscore",
    "mean_best_ttmscore",
    "mean_best_min_directional_tm",
    "mean_best_alntmscore",
    "novel_rate_best_qtmscore_below_0.5",
    "novel_rate_best_ttmscore_below_0.5",
    "novel_rate_best_min_directional_tm_below_0.5",
    "novel_rate_best_alntmscore_below_0.5",
    "foldseek_exact_tmscore",
    "foldseek_exhaustive_search",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _run_search(
    *,
    foldseek_bin: Path,
    query_dir: Path,
    target_db: Path,
    result_tsv: Path,
    tmp_dir: Path,
    threads: int,
    max_seqs: int,
    exhaustive: bool,
) -> None:
    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(foldseek_bin),
        "easy-search",
        str(query_dir),
        str(target_db),
        str(result_tsv),
        str(tmp_dir),
        "--threads",
        str(threads),
        "--max-seqs",
        str(max_seqs),
        "--alignment-type",
        "1",
        "--exact-tmscore",
        "1",
        "--tmalign-hit-order",
        "0",
        "--format-output",
        "query,target,qtmscore,ttmscore,alntmscore,evalue",
    ]
    if exhaustive:
        command.extend(["--exhaustive-search", "1"])
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "FoldSeek failed:\n{}\n{}".format(completed.stdout, completed.stderr)
        )


def _read_hits(path: Path) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
            fieldnames=[
                "query",
                "target",
                "qtmscore",
                "ttmscore",
                "alntmscore",
                "evalue",
            ],
        )
        for row in reader:
            hits.setdefault(Path(row["query"]).stem, []).append(row)
    return hits


def _max_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else math.nan


def evaluate_root(
    row: dict[str, str],
    *,
    foldseek_bin: Path,
    target_db: Path,
    target_name: str,
    output_dir: Path,
    threads: int,
    max_seqs: int,
    exhaustive: bool,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(row["output_root"])
    selected = _short_confident(root)
    label = "{}_step_{}".format(row.get("run_tag", "run"), row.get("step", "unknown"))
    run_dir = output_dir / label
    query_dir = run_dir / "queries"
    result_tsv = run_dir / "foldseek_directional.tsv"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if selected and (force or not result_tsv.is_file()):
        query_dir.mkdir(parents=True, exist_ok=True)
        for sample, item in selected.items():
            destination = query_dir / (sample + ".pdb")
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(Path(item["backbone_pdb"]))
        _run_search(
            foldseek_bin=foldseek_bin,
            query_dir=query_dir,
            target_db=target_db,
            result_tsv=result_tsv,
            tmp_dir=run_dir / "tmp",
            threads=threads,
            max_seqs=max_seqs,
            exhaustive=exhaustive,
        )

    hits = _read_hits(result_tsv) if result_tsv.is_file() else {}
    details: list[dict[str, Any]] = []
    for sample, item in selected.items():
        sample_hits = hits.get(sample, [])
        best_by = {}
        for key in ("qtmscore", "ttmscore", "alntmscore"):
            finite = [hit for hit in sample_hits if math.isfinite(_float(hit.get(key)))]
            best_by[key] = (
                max(finite, key=lambda hit: _float(hit[key])) if finite else None
            )
        row_out: dict[str, Any] = {
            "sweep": row.get("run_tag", ""),
            "step": row.get("step", ""),
            "sample": sample,
            "scTM_selected": item["scTM"],
            "pLDDT_selected": item["pLDDT"],
            "n_hits": len(sample_hits),
        }
        for key in ("qtmscore", "ttmscore", "alntmscore"):
            best = best_by[key]
            row_out["best_" + key] = math.nan if best is None else _float(best[key])
            row_out["best_" + key + "_target"] = "" if best is None else best["target"]
        min_hits = [
            hit
            for hit in sample_hits
            if math.isfinite(_float(hit.get("qtmscore")))
            and math.isfinite(_float(hit.get("ttmscore")))
        ]
        best_min_hit = max(
            min_hits,
            key=lambda hit: min(_float(hit["qtmscore"]), _float(hit["ttmscore"])),
            default=None,
        )
        row_out["best_min_directional_tm"] = (
            math.nan
            if best_min_hit is None
            else min(_float(best_min_hit["qtmscore"]), _float(best_min_hit["ttmscore"]))
        )
        row_out["best_min_directional_tm_target"] = (
            "" if best_min_hit is None else best_min_hit["target"]
        )
        details.append(row_out)

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

    summary: dict[str, Any] = {
        "sweep": row.get("run_tag", ""),
        "step": row.get("step", ""),
        "output_root": str(root),
        "target_name": target_name,
        "n_genie_short_confident": len(selected),
        "n_queries": len(details),
        "n_hits_total": sum(int(item["n_hits"]) for item in details),
        "mean_best_qtmscore": mean("best_qtmscore"),
        "mean_best_ttmscore": mean("best_ttmscore"),
        "mean_best_min_directional_tm": mean("best_min_directional_tm"),
        "mean_best_alntmscore": mean("best_alntmscore"),
        "novel_rate_best_qtmscore_below_0.5": rate("best_qtmscore"),
        "novel_rate_best_ttmscore_below_0.5": rate("best_ttmscore"),
        "novel_rate_best_min_directional_tm_below_0.5": rate("best_min_directional_tm"),
        "novel_rate_best_alntmscore_below_0.5": rate("best_alntmscore"),
        "foldseek_exact_tmscore": True,
        "foldseek_exhaustive_search": exhaustive,
    }
    return details, summary


def _rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if str(row.get("complete", "")).lower() in {"1", "true", "yes"}
            and int(float(row.get("n_backbones") or 0)) > 0
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", action="append", type=Path, required=True)
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--foldseek-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-seqs", type=int, default=10000)
    parser.add_argument("--exhaustive-search", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_details: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for summary_csv in args.summary_csv:
        for row in _rows(summary_csv):
            details, summary = evaluate_root(
                row,
                foldseek_bin=args.foldseek_bin,
                target_db=args.target_db,
                target_name=args.target_name,
                output_dir=args.output_dir,
                threads=args.threads,
                max_seqs=args.max_seqs,
                exhaustive=args.exhaustive_search,
                force=args.force,
            )
            all_details.extend(details)
            all_summaries.append(summary)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.summary_out, all_summaries, SUMMARY_FIELDS)
    _write_csv(args.details_out, all_details, DETAIL_FIELDS)
    print("wrote {} and {}".format(args.summary_out, args.details_out))


if __name__ == "__main__":
    main()
