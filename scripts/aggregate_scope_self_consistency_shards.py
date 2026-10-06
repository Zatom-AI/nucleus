#!/usr/bin/env python3
"""Aggregate self-consistency detail JSON files from sharded zevals runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def aggregate(shard_root: Path, out_json: Path, out_csv: Path) -> None:
    details_paths = sorted(
        shard_root.glob("shard_*/zevals/self_consistency_details.json")
    )
    per_backbone: list[dict[str, object]] = []
    per_refold: list[dict[str, object]] = []
    for details_path in details_paths:
        payload = json.loads(details_path.read_text())
        shard = details_path.parts[-3]
        for row in payload.get("per_backbone", []):
            row = dict(row)
            row["shard"] = shard
            per_backbone.append(row)
        for row in payload.get("per_refold", []):
            row = dict(row)
            row["shard"] = shard
            per_refold.append(row)

    min_rmsds = _finite([row.get("min_scRMSD") for row in per_backbone])
    max_tms = _finite([row.get("max_scTM") for row in per_backbone])
    designable_count = sum(bool(row.get("designable")) for row in per_backbone)
    n_backbones = len(per_backbone)
    thresholds = _finite([row.get("sc_rmsd_threshold") for row in per_backbone])
    aggregate_payload = {
        "n_shards_with_details": len(details_paths),
        "n_backbones": n_backbones,
        "n_refolds": len(per_refold),
        "designable_count": designable_count,
        "designability": designable_count / n_backbones if n_backbones else 0.0,
        "mean_min_scRMSD": statistics.fmean(min_rmsds) if min_rmsds else math.inf,
        "median_min_scRMSD": statistics.median(min_rmsds) if min_rmsds else math.inf,
        "mean_max_scTM": statistics.fmean(max_tms) if max_tms else None,
        "median_max_scTM": statistics.median(max_tms) if max_tms else None,
        "sc_rmsd_threshold": thresholds[0] if thresholds else None,
    }
    payload = {
        "aggregate": aggregate_payload,
        "per_backbone": per_backbone,
        "per_refold": per_refold,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "shard",
            "sample",
            "backbone_pdb",
            "prediction_pdb",
            "sequence_index",
            "sequence",
            "scRMSD",
            "scTM",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_refold)

    print(json.dumps(aggregate_payload, indent=2, allow_nan=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.shard_root, args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
