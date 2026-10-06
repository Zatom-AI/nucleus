#!/usr/bin/env python3
"""Convert one completed self-consistency result into sweep-summary CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

FIELDS = [
    "step",
    "step_name",
    "ckpt_path",
    "run_tag",
    "output_root",
    "complete",
    "n_backbones",
    "n_refolds",
    "designability",
    "designable_count",
    "mean_min_scRMSD",
    "median_min_scRMSD",
    "mean_max_scTM",
    "median_max_scTM",
    "all_refolds_mean_scRMSD",
    "all_refolds_median_scRMSD",
    "all_refolds_mean_scTM",
    "all_refolds_median_scTM",
]


def finite(values: list[object]) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-json", type=Path, required=True)
    parser.add_argument("--refolds-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.details_json.read_text())
    aggregate = payload["aggregate"]
    with args.refolds_csv.open(newline="") as handle:
        refolds = list(csv.DictReader(handle))
    rmsds = finite([row.get("scRMSD") for row in refolds])
    tms = finite([row.get("scTM") for row in refolds])

    row = {
        "step": args.step,
        "step_name": f"step-{args.step:08d}",
        "ckpt_path": str(args.checkpoint),
        "run_tag": args.output_root.name,
        "output_root": str(args.output_root),
        "complete": 1,
        "n_backbones": aggregate.get("n_backbones", ""),
        "n_refolds": aggregate.get("n_refolds", ""),
        "designability": aggregate.get("designability", ""),
        "designable_count": aggregate.get("designable_count", ""),
        "mean_min_scRMSD": aggregate.get("mean_min_scRMSD", ""),
        "median_min_scRMSD": aggregate.get("median_min_scRMSD", ""),
        "mean_max_scTM": aggregate.get("mean_max_scTM", ""),
        "median_max_scTM": aggregate.get("median_max_scTM", ""),
        "all_refolds_mean_scRMSD": statistics.fmean(rmsds) if rmsds else "",
        "all_refolds_median_scRMSD": statistics.median(rmsds) if rmsds else "",
        "all_refolds_mean_scTM": statistics.fmean(tms) if tms else "",
        "all_refolds_median_scTM": statistics.median(tms) if tms else "",
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
