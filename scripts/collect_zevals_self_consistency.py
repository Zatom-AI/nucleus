#!/usr/bin/env python3
"""Collect per-sample scRMSD and optional TMalign scores from zevals outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path


def _read_fasta(path: Path) -> list[str]:
    if not path.exists():
        return []
    seqs: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        seqs.append("".join(current))
    return seqs


def _tm_score(
    ref_pdb: Path,
    pred_pdb: Path,
    tmalign_bin: str,
    timeout_seconds: float,
) -> float | None:
    if shutil.which(tmalign_bin) is None and not Path(tmalign_bin).is_file():
        return None
    try:
        result = subprocess.run(
            [tmalign_bin, str(ref_pdb), str(pred_pdb), "-outfmt", "2"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                return float(parts[2])
            except ValueError:
                return None
    return None


def _finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def collect(
    backbone_dir: Path,
    zevals_output_dir: Path,
    out_json: Path,
    out_csv: Path,
    tmalign_bin: str,
    sc_rmsd_threshold: float,
    tmalign_timeout_seconds: float = 30.0,
    sc_atom: str = "CA",
) -> None:
    from zevals.proteins.designability import _atom_rmsd

    designability_dir = zevals_output_dir / "designability"
    tmalign_available = shutil.which(tmalign_bin) is not None or Path(
        tmalign_bin
    ).is_file()
    rows: list[dict[str, object]] = []
    per_backbone: list[dict[str, object]] = []

    for backbone in sorted(backbone_dir.glob("*.pdb")):
        sample_dir = designability_dir / backbone.stem
        fasta = sample_dir / "designed.fasta"
        structures_dir = sample_dir / "structures"
        seqs = _read_fasta(fasta)
        pred_paths = (
            sorted([*structures_dir.glob("*.pdb"), *structures_dir.glob("*.cif")])
            if structures_dir.exists()
            else []
        )

        sample_rmsds: list[float] = []
        sample_tms: list[float | None] = []
        for idx, pred in enumerate(pred_paths):
            rmsd = _atom_rmsd(backbone, pred, sc_atom)
            tm_score = _tm_score(
                backbone,
                pred,
                tmalign_bin,
                timeout_seconds=tmalign_timeout_seconds,
            )
            sequence = seqs[idx] if idx < len(seqs) else ""
            row = {
                "sample": backbone.stem,
                "backbone_pdb": str(backbone),
                "prediction_pdb": str(pred),
                "sequence_index": idx,
                "sequence": sequence,
                "scRMSD": rmsd,
                "scTM": tm_score,
                "sc_atom": sc_atom,
            }
            rows.append(row)
            sample_rmsds.append(rmsd)
            sample_tms.append(tm_score)

        finite_rmsds = _finite(sample_rmsds)
        finite_tms = _finite(sample_tms)
        min_rmsd = min(finite_rmsds) if finite_rmsds else math.inf
        max_tm = max(finite_tms) if finite_tms else None
        per_backbone.append(
            {
                "sample": backbone.stem,
                "backbone_pdb": str(backbone),
                "n_sequences": len(pred_paths),
                "min_scRMSD": min_rmsd,
                "max_scTM": max_tm,
                "designable": min_rmsd < sc_rmsd_threshold,
                "sc_rmsd_threshold": sc_rmsd_threshold,
                "sc_atom": sc_atom,
            }
        )

    min_rmsds = _finite([float(row["min_scRMSD"]) for row in per_backbone])
    max_tms = _finite([row["max_scTM"] for row in per_backbone])
    designable_count = sum(bool(row["designable"]) for row in per_backbone)
    n_backbones = len(per_backbone)
    aggregate = {
        "n_backbones": n_backbones,
        "n_refolds": len(rows),
        "designable_count": designable_count,
        "designability": designable_count / n_backbones if n_backbones else 0.0,
        "mean_min_scRMSD": statistics.fmean(min_rmsds) if min_rmsds else math.inf,
        "median_min_scRMSD": statistics.median(min_rmsds) if min_rmsds else math.inf,
        "mean_max_scTM": statistics.fmean(max_tms) if max_tms else None,
        "median_max_scTM": statistics.median(max_tms) if max_tms else None,
        "sc_rmsd_threshold": sc_rmsd_threshold,
        "tmalign_timeout_seconds": tmalign_timeout_seconds,
        "tmalign_available": tmalign_available,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "per_backbone": per_backbone,
                "per_refold": rows,
            },
            indent=2,
            allow_nan=True,
        )
        + "\n"
    )

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "backbone_pdb",
                "prediction_pdb",
                "sequence_index",
                "sequence",
                "scRMSD",
                "scTM",
                "sc_atom",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(aggregate, indent=2, allow_nan=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--zevals-output-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--tmalign-bin", default="TMalign")
    parser.add_argument("--tmalign-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--sc-rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--sc-atom", default="CA")
    args = parser.parse_args()

    collect(
        backbone_dir=args.backbone_dir,
        zevals_output_dir=args.zevals_output_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        tmalign_bin=args.tmalign_bin,
        sc_rmsd_threshold=args.sc_rmsd_threshold,
        tmalign_timeout_seconds=args.tmalign_timeout_seconds,
        sc_atom=args.sc_atom,
    )


if __name__ == "__main__":
    main()
