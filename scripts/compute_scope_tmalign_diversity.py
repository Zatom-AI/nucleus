#!/usr/bin/env python3
"""Compute TM-align complete-linkage diversity for a scope self-consistency run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
ZEVALS_SRC = REPO_DIR / "models" / "zevals" / "src"
if str(ZEVALS_SRC) not in sys.path:
    sys.path.insert(0, str(ZEVALS_SRC))

from zevals.proteins.diversity import compute_tmalign_complete_linkage_diversity


def _parse_thresholds(value: str | None) -> list[float] | None:
    if value is None:
        return None
    thresholds = [float(item) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("--thresholds cannot be empty")
    return thresholds


def _load_backbones(sc_root: Path, query_set: str) -> tuple[list[Path], int, int]:
    details_path = sc_root / "self_consistency_details.json"
    if not details_path.is_file():
        raise FileNotFoundError(details_path)

    details = json.loads(details_path.read_text())
    rows = details["per_backbone"]
    backbones: list[Path] = []
    n_designable = 0
    for row in rows:
        is_designable = bool(row.get("designable"))
        n_designable += int(is_designable)
        if query_set == "designable" and not is_designable:
            continue
        backbones.append(Path(row["backbone_pdb"]))
    return backbones, len(rows), n_designable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-consistency-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--query-set", choices=["all", "designable"], default="all")
    parser.add_argument("--tmalign-bin", default="TMalign")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--symmetrize", choices=["min", "max", "mean"], default="min")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    sc_root = args.self_consistency_root
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else sc_root / "diversity" / f"tmalign_complete_linkage_{args.query_set}"
    )
    backbones, n_total, n_designable = _load_backbones(sc_root, args.query_set)
    metrics = compute_tmalign_complete_linkage_diversity(
        backbones,
        output_dir=output_dir,
        tmalign_bin=args.tmalign_bin,
        tm_thresholds=_parse_thresholds(args.thresholds),
        symmetrize=args.symmetrize,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
    )
    metrics.update(
        {
            "self_consistency_root": str(sc_root),
            "query_set": args.query_set,
            "n_total_backbones": n_total,
            "n_designable": n_designable,
            "n_queries": len(backbones),
            "symmetrization": args.symmetrize,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tmalign_diversity_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
