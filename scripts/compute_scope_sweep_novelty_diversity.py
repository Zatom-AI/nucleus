#!/usr/bin/env python3
"""Compute FoldSeek novelty and TM-align diversity over a SCOPe sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
ZEVALS_SRC = REPO_DIR / "models" / "zevals" / "src"
if str(ZEVALS_SRC) not in sys.path:
    sys.path.insert(0, str(ZEVALS_SRC))

from zevals.oracles.structure_search.foldseek import FoldSeekSearcher
from zevals.proteins.diversity import compute_tmalign_complete_linkage_diversity


DEFAULT_SUMMARY_OUT = "novelty_diversity_summary.csv"


def parse_metrics(value: str) -> set[str]:
    metrics = {item.strip() for item in value.split(",") if item.strip()}
    unknown = metrics - {"novelty", "diversity"}
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown metrics {sorted(unknown)}; use novelty,diversity"
        )
    if not metrics:
        raise argparse.ArgumentTypeError("--metrics cannot be empty")
    return metrics


def parse_float(value: str | int | float | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def load_sweep_rows(
    summary_csv: Path,
    *,
    refolds_per_backbone: int,
    allow_partial_refolds: bool,
) -> list[dict[str, str]]:
    if summary_csv.is_dir():
        summary_csv = summary_csv / "summary.csv"
    with summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    kept: list[dict[str, str]] = []
    for row in rows:
        n_backbones = int(parse_float(row.get("n_backbones") or 0))
        n_refolds = int(parse_float(row.get("n_refolds") or 0))
        complete = str(row.get("complete", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not complete or n_backbones <= 0:
            continue
        if not allow_partial_refolds and n_refolds != refolds_per_backbone * n_backbones:
            continue
        output_root = Path(row["output_root"])
        if not (output_root / "self_consistency_details.json").is_file():
            continue
        kept.append(row)

    if not kept:
        raise ValueError(f"No usable completed self-consistency rows in {summary_csv}")
    return kept


def load_single_row(
    sc_root: Path,
    *,
    refolds_per_backbone: int,
    allow_partial_refolds: bool,
) -> list[dict[str, str]]:
    details_path = sc_root / "self_consistency_details.json"
    if not details_path.is_file():
        raise FileNotFoundError(details_path)

    payload = json.loads(details_path.read_text())
    aggregate = payload.get("aggregate", {})
    n_backbones = int(parse_float(aggregate.get("n_backbones") or 0))
    n_refolds = int(parse_float(aggregate.get("n_refolds") or 0))
    if n_backbones <= 0:
        raise ValueError(f"No backbones found in {details_path}")
    if not allow_partial_refolds and n_refolds != refolds_per_backbone * n_backbones:
        raise ValueError(
            f"Incomplete refolds in {details_path}: n_refolds={n_refolds}, "
            f"expected={refolds_per_backbone * n_backbones}. Pass "
            "--allow-partial-refolds to override."
        )

    return [
        {
            "step": "",
            "step_name": sc_root.name,
            "run_tag": sc_root.name,
            "output_root": str(sc_root),
            "n_backbones": str(n_backbones),
            "n_refolds": str(n_refolds),
            "complete": "True",
        }
    ]


def load_backbones(sc_root: Path, query_set: str) -> tuple[list[tuple[Path, bool]], int, int]:
    details_path = sc_root / "self_consistency_details.json"
    if not details_path.is_file():
        raise FileNotFoundError(details_path)

    details = json.loads(details_path.read_text())
    selected: list[tuple[Path, bool]] = []
    n_total = 0
    n_designable = 0
    for row in details["per_backbone"]:
        n_total += 1
        is_designable = bool(row.get("designable"))
        n_designable += int(is_designable)
        if query_set == "designable" and not is_designable:
            continue
        pdb_path = Path(row["backbone_pdb"])
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)
        selected.append((pdb_path, is_designable))

    if not selected:
        raise ValueError(f"No {query_set} backbones found in {details_path}")
    return selected, n_total, n_designable


def stable_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def compute_foldseek_novelty(
    *,
    sc_root: Path,
    target_db: Path,
    target_name: str,
    output_subdir: Path,
    query_set: str,
    foldseek_bin: Path,
    tm_threshold: float,
    force: bool,
) -> dict[str, object]:
    out_dir = sc_root / output_subdir
    metrics_path = out_dir / "novelty_metrics.json"
    hits_path = out_dir / "best_foldseek_hits.csv"
    if not force and metrics_path.is_file() and hits_path.is_file():
        print(f"Skipping novelty, already complete: {out_dir}", flush=True)
        return json.loads(metrics_path.read_text())

    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected, n_total, n_designable = load_backbones(sc_root, query_set)
    query_dir = out_dir / f"{query_set}_backbones"
    query_dir.mkdir(parents=True, exist_ok=True)
    for src, _is_designable in selected:
        stable_symlink(src, query_dir / src.name)

    foldseek = FoldSeekSearcher(
        binary_path=str(foldseek_bin),
        tmp_dir=out_dir / "tmp",
    )
    result_tsv = out_dir / "novelty_search.tsv"
    results = foldseek.search(query_dir, target_db, result_tsv)

    best = {}
    for result in results:
        query = Path(result.query_id).stem
        if result.hits:
            best[query] = max(result.hits, key=lambda hit: hit["tm_score"])

    rows = []
    for pdb, is_designable in selected:
        hit = best.get(pdb.stem)
        rows.append(
            {
                "query": pdb.stem,
                "designable": int(is_designable),
                "best_target": "" if hit is None else hit["target"],
                "best_tm": 0.0 if hit is None else hit["tm_score"],
                "evalue": "" if hit is None else hit["evalue"],
            }
        )

    best_tms = [float(row["best_tm"]) for row in rows]
    metrics: dict[str, object] = {
        "self_consistency_root": str(sc_root),
        "target_db": str(target_db),
        "target_name": target_name,
        "query_set": query_set,
        "n_total_backbones": n_total,
        "n_designable": n_designable,
        "n_queries": len(selected),
        "novelty_mean_best_tm": sum(best_tms) / len(best_tms),
        f"novel_rate_tm_below_{tm_threshold:g}": sum(
            value < tm_threshold for value in best_tms
        )
        / len(best_tms),
        "lower_mean_best_tm_is_more_novel": True,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    with hits_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote novelty {out_dir}", flush=True)
    return metrics


def compute_diversity(
    *,
    sc_root: Path,
    output_subdir: Path,
    query_set: str,
    tmalign_bin: Path,
    symmetrize: str,
    chunk_size: int,
    n_workers: int,
    force: bool,
) -> dict[str, object]:
    out_dir = sc_root / output_subdir
    metrics_path = out_dir / "tmalign_diversity_metrics.json"
    if not force and metrics_path.is_file():
        print(f"Skipping diversity, already complete: {out_dir}", flush=True)
        return json.loads(metrics_path.read_text())

    if force and out_dir.exists():
        shutil.rmtree(out_dir)

    selected, n_total, n_designable = load_backbones(sc_root, query_set)
    metrics = compute_tmalign_complete_linkage_diversity(
        [pdb for pdb, _is_designable in selected],
        output_dir=out_dir,
        tmalign_bin=str(tmalign_bin),
        symmetrize=symmetrize,
        chunk_size=chunk_size,
        n_workers=n_workers,
    )
    metrics.update(
        {
            "self_consistency_root": str(sc_root),
            "query_set": query_set,
            "n_total_backbones": n_total,
            "n_designable": n_designable,
            "n_queries": len(selected),
            "symmetrization": symmetrize,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote diversity {out_dir}", flush=True)
    return metrics


def metric_value(payload: dict[str, object] | None, key: str) -> object:
    if not payload:
        return ""
    value = payload.get(key, "")
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def summarize_outputs(
    rows: list[dict[str, str]],
    *,
    out_csv: Path,
    novelty_subdir: Path,
    diversity_subdir: Path,
    novelty_tm_threshold: float,
) -> None:
    novel_rate_key = f"novel_rate_tm_below_{novelty_tm_threshold:g}"
    summary_rows = []
    for row in rows:
        sc_root = Path(row["output_root"])
        novelty_metrics_path = sc_root / novelty_subdir / "novelty_metrics.json"
        diversity_metrics_path = sc_root / diversity_subdir / "tmalign_diversity_metrics.json"
        novelty = (
            json.loads(novelty_metrics_path.read_text())
            if novelty_metrics_path.is_file()
            else None
        )
        diversity = (
            json.loads(diversity_metrics_path.read_text())
            if diversity_metrics_path.is_file()
            else None
        )
        summary_rows.append(
            {
                "step": row["step"],
                "step_name": row.get("step_name", ""),
                "run_tag": row.get("run_tag", ""),
                "output_root": row["output_root"],
                "novelty_complete": novelty is not None,
                "diversity_complete": diversity is not None,
                "novelty_mean_best_tm": metric_value(novelty, "novelty_mean_best_tm"),
                novel_rate_key: metric_value(novelty, novel_rate_key),
                "novelty_n_queries": metric_value(novelty, "n_queries"),
                "tmalign_diversity_auc": metric_value(
                    diversity, "tmalign_diversity_auc"
                ),
                "tmalign_cluster_fraction_at_0.5": metric_value(
                    diversity, "tmalign_cluster_fraction_at_0.5"
                ),
                "tmalign_n_clusters_at_0.5": metric_value(
                    diversity, "tmalign_n_clusters_at_0.5"
                ),
                "tmalign_mean_pairwise_tm": metric_value(
                    diversity, "tmalign_mean_pairwise_tm"
                ),
                "tmalign_median_pairwise_tm": metric_value(
                    diversity, "tmalign_median_pairwise_tm"
                ),
                "tmalign_n_missing_pair_scores": metric_value(
                    diversity, "tmalign_n_missing_pair_scores"
                ),
                "diversity_n_queries": metric_value(diversity, "n_queries"),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {out_csv}", flush=True)


def run_worker(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    assigned = [
        row
        for index, row in enumerate(rows)
        if index % args.num_workers == args.worker_index
    ]
    print(
        f"worker {args.worker_index}/{args.num_workers}: "
        f"{len(assigned)} of {len(rows)} checkpoint roots",
        flush=True,
    )
    for item_index, row in enumerate(assigned, start=1):
        sc_root = Path(row["output_root"])
        step_name = row.get("step_name") or f"step-{int(row['step']):08d}"
        print(
            f"worker {args.worker_index}: [{item_index}/{len(assigned)}] {step_name} {sc_root}",
            flush=True,
        )
        if "novelty" in args.metrics:
            compute_foldseek_novelty(
                sc_root=sc_root,
                target_db=args.foldseek_target_db,
                target_name=args.foldseek_target_name,
                output_subdir=args.novelty_output_subdir,
                query_set=args.novelty_query_set,
                foldseek_bin=args.foldseek_bin,
                tm_threshold=args.novelty_tm_threshold,
                force=args.force,
            )
        if "diversity" in args.metrics:
            compute_diversity(
                sc_root=sc_root,
                output_subdir=args.diversity_output_subdir,
                query_set=args.diversity_query_set,
                tmalign_bin=args.tmalign_bin,
                symmetrize=args.tmalign_symmetrize,
                chunk_size=args.tmalign_chunk_size,
                n_workers=args.tmalign_n_workers,
                force=args.force,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary-csv", type=Path)
    source.add_argument("--self-consistency-root", type=Path)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--metrics", type=parse_metrics, default=parse_metrics("novelty,diversity"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refolds-per-backbone", type=int, default=8)
    parser.add_argument("--allow-partial-refolds", action="store_true")
    parser.add_argument("--foldseek-bin", type=Path, default=Path("foldseek"))
    parser.add_argument("--foldseek-target-db", type=Path, default=None)
    parser.add_argument("--foldseek-target-name", default="scope_train")
    parser.add_argument("--novelty-query-set", choices=["all", "designable"], default="all")
    parser.add_argument(
        "--novelty-output-subdir",
        type=Path,
        default=Path("novelty/foldseek_scope_train_all"),
    )
    parser.add_argument("--novelty-tm-threshold", type=float, default=0.5)
    parser.add_argument("--tmalign-bin", type=Path, default=Path("TMalign"))
    parser.add_argument("--diversity-query-set", choices=["all", "designable"], default="all")
    parser.add_argument(
        "--diversity-output-subdir",
        type=Path,
        default=Path("diversity/tmalign_complete_linkage_all"),
    )
    parser.add_argument("--tmalign-symmetrize", choices=["min", "max", "mean"], default="min")
    parser.add_argument("--tmalign-chunk-size", type=int, default=8)
    parser.add_argument("--tmalign-n-workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summary_csv is not None and args.summary_csv.is_dir():
        args.summary_csv = args.summary_csv / "summary.csv"
    if args.worker_index < 0 or args.num_workers <= 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must be in [0, --num-workers)")
    if "novelty" in args.metrics:
        if args.foldseek_target_db is None:
            raise ValueError("--foldseek-target-db is required when running novelty")
        if not (
            args.foldseek_target_db.exists()
            or args.foldseek_target_db.with_suffix(args.foldseek_target_db.suffix + ".dbtype").exists()
            or Path(f"{args.foldseek_target_db}.dbtype").exists()
        ):
            raise FileNotFoundError(args.foldseek_target_db)

    if args.summary_csv is not None:
        rows = load_sweep_rows(
            args.summary_csv,
            refolds_per_backbone=args.refolds_per_backbone,
            allow_partial_refolds=args.allow_partial_refolds,
        )
        default_summary_out = args.summary_csv.parent / DEFAULT_SUMMARY_OUT
    else:
        rows = load_single_row(
            args.self_consistency_root,
            refolds_per_backbone=args.refolds_per_backbone,
            allow_partial_refolds=args.allow_partial_refolds,
        )
        default_summary_out = args.self_consistency_root / DEFAULT_SUMMARY_OUT

    summary_out = args.summary_out or default_summary_out
    if args.summarize_only:
        summarize_outputs(
            rows,
            out_csv=summary_out,
            novelty_subdir=args.novelty_output_subdir,
            diversity_subdir=args.diversity_output_subdir,
            novelty_tm_threshold=args.novelty_tm_threshold,
        )
        return

    run_worker(args, rows)
    if args.num_workers == 1:
        summarize_outputs(
            rows,
            out_csv=summary_out,
            novelty_subdir=args.novelty_output_subdir,
            diversity_subdir=args.diversity_output_subdir,
            novelty_tm_threshold=args.novelty_tm_threshold,
        )


if __name__ == "__main__":
    main()
