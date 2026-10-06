#!/usr/bin/env python3
"""SCOPe advanced novelty/diversity audit.

This is deliberately separate from the historical FoldSeek summaries.  It
computes a configurable scRMSD-filtered protocol from cached generated
backbones and direct model sequences:

* designability: configurable CA scRMSD threshold, with best-of-8 and
  strict-all-refolds views;
* novelty: exact TMalign against the full canonical SCOPe-2k training set,
  query-normalized by the generated backbone;
* structural diversity: all-to-all TMalign, conservative min-directional
  symmetrization, complete linkage, threshold-0.5 counts and AUC;
* sequence diversity: MMseqs2 at 10% identity and 70% coverage;
* joint sequence/structure diversity: complete-linkage clustering after
  intersecting the MMseqs2 sequence edge relation with TM >= 0.5.

The script is resumable at the cohort level and at the expensive TMalign
chunk level.  It has two phases: workers evaluate manifest rows, then a
single merge invocation writes tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "models/zevals/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "models/zevals/src"))

from zevals.proteins.diversity import (  # noqa: E402
    _cluster_count_at_threshold,
    _complete_linkage_merge_tms,
    compute_tmalign_complete_linkage_diversity,
)

PLOT_COLORS = {
    "zatom2_scratch": "#d62728",  # red, as in the scope-transfer figure
    "zatom2_joint_fc": "#174a7e",  # dark blue
    "zatom2_joint_properties_fc": "#3c82b8",  # medium blue
    "zatom2_joint_no_fc": "#73add0",  # light blue
    "rfd3": "#20242b",  # black/charcoal control
}
ALL_BACKBONE_CLUSTER_COUNT_KEY = "diversity_all_backbones_n_clusters_at_0_5"
ALL_BACKBONE_CLUSTER_FRACTION_KEY = "diversity_all_backbones_cluster_fraction_at_0_5"
AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "HYP": "P",
}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("rows", value.get("manifest", []))
    if not isinstance(value, list):
        raise ValueError(f"Manifest must be a list or {{rows: [...]}}: {path}")
    return [dict(row) for row in value]


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if str(row.get("complete", "")).lower() in {"1", "true", "yes"}
        ]


def _root_local_index(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    backbones: dict[str, Path] = {}
    generated: dict[str, Path] = {}
    for path in root.glob("shards/*/backbones/*.pdb"):
        backbones[path.name] = path
    for path in root.glob("shards/*/generated/*.cif.gz"):
        key = path.name[:-7] if path.name.endswith(".cif.gz") else path.stem
        generated[key] = path
    for path in root.glob("shards/*/generated/*.cif"):
        generated.setdefault(path.stem, path)
    return backbones, generated


def _sample_length(sample: str) -> int | None:
    match = re.search(r"(?:^|_)len(\d+)(?:_|$)", sample)
    return int(match.group(1)) if match else None


def _load_cohort(
    row: dict[str, Any], threshold: float, expected_refolds: int
) -> dict[str, Any]:
    root = Path(row["output_root"])
    details_path = root / "self_consistency_details.json"
    if not details_path.is_file():
        raise FileNotFoundError(details_path)
    details = json.loads(details_path.read_text())
    local_backbones, generated = _root_local_index(root)
    lo = row.get("length_min")
    hi = row.get("length_max")
    lo = int(lo) if lo not in (None, "") else None
    hi = int(hi) if hi not in (None, "") else None
    by_sample_refolds: dict[str, list[dict[str, Any]]] = {}
    for refold in details.get("per_refold", []):
        by_sample_refolds.setdefault(str(refold["sample"]), []).append(refold)
    entries: list[dict[str, Any]] = []
    for backbone in details.get("per_backbone", []):
        sample = str(backbone["sample"])
        length = _sample_length(sample)
        if lo is not None and (length is None or length < lo):
            continue
        if hi is not None and (length is None or length > hi):
            continue
        name = Path(str(backbone.get("backbone_pdb", sample + ".pdb"))).name
        pdb = local_backbones.get(name)
        if pdb is None:
            candidate = (
                root
                / "shards"
                / str(backbone.get("shard", "shard_00"))
                / "backbones"
                / name
            )
            if candidate.is_file():
                pdb = candidate
        if pdb is None:
            raise FileNotFoundError(f"backbone for {sample} not found below {root}")
        cif = generated.get(Path(name).stem)
        refs = by_sample_refolds.get(sample, [])
        if len(refs) != expected_refolds:
            raise ValueError(
                f"Expected {expected_refolds} refolds for {sample}, found {len(refs)}"
            )
        sequence_indices = [ref.get("sequence_index") for ref in refs]
        if all(index is not None for index in sequence_indices) and {
            int(index) for index in sequence_indices
        } != set(range(expected_refolds)):
            raise ValueError(
                f"Refold sequence indices for {sample} are not exactly "
                f"0..{expected_refolds - 1}: {sequence_indices}"
            )
        atoms = {
            str(ref.get("sc_atom", "")).upper()
            for ref in refs
            if ref.get("sc_atom") not in (None, "")
        }
        if atoms and atoms != {"CA"}:
            raise ValueError(
                f"Expected cached CA scRMSD values for {sample}, found sc_atom={sorted(atoms)}"
            )
        finite = [_float(ref.get("scRMSD")) for ref in refs]
        finite = [value for value in finite if math.isfinite(value)]
        if len(finite) != expected_refolds:
            raise ValueError(
                f"Expected {expected_refolds} finite scRMSD values for {sample}, "
                f"found {len(finite)}"
            )
        best = bool(finite) and min(finite) < threshold
        # Strict all-refolds is intentional and is reported separately from
        # best-of-8. A missing/invalid refold cannot pass this filter.
        strict_all = (
            bool(refs)
            and len(finite) == len(refs)
            and all(value < threshold for value in finite)
        )
        entries.append(
            {
                "sample": sample,
                "length": length,
                "backbone_pdb": str(pdb),
                "generated_cif": str(cif) if cif else "",
                "refolds": refs,
                "best_of_8_designable": best,
                "all_refolds_designable": strict_all,
                "n_refolds": len(refs),
                "n_passing_refolds": sum(value < threshold for value in finite),
                "min_scRMSD": min(finite) if finite else math.nan,
            }
        )
    if not entries:
        raise ValueError(f"No cohort entries selected from {root} with row={row}")
    return {
        "row": row,
        "root": root,
        "entries": entries,
        "n_all": len(entries),
        "threshold": threshold,
        "expected_refolds": expected_refolds,
        "canonical": details,
    }


def _designability_metrics(
    entries: list[dict[str, Any]], threshold: float, expected_refolds: int
) -> dict[str, Any]:
    passing_refolds = sum(int(entry["n_passing_refolds"]) for entry in entries)
    total_refolds = len(entries) * expected_refolds
    return {
        "n_backbones": len(entries),
        "threshold_A": threshold,
        "expected_refolds_per_backbone": expected_refolds,
        "best_of_8_count": int(sum(entry["best_of_8_designable"] for entry in entries)),
        "best_of_8_rate": float(
            sum(entry["best_of_8_designable"] for entry in entries) / len(entries)
        ),
        "per_sequence_success_count": passing_refolds,
        "per_sequence_success_total": total_refolds,
        "per_sequence_success_rate": float(passing_refolds / total_refolds),
        "all_refolds_count": int(
            sum(entry["all_refolds_designable"] for entry in entries)
        ),
        "all_refolds_rate": float(
            sum(entry["all_refolds_designable"] for entry in entries) / len(entries)
        ),
        "all_refolds_definition": (
            "all finite refolds for a generated backbone pass CA scRMSD "
            f"< {threshold:g} A"
        ),
    }


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", errors="replace")
        if path.name.endswith(".gz")
        else path.open("r", errors="replace")
    )


def _cif_sequence(path: Path) -> str:
    """Read one direct protein sequence from the generated all-atom CIF."""
    if not path:
        return ""
    with _open_text(path) as handle:
        lines = handle.readlines()
    headers: list[str] = []
    in_loop = False
    data_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "loop_":
            in_loop = True
            headers = []
            continue
        if in_loop and stripped.startswith("_atom_site."):
            headers.append(stripped.split()[0])
            continue
        if in_loop and headers and stripped and not stripped.startswith("_"):
            data_start = index
            break
    if not headers or data_start is None:
        return ""
    try:
        atom_col = headers.index("_atom_site.label_atom_id")
    except ValueError:
        atom_col = (
            headers.index("_atom_site.auth_atom_id")
            if "_atom_site.auth_atom_id" in headers
            else -1
        )
    comp_col = (
        headers.index("_atom_site.label_comp_id")
        if "_atom_site.label_comp_id" in headers
        else -1
    )
    seq_col = (
        headers.index("_atom_site.label_seq_id")
        if "_atom_site.label_seq_id" in headers
        else -1
    )
    asym_col = (
        headers.index("_atom_site.label_asym_id")
        if "_atom_site.label_asym_id" in headers
        else -1
    )
    if min(atom_col, comp_col, seq_col, asym_col) < 0:
        return ""
    residues: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("#"):
                break
            continue
        try:
            parts = shlex.split(stripped, comments=False, posix=True)
        except ValueError:
            continue
        if len(parts) < len(headers) or parts[0] not in {"ATOM", "HETATM"}:
            continue
        if parts[atom_col] != "CA":
            continue
        key = (parts[asym_col], parts[seq_col])
        if key in seen:
            continue
        seen.add(key)
        comp = parts[comp_col].upper()
        residues.append((parts[asym_col], parts[seq_col], AA3.get(comp, "X")))
    return "".join(item[2] for item in residues)


def _write_fasta(
    path: Path, entries: list[dict[str, Any]], indices: list[int]
) -> list[str]:
    sequences: list[str] = []
    lines: list[str] = []
    for seq_idx, entry_idx in enumerate(indices):
        generated_cif = entries[entry_idx]["generated_cif"]
        if not generated_cif:
            raise ValueError(
                "Sequence/joint diversity requires the generated all-atom CIF "
                f"for {entries[entry_idx]['sample']}"
            )
        sequence = _cif_sequence(Path(generated_cif))
        if not sequence:
            raise ValueError(
                "Could not recover a protein sequence from generated CIF "
                f"{generated_cif}"
            )
        sequences.append(sequence)
        lines.extend([f">seq_{seq_idx:06d}", sequence])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return sequences


def _parse_tmalign_outfmt2(text: str) -> Iterable[tuple[str, str, float, float]]:
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            parts = line.split()
        if len(parts) < 4:
            continue
        try:
            yield (
                Path(parts[0]).stem,
                Path(parts[1]).stem,
                float(parts[2]),
                float(parts[3]),
            )
        except (TypeError, ValueError):
            continue


def _run_dir_tmalign(
    binary: str,
    query_dir: Path,
    query_list: Path,
    target_dir: Path,
    target_list: Path,
    output: Path,
) -> None:
    tmp = output.with_suffix(".tmp")
    command = [
        binary,
        "-dir1",
        f"{query_dir}/",
        str(query_list),
        "-dir2",
        f"{target_dir}/",
        str(target_list),
        "-outfmt",
        "2",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode:
        raise RuntimeError(
            f"TMalign failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    tmp.write_text(completed.stdout)
    tmp.replace(output)


def _read_novelty_scores(
    path: Path,
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    scores: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            scores[row["sample"]] = {
                key: _float(row.get(key))
                for key in (
                    "max_tmalign_tm_query",
                    "max_tmalign_tm_target",
                    "max_tmalign_tm_min",
                )
            }
    return scores, rows


def _exact_novelty(
    cohort: dict[str, Any],
    output_dir: Path,
    targets: list[Path],
    tmalign_bin: str,
    pair_workers: int,
    chunk_size: int,
    cached_score_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    entries = cohort["entries"]
    union_indices = [
        index
        for index, entry in enumerate(entries)
        if entry["best_of_8_designable"] or entry["all_refolds_designable"]
    ]
    result_dir = output_dir / "novelty_tmalign_query"
    result_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = result_dir / "per_backbone.csv"
    if cached_score_path is not None:
        cached_scores, cached_rows = _read_novelty_scores(cached_score_path)
        required = {entries[index]["sample"] for index in union_indices}
        missing = sorted(required - set(cached_scores))
        if missing:
            raise ValueError(
                f"Cached novelty scores do not cover {len(missing)} selected samples; "
                f"first missing={missing[0]} source={cached_score_path}"
            )
        _csv_write(
            per_query_path, [row for row in cached_rows if row["sample"] in required]
        )
        _json_dump(
            result_dir / "cache_provenance.json",
            {
                "source": str(cached_score_path),
                "n_source_scores": len(cached_scores),
                "n_selected_scores": len(required),
            },
        )
        score_by_sample = {sample: cached_scores[sample] for sample in required}
        metrics: dict[str, Any] = {}
        for mode in ("best_of_8", "all_refolds"):
            selected = [
                entry
                for entry in entries
                if entry[f"{mode}_designable"] and entry["sample"] in score_by_sample
            ]
            values = [
                score_by_sample[entry["sample"]]["max_tmalign_tm_query"]
                for entry in selected
            ]
            values = [value for value in values if math.isfinite(value)]
            metrics[mode] = {
                "n_designable": len(selected),
                "n_rescored": len(values),
                "mean_max_tmalign_tm_query": float(np.mean(values))
                if values
                else math.nan,
                "novel_count_max_tm_lt_0_5": int(sum(value < 0.5 for value in values)),
                "novel_rate_max_tm_lt_0_5": float(
                    sum(value < 0.5 for value in values) / len(values)
                )
                if values
                else math.nan,
                "criterion": "max exact TMalign TM-score normalized by generated backbone (Chain 1) < 0.5",
            }
        return metrics, score_by_sample
    query_manifest = {
        "samples": [entries[index]["sample"] for index in union_indices],
        "targets": [str(path) for path in targets],
        "target_count": len(targets),
        "tmalign_bin": tmalign_bin,
    }
    cached = False
    if (result_dir / "manifest.json").is_file() and per_query_path.is_file():
        try:
            cached = (
                json.loads((result_dir / "manifest.json").read_text()) == query_manifest
            )
        except json.JSONDecodeError:
            cached = False
    if not cached:
        query_dir = result_dir / "queries"
        query_dir.mkdir(parents=True, exist_ok=True)
        query_names: list[str] = []
        for query_index, entry_index in enumerate(union_indices):
            name = f"query_{query_index:06d}.pdb"
            link = query_dir / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(Path(entries[entry_index]["backbone_pdb"]).resolve())
            query_names.append(name)
        target_dir = targets[0].parent
        target_list = result_dir / "target_list.txt"
        target_list.write_text("\n".join(path.name for path in targets) + "\n")
        chunk_dir = result_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[tuple[int, list[str], Path, Path]] = []
        for chunk_index in range(0, len(query_names), chunk_size):
            names = query_names[chunk_index : chunk_index + chunk_size]
            number = chunk_index // chunk_size
            query_list = chunk_dir / f"queries_{number:05d}.txt"
            output = chunk_dir / f"results_{number:05d}.tsv"
            query_list.write_text("\n".join(names) + "\n")
            chunks.append((number, names, query_list, output))
        pending = [
            chunk
            for chunk in chunks
            if not chunk[3].is_file() or chunk[3].stat().st_size == 0
        ]
        if pending:
            with ThreadPoolExecutor(
                max_workers=min(pair_workers, len(pending))
            ) as executor:
                futures = {
                    executor.submit(
                        _run_dir_tmalign,
                        tmalign_bin,
                        query_dir,
                        query_list,
                        target_dir,
                        target_list,
                        output,
                    ): number
                    for number, _names, query_list, output in pending
                }
                for future in as_completed(futures):
                    future.result()
                    print(
                        f"exact novelty chunk {futures[future] + 1}/{len(chunks)} complete",
                        flush=True,
                    )
        maxima: dict[str, dict[str, float]] = {}
        for _number, names, _query_list, output in chunks:
            name_stems = {Path(name).stem for name in names}
            for query_id, target_id, query_tm, target_tm in _parse_tmalign_outfmt2(
                output.read_text()
            ):
                if query_id not in name_stems:
                    continue
                current = maxima.setdefault(
                    query_id,
                    {
                        "max_query": math.nan,
                        "max_target": math.nan,
                        "max_min": math.nan,
                    },
                )
                current["max_query"] = (
                    max(current["max_query"], query_tm)
                    if math.isfinite(current["max_query"])
                    else query_tm
                )
                current["max_target"] = (
                    max(current["max_target"], target_tm)
                    if math.isfinite(current["max_target"])
                    else target_tm
                )
                score_min = min(query_tm, target_tm)
                current["max_min"] = (
                    max(current["max_min"], score_min)
                    if math.isfinite(current["max_min"])
                    else score_min
                )
        rows: list[dict[str, Any]] = []
        for query_index, entry_index in enumerate(union_indices):
            query_id = f"query_{query_index:06d}"
            scores = maxima.get(query_id, {})
            rows.append(
                {
                    "sample": entries[entry_index]["sample"],
                    "backbone_pdb": entries[entry_index]["backbone_pdb"],
                    "max_tmalign_tm_query": scores.get("max_query", math.nan),
                    "max_tmalign_tm_target": scores.get("max_target", math.nan),
                    "max_tmalign_tm_min": scores.get("max_min", math.nan),
                    "n_targets": len(targets),
                }
            )
        _csv_write(per_query_path, rows)
        _json_dump(result_dir / "manifest.json", query_manifest)
    score_by_sample: dict[str, dict[str, float]] = {}
    with per_query_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            score_by_sample[row["sample"]] = {
                key: _float(row.get(key))
                for key in (
                    "max_tmalign_tm_query",
                    "max_tmalign_tm_target",
                    "max_tmalign_tm_min",
                )
            }
    metrics: dict[str, Any] = {}
    for mode in ("best_of_8", "all_refolds"):
        selected = [
            entry
            for entry in entries
            if entry[f"{mode}_designable"] and entry["sample"] in score_by_sample
        ]
        values = [
            score_by_sample[entry["sample"]]["max_tmalign_tm_query"]
            for entry in selected
        ]
        values = [value for value in values if math.isfinite(value)]
        metrics[mode] = {
            "n_designable": len(selected),
            "n_rescored": len(values),
            "mean_max_tmalign_tm_query": float(np.mean(values)) if values else math.nan,
            "novel_count_max_tm_lt_0_5": int(sum(value < 0.5 for value in values)),
            "novel_rate_max_tm_lt_0_5": float(
                sum(value < 0.5 for value in values) / len(values)
            )
            if values
            else math.nan,
            "criterion": "max exact TMalign TM-score normalized by generated backbone (Chain 1) < 0.5",
        }
    return metrics, score_by_sample


def _read_tm_matrix(
    path: Path, expected_entries: list[dict[str, Any]] | None = None
) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        return np.zeros((0, 0), dtype=float)
    if expected_entries is not None:
        expected_ids = [
            f"sample_{index:05d}_{Path(entry['backbone_pdb']).stem}"
            for index, entry in enumerate(expected_entries)
        ]
        header_ids = rows[0][1:]
        row_ids = [row[0] for row in rows[1:]]
        if header_ids != expected_ids or row_ids != expected_ids:
            raise ValueError(
                f"Cached TM matrix order does not match the selected cohort: {path}"
            )
    return np.asarray(
        [[float(value) for value in row[1:]] for row in rows[1:]], dtype=float
    )


def _subset_diversity(
    matrix: np.ndarray, indices: list[int], output_dir: Path, prefix: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not indices:
        return {
            "n_structures": 0,
            "n_clusters_at_0_5": 0,
            "cluster_fraction_at_0_5": 0.0,
            "cluster_fraction_over_all_at_0_5": 0.0,
            "tmalign_diversity_auc": math.nan,
            "mean_pairwise_tm": math.nan,
            "median_pairwise_tm": math.nan,
        }
    sub = matrix[np.ix_(indices, indices)]
    n = sub.shape[0]
    merges = _complete_linkage_merge_tms(sub)
    thresholds = np.linspace(0.0, 1.0, 101)
    counts = np.asarray(
        [_cluster_count_at_threshold(n, merges, float(t)) for t in thresholds]
    )
    fractions = counts / float(n)
    if n > 1:
        upper = sub[np.triu_indices(n, 1)]
        mean_tm = float(np.mean(upper))
        median_tm = float(np.median(upper))
    else:
        mean_tm = median_tm = math.nan
    curve_rows = [
        {
            "tm_threshold": float(t),
            "n_clusters": int(count),
            "cluster_fraction": float(frac),
        }
        for t, count, frac in zip(thresholds, counts, fractions)
    ]
    _csv_write(output_dir / f"{prefix}_complete_linkage_curve.csv", curve_rows)
    return {
        "n_structures": int(n),
        "n_clusters_at_0_5": int(_cluster_count_at_threshold(n, merges, 0.5)),
        "cluster_fraction_at_0_5": float(
            _cluster_count_at_threshold(n, merges, 0.5) / n
        ),
        "cluster_fraction_over_all_at_0_5": float(
            _cluster_count_at_threshold(n, merges, 0.5) / matrix.shape[0]
        )
        if matrix.shape[0]
        else 0.0,
        "tmalign_diversity_auc": float(
            np.trapezoid(fractions, thresholds)
            if hasattr(np, "trapezoid")
            else np.trapz(fractions, thresholds)
        ),
        "mean_pairwise_tm": mean_tm,
        "median_pairwise_tm": median_tm,
    }


def _run_mmseqs_cluster(
    fasta: Path,
    output_dir: Path,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "cluster"
    tsv = Path(str(prefix) + "_cluster.tsv")
    rep = Path(str(prefix) + "_rep_seq.fasta")
    if not tsv.is_file() or not rep.is_file():
        tmp = output_dir / "tmp"
        command = [
            mmseqs_bin,
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--threads",
            str(threads),
        ]
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if completed.returncode:
            raise RuntimeError(f"MMseqs2 clustering failed: {completed.stderr[-3000:]}")
    clusters: set[str] = set()
    if tsv.is_file():
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            if parts:
                clusters.add(parts[0])
    return len(clusters)


def _run_mmseqs_search(
    fasta: Path,
    output_dir: Path,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / "all_vs_all.tsv"
    if result.is_file() and result.stat().st_size:
        return result
    tmp = output_dir / "tmp_search"
    command = [
        mmseqs_bin,
        "easy-search",
        str(fasta),
        str(fasta),
        str(result),
        str(tmp),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--max-seqs",
        "1000000",
        "--format-output",
        "query,target,pident,qcov,tcov",
        "--threads",
        str(threads),
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 all-vs-all failed: {completed.stderr[-3000:]}")
    return result


def _fraction(value: str) -> float:
    number = _float(value)
    if not math.isfinite(number):
        return math.nan
    return number / 100.0 if number > 1.0 else number


def _joint_clusters(structural: np.ndarray, search_tsv: Path, n: int) -> int:
    sequence_edges = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(sequence_edges, True)
    for line in search_tsv.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            left = int(parts[0].rsplit("_", 1)[1])
            right = int(parts[1].rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        pident = _fraction(parts[2])
        qcov = _fraction(parts[3])
        tcov = _fraction(parts[4])
        coverage_ok = (math.isfinite(qcov) and qcov >= 0.7) and (
            math.isfinite(tcov) and tcov >= 0.7
        )
        if (
            0 <= left < n
            and 0 <= right < n
            and math.isfinite(pident)
            and pident >= 0.1
            and coverage_ok
        ):
            sequence_edges[left, right] = sequence_edges[right, left] = True
    joint = np.where(sequence_edges & (structural >= 0.5), 1.0, 0.0)
    merges = _complete_linkage_merge_tms(joint)
    return _cluster_count_at_threshold(n, merges, 0.5)


def _sequence_metrics(
    cohort: dict[str, Any],
    output_dir: Path,
    matrix: np.ndarray,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> dict[str, Any]:
    entries = cohort["entries"]
    output: dict[str, Any] = {}
    for mode in ("best_of_8", "all_refolds"):
        indices = [i for i, entry in enumerate(entries) if entry[f"{mode}_designable"]]
        mode_dir = output_dir / "sequences" / mode
        fasta = mode_dir / "sequences.fasta"
        _write_fasta(fasta, entries, indices)
        if not indices:
            output[mode] = {
                "n_sequences": 0,
                "n_clusters": 0,
                "cluster_fraction": 0.0,
                "joint_n_clusters": 0,
                "sequence_source": "generated all-atom CIF residue identities",
            }
            continue
        n_clusters = _run_mmseqs_cluster(
            fasta,
            mode_dir / "mmseqs_cluster",
            mmseqs_bin,
            min_seq_id,
            coverage,
            threads,
        )
        search = _run_mmseqs_search(
            fasta, mode_dir / "mmseqs_search", mmseqs_bin, min_seq_id, coverage, threads
        )
        submatrix = matrix[np.ix_(indices, indices)]
        joint = _joint_clusters(submatrix, search, len(indices))
        output[mode] = {
            "n_sequences": len(indices),
            "n_clusters": n_clusters,
            "cluster_fraction": float(n_clusters / len(indices)) if indices else 0.0,
            "joint_n_clusters": joint,
            "joint_cluster_fraction": float(joint / len(indices)) if indices else 0.0,
            "sequence_identity_threshold": min_seq_id,
            "sequence_coverage_threshold": coverage,
            "mmseqs_cov_mode": 0,
            "joint_definition": "complete-linkage clusters whose every pair has TM-align min-sym TM >= 0.5 and MMseqs2 identity/coverage pass",
            "sequence_source": "generated all-atom CIF residue identities",
        }
    return output


def evaluate_row(
    row: dict[str, Any],
    output_root: Path,
    canonical_targets: Path,
    tmalign_bin: str,
    mmseqs_bin: str,
    threshold: float,
    pair_workers: int,
    chunk_size: int,
    mmseqs_threads: int,
    expected_refolds: int,
    cache_source_root: Path | None = None,
) -> dict[str, Any]:
    row_id = str(row.get("row_id") or row.get("label") or Path(row["output_root"]).name)
    row_output = output_root / _safe_name(row_id)
    metrics_path = row_output / "advanced_metrics.json"
    if metrics_path.is_file():
        try:
            cached = json.loads(metrics_path.read_text())
            protocol = cached.get("protocol", {})
            if (
                protocol.get("sc_rmsd_threshold_A") == threshold
                and protocol.get("expected_refolds_per_backbone") == expected_refolds
            ):
                print(f"reusing {row_id}", flush=True)
                return cached
        except json.JSONDecodeError:
            pass
    cohort = _load_cohort(row, threshold, expected_refolds)
    entries = cohort["entries"]
    all_paths = [Path(entry["backbone_pdb"]) for entry in entries]
    row_output.mkdir(parents=True, exist_ok=True)
    threshold_tag = f"{threshold:g}A".replace(".", "p")
    _csv_write(
        row_output / f"designability_scRMSD_{threshold_tag}.csv",
        [
            {
                "sample": entry["sample"],
                "length": entry["length"],
                "min_scRMSD": entry["min_scRMSD"],
                "best_of_8_designable": int(entry["best_of_8_designable"]),
                "all_refolds_designable": int(entry["all_refolds_designable"]),
                "n_refolds": entry["n_refolds"],
                "n_passing_refolds": entry["n_passing_refolds"],
            }
            for entry in entries
        ],
    )
    designability = _designability_metrics(entries, threshold, expected_refolds)
    cache_row_output = (
        cache_source_root / _safe_name(row_id) if cache_source_root else None
    )
    cached_novelty = (
        cache_row_output / "novelty_tmalign_query" / "per_backbone.csv"
        if cache_row_output
        else None
    )
    if cached_novelty is not None and not cached_novelty.is_file():
        raise FileNotFoundError(cached_novelty)
    targets = sorted(canonical_targets.glob("*.pdb"))
    if not targets:
        raise FileNotFoundError(f"No canonical SCOPe targets below {canonical_targets}")
    novelty, _scores = _exact_novelty(
        cohort,
        row_output,
        targets,
        tmalign_bin,
        pair_workers,
        chunk_size,
        cached_score_path=cached_novelty,
    )
    structural_dir = row_output / "structural_diversity" / "all_backbones"
    if cache_row_output is not None:
        cached_matrix = (
            cache_row_output
            / "structural_diversity"
            / "all_backbones"
            / "tmalign_tm_matrix.csv"
        )
        cached_metrics = cache_row_output / "advanced_metrics.json"
        if not cached_matrix.is_file():
            raise FileNotFoundError(cached_matrix)
        if not cached_metrics.is_file():
            raise FileNotFoundError(cached_metrics)
        matrix = _read_tm_matrix(cached_matrix, expected_entries=entries)
        cached_result = json.loads(cached_metrics.read_text())
        all_backbone_metrics = dict(cached_result["diversity"]["all_backbones"])
        if int(all_backbone_metrics.get("n_structures", -1)) != len(entries):
            raise ValueError(f"Cached all-backbone metrics do not match {row_id}")
        _json_dump(
            structural_dir / "cache_provenance.json",
            {
                "matrix_source": str(cached_matrix),
                "metrics_source": str(cached_metrics),
                "n_structures": len(entries),
            },
        )
    else:
        all_structural = compute_tmalign_complete_linkage_diversity(
            all_paths,
            structural_dir,
            tmalign_bin=tmalign_bin,
            symmetrize="min",
            chunk_size=chunk_size,
            n_workers=pair_workers,
        )
        matrix = _read_tm_matrix(
            structural_dir / "tmalign_tm_matrix.csv", expected_entries=entries
        )
        all_backbone_metrics = {
            key.removeprefix("tmalign_"): value for key, value in all_structural.items()
        }
    if matrix.shape != (len(entries), len(entries)):
        raise ValueError(
            f"TM matrix shape {matrix.shape} != {(len(entries), len(entries))} for {row_id}"
        )
    diversity = {"all_backbones": all_backbone_metrics}
    for mode in ("best_of_8", "all_refolds"):
        indices = [i for i, entry in enumerate(entries) if entry[f"{mode}_designable"]]
        diversity[mode] = _subset_diversity(
            matrix, indices, row_output / "structural_diversity" / mode, mode
        )
    sequence = _sequence_metrics(
        cohort, row_output, matrix, mmseqs_bin, 0.1, 0.7, mmseqs_threads
    )
    result = {
        "protocol": {
            "sc_rmsd_threshold_A": threshold,
            "expected_refolds_per_backbone": expected_refolds,
            "sc_atom": "CA",
            "pLDDT_gate": False,
            "tmalign_novelty_target": str(canonical_targets),
            "tmalign_novelty_score": "query-normalized Chain_1 score",
            "novelty_threshold": 0.5,
            "structural_tm_symmetrization": "min",
            "structural_linkage": "complete",
            "structural_threshold_grid": "101 points from 0 to 1 inclusive",
            "sequence_identity_threshold": 0.1,
            "sequence_coverage_threshold": 0.7,
            "sequence_mmseqs_cov_mode": 0,
            "cache_source_root": str(cache_source_root) if cache_source_root else None,
        },
        "row": row,
        "designability": designability,
        "novelty": novelty,
        "diversity": diversity,
        "sequence": sequence,
    }
    _json_dump(metrics_path, result)
    return result


def _manifest_row(
    *,
    output_root: Path,
    row_id: str,
    model: str,
    label: str,
    cohort: str,
    step: int | str = "",
    checkpoint: str = "",
    length_min: int | None = None,
    length_max: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_id": row_id,
        "model": model,
        "label": label,
        "step": step,
        "checkpoint": checkpoint,
        "output_root": str(output_root),
        "cohort": cohort,
    }
    if length_min is not None:
        row["length_min"] = length_min
    if length_max is not None:
        row["length_max"] = length_max
    return row


def build_input_manifest(
    *,
    self_consistency_root: Path | None,
    summary_csv: Path | None,
    model: str,
    label: str,
    cohort: str,
    row_id: str | None,
    length_min: int | None,
    length_max: int | None,
) -> list[dict[str, Any]]:
    """Build model-agnostic rows from one result root or a sweep summary."""
    if (self_consistency_root is None) == (summary_csv is None):
        raise ValueError(
            "Specify exactly one of --self-consistency-root or --summary-csv"
        )
    if self_consistency_root is not None:
        resolved_id = row_id or self_consistency_root.name
        return [
            _manifest_row(
                output_root=self_consistency_root,
                row_id=resolved_id,
                model=model,
                label=label,
                cohort=cohort,
                length_min=length_min,
                length_max=length_max,
            )
        ]

    assert summary_csv is not None
    rows: list[dict[str, Any]] = []
    for summary_row in sorted(
        _read_summary(summary_csv), key=lambda item: int(item["step"])
    ):
        step = int(summary_row["step"])
        rows.append(
            _manifest_row(
                output_root=Path(summary_row["output_root"]),
                row_id=f"{model}_step_{step:08d}",
                model=model,
                label=label,
                cohort=cohort,
                step=step,
                checkpoint=summary_row.get("ckpt_path")
                or summary_row.get("step_name", ""),
                length_min=length_min,
                length_max=length_max,
            )
        )
    if not rows:
        raise ValueError(f"No completed checkpoints found in {summary_csv}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    _json_dump(path, {"protocol": "scope_paper_metrics_v1", "rows": rows})
    print(f"wrote {len(rows)} rows to {path}", flush=True)


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.get("row", {}))
    flat: dict[str, Any] = {}
    flat.update(row)
    designability = result["designability"]
    flat.update(
        {
            f"designability_{key}": value
            for key, value in designability.items()
            if key != "all_refolds_definition"
        }
    )
    for mode, values in result["novelty"].items():
        flat.update(
            {
                f"novelty_{mode}_{key}": value
                for key, value in values.items()
                if key != "criterion"
            }
        )
    for mode, values in result["diversity"].items():
        flat.update({f"diversity_{mode}_{key}": value for key, value in values.items()})
    for mode, values in result["sequence"].items():
        flat.update(
            {
                f"sequence_{mode}_{key}": value
                for key, value in values.items()
                if key not in {"sequence_source", "joint_definition"}
            }
        )
    return flat


def _paper_metric_row(result: dict[str, Any]) -> dict[str, Any]:
    """Return the focused manuscript-facing metrics for one evaluated cohort."""
    flat = _flatten(result)
    identity = {
        key: flat.get(key, "")
        for key in (
            "row_id",
            "model",
            "label",
            "cohort",
            "step",
            "checkpoint",
            "output_root",
        )
    }
    return {
        **identity,
        "n_backbones": flat["designability_n_backbones"],
        "designability_count": flat["designability_best_of_8_count"],
        "designability_rate": flat["designability_best_of_8_rate"],
        "per_sequence_success_count": flat["designability_per_sequence_success_count"],
        "per_sequence_success_total": flat["designability_per_sequence_success_total"],
        "per_sequence_success_rate": flat["designability_per_sequence_success_rate"],
        "novel_designable_count": flat["novelty_best_of_8_novel_count_max_tm_lt_0_5"],
        "novelty_rate": flat["novelty_best_of_8_novel_rate_max_tm_lt_0_5"],
        "mean_nearest_train_tm": flat["novelty_best_of_8_mean_max_tmalign_tm_query"],
        "distinct_tm_clusters": flat["diversity_best_of_8_n_clusters_at_0_5"],
        "sequence_clusters": flat["sequence_best_of_8_n_clusters"],
        "joint_sequence_structure_clusters": flat[
            "sequence_best_of_8_joint_n_clusters"
        ],
    }


def _read_results(
    output_root: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(
            row.get("row_id") or row.get("label") or Path(row["output_root"]).name
        )
        path = output_root / _safe_name(row_id) / "advanced_metrics.json"
        if not path.is_file():
            print(f"missing result: {path}", file=sys.stderr)
            continue
        results.append(json.loads(path.read_text()))
    return results


def _plot_sweeps(
    results: list[dict[str, Any]],
    output_dir: Path,
    designability_filtered_diversity: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    sweep = [
        result
        for result in results
        if result.get("row", {}).get("cohort") == "checkpoint_sweep"
    ]
    if not sweep:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {
        float(result.get("protocol", {}).get("sc_rmsd_threshold_A")) for result in sweep
    }
    if len(thresholds) != 1:
        raise ValueError(
            f"Sweep results contain mixed scRMSD thresholds: {sorted(thresholds)}"
        )
    sc_rmsd_threshold = thresholds.pop()

    def display_step(row: dict[str, Any]) -> float:
        """Return the comparable training-step coordinate for sweep plots.

        The three Zatom-2 fine-tuning runs start from pretrained models at
        raw step 100k, whereas the scratch run starts at raw step 0.  For
        trajectory comparisons, plot fine-tune progress relative to that
        initialization point.  Keep RFD3 on its actual retained checkpoint
        coordinates: its first available sweep checkpoint is 5k.
        """
        step = float(row.get("step") or 0)
        if str(row.get("model", "")).startswith("zatom2_joint_"):
            step -= 100000.0
        return step / 1000.0

    for result in sweep:
        result["_flat"] = _flatten(result)

    plotted_x = [display_step(result["_flat"]) for result in sweep]
    x_min = min(plotted_x) if plotted_x else 0.0
    x_max = max(plotted_x) if plotted_x else 50.0
    x_pad = max(1.0, 0.02 * (x_max - x_min))
    common_xlim = (x_min - x_pad, x_max + x_pad)
    common_xticks = np.arange(
        math.floor(x_min / 5.0) * 5.0, math.ceil(x_max / 5.0) * 5.0 + 0.1, 5.0
    )

    def undefined_at_zero(row: dict[str, Any], key: str) -> bool:
        """Whether a conditional metric is undefined at plotted x=0."""
        if key.startswith("designability_") or "all_backbones" in key:
            return False
        if "best_of_8" in key:
            return _float(row.get("designability_best_of_8_count")) <= 0.0
        if "all_refolds" in key:
            return _float(row.get("designability_all_refolds_count")) <= 0.0
        return False

    if designability_filtered_diversity:
        structural_population_labels = [
            (
                "diversity_all_refolds_n_clusters_at_0_5",
                "Diversity – distinct clusters (TM; strict all-refolds designable) ↑",
                False,
            ),
            (
                "diversity_all_refolds_cluster_fraction_at_0_5",
                "Diversity rate (TM; strict all-refolds designable) ↑",
                False,
            ),
            (
                "diversity_best_of_8_n_clusters_at_0_5",
                "Diversity – distinct clusters (TM; best-of-8 designable) ↑",
                False,
            ),
            (
                "diversity_best_of_8_cluster_fraction_at_0_5",
                "Diversity rate (TM; best-of-8 designable) ↑",
                False,
            ),
            (
                "diversity_all_refolds_tmalign_diversity_auc",
                "Diversity AUC – structural (strict all-refolds designable) ↑",
                False,
            ),
            (
                "diversity_best_of_8_tmalign_diversity_auc",
                "Diversity AUC – structural (best-of-8 designable) ↑",
                False,
            ),
        ]
    else:
        structural_population_labels = [
            (
                ALL_BACKBONE_CLUSTER_COUNT_KEY,
                "Diversity – distinct clusters (TM; all backbones) ↑",
                False,
            ),
            (
                ALL_BACKBONE_CLUSTER_FRACTION_KEY,
                "Diversity rate (TM; all backbones) ↑",
                False,
            ),
            (
                "diversity_best_of_8_n_clusters_at_0_5",
                "Diversity – distinct clusters (TM; best-of-8 designable) ↑",
                False,
            ),
            (
                "diversity_best_of_8_cluster_fraction_at_0_5",
                "Diversity rate (TM; best-of-8 designable) ↑",
                False,
            ),
            (
                "diversity_all_backbones_diversity_auc",
                "Diversity AUC – structural (all backbones) ↑",
                False,
            ),
            (
                "diversity_best_of_8_tmalign_diversity_auc",
                "Diversity AUC – structural (best-of-8 designable) ↑",
                False,
            ),
        ]
    labels = [
        ("designability_best_of_8_rate", "Designability fraction (best-of-8) ↑", False),
        (
            "designability_all_refolds_rate",
            "Designability fraction (strict all-refolds) ↑",
            False,
        ),
        (
            "novelty_best_of_8_mean_max_tmalign_tm_query",
            "Novelty (nearest TM score; best-of-8 designable) ↓",
            True,
        ),
        (
            "novelty_all_refolds_mean_max_tmalign_tm_query",
            "Novelty (nearest TM score; all-refolds designable) ↓",
            True,
        ),
        (
            "novelty_best_of_8_novel_rate_max_tm_lt_0_5",
            "Novelty rate (TM score < 0.5; best-of-8 designable) ↑",
            False,
        ),
        (
            "novelty_all_refolds_novel_rate_max_tm_lt_0_5",
            "Novelty rate (TM score < 0.5; all-refolds designable) ↑",
            False,
        ),
        *structural_population_labels,
        (
            "sequence_all_refolds_n_clusters",
            "Diversity – distinct clusters (sequence; all-refolds) ↑",
            False,
        ),
        (
            "sequence_best_of_8_n_clusters",
            "Diversity – distinct clusters (sequence; best-of-8 designable) ↑",
            False,
        ),
        (
            "sequence_all_refolds_joint_n_clusters",
            "Diversity – distinct clusters (joint sequence–structure; all-refolds) ↑",
            False,
        ),
        (
            "sequence_best_of_8_joint_n_clusters",
            "Diversity – distinct clusters (joint sequence–structure; best-of-8 designable) ↑",
            False,
        ),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(24, 22), constrained_layout=True)
    for ax, (key, ylabel, lower_better) in zip(axes.flat, labels):
        has_undefined_zero = False
        for model in [
            "zatom2_scratch",
            "zatom2_joint_fc",
            "zatom2_joint_properties_fc",
            "zatom2_joint_no_fc",
            "rfd3",
        ]:
            rows = [
                result["_flat"]
                for result in sweep
                if result.get("row", {}).get("model") == model
                and key in result["_flat"]
            ]
            rows = sorted(
                rows,
                key=lambda row: (
                    float(row.get("step") or 0),
                    str(row.get("row_id", "")),
                ),
            )
            if not rows:
                continue
            x = [display_step(row) for row in rows]
            y = [_float(row.get(key)) for row in rows]
            finite = np.isfinite(np.asarray(y, dtype=float))
            if np.any(finite):
                ax.plot(
                    np.asarray(x)[finite],
                    np.asarray(y)[finite],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=rows[0].get("label", model),
                    color=PLOT_COLORS.get(model),
                )
            if any(
                abs(x_i) < 1e-9 and undefined_at_zero(row, key)
                for x_i, row in zip(x, rows)
            ):
                has_undefined_zero = True
        ax.set_ylabel(ylabel, fontsize=9, labelpad=7)
        ax.set_xlabel("Comparable training step (10^3)")
        ax.set_xlim(*common_xlim)
        ax.set_xticks(common_xticks)
        ax.grid(alpha=0.25)
        if lower_better:
            ax.text(
                0.02, 0.05, "lower = more novel", transform=ax.transAxes, fontsize=9
            )
        if has_undefined_zero:
            ax.axvline(0.0, color="0.35", linestyle=":", linewidth=0.9, alpha=0.8)
            ax.text(
                0.03,
                0.93,
                "N/A at x=0\n(0 designable)",
                transform=ax.transAxes,
                fontsize=7.5,
                color="0.30",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.75,
                    "pad": 2.0,
                },
            )
    for ax in axes.flat[len(labels) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best", fontsize=8)
    population_note = (
        "designability-filtered diversity; " if designability_filtered_diversity else ""
    )
    fig.suptitle(
        f"SCOPe-2k advanced novelty/diversity audit: Zatom-2 and RFD3 sweeps "
        f"({population_note}CA scRMSD < {sc_rmsd_threshold:g} Å)",
        fontsize=18,
        fontweight="bold",
    )
    grid_stem = (
        "advanced_metrics_grid_designability_filtered"
        if designability_filtered_diversity
        else "advanced_metrics_grid"
    )
    fig.savefig(output_dir / f"{grid_stem}.png", dpi=220)
    fig.savefig(output_dir / f"{grid_stem}.pdf")
    if not designability_filtered_diversity:
        fig.savefig(output_dir / "advanced_metrics_grid_all_metrics.png", dpi=220)
    plt.close(fig)

    # Paper-focused view of panels 1, 3, and 16 from the full audit grid.  These
    # expose the late-training designability/novelty/diversity trade-off without
    # reproducing every diagnostic panel in the appendix.
    focused_metrics = [
        ("designability_best_of_8_rate", "Designability\n(best of 8) ↑", False),
        (
            "novelty_best_of_8_mean_max_tmalign_tm_query",
            "Nearest-training TM\n(designable) ↓",
            True,
        ),
        (
            "sequence_best_of_8_joint_n_clusters",
            "Joint clusters\n(designable) ↑",
            False,
        ),
    ]
    focused_models = [
        "zatom2_scratch",
        "zatom2_joint_fc",
        "zatom2_joint_properties_fc",
        "zatom2_joint_no_fc",
        "rfd3",
    ]
    focused_names = {
        "zatom2_scratch": "Zatom-2 from scratch",
        "zatom2_joint_fc": "Gen. + structure",
        "zatom2_joint_properties_fc": "Gen. + structure + MLIP",
        "zatom2_joint_no_fc": "Gen. + structure (no FC)",
        "rfd3": "RFD3/EDM from scratch",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (key, ylabel, lower_better) in zip(axes, focused_metrics):
        has_undefined_zero = False
        for model in focused_models:
            rows = [
                result["_flat"]
                for result in sweep
                if result.get("row", {}).get("model") == model
                and key in result["_flat"]
            ]
            rows = sorted(
                rows,
                key=lambda row: (
                    float(row.get("step") or 0),
                    str(row.get("row_id", "")),
                ),
            )
            if not rows:
                continue
            x = [display_step(row) for row in rows]
            y = [_float(row.get(key)) for row in rows]
            finite = np.isfinite(np.asarray(y, dtype=float))
            if np.any(finite):
                ax.plot(
                    np.asarray(x)[finite],
                    np.asarray(y)[finite],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=focused_names[model],
                    color=PLOT_COLORS.get(model),
                )
            if any(
                abs(x_i) < 1e-9 and undefined_at_zero(row, key)
                for x_i, row in zip(x, rows)
            ):
                has_undefined_zero = True
        ax.set_ylabel(ylabel)
        ax.set_xlabel("SCOPe fine-tuning step (10³)")
        ax.set_xlim(*common_xlim)
        ax.set_xticks(common_xticks)
        ax.grid(alpha=0.25)
        if lower_better:
            ax.text(
                0.02, 0.05, "lower = more novel", transform=ax.transAxes, fontsize=8
            )
        if has_undefined_zero:
            ax.axvline(0.0, color="0.35", linestyle=":", linewidth=0.9, alpha=0.8)
            ax.text(
                0.03,
                0.93,
                "N/A at step 0\n(0 designable)",
                transform=ax.transAxes,
                fontsize=7.5,
                color="0.30",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.75,
                    "pad": 2.0,
                },
            )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        f"SCOPe-2k checkpoint trajectories (CA scRMSD < {sc_rmsd_threshold:g} Å)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.91))
    fig.savefig(
        output_dir / "scope_transfer_overfitting_sweep.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "scope_transfer_overfitting_sweep.pdf", bbox_inches="tight"
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        [
            "designability_best_of_8_rate",
            "novelty_best_of_8_novel_rate_max_tm_lt_0_5",
            "diversity_best_of_8_tmalign_diversity_auc",
        ],
        [
            f"Designability fraction (scRMSD < {sc_rmsd_threshold:g} Å)",
            "Exact TMalign novelty rate (max TM < 0.5)",
            "Structural diversity AUC",
        ],
    ):
        for model in [
            "zatom2_scratch",
            "zatom2_joint_fc",
            "zatom2_joint_properties_fc",
            "zatom2_joint_no_fc",
            "rfd3",
        ]:
            rows = sorted(
                [
                    result["_flat"]
                    for result in sweep
                    if result.get("row", {}).get("model") == model
                ],
                key=lambda row: float(row.get("step") or 0),
            )
            if not rows:
                continue
            ax.plot(
                [display_step(row) for row in rows],
                [_float(row.get(key)) for row in rows],
                marker="o",
                linewidth=2,
                color=PLOT_COLORS.get(model),
                label=rows[0].get("label", model),
            )
        ax.set_title(title)
        ax.set_xlabel("Comparable training step (10^3)")
        ax.set_xlim(*common_xlim)
        ax.set_xticks(common_xticks)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"SCOPe transfer trajectories under the scRMSD < {sc_rmsd_threshold:g} Å protocol",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_dir / "advanced_scope_transfer_trajectories.png", dpi=220)
    fig.savefig(output_dir / "advanced_scope_transfer_trajectories.pdf")
    plt.close(fig)


def _write_extended_tables(results: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [_flatten(result) for result in results]
    _csv_write(output_dir / "table3_extended_metrics.csv", rows)
    columns = [
        "row_id",
        "label",
        "cohort",
        "model",
        "n_backbones",
        "designability_best_of_8_count",
        "designability_best_of_8_rate",
        "designability_all_refolds_count",
        "designability_all_refolds_rate",
        "novelty_best_of_8_mean_max_tmalign_tm_query",
        "novelty_best_of_8_novel_count_max_tm_lt_0_5",
        "novelty_best_of_8_novel_rate_max_tm_lt_0_5",
        "novelty_all_refolds_mean_max_tmalign_tm_query",
        "novelty_all_refolds_novel_count_max_tm_lt_0_5",
        "novelty_all_refolds_novel_rate_max_tm_lt_0_5",
        "diversity_all_backbones_n_clusters_at_0_5",
        "diversity_all_backbones_tmalign_diversity_auc",
        "diversity_best_of_8_n_clusters_at_0_5",
        "diversity_best_of_8_tmalign_diversity_auc",
        "diversity_all_refolds_n_clusters_at_0_5",
        "diversity_all_refolds_tmalign_diversity_auc",
        "sequence_best_of_8_n_clusters",
        "sequence_best_of_8_joint_n_clusters",
        "sequence_all_refolds_n_clusters",
        "sequence_all_refolds_joint_n_clusters",
    ]
    with (output_dir / "table3_extended_metrics.md").open("w") as handle:
        handle.write("# Table 3 advanced novelty/diversity extension\n\n")
        threshold = results[0].get("protocol", {}).get("sc_rmsd_threshold_A", "?")
        handle.write(
            f"All designability filters use CA scRMSD < {threshold:g} Å and no pLDDT gate. `all_refolds` is the strict all-eight filter. Exact novelty uses query-normalized TMalign against the canonical SCOPe-2k train set.\n\n"
        )
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(str(row.get(column, "")) for column in columns)
                + " |\n"
            )


def merge_results(
    manifest: Path,
    output_root: Path,
    plots_root: Path,
    designability_filtered_diversity_grid: bool = False,
) -> None:
    rows = _read_manifest(manifest)
    results = _read_results(output_root, rows)
    if not results:
        raise RuntimeError("No completed advanced metrics results to merge")
    cohort_names = {result.get("row", {}).get("cohort") for result in results}
    _csv_write(
        output_root / "advanced_metrics_summary.csv",
        [_flatten(result) for result in results],
    )
    _csv_write(
        output_root / "paper_metrics.csv",
        [_paper_metric_row(result) for result in results],
    )
    if "checkpoint_sweep" in cohort_names:
        _plot_sweeps(
            results,
            plots_root,
            designability_filtered_diversity=designability_filtered_diversity_grid,
        )
    if any(name and name.startswith("table3") for name in cohort_names):
        _write_extended_tables(results, output_root)
    _json_dump(
        output_root / "merge_manifest.json",
        {
            "n_results": len(results),
            "cohorts": sorted(str(name) for name in cohort_names),
        },
    )
    print(f"merged {len(results)} result rows under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--self-consistency-root", type=Path)
    source.add_argument("--summary-csv", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", default="model")
    parser.add_argument("--label", default="Model")
    parser.add_argument("--cohort")
    parser.add_argument("--row-id")
    parser.add_argument("--length-min", type=int)
    parser.add_argument("--length-max", type=int)
    parser.add_argument("--plots-root", type=Path)
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--canonical-target-dir", type=Path)
    parser.add_argument(
        "--tmalign-bin", default=os.environ.get("TMALIGN_BIN", "TMalign")
    )
    parser.add_argument("--mmseqs-bin", default=os.environ.get("MMSEQS_BIN", "mmseqs"))
    parser.add_argument("--sc-rmsd-threshold", type=float, default=1.5)
    parser.add_argument("--expected-refolds", type=int, default=8)
    parser.add_argument("--pair-workers", type=int, default=8)
    parser.add_argument("--tmalign-chunk-size", type=int, default=8)
    parser.add_argument("--mmseqs-threads", type=int, default=8)
    parser.add_argument("--cache-source-root", type=Path)
    parser.add_argument("--designability-filtered-diversity-grid", action="store_true")
    args = parser.parse_args()

    if args.expected_refolds <= 0:
        parser.error("--expected-refolds must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, --num-workers)")
    if args.length_min is not None and args.length_max is not None:
        if args.length_min > args.length_max:
            parser.error("--length-min cannot exceed --length-max")

    source_requested = (
        args.self_consistency_root is not None or args.summary_csv is not None
    )
    if source_requested:
        cohort = args.cohort or (
            "checkpoint_selection"
            if args.summary_csv is not None
            else "paper_evaluation"
        )
        rows = build_input_manifest(
            self_consistency_root=args.self_consistency_root,
            summary_csv=args.summary_csv,
            model=args.model,
            label=args.label,
            cohort=cohort,
            row_id=args.row_id,
            length_min=args.length_min,
            length_max=args.length_max,
        )
        if args.manifest_out is not None:
            write_manifest(args.manifest_out, rows)
            return
        if args.output_root is None:
            base = (
                args.self_consistency_root
                if args.self_consistency_root is not None
                else args.summary_csv.parent
            )
            args.output_root = base / "paper_metrics"
        args.manifest = args.output_root / "manifest.json"
        write_manifest(args.manifest, rows)
    elif args.manifest_out is not None:
        parser.error("--manifest-out requires --self-consistency-root or --summary-csv")

    if args.manifest is None or args.output_root is None:
        parser.error(
            "specify a source (--self-consistency-root/--summary-csv), or both "
            "--manifest and --output-root"
        )
    if args.merge:
        merge_results(
            args.manifest,
            args.output_root,
            args.plots_root or args.output_root / "plots",
            designability_filtered_diversity_grid=args.designability_filtered_diversity_grid,
        )
        return
    if args.canonical_target_dir is None:
        parser.error("metric evaluation requires --canonical-target-dir")
    rows = _read_manifest(args.manifest)
    worker_index = args.worker_index if args.worker_index is not None else 0
    assigned = [
        row
        for index, row in enumerate(rows)
        if index % args.num_workers == worker_index
    ]
    for index, row in enumerate(assigned, start=1):
        print(
            f"worker {worker_index}: {index}/{len(assigned)} {row.get('row_id')}",
            flush=True,
        )
        evaluate_row(
            row,
            args.output_root,
            args.canonical_target_dir,
            args.tmalign_bin,
            args.mmseqs_bin,
            args.sc_rmsd_threshold,
            args.pair_workers,
            args.tmalign_chunk_size,
            args.mmseqs_threads,
            args.expected_refolds,
            args.cache_source_root,
        )
    print(f"worker {worker_index}: completed {len(assigned)} rows", flush=True)
    if args.worker_index is None and args.num_workers == 1:
        merge_results(
            args.manifest,
            args.output_root,
            args.plots_root or args.output_root / "plots",
            designability_filtered_diversity_grid=(
                args.designability_filtered_diversity_grid
            ),
        )


if __name__ == "__main__":
    main()
