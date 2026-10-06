"""Protein diversity: cluster-based structural and sequence diversity.

Both metrics follow the FrameDiff / Chroma convention:
  Structural diversity = FoldSeek clusters / total samples
  Sequence diversity   = MMseqs2 clusters / total samples

Higher = more diverse. A value of 1.0 means every sample is in its own cluster.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from zevals.oracles.base import run_subprocess
from zevals.oracles.structure_search.foldseek import FoldSeekSearcher


def compute_structural_diversity(
    backbone_pdbs: list[Path],
    foldseek: FoldSeekSearcher,
    output_dir: Path,
    min_tm: float = 0.5,
) -> dict[str, float]:
    """FoldSeek all-vs-all cluster diversity.

    Args:
        backbone_pdbs: Paths to generated structure files (`.cif` or `.pdb`).
                       Should be the *designable* subset only (see
                       `designability.compute_designability`'s second return
                       value) — diversity is only meaningful over structures
                       that are themselves physically plausible.
        foldseek:      FoldSeek oracle (injected from config).
        output_dir:    Directory for cluster output files.
        min_tm:        TM-score threshold for clustering.

    Returns:
        {"structural_diversity": float in [0, 1], "n_clusters": int}
    """
    if not backbone_pdbs:
        return {"structural_diversity": 0.0, "n_clusters": 0}

    # FoldSeek clusters every structure file in the directory, which may include
    # non-designable backbones sharing the same parent dir — filter cluster
    # membership down to `backbone_pdbs` before counting, so the numerator
    # and denominator both describe the same (designable) population.
    pdb_dir = backbone_pdbs[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_tsv = output_dir / "foldseek_clusters.tsv"

    wanted_stems = {p.stem for p in backbone_pdbs}
    results = foldseek.cluster(pdb_dir, cluster_tsv, min_tm=min_tm)
    n_clusters = len(
        {r.cluster_id for r in results if r.cluster_id and r.query_id in wanted_stems}
    )
    n_total = len(backbone_pdbs)

    return {
        "structural_diversity": n_clusters / n_total if n_total else 0.0,
        "n_clusters": n_clusters,
    }


def compute_sequence_diversity(
    sequences: list[str],
    output_dir: Path,
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.3,
    dbtype: int = 1,
) -> dict[str, float]:
    """MMseqs2 cluster-based sequence diversity.

    Args:
        sequences:    Designed sequences (single-letter code) — amino acid
                      by default; pass real nucleotide sequences (A/C/G/T/U,
                      not a model's internal alphabet) with `dbtype=2` for
                      RNA/DNA (see `rna.diversity.compute_sequence_diversity`,
                      which does exactly that for NA-MPNN's designs).
        output_dir:   Scratch directory for MMseqs2 intermediate files.
        mmseqs_bin:   Path to MMseqs2 binary (default: 'mmseqs' on PATH).
        min_seq_id:   Sequence identity threshold for clustering.
        dbtype:       `mmseqs createdb --dbtype` — 1 (amino acid, default)
                      or 2 (nucleotide). Left at 0 (auto-detect) this would
                      silently misclassify short reduced-alphabet nucleotide
                      sequences as amino acid (verified: MMseqs2's heuristic
                      does exactly this for NA-MPNN's designs) — always pass
                      it explicitly rather than relying on auto-detect.

    Returns:
        {"sequence_diversity": float in [0, 1], "n_seq_clusters": int}
    """
    if not sequences:
        return {"sequence_diversity": 0.0, "n_seq_clusters": 0}

    # Unlike FoldSeek's `easy-cluster` (used above), plain `mmseqs cluster`
    # refuses to run if its output DB already exists — fatal for a directory
    # reused across validation epochs (ZevalSuite's primary use case), so
    # clear any stale DB from a previous call before starting.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / "sequences.fasta"
    lines = []
    for i, seq in enumerate(sequences):
        lines += [f">seq_{i}", seq]
    fasta_path.write_text("\n".join(lines) + "\n")

    db = output_dir / "seqdb"
    cluster_db = output_dir / "clusterdb"
    tmp = output_dir / "mmseqs_tmp"

    # `mmseqs cluster`'s nucleotide path (dbtype=2) runs an internal
    # extractframes step whose worker pool is sized off --threads; on a
    # high-core-count host (e.g. a 244-core login node) with a handful of
    # designed sequences, --threads defaulting to every visible core wildly
    # oversubscribes that step and crashes ("Extractframes died") — verified
    # directly. Capping threads to the dataset size (and a sane ceiling)
    # fixes it and costs nothing on a real-sized run.
    threads = max(1, min(os.cpu_count() or 1, 8, len(sequences)))

    subprocess.run(
        [mmseqs_bin, "createdb", str(fasta_path), str(db), "--dbtype", str(dbtype)],
        check=True,
    )
    subprocess.run(
        [
            mmseqs_bin,
            "cluster",
            str(db),
            str(cluster_db),
            str(tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--threads",
            str(threads),
        ],
        check=True,
    )

    tsv_path = output_dir / "clusters.tsv"
    subprocess.run(
        [mmseqs_bin, "createtsv", str(db), str(db), str(cluster_db), str(tsv_path)],
        check=True,
    )

    clusters = set()
    for line in tsv_path.read_text().splitlines():
        parts = line.split("\t")
        if parts:
            clusters.add(parts[0])

    n_clusters = len(clusters)
    n_total = len(sequences)
    return {
        "sequence_diversity": n_clusters / n_total if n_total else 0.0,
        "n_seq_clusters": n_clusters,
    }


def compute_tmalign_complete_linkage_diversity(
    backbone_pdbs: list[Path],
    output_dir: Path,
    tmalign_bin: str = "TMalign",
    tm_thresholds: list[float] | np.ndarray | None = None,
    symmetrize: str = "min",
    chunk_size: int = 8,
    n_workers: int = 1,
) -> dict[str, float]:
    """TM-align complete-linkage diversity AUC over TM-score thresholds.

    This is the pairwise-TM metric described by the user: run all-vs-all
    TM-align, symmetrize the asymmetric TM matrix, form complete-linkage
    clusters where every pair in a cluster must have TM >= threshold, sweep
    thresholds from 0 to 1, and integrate cluster_fraction vs threshold.

    Args:
        backbone_pdbs: Structure files to compare. Pass all generated
                       backbones for unconditional-sample diversity, or a
                       filtered subset if matching an older designable-only
                       protocol.
        output_dir: Directory for the raw TMalign output, TM matrix, and
                    threshold curve.
        tmalign_bin: Path to TMalign from USalign.
        tm_thresholds: Optional threshold grid. Defaults to 101 evenly spaced
                       points from 0.0 to 1.0.
        symmetrize: How to combine TMalign's length-normalized asymmetric
                    pair scores. ``"min"`` is the conservative default implied
                    by complete-linkage "all pairs above threshold" wording.
        chunk_size: Number of query structures per TMalign directory-mode call.
                    Chunks are written independently, so reruns can resume.
        n_workers: Number of TMalign chunk subprocesses to run concurrently.

    Returns:
        Numeric metrics including ``tmalign_diversity_auc``. Higher AUC means
        a more diverse set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(backbone_pdbs)
    if n_total == 0:
        return {
            "tmalign_diversity_auc": 0.0,
            "tmalign_cluster_fraction_at_0.5": 0.0,
            "tmalign_n_clusters_at_0.5": 0.0,
            "tmalign_mean_pairwise_tm": float("nan"),
            "tmalign_median_pairwise_tm": float("nan"),
            "tmalign_n_structures": 0.0,
            "tmalign_n_missing_pair_scores": 0.0,
        }

    thresholds = _threshold_grid(tm_thresholds)
    if n_total == 1:
        ids = [backbone_pdbs[0].stem]
        tm_matrix = np.ones((1, 1), dtype=float)
        cluster_counts = np.ones_like(thresholds, dtype=int)
        fractions = np.ones_like(thresholds, dtype=float)
        _write_tm_matrix(output_dir / "tmalign_tm_matrix.csv", ids, tm_matrix)
        _write_threshold_curve(
            output_dir / "tmalign_complete_linkage_curve.csv",
            thresholds,
            cluster_counts,
            fractions,
        )
        return {
            "tmalign_diversity_auc": _trapezoid_auc(fractions, thresholds),
            "tmalign_cluster_fraction_at_0.5": 1.0,
            "tmalign_n_clusters_at_0.5": 1.0,
            "tmalign_mean_pairwise_tm": float("nan"),
            "tmalign_median_pairwise_tm": float("nan"),
            "tmalign_n_structures": 1.0,
            "tmalign_n_missing_pair_scores": 0.0,
        }

    staged_dir = output_dir / "tmalign_inputs"
    staged_paths, ids = _stage_tmalign_inputs(backbone_pdbs, staged_dir)
    tm_matrix, missing_pairs = _run_tmalign_all_vs_all(
        staged_dir=staged_dir,
        staged_paths=staged_paths,
        ids=ids,
        output_tsv=output_dir / "tmalign_pairwise.tsv",
        tmalign_bin=tmalign_bin,
        chunk_size=chunk_size,
        n_workers=n_workers,
    )
    sym_tm = _symmetrize_tm_matrix(tm_matrix, symmetrize=symmetrize)
    merge_tms = _complete_linkage_merge_tms(sym_tm)
    cluster_counts, fractions = _cluster_curve_from_merges(
        n_total=n_total,
        merge_tms=merge_tms,
        thresholds=thresholds,
    )

    _write_tm_matrix(output_dir / "tmalign_tm_matrix.csv", ids, sym_tm)
    _write_threshold_curve(
        output_dir / "tmalign_complete_linkage_curve.csv",
        thresholds,
        cluster_counts,
        fractions,
    )

    upper = sym_tm[np.triu_indices(n_total, k=1)]
    clusters_at_half = _cluster_count_at_threshold(n_total, merge_tms, threshold=0.5)
    return {
        "tmalign_diversity_auc": _trapezoid_auc(fractions, thresholds),
        "tmalign_cluster_fraction_at_0.5": float(clusters_at_half / n_total),
        "tmalign_n_clusters_at_0.5": float(clusters_at_half),
        "tmalign_mean_pairwise_tm": float(np.mean(upper)),
        "tmalign_median_pairwise_tm": float(np.median(upper)),
        "tmalign_n_structures": float(n_total),
        "tmalign_n_missing_pair_scores": float(missing_pairs),
    }


def _threshold_grid(
    tm_thresholds: list[float] | np.ndarray | None,
) -> np.ndarray:
    thresholds = (
        np.linspace(0.0, 1.0, 101, dtype=float)
        if tm_thresholds is None
        else np.asarray(tm_thresholds, dtype=float)
    )
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("tm_thresholds must be a non-empty 1D sequence")
    thresholds = np.unique(thresholds)
    if thresholds[0] < 0.0 or thresholds[-1] > 1.0:
        raise ValueError("TM-score thresholds must be within [0, 1]")
    return thresholds


def _stage_tmalign_inputs(
    backbone_pdbs: list[Path],
    staged_dir: Path,
) -> tuple[list[Path], list[str]]:
    shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []
    ids: list[str] = []
    for idx, src in enumerate(backbone_pdbs):
        if not src.is_file():
            raise FileNotFoundError(src)
        suffix = src.suffix if src.suffix else ".pdb"
        stem = f"sample_{idx:05d}_{src.stem}"
        dst = staged_dir / f"{stem}{suffix}"
        dst.symlink_to(src.resolve())
        staged_paths.append(dst)
        ids.append(stem)
    return staged_paths, ids


def _run_tmalign_all_vs_all(
    staged_dir: Path,
    staged_paths: list[Path],
    ids: list[str],
    output_tsv: Path,
    tmalign_bin: str,
    chunk_size: int,
    n_workers: int,
) -> tuple[np.ndarray, int]:
    n_total = len(staged_paths)
    tm_matrix = np.zeros((n_total, n_total), dtype=float)
    np.fill_diagonal(tm_matrix, 1.0)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")

    target_list = output_tsv.parent / "tmalign_target_list.txt"
    target_list.write_text("\n".join(path.name for path in staged_paths) + "\n")
    chunk_dir = output_tsv.parent / "tmalign_pairwise_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[int, list[Path], Path, Path]] = []
    for chunk_idx, start in enumerate(range(0, n_total, chunk_size)):
        query_paths = staged_paths[start : start + chunk_size]
        query_list = chunk_dir / f"chunk_{chunk_idx:04d}_queries.txt"
        query_list.write_text("\n".join(path.name for path in query_paths) + "\n")
        chunk_output = chunk_dir / f"chunk_{chunk_idx:04d}.tsv"
        chunks.append((chunk_idx, query_paths, query_list, chunk_output))

    pending = [chunk for chunk in chunks if not _chunk_complete(chunk[3])]
    if pending:
        print(
            f"Running TMalign in {len(chunks)} chunks "
            f"({len(pending)} pending, chunk_size={chunk_size}, n_workers={n_workers})",
            flush=True,
        )
    else:
        print(
            f"All {len(chunks)} TMalign chunks already complete; reusing them",
            flush=True,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _run_tmalign_chunk,
                    tmalign_bin,
                    staged_dir,
                    target_list,
                    query_list,
                    chunk_output,
                ): chunk_idx
                for chunk_idx, _query_paths, query_list, chunk_output in pending
            }
            completed = len(chunks) - len(pending)
            for future in as_completed(futures):
                chunk_idx = futures[future]
                future.result()
                completed += 1
                print(
                    f"TMalign chunk {chunk_idx + 1}/{len(chunks)} complete "
                    f"({completed}/{len(chunks)} total)",
                    flush=True,
                )

    with output_tsv.open("w") as out_handle:
        for _chunk_idx, _query_paths, _query_list, chunk_output in chunks:
            chunk_text = chunk_output.read_text()
            out_handle.write(chunk_text)
            if not chunk_text.endswith("\n"):
                out_handle.write("\n")

    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}
    seen_pairs: set[tuple[int, int]] = set()
    for row in _parse_tmalign_outfmt2(output_tsv.read_text()):
        query_id, target_id, tm_query, tm_target = row
        if query_id not in id_to_idx or target_id not in id_to_idx:
            continue
        i = id_to_idx[query_id]
        j = id_to_idx[target_id]
        if i == j:
            continue
        tm_matrix[i, j] = max(tm_matrix[i, j], tm_query)
        tm_matrix[j, i] = max(tm_matrix[j, i], tm_target)
        seen_pairs.add((min(i, j), max(i, j)))

    expected_pairs = n_total * (n_total - 1) // 2
    missing_pairs = expected_pairs - len(seen_pairs)
    return tm_matrix, missing_pairs


def _chunk_complete(chunk_output: Path) -> bool:
    return chunk_output.is_file() and chunk_output.stat().st_size > 0


def _run_tmalign_chunk(
    tmalign_bin: str,
    staged_dir: Path,
    target_list: Path,
    query_list: Path,
    chunk_output: Path,
) -> None:
    tmp_output = chunk_output.with_suffix(".tmp")
    cmd = [
        tmalign_bin,
        "-dir1",
        f"{staged_dir}/",
        str(query_list),
        "-dir2",
        f"{staged_dir}/",
        str(target_list),
        "-outfmt",
        "2",
    ]
    result = run_subprocess(cmd, capture_output=True, text=True)
    tmp_output.write_text(result.stdout)
    tmp_output.replace(chunk_output)


def _parse_tmalign_outfmt2(stdout: str) -> list[tuple[str, str, float, float]]:
    rows: list[tuple[str, str, float, float]] = []
    for line in stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            parts = line.split()
        if len(parts) < 4:
            continue
        try:
            query_id = Path(parts[0]).stem
            target_id = Path(parts[1]).stem
            tm_query = float(parts[2])
            tm_target = float(parts[3])
        except ValueError:
            continue
        rows.append((query_id, target_id, tm_query, tm_target))
    return rows


def _symmetrize_tm_matrix(tm_matrix: np.ndarray, symmetrize: str = "min") -> np.ndarray:
    if symmetrize == "min":
        sym_tm = np.minimum(tm_matrix, tm_matrix.T)
    elif symmetrize == "max":
        sym_tm = np.maximum(tm_matrix, tm_matrix.T)
    elif symmetrize == "mean":
        sym_tm = (tm_matrix + tm_matrix.T) / 2.0
    else:
        raise ValueError("symmetrize must be one of: min, max, mean")
    sym_tm = np.clip(sym_tm, 0.0, 1.0)
    np.fill_diagonal(sym_tm, 1.0)
    return sym_tm


def _complete_linkage_merge_tms(sym_tm: np.ndarray) -> np.ndarray:
    """Return complete-linkage merge similarities from a symmetric TM matrix."""
    n_total = sym_tm.shape[0]
    if n_total <= 1:
        return np.array([], dtype=float)

    active = set(range(n_total))
    distances: dict[tuple[int, int], float] = {}
    for i in range(n_total):
        for j in range(i + 1, n_total):
            distances[(i, j)] = 1.0 - float(sym_tm[i, j])

    merge_distances: list[float] = []
    next_cluster_id = n_total
    while len(active) > 1:
        (left, right), best_distance = min(
            distances.items(),
            key=lambda item: (item[1], item[0]),
        )
        merge_distances.append(best_distance)

        remaining = sorted(active - {left, right})
        new_distances = {
            cluster_id: max(
                distances[_pair_key(left, cluster_id)],
                distances[_pair_key(right, cluster_id)],
            )
            for cluster_id in remaining
        }

        distances = {
            pair: distance
            for pair, distance in distances.items()
            if left not in pair and right not in pair
        }
        for cluster_id, distance in new_distances.items():
            distances[_pair_key(next_cluster_id, cluster_id)] = distance

        active.remove(left)
        active.remove(right)
        active.add(next_cluster_id)
        next_cluster_id += 1

    return 1.0 - np.asarray(merge_distances, dtype=float)


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _cluster_curve_from_merges(
    n_total: int,
    merge_tms: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_counts = np.asarray(
        [
            _cluster_count_at_threshold(n_total, merge_tms, threshold)
            for threshold in thresholds
        ],
        dtype=int,
    )
    fractions = (
        cluster_counts.astype(float) / float(n_total) if n_total else cluster_counts
    )
    return cluster_counts, fractions


def _cluster_count_at_threshold(
    n_total: int,
    merge_tms: np.ndarray,
    threshold: float,
) -> int:
    return int(n_total - np.count_nonzero(merge_tms >= threshold))


def _write_tm_matrix(path: Path, ids: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", *ids])
        for sample_id, row in zip(ids, matrix):
            writer.writerow([sample_id, *[f"{float(value):.6g}" for value in row]])


def _write_threshold_curve(
    path: Path,
    thresholds: np.ndarray,
    cluster_counts: np.ndarray,
    cluster_fractions: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tm_threshold", "n_clusters", "cluster_fraction"])
        for threshold, count, fraction in zip(
            thresholds, cluster_counts, cluster_fractions
        ):
            writer.writerow(
                [f"{float(threshold):.6g}", int(count), f"{float(fraction):.6g}"]
            )


def _trapezoid_auc(values: np.ndarray, thresholds: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, thresholds))
    return float(np.trapz(values, thresholds))
