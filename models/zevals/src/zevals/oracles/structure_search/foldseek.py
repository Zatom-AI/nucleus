"""FoldSeek structure search and clustering oracle.

FoldSeek is a standalone binary — no Python env or repo_path needed.
Just point `binary_path` at your FoldSeek installation (or ensure it's on PATH).

Used for:
  - Novelty: easy-search generated structures against PDB / AFDB
  - Structural diversity: easy-cluster generated structures, report n_clusters / n_total

Install:
  conda install -c bioconda foldseek
  # or download from https://github.com/steineggerlab/foldseek/releases

Hydra config: configs/oracle/structure_search/foldseek.yaml
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from zevals.oracles.base import BaseStructureSearcher, SearchResult, run_subprocess


class FoldSeekSearcher(BaseStructureSearcher):
    """Wraps the FoldSeek binary for structure search and all-vs-all clustering.

    Hydra config: oracle/structure_search/foldseek.yaml
      _target_:     zevals.oracles.structure_search.foldseek.FoldSeekSearcher
      binary_path:  null        # null = use 'foldseek' from PATH
      tmp_dir:      /tmp/zevals/foldseek
      sensitivity:  9.5         # foldseek -s parameter
      max_seqs:     1000
    """

    def __init__(
        self,
        sensitivity: float = 9.5,
        max_seqs: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sensitivity = sensitivity
        self.max_seqs = max_seqs

    @property
    def _bin(self) -> str:
        return self.binary_path or "foldseek"

    def search(
        self,
        query_dir: Path,
        target_db: Path,
        output_tsv: Path,
    ) -> list[SearchResult]:
        """Search structure files in `query_dir` against a pre-built FoldSeek database.

        Args:
            query_dir:   Directory of query structure files (`.cif` or `.pdb`;
                         FoldSeek dispatches on each file's own suffix).
            target_db:   Path to a pre-built FoldSeek target DB (e.g. PDB or AFDB).
            output_tsv:  Path where the result TSV will be written.

        Returns:
            One SearchResult per query structure, with TM-score hits.
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._bin,
            "easy-search",
            str(query_dir),
            str(target_db),
            str(output_tsv),
            str(self.tmp_dir / "search_tmp"),
            "-s",
            str(self.sensitivity),
            "--max-seqs",
            str(self.max_seqs),
            "--alignment-type",
            "1",  # TM-score alignment
            "--tmscore-threshold",
            "0.0",
            "--format-output",
            "query,target,alntmscore,evalue",
        ]
        run_subprocess(cmd)
        return _parse_search_tsv(output_tsv)

    def cluster(
        self,
        structure_dir: Path,
        output_tsv: Path,
        min_tm: float = 0.5,
    ) -> list[SearchResult]:
        """All-vs-all cluster structure files in `structure_dir` by TM-score.

        Args:
            structure_dir: Directory of structure files (`.cif` or `.pdb`) to cluster.
            output_tsv:    Path where cluster TSV will be written.
            min_tm:        Minimum TM-score to group two structures in the same cluster.

        Returns:
            One SearchResult per query structure, with cluster_id set.
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._bin,
            "easy-cluster",
            str(structure_dir),
            str(output_tsv.with_suffix("")),
            str(self.tmp_dir / "cluster_tmp"),
            "--alignment-type",
            "1",
            "--tmscore-threshold",
            str(min_tm),
            "-s",
            str(self.sensitivity),
        ]
        run_subprocess(cmd)
        return _parse_cluster_tsv(output_tsv)


def _parse_search_tsv(tsv_path: Path) -> list[SearchResult]:
    results: dict[str, SearchResult] = {}
    if not tsv_path.exists():
        return []
    with tsv_path.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            query, target, tm_score, evalue = row[0], row[1], row[2], row[3]
            if query not in results:
                results[query] = SearchResult(query_id=query, hits=[])
            results[query].hits.append(
                {"target": target, "tm_score": float(tm_score), "evalue": float(evalue)}
            )
    return list(results.values())


def _parse_cluster_tsv(tsv_path: Path) -> list[SearchResult]:
    """FoldSeek easy-cluster writes <cluster_rep>\t<member> lines."""
    results: list[SearchResult] = []
    cluster_tsv = Path(str(tsv_path.with_suffix("")) + "_cluster.tsv")
    if not cluster_tsv.exists():
        return []
    with cluster_tsv.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            rep, member = row[0], row[1]
            results.append(SearchResult(query_id=member, hits=[], cluster_id=rep))
    return results
