"""USalign (TMalign + qTMclust) structure search and clustering oracle, for RNA.

RNA structural comparison uses TM-score (sequence-independent, alignment-
based) rather than FoldSeek's protein-structure-alphabet approach, which
doesn't understand nucleic acids. USalign (https://github.com/pylelab/USalign,
Zhang/Pyle lab) is the RNA-FrameFlow/RFDPoly-convention tool for this —
TMalign for pairwise/all-vs-all search, qTMclust for clustering. Neither
ships pre-built binaries; `zevals setup --tools usalign` compiles both from
source (see `scripts/install_usalign.sh`; needs `g++` on PATH).

Every call hardcodes `-mol RNA -atom " C1'"`. TMalign/qTMclust's own default
reference atom for RNA/DNA is C3', not C1' — this oracle intentionally
overrides that default so it's consistent with `oracles.inverse_folders.
na_mpnn`, which already uses C1' as nucleic acid's structural reference atom
(see `_na_mpnn/featurize.py`'s `na_ref_atom`). Using the same atom in both
the inverse-folding model and the structure-comparison tool keeps one
consistent geometric picture of "the RNA backbone" across zevals.

Unlike FoldSeek, USalign has no compiled-database concept — `target_db` in
`search()` means a directory of reference PDB files, aligned against
directly, not a pre-built binary index.

Hydra config: configs/oracle/structure_search/usalign.yaml
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from zevals.oracles.base import (
    BaseStructureSearcher,
    SearchResult,
    resolve_binary,
    run_subprocess,
)
from zevals.utils.io import glob_structures

_ATOM = " C1'"
_MOL = "RNA"


class USalignSearcher(BaseStructureSearcher):
    """Wraps TMalign (search) + qTMclust (cluster) for RNA structures.

    Hydra config: oracle/structure_search/usalign.yaml
      _target_:    zevals.oracles.structure_search.usalign.USalignSearcher
      binary_path: null   # null = 'TMalign'/'qTMclust' from PATH; otherwise
                           # the directory holding both binaries (matches
                           # `zevals setup --tools usalign`'s --install-dir,
                           # which installs both into the same directory).
      tmp_dir:     /tmp/zevals/usalign
    """

    def __init__(self, *args, search_workers: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.search_workers = max(1, int(search_workers))

    @property
    def _tmalign_bin(self) -> str:
        return _binary_in_dir(self.binary_path, "TMalign")

    @property
    def _qtmclust_bin(self) -> str:
        return _binary_in_dir(self.binary_path, "qTMclust")

    def search(
        self,
        query_dir: Path,
        target_db: Path,
        output_tsv: Path,
    ) -> list[SearchResult]:
        """All-vs-all TMalign between every structure in `query_dir` and `target_db`.

        Args:
            query_dir:  Directory of query structure files (`.cif` or `.pdb`).
            target_db:  Directory of reference/target structure files
                        (`.cif` or `.pdb`) — NOT a compiled database
                        (USalign has no such concept).
            output_tsv: Path the raw TMalign tabular output is saved to.

        Returns:
            One SearchResult per query structure, with one hit per target.
            `tm_score` is TM-score normalized by the *target*'s length (the
            convention for "does a close reference structure exist").
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)

        query_pdbs = glob_structures(Path(query_dir))
        target_pdbs = glob_structures(Path(target_db))
        if not query_pdbs or not target_pdbs:
            return []

        target_list = self.tmp_dir / "target_list.txt"
        target_list.write_text("\n".join(p.name for p in target_pdbs) + "\n")
        worker_count = min(self.search_workers, len(query_pdbs))
        shards = [query_pdbs[index::worker_count] for index in range(worker_count)]

        def run_shard(index: int, paths: list[Path]) -> str:
            query_list = self.tmp_dir / f"query_list_{index:03d}.txt"
            query_list.write_text("\n".join(path.name for path in paths) + "\n")
            cmd = [
                self._tmalign_bin,
                "-dir1",
                f"{query_dir}/",
                str(query_list),
                "-dir2",
                f"{target_db}/",
                str(target_list),
                "-mol",
                _MOL,
                "-atom",
                _ATOM,
                "-outfmt",
                "2",
            ]
            result = run_subprocess(cmd, capture_output=True, text=True)
            return result.stdout

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            outputs = list(
                executor.map(
                    lambda indexed: run_shard(*indexed),
                    enumerate(shards),
                )
            )
        stdout = "\n".join(outputs)
        output_tsv.write_text(stdout)
        return _parse_tmalign_output(stdout)

    def cluster(
        self,
        structure_dir: Path,
        output_tsv: Path,
        min_tm: float = 0.45,
    ) -> list[SearchResult]:
        """All-vs-all qTMclust of every structure in `structure_dir`.

        Args:
            structure_dir: Directory of structure files (`.cif` or `.pdb`) to cluster.
            output_tsv:    Path the cluster assignment is written to
                            (qTMclust's own format: one line per cluster,
                            tab-separated `repr<TAB>member<TAB>member...`).
            min_tm:        TM-score cutoff for two structures to share a
                            cluster — must be in qTMclust's valid range,
                            [0.45, 1) (RNA-FrameFlow's 0.45 default clears it).

        Returns:
            One SearchResult per structure, `cluster_id` set to its cluster
            representative's filename stem.
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)

        pdbs = glob_structures(Path(structure_dir))
        if not pdbs:
            return []

        chain_list = self.tmp_dir / "chain_list.txt"
        chain_list.write_text("\n".join(p.name for p in pdbs) + "\n")

        cmd = [
            self._qtmclust_bin,
            "-dir",
            f"{structure_dir}/",
            str(chain_list),
            "-TMcut",
            str(min_tm),
            "-mol",
            _MOL,
            "-atom",
            _ATOM,
            "-o",
            str(output_tsv),
        ]
        run_subprocess(cmd)
        return _parse_qtmclust_output(output_tsv)


def _binary_in_dir(binary_path: str | None, name: str) -> str:
    """Resolve `name` (TMalign or qTMclust) inside the `binary_path`
    directory — unlike FoldSeek's single-binary `binary_path`, USalign needs
    two, so this field is always a directory, never a path to one binary
    (avoids an existence-check-dependent, ambiguous "is this a file or a
    not-yet-created directory" guess)."""
    if binary_path is None:
        return resolve_binary(name)
    return str(Path(binary_path) / name)


def _parse_tmalign_output(stdout: str) -> list[SearchResult]:
    """Parse `TMalign -outfmt 2` output: a header line (`#PDBchain1
    PDBchain2 TM1 TM2 RMSD ...`) followed by one tab-separated row per
    (query, target) pair."""
    results: dict[str, SearchResult] = {}
    for line in stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        query_id = Path(parts[0]).stem
        target_id = Path(parts[1]).stem
        tm1 = float(parts[2])  # normalized by the query's length
        tm2 = float(parts[3])  # normalized by the target's length
        if query_id not in results:
            results[query_id] = SearchResult(query_id=query_id, hits=[])
        results[query_id].hits.append(
            {
                "target": target_id,
                # Keep the established field for callers that explicitly use
                # reference-containment normalization.
                "tm_score": tm2,
                "query_tm_score": tm1,
                "target_tm_score": tm2,
            }
        )
    return list(results.values())


def _parse_qtmclust_output(output_path: Path) -> list[SearchResult]:
    """Parse qTMclust's `-o` output: one line per cluster,
    `repr<TAB>member<TAB>member...` (the representative is itself a member)."""
    results: list[SearchResult] = []
    if not output_path.exists():
        return results
    for line in output_path.read_text().splitlines():
        fields = [f for f in line.strip().split("\t") if f]
        if not fields:
            continue
        repr_id = Path(fields[0]).stem
        for member in fields:
            results.append(
                SearchResult(query_id=Path(member).stem, hits=[], cluster_id=repr_id)
            )
    return results
