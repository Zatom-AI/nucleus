"""RiboSeek structural-alphabet prefilter with exact USalign refinement."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from zevals.oracles.base import BaseStructureSearcher, SearchResult
from zevals.utils.io import glob_structures


class RiboSeekPrefilteredSearcher(BaseStructureSearcher):
    """Shortlist RNA references with RiboSeek, then report USalign TM-scores.

    RiboSeek's combined structural-alphabet score is a ranking signal, not a
    TM-score. This wrapper therefore never exposes it as novelty: the retained
    candidates are refined by ``refinement_searcher`` and only those TM-scores
    enter the public search result. Structural clustering is delegated without
    prefiltering so the existing qTMclust definition remains unchanged.
    """

    def __init__(
        self,
        refinement_searcher: BaseStructureSearcher,
        database_path: str,
        top_n: int = 256,
        prefilter_candidates: int = 1000,
        tmp_dir: str = "/tmp/zevals/riboseek",
        **kwargs: Any,
    ) -> None:
        super().__init__(tmp_dir=tmp_dir, **kwargs)
        if top_n <= 0 or prefilter_candidates < top_n:
            raise ValueError("RiboSeek requires 0 < top_n <= prefilter_candidates")
        self.refinement_searcher = refinement_searcher
        self.database_path = Path(database_path).expanduser()
        self.top_n = top_n
        self.prefilter_candidates = prefilter_candidates
        self._searcher: Any | None = None

    def search(
        self,
        query_dir: Path,
        target_db: Path,
        output_tsv: Path,
    ) -> list[SearchResult]:
        """Prefilter each query against an encoded DB and refine its shortlist."""
        queries = glob_structures(Path(query_dir))
        targets = glob_structures(Path(target_db))
        if not queries or not targets:
            return []
        target_by_key = _target_structure_index(targets)
        searcher = self._get_searcher()
        results: list[SearchResult] = []
        output_lines = ["query\ttarget\ttm_score\triboseek_combined_score"]
        for query_index, query in enumerate(queries):
            riboseek_hits = searcher.search(
                query,
                top_n=self.top_n,
                prefilter=True,
                prefilter_candidates=self.prefilter_candidates,
            )
            resolved: list[tuple[Path, float]] = []
            for hit in riboseek_hits:
                target = _resolve_target(str(hit["chain"]), target_by_key)
                if target is not None and all(target != item[0] for item in resolved):
                    resolved.append((target, float(hit["combined_score"])))
            if not resolved:
                raise RuntimeError(
                    "RiboSeek hits could not be mapped to reference structures. "
                    "Build the encoded database from the same target_db directory."
                )

            work_dir = self.tmp_dir / f"query_{query_index:06d}_{query.stem}"
            if work_dir.exists():
                shutil.rmtree(work_dir)
            query_stage = work_dir / "query"
            target_stage = work_dir / "targets"
            query_stage.mkdir(parents=True)
            target_stage.mkdir()
            shutil.copy2(query, query_stage / query.name)
            score_by_stem: dict[str, float] = {}
            for target, score in resolved:
                shutil.copy2(target, target_stage / target.name)
                score_by_stem[target.stem] = score
            refined = self.refinement_searcher.search(
                query_stage,
                target_stage,
                work_dir / "usalign.tsv",
            )
            if len(refined) != 1:
                raise RuntimeError(
                    f"USalign refinement returned {len(refined)} results for {query}"
                )
            result = refined[0]
            for hit in result.hits:
                score = score_by_stem.get(str(hit["target"]), float("nan"))
                hit["riboseek_combined_score"] = score
                output_lines.append(
                    f"{result.query_id}\t{hit['target']}\t{hit['tm_score']}\t{score}"
                )
            results.append(result)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        output_tsv.write_text("\n".join(output_lines) + "\n")
        return results

    def cluster(
        self,
        structure_dir: Path,
        output_tsv: Path,
        min_tm: float = 0.5,
    ) -> list[SearchResult]:
        """Delegate exact all-vs-all clustering to the refinement searcher."""
        return self.refinement_searcher.cluster(structure_dir, output_tsv, min_tm)

    def _get_searcher(self) -> Any:
        if self._searcher is None:
            if not self.database_path.is_file():
                raise FileNotFoundError(
                    f"RiboSeek encoded database is missing: {self.database_path}. "
                    "Build it with `riboseek build-db <target_db> -o <db.json>`."
                )
            try:
                from riboseek import Searcher  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "RiboSeek is optional; install `zevals[rna]` or "
                    "`pip install riboseek==0.1.0`."
                ) from exc
            self._searcher = Searcher.from_pretrained(db=self.database_path)
        return self._searcher


def _target_structure_index(targets: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for target in targets:
        if target.stem in index:
            raise ValueError(f"Duplicate target structure stem: {target.stem!r}")
        index[target.stem] = target
    return index


def _resolve_target(chain_key: str, target_by_key: dict[str, Path]) -> Path | None:
    matches = [
        (stem, path)
        for stem, path in target_by_key.items()
        if chain_key == stem or chain_key.startswith(stem + "_")
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]
