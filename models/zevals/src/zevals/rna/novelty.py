"""RNA structural novelty via TMalign (USalign) search against PDB + RNAsolo.

Same shape as protein novelty, but FoldSeek doesn't understand nucleic acids
— RNA uses `oracles.structure_search.usalign.USalignSearcher` instead, which
runs an all-vs-all TMalign between the generated structures and `target_db`.
Unlike FoldSeek, USalign has no compiled-database concept: `target_db` here
is a directory of reference structure files (`.cif` or `.pdb`; PDB + RNAsolo
recommended), not a pre-built binary index.
"""

from __future__ import annotations

from pathlib import Path

from zevals.oracles.base import BaseStructureSearcher


def compute_novelty(
    backbone_pdbs: list[Path],
    structure_searcher: BaseStructureSearcher,
    target_db: Path,
    output_dir: Path,
    tm_threshold: float = 0.45,
) -> dict[str, float]:
    """Fraction of generated RNA structures with no close PDB/RNAsolo homologue.

    Args:
        backbone_pdbs:      Paths to generated RNA structure files (`.cif` or `.pdb`).
        structure_searcher: USalign oracle (injected from config).
        target_db:          Directory of reference structure files (`.cif`
                             or `.pdb`; PDB + RNAsolo recommended) — NOT a
                             compiled database.
        output_dir:         Directory for search result output.
        tm_threshold:       Max TM-score to consider a structure novel (0.45 for RNA).

    Returns:
        {"novelty": mean best TM-score, "novel_rate": threshold fraction}
    """
    if not backbone_pdbs:
        return {"novelty": float("nan"), "novel_rate": 0.0}

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = backbone_pdbs[0].parent
    result_tsv = output_dir / "rna_novelty_search.tsv"

    search_results = structure_searcher.search(pdb_dir, target_db, result_tsv)

    best_query_tm: dict[str, float] = {}
    best_target_tm: dict[str, float] = {}
    for res in search_results:
        best_query_tm[res.query_id] = max(
            (h.get("query_tm_score", h["tm_score"]) for h in res.hits),
            default=0.0,
        )
        best_target_tm[res.query_id] = max(
            (h.get("target_tm_score", h["tm_score"]) for h in res.hits),
            default=0.0,
        )

    expected_queries = {path.stem for path in backbone_pdbs}
    missing_queries = sorted(expected_queries - set(best_query_tm))
    if missing_queries:
        raise RuntimeError(
            "TMalign novelty search returned no result for "
            f"{len(missing_queries)} query structure(s): {missing_queries[:5]}"
        )

    all_best_query = [best_query_tm[path.stem] for path in backbone_pdbs]
    all_best_target = [best_target_tm[path.stem] for path in backbone_pdbs]
    novel_query = [value for value in all_best_query if value < tm_threshold]
    novel_target = [value for value in all_best_target if value < tm_threshold]
    return {
        # Query normalization answers whether the complete generated fold has
        # a close reference. Target normalization is retained as a separate
        # reference-containment diagnostic, since short references otherwise
        # dominate variable-length searches.
        "novelty": sum(all_best_query) / len(all_best_query),
        "novel_rate": len(novel_query) / len(backbone_pdbs),
        "target_normalized_novelty": sum(all_best_target) / len(all_best_target),
        "target_normalized_novel_rate": len(novel_target) / len(backbone_pdbs),
    }
