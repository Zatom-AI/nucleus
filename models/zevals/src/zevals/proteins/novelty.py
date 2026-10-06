"""Protein structural novelty via FoldSeek pdbTM search.

Novelty (primary) = mean of each generated structure's best (max) TM-score
hit against PDB + AFDB — lower is more novel. This matches METRICS.md's
stated headline metric (Proteína reports this raw average directly).
`novel_rate` is a secondary, threshold-based view: the fraction of samples
whose best hit falls below `tm_threshold` (no close structural homologue).

Uses FoldSeek `easy-search` against a pre-built PDB/AFDB database. Should be
called with only the *designable* subset of backbones (see
`designability.compute_designability`'s second return value) — novelty is
only meaningful for structures that are themselves physically plausible.
"""

from __future__ import annotations

from pathlib import Path

from zevals.oracles.structure_search.foldseek import FoldSeekSearcher


def compute_novelty(
    backbone_pdbs: list[Path],
    foldseek: FoldSeekSearcher,
    target_db: Path,
    output_dir: Path,
    tm_threshold: float = 0.5,
) -> dict[str, float]:
    """Fraction of generated structures with no close PDB/AFDB homologue.

    Args:
        backbone_pdbs:  Paths to generated backbone structure files (`.cif` or `.pdb`).
        foldseek:       FoldSeek oracle (injected from config).
        target_db:      Path to pre-built FoldSeek DB (PDB or AFDB).
        output_dir:     Directory for search result TSV.
        tm_threshold:   Max TM-score to consider a structure novel.

    Returns:
        {"novelty": mean best TM-score (lower = more novel), "novel_rate":
         fraction of samples below `tm_threshold`}
    """
    if not backbone_pdbs:
        return {"novelty": float("nan"), "novel_rate": 0.0}

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = backbone_pdbs[0].parent
    result_tsv = output_dir / "novelty_search.tsv"

    search_results = foldseek.search(pdb_dir, target_db, result_tsv)

    # Map query id → best (max) TM-score hit.
    best_tm: dict[str, float] = {}
    for res in search_results:
        if res.hits:
            best_tm[res.query_id] = max(h["tm_score"] for h in res.hits)
        else:
            best_tm[res.query_id] = 0.0

    all_best = [best_tm.get(p.stem, 0.0) for p in backbone_pdbs]
    novel = [v for v in all_best if v < tm_threshold]

    return {
        "novelty": sum(all_best) / len(all_best) if all_best else float("nan"),
        "novel_rate": len(novel) / len(backbone_pdbs),
    }
