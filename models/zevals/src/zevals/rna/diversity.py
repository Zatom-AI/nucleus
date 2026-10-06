"""RNA diversity: structural (qTMclust, via USalign) and sequence (MMseqs2).

Structural diversity = qTMclust clusters / total samples
Sequence diversity   = MMseqs2 clusters / total samples

qTMclust is part of the USalign toolkit and is the standard tool for RNA
structural clustering (following RNA-FrameFlow / RFDPoly) — FoldSeek's own
clustering doesn't understand nucleic acids, so RNA uses a dedicated oracle
(`oracles.structure_search.usalign.USalignSearcher`) rather than reusing
`proteins.diversity`'s FoldSeek-based one.

Sequence diversity reuses `proteins.diversity`'s MMseqs2 plumbing (that part
genuinely is molecule-agnostic), but with two RNA-specific fixes so it's a
real nucleotide search, not an accident: designed sequences arrive in
NA-MPNN's own internal one-letter alphabet (lowercase, DNA/RNA-disambiguating
— see `_na_mpnn/featurize.py`), which (a) MMseqs2's auto-detect heuristic
mis-scores as "amino acid" (verified — left at `--dbtype 0` it silently ran
protein-mode search over RNA designs) and (b) MMseqs2 can't safely nucleotide-
search directly anyway, since it isn't the real biological alphabet. Both are
fixed below: translate to standard IUPAC nucleotide codes and pass
`--dbtype 2` explicitly.
"""

from __future__ import annotations

from pathlib import Path

from zevals.oracles.base import BaseStructureSearcher


def compute_structural_diversity(
    backbone_pdbs: list[Path],
    structure_searcher: BaseStructureSearcher,
    output_dir: Path,
    min_tm: float = 0.45,
) -> dict[str, float]:
    """qTMclust all-vs-all cluster diversity for RNA structures.

    Args:
        backbone_pdbs:      Paths to generated RNA structure files (`.cif` or `.pdb`).
        structure_searcher: USalign oracle (injected from config).
        output_dir:         Scratch directory for qTMclust output.
        min_tm:              TM-score threshold for clustering (0.45 for RNA,
                              qTMclust's own valid range is [0.45, 1)).

    Returns:
        {"structural_diversity": float in [0, 1], "n_clusters": int}
    """
    if not backbone_pdbs:
        return {"structural_diversity": 0.0, "n_clusters": 0}

    # qTMclust clusters every structure file in the directory, which may include
    # samples outside `backbone_pdbs` sharing the same parent dir — filter
    # cluster membership down to `backbone_pdbs` before counting, so the
    # numerator and denominator both describe the same population (mirrors
    # proteins.diversity.compute_structural_diversity's designable-subset
    # filtering).
    pdb_dir = backbone_pdbs[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_out = output_dir / "qtmclust_clusters.txt"

    wanted_stems = {p.stem for p in backbone_pdbs}
    results = structure_searcher.cluster(pdb_dir, cluster_out, min_tm=min_tm)
    returned_stems = {result.query_id for result in results}
    missing_stems = sorted(wanted_stems - returned_stems)
    if missing_stems:
        raise RuntimeError(
            "qTMclust returned no cluster assignment for "
            f"{len(missing_stems)} RNA structure(s): {missing_stems[:5]}"
        )
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
) -> dict[str, float]:
    """MMseqs2 cluster-based RNA sequence diversity, in real nucleotide mode.

    Args:
        sequences:   Designed sequences straight out of NA-MPNN's
                     `decode_sequence` — its own one-letter alphabet,
                     `/`-joined per chain for multi-chain designs. Translated
                     to standard IUPAC nucleotide codes and split on `/`
                     before clustering (a literal `/` isn't a valid
                     nucleotide character — MMseqs2 would otherwise choke on
                     or silently mis-score multi-chain designs).
        output_dir:  Scratch directory for MMseqs2 intermediate files.
        mmseqs_bin:  Path to MMseqs2 binary (default: 'mmseqs' on PATH).
        min_seq_id:  Sequence identity threshold for clustering.

    Returns:
        {"sequence_diversity": float in [0, 1], "n_seq_clusters": int}
    """
    from zevals.oracles.inverse_folders._na_mpnn.featurize import (
        to_standard_nucleotide_alphabet,
    )
    from zevals.proteins.diversity import (
        compute_sequence_diversity as _protein_seq_div,
    )

    chains = [
        to_standard_nucleotide_alphabet(chain)
        for seq in sequences
        for chain in seq.split("/")
    ]

    result = _protein_seq_div(
        chains, output_dir, mmseqs_bin=mmseqs_bin, min_seq_id=min_seq_id, dbtype=2
    )
    return {
        "sequence_diversity": result["sequence_diversity"],
        "n_seq_clusters": result["n_seq_clusters"],
    }
