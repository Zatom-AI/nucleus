"""Crystal internal diversity.

Diversity is measured as the mean pairwise structural dissimilarity over all
generated crystals, using pymatgen's StructureComparator or fingerprint-based
distance.

Requires: pymatgen
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher  # type: ignore[import]
from pymatgen.core import Structure  # type: ignore[import]


def compute_diversity(
    generated_structures: list[Structure],
    use_fingerprint: bool = True,
) -> dict[str, float]:
    """Mean pairwise structural diversity among valid generated crystals.

    Args:
        generated_structures:  Generated PyMatGen ``Structure`` objects.
        use_fingerprint: If True, use compositional fingerprints for speed;
                         otherwise use full StructureMatcher (slow for large sets).

    Returns:
        {"diversity_rate": float in [0, 1]}
    """
    structures: list[Any] = list(generated_structures)

    if len(structures) < 2:
        return {"diversity_rate": 0.0}

    if use_fingerprint:
        return _fingerprint_diversity(structures)

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    total_pairs = 0
    dissimilar_pairs = 0

    for left, right in combinations(structures, 2):
        try:
            is_match = matcher.fit(left, right)
        except Exception:
            continue

        total_pairs += 1
        dissimilar_pairs += 0 if is_match else 1

    return {"diversity_rate": dissimilar_pairs / total_pairs if total_pairs else 0.0}


def _fingerprint_diversity(structures: list[Any]) -> dict[str, float]:
    """Fast composition-fingerprint-based diversity."""
    try:
        from matminer.featurizers.site.fingerprint import (  # type: ignore[import]
            CrystalNNFingerprint,
        )
    except ImportError:
        # Fall back to composition-level diversity if matminer unavailable.
        return _composition_diversity(structures)

    fp = CrystalNNFingerprint.from_preset("ops")
    vectors: list[np.ndarray] = []
    for structure in structures:
        try:
            site_fps = [fp.featurize(structure, i) for i in range(len(structure))]
        except Exception:
            continue
        vec = np.asarray(site_fps, dtype=float).mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0 and np.isfinite(norm):
            vectors.append(vec / norm)

    n = len(vectors)
    if n < 2:
        return {"diversity_rate": 0.0}

    x = np.vstack(vectors)
    sim_sum = float(np.sum(x @ x.T))
    mean_similarity = (sim_sum - n) / (n * (n - 1))
    return {"diversity_rate": 1.0 - mean_similarity}


def _composition_diversity(structures: list[Any]) -> dict[str, float]:
    """Coarse diversity based on distinct reduced compositions."""
    formulas = {s.composition.reduced_formula for s in structures}
    return {"diversity_rate": len(formulas) / len(structures)}
