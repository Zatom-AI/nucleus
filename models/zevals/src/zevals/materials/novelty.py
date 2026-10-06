"""Crystal novelty: composition and structure-level.

A generated crystal is novel if it does not match any structure in the
reference dataset (e.g., Materials Project, ICSD) by composition fingerprint
or by structural prototype similarity.

Requires: pymatgen
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher  # type: ignore[import]
from pymatgen.core import Structure  # type: ignore[import]


def compute_novelty(
    generated_structures: list[Structure],
    reference_structures: list[Structure],
    fingerprint_threshold: float = 0.99,
    use_fingerprint: bool = True,
    reference_fingerprints: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """Fraction of generated crystals not matching any reference structure.

    By default this uses a fingerprint similarity check, which caches
    fingerprints for the reference structures and is much faster than full
    structural matching. Set ``use_fingerprint=False`` to fall back to
    ``StructureMatcher``.

    Args:
        generated_structures:  Generated PyMatGen Structure objects.
        reference_structures:  PyMatGen Structure objects for the reference set.
        fingerprint_threshold: Cosine similarity threshold for a fingerprint
            match when ``use_fingerprint`` is enabled.
        use_fingerprint: If True, compare normalized CrystalNN fingerprints;
            otherwise use pymatgen StructureMatcher.
        reference_fingerprints: Optional precomputed fingerprints for the
            reference structures. If provided, novelty skips recomputing the
            reference-side fingerprints.

    Returns:
        {"novel_rate": float in [0, 1]}
    """
    if use_fingerprint:
        return _fingerprint_novelty(
            generated_structures,
            reference_structures,
            fingerprint_threshold=fingerprint_threshold,
            reference_fingerprints=reference_fingerprints,
        )

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)

    novel_count = 0
    valid_count = 0
    for gen_struct in generated_structures:
        try:
            is_novel = all(
                not matcher.fit(gen_struct, ref) for ref in reference_structures
            )
        except Exception:
            continue
        valid_count += 1
        if is_novel:
            novel_count += 1

    return {"novel_rate": novel_count / valid_count if valid_count else 0.0}


def _fingerprint_novelty(
    generated_structures: list[Structure],
    reference_structures: list[Structure],
    *,
    fingerprint_threshold: float,
    reference_fingerprints: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """Fast novelty computation based on normalized CrystalNN fingerprints."""
    try:
        from matminer.featurizers.site.fingerprint import (  # type: ignore[import]
            CrystalNNFingerprint,
        )
    except ImportError:
        return _matcher_novelty(generated_structures, reference_structures)

    fp = CrystalNNFingerprint.from_preset("ops")
    if reference_fingerprints is None:
        reference_fingerprints = compute_structure_fingerprints(reference_structures)
    else:
        reference_fingerprints = [
            np.asarray(fingerprint, dtype=float)
            for fingerprint in reference_fingerprints
            if fingerprint is not None
        ]

    generated_fingerprints: list[np.ndarray] = []
    for gen_struct in generated_structures:
        gen_fingerprint = _structure_fingerprint(gen_struct, fp)
        if gen_fingerprint is None:
            continue
        generated_fingerprints.append(gen_fingerprint)

    return _novelty_from_fingerprints(
        generated_fingerprints,
        reference_fingerprints,
        fingerprint_threshold=fingerprint_threshold,
    )


def _matcher_novelty(
    generated_structures: list[Structure],
    reference_structures: list[Structure],
) -> dict[str, float]:
    """Fallback novelty computation using StructureMatcher."""
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)

    novel_count = 0
    valid_count = 0
    for gen_struct in generated_structures:
        try:
            is_novel = all(
                not matcher.fit(gen_struct, ref) for ref in reference_structures
            )
        except Exception:
            continue
        valid_count += 1
        if is_novel:
            novel_count += 1

    return {"novel_rate": novel_count / valid_count if valid_count else 0.0}


def compute_structure_fingerprints(
    structures: Sequence[Structure],
) -> list[np.ndarray]:
    """Compute normalized CrystalNN fingerprints for a set of structures.

    This helper is useful for caching reference fingerprints once and reusing
    them across multiple novelty evaluations.
    """
    try:
        from matminer.featurizers.site.fingerprint import (  # type: ignore[import]
            CrystalNNFingerprint,
        )
    except ImportError:
        return []

    fp = CrystalNNFingerprint.from_preset("ops")
    fingerprints: list[np.ndarray] = []
    for structure in structures:
        fingerprint = _structure_fingerprint(structure, fp)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def _novelty_from_fingerprints(
    generated_fingerprints: Sequence[np.ndarray],
    reference_fingerprints: Sequence[np.ndarray],
    *,
    fingerprint_threshold: float,
) -> dict[str, float]:
    """Compute novelty from precomputed normalized fingerprints.

    Uses a single matrix multiplication to score all generated fingerprints
    against all reference fingerprints, then marks a structure novel if every
    reference similarity is below the threshold.
    """
    if not generated_fingerprints:
        return {"novel_rate": 0.0}

    if not reference_fingerprints:
        return {"novel_rate": 1.0}

    x = np.vstack(generated_fingerprints)
    y = np.vstack(reference_fingerprints)
    similarity_matrix = x @ y.T
    novel_mask = np.all(similarity_matrix < fingerprint_threshold, axis=1)
    return {"novel_rate": float(np.mean(novel_mask))}


def _structure_fingerprint(structure: Structure, fp: Any) -> np.ndarray | None:
    """Return a normalized mean site fingerprint for a structure."""
    try:
        site_fps = [fp.featurize(structure, i) for i in range(len(structure))]
    except Exception:
        return None

    if not site_fps:
        return None

    vec = np.asarray(site_fps, dtype=float).mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        return None
    return vec / norm
