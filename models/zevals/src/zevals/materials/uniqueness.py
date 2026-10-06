"""Crystal sample uniqueness metrics."""

from __future__ import annotations

from pymatgen.analysis.structure_matcher import StructureMatcher  # type: ignore[import]
from pymatgen.core import Structure  # type: ignore[import]


def compute_uniqueness(
    generated_structures: list[Structure],
    *,
    return_unique_structures: bool = False,
) -> dict[str, float | list[Structure]]:
    """Fraction of valid generated structures that are unique.

    This mirrors the Zatom-1 material definition:

    ``len(group_structures(valid_structures)) / len(valid_structures)``

    Args:
        generated_structures: Valid generated structures to deduplicate.
        return_unique_structures: If True, include the representative unique
            structures under ``unique_structures`` so callers can reuse the
            grouped subset without recomputing matches.

    Returns:
        {"num_unique": int, "unique_rate": float in [0, 1]}
        and optionally ``"unique_structures": list[Structure]``.
    """
    if not generated_structures:
        result: dict[str, float | list[Structure]] = {
            "num_unique": 0,
            "unique_rate": 0.0,
        }
        if return_unique_structures:
            result["unique_structures"] = []
        return result  # type: ignore[return-value]

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    unique_groups = matcher.group_structures(generated_structures)
    unique_structures = [group[0] for group in unique_groups if group]
    num_unique = len(unique_groups)
    result: dict[str, float | list[Structure]] = {
        "num_unique": num_unique,
        "unique_rate": num_unique / len(generated_structures),
    }
    if return_unique_structures:
        result["unique_structures"] = unique_structures
    return result  # type: ignore[return-value]


compute_sample_uniqueness = compute_uniqueness
