"""Crystal structure validity checks.

Checks include:
  - Charge balance (electroneutrality)
  - Reasonable inter-atomic distances (no clashes, no voids)
  - Space-group consistency

Requires: pymatgen
"""

from __future__ import annotations

from collections.abc import Sequence

from zevals.materials.crystal import Crystal


def is_valid_structure(
    sample: Crystal,
) -> bool:
    """Return True if the sample passes the material validity definition.

    Args:
        sample: Crystal object with an already-computed ``valid`` flag.

    Returns:
        True if the sample is physically plausible.
    """
    if not isinstance(sample, Crystal):
        raise TypeError("materials.validity requires Crystal objects.")
    return bool(sample.valid)


def compute_validity(
    samples: Sequence[Crystal],
) -> dict[str, float]:
    """Fraction of generated structures that pass each validity check.

    Returns:
        {"valid_rate": float in [0, 1], "comp_valid_rate": float in [0, 1],
         "struct_valid_rate": float in [0, 1]}
    """
    if not samples:
        return {"valid_rate": 0.0, "comp_valid_rate": 0.0, "struct_valid_rate": 0.0}

    valid = [sample for sample in samples if is_valid_structure(sample)]
    comp_valid = [sample for sample in samples if bool(sample.comp_valid)]
    struct_valid = [sample for sample in samples if bool(sample.struct_valid)]
    total = float(len(samples))
    return {
        "valid_rate": len(valid) / total,
        "comp_valid_rate": len(comp_valid) / total,
        "struct_valid_rate": len(struct_valid) / total,
    }
