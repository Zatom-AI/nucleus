"""Small-molecule sample uniqueness metrics."""

from __future__ import annotations


def compute_uniqueness(smiles_list: list[str]) -> dict[str, float]:
    """Fraction of samples that are unique SMILES.

    This mirrors the Zatom-1 definition, which computes uniqueness as:

    ``len(unique(_valid)_smiles) / len((valid_)smiles_list)``

    Returns:
        {"num_unique": int, "unique_rate": float in [0, 1]}
    """
    if not smiles_list:
        return {"num_unique": 0, "unique_rate": 0.0}

    unique_smiles = set(smiles_list)
    num_unique = len(unique_smiles)
    unique_rate = num_unique / len(smiles_list)
    return {"num_unique": num_unique, "unique_rate": unique_rate}


compute_sample_uniqueness = compute_uniqueness
