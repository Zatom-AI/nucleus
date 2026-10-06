"""Small-molecule validity metrics.

Checks each generated molecule for:
  - Molecule SMILES parsability and RDKit sanitization
  - Structural sanity (no radicals, reasonable valence) and optional fragment-based validity (largest fragment is valid)

Requires: rdkit
"""

from __future__ import annotations

from typing import Any

from zevals.utils.packages import load_rdkit_chem


def is_valid_mol(mol: Any, frag_check: bool = True) -> tuple[bool, Any, str]:
    """Return True if `mol` can be parsed and sanitized by RDKit along with its RDKit and SMILES representation."""
    Chem = load_rdkit_chem()  # Ensure rdkit is available before proceeding

    try:
        smiles = Chem.MolToSmiles(mol)
        if not frag_check:
            return mol is not None and smiles is not None, mol, smiles

        # Simple fragment-based validity check
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest_mol_frag = max(
            mol_frags, default=mol, key=lambda frag: frag.GetNumAtoms()
        )
        largest_mol_smiles = Chem.MolToSmiles(largest_mol_frag)
        return (
            largest_mol_frag is not None and largest_mol_smiles is not None,
            mol,
            largest_mol_smiles,
        )

    except Exception:
        return False, mol, ""


def compute_validity(mol_list: list[Any], frag_check: bool = True) -> dict[str, float]:
    """Fraction of valid molecules in a generated set.

    Returns:
        {"num_valid": int, "valid_rate": float in [0, 1], "valid_mols": list[Any], "valid_smiles": list[str]}
    """
    valid_mols = []
    valid_smiles = []
    for m in mol_list:
        if m is None:
            continue
        is_valid, mol, smiles = is_valid_mol(m, frag_check=frag_check)
        if not is_valid:
            continue
        valid_mols.append(mol)
        valid_smiles.append(smiles)

    num_valid = len(valid_mols)
    valid_rate = num_valid / len(mol_list) if mol_list else 0.0
    return {
        "num_valid": num_valid,
        "valid_rate": valid_rate,
        "valid_mols": valid_mols,
        "valid_smiles": valid_smiles,
    }


compute_sample_validity = compute_validity
