"""Small-molecule internal diversity.

Internal diversity is the mean pairwise Tanimoto *distance* (1 - similarity)
over all pairs of unique SMILES.

Requires: rdkit
"""

from __future__ import annotations

from zevals.utils.packages import load_rdkit_chem


def compute_diversity(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, float]:
    """Mean pairwise Tanimoto distance among unique generated SMILES.

    Args:
        smiles_list: SMILES strings from the generative model.
        radius:      Morgan fingerprint radius.
        n_bits:      Morgan fingerprint bit-vector length.

    Returns:
        {"diversity_rate": float in [0, 1]}
    """
    Chem = load_rdkit_chem()  # Ensure rdkit is available before proceeding

    try:
        from rdkit import DataStructs  # type: ignore[import]
        from rdkit.Chem import AllChem  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "DataStructs and AllChem require working rdkit. Install rdkit: pip install rdkit"
        ) from exc

    unique_smiles = set(smiles_list)
    if len(unique_smiles) < 2:
        return {"diversity_rate": 0.0}

    fps = []
    for smiles in unique_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None

        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

    if len(fps) < 2:
        return {"diversity_rate": 0.0}

    total, count = 0.0, 0
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
        total += sum(1.0 - s for s in sims)
        count += len(sims)

    diversity_rate = total / count if count else 0.0
    return {"diversity_rate": diversity_rate}
