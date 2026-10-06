"""Small-molecule novelty against a reference set.

Zatom-1 and previous works define novelty as the generated fraction of unique
(valid) SMILES that do not appear in the reference / training set, normalized by
the total number of (valid) generated samples:

``len((unique_valid_)smiles - reference_smiles) / len((valid_)smiles)``

For large reference sets, FPSim2 novelty uses the mean nearest-neighbor
Tanimoto similarity against a precomputed FPSim2 database:

``1 - mean(top_1_tanimoto_similarity)``
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from zevals.utils.packages import load_fpsim2_engine


def _fp2sim_novelty(smiles: Sequence[str], ref_fp: str, n_threads: int) -> float:
    FPSim2Engine = load_fpsim2_engine()
    eng = FPSim2Engine(ref_fp)

    def _one(smi):
        return float(
            eng.top_k(
                smi,
                threshold=0.0,
                k=1,
                metric="tanimoto",
                full_sanitization=False,
                n_workers=1,
            )[0][1]
        )

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        sims = list(pool.map(_one, smiles))
    return 1.0 - float(np.mean(sims)) if sims else 0.0


def compute_novelty(
    generated: Sequence[str],
    reference: Sequence[str] | None = None,
    *,
    reference_fp: str | Path | None = None,
    n_threads: int = 1,
) -> dict[str, float]:
    """Compute molecule novelty against reference SMILES or a FPSim2 database.

    Args:
        generated: SMILES strings from the generative model.
        reference: SMILES strings from the training / reference set. Used only
            when `reference_fp` is not provided.
        reference_fp: Precomputed FPSim2 reference database path.
        n_threads: Number of Python worker threads for FPSim2 queries.

    Returns:
        For SMILES references: {"num_novel": int, "novel_rate": float in [0, 1]}.
        For FPSim2 references: {"novel_rate": float, "mean_nearest_similarity": float}.
    """
    if not generated:
        return {"num_novel": 0, "novel_rate": 0.0}

    if reference_fp is not None:
        if n_threads < 1:
            raise ValueError("n_threads must be at least 1 for FPSim2 novelty")
        novel_rate = _fp2sim_novelty(generated, str(reference_fp), n_threads)
        return {
            "novel_rate": novel_rate,
            "mean_nearest_similarity": 1.0 - novel_rate,
        }

    if reference is None:
        return {"num_novel": 0, "novel_rate": 0.0}

    reference_smiles = set(reference)
    unique_generated_smiles = set(generated)
    novel_smiles = unique_generated_smiles.difference(reference_smiles)
    novel_rate = len(novel_smiles) / len(generated)
    return {"num_novel": len(novel_smiles), "novel_rate": novel_rate}
