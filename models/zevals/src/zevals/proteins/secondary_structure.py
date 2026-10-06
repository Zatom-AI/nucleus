"""Secondary structure content via the P-SEA algorithm (Biotite).

Unlike DSSP, P-SEA assigns secondary structure from Cα coordinates alone
(distances + dihedral angles), which makes it usable on backbone-only
generated structures. Computed over the *full* generated set, not just the
designable subset (METRICS.md: "averaged over the full generated set").
"""

from __future__ import annotations

from pathlib import Path


def compute_secondary_structure(backbone_pdbs: list[Path]) -> dict[str, float]:
    """Fraction of residues assigned to alpha-helix / beta-strand / coil.

    Args:
        backbone_pdbs: Paths to generated backbone structure files (`.cif` or
                       `.pdb`) — `biotite`'s `load_structure` dispatches on
                       the file's own suffix.

    Returns:
        {"alpha_frac": float, "beta_frac": float, "coil_frac": float},
        each averaged over every residue across every input structure.
        Structures that fail to parse are skipped.
    """
    import biotite.structure as struc
    import biotite.structure.io as strucio

    n_alpha = n_beta = n_coil = 0
    for pdb_path in backbone_pdbs:
        try:
            atom_array = strucio.load_structure(str(pdb_path))
            sse = struc.annotate_sse(atom_array)
        except Exception:
            continue
        n_alpha += int((sse == "a").sum())
        n_beta += int((sse == "b").sum())
        n_coil += int((sse == "c").sum())

    # Normalize by classified residues (a+b+c), not total residue count —
    # P-SEA leaves chain-break/non-peptide residues unclassified ('').
    n_total = n_alpha + n_beta + n_coil
    if n_total == 0:
        return {
            "alpha_frac": float("nan"),
            "beta_frac": float("nan"),
            "coil_frac": float("nan"),
        }

    return {
        "alpha_frac": n_alpha / n_total,
        "beta_frac": n_beta / n_total,
        "coil_frac": n_coil / n_total,
    }
