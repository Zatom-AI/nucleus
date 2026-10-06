"""PDB parsing and tensor featurization for NA-MPNN, vendored and trimmed
from `inference/data_utils.py` + the constants block in `inference/run.py`
in https://github.com/baker-laboratory/NA-MPNN (MIT License, Copyright (c)
2025 Andrew Kubaney).

Trimmed relative to upstream: only backbone atoms are parsed (no
`parse_all_atoms` side-chain mode), every protein/DNA/RNA residue present is
designable (no fixed positions, no `design_na_only` chain restriction, no
symmetry ties), and residues with any missing backbone atom are masked out
rather than optionally loaded (`load_residues_with_missing_atoms=0`,
upstream's own default) — matching the trims already applied to
`_protein_mpnn`/`_ligand_mpnn`.

`na_shared_tokens=1` (upstream's default, and what both bundled checkpoints
were trained with) is hardcoded rather than exposed as an option: RNA's
one-letter tokens (A/C/G/U/RX) are aliased onto DNA's softmax slots
(DA/DC/DG/DT/DX) so the model's 33-way output never has to distinguish a
ribose from a deoxyribose backbone at the token-identity level — only the
input structure's O2' atom (or lack of it) does that. `decode_sequence`
below reverses the alias post-hoc: it always decodes into DNA one-letter
codes first, then swaps to the RNA code for positions whose *original*
residue actually had an O2' atom. Concretely: designing a sequence for a
pure-RNA backbone still routes through the DA/DC/DG/DT/DX softmax slots
internally, and comes back out as A/C/G/U anyway, because the sampling loop
would otherwise be able to select `A`'s/`RX`'s own (untrained, always-biased
-1e8) slots — see `_DEFAULT_BIAS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Atom / polymer-type vocabularies
# ---------------------------------------------------------------------------

_PROTEIN_BACKBONE_ATOMS = ["N", "CA", "C", "O"]
_DNA_BACKBONE_ATOMS = [
    "OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
]  # fmt: skip
_NA_BACKBONE_ATOMS = [
    "OP1",
    "OP2",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]  # fmt: skip  (RNA has every DNA backbone atom plus O2')

ATOM_TYPES = _PROTEIN_BACKBONE_ATOMS + _NA_BACKBONE_ATOMS  # 16 atoms
ATOM_ORDER = {name: i for i, name in enumerate(ATOM_TYPES)}

POLYTYPES = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
POLYTYPE_TO_INT = {name: i for i, name in enumerate(POLYTYPES)}

RESTYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    "DA", "DC", "DG", "DT", "DX",
    "A", "C", "G", "U", "RX",
    "MAS", "PAD",
]  # fmt: skip

RESTYPE_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "UNK": "X",
    "DA": "a", "DC": "c", "DG": "g", "DT": "t", "DX": "x",
    "A": "b", "C": "d", "G": "h", "U": "u", "RX": "y",
    "MAS": "-", "PAD": "+",
}  # fmt: skip

RESTYPE_TO_INT = {name: i for i, name in enumerate(RESTYPES)}
for _rna_name, _dna_name in [
    ("A", "DA"),
    ("C", "DC"),
    ("G", "DG"),
    ("U", "DT"),
    ("RX", "DX"),
]:
    RESTYPE_TO_INT[_rna_name] = RESTYPE_TO_INT[_dna_name]  # na_shared_tokens=1

RESTYPE_INT_TO_ONE: dict[int, str] = {}
for _name in RESTYPES:
    RESTYPE_INT_TO_ONE.setdefault(RESTYPE_TO_INT[_name], RESTYPE_3_TO_1[_name])

DNA_CHAR_TO_RNA_CHAR = {
    RESTYPE_3_TO_1["DA"]: RESTYPE_3_TO_1["A"],
    RESTYPE_3_TO_1["DC"]: RESTYPE_3_TO_1["C"],
    RESTYPE_3_TO_1["DG"]: RESTYPE_3_TO_1["G"],
    RESTYPE_3_TO_1["DT"]: RESTYPE_3_TO_1["U"],
    RESTYPE_3_TO_1["DX"]: RESTYPE_3_TO_1["RX"],
}

# NA-MPNN's own one-letter codes -> standard IUPAC nucleotide codes. Generic
# downstream tools (e.g. MMseqs2's nucleotide search mode) expect the real
# biological alphabet — uppercase A/C/G/T for DNA, A/C/G/U for RNA — not
# NA-MPNN's internal DNA/RNA-disambiguating one (lowercase a/c/g/t/x for
# DNA, b/d/h/u/y for RNA). Protein one-letter codes are already standard
# uppercase amino acid letters and aren't in this table (left unchanged by
# `to_standard_nucleotide_alphabet` below).
STANDARD_NUCLEOTIDE_CHAR = {
    RESTYPE_3_TO_1["DA"]: "A",
    RESTYPE_3_TO_1["DC"]: "C",
    RESTYPE_3_TO_1["DG"]: "G",
    RESTYPE_3_TO_1["DT"]: "T",
    RESTYPE_3_TO_1["DX"]: "N",
    RESTYPE_3_TO_1["A"]: "A",
    RESTYPE_3_TO_1["C"]: "C",
    RESTYPE_3_TO_1["G"]: "G",
    RESTYPE_3_TO_1["U"]: "U",
    RESTYPE_3_TO_1["RX"]: "N",
}

# Positions the sampling loop must never select: UNK/DX/RX/MAS/PAD (blocked
# explicitly in `NAMPNN.sample`) plus A/C/G/U/RX's own now-unaliased softmax
# slots, which `na_shared_tokens` leaves untrained and unreachable via
# `RESTYPE_TO_INT` but which the raw 33-way softmax can still assign
# probability mass to unless biased away here.
_OMIT_ONE_LETTER = {"X", "b", "d", "h", "u", "y"}
_ALPHABET = [RESTYPE_3_TO_1[name] for name in RESTYPES]
_DEFAULT_BIAS = torch.tensor(
    [-1e8 if ch in _OMIT_ONE_LETTER else 0.0 for ch in _ALPHABET], dtype=torch.float32
)


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------


def _aligned_coords(
    macromolecule_atoms, reference_dict: dict, atom_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Align one atom type's coordinates onto the residue index used by `reference_dict`."""
    atom_sel = macromolecule_atoms.select(f"name {atom_name}")
    coords = np.zeros([len(reference_dict), 3], np.float32)
    mask = np.zeros([len(reference_dict)], np.int32)
    if atom_sel is not None:
        atom_coords = atom_sel.getCoords()
        atom_resnums = atom_sel.getResnums()
        atom_chain_ids = atom_sel.getChids()
        atom_icodes = atom_sel.getIcodes()
        for i in range(len(atom_resnums)):
            code = f"{atom_chain_ids[i]}_{atom_resnums[i]}_{atom_icodes[i]}"
            if code in reference_dict:
                coords[reference_dict[code], :] = atom_coords[i]
                mask[reference_dict[code]] = 1
    return coords, mask


@dataclass
class ParsedNABackbone:
    """A single (possibly multi-chain, multi-polymer-type) parsed backbone."""

    name: str
    X: np.ndarray  # [L, 16, 3] backbone coords (protein N/CA/C/O + NA atoms)
    X_m: np.ndarray  # [L, 16] per-atom mask
    R_idx: np.ndarray  # [L] residue index (PDB numbering)
    chain_labels: np.ndarray  # [L] integer chain id per residue
    chain_letters: list[str]  # [L] chain letter per residue
    protein_mask: np.ndarray  # [L]
    dna_mask: np.ndarray  # [L]
    rna_mask: np.ndarray  # [L]
    rna_mask_for_token_conversion: np.ndarray  # [L] 1 where residue has O2'
    polymer_type: np.ndarray  # [L] int, indexes into POLYTYPE_TO_INT
    S: np.ndarray  # [L] native sequence, RESTYPE_TO_INT-encoded


def parse_pdb_na(path: Path) -> ParsedNABackbone:
    """Parse a PDB or mmCIF file's protein and/or nucleic-acid backbone atoms.

    Args:
        path: Path to a single-model PDB or CIF file (dispatched by
            `.suffix`). Every protein and nucleic-acid residue with a
            complete backbone is included; everything else (ligands,
            waters, incomplete residues) is dropped.

    Returns:
        ParsedNABackbone with one entry per complete-backbone residue,
        in the order prody's `protein or nucleic` selection returns them.
    """
    from prody import confProDy, parseMMCIF, parsePDB

    confProDy(verbosity="none")

    parse = parseMMCIF if path.suffix == ".cif" else parsePDB
    atoms = parse(str(path))
    atoms = atoms.select("occupancy > 0")
    macromolecule_atoms = atoms.select("protein or nucleic")
    if macromolecule_atoms is None:
        raise ValueError(f"No protein or nucleic-acid atoms found in {path}")

    reference_atoms = macromolecule_atoms.select(
        "(protein and name CA) or (nucleic and name C1')"
    )
    ref_resnums = reference_atoms.getResnums()
    ref_chain_ids = reference_atoms.getChids()
    ref_icodes = reference_atoms.getIcodes()

    reference_dict: dict[str, int] = {}
    for i in range(len(ref_resnums)):
        code = f"{ref_chain_ids[i]}_{ref_resnums[i]}_{ref_icodes[i]}"
        reference_dict[code] = i
    L = len(reference_dict)

    X = np.zeros([L, len(ATOM_TYPES), 3], np.float32)
    X_m = np.zeros([L, len(ATOM_TYPES)], np.int32)
    for atom_name in ATOM_TYPES:
        coords, mask = _aligned_coords(macromolecule_atoms, reference_dict, atom_name)
        X[:, ATOM_ORDER[atom_name], :] = coords
        X_m[:, ATOM_ORDER[atom_name]] = mask

    protein_idx = [ATOM_ORDER[a] for a in _PROTEIN_BACKBONE_ATOMS]
    dna_idx = [ATOM_ORDER[a] for a in _DNA_BACKBONE_ATOMS]
    na_idx = [ATOM_ORDER[a] for a in _NA_BACKBONE_ATOMS]

    protein_mask = X_m[:, protein_idx].prod(axis=-1)
    rna_mask = X_m[:, na_idx].prod(axis=-1)
    # DNA shares every backbone atom with RNA except O2' — subtract rna_mask
    # so a residue with a complete RNA backbone isn't double-counted as DNA.
    dna_mask = X_m[:, dna_idx].prod(axis=-1) - rna_mask

    rna_mask_for_token_conversion = X_m[:, ATOM_ORDER["O2'"]]

    polymer_type = (
        protein_mask * POLYTYPE_TO_INT["PP"]
        + dna_mask * POLYTYPE_TO_INT["DNA"]
        + rna_mask * POLYTYPE_TO_INT["RNA"]
        + (1 - protein_mask - dna_mask - rna_mask) * POLYTYPE_TO_INT["UNK"]
    ).astype(np.int64)

    resnames = reference_atoms.getResnames()
    S = np.empty(L, dtype=np.int64)
    for i, resname in enumerate(resnames):
        if resname in RESTYPE_TO_INT:
            S[i] = RESTYPE_TO_INT[resname]
        elif dna_mask[i]:
            S[i] = RESTYPE_TO_INT["DX"]
        elif rna_mask[i]:
            S[i] = RESTYPE_TO_INT["RX"]
        else:
            S[i] = RESTYPE_TO_INT["UNK"]

    return ParsedNABackbone(
        name=Path(path).stem,
        X=X,
        X_m=X_m,
        R_idx=np.array(ref_resnums, dtype=np.int64),
        chain_labels=np.array(reference_atoms.getChindices(), dtype=np.int64),
        chain_letters=list(np.array(ref_chain_ids)),
        protein_mask=protein_mask,
        dna_mask=dna_mask,
        rna_mask=rna_mask,
        rna_mask_for_token_conversion=rna_mask_for_token_conversion,
        polymer_type=polymer_type,
        S=S,
    )


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


@dataclass
class FeaturizedNABackbone:
    """Batch-ready (batch size 1) tensors for `NAMPNN.sample`."""

    feature_dict: dict[str, torch.Tensor]
    chain_letters: list[str]  # per-residue chain letter, same order as S
    rna_mask_for_token_conversion: np.ndarray  # per-residue, same order as S


def featurize_na_backbone(
    backbone: ParsedNABackbone, device: torch.device
) -> FeaturizedNABackbone:
    """Convert a `ParsedNABackbone` into `NAMPNN.sample`-ready tensors.

    Every present residue is marked designable (`chain_mask` = the residue
    mask — no fixed positions), matching `ProteinMPNNFolder`/
    `LigandMPNNFolder`'s "every chain in the PDB is designable" convention.
    """
    L = backbone.X.shape[0]
    mask = (backbone.protein_mask + backbone.dna_mask + backbone.rna_mask).astype(
        np.float32
    )

    # Renumber repeated (R_idx, insertion-code) collisions so R_idx stays
    # strictly non-decreasing — mirrors upstream `featurize`'s handling of
    # insertion codes, which the raw PDB residue number alone can't encode.
    R_idx_renumbered = np.empty(L, dtype=np.int64)
    count, prev = 0, None
    for i, r in enumerate(backbone.R_idx):
        if prev is not None and r == prev:
            count += 1
        R_idx_renumbered[i] = r + count
        prev = r

    def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device).unsqueeze(0)

    feature_dict = {
        "X": to_tensor(backbone.X, torch.float32),
        "X_m": to_tensor(backbone.X_m, torch.int32),
        "mask": to_tensor(mask, torch.float32),
        "R_idx": to_tensor(R_idx_renumbered, torch.long),
        "chain_labels": to_tensor(backbone.chain_labels, torch.long),
        "protein_mask": to_tensor(
            backbone.protein_mask.astype(np.float32), torch.float32
        ),
        "dna_mask": to_tensor(backbone.dna_mask.astype(np.float32), torch.float32),
        "rna_mask": to_tensor(backbone.rna_mask.astype(np.float32), torch.float32),
        "R_polymer_type": to_tensor(backbone.polymer_type, torch.long),
        "S": to_tensor(backbone.S, torch.long),
        "chain_mask": to_tensor(mask, torch.float32),
        "bias": _DEFAULT_BIAS.to(device)[None, None, :].repeat(1, L, 1),
    }
    return FeaturizedNABackbone(
        feature_dict=feature_dict,
        chain_letters=backbone.chain_letters,
        rna_mask_for_token_conversion=backbone.rna_mask_for_token_conversion,
    )


def decode_sequence(
    S: torch.Tensor,
    rna_mask_for_token_conversion: np.ndarray,
    chain_letters: list[str],
) -> str:
    """Decode a sampled `S` row (shape [L]) into a `/`-joined, per-chain
    one-letter string, grouped by sorted chain letter (matching upstream's
    `mask_c` grouping — not necessarily first-appearance order)."""
    chars = []
    for i, idx in enumerate(S.tolist()):
        one = RESTYPE_INT_TO_ONE.get(idx, "X")
        if rna_mask_for_token_conversion[i]:
            one = DNA_CHAR_TO_RNA_CHAR.get(one, one)
        chars.append(one)

    chain_list = sorted(set(chain_letters))
    if len(chain_list) <= 1:
        return "".join(chars)
    parts = [
        "".join(c for c, cl in zip(chars, chain_letters) if cl == chain)
        for chain in chain_list
    ]
    return "/".join(parts)


def to_standard_nucleotide_alphabet(seq: str) -> str:
    """Translate one chain of a `decode_sequence` output out of NA-MPNN's
    internal one-letter alphabet into standard IUPAC nucleotide codes (see
    `STANDARD_NUCLEOTIDE_CHAR`). Protein positions pass through unchanged.

    Takes a single chain, not a `/`-joined multi-chain string — split on
    `/` first if needed (see `rna.diversity.compute_sequence_diversity`,
    which needs each chain as its own sequence for MMseqs2 anyway, since a
    literal `/` isn't a valid nucleotide/protein character).
    """
    return "".join(STANDARD_NUCLEOTIDE_CHAR.get(c, c) for c in seq)
