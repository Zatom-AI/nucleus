"""PDB parsing and tensor featurization for ProteinMPNN, vendored and trimmed
from `protein_mpnn_utils.py` in https://github.com/dauparas/ProteinMPNN
(MIT License, Copyright (c) 2022 Justas Dauparas).

Upstream's `tied_featurize` supports fixed positions, tied multimer
positions, PSSM biasing and per-residue AA biasing — none of which zevals
needs for the designability metric (full redesign of every residue in a
single backbone, no side constraints). `featurize_backbone` below covers
exactly that case: every chain in the PDB is masked (i.e. designed), in
the same order it appears in the file, matching upstream's behaviour when
`chain_id_jsonl` is not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

_ALPHA_1 = list("ARNDCQEGHILKMFPSTWYV-")
_ALPHA_3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "GAP",
]  # fmt: skip
_AA_3_TO_N = {a: n for n, a in enumerate(_ALPHA_3)}
_AA_N_TO_1 = {n: a for n, a in enumerate(_ALPHA_1)}

_CHAIN_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + [
    str(i) for i in range(300)
]


def _parse_pdb_biounit(
    path_to_pdb: str, atoms: list[str], chain: str
) -> tuple[np.ndarray | str, list[str] | str]:
    """Extract per-residue coordinates + sequence for one chain of a PDB file.

    Returns ("no_chain", "no_chain") if `chain` is absent from the file.
    """
    xyz: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    seq: dict[int, dict[str, str]] = {}
    min_resn, max_resn = 1e6, -1e6

    for raw_line in open(path_to_pdb, "rb"):
        line = raw_line.decode("utf-8", "ignore").rstrip()
        if line[:6] == "HETATM" and line[17:20] == "MSE":
            line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")
        if line[:4] != "ATOM":
            continue
        if line[21:22] != chain:
            continue

        atom = line[12:16].strip()
        resi = line[17:20]
        resn_field = line[22:27].strip()
        x, y, z = (float(line[i : i + 8]) for i in [30, 38, 46])

        if resn_field[-1].isalpha():
            resa, resn = resn_field[-1], int(resn_field[:-1]) - 1
        else:
            resa, resn = "", int(resn_field) - 1
        min_resn, max_resn = min(min_resn, resn), max(max_resn, resn)

        xyz.setdefault(resn, {}).setdefault(resa, {})
        seq.setdefault(resn, {}).setdefault(resa, resi)
        if atom not in xyz[resn][resa]:
            xyz[resn][resa][atom] = np.array([x, y, z])

    if max_resn < min_resn:
        return "no_chain", "no_chain"

    seq_out, xyz_out = [], []
    for resn in range(int(min_resn), int(max_resn) + 1):
        if resn in seq:
            for k in sorted(seq[resn]):
                seq_out.append(_AA_3_TO_N.get(seq[resn][k], 20))
        else:
            seq_out.append(20)
        if resn in xyz:
            for k in sorted(xyz[resn]):
                for atom in atoms:
                    if atom in xyz[resn][k]:
                        xyz_out.append(xyz[resn][k][atom])
                    else:
                        xyz_out.append(np.full(3, np.nan))
        else:
            for _atom in atoms:
                xyz_out.append(np.full(3, np.nan))

    sequence = "".join(_AA_N_TO_1.get(n, "-") for n in seq_out)
    return np.array(xyz_out).reshape(-1, len(atoms), 3), [sequence]


@dataclass
class ParsedBackbone:
    """A single (possibly multi-chain) backbone parsed from a PDB file."""

    name: str
    chain_letters: list[str]  # chain order as it appears in the file
    seq_by_chain: dict[str, str]
    coords_by_chain: dict[str, np.ndarray]  # [chain_len, n_atoms, 3]


def parse_pdb(path_to_pdb: Path, ca_only: bool = False) -> ParsedBackbone:
    """Parse all chains of a PDB file into coordinates + sequence.

    Args:
        path_to_pdb: Path to a single-model PDB file.
        ca_only:     If True, only Cα coordinates are read (matches
                     CA-ProteinMPNN checkpoints). Otherwise N, CA, C, O.

    Returns:
        ParsedBackbone with one entry per chain found in the file.
    """
    atoms = ["CA"] if ca_only else ["N", "CA", "C", "O"]
    seq_by_chain: dict[str, str] = {}
    coords_by_chain: dict[str, np.ndarray] = {}
    for letter in _CHAIN_LETTERS:
        xyz, seq = _parse_pdb_biounit(str(path_to_pdb), atoms=atoms, chain=letter)
        if isinstance(xyz, str):
            continue
        seq_by_chain[letter] = seq[0]
        coords_by_chain[letter] = xyz

    return ParsedBackbone(
        name=Path(path_to_pdb).stem,
        chain_letters=list(seq_by_chain.keys()),
        seq_by_chain=seq_by_chain,
        coords_by_chain=coords_by_chain,
    )


@dataclass
class FeaturizedBackbone:
    """Batch-ready (batch size 1) tensors for `ProteinMPNN.forward`/`.sample`."""

    X: torch.Tensor  # [1, L, n_atoms, 3] (or [1, L, 3] if ca_only)
    S: torch.Tensor  # [1, L] native sequence, integer-encoded
    mask: torch.Tensor  # [1, L] 1.0 for real residues
    chain_M: torch.Tensor  # [1, L] 1.0 for designed positions (all of them here)
    chain_M_pos: torch.Tensor  # [1, L] 1.0 unless fixed (always 1.0 here)
    chain_encoding_all: torch.Tensor  # [1, L] integer chain id per residue
    residue_idx: torch.Tensor  # [1, L] chain-offset residue index
    omit_AA_mask: torch.Tensor  # [1, L, 21]
    bias_by_res: torch.Tensor  # [1, L, 21]
    chain_lengths: list[int]  # residue count per chain, in `chain_letters` order


def featurize_backbone(
    backbone: ParsedBackbone, device: torch.device, ca_only: bool = False
) -> FeaturizedBackbone:
    """Convert a `ParsedBackbone` into tensors with every chain marked as
    designed (equivalent to running upstream ProteinMPNN with no
    `--chain_id_jsonl` / `--fixed_positions_jsonl`)."""
    chain_letters = backbone.chain_letters
    chain_lengths = [len(backbone.seq_by_chain[c]) for c in chain_letters]
    L = sum(chain_lengths)

    x_chunks, encoding_chunks, residue_idx = [], [], np.zeros(L, dtype=np.int64)
    seq = ""
    offset = 0
    for chain_num, letter in enumerate(chain_letters):
        n = chain_lengths[chain_num]
        coords = backbone.coords_by_chain[letter]
        x_chunks.append(coords)
        encoding_chunks.append(np.full(n, chain_num + 1, dtype=np.int64))
        residue_idx[offset : offset + n] = 100 * chain_num + np.arange(n)
        seq += backbone.seq_by_chain[letter].replace("-", "X")
        offset += n

    X = np.concatenate(x_chunks, axis=0)  # [L, n_atoms, 3]
    if ca_only:
        X = X[:, 0, :]  # -> [L, 3]

    S = np.array([ALPHABET.index(a) for a in seq], dtype=np.int64)
    mask = np.isfinite(X.reshape(L, -1).sum(-1)).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device).unsqueeze(0)

    return FeaturizedBackbone(
        X=to_tensor(X, torch.float32),
        S=to_tensor(S, torch.long),
        mask=to_tensor(mask, torch.float32),
        chain_M=to_tensor(np.ones(L, dtype=np.float32), torch.float32),
        chain_M_pos=to_tensor(np.ones(L, dtype=np.float32), torch.float32),
        chain_encoding_all=to_tensor(np.concatenate(encoding_chunks), torch.long),
        residue_idx=to_tensor(residue_idx, torch.long),
        omit_AA_mask=to_tensor(
            np.zeros((L, len(ALPHABET)), dtype=np.float32), torch.float32
        ),
        bias_by_res=to_tensor(
            np.zeros((L, len(ALPHABET)), dtype=np.float32), torch.float32
        ),
        chain_lengths=chain_lengths,
    )


def decode_sequence(S: torch.Tensor, chain_lengths: list[int]) -> str:
    """Decode a sampled `S` row (shape [L]) into a `/`-joined chain string."""
    flat = "".join(ALPHABET[i] for i in S.tolist())
    if len(chain_lengths) <= 1:
        return flat
    parts, offset = [], 0
    for n in chain_lengths:
        parts.append(flat[offset : offset + n])
        offset += n
    return "/".join(parts)


def sequence_scores(
    S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sequence average negative log-likelihood (lower = more confident)."""
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    return torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
