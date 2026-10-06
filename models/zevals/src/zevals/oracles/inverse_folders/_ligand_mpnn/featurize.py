"""PDB + ligand-context parsing and featurization for LigandMPNN, vendored
and trimmed from `data_utils.py` in https://github.com/dauparas/LigandMPNN
(MIT License, Copyright (c) 2024 Justas Dauparas).

Uses ProDy (https://pypi.org/project/ProDy) for PDB parsing — the same
library upstream depends on for robust HETATM/element typing; hand-rolling
that parsing (metals, waters, non-standard residues, alternate locations)
would just be a worse reimplementation of what ProDy already does.

Trimmed relative to upstream `parse_PDB` + `get_nearest_neighbours` +
`featurize`: every residue in the file is designed (no fixed positions, no
per-residue AA bias/omission, no symmetry ties) and only backbone (N, CA,
C, O) + nearest ligand-atom context is parsed (no all-37-atom / side-chain
context mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from zevals.oracles.inverse_folders._protein_mpnn.featurize import ALPHABET

_RESTYPE_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}  # fmt: skip

# Index+1 = atomic number; matches upstream `element_dict`/`element_dict_rev`.
_ELEMENT_LIST = [
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
    "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR",
    "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR",
    "RB", "SR", "Y", "ZR", "NB", "MB", "TC", "RU", "RH", "PD", "AG", "CD",
    "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
    "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF",
    "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO",
    "AT", "RN", "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM",
    "BK", "CF", "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS",
    "MT", "DS", "RG", "CN", "UUT", "FL", "UUP", "LV", "UUS", "UUO",
]  # fmt: skip
_ELEMENT_TO_INT = {el: i + 1 for i, el in enumerate(_ELEMENT_LIST)}


@dataclass
class ParsedLigandBackbone:
    """A single (possibly multi-chain) backbone + its ligand atom context."""

    name: str
    X: np.ndarray  # [L, 4, 3] N, CA, C, O
    mask: np.ndarray  # [L] 1.0 where all 4 backbone atoms are present
    S: np.ndarray  # [L] native sequence, integer-encoded
    R_idx: np.ndarray  # [L] residue index (PDB numbering)
    chain_labels: np.ndarray  # [L] integer chain id per residue
    chain_lengths: list[int]  # residues per chain, in file order
    Y_all: np.ndarray  # [M_total, 3] all candidate ligand-context atom coords
    Y_t_all: np.ndarray  # [M_total] element type (1-118) per candidate atom


def _aligned_coords(
    protein_atoms, ca_dict: dict, atom_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Align one backbone atom type's coordinates onto the residue index used by `ca_dict`."""
    atom_sel = protein_atoms.select(f"name {atom_name}")
    coords = np.zeros([len(ca_dict), 3], np.float32)
    mask = np.zeros([len(ca_dict)], np.int32)
    if atom_sel is not None:
        atom_coords = atom_sel.getCoords()
        atom_resnums = atom_sel.getResnums()
        atom_chain_ids = atom_sel.getChids()
        atom_icodes = atom_sel.getIcodes()
        for i in range(len(atom_resnums)):
            code = f"{atom_chain_ids[i]}_{atom_resnums[i]}_{atom_icodes[i]}"
            if code in ca_dict:
                coords[ca_dict[code], :] = atom_coords[i]
                mask[ca_dict[code]] = 1
    return coords, mask


def parse_pdb_with_ligand(path: Path) -> ParsedLigandBackbone:
    """Parse a PDB file's protein backbone and its candidate ligand-context atoms.

    Args:
        path: Path to a single-model PDB file. Protein ATOM records provide
            the backbone; all non-water HETATM (and non-protein ATOM, e.g.
            nucleic acids) records are candidate ligand context.

    Returns:
        ParsedLigandBackbone with backbone arrays and the *unfiltered* pool
        of candidate context atoms (nearest-neighbor selection down to
        `atom_context_num` happens later, per-residue, in
        `featurize_ligand_backbone`).
    """
    from prody import confProDy, parsePDB

    confProDy(verbosity="none")

    atoms = parsePDB(str(path))
    atoms = atoms.select("occupancy > 0")
    protein_atoms = atoms.select("protein")
    other_atoms = atoms.select("not protein and not water")

    if protein_atoms is None:
        raise ValueError(f"No protein atoms found in {path}")

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    ca_dict: dict[str, int] = {}
    for i in range(len(CA_resnums)):
        code = f"{CA_chain_ids[i]}_{CA_resnums[i]}_{CA_icodes[i]}"
        ca_dict[code] = i

    L = len(ca_dict)
    xyz4 = np.zeros([L, 4, 3], np.float32)
    xyz4_m = np.zeros([L, 4], np.int32)
    for atom_idx, atom_name in enumerate(["N", "CA", "C", "O"]):
        coords, m = _aligned_coords(protein_atoms, ca_dict, atom_name)
        xyz4[:, atom_idx, :] = coords
        xyz4_m[:, atom_idx] = m
    mask = xyz4_m.prod(axis=1).astype(np.float32)

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int64)
    R_idx = np.array(CA_resnums, dtype=np.int64)
    resnames = CA_atoms.getResnames()
    seq = "".join(_RESTYPE_3TO1.get(aa, "X") for aa in resnames)
    S = np.array([ALPHABET.index(a) for a in seq], dtype=np.int64)

    chain_lengths: list[int] = []
    prev_label = None
    for label in chain_labels:
        if label == prev_label:
            chain_lengths[-1] += 1
        else:
            chain_lengths.append(1)
        prev_label = label

    if other_atoms is not None and len(other_atoms) > 0:
        Y_all = np.array(other_atoms.getCoords(), dtype=np.float32)
        elems = [str(e).upper() for e in other_atoms.getElements()]
        Y_t_all = np.array([_ELEMENT_TO_INT.get(e, 0) for e in elems], dtype=np.int64)
        keep = (Y_t_all != _ELEMENT_TO_INT["H"]) & (
            Y_t_all != 0
        )  # drop H + unknown elements
        Y_all = Y_all[keep]
        Y_t_all = Y_t_all[keep]
    else:
        Y_all = np.zeros([0, 3], np.float32)
        Y_t_all = np.zeros([0], np.int64)

    return ParsedLigandBackbone(
        name=Path(path).stem,
        X=xyz4,
        mask=mask,
        S=S,
        R_idx=R_idx,
        chain_labels=chain_labels,
        chain_lengths=chain_lengths,
        Y_all=Y_all,
        Y_t_all=Y_t_all,
    )


@dataclass
class FeaturizedLigandBackbone:
    """Batch-ready (batch size 1) tensors for `LigandMPNN.sample`."""

    feature_dict: dict[
        str, torch.Tensor
    ]  # X, S, mask, R_idx, chain_labels, Y, Y_t, Y_m
    chain_lengths: list[int]
    num_context_atoms: int  # count of real (non-padding) ligand atoms found


def _nearest_ligand_atoms(
    CB: torch.Tensor,
    mask: torch.Tensor,
    Y: torch.Tensor,
    Y_t: torch.Tensor,
    atom_context_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-residue top-`atom_context_num` nearest ligand atoms by Cβ distance.

    Mirrors upstream `get_nearest_neighbours`: pads with zeros (masked out
    via `Y_m`) when fewer than `atom_context_num` candidate atoms exist.
    """
    device = CB.device
    L = CB.shape[0]
    M = Y.shape[0]
    if M == 0:
        return (
            torch.zeros([L, atom_context_num, 3], dtype=torch.float32, device=device),
            torch.zeros([L, atom_context_num], dtype=torch.int64, device=device),
            torch.zeros([L, atom_context_num], dtype=torch.int32, device=device),
        )

    Y_m_flat = torch.ones([M], dtype=torch.int32, device=device)
    mask_CBY = mask[:, None] * Y_m_flat[None, :]
    L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    k = min(atom_context_num, M)
    nn_idx = torch.argsort(L2_AB, -1)[:, :k]

    Y_r = Y[None, :, :].repeat(L, 1, 1)
    Y_t_r = Y_t[None, :].repeat(L, 1)
    Y_m_r = Y_m_flat[None, :].repeat(L, 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y_out = torch.zeros([L, atom_context_num, 3], dtype=torch.float32, device=device)
    Y_t_out = torch.zeros([L, atom_context_num], dtype=torch.int64, device=device)
    Y_m_out = torch.zeros([L, atom_context_num], dtype=torch.int32, device=device)
    Y_out[:, :k] = Y_tmp
    Y_t_out[:, :k] = Y_t_tmp
    Y_m_out[:, :k] = Y_m_tmp
    return Y_out, Y_t_out, Y_m_out


def featurize_ligand_backbone(
    backbone: ParsedLigandBackbone, device: torch.device, atom_context_num: int
) -> FeaturizedLigandBackbone:
    """Convert a `ParsedLigandBackbone` into `LigandMPNN.sample`-ready tensors.

    Every residue is marked designable (`chain_mask` all ones — no fixed
    positions), matching running upstream LigandMPNN with no
    `--fixed_residues`/`--redesigned_residues`.
    """
    L = backbone.X.shape[0]

    def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device).unsqueeze(0)

    X = to_tensor(backbone.X, torch.float32)
    mask = to_tensor(backbone.mask, torch.float32)
    S = to_tensor(backbone.S, torch.long)
    R_idx = to_tensor(backbone.R_idx, torch.long)
    chain_labels = to_tensor(backbone.chain_labels, torch.long)

    CB_N = X[0, :, 0, :]
    CB_CA = X[0, :, 1, :]
    CB_C = X[0, :, 2, :]
    b = CB_CA - CB_N
    c = CB_C - CB_CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CB_CA

    Y_all = torch.as_tensor(backbone.Y_all, dtype=torch.float32, device=device)
    Y_t_all = torch.as_tensor(backbone.Y_t_all, dtype=torch.int64, device=device)
    Y, Y_t, Y_m = _nearest_ligand_atoms(CB, mask[0], Y_all, Y_t_all, atom_context_num)

    feature_dict = {
        "X": X,
        "mask": mask,
        "S": S,
        "R_idx": R_idx,
        "chain_labels": chain_labels,
        "chain_mask": torch.ones([1, L], dtype=torch.float32, device=device),
        "bias": torch.zeros([1, L, 21], dtype=torch.float32, device=device),
        "Y": Y.unsqueeze(0),
        "Y_t": Y_t.unsqueeze(0),
        "Y_m": Y_m.unsqueeze(0),
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
    }
    return FeaturizedLigandBackbone(
        feature_dict=feature_dict,
        chain_lengths=backbone.chain_lengths,
        num_context_atoms=int(Y_m.sum().item()),
    )
