from __future__ import annotations

import logging
from typing import Any

import numpy as np
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.ml.transforms.base import TransformedDict
from atomworks.ml.transforms.rdkit_utils import AddRDKitMoleculesForAtomizedMolecules
from biotite.structure import AtomArray, BondList
from rdkit import Chem

from nucleus.utils.materials import get_material_lattice

try:
    from openbabel import openbabel
except ImportError:  # pragma: no cover - optional fallback
    openbabel = None


LOGGER = logging.getLogger(__name__)

# These are deliberately conservative relative to atomistic length scales. At the
# CT model's 5 Å cutoff, a 0.1 Å cell height already needs 50 replicas per side.
_MIN_AFD_RELATIVE_CELL_VOLUME = 1e-6
_MIN_AFD_CELL_HEIGHT_ANGSTROM = 0.1


def _is_valid_afd_periodic_cell(lattice_vectors: Any) -> bool:
    """Return whether a cell is safe for periodic AFD graph construction.

    An absolute determinant threshold is not sufficient here: the determinant
    scales cubically with the cell lengths, and a large but nearly coplanar cell
    can have a non-trivial determinant while requiring millions of periodic
    replicas. Check both the scale-free volume of the normalized cell and the
    physical distance between opposite cell planes.
    """
    cell = np.asarray(lattice_vectors, dtype=np.float64)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        return False

    vector_lengths = np.linalg.norm(cell, axis=1)
    if not np.all(np.isfinite(vector_lengths)) or np.any(vector_lengths <= 0.0):
        return False

    volume = abs(float(np.linalg.det(cell)))
    length_product = float(np.prod(vector_lengths))
    if not np.isfinite(volume) or not np.isfinite(length_product) or volume <= 0.0:
        return False

    relative_volume = volume / length_product
    if (
        not np.isfinite(relative_volume)
        or relative_volume < _MIN_AFD_RELATIVE_CELL_VOLUME
    ):
        return False

    a, b, c = cell
    opposite_face_areas = np.asarray(
        [
            np.linalg.norm(np.cross(b, c)),
            np.linalg.norm(np.cross(c, a)),
            np.linalg.norm(np.cross(a, b)),
        ]
    )
    if not np.all(np.isfinite(opposite_face_areas)) or np.any(
        opposite_face_areas <= 0.0
    ):
        return False

    cell_heights = volume / opposite_face_areas
    return bool(
        np.all(np.isfinite(cell_heights))
        and np.all(cell_heights >= _MIN_AFD_CELL_HEIGHT_ANGSTROM)
    )


def _ensure_atomarray_has_bonds(atom_array: AtomArray, *, example_id: str) -> AtomArray:
    """Validate that the input atom array already has a usable bond graph."""
    bonds = getattr(atom_array, "bonds", None)
    if bonds is None:
        raise ValueError(
            f"{example_id}: atom_array.bonds is missing. "
            "This script assumes bond annotations are already attached."
        )

    if isinstance(bonds, BondList):
        bond_array = bonds.as_array()
        if bond_array.size == 0:
            raise ValueError(
                f"{example_id}: atom_array.bonds is empty. "
                "This script expects known bonds to be attached."
            )
        return atom_array

    bond_array = np.asarray(bonds)
    if bond_array.size == 0:
        raise ValueError(
            f"{example_id}: atom_array.bonds is empty. "
            "This script expects known bonds to be attached."
        )
    if bond_array.ndim != 2 or bond_array.shape[1] != 3:
        raise ValueError(
            f"{example_id}: atom_array.bonds must be an (N, 3) bond array."
        )
    atom_array.bonds = BondList(atom_array.array_length(), bond_array.astype(int))
    return atom_array


def _atom_array_to_pdb_block(atom_array: AtomArray) -> str:
    """Serialize an AtomArray into a minimal PDB block for RDKit proximity bonding."""
    coords = np.asarray(atom_array.coord)
    if coords.ndim == 1:
        coords = coords[None, :]
    n_atoms = len(coords)

    def _annotation_array(name: str, default: Any) -> np.ndarray:
        value = getattr(atom_array, name, None)
        if value is None:
            return np.full(n_atoms, default)
        array = np.asarray(value)
        if array.ndim == 0:
            return np.full(n_atoms, array.item())
        if len(array) != n_atoms:
            array = np.resize(array, n_atoms)
        return array

    atom_names = _annotation_array("atom_name", "C")
    res_names = _annotation_array("res_name", "LIG")
    chain_ids = _annotation_array("chain_id", "A")
    res_ids = _annotation_array("res_id", 1)
    elements = _annotation_array("element", "C")

    lines: list[str] = []
    for idx, (coord, atom_name, res_name, chain_id, res_id, element) in enumerate(
        zip(coords, atom_names, res_names, chain_ids, res_ids, elements, strict=True),
        start=1,
    ):
        x, y, z = (float(coord[0]), float(coord[1]), float(coord[2]))
        lines.append(
            (
                f"HETATM{idx:5d} {str(atom_name).strip()[:4]:>4s} {str(res_name).strip()[:3]:>3s} "
                f"{str(chain_id).strip()[:1] or 'A':1s}{int(res_id):4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {str(element).strip().title()[:2]:>2s}  "
            )
        )

    lines.append("END")
    return "\n".join(lines) + "\n"


def _atom_array_coords(atom_array: Any) -> np.ndarray:
    """Return an (N, 3) coordinate array for an AtomArray-like object."""
    if hasattr(atom_array, "coord"):
        coords = np.asarray(getattr(atom_array, "coord"))
    elif hasattr(atom_array, "get_positions"):
        coords = np.asarray(atom_array.get_positions())
    else:
        raise ValueError("AtomArray object does not expose coordinates.")

    if coords.ndim == 1:
        coords = coords[None, :]
    if coords.shape[-1] != 3:
        raise ValueError("AtomArray coordinates must have shape (N, 3).")

    return coords


def _atom_array_elements(atom_array: Any, n_atoms: int) -> np.ndarray:
    """Return a per-atom element array with a carbon fallback."""
    if hasattr(atom_array, "element"):
        elements = getattr(atom_array, "element")
        if elements is not None:
            elements = np.asarray(elements)
    elif hasattr(atom_array, "get_chemical_symbols"):
        elements = np.asarray(atom_array.get_chemical_symbols())
    else:
        elements = None

    if elements is None:
        return np.full(n_atoms, "C", dtype="<U2")

    if elements.ndim == 0:
        return np.full(n_atoms, str(elements.item()).title(), dtype="<U2")

    if len(elements) != n_atoms:
        elements = np.resize(elements, n_atoms)

    normalized = np.array(
        [str(value).strip().title() or "C" for value in elements], dtype="<U4"
    )
    return normalized


def _atomic_numbers_from_atom_array(atom_array: Any, n_atoms: int) -> np.ndarray | None:
    """Return per-atom atomic numbers, or None if any element is unknown."""
    atomic_numbers = getattr(atom_array, "atomic_number", None)
    if atomic_numbers is not None:
        z = np.asarray(atomic_numbers)
        if z.ndim == 0:
            z = np.full(n_atoms, z.item())
        if len(z) != n_atoms:
            z = np.resize(z, n_atoms)
        z = z.astype(np.int64, copy=False)
        return z if np.all(z > 0) else None

    elements = _atom_array_elements(atom_array, n_atoms)
    z = []
    for element in elements:
        key = str(element).strip().upper()
        if key == "D":
            key = "H"
        z.append(ELEMENT_NAME_TO_ATOMIC_NUMBER.get(key, 0))

    atomic_numbers = np.asarray(z, dtype=np.int64)
    return atomic_numbers if np.all(atomic_numbers > 0) else None


def _atom_array_to_afd_record(atom_array: Any) -> dict[str, np.ndarray] | None:
    """Convert an AtomArray-like object into an AFD featurization record."""
    try:
        coords = _atom_array_coords(atom_array).astype(np.float32, copy=False)
    except Exception:
        return None

    if coords.ndim != 2 or coords.shape[-1] != 3 or len(coords) == 0:
        return None
    if not np.all(np.isfinite(coords)):
        return None

    atomic_numbers = _atomic_numbers_from_atom_array(atom_array, len(coords))
    if atomic_numbers is None:
        return None

    return {"z": atomic_numbers, "pos": coords}


def _atom_array_to_material_afd_record(
    atom_array: Any,
) -> dict[str, np.ndarray] | None:
    """Convert a periodic material AtomArray into an AFD featurization record."""
    record = _atom_array_to_afd_record(atom_array)
    if record is None:
        return None

    try:
        lattice = get_material_lattice(atom_array)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if lattice is None:
        return None

    _, _, lattice_vectors, material_mask = lattice
    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float32)
    material_mask = np.asarray(material_mask, dtype=bool)
    if material_mask.shape != record["z"].shape or not np.any(material_mask):
        return None
    if not _is_valid_afd_periodic_cell(lattice_vectors):
        return None

    return {
        "z": record["z"][material_mask],
        "pos": record["pos"][material_mask],
        "cell": lattice_vectors[None, :, :],
        "pbc": np.ones((1, 3), dtype=bool),
    }


def _atom_array_to_xyz_block(atom_array: Any, *, title: str) -> str:
    """Serialize an AtomArray-like object into a minimal XYZ block."""
    coords = _atom_array_coords(atom_array)
    elements = _atom_array_elements(atom_array, len(coords))

    lines = [str(len(coords)), title]
    for element, coord in zip(elements, coords, strict=True):
        x, y, z = (float(coord[0]), float(coord[1]), float(coord[2]))
        lines.append(f"{str(element).strip().title() or 'C'} {x:.8f} {y:.8f} {z:.8f}")

    return "\n".join(lines) + "\n"


def _smiles_from_rdkit_mol(mol: Chem.Mol) -> str | None:
    """Canonicalize an RDKit molecule into a SMILES string."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None


def _postprocess_openbabel_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Match the previous stripped-atom behavior, then sanitize the RDKit mol."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    editable_mol = Chem.RWMol(mol)
    for atom in editable_mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)

    processed_mol = editable_mol.GetMol()
    try:
        Chem.SanitizeMol(processed_mol)
    except Exception:
        return None
    return processed_mol


def _openbabel_obmol_from_atom_array(atom_array: Any, *, example_id: str) -> Any | None:
    """Use Open Babel bond perception to build an ``OBMol``."""
    if openbabel is None:
        return None

    xyz_block = _atom_array_to_xyz_block(atom_array, title=example_id)
    try:
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("xyz")
        ob_mol = openbabel.OBMol()
        if not ob_conversion.ReadString(ob_mol, xyz_block):
            return None

        ob_mol.ConnectTheDots()
        ob_mol.PerceiveBondOrders()
    except Exception:
        return None
    return ob_mol


def _openbabel_mol_from_atom_array(
    atom_array: Any, *, example_id: str
) -> Chem.Mol | None:
    """Use Open Babel bond perception to build an RDKit molecule."""
    ob_mol = _openbabel_obmol_from_atom_array(
        atom_array,
        example_id=example_id,
    )
    if ob_mol is None:
        return None

    try:
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetOutFormat("sdf")
        sdf_block = ob_conversion.WriteString(ob_mol)
        if not sdf_block:
            return None

        mol_block = sdf_block.split("$$$$", 1)[0]
        tmp_mol = Chem.MolFromMolBlock(
            mol_block,
            sanitize=False,
            removeHs=False,
        )
    except Exception:
        return None

    if tmp_mol is None:
        return None

    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    if tmp_mol.GetNumConformers():
        mol.AddConformer(tmp_mol.GetConformer(0), assignId=True)
    for bond in tmp_mol.GetBonds():
        mol.AddBond(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType(),
        )
    return _postprocess_openbabel_mol(mol.GetMol())


def _rdkit_smiles_from_atom_array(
    atom_array: Any, *, example_id: str, return_mol_only: bool = False
) -> str | Chem.Mol | None:
    """Build an RDKit molecule via OpenBabel bond perception."""
    mol = _openbabel_mol_from_atom_array(atom_array, example_id=example_id)
    if mol is None:
        return None

    if return_mol_only:
        return mol

    smiles = _smiles_from_rdkit_mol(mol)
    if smiles:
        return smiles

    return None


def _atom_array_to_smiles(
    atom_array: Any, *, example_id: str = "atom_array", return_mol_only: bool = False
) -> str | Chem.Mol | None:
    """Best-effort conversion from an AtomArray-like object to a SMILES string or RDKit Mol."""
    mol_or_smiles = _rdkit_smiles_from_atom_array(
        atom_array, example_id=example_id, return_mol_only=return_mol_only
    )
    if mol_or_smiles is not None:
        return mol_or_smiles

    return None


def _normalize_atom_array_for_rdkit(
    atom_array: AtomArray, *, example_id: str
) -> AtomArray:
    """Attach the annotations required by AddRDKitMoleculesForAtomizedMolecules."""
    try:
        atom_array = _ensure_atomarray_has_bonds(atom_array, example_id=example_id)
    except ValueError:
        pdb_block = _atom_array_to_pdb_block(atom_array)
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
        if mol is None or mol.GetNumAtoms() == 0:
            raise
        atom_array = atom_array_from_rdkit(
            mol, remove_hydrogens=False, remove_inferred_atoms=False
        )
        atom_array = _ensure_atomarray_has_bonds(atom_array, example_id=example_id)

    n_atoms = atom_array.array_length()
    if (
        not hasattr(atom_array, "atomize")
        or getattr(atom_array, "atomize", None) is None
    ):
        atom_array.set_annotation("atomize", np.ones(n_atoms, dtype=bool))
    if (
        not hasattr(atom_array, "pn_unit_iid")
        or getattr(atom_array, "pn_unit_iid", None) is None
    ):
        atom_array.set_annotation(
            "pn_unit_iid", np.full(n_atoms, f"{example_id}_0", dtype="<U32")
        )

    return atom_array


def _atomworks_smiles_from_atom_array(
    atom_array: AtomArray, *, example_id: str, return_mol_only: bool = False
) -> str | Chem.Mol | None:
    """Attempt the legacy AtomWorks transform path as a fallback."""
    try:
        atom_array = _normalize_atom_array_for_rdkit(atom_array, example_id=example_id)
    except Exception:
        return None

    transform = AddRDKitMoleculesForAtomizedMolecules(hydrogen_policy="keep")
    data = TransformedDict({"atom_array": atom_array})
    # AtomWorks expects this transform to follow AtomizeByCCDName in a pipeline.
    # For this standalone export we synthesize the minimal history entry it needs.
    data.__transform_history__ = [
        {
            "name": "AtomizeByCCDName",
            "instance": "synthetic-atomize-by-ccd-name",
        }
    ]

    try:
        data = transform(data)
    except Exception:
        return None

    rdkit_mols = data.get("rdkit", {})
    if not rdkit_mols:
        return None

    # A single molecule example should produce a single RDKit Mol.
    mol = next(iter(rdkit_mols.values()))
    if return_mol_only:
        return mol
    return _smiles_from_rdkit_mol(mol)


def _smiles_from_atom_array(
    atom_array: AtomArray,
    *,
    example_id: str,
    log_failures: bool = True,
) -> str | None:
    """Convert an AtomArray to a canonical RDKit SMILES string."""
    smiles = _atom_array_to_smiles(atom_array, example_id=example_id)
    if smiles:
        return smiles

    smiles = _atomworks_smiles_from_atom_array(atom_array, example_id=example_id)
    if smiles:
        return smiles

    if log_failures:
        LOGGER.warning(
            "%s: conversion failed; no valid molecule-derived SMILES was produced",
            example_id,
        )
    return None


def _mol_from_atom_array(
    atom_array: AtomArray,
    *,
    example_id: str,
    infer_bonds: bool = True,
    log_failures: bool = True,
) -> Any | None:
    """Convert an AtomArray to a canonical RDKit Mol object."""
    if not infer_bonds:
        mol = _atomworks_smiles_from_atom_array(
            atom_array, example_id=example_id, return_mol_only=True
        )
        if mol:
            return mol

    mol = _atom_array_to_smiles(atom_array, example_id=example_id, return_mol_only=True)
    if mol:
        return mol

    if log_failures:
        LOGGER.warning(
            "%s: conversion failed; no valid RDKit Mol was produced",
            example_id,
        )
    return None
