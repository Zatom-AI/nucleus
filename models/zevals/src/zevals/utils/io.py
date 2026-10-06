"""Structure loading and saving utilities.

Thin wrappers around common I/O operations so domain modules stay format-agnostic.

Two, deliberately distinct, CIF readers exist here — don't conflate them:
  - `load_structure_cif` — pymatgen `Structure`, for *materials* (periodic
    crystals: unit cell, fractional coordinates). Architecturally wrong for
    biomolecules, which have neither.
  - `load_structure_mmcif` — Bio.PDB `Structure`, for *biomolecule* mmCIF
    (proteins/RNA/ligands): same object model `load_structure_pdb` already
    returns, just parsed from mmCIF instead of legacy PDB text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_smiles(path: Path) -> list[str]:
    """Load SMILES strings from a .smi file (one per line, strip comments)."""
    lines = path.read_text().splitlines()
    return [
        line.split()[0] for line in lines if line.strip() and not line.startswith("#")
    ]


def load_structure_pdb(path: Path) -> Any:
    """Parse a PDB file into a Bio.PDB Structure object."""
    try:
        from Bio.PDB import PDBParser  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install biopython: pip install biopython") from exc

    parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def load_structure_mmcif(path: Path) -> Any:
    """Parse a biomolecule mmCIF file into a Bio.PDB Structure object.

    Same return type as `load_structure_pdb` (both are `Bio.PDB.Structure`),
    so callers that already walk `.get_residues()`/`"CA" in residue` etc.
    need no further changes beyond picking the right loader for the file's
    extension. Not to be confused with `load_structure_cif` (pymatgen,
    materials-only — see module docstring).
    """
    try:
        from Bio.PDB import MMCIFParser  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install biopython: pip install biopython") from exc

    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def load_structure_cif(path: Path) -> Any:
    """Parse a periodic CIF into a pymatgen Structure.

    Pymatgen is attempted first. Zatom/AtomWorks CIFs use PDBx atom labels
    that some pymatgen versions reject, so the fallback reads the same
    fractional coordinates and cell parameters directly. Fractional fields
    are intentionally preferred over Cartesian fields because they define a
    periodic structure independently of the lattice-vector orientation.
    """
    try:
        import numpy as np
        from atomworks.io.utils.io_utils import read_any
        from pymatgen.core import Lattice, Structure  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install atomworks and pymatgen to load CIF files.") from exc

    try:
        return Structure.from_file(str(path))
    except Exception:
        try:
            block = read_any(path).block
            cell = block["cell"]
            lengths = [
                float(cell[key].as_array(float)[0])
                for key in ("length_a", "length_b", "length_c")
            ]
            angles = [
                float(cell[key].as_array(float)[0])
                for key in ("angle_alpha", "angle_beta", "angle_gamma")
            ]
            atom_site = block["atom_site"]
            species = atom_site["type_symbol"].as_array(str)
            fractional_keys = ("fract_x", "fract_y", "fract_z")
            if all(key in atom_site for key in fractional_keys):
                coordinates = np.stack(
                    [atom_site[key].as_array(float) for key in fractional_keys],
                    axis=-1,
                )
                are_cartesian = False
            else:
                cartesian_keys = ("Cartn_x", "Cartn_y", "Cartn_z")
                coordinates = np.stack(
                    [atom_site[key].as_array(float) for key in cartesian_keys],
                    axis=-1,
                )
                are_cartesian = True
            return Structure(
                Lattice.from_parameters(*lengths, *angles),
                species,
                coordinates,
                coords_are_cartesian=are_cartesian,
            )
        except Exception as fallback_error:
            raise ValueError(
                f"Could not parse periodic CIF {path}."
            ) from fallback_error


def save_structure_pdb(structure: Any, path: Path) -> None:
    """Write a Bio.PDB Structure to a PDB file."""
    try:
        from Bio.PDB import PDBIO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install biopython: pip install biopython") from exc

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def convert_cif_to_pdb(cif_path: Path, out_path: Path) -> Path:
    """Fallback conversion for tools that can't read mmCIF directly.

    Prefer handing CIF to a library that reads it natively wherever one
    exists (ProDy, Biotite, TMalign/qTMclust/FoldSeek all already do) — this
    is only for the remaining callers that are hard-coded to PDB text (e.g.
    `_protein_mpnn`'s vendored fixed-column parser).

    Uses Biopython for both ends (`MMCIFParser` -> `PDBIO`), not because
    it's the most feature-complete PDB writer available, but because it
    already raises loudly — `Bio.PDB.PDBIOException` — on exactly the ways
    this conversion can lose information relative to the source CIF: a
    chain ID longer than one character, a residue number above PDB's 4-digit
    field width, or an atom serial number overflow. That exception is left
    to propagate (only wrapped with the source path for context) rather than
    caught, so a lossy conversion fails the caller instead of silently
    writing a corrupted structure.

    Args:
        cif_path: Source mmCIF file.
        out_path: Destination PDB file path (parent dir must already exist).

    Returns:
        `out_path`, for chaining.

    Raises:
        Bio.PDB.PDBExceptions.PDBIOException: if `structure` can't be
            represented losslessly in PDB's fixed-width fields.
    """
    try:
        from Bio.PDB.PDBExceptions import PDBIOException  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install biopython: pip install biopython") from exc

    structure = load_structure_mmcif(cif_path)

    n_models = len(list(structure))
    if n_models > 1:
        logger.warning(
            "%s has %d models; converting only the first to PDB "
            "(Bio.PDB.PDBIO writes whichever model .set_structure() was "
            "given, silently, so this is called out explicitly here).",
            cif_path,
            n_models,
        )

    try:
        save_structure_pdb(structure, out_path)
    except PDBIOException as exc:
        raise PDBIOException(
            f"Can't losslessly convert {cif_path} to PDB: {exc}"
        ) from exc
    return out_path


def glob_cifs(directory: Path) -> list[Path]:
    """Return all CIF files in a directory, sorted by name."""
    return sorted(directory.glob("*.cif"))


def glob_sdfs(directory: Path) -> list[Path]:
    """Return all SDF files in a directory, sorted by name."""
    return sorted(directory.glob("*.sdf"))


def glob_structures(directory: Path) -> list[Path]:
    """Return every `.cif`/`.pdb` biomolecule structure file in a directory.

    When both `<stem>.cif` and `<stem>.pdb` exist for the same sample, only
    the `.cif` is kept (logged, not silently dropped) — CIF is the preferred
    format going forward and carries strictly more information, so a stray
    leftover `.pdb` shouldn't cause the sample to be double-counted or the
    lower-fidelity file to win.
    """
    cifs = {p.stem: p for p in directory.glob("*.cif")}
    pdbs = {p.stem: p for p in directory.glob("*.pdb")}

    dupes = sorted(set(cifs) & set(pdbs))
    for stem in dupes:
        logger.debug(
            "%s has both %s.cif and %s.pdb — using the .cif, ignoring the .pdb",
            directory,
            stem,
            stem,
        )

    merged = {**pdbs, **cifs}  # cifs second so it wins on stem collision
    return sorted(merged.values())
