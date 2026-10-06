"""Lossless scientific-field normalization for generated materials mmCIFs.

This module deliberately imports only the standard library at module scope. It
is also imported directly by the isolated CHGNet worker, without installing
``zevals`` or AtomWorks in the oracle environment.
"""

from __future__ import annotations

import gzip
from pathlib import Path

MATERIAL_CIF_NORMALIZATION_VERSION = 1

# AtomWorks writes the PDBx/mmCIF dictionary, while materials CIF readers expect
# the core CIF dictionary. Only documented equivalents are aliased: Cartesian
# coordinates are deliberately not used to reconstruct fractional coordinates.
_TAG_ALIASES = {
    "_cell.length_a": "_cell_length_a",
    "_cell.length_b": "_cell_length_b",
    "_cell.length_c": "_cell_length_c",
    "_cell.angle_alpha": "_cell_angle_alpha",
    "_cell.angle_beta": "_cell_angle_beta",
    "_cell.angle_gamma": "_cell_angle_gamma",
    "_cell.volume": "_cell_volume",
    "_space_group.name_h-m_alt": "_space_group_name_H-M_alt",
    "_space_group.name_hall": "_space_group_name_Hall",
    "_space_group.it_number": "_space_group_IT_number",
    "_symmetry.space_group_name_h-m": "_symmetry_space_group_name_H-M",
    "_symmetry.int_tables_number": "_symmetry_Int_Tables_number",
    "_symmetry.space_group_name_hall": "_symmetry_space_group_name_Hall",
    "_space_group_symop.id": "_space_group_symop_id",
    "_space_group_symop.operation_xyz": "_space_group_symop_operation_xyz",
    "_symmetry_equiv.id": "_symmetry_equiv_pos_site_id",
    "_symmetry_equiv.pos_as_xyz": "_symmetry_equiv_pos_as_xyz",
    "_atom_site.label_atom_id": "_atom_site_label",
    "_atom_site.label": "_atom_site_label",
    "_atom_site.type_symbol": "_atom_site_type_symbol",
    "_atom_site.fract_x": "_atom_site_fract_x",
    "_atom_site.fract_y": "_atom_site_fract_y",
    "_atom_site.fract_z": "_atom_site_fract_z",
    "_atom_site.occupancy": "_atom_site_occupancy",
    "_atom_site.u_iso_or_equiv": "_atom_site_U_iso_or_equiv",
    "_atom_site.b_iso_or_equiv": "_atom_site_B_iso_or_equiv",
}
_CORE_CASE = {value.lower(): value for value in _TAG_ALIASES.values()}


class MaterialCifConversionError(ValueError):
    """A recognized materials mmCIF cannot be normalized unambiguously."""


def normalize_material_cif(path: Path) -> bytes:
    """Return materials-CIF bytes without changing the source file.

    Core CIF inputs are returned byte-for-byte, including their gzip framing.
    PDBx/mmCIF inputs are parsed into CIF blocks and their equivalent cell,
    symmetry, and atom-site tags are renamed. Values, site order, occupancies,
    and fractional coordinates are preserved; no wrapping, sorting, lattice
    reconstruction, symmetry finding, or floating-point rounding is performed.
    If ``path`` ends in ``.gz``, normalized output uses deterministic gzip.

    Malformed input remains an unsuccessful scientific attempt for callers to
    account for, not a reason to omit a sample. In particular, callers should
    classify ``MaterialCifConversionError`` as a parsing failure. Files without
    recognized mmCIF tags are left to the downstream reader to validate.

    Args:
        path: Source ``.cif`` or ``.cif.gz`` path.

    Returns:
        Bytes suitable for a destination with the same compression extension.

    Raises:
        MaterialCifConversionError: Recognized tags cannot be safely converted.
        OSError: The source cannot be read or decompressed.
        ImportError: pymatgen is unavailable when conversion is required.
    """
    path = Path(path)
    original = path.read_bytes()
    compressed = path.name.lower().endswith(".gz")
    source = gzip.decompress(original) if compressed else original
    # This is only a fast-path guard, not the parser: aliases are applied to
    # parsed data keys, never to values, quoted strings, or comments.
    lowered = source.lower()
    if not any(tag.encode("ascii") in lowered for tag in _TAG_ALIASES):
        return original

    from pymatgen.io.cif import CifBlock, CifFile

    try:
        document = CifFile.from_str(source.decode("utf-8"))
        if not document.data:
            raise ValueError("no CIF data blocks")
        changed = False
        for name, block in list(document.data.items()):
            if not any(key.lower() in _TAG_ALIASES for key in block.data):
                continue
            changed = True
            renamed = {
                key: _TAG_ALIASES.get(key.lower(), _CORE_CASE.get(key.lower(), key))
                for key in block.data
            }
            data = {}
            for key, value in block.data.items():
                target = renamed[key]
                if target in data and data[target] != value:
                    raise ValueError(f"conflicting values for {target}")
                data[target] = value
            loops = []
            assigned = set()
            for loop in block.loops:
                normalized = list(dict.fromkeys(renamed[key] for key in loop))
                if assigned.intersection(normalized):
                    raise ValueError("aliased tags occur in different CIF loops")
                assigned.update(normalized)
                lengths = {len(data[key]) for key in normalized}
                if len(lengths) != 1:
                    raise ValueError("atom-site or symmetry loop lengths differ")
                loops.append(normalized)
            document.data[name] = CifBlock(data, loops, block.header)
        if not changed:
            return original
        normalized_bytes = str(document).encode("utf-8")
    except (ValueError, UnicodeError, IndexError, KeyError, ZeroDivisionError) as exc:
        raise MaterialCifConversionError(
            f"Cannot normalize {path.name}: {exc}"
        ) from exc
    return gzip.compress(normalized_bytes, mtime=0) if compressed else normalized_bytes
