"""Materials mmCIF normalization without changing generated scientific values."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from zevals.materials._material_cif import (
    MaterialCifConversionError,
    normalize_material_cif,
)

pytest.importorskip("pymatgen.io.cif")
from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.cif import CifFile  # noqa: E402

MMCIF = """data_generated
_entry.id generated
_cell.length_a 3.8433204908215908
_cell.length_b 3.820735916509583
_cell.length_c 5.651303582127122
_cell.angle_alpha 96.23899841308594
_cell.angle_beta 96.23899841308594
_cell.angle_gamma 89.0770263671875
_space_group.name_H-M_alt 'P 1'
_space_group.IT_number 1
_symmetry.space_group_name_H-M 'P 1'
_symmetry.Int_Tables_number 1
loop_
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.fract_x
_atom_site.fract_y
_atom_site.fract_z
_atom_site.occupancy
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
'S 2' S 0.03466796875 0.0537109375 0.7734375 0.75 -0.33846945 -0.26223552 4.319827
O1 O 0.9739990234375 0.940673828125 0.26171875 1.0 3.640547 3.4354312 1.4617597
"""


def _write(tmp_path, text=MMCIF, *, compressed=False):
    path = tmp_path / ("input.cif.gz" if compressed else "input.cif")
    payload = text.encode()
    path.write_bytes(gzip.compress(payload, mtime=17) if compressed else payload)
    return path


@pytest.mark.parametrize("compressed", [False, True])
def test_normalization_preserves_fractional_sites_and_lattice(tmp_path, compressed):
    source = _write(tmp_path, compressed=compressed)
    original = source.read_bytes()
    output = normalize_material_cif(source)
    assert source.read_bytes() == original
    text = (gzip.decompress(output) if compressed else output).decode()
    block = next(iter(CifFile.from_str(text).data.values())).data
    assert block["_cell_length_a"] == "3.8433204908215908"
    assert block["_cell_angle_alpha"] == "96.23899841308594"
    assert block["_atom_site_label"] == ["S 2", "O1"]
    assert block["_atom_site_type_symbol"] == ["S", "O"]
    assert block["_atom_site_occupancy"] == ["0.75", "1.0"]
    assert block["_atom_site_fract_x"] == ["0.03466796875", "0.9739990234375"]
    assert block["_atom_site_fract_y"] == ["0.0537109375", "0.940673828125"]
    assert block["_atom_site_fract_z"] == ["0.7734375", "0.26171875"]
    structure = Structure.from_str(text, fmt="cif")
    expected_lattice = Lattice.from_parameters(
        3.8433204908215908,
        3.820735916509583,
        5.651303582127122,
        96.23899841308594,
        96.23899841308594,
        89.0770263671875,
    )
    np.testing.assert_allclose(
        structure.lattice.matrix, expected_lattice.matrix, atol=1e-12, rtol=0
    )
    by_element = {next(iter(site.species)).symbol: site for site in structure}
    assert by_element["S"].species["S"] == 0.75
    np.testing.assert_array_equal(
        by_element["S"].frac_coords, [0.03466796875, 0.0537109375, 0.7734375]
    )
    np.testing.assert_array_equal(
        by_element["O"].frac_coords, [0.9739990234375, 0.940673828125, 0.26171875]
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_standard_cif_and_unrecognized_invalid_files_unchanged(tmp_path, compressed):
    for text in (
        "data_standard\n_cell_length_a 3.0\n# comment formatting\n",
        "not a valid CIF\n",
        "data_note\n_note 'the _atom_site.fract_x field is useful'\n",
    ):
        source = _write(tmp_path, text, compressed=compressed)
        assert normalize_material_cif(source) == source.read_bytes()


def test_deterministic_gzip_and_idempotence(tmp_path):
    source = _write(tmp_path, compressed=True)
    first = normalize_material_cif(source)
    assert first == normalize_material_cif(source)
    assert first[4:8] == b"\0\0\0\0"
    destination = tmp_path / "normalized.cif.gz"
    destination.write_bytes(first)
    assert normalize_material_cif(destination) == first


def test_uppercase_compression_extension(tmp_path):
    source = tmp_path / "input.CIF.GZ"
    source.write_bytes(gzip.compress(MMCIF.encode(), mtime=17))
    normalized = normalize_material_cif(source)
    assert "_atom_site_fract_x" in gzip.decompress(normalized).decode()
    source.write_bytes(normalized)
    assert normalize_material_cif(source) == normalized


def test_conflicting_aliases_fail_without_overwriting(tmp_path):
    source = _write(tmp_path, MMCIF + "_cell_length_a 900\n")
    with pytest.raises(MaterialCifConversionError, match="conflicting values"):
        normalize_material_cif(source)
    assert source.read_text().endswith("_cell_length_a 900\n")


def test_case_insensitive_mmcif_tags(tmp_path):
    source = _write(tmp_path, MMCIF.replace("_atom_site.fract_x", "_ATOM_SITE.FRACT_X"))
    text = normalize_material_cif(source).decode()
    assert "_atom_site_fract_x" in text
    assert "_ATOM_SITE.FRACT_X" not in text


def test_empty_mmcif_is_classified_as_conversion_failure(tmp_path):
    source = _write(tmp_path, "_atom_site.fract_x 0.0\n")
    with pytest.raises(MaterialCifConversionError, match="no CIF data blocks"):
        normalize_material_cif(source)


def test_module_can_load_directly_without_third_party_imports(monkeypatch):
    import builtins

    module_path = Path(__file__).parents[1] / "src/zevals/materials/_material_cif.py"
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert not name.startswith(("pymatgen", "zevals", "biotite", "numpy"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    spec = importlib.util.spec_from_file_location("isolated_material_cif", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.normalize_material_cif)
