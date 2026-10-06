"""Unit tests for `utils/io.py`'s biomolecule CIF support added this session:
`load_structure_mmcif`, `convert_cif_to_pdb` (the CIF->PDB fallback for tools
that can't read mmCIF directly), and `glob_structures` (the dual-`.cif`/
`.pdb` directory listing used by `pipeline/loaders.py`).

Structures are written via Biopython's own `StructureBuilder` + `PDBIO`/
`MMCIFIO`, not hand-rolled text, so the fixed-width/mmCIF-loop formatting is
guaranteed correct — same convention as `test_proteins_metrics.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.utils.io import (  # noqa: E402
    convert_cif_to_pdb,
    glob_structures,
    load_structure_mmcif,
    load_structure_pdb,
)


def _write_ca_structure(path: Path, coords, chain_id: str = "A") -> None:
    """Write a minimal valid CA-only structure, PDB or CIF by `path.suffix`,
    via Biopython's own writers (no hand-rolled text)."""
    from Bio.PDB import StructureBuilder
    from Bio.PDB.mmcifio import MMCIFIO
    from Bio.PDB.PDBIO import PDBIO

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("s")
    sb.init_model(0)
    sb.init_chain(chain_id)
    sb.init_seg(" ")
    for i, (x, y, z) in enumerate(coords):
        sb.init_residue("ALA", " ", i + 1, " ")
        sb.init_atom("CA", (x, y, z), 0.0, 1.0, " ", "CA", i, element="C")
    structure = sb.get_structure()

    io = MMCIFIO() if path.suffix == ".cif" else PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def test_load_structure_mmcif_reads_ca_coords(tmp_path: Path):
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
    cif_path = tmp_path / "sample.cif"
    _write_ca_structure(cif_path, coords)

    structure = load_structure_mmcif(cif_path)
    residues = [r for r in structure.get_residues() if "CA" in r]

    assert len(residues) == 3
    got = np.array([r["CA"].get_coord() for r in residues])
    assert np.allclose(got, coords, atol=1e-2)


def test_convert_cif_to_pdb_round_trips_coordinates(tmp_path: Path):
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
    cif_path = tmp_path / "sample.cif"
    _write_ca_structure(cif_path, coords)

    out_path = convert_cif_to_pdb(cif_path, tmp_path / "converted.pdb")

    assert out_path.is_file()
    structure = load_structure_pdb(out_path)
    residues = [r for r in structure.get_residues() if "CA" in r]
    got = np.array([r["CA"].get_coord() for r in residues])
    assert np.allclose(got, coords, atol=1e-2)


def test_convert_cif_to_pdb_raises_loudly_on_long_chain_id(tmp_path: Path):
    """A chain ID longer than PDB's 1-character field can't be represented
    losslessly — must fail, not silently truncate."""
    from Bio.PDB.PDBExceptions import PDBIOException

    coords = [(0.0, 0.0, 0.0)]
    cif_path = tmp_path / "sample.cif"
    _write_ca_structure(cif_path, coords, chain_id="LONGCHAIN")

    with pytest.raises(PDBIOException):
        convert_cif_to_pdb(cif_path, tmp_path / "converted.pdb")


def test_convert_cif_to_pdb_warns_on_multiple_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Exercises the multi-model warning in isolation from Biopython's own
    mmCIF *writer* (constructing a real multi-model mmCIF fixture that
    MMCIFIO round-trips cleanly is a separate concern from what this check
    guards against — a real deposited/ensemble CIF having >1 model)."""
    import zevals.utils.io as io_module

    class _FakeModel:
        pass

    class _FakeStructure:
        def __iter__(self):
            return iter([_FakeModel(), _FakeModel()])

    cif_path = tmp_path / "multimodel.cif"
    cif_path.touch()
    monkeypatch.setattr(io_module, "load_structure_mmcif", lambda p: _FakeStructure())
    monkeypatch.setattr(io_module, "save_structure_pdb", lambda structure, path: None)

    with caplog.at_level("WARNING"):
        io_module.convert_cif_to_pdb(cif_path, tmp_path / "converted.pdb")

    assert any("models" in rec.message for rec in caplog.records)


def test_glob_structures_returns_both_extensions(tmp_path: Path):
    (tmp_path / "a.pdb").touch()
    (tmp_path / "b.cif").touch()
    (tmp_path / "not_a_structure.txt").touch()

    result = glob_structures(tmp_path)

    assert [p.name for p in result] == ["a.pdb", "b.cif"]


def test_glob_structures_prefers_cif_on_same_stem(tmp_path: Path):
    (tmp_path / "sample.pdb").touch()
    (tmp_path / "sample.cif").touch()

    result = glob_structures(tmp_path)

    assert len(result) == 1
    assert result[0].suffix == ".cif"


def test_glob_structures_empty_directory_returns_empty_list(tmp_path: Path):
    assert glob_structures(tmp_path) == []
