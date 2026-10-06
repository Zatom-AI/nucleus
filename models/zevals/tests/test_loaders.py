"""Unit tests for `pipeline/loaders.py` — the per-modality folder loaders
that back `ZevalSuite.evaluate()`. Uses real rdkit/pymatgen (both installed
in this environment) rather than faking them, since the loaders only do
simple format conversion here (no fingerprint/similarity math worth
mocking, unlike the domain metric functions tested elsewhere)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.pipeline.ligands import LigandSample  # noqa: E402
from zevals.pipeline.loaders import (  # noqa: E402
    LOADER_REGISTRY,
    load_ligands,
    load_materials,
    load_molecules,
    load_proteins,
    load_rna,
)


def test_loader_registry_covers_all_five_modalities():
    assert set(LOADER_REGISTRY) == {
        "molecules",
        "materials",
        "proteins",
        "rna",
        "ligands",
    }
    assert LOADER_REGISTRY["molecules"] is load_molecules
    assert LOADER_REGISTRY["materials"] is load_materials
    assert LOADER_REGISTRY["proteins"] is load_proteins
    assert LOADER_REGISTRY["rna"] is load_rna
    assert LOADER_REGISTRY["ligands"] is load_ligands


def test_load_proteins_globs_and_sorts_pdb_files(tmp_path: Path):
    (tmp_path / "b.pdb").touch()
    (tmp_path / "a.pdb").touch()
    (tmp_path / "not_a_pdb.txt").touch()

    result = load_proteins(tmp_path)

    assert [p.name for p in result] == ["a.pdb", "b.pdb"]


def test_load_proteins_also_globs_cif_files(tmp_path: Path):
    """CIF is the preferred format going forward — load_proteins must pick
    up .cif samples alongside (or instead of) .pdb ones."""
    (tmp_path / "a.pdb").touch()
    (tmp_path / "b.cif").touch()

    result = load_proteins(tmp_path)

    assert [p.name for p in result] == ["a.pdb", "b.cif"]


def test_load_proteins_prefers_cif_over_pdb_for_same_sample(tmp_path: Path):
    (tmp_path / "sample.pdb").touch()
    (tmp_path / "sample.cif").touch()

    result = load_proteins(tmp_path)

    assert [p.name for p in result] == ["sample.cif"]


def test_load_rna_matches_load_proteins_shape(tmp_path: Path):
    (tmp_path / "x.pdb").touch()
    assert [p.name for p in load_rna(tmp_path)] == ["x.pdb"]


def test_load_rna_also_globs_cif_files(tmp_path: Path):
    (tmp_path / "x.cif").touch()
    assert [p.name for p in load_rna(tmp_path)] == ["x.cif"]


def test_load_proteins_ignores_ground_truth_dir_argument(tmp_path: Path):
    """ground_truth_dir is accepted for uniform dispatch but is a no-op for
    proteins/rna — passing it must not raise or change the result."""
    (tmp_path / "a.pdb").touch()
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()

    assert [p.name for p in load_proteins(tmp_path, gt_dir)] == ["a.pdb"]


def test_load_molecules_reads_smi_file(tmp_path: Path):
    from rdkit import Chem

    (tmp_path / "gen.smi").write_text("CCO\nc1ccccc1\n# a comment\nCC(=O)O\n")

    mols = load_molecules(tmp_path)

    assert [Chem.MolToSmiles(m) for m in mols] == [
        Chem.CanonSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]
    ]
    # Bare SMILES carry no 3D information — no conformer to check.
    assert all(m.GetNumConformers() == 0 for m in mols)


def test_load_molecules_reads_and_converts_sdf_file(tmp_path: Path):
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(tmp_path / "gen.sdf"))
    writer.write(mol)
    writer.close()

    mols = load_molecules(tmp_path)

    assert [Chem.MolToSmiles(m) for m in mols] == [Chem.CanonSmiles("CCO")]


def test_load_molecules_preserves_3d_conformer_from_sdf(tmp_path: Path):
    """The whole point of loading Mol objects instead of flattening to SMILES
    immediately — a real 3D conformer (Zatom's native output) must survive
    the round trip so PoseBusters/UFF conformer-quality checks have
    something to check."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    writer = Chem.SDWriter(str(tmp_path / "gen.sdf"))
    writer.write(mol)
    writer.close()

    mols = load_molecules(tmp_path)

    assert len(mols) == 1
    assert mols[0].GetNumConformers() == 1


def test_load_molecules_combines_smi_and_sdf_in_one_directory(tmp_path: Path):
    from rdkit import Chem

    (tmp_path / "a.smi").write_text("CCN\n")
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(tmp_path / "b.sdf"))
    writer.write(mol)
    writer.close()

    mols = load_molecules(tmp_path)

    assert sorted(Chem.MolToSmiles(m) for m in mols) == sorted(
        Chem.CanonSmiles(s) for s in ["CCN", "CCO"]
    )


def test_load_molecules_ground_truth_dir_is_unused(tmp_path: Path):
    from rdkit import Chem

    (tmp_path / "gen.smi").write_text("CCO\n")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "ref.smi").write_text("CCN\n")

    # Only samples_dir is loaded — ground truth is a placeholder right now.
    mols = load_molecules(tmp_path, gt_dir)
    assert [Chem.MolToSmiles(m) for m in mols] == [Chem.CanonSmiles("CCO")]


def test_load_materials_reads_cif_files(tmp_path: Path):
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.to(filename=str(tmp_path / "sample.cif"))

    structures = load_materials(tmp_path)

    assert len(structures) == 1
    assert {site.species_string for site in structures[0]} == {"Na", "Cl"}


def test_load_materials_empty_directory_returns_empty_list(tmp_path: Path):
    assert load_materials(tmp_path) == []


def test_load_ligands_reads_valid_subdirectory(tmp_path: Path):
    sample_dir = tmp_path / "lig001"
    sample_dir.mkdir()
    (sample_dir / "ligand.sdf").touch()
    (sample_dir / "protein.pdb").touch()

    samples = load_ligands(tmp_path)

    assert len(samples) == 1
    assert isinstance(samples[0], LigandSample)
    assert samples[0].ligand_path == sample_dir / "ligand.sdf"
    assert samples[0].protein_path == sample_dir / "protein.pdb"
    assert samples[0].reference_ligand_path is None


def test_load_ligands_protein_pdb_is_optional(tmp_path: Path):
    sample_dir = tmp_path / "lig001"
    sample_dir.mkdir()
    (sample_dir / "ligand.sdf").touch()

    samples = load_ligands(tmp_path)

    assert samples[0].protein_path is None


def test_load_ligands_skips_subdirectory_missing_ligand_sdf(tmp_path: Path):
    good = tmp_path / "lig001"
    good.mkdir()
    (good / "ligand.sdf").touch()

    bad = tmp_path / "lig002"
    bad.mkdir()
    (bad / "protein.pdb").touch()  # no ligand.sdf

    with pytest.warns(UserWarning, match="lig002"):
        samples = load_ligands(tmp_path)

    assert [s.ligand_path.parent.name for s in samples] == ["lig001"]


def test_load_ligands_ignores_non_directory_entries(tmp_path: Path):
    sample_dir = tmp_path / "lig001"
    sample_dir.mkdir()
    (sample_dir / "ligand.sdf").touch()
    (tmp_path / "stray_file.txt").touch()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected for a stray file
        samples = load_ligands(tmp_path)

    assert len(samples) == 1


def test_load_ligands_matches_reference_by_id(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    gt_dir = tmp_path / "gt"
    sample_dir = samples_dir / "lig001"
    sample_dir.mkdir(parents=True)
    (sample_dir / "ligand.sdf").touch()

    ref_dir = gt_dir / "lig001"
    ref_dir.mkdir(parents=True)
    (ref_dir / "reference_ligand.sdf").touch()

    samples = load_ligands(samples_dir, gt_dir)

    assert samples[0].reference_ligand_path == ref_dir / "reference_ligand.sdf"


def test_load_ligands_missing_reference_leaves_field_none(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    gt_dir = tmp_path / "gt"
    sample_dir = samples_dir / "lig001"
    sample_dir.mkdir(parents=True)
    (sample_dir / "ligand.sdf").touch()
    gt_dir.mkdir()  # exists, but no lig001/ subdirectory inside it

    samples = load_ligands(samples_dir, gt_dir)

    assert samples[0].reference_ligand_path is None
