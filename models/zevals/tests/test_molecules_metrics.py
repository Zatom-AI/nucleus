from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeMol:
    """A simple fake molecule class that mimics the behavior of RDKit's Mol for testing purposes."""

    def __init__(self, label: str):
        self.label = label

    def GetNumAtoms(self):
        return max(len(self.label.replace(".", "")), 1)

    def GetNumConformers(self):
        return 1

    def __str__(self):
        return self.label


def _install_fake_rdkit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installs a fake rdkit module that mimics the necessary functionality for testing molecule metrics."""
    rdkit = ModuleType("rdkit")
    chem = ModuleType("rdkit.Chem")
    datastructs = ModuleType("rdkit.DataStructs")
    allchem = ModuleType("rdkit.Chem.AllChem")
    rdmolops = ModuleType("rdkit.Chem.rdmolops")

    def mol_from_smiles(smiles: str):
        """Mimics rdkit.Chem.MolFromSmiles behavior for testing."""
        if smiles == "bad":
            return None
        return FakeMol(smiles)

    def mol_to_smiles(mol, *args, **kwargs):
        """Mimics rdkit.Chem.MolToSmiles behavior for testing."""
        return mol.label.lower()

    def get_fp(mol, radius: int, nBits: int):
        """Mimics rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect behavior for testing."""
        return frozenset(mol.label.lower())

    def get_frags(mol, asMols: bool = True):
        """Mimics rdkit.Chem.rdmolops.GetMolFrags behavior for testing."""
        fragments = [FakeMol(fragment) for fragment in mol.label.split(".") if fragment]
        if not fragments:
            fragments = [mol]
        if asMols:
            return fragments
        return tuple(tuple(range(fragment.GetNumAtoms())) for fragment in fragments)

    def bulk_tanimoto_similarity(fp, fps):
        """Mimics rdkit.DataStructs.BulkTanimotoSimilarity behavior for testing."""
        fp_set = set(fp)
        sims = []
        for other in fps:
            other_set = set(other)
            union = fp_set | other_set
            sims.append(len(fp_set & other_set) / len(union) if union else 1.0)
        return sims

    chem.MolFromSmiles = mol_from_smiles
    chem.MolToSmiles = mol_to_smiles
    chem.AllChem = allchem
    chem.rdmolops = rdmolops
    rdmolops.GetMolFrags = get_frags
    datastructs.BulkTanimotoSimilarity = bulk_tanimoto_similarity
    allchem.GetMorganFingerprintAsBitVect = get_fp

    rdkit.Chem = chem
    rdkit.DataStructs = datastructs

    monkeypatch.setitem(sys.modules, "rdkit", rdkit)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem)
    monkeypatch.setitem(sys.modules, "rdkit.DataStructs", datastructs)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.AllChem", allchem)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.rdmolops", rdmolops)


def _install_fake_posebusters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installs a fake posebusters module that mimics the necessary functionality for testing posebusters metrics."""
    posebusters = ModuleType("posebusters")

    class FakePoseBusters:
        """A fake PoseBusters class that mimics the behavior of the real one for testing purposes."""

        def __init__(self, config: str):
            self.config = config

        def bust(self, molecules, *_args):
            """Mimics PoseBusters.bust behavior for testing."""
            count = len(list(molecules))
            internal_steric_clash = [True] * count
            if count:
                internal_steric_clash[-1] = False
            return pd.DataFrame(
                {
                    "mol_pred_loaded": [True] * count,
                    "sanitization": [True] * count,
                    "inchi_convertible": [True] * count,
                    "all_atoms_connected": [True] * count,
                    "no_radicals": [True] * count,
                    "bond_lengths": [True] * count,
                    "bond_angles": [True] * count,
                    "internal_steric_clash": internal_steric_clash,
                    "aromatic_ring_flatness": [True] * count,
                    "non-aromatic_ring_non-flatness": [True] * count,
                    "double_bond_flatness": [True] * count,
                    "internal_energy": [True] * count,
                }
            )

    posebusters.PoseBusters = FakePoseBusters
    monkeypatch.setitem(__import__("sys").modules, "posebusters", posebusters)


def test_molecule_metrics_match_expected_definitions(monkeypatch: pytest.MonkeyPatch):
    """Tests that the molecule metrics compute the expected values based on the provided samples and reference."""
    _install_fake_rdkit(monkeypatch)

    from zevals.molecules.diversity import compute_diversity
    from zevals.molecules.novelty import compute_novelty
    from zevals.molecules.uniqueness import compute_uniqueness
    from zevals.molecules.validity import compute_validity

    reference = ["b"]

    mol_samples = [FakeMol("A"), FakeMol("a"), FakeMol("B"), None]
    validity = compute_validity(mol_samples)
    assert validity["num_valid"] == 3
    assert validity["valid_rate"] == pytest.approx(0.75)
    assert validity["valid_smiles"] == ["a", "a", "b"]

    valid_smiles = validity["valid_smiles"]
    assert compute_uniqueness(valid_smiles) == {
        "num_unique": 2,
        "unique_rate": pytest.approx(2 / 3),
    }
    assert compute_novelty(valid_smiles, reference) == {
        "num_novel": 1,
        "novel_rate": pytest.approx(1 / 3),
    }
    assert compute_diversity(valid_smiles) == {
        "diversity_rate": pytest.approx(1.0),
    }


def test_molecule_novelty_can_use_fpsim2_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that molecule novelty can query a precomputed FPSim2 reference database."""
    from zevals.molecules import novelty

    class FakeFPSim2Engine:
        def __init__(self, ref_fp: str):
            self.ref_fp = ref_fp

        def top_k(self, smi, *, threshold, k, metric, full_sanitization, n_workers):
            assert self.ref_fp == "reference.h5"
            assert threshold == 0.0
            assert k == 1
            assert metric == "tanimoto"
            assert full_sanitization is False
            assert n_workers == 1
            return [(0, {"a": 0.25, "b": 0.75}[smi])]

    monkeypatch.setattr(novelty, "load_fpsim2_engine", lambda: FakeFPSim2Engine)

    assert novelty.compute_novelty(
        ["a", "b"], reference_fp="reference.h5", n_threads=1
    ) == {
        "novel_rate": pytest.approx(0.5),
        "mean_nearest_similarity": pytest.approx(0.5),
    }


def test_posebusters_breakdown_is_exposed(monkeypatch: pytest.MonkeyPatch):
    """Tests that the PoseBusters metrics compute the expected values based on the provided samples."""
    _install_fake_posebusters(monkeypatch)

    from zevals.molecules.posebusters import (
        POSEBUSTERS_KEYS,
        compute_posebusters_metrics,
    )

    metrics = compute_posebusters_metrics(["mol-1", "mol-2"])

    assert list(metrics.keys()) == list(POSEBUSTERS_KEYS)
    assert metrics["posebusters_rate"] == pytest.approx(0.5)
    assert metrics["internal_steric_clash"] == pytest.approx(0.5)


def _install_fake_uff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installs a fake rdkit.Chem.rdForceFieldHelpers for UFF force-norm tests."""

    class FakeConformer:
        def GetId(self):
            return 0

    class FakeForceField:
        def __init__(self, grad):
            self._grad = grad

        def CalcGrad(self):
            return self._grad

    def uff_has_all_params(mol):
        return getattr(mol, "has_uff_params", True)

    def uff_get_force_field(mol, confId=0):
        return FakeForceField(mol.gradient)

    rdff = ModuleType("rdkit.Chem.rdForceFieldHelpers")
    rdff.UFFHasAllMoleculeParams = uff_has_all_params
    rdff.UFFGetMoleculeForceField = uff_get_force_field
    monkeypatch.setitem(sys.modules, "rdkit.Chem.rdForceFieldHelpers", rdff)
    return FakeConformer


def test_compute_uff_force_norm_metrics_scores_valid_conformers(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests the UFF median force-norm metric against fake conformers with
    known gradients, and confirms conformer-less/param-less mols are
    excluded from the median but still counted in the success-rate denominator."""
    FakeConformer = _install_fake_uff(monkeypatch)

    from zevals.molecules.conformer_quality import compute_uff_force_norm_metrics

    good_mol = FakeMol("good")
    good_mol.gradient = [[3.0, 4.0, 0.0]]  # norm = 5.0, one atom
    good_mol.GetConformer = lambda: FakeConformer()

    no_conformer_mol = FakeMol("noconf")
    no_conformer_mol.GetNumConformers = lambda: 0

    no_params_mol = FakeMol("noparams")
    no_params_mol.has_uff_params = False
    no_params_mol.GetConformer = lambda: FakeConformer()

    metrics = compute_uff_force_norm_metrics(
        [good_mol, no_conformer_mol, no_params_mol], total_samples=4
    )

    assert metrics["uff_median_force_norm"] == pytest.approx(5.0)
    assert metrics["uff_median_force_norm_success_rate"] == pytest.approx(1 / 4)
    assert "uff_mean_force_norm_mae_eV_per_angstrom" not in metrics


def test_compute_uff_force_norm_metrics_compares_each_conformer_to_target(
    monkeypatch: pytest.MonkeyPatch,
):
    """UFF target MAE is reduced over per-conformer mean force norms."""
    FakeConformer = _install_fake_uff(monkeypatch)

    from zevals.molecules.conformer_quality import compute_uff_force_norm_metrics

    one = FakeMol("one")
    one.gradient = [[1.0, 0.0, 0.0]]
    one.GetConformer = lambda: FakeConformer()

    nine = FakeMol("nine")
    nine.gradient = [[0.0, 0.0, 9.0]]
    nine.GetConformer = lambda: FakeConformer()

    metrics = compute_uff_force_norm_metrics([one, nine], target_mean_force_norm=0.2)

    # UFF's [1, 9] kcal/mol/A are converted to eV/A before the target error.
    conversion = 0.0433641153087705
    expected = (abs(conversion - 0.2) + abs(9 * conversion - 0.2)) / 2
    assert metrics["uff_mean_force_norm_mae_eV_per_angstrom"] == pytest.approx(expected)


def test_compute_uff_force_norm_metrics_keeps_target_key_when_nothing_scores():
    """A requested target keeps a stable zero MAE when no conformer scores."""
    from zevals.molecules.conformer_quality import compute_uff_force_norm_metrics

    metrics = compute_uff_force_norm_metrics([], target_mean_force_norm=2.0)

    assert metrics["uff_mean_force_norm_mae_eV_per_angstrom"] == 0.0


def test_pipeline_wires_posebusters_and_uff_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that MoleculeEvalPipeline actually calls PoseBusters/UFF (not
    just validity/uniqueness/diversity/novelty) when both flags default on."""
    _install_fake_rdkit(monkeypatch)
    _install_fake_posebusters(monkeypatch)
    FakeConformer = _install_fake_uff(monkeypatch)

    from zevals.pipeline.molecules import MoleculeEvalPipeline

    mols = [FakeMol("A"), FakeMol("B")]
    for m in mols:
        m.gradient = [[1.0, 0.0, 0.0]]
        m.GetConformer = lambda: FakeConformer()

    pipeline = MoleculeEvalPipeline()
    results = pipeline.run(mols)

    assert "posebusters_rate" in results.aggregate
    assert "internal_steric_clash" in results.aggregate
    assert results.aggregate["uff_median_force_norm"] == pytest.approx(1.0)
    assert results.aggregate["uff_median_force_norm_success_rate"] == pytest.approx(1.0)


def test_pipeline_skips_posebusters_and_uff_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that disabling compute_posebusters/compute_uff_force_norm falls
    back to zero-filled keys rather than calling either check."""
    _install_fake_rdkit(monkeypatch)

    from zevals.molecules.posebusters import POSEBUSTERS_KEYS
    from zevals.pipeline.molecules import MoleculeEvalPipeline

    pipeline = MoleculeEvalPipeline(
        compute_posebusters=False, compute_uff_force_norm=False
    )
    results = pipeline.run([FakeMol("A")])

    assert all(results.aggregate[key] == 0.0 for key in POSEBUSTERS_KEYS)
    assert results.aggregate["uff_median_force_norm"] == 0.0
    assert results.aggregate["uff_median_force_norm_success_rate"] == 0.0


def test_pipeline_includes_uniqueness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Tests that the MoleculeEvalPipeline includes uniqueness metrics."""
    _install_fake_rdkit(monkeypatch)

    from rdkit import Chem

    from zevals.pipeline.molecules import MoleculeEvalPipeline

    reference = tmp_path / "reference.smi"
    reference.write_text("b\n")

    # run() takes already-parsed Mol objects (what loaders.load_molecules
    # produces), matching real usage — not raw SMILES strings.
    mols = [Chem.MolFromSmiles(s) for s in ["A", "a", "B", "bad"]]
    pipeline = MoleculeEvalPipeline(
        reference_smiles_path=reference,
        compute_posebusters=False,
        compute_uff_force_norm=False,
    )
    results = pipeline.run(mols)

    assert results.aggregate["num_valid"] == 3
    assert results.aggregate["valid_rate"] == pytest.approx(0.75)
    assert results.aggregate["unique_rate"] == pytest.approx(2 / 3)
    # Zatom-2/nucleus convention: novel_rate normalizes by unique count (2),
    # not valid count (3) — "b" is the only novel one among {"a", "b"}.
    assert results.aggregate["novel_rate"] == pytest.approx(1 / 2)


def test_pipeline_can_use_fpsim2_reference_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Tests that MoleculeEvalPipeline forwards the FPSim2 reference path."""
    _install_fake_rdkit(monkeypatch)

    from rdkit import Chem

    from zevals.pipeline import molecules as molecule_pipeline

    captured = {}

    def fake_novelty(smiles, reference=None, *, reference_fp=None, n_threads=1):
        captured["smiles"] = list(smiles)
        captured["reference"] = reference
        captured["reference_fp"] = reference_fp
        captured["n_threads"] = n_threads
        return {"novel_rate": 0.42}

    monkeypatch.setattr(molecule_pipeline, "compute_novelty", fake_novelty)

    reference_fp = tmp_path / "reference.h5"
    pipeline = molecule_pipeline.MoleculeEvalPipeline(
        reference_fingerprint_path=reference_fp,
        novelty_n_threads=3,
        compute_posebusters=False,
        compute_uff_force_norm=False,
    )
    mols = [Chem.MolFromSmiles(s) for s in ["A", "a", "B", "bad"]]
    results = pipeline.run(mols)

    assert results.aggregate["novel_rate"] == pytest.approx(0.42)
    assert captured == {
        "smiles": ["a", "a", "b"],
        "reference": None,
        "reference_fp": reference_fp,
        "n_threads": 3,
    }
