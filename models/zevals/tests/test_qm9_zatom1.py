from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib import metadata

import numpy as np
import pandas as pd
import pytest

from zevals.molecules import qm9_zatom1 as scorer


def _methane() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([6, 1, 1, 1, 1]),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.629437, 0.629437, 0.629437],
                [-0.629437, -0.629437, 0.629437],
                [-0.629437, 0.629437, -0.629437],
                [0.629437, -0.629437, -0.629437],
            ]
        ),
    )


@pytest.fixture
def backend():
    for package, expected in scorer.QM9_ZATOM1_PACKAGE_VERSIONS.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            pytest.skip(
                "Requires Table 2 chemistry environment: uv run --no-sync "
                "--with-requirements models/zatom2/docs/table2_qm9.requirements.txt "
                "python -m pytest"
            )
    return scorer._load_backend()


def _record_posebusters(monkeypatch, backend, frame=None):
    Chem, _, _, PoseBusters, _ = backend
    recorded = {}
    config = PoseBusters(config="mol").config

    class RecordingBuster:
        def __init__(self, *, config):
            assert config == "mol"
            self.config = deepcopy(recorded["config"])

        def bust(self, molecules, mol_true, mol_cond):
            assert mol_true is None and mol_cond is None
            recorded["molecules"] = molecules
            if frame is not None:
                return frame
            rows = []
            for molecule in molecules:
                row = dict.fromkeys(scorer.QM9_ZATOM1_POSEBUSTERS_KEYS, True)
                row["all_atoms_connected"] = (
                    len(Chem.rdmolops.GetMolFrags(molecule)) == 1
                )
                rows.append(row)
            return pd.DataFrame(rows)

    recorded["config"] = config
    monkeypatch.setattr(
        scorer,
        "_load_backend",
        lambda: (*backend[:3], RecordingBuster, backend[4]),
    )
    return recorded


def test_conversion_matches_upstream_pdb_file_and_retains_geometry(backend, tmp_path):
    Chem, Molecule, BabelMolAdaptor, _, _ = backend
    sample = _methane()
    positions_before = sample[1].copy()
    pdb_path = tmp_path / "upstream.pdb"
    Molecule(sample[0], sample[1]).to(str(pdb_path), fmt="pdb")
    upstream = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    converted, smiles = scorer._convert_sample(sample, Chem, Molecule, BabelMolAdaptor)

    assert smiles == "[H]C([H])([H])[H]"
    assert converted.GetNumAtoms() == 5
    assert Chem.MolToSmiles(converted) == Chem.MolToSmiles(upstream)
    np.testing.assert_array_equal(
        converted.GetConformer().GetPositions(),
        upstream.GetConformer().GetPositions(),
    )
    # PDB's three-decimal output is part of the original conversion protocol.
    np.testing.assert_array_equal(
        converted.GetConformer().GetPositions(), np.round(sample[1], 3)
    )
    np.testing.assert_array_equal(sample[1], positions_before)


def test_denominators_and_full_disconnected_molecules(backend, monkeypatch):
    recorded = _record_posebusters(monkeypatch, backend)
    numbers, positions = _methane()
    disconnected = (
        np.concatenate([numbers, numbers]),
        np.concatenate([positions, positions + 10.0]),
    )
    water = (
        np.array([8, 1, 1]),
        np.array([[0, 0, 0], [0.958, 0, 0], [-0.240, 0.927, 0]]),
    )
    samples = [_methane(), _methane(), water, disconnected, None]
    result = scorer.compute_qm9_zatom1_generation_metrics(samples)

    assert result["counts"]["generated"] == 5
    assert result["counts"]["valid"] == 4
    assert result["counts"]["unique"] == 2
    assert result["counts"]["posebusters_evaluated"] == 4
    assert result["denominators"]["valid"] == 5
    assert result["denominators"]["unique"] == 4
    assert result["rates"]["valid"] == 4 / 5
    assert result["rates"]["unique"] == 1 / 2
    assert result["rates"]["all_atoms_connected"] == 3 / 4
    assert result["percent"]["all_atoms_connected"] == 75.0
    assert [mol.GetNumAtoms() for mol in recorded["molecules"]] == [5, 5, 3, 10]
    assert set(result["rates"]) == {
        "valid",
        "unique",
        *scorer.QM9_ZATOM1_POSEBUSTERS_KEYS,
    }


@pytest.mark.parametrize("samples", [[], [None], [([], [])]])
def test_empty_or_invalid_samples_keep_attempted_count(backend, monkeypatch, samples):
    recorded = _record_posebusters(monkeypatch, backend)
    result = scorer.compute_qm9_zatom1_generation_metrics(samples)
    assert result["counts"]["generated"] == len(samples)
    assert result["counts"]["valid"] == 0
    assert all(value == 0 for value in result["rates"].values())
    assert "unique" in result["zero_denominator_metrics"]
    assert ("valid" in result["zero_denominator_metrics"]) == (not samples)
    assert "molecules" not in recorded


@pytest.mark.parametrize(
    "sample",
    [([0], [[0, 0, 0]]), ([119], [[0, 0, 0]]), ([6], [[np.nan, 0, 0]])],
)
def test_invalid_atomic_numbers_and_coordinates(backend, sample):
    assert scorer._convert_sample(sample, *backend[:3]) is None


def test_native_missing_checks_match_upstream_means(backend, monkeypatch):
    frame = pd.DataFrame(
        [{key: True for key in scorer.QM9_ZATOM1_POSEBUSTERS_KEYS}] * 2
    )
    frame["internal_energy"] = [True, np.nan]
    frame["bond_angles"] = [np.nan, np.nan]
    _record_posebusters(monkeypatch, backend, frame)
    result = scorer.compute_qm9_zatom1_generation_metrics([_methane(), _methane()])

    assert result["rates"]["internal_energy"] == frame["internal_energy"].mean()
    assert result["denominators"]["internal_energy"] == 1
    assert result["counts"]["posebusters_missing_checks"]["internal_energy"] == 1
    assert result["rates"]["bond_angles"] == 0
    assert result["denominators"]["bond_angles"] == 0
    assert result["zero_denominator_metrics"] == ["bond_angles"]


@pytest.mark.parametrize("failure", ["missing_column", "missing_row", "nonbinary"])
def test_posebusters_schema_errors_fail_explicitly(backend, monkeypatch, failure):
    frame = pd.DataFrame(
        [{key: True for key in scorer.QM9_ZATOM1_POSEBUSTERS_KEYS}] * 2
    )
    if failure == "missing_column":
        frame = frame.drop(columns="bond_lengths")
    elif failure == "missing_row":
        frame = frame.iloc[:1]
    else:
        frame["bond_lengths"] = [0.25, 1]
    _record_posebusters(monkeypatch, backend, frame)
    with pytest.raises(RuntimeError, match="PoseBusters Table 2"):
        scorer.compute_qm9_zatom1_generation_metrics([_methane(), _methane()])


def test_effective_config_and_versions_are_recorded(backend, monkeypatch):
    recorded = _record_posebusters(monkeypatch, backend)
    result = scorer.compute_qm9_zatom1_generation_metrics([])
    protocol = result["protocol"]
    pb = protocol["posebusters"]
    assert protocol["package_versions"] == scorer.QM9_ZATOM1_PACKAGE_VERSIONS
    assert not protocol["pdb_remove_hydrogens"]
    assert not protocol["predicted_geometry_optimized"]
    assert pb["config_name"] == "mol"
    assert pb["reference_ensemble_random_seed"] == 42
    assert pb["config"] == recorded["config"]
    digest = hashlib.sha256(
        json.dumps(pb["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert pb["config_sha256"] == digest
    assert result["digests"]["posebusters_config_sha256"] == digest
    energy_modules = [
        module
        for module in pb["config"]["modules"]
        if module["function"] == "energy_ratio"
    ]
    assert len(energy_modules) == 1
    assert energy_modules[0]["parameters"] == {
        "threshold_energy_ratio": 100.0,
        "ensemble_number_conformations": 50,
        "inchi_strict": False,
    }


def test_real_posebusters_matches_upstream_file_based_pipeline(backend, tmp_path):
    Chem, Molecule, _, PoseBusters, _ = backend
    numbers, positions = _methane()
    pdb_path = tmp_path / "upstream.pdb"
    Molecule(numbers, positions).to(str(pdb_path), fmt="pdb")
    upstream = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    expected = PoseBusters(config="mol").bust([upstream], None, None).mean()

    result = scorer.compute_qm9_zatom1_generation_metrics([_methane()])
    for key in scorer.QM9_ZATOM1_POSEBUSTERS_KEYS:
        assert result["rates"][key] == float(expected[key])
    assert result["rates"]["valid"] == result["rates"]["unique"] == 1


@pytest.mark.parametrize("package", list(scorer.QM9_ZATOM1_PACKAGE_VERSIONS))
def test_wrong_chemistry_versions_fail_before_evaluating(monkeypatch, package):
    def fake_version(name):
        if name == package:
            return "0.0.0"
        return scorer.QM9_ZATOM1_PACKAGE_VERSIONS[name]

    monkeypatch.setattr(scorer.metadata, "version", fake_version)
    with pytest.raises(RuntimeError, match=f"requires {package}"):
        scorer.compute_qm9_zatom1_generation_metrics([])


def test_missing_chemistry_dependency_fails_before_evaluating(monkeypatch):
    def missing_version(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(scorer.metadata, "version", missing_version)
    with pytest.raises(RuntimeError, match="found missing"):
        scorer.compute_qm9_zatom1_generation_metrics([None])
