from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from biotite.structure import AtomArray
from pymatgen.core import Lattice, Structure

from nucleus.utils.materials import annotate_material_properties


def _install_fake_matminer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal matminer stub so the helper module can be imported."""
    matminer = ModuleType("matminer")
    matminer.__path__ = []  # type: ignore[attr-defined]

    featurizers = ModuleType("matminer.featurizers")
    featurizers.__path__ = []  # type: ignore[attr-defined]

    site = ModuleType("matminer.featurizers.site")
    site.__path__ = []  # type: ignore[attr-defined]

    fingerprint = ModuleType("matminer.featurizers.site.fingerprint")

    class FakeCrystalNNFingerprint:
        @classmethod
        def from_preset(cls, preset: str):
            return cls()

        def featurize(self, structure, site_index: int):
            return np.array([float(site_index + 1), float(len(structure))])

    fingerprint.CrystalNNFingerprint = FakeCrystalNNFingerprint
    site.fingerprint = fingerprint
    featurizers.site = site
    matminer.featurizers = featurizers

    monkeypatch.setitem(sys.modules, "matminer", matminer)
    monkeypatch.setitem(sys.modules, "matminer.featurizers", featurizers)
    monkeypatch.setitem(sys.modules, "matminer.featurizers.site", site)
    monkeypatch.setitem(
        sys.modules, "matminer.featurizers.site.fingerprint", fingerprint
    )


def test_atom_array_to_structure_converts_material_atoms(
    monkeypatch: pytest.MonkeyPatch,
):
    """AtomArray -> Structure conversion should use the material lattice helpers."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    atom_array = AtomArray(2)
    atom_array.coord = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=float)
    atom_array.element = np.asarray(["C", "O"], dtype=object)
    atom_array.set_annotation("is_material", np.ones(2, dtype=bool))
    atom_array.set_annotation("material_lattice_lengths", np.ones((2, 3)))
    atom_array.set_annotation("material_lattice_angles", np.full((2, 3), 90.0))
    atom_array.set_annotation(
        "material_lattice_vectors", np.repeat(np.eye(3)[None], 2, axis=0)
    )

    structure = mf.atom_array_to_structure(atom_array)

    assert structure is not None
    assert isinstance(structure, Structure)
    assert structure.composition.reduced_formula == "CO"
    np.testing.assert_allclose(
        structure.cart_coords, np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    )


def test_atom_array_to_structure_prefers_fractional_coordinates(
    monkeypatch: pytest.MonkeyPatch,
):
    """Zero or differently oriented Cartesian fields must not define a crystal."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    lattice = Lattice.from_parameters(4.1, 5.2, 6.3, 77.0, 83.0, 68.0).matrix
    fractional = np.asarray([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]], dtype=float)
    atom_array = AtomArray(2)
    atom_array.coord = np.zeros((2, 3), dtype=float)
    atom_array.element = np.asarray(["Si", "O"], dtype=object)
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional,
        lattice_lengths=np.linalg.norm(lattice, axis=1),
        lattice_angles=np.asarray([77.0, 83.0, 68.0]),
        lattice_vectors=lattice,
    )

    structure = mf.atom_array_to_structure(atom_array)

    assert structure is not None
    np.testing.assert_allclose(
        structure.frac_coords,
        fractional,
        atol=1e-12,
    )
    assert not np.allclose(structure.cart_coords, 0.0)


def test_structure_fingerprint_normalizes_mean_site_vector(
    monkeypatch: pytest.MonkeyPatch,
):
    """The structure fingerprint helper should average site fingerprints and normalize them."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    structure = Structure(
        Lattice.cubic(1.0),
        species=["C", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )

    class FakeFeaturizer:
        def featurize(self, _structure, site_index: int):
            return np.array([1.0 + site_index, 2.0], dtype=float)

    fingerprint = mf.structure_fingerprint(structure, FakeFeaturizer())

    assert fingerprint is not None
    np.testing.assert_allclose(np.linalg.norm(fingerprint), 1.0)
    np.testing.assert_allclose(fingerprint, np.array([0.6, 0.8]))


def test_compute_structure_fingerprints_skips_invalid_structures(
    monkeypatch: pytest.MonkeyPatch,
):
    """Invalid structures should be skipped, and valid ones should be fingerprinted."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    structures = [
        Structure(
            Lattice.cubic(1.0),
            species=["C"],
            coords=[[0, 0, 0]],
            coords_are_cartesian=False,
        ),
        Structure(
            Lattice.cubic(1.0),
            species=["O"],
            coords=[[0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        ),
    ]

    fingerprint = mf.compute_structure_fingerprints(structures)

    assert len(fingerprint) == 2
    np.testing.assert_allclose(np.linalg.norm(fingerprint[0]), 1.0)
    np.testing.assert_allclose(np.linalg.norm(fingerprint[1]), 1.0)


def test_record_to_structure_uses_dataset_lattice_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    """Dataset records should convert directly from their stored lattice fields."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    record = {
        "atom_array": type(
            "FakeAtomArray",
            (),
            {"element": np.asarray(["C", "O"], dtype=object)},
        )(),
        "fractional_coordinates": np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float
        ),
        "lattice_vectors": np.eye(3),
    }

    structure = mf.record_to_structure(record)

    assert structure is not None
    assert isinstance(structure, Structure)
    assert structure.composition.reduced_formula == "CO"
    np.testing.assert_allclose(
        structure.frac_coords,
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    )


def test_record_to_structure_prefers_fractional_over_inconsistent_cartesian(
    monkeypatch: pytest.MonkeyPatch,
):
    """Dataset Cartesian coordinates may use a different lattice orientation."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    fractional = np.asarray([[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]])
    record = {
        "chemical_symbols": ["Na", "Cl"],
        "fractional_coordinates": fractional,
        "cartesian_coordinates": np.zeros((2, 3)),
        "lattice_vectors": Lattice.from_parameters(
            4.0, 5.0, 6.0, 75.0, 80.0, 70.0
        ).matrix,
    }

    structure = mf.record_to_structure(record)

    assert structure is not None
    np.testing.assert_allclose(structure.frac_coords, fractional, atol=1e-12)
    assert not np.allclose(structure.cart_coords, 0.0)


def test_load_reference_fingerprints_from_path_handles_npy_and_npz(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The cache loader should accept both .npy and .npz files."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    npy_path = tmp_path / "reference_fingerprints.npy"
    np.save(npy_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    npz_path = tmp_path / "reference_fingerprints.npz"
    np.savez(
        npz_path,
        reference_fingerprints=np.asarray([[0.5, 0.5]], dtype=np.float32),
    )

    npy_loaded = mf.load_reference_fingerprints_from_path(npy_path)
    npz_loaded = mf.load_reference_fingerprints_from_path(npz_path)

    assert len(npy_loaded or []) == 2
    assert len(npz_loaded or []) == 1
    np.testing.assert_allclose(npy_loaded[0], np.array([1.0, 0.0]))
    np.testing.assert_allclose(npy_loaded[1], np.array([0.0, 1.0]))
    np.testing.assert_allclose(npz_loaded[0], np.array([0.5, 0.5]))


def test_load_reference_fingerprints_from_path_rejects_unknown_suffix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The cache loader should reject unsupported file extensions."""
    _install_fake_matminer(monkeypatch)
    sys.modules.pop("nucleus.utils.material_fingerprints", None)
    from nucleus.utils import material_fingerprints as mf

    bad_path = tmp_path / "reference_fingerprints.txt"
    bad_path.write_text("not a cache", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported fingerprint cache format"):
        mf.load_reference_fingerprints_from_path(bad_path)
