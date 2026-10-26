from __future__ import annotations

import pickle

import lmdb
import numpy as np
import pytest
from biotite.structure import AtomArray, BondList

from nucleus.utils.atom_array import openbabel
from nucleus.utils.bond_sidecar import (
    ATOMIC_NUMBERS_HASH_ALGORITHM,
    BOND_SIDECAR_METADATA_KEY,
    BondSidecarASEAtomsLoader,
    decode_bond_record,
    encode_bond_record,
    hash_atomic_numbers,
    openbabel_bond_record,
    read_bond_sidecar_metadata,
    write_fragment_status_index_cache,
)


def _water_atom_array() -> AtomArray:
    atom_array = AtomArray(3)
    atom_array.coord = np.asarray(
        [
            [0.000000, 0.000000, 0.000000],
            [0.957200, 0.000000, 0.000000],
            [-0.239987, 0.927297, 0.000000],
        ]
    )
    atom_array.element = np.asarray(["O", "H", "H"])
    return atom_array


def _ammonium_atom_array() -> AtomArray:
    atom_array = AtomArray(5)
    atom_array.coord = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.589, 0.589, 0.589],
            [0.589, -0.589, -0.589],
            [-0.589, 0.589, -0.589],
            [-0.589, -0.589, 0.589],
        ]
    )
    atom_array.element = np.asarray(["N", "H", "H", "H", "H"])
    return atom_array


def _ligand_atom_array() -> AtomArray:
    atom_array = AtomArray(3)
    atom_array.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-0.5, 0.8, 0.0],
        ]
    )
    atom_array.element = np.asarray(["C", "O", "H"])
    atom_array.chain_id = np.asarray(["A", "A", "A"])
    atom_array.res_id = np.asarray([1, 1, 1])
    atom_array.res_name = np.asarray(["LIG", "LIG", "LIG"])
    atom_array.atom_name = np.asarray(["C1", "O1", "H1"])
    atom_array.hetero = np.ones(3, dtype=bool)
    return atom_array


def _write_bond_sidecar(path, records: dict[str, np.ndarray]) -> None:
    path.mkdir()
    environment = lmdb.open(str(path), map_size=1 << 20)
    with environment.begin(write=True) as transaction:
        transaction.put(
            BOND_SIDECAR_METADATA_KEY,
            pickle.dumps(
                {
                    "atomic_numbers_hash_algorithm": (ATOMIC_NUMBERS_HASH_ALGORITHM),
                    "complete": True,
                }
            ),
        )
        for example_id, bond_edges in records.items():
            record = {
                "bond_edges": np.asarray(bond_edges, dtype=np.int32),
                "num_atoms": 3,
                "atomic_numbers_hash": hash_atomic_numbers([6, 8, 1]),
                "has_disconnected_fragments": False,
                "method": "test",
                "method_version": "1",
            }
            transaction.put(
                example_id.encode("utf-8"),
                encode_bond_record(record),
            )
    environment.close()


def _write_fragment_status_sidecar(path, fragment_statuses: dict[str, bool]) -> None:
    path.mkdir()
    environment = lmdb.open(str(path), map_size=1 << 20)
    with environment.begin(write=True) as transaction:
        transaction.put(
            BOND_SIDECAR_METADATA_KEY,
            pickle.dumps({"complete": True, "dataset": "test"}),
        )
        for example_id, fragment_status in fragment_statuses.items():
            transaction.put(
                example_id.encode("utf-8"),
                pickle.dumps(
                    {"has_disconnected_fragments": fragment_status},
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )
    environment.close()


def _add_structure_prediction_token_bonds(data):
    from nucleus.transforms.pipelines import (
        StructurePredictionAddAF3TokenBondFeatures,
    )

    atom_array = data["atom_array"]
    atom_array.set_annotation("atomize", np.ones(len(atom_array), dtype=bool))
    atom_array.set_annotation("is_polymer", np.zeros(len(atom_array), dtype=bool))
    data["is_inference"] = True
    return StructurePredictionAddAF3TokenBondFeatures().forward(data)


def test_hash_atomic_numbers_is_order_sensitive_and_platform_stable() -> None:
    assert hash_atomic_numbers([8, 1, 1]) == (
        "aec2b0f19fba29cefac27eda8d9aaa10db54cc91e38186b43613840583d1e7a3"
    )
    assert hash_atomic_numbers([8, 1, 1]) != hash_atomic_numbers([1, 8, 1])


def test_write_fragment_status_index_cache_uses_zero_based_dataset_order(
    tmp_path,
) -> None:
    sidecar_path = tmp_path / "fragment-status.lmdb"
    output_path = tmp_path / "connected.txt"
    _write_fragment_status_sidecar(
        sidecar_path,
        {
            "example-a": False,
            "example-b": True,
            "example-c": False,
        },
    )

    selected_count = write_fragment_status_index_cache(
        example_ids=["example-a", "example-b", "example-c"],
        dataset_length=3,
        bond_sidecar_path=sidecar_path,
        output_path=output_path,
        data_id="connected",
        has_disconnected_fragments=False,
        expected_count=2,
    )

    assert selected_count == 2
    assert output_path.read_text(encoding="utf-8") == (
        "# data_id=connected dataset_len=3\n0\n2\n"
    )
    assert read_bond_sidecar_metadata(sidecar_path) == {
        "complete": True,
        "dataset": "test",
    }


def test_write_fragment_status_index_cache_is_atomic_on_count_mismatch(
    tmp_path,
) -> None:
    sidecar_path = tmp_path / "fragment-status.lmdb"
    output_path = tmp_path / "connected.txt"
    _write_fragment_status_sidecar(sidecar_path, {"example-a": False})

    with pytest.raises(ValueError, match="Selected 1 indices, expected 2"):
        write_fragment_status_index_cache(
            example_ids=["example-a"],
            dataset_length=1,
            bond_sidecar_path=sidecar_path,
            output_path=output_path,
            data_id="connected",
            has_disconnected_fragments=False,
            expected_count=2,
        )

    assert not output_path.exists()


def test_attach_bonds_replaces_existing_graph_with_validated_sidecar(tmp_path) -> None:
    sidecar_path = tmp_path / "bonds.lmdb"
    _write_bond_sidecar(
        sidecar_path,
        {"source-example": np.asarray([[0, 1], [0, 2]])},
    )
    atom_array = _ligand_atom_array()
    atom_array.bonds = BondList(3, np.asarray([[1, 2]], dtype=np.int32))

    loader = BondSidecarASEAtomsLoader(bond_sidecar_path=sidecar_path)
    try:
        has_disconnected_fragments = loader.attach_bonds(
            atom_array,
            example_id="source-example",
        )
    finally:
        loader.close()

    assert has_disconnected_fragments is False
    np.testing.assert_array_equal(
        atom_array.bonds.as_array()[:, :2],
        np.asarray([[0, 1], [0, 2]]),
    )


def test_fragment_status_validates_atomic_number_order(tmp_path) -> None:
    sidecar_path = tmp_path / "bonds.lmdb"
    _write_bond_sidecar(
        sidecar_path,
        {"source-example": np.asarray([[0, 1], [0, 2]])},
    )
    loader = BondSidecarASEAtomsLoader(bond_sidecar_path=sidecar_path)
    try:
        assert not loader.fragment_status(
            example_id="source-example",
            atomic_numbers=np.asarray([6, 8, 1]),
        )
        with pytest.raises(ValueError, match="atomic-number hash"):
            loader.fragment_status(
                example_id="source-example",
                atomic_numbers=np.asarray([8, 6, 1]),
            )
    finally:
        loader.close()


def test_multi_example_contig_validation_uses_per_spec_sidecar_ids(
    tmp_path,
) -> None:
    from nucleus.inference_input_parsing import ZatomDesignInputSpecification
    from nucleus.utils.datasets import ZatomContigJsonDataset

    sidecar_path = tmp_path / "bonds.lmdb"
    _write_bond_sidecar(
        sidecar_path,
        {
            "training-example-a": np.asarray([[0, 1], [0, 2]]),
            "training-example-b": np.asarray([[0, 1], [1, 2]]),
        },
    )

    def structure_prediction_specification(source_example_id):
        return ZatomDesignInputSpecification.safe_init(
            example_id=source_example_id,
            input=None,
            atom_array_input=_ligand_atom_array(),
            contig="A1",
            select_fixed_atoms=False,
            select_unfixed_sequence=False,
        )

    dataset = ZatomContigJsonDataset(
        data={
            "validation-example-a": structure_prediction_specification(
                "training-example-a"
            ),
            "validation-example-b": structure_prediction_specification(
                "training-example-b"
            ),
        },
        cif_parser_args={},
        transform=_add_structure_prediction_token_bonds,
        name="bond-sidecar-validation-test",
        subset_to_keys=None,
        eval_every_n=1,
        bond_sidecar_path=sidecar_path,
    )

    first = dataset[0]
    second = dataset[1]

    assert first["example_id"] == "validation-example-a"
    assert second["example_id"] == "validation-example-b"
    assert first["has_disconnected_fragments"] is False
    assert second["has_disconnected_fragments"] is False
    np.testing.assert_array_equal(
        first["atom_array"].bonds.as_array()[:, :2],
        np.asarray([[0, 1], [0, 2]]),
    )
    np.testing.assert_array_equal(
        second["atom_array"].bonds.as_array()[:, :2],
        np.asarray([[0, 1], [1, 2]]),
    )
    first_token_bonds = first["feats"]["token_bonds"]
    second_token_bonds = second["feats"]["token_bonds"]
    assert first_token_bonds.shape == (3, 3)
    assert second_token_bonds.shape == (3, 3)
    assert first_token_bonds[0, 2] and not first_token_bonds[1, 2]
    assert second_token_bonds[1, 2] and not second_token_bonds[0, 2]


@pytest.mark.skipif(openbabel is None, reason="Open Babel is not installed")
def test_contig_validation_perceives_bonds_without_sidecar() -> None:
    from nucleus.inference_input_parsing import ZatomDesignInputSpecification
    from nucleus.utils.datasets import ZatomContigJsonDataset

    native_atom_array = _ligand_atom_array()
    expected_record = openbabel_bond_record(
        native_atom_array,
        example_id="on-the-fly-example",
    )
    specification = ZatomDesignInputSpecification.safe_init(
        input=None,
        atom_array_input=native_atom_array,
        contig="A1",
        select_fixed_atoms=False,
        select_unfixed_sequence=False,
    )
    dataset = ZatomContigJsonDataset(
        data={"on-the-fly-example": specification},
        cif_parser_args={},
        transform=_add_structure_prediction_token_bonds,
        name="on-the-fly-bond-validation-test",
        subset_to_keys=None,
        eval_every_n=1,
    )

    data = dataset[0]

    np.testing.assert_allclose(data["atom_array"].coord, 0.0)
    np.testing.assert_array_equal(
        data["atom_array"].bonds.as_array()[:, :2],
        expected_record["bond_edges"],
    )
    assert data["has_disconnected_fragments"] is bool(
        expected_record["has_disconnected_fragments"]
    )
    assert data["feats"]["token_bonds"].sum() == 2 * len(expected_record["bond_edges"])


@pytest.mark.skipif(openbabel is None, reason="Open Babel is not installed")
def test_contig_inference_perceives_bonds_without_sidecar() -> None:
    # `zatom2 design` loads specs through RFD3's ContigJsonDataset rather than
    # ZatomContigJsonDataset, so perception must happen in to_pipeline_input.
    from nucleus.inference_input_parsing import (
        ZatomDesignInputSpecification,
        install_molecule_contig_support,
    )
    from rfd3.inference.datasets import ContigJsonDataset

    install_molecule_contig_support()
    native_atom_array = _ligand_atom_array()
    expected_record = openbabel_bond_record(
        native_atom_array,
        example_id="on-the-fly-example",
    )
    specification = ZatomDesignInputSpecification.safe_init(
        input=None,
        atom_array_input=native_atom_array,
        contig="A1",
        select_fixed_atoms=False,
        select_unfixed_sequence=False,
    )
    dataset = ContigJsonDataset(
        data={"on-the-fly-example": specification},
        cif_parser_args={},
        transform=_add_structure_prediction_token_bonds,
        name="on-the-fly-bond-inference-test",
        subset_to_keys=None,
        eval_every_n=1,
    )

    data = dataset[0]

    np.testing.assert_allclose(data["atom_array"].coord, 0.0)
    np.testing.assert_array_equal(
        data["atom_array"].bonds.as_array()[:, :2],
        expected_record["bond_edges"],
    )
    assert data["has_disconnected_fragments"] is bool(
        expected_record["has_disconnected_fragments"]
    )
    assert data["feats"]["token_bonds"].sum() == 2 * len(expected_record["bond_edges"])


@pytest.mark.skipif(openbabel is None, reason="Open Babel is not installed")
def test_openbabel_bond_record_stores_edges_without_bond_types() -> None:
    record = openbabel_bond_record(_water_atom_array(), example_id="water")

    assert record["num_atoms"] == 3
    assert record["method"] == "openbabel"
    assert record["bond_edges"].dtype == np.int32
    np.testing.assert_array_equal(
        record["bond_edges"],
        np.asarray([[0, 1], [0, 2]], dtype=np.int32),
    )

    # Biotite accepts [E, 2] and assigns BondType.ANY internally.
    bonds = BondList(record["num_atoms"], record["bond_edges"])
    assert bonds.as_array().shape == (2, 3)


@pytest.mark.skipif(openbabel is None, reason="Open Babel is not installed")
def test_bond_record_pickle_round_trip() -> None:
    record = openbabel_bond_record(_water_atom_array(), example_id="water")
    decoded = decode_bond_record(encode_bond_record(record))

    assert decoded.keys() == record.keys()
    np.testing.assert_array_equal(decoded["bond_edges"], record["bond_edges"])
    assert decoded["atomic_numbers_hash"] == record["atomic_numbers_hash"]


@pytest.mark.skipif(openbabel is None, reason="Open Babel is not installed")
def test_openbabel_bond_record_does_not_require_rdkit_sanitization() -> None:
    record = openbabel_bond_record(_ammonium_atom_array(), example_id="ammonium")

    np.testing.assert_array_equal(
        record["bond_edges"],
        np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int32),
    )
