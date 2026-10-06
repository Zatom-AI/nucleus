import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atomworks.ml.datasets import ASELMDBDataset
from atomworks.ml.datasets.loaders import create_ase_materials_loader
from torch_geometric.data import Data, InMemoryDataset

from scripts import prepare_qm9_mp20_lmdb as preparation


def test_qm9_selection_is_seeded_and_mp20_selection_preserves_source_order():
    first = preparation.select_indices("qm9", dataset_size=20, num_examples=8, seed=42)
    second = preparation.select_indices("qm9", dataset_size=20, num_examples=8, seed=42)
    different_seed = preparation.select_indices(
        "qm9", dataset_size=20, num_examples=8, seed=43
    )

    assert first == second
    assert first != different_seed
    assert len(first) == len(set(first)) == 8
    assert preparation.select_indices(
        "mp20", dataset_size=20, num_examples=8, seed=42
    ) == list(range(8))


def test_selection_rejects_more_examples_than_the_source_contains():
    with pytest.raises(ValueError, match="contains only 10"):
        preparation.select_indices("qm9", dataset_size=10, num_examples=11, seed=42)


def test_qm9_record_preserves_hydrogens_positions_and_identifiers():
    example = SimpleNamespace(
        z=torch.tensor([6, 1, 1]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        name="gdb_1",
        smiles="[H]C[H]",
    )

    record = preparation.qm9_record(example, source_index=7)

    assert record.atoms.numbers.tolist() == [6, 1, 1]
    np.testing.assert_allclose(record.atoms.positions, example.pos.numpy())
    assert not record.atoms.pbc.any()
    assert record.key_value_pairs == {
        "dataset": "qm9",
        "source_index": 7,
        "source_id": "gdb_1",
    }
    assert record.data == {"smiles": "[H]C[H]"}


def test_mp20_record_preserves_periodic_geometry_and_space_group():
    cell = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    fractional_coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]])
    example = SimpleNamespace(
        atom_types=torch.tensor([14, 8]),
        frac_coords=fractional_coordinates,
        cell=cell.unsqueeze(0),
        id="mp-test",
        spacegroup=torch.tensor([221]),
    )

    record = preparation.mp20_record(example, source_index=3)

    assert record.atoms.numbers.tolist() == [14, 8]
    assert record.atoms.pbc.all()
    np.testing.assert_allclose(record.atoms.cell.array, cell.numpy())
    np.testing.assert_allclose(
        record.atoms.get_scaled_positions(wrap=False),
        fractional_coordinates.numpy(),
    )
    assert record.key_value_pairs["source_id"] == "mp-test"
    assert record.key_value_pairs["space_group"] == 221


def test_write_shards_produces_atomworks_readable_lmdb_and_manifest(tmp_path):
    examples = [
        SimpleNamespace(
            z=torch.tensor([6, 1]),
            pos=torch.tensor([[float(index), 0.0, 0.0], [0.0, 1.0, 0.0]]),
            name=f"gdb_{index}",
            smiles="C",
        )
        for index in range(3)
    ]
    output_dir = tmp_path / "qm9"
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "data_v3.pt").write_bytes(b"synthetic-source-artifact")

    shard_paths = preparation.write_shards(
        dataset=examples,
        dataset_name="qm9",
        indices=[2, 0, 1],
        output_dir=output_dir,
        shard_size=2,
        converter=preparation.qm9_record,
        seed=42,
        source_root=source_root,
    )

    assert [path.name for path in shard_paths] == [
        "data0000.aselmdb",
        "data0001.aselmdb",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["dataset"] == "qm9"
    assert manifest["source_num_examples"] == 3
    assert manifest["num_examples"] == 3
    assert manifest["selection_seed"] == 42

    with ASELMDBDataset(
        paths=output_dir,
        name="prepared_qm9_test",
        return_type="record",
        readonly=True,
        readahead=False,
    ) as dataset:
        assert len(dataset) == 3
        assert [dataset[index]["atoms"].info["source_index"] for index in range(3)] == [
            2,
            0,
            1,
        ]
        assert dataset[0]["atoms"].numbers.tolist() == [6, 1]

    with pytest.raises(FileExistsError, match="already contains"):
        preparation.write_shards(
            dataset=examples,
            dataset_name="qm9",
            indices=[0],
            output_dir=output_dir,
            shard_size=2,
            converter=preparation.qm9_record,
            seed=42,
            source_root=source_root,
        )


def test_qm9_cli_reads_flat_zatom_archive_layout(tmp_path):
    source_root = tmp_path / "qm9"
    source_root.mkdir()
    InMemoryDataset.save(
        [
            Data(
                z=torch.tensor([6, 1]),
                pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                name="gdb_1",
                smiles="C",
            ),
            Data(
                z=torch.tensor([8, 1]),
                pos=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                name="gdb_2",
                smiles="O",
            ),
        ],
        source_root / "data_v3.pt",
    )
    output_dir = tmp_path / "prepared_qm9"

    preparation.main(
        [
            "qm9",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_dir),
            "--num-examples",
            "2",
            "--seed",
            "42",
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["source_artifact"] == str(source_root / "data_v3.pt")
    assert manifest["num_examples"] == 2


def test_mp20_cli_reads_zatom_artifact_and_writes_material_loader_contract(tmp_path):
    source_root = tmp_path / "mp_20"
    processed_path = source_root / "mp20.pt"
    processed_path.parent.mkdir(parents=True)
    cell = torch.eye(3).unsqueeze(0) * 4.0
    InMemoryDataset.save(
        [
            Data(
                atom_types=torch.tensor([14, 8]),
                frac_coords=torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]]),
                cell=cell,
                id="mp-test",
                spacegroup=torch.tensor([221]),
            )
        ],
        processed_path,
    )
    output_dir = tmp_path / "prepared_mp20"

    preparation.main(
        [
            "mp20",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_dir),
            "--num-examples",
            "1",
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["source_artifact"] == str(processed_path)
    assert len(manifest["source_artifact_sha256"]) == 64
    with ASELMDBDataset(
        paths=output_dir,
        name="prepared_mp20_test",
        return_type="record",
        loader=create_ase_materials_loader(
            wrap_fractional_coordinates=True,
            include_atom_array=True,
        ),
        readonly=True,
        readahead=False,
    ) as dataset:
        assert len(dataset) == 1
        record = dataset[0]
        assert record["atoms"].pbc.all()
        assert record["space_group"] == 221
        np.testing.assert_allclose(
            record["fractional_coordinates"],
            [[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]],
        )
