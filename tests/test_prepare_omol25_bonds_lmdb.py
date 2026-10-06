"""Tests for the OMol25 bond-sidecar preparation script."""

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray


@pytest.fixture(scope="module")
def bonds_script():
    """Load the bond-sidecar script as an importable module."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_omol25_bonds_lmdb.py"
    )
    module_name = "prepare_omol25_bonds_lmdb"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeDataset:
    """A fake dataset class to simulate the behavior of the actual dataset used in the script."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        """Return the row at the specified index."""
        return self.rows[index]

    def idx_to_id(self, index):
        """Return a fake entry ID for the given index."""
        return f"example-{index}"


@pytest.mark.parametrize(
    ("data_id", "expected_entry_ids"),
    [
        ("geom_orca6", ["example-0"]),
        ("all", ["example-0", "example-1"]),
    ],
)
def test_iter_tasks_supports_all_data_ids(
    bonds_script,
    monkeypatch,
    data_id,
    expected_entry_ids,
):
    """Test that _iter_tasks correctly filters rows based on the data_id parameter."""
    atom_arrays = [object(), object()]
    rows = [
        {"data": {"data_id": "geom_orca6"}, "atom_array": atom_arrays[0]},
        {"data": {"data_id": "omol_other"}, "atom_array": atom_arrays[1]},
    ]
    dataset = _FakeDataset(rows)
    monkeypatch.setattr(
        bonds_script,
        "ASE_ATOMS_LOADER",
        lambda row: {"atom_array": row["atom_array"]},
    )
    stats = bonds_script.ScanStats()

    tasks = list(
        bonds_script._iter_tasks(
            dataset,
            range(len(rows)),
            data_id=data_id,
            max_records=None,
            scan_progress_every=0,
            stats=stats,
        )
    )

    assert [task.entry_id for task in tasks] == expected_entry_ids
    assert stats.scanned_count == len(rows)
    assert stats.matched_count == len(expected_entry_ids)


@pytest.mark.parametrize(
    ("bond_edges", "expected_disconnected", "expected_fragment_count"),
    [
        ([[0, 1], [1, 2], [2, 3]], False, 1),
        ([[0, 1], [2, 3]], True, 2),
    ],
)
def test_disconnected_fragment_assessment(
    bonds_script,
    bond_edges,
    expected_disconnected,
    expected_fragment_count,
):
    """The RDKit assessment uses the stored edges and canonical-SMILES dots."""
    atom_array = AtomArray(4)
    atom_array.element = np.asarray(["H", "H", "H", "H"])

    disconnected, fragment_count = bonds_script._disconnected_fragment_assessment(
        atom_array,
        np.asarray(bond_edges, dtype=np.int32),
    )

    assert disconnected is expected_disconnected
    assert fragment_count == expected_fragment_count


def test_process_task_stores_disconnected_fragment_flag(
    bonds_script,
    monkeypatch,
):
    """The serialized sidecar record includes the native boolean flag."""
    atom_array = AtomArray(4)
    atom_array.element = np.asarray(["H", "H", "H", "H"])
    bond_edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
    monkeypatch.setattr(
        bonds_script,
        "openbabel_bond_record",
        lambda atom_array, example_id: {"bond_edges": bond_edges},
    )

    result = bonds_script._process_task(
        bonds_script.BondTask(entry_id="disconnected", atom_array=atom_array)
    )
    assert result.error is None
    assert result.has_disconnected_fragments is True
    assert result.fragment_count == 2
    assert result.value is not None
    record = pickle.loads(result.value)
    assert record["has_disconnected_fragments"] is True


@pytest.mark.parametrize(
    ("split_name", "expected"),
    [
        ("train_4M", "OMol25_train_4M"),
        ("val", "OMol25_val"),
        ("test", "OMol25_test"),
    ],
)
def test_dataset_name_uses_input_split(bonds_script, split_name, expected):
    """Sidecar metadata identifies the input split rather than always training."""
    assert bonds_script._dataset_name(Path("/data/omol25/raw") / split_name) == expected
