import sys

import numpy as np
from biotite.structure import AtomArray
from rfd3na.transforms.design_transforms import AddAdditional1dFeaturesToFeats
from rfd3na.transforms.hbonds_hbplus import CalculateHbondsPlus

from nucleus.transforms.hbplus import SafeCalculateHbondsPlus


def _atom_array(*, res_name: str = "ALA", n_chain_iids: int = 1) -> AtomArray:
    atom_array = AtomArray(n_chain_iids)
    atom_array.atom_name = np.full(n_chain_iids, "CA")
    atom_array.res_name = np.full(n_chain_iids, res_name)
    atom_array.chain_id = np.full(n_chain_iids, "A")
    atom_array.res_id = np.arange(1, n_chain_iids + 1)
    atom_array.coord = np.zeros((n_chain_iids, 3), dtype=np.float32)
    atom_array.set_annotation(
        "chain_iid", np.asarray([f"chain_{index}" for index in range(n_chain_iids)])
    )
    return atom_array


def test_safe_hbplus_skips_long_residue_names_without_calling_hbplus(monkeypatch):
    """Valid mmCIF-only residue names must not fail the data pipeline."""
    atom_array = _atom_array(res_name="LONG")

    def should_not_run(*args, **kwargs):
        raise AssertionError("HBPLUS should be skipped for long residue names")

    monkeypatch.setattr(CalculateHbondsPlus, "forward", should_not_run)
    data = SafeCalculateHbondsPlus()({"atom_array": atom_array})

    assert data["log_dict"] == {
        "hbplus_available": 0,
        "hbplus_unavailable_res_name_too_long": 1,
    }
    assert "active_donor" not in atom_array.get_annotation_categories()
    assert "active_acceptor" not in atom_array.get_annotation_categories()


def test_safe_hbplus_skips_more_than_legacy_pdb_chain_limit(monkeypatch):
    """HBPLUS cannot represent more than 52 chain instances in legacy PDB."""
    atom_array = _atom_array(n_chain_iids=53)
    monkeypatch.setenv("HBPLUS_PATH", sys.executable)
    data = SafeCalculateHbondsPlus()({"atom_array": atom_array})

    assert data["log_dict"] == {
        "hbplus_available": 0,
        "hbplus_unavailable_too_many_chain_iids": 1,
    }


def test_safe_hbplus_transfers_features_without_mutating_structure(monkeypatch):
    """Only HBPLUS annotations, not temporary-PDB mutations, reach downstream."""
    atom_array = _atom_array()
    original_chain_id = atom_array.chain_id.copy()
    monkeypatch.setenv("HBPLUS_PATH", sys.executable)

    def fake_hbplus_forward(self, data):
        working_array = data["atom_array"]
        working_array.chain_id[:] = "Z"
        working_array.set_annotation("active_donor", np.array([1]))
        working_array.set_annotation("active_acceptor", np.array([0]))
        data.setdefault("log_dict", {})["hbond_total_count"] = 1
        return data

    monkeypatch.setattr(CalculateHbondsPlus, "forward", fake_hbplus_forward)
    data = SafeCalculateHbondsPlus()({"atom_array": atom_array})

    assert np.array_equal(atom_array.chain_id, original_chain_id)
    assert atom_array.active_donor.tolist() == [1]
    assert atom_array.active_acceptor.tolist() == [0]
    assert data["log_dict"]["hbplus_available"] == 1
    assert data["log_dict"]["hbond_total_count"] == 1


def test_unavailable_hbplus_features_are_zero_filled_for_the_model(monkeypatch):
    """Skipped HBPLUS must satisfy the existing atom-feature contract."""
    atom_array = _atom_array(res_name="LONG")
    monkeypatch.delenv("HBPLUS_PATH", raising=False)
    data = SafeCalculateHbondsPlus()({"atom_array": atom_array})
    data = AddAdditional1dFeaturesToFeats(
        token_1d_features={},
        atom_1d_features={"active_donor": 1, "active_acceptor": 1},
        autofill_zeros_if_not_present_in_atomarray=True,
    )(data)

    assert data["feats"]["active_donor"].tolist() == [[0.0]]
    assert data["feats"]["active_acceptor"].tolist() == [[0.0]]


def test_safe_hbplus_converts_tool_errors_to_an_unavailable_feature(monkeypatch):
    """A failing external tool must not reject an otherwise usable crop."""
    atom_array = _atom_array()
    monkeypatch.setenv("HBPLUS_PATH", sys.executable)

    def failing_hbplus_forward(self, data):
        raise RuntimeError("HBPLUS did not write an hb2 file")

    monkeypatch.setattr(CalculateHbondsPlus, "forward", failing_hbplus_forward)
    data = SafeCalculateHbondsPlus()({"atom_array": atom_array})

    assert data["log_dict"] == {
        "hbplus_available": 0,
        "hbplus_unavailable_runtime_error": 1,
    }
    assert "active_donor" not in atom_array.get_annotation_categories()
