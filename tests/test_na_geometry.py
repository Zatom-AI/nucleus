import sys

import numpy as np
from biotite.structure import AtomArray
from rfd3na.transforms.na_geom import CalculateNucleicAcidGeomFeats

from nucleus.transforms.na_geometry import SafeCalculateNucleicAcidGeomFeats


def _atom_array(res_name: str = "A") -> AtomArray:
    atom_array = AtomArray(1)
    atom_array.atom_name = ["C1'"]
    atom_array.res_name = [res_name]
    atom_array.chain_id = ["A"]
    atom_array.res_id = [1]
    atom_array.coord = np.zeros((1, 3), dtype=np.float32)
    atom_array.set_annotation("chain_iid", np.array(["chain_a"]))
    return atom_array


def test_safe_na_geometry_masks_unsupported_legacy_pdb_input(monkeypatch):
    """Long mmCIF residue names must be unknown, never labelled unpaired."""
    atom_array = _atom_array(res_name="LONG")
    monkeypatch.setenv("HBPLUS_PATH", sys.executable)
    data = SafeCalculateNucleicAcidGeomFeats(is_inference=False)(
        {"atom_array": atom_array}
    )

    assert data["log_dict"] == {
        "na_ss_available": 0,
        "na_ss_unavailable_res_name_too_long": 1,
    }
    assert atom_array.bp_partners.tolist() == [None]


def test_safe_na_geometry_keeps_only_bp_partners_from_working_copy(monkeypatch):
    """Temporary legacy-PDB mutations cannot leak into model coordinates."""
    atom_array = _atom_array()
    original_chain_id = atom_array.chain_id.copy()
    monkeypatch.setenv("HBPLUS_PATH", sys.executable)

    def fake_geometry_forward(self, data):
        working_array = data["atom_array"]
        working_array.chain_id[:] = "Z"
        working_array.set_annotation("bp_partners", np.array([[]], dtype=object))
        return data

    monkeypatch.setattr(CalculateNucleicAcidGeomFeats, "forward", fake_geometry_forward)
    data = SafeCalculateNucleicAcidGeomFeats(is_inference=False)(
        {"atom_array": atom_array}
    )

    assert np.array_equal(atom_array.chain_id, original_chain_id)
    assert atom_array.bp_partners.tolist() == [[]]
    assert data["log_dict"]["na_ss_available"] == 1
