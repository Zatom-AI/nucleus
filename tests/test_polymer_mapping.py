import numpy as np
import pytest
from atomworks.enums import ChainType
from biotite.structure import AtomArray

import nucleus.transforms.polymer_mapping as polymer_mapping
from nucleus.transforms.polymer_mapping import (
    ContextAwareAssignTypes,
    MapPolymerNucleotideParents,
    UnsupportedPolymerNucleotideError,
)


def _token_array(
    res_name: str,
    *,
    is_polymer: bool,
    atom_names: list[str],
    chain_type: ChainType | None = None,
) -> AtomArray:
    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = atom_names
    atom_array.res_name = np.full(len(atom_names), res_name)
    atom_array.chain_id = np.full(len(atom_names), "A")
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.token_id = np.zeros(len(atom_names), dtype=int)
    atom_array.set_annotation("is_polymer", np.full(len(atom_names), is_polymer))
    if chain_type is not None:
        atom_array.set_annotation(
            "chain_type", np.full(len(atom_names), chain_type.value, dtype=np.int8)
        )
    return atom_array


def test_maps_polymer_modified_rna_and_records_dropped_atoms():
    atom_array = _token_array(
        "2MG",
        is_polymer=True,
        atom_names=["C1'", "N9", "C8", *[f"X{i}" for i in range(20)], "CM2"],
    )
    data = MapPolymerNucleotideParents()({"atom_array": atom_array})

    mapped = data["atom_array"]
    assert set(mapped.res_name) == {"G"}
    assert set(mapped.nucleus_original_res_name) == {"2MG"}
    assert set(mapped.nucleus_mapping_status) == {"mapped"}
    assert "CM2" not in mapped.atom_name
    assert data["polymer_mapping_summary"] == {
        "mapped:2MG->G": 1,
        "dropped_atoms:2MG": 21,
    }


def test_stanford_parent_atom_filter_removes_noncanonical_heavy_atoms():
    canonical_g_atoms = [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
        "C5",
        "C6",
        "O6",
        "N7",
        "C8",
    ]
    atom_array = _token_array(
        "G7M",
        is_polymer=True,
        atom_names=[*canonical_g_atoms, "CN7"],
        chain_type=ChainType.RNA,
    )

    mapped = MapPolymerNucleotideParents()({"atom_array": atom_array})["atom_array"]

    assert len(mapped) == len(canonical_g_atoms)
    assert "CN7" not in mapped.atom_name
    assert set(mapped.res_name) == {"G"}


@pytest.mark.parametrize("res_name", ["GDP", "ATP", "UDP"])
def test_free_nucleotide_like_components_remain_ligands(res_name):
    atom_array = _token_array(res_name, is_polymer=False, atom_names=["P"])
    data = MapPolymerNucleotideParents()({"atom_array": atom_array})
    data = ContextAwareAssignTypes()(data)

    assert set(data["atom_array"].res_name) == {res_name}
    assert data["atom_array"].is_ligand.all()
    assert not data["atom_array"].is_rna.any()
    assert not data["atom_array"].is_dna.any()


def test_rna_chain_context_does_not_reclassify_amino_acid_or_cap():
    """Chain-level RNA labels must not override residue-level chemistry."""
    atom_array = AtomArray(2)
    atom_array.atom_name = ["CA", "C"]
    atom_array.res_name = ["VAL", "ACE"]
    atom_array.chain_id = ["A", "A"]
    atom_array.res_id = [1, 2]
    atom_array.token_id = [0, 1]
    atom_array.set_annotation("is_polymer", np.ones(2, dtype=bool))
    atom_array.set_annotation(
        "chain_type", np.full(2, ChainType.RNA.value, dtype=np.int8)
    )

    data = MapPolymerNucleotideParents()({"atom_array": atom_array})
    data = ContextAwareAssignTypes()(data)
    mapped = data["atom_array"]

    assert mapped.is_protein.tolist() == [True, False]
    assert mapped.is_ligand.tolist() == [False, True]
    assert not mapped.is_rna.any()
    assert not mapped.is_dna.any()


def test_stanford_nakb_fallback_maps_polymer_rna():
    atom_array = _token_array(
        "G5J",
        is_polymer=True,
        atom_names=["C1'"],
        chain_type=ChainType.RNA,
    )
    data = MapPolymerNucleotideParents()({"atom_array": atom_array})

    assert set(data["atom_array"].res_name) == {"G"}


def test_modified_dna_maps_to_a_deoxy_parent():
    """DNA modifications must not be mapped to a 23-slot RNA parent."""
    atom_array = _token_array(
        "6MA",
        is_polymer=True,
        atom_names=["C1'", "N9", "C8"],
        chain_type=ChainType.DNA,
    )

    data = MapPolymerNucleotideParents()({"atom_array": atom_array})

    assert set(data["atom_array"].res_name) == {"DA"}


def test_stanford_nakb_parent_takes_precedence_over_ccd_parent(monkeypatch):
    """Stanford/NAKB resolves an overlap without relying on CCD precedence."""
    monkeypatch.setattr(
        polymer_mapping, "_atomworks_rna_parent_by_ccd", lambda: {"G5J": "A"}
    )

    assert polymer_mapping._rna_parent_for_ccd("G5J") == "G"


def test_unsupported_polymer_nucleotide_is_explicitly_rejected():
    atom_array = _token_array(
        "ZZZ",
        is_polymer=True,
        atom_names=["C1'"],
        chain_type=ChainType.RNA,
    )

    with pytest.raises(UnsupportedPolymerNucleotideError, match="ZZZ"):
        MapPolymerNucleotideParents()({"atom_array": atom_array})
