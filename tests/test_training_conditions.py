import numpy as np

from nucleus.transforms.training_conditions import LigandAwareIslandCondition


class _AtomArrayStub:
    def __init__(self, *, protein=False, ligand=False, nucleotide=False):
        self.is_protein = np.asarray([protein], dtype=bool)
        self.is_ligand = np.asarray([ligand], dtype=bool)
        self.is_dna = np.asarray([False], dtype=bool)
        self.is_rna = np.asarray([nucleotide], dtype=bool)


def _condition(name, mode):
    return LigandAwareIslandCondition(
        name=name,
        frequency=1.0,
        molecule_conditioning_mode=mode,
        island_sampling_kwargs={},
        p_diffuse_motif_sidechains=0.0,
        p_diffuse_subgraph_atoms=0.0,
        subgraph_sampling_kwargs={},
        p_fix_motif_coordinates=0.0,
        p_fix_motif_sequence=0.0,
        p_unindex_motif_tokens=0.0,
        association_scheme="atom23",
    )


def _data(
    *,
    protein=False,
    ligand=False,
    nucleotide=False,
    has_disconnected_fragments=None,
):
    data = {
        "atom_array": _AtomArrayStub(
            protein=protein,
            ligand=ligand,
            nucleotide=nucleotide,
        )
    }
    if has_disconnected_fragments is not None:
        data["has_disconnected_fragments"] = has_disconnected_fragments
    return data


def test_ligand_conditioned_nucleotide_requires_no_protein():
    condition = _condition(
        "ligand_conditioned_nucleotide",
        "ligand_conditioned_nucleotide_design",
    )

    assert condition.is_valid_for_example(
        _data(protein=False, ligand=True, nucleotide=True)
    )
    assert not condition.is_valid_for_example(
        _data(protein=True, ligand=True, nucleotide=True)
    )


def test_ligand_conditioned_protein_requires_no_nucleotide():
    condition = _condition(
        "ligand_conditioned_protein",
        "ligand_conditioned_protein_design",
    )

    assert condition.is_valid_for_example(
        _data(protein=True, ligand=True, nucleotide=False)
    )
    assert not condition.is_valid_for_example(
        _data(protein=True, ligand=True, nucleotide=True)
    )


def test_three_component_conditions_require_all_three_modalities():
    protein_ligand_nucleotide = _condition(
        "protein_ligand_conditioned_nucleotide",
        "protein_ligand_conditioned_nucleotide_design",
    )
    nucleotide_ligand_protein = _condition(
        "nucleotide_ligand_conditioned_protein",
        "nucleotide_ligand_conditioned_protein_design",
    )

    three_component = _data(protein=True, ligand=True, nucleotide=True)
    two_component = _data(protein=False, ligand=True, nucleotide=True)

    assert protein_ligand_nucleotide.is_valid_for_example(three_component)
    assert nucleotide_ligand_protein.is_valid_for_example(three_component)
    assert not protein_ligand_nucleotide.is_valid_for_example(two_component)
    assert not nucleotide_ligand_protein.is_valid_for_example(two_component)


def test_disconnected_molecules_are_excluded_only_from_structure_prediction():
    structure_prediction = _condition(
        "structure_prediction",
        "sequence_fixed_structure_design",
    )
    unconditional = _condition("unconditional", "unconditional_design")

    connected = _data(ligand=True, has_disconnected_fragments=False)
    disconnected = _data(ligand=True, has_disconnected_fragments=True)

    assert structure_prediction.is_valid_for_example(connected)
    assert not structure_prediction.is_valid_for_example(disconnected)
    assert unconditional.is_valid_for_example(disconnected)

    # Modalities without a bond sidecar retain their previous behavior.
    assert structure_prediction.is_valid_for_example(_data(ligand=True))


def test_ligand_only_rejects_condition_requiring_polymer_context():
    condition = _condition("island", "protein_conditioned_molecule_design")

    assert not condition.is_valid_for_example(_data(ligand=True))


def test_ligand_only_accepts_each_supported_molecule_mode():
    for mode in LigandAwareIslandCondition.MOLECULE_ONLY_MODES:
        condition = _condition("molecule_only", mode)

        assert condition.is_valid_for_example(_data(ligand=True))
