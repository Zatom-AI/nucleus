import pytest

from nucleus.inference_input_parsing import rewrite_inputless_generated_contig
from nucleus.utils.datasets import ZatomContigJsonDataset


@pytest.mark.parametrize(
    "contig",
    [
        "19R",
        "13-13R,/0,13-13R",
        "24D",
        "32-32R,/0,90-90P",
        "7M",
        "4m",
    ],
)
def test_inputless_generated_contig_uses_length_path(contig):
    spec = {"contig": contig, "plddt_enhanced": False}

    rewritten = rewrite_inputless_generated_contig(spec)

    assert rewritten == {"length": contig, "plddt_enhanced": False}
    assert spec == {"contig": contig, "plddt_enhanced": False}


@pytest.mark.parametrize(
    "spec",
    [
        {"input": "input.cif", "contig": "10R"},
        {"atom_array_input": object(), "contig": "10R"},
        {"contig": "A1-4,10R"},
        {"contig": "10R", "length": 10},
        {"contig": "not-a-contig"},
    ],
)
def test_generated_contig_rewrite_preserves_other_specifications(spec):
    assert rewrite_inputless_generated_contig(spec) is spec


def test_zatom_contig_dataset_builds_inputless_mixed_polymer_complex():
    dataset = ZatomContigJsonDataset(
        data={"mixed": {"contig": "2R,/0,3P", "plddt_enhanced": False}},
        parser_backend="rfd3na",
        cif_parser_args={},
        transform=lambda data: data,
        name="generated-contig-test",
        subset_to_keys=None,
        eval_every_n=1,
    )

    data = dataset[0]
    atom_array = data["atom_array"]

    assert set(atom_array.chain_id) == {"A", "B"}
    rna_chain = atom_array[atom_array.chain_id == "A"]
    protein_chain = atom_array[atom_array.chain_id == "B"]
    assert set(rna_chain.res_name) == {"A"}
    assert set(protein_chain.res_name) == {"ALA"}
    assert set(rna_chain.res_id) == {1, 2}
    assert set(protein_chain.res_id) == {1, 2, 3}
