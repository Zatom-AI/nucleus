"""Unit tests for the vendored NA-MPNN port: the DNA/RNA shared-token
aliasing + post-hoc decode (the trickiest, checkpoint-independent logic in
`_na_mpnn/featurize.py`), and PDB parsing/featurization plumbing for a
synthetic protein+DNA structure.

Real end-to-end correctness (checkpoint load + design() on NA-MPNN's own
`inference/examples/{4oqu,1am9}.pdb` — pure RNA and protein+DNA respectively)
was verified manually against a freshly downloaded checkpoint this session;
that's excluded here since it needs a real network fetch + GPU/CPU inference,
not appropriate for a fast unit-test pass.

`parse_pdb_na`'s CIF branch (added for CIF support) is similarly verified
manually against real downloaded RCSB mmCIF files this session — both a
pure-RNA structure (1EHZ) and a real protein+DNA complex (1AM9) parsed via
`.cif` and `.pdb` gave bit-identical `S`/`X`/`protein_mask`/`dna_mask`
tensors. That's not repeated here as an automated test because a synthetic,
hand-built mmCIF fixture doesn't reliably round-trip through ProDy's
`.select("nucleic")` the way a real deposited mmCIF does (ProDy's nucleic-
acid selection leans on `_entity_poly`/`_chem_comp` metadata a minimal
synthetic CIF writer doesn't populate) — the dispatch *logic* itself
(`.cif` -> `parseMMCIF`, else `parsePDB`) is what's tested below instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.oracles.inverse_folders._na_mpnn.featurize import (  # noqa: E402
    ATOM_ORDER,
    DNA_CHAR_TO_RNA_CHAR,
    POLYTYPE_TO_INT,
    RESTYPE_INT_TO_ONE,
    RESTYPE_TO_INT,
    decode_sequence,
    featurize_na_backbone,
    parse_pdb_na,
    to_standard_nucleotide_alphabet,
)


def test_rna_tokens_alias_onto_dna_softmax_slots() -> None:
    """na_shared_tokens=1 (hardcoded): A/C/G/U/RX must resolve to the exact
    same int as their DNA counterpart, matching what both bundled
    checkpoints were trained with."""
    for rna_name, dna_name in [
        ("A", "DA"), ("C", "DC"), ("G", "DG"), ("U", "DT"), ("RX", "DX"),
    ]:  # fmt: skip
        assert RESTYPE_TO_INT[rna_name] == RESTYPE_TO_INT[dna_name]

    # And the now-dead original slots must never be reachable via lookup —
    # only via the raw softmax, which `_DEFAULT_BIAS` drives to ~0.
    seen_ints = {RESTYPE_TO_INT[n] for n in RESTYPE_TO_INT}
    assert len(seen_ints) == 33 - 5  # 5 aliased names collapse onto 5 others


def test_restype_int_to_one_decodes_aliased_slots_as_dna() -> None:
    """Decoding always yields the DNA one-letter code first (lowercase
    a/c/g/t/x) — RNA conversion happens later, post-hoc, in decode_sequence."""
    assert RESTYPE_INT_TO_ONE[RESTYPE_TO_INT["DA"]] == "a"
    assert RESTYPE_INT_TO_ONE[RESTYPE_TO_INT["A"]] == "a"  # same (aliased) int


def test_dna_char_to_rna_char_covers_all_five_pairs() -> None:
    assert DNA_CHAR_TO_RNA_CHAR == {
        "a": "b", "c": "d", "g": "h", "t": "u", "x": "y",
    }  # fmt: skip


def test_to_standard_nucleotide_alphabet_translates_rna_and_dna() -> None:
    # RNA: b/d/h/u/y -> A/C/G/U/N
    assert to_standard_nucleotide_alphabet("bdhuy") == "ACGUN"
    # DNA: a/c/g/t/x -> A/C/G/T/N
    assert to_standard_nucleotide_alphabet("acgtx") == "ACGTN"


def test_to_standard_nucleotide_alphabet_leaves_protein_and_separator_unchanged() -> (
    None
):
    # Uppercase amino acid letters and the multi-chain "/" separator aren't
    # in the translation table — passed through as-is (chain-splitting is
    # the caller's job, e.g. rna.diversity.compute_sequence_diversity).
    assert to_standard_nucleotide_alphabet("MKV/bdh") == "MKV/ACG"


def test_decode_sequence_converts_rna_positions_only() -> None:
    """A DA/DA/DG (DNA) run should decode unchanged; the same ints with
    rna_mask_for_token_conversion=1 should decode as their RNA equivalents."""
    S = torch.tensor([RESTYPE_TO_INT["DA"], RESTYPE_TO_INT["DC"], RESTYPE_TO_INT["DG"]])
    chain_letters = ["A", "A", "A"]

    dna_seq = decode_sequence(S, [0, 0, 0], chain_letters)
    assert dna_seq == "acg"

    rna_seq = decode_sequence(S, [1, 1, 1], chain_letters)
    assert rna_seq == "bdh"

    mixed_seq = decode_sequence(S, [0, 1, 0], chain_letters)
    assert mixed_seq == "adg"


def test_decode_sequence_groups_by_sorted_chain_letter() -> None:
    S = torch.tensor(
        [RESTYPE_TO_INT["ALA"], RESTYPE_TO_INT["DA"], RESTYPE_TO_INT["ARG"]]
    )
    # Chain B appears before chain A in file order — output must still be
    # sorted (A, then B), matching upstream's mask_c-based grouping.
    chain_letters = ["B", "A", "B"]
    seq = decode_sequence(S, [0, 0, 0], chain_letters)
    assert seq == "a/AR"


_PDB_LINE = (
    "ATOM  {serial:>5} {atom:<4} {resname:>3} {chain}{resnum:>4}    "
    "{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {elem}\n"
)


def _atom(serial, atom, resname, chain, resnum, xyz, elem):
    x, y, z = xyz
    return _PDB_LINE.format(
        serial=serial, atom=atom, resname=resname, chain=chain,
        resnum=resnum, x=x, y=y, z=z, elem=elem,
    )  # fmt: skip


def _write_protein_dna_pdb(path: Path) -> None:
    """One protein residue (chain A) + one DNA residue (chain B), with a
    complete backbone each — enough to exercise both polymer-type branches
    of `parse_pdb_na` without needing a real structure."""
    lines = []
    serial = 1
    protein_backbone = {
        "N": (0.0, 0.0, 0.0), "CA": (1.5, 0.0, 0.0),
        "C": (2.3, 1.3, 0.0), "O": (3.5, 1.4, 0.0),
    }  # fmt: skip
    for atom, xyz in protein_backbone.items():
        lines.append(_atom(serial, atom, "ALA", "A", 1, xyz, atom[0]))
        serial += 1

    dna_backbone = {
        "OP1": (10.0, 0.0, 0.0), "OP2": (10.5, 1.0, 0.0), "P": (11.0, 0.0, 0.0),
        "O5'": (12.0, 0.5, 0.0), "C5'": (13.0, 0.0, 0.0), "C4'": (14.0, 0.5, 0.0),
        "O4'": (14.5, 1.5, 0.0), "C3'": (15.0, 0.0, 0.0), "O3'": (16.0, 0.5, 0.0),
        "C2'": (15.5, -1.0, 0.0), "C1'": (14.8, 2.5, 0.0),
    }  # fmt: skip
    for atom, xyz in dna_backbone.items():
        elem = "P" if atom == "P" else atom[0]
        lines.append(_atom(serial, atom, "DA", "B", 1, xyz, elem))
        serial += 1

    path.write_text("".join(lines) + "END\n")


def test_parse_pdb_na_separates_protein_and_dna_residues(tmp_path: Path) -> None:
    pdb = tmp_path / "mixed.pdb"
    _write_protein_dna_pdb(pdb)

    backbone = parse_pdb_na(pdb)
    assert backbone.X.shape == (2, len(ATOM_ORDER), 3)
    assert list(backbone.protein_mask) == [1, 0]
    assert list(backbone.dna_mask) == [0, 1]
    assert list(backbone.rna_mask) == [0, 0]
    assert backbone.polymer_type[0] == POLYTYPE_TO_INT["PP"]
    assert backbone.polymer_type[1] == POLYTYPE_TO_INT["DNA"]
    assert backbone.S[0] == RESTYPE_TO_INT["ALA"]
    assert backbone.S[1] == RESTYPE_TO_INT["DA"]
    assert backbone.chain_letters == ["A", "B"]


def test_parse_pdb_na_dispatches_on_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`.cif` -> `parseMMCIF`, everything else -> `parsePDB` — the actual
    parsing correctness for each is verified manually against real RCSB
    files (see module docstring); this just confirms the right prody
    function gets called for each extension."""
    import zevals.oracles.inverse_folders._na_mpnn.featurize as featurize_module

    calls = []

    class _FakeAtoms:
        def select(self, *args, **kwargs):
            return self

        def getResnums(self):
            return []

        def getChids(self):
            return []

        def getIcodes(self):
            return []

        def getChindices(self):
            return []

        def getResnames(self):
            return []

    def fake_parse_mmcif(path, **kwargs):
        calls.append(("mmcif", path))
        return _FakeAtoms()

    def fake_parse_pdb(path, **kwargs):
        calls.append(("pdb", path))
        return _FakeAtoms()

    import prody

    monkeypatch.setattr(prody, "parseMMCIF", fake_parse_mmcif)
    monkeypatch.setattr(prody, "parsePDB", fake_parse_pdb)
    monkeypatch.setattr(featurize_module, "np", featurize_module.np)  # sanity no-op

    cif_path = tmp_path / "sample.cif"
    pdb_path = tmp_path / "sample.pdb"
    cif_path.touch()
    pdb_path.touch()

    for path in (cif_path, pdb_path):
        try:
            parse_pdb_na(path)
        except Exception:
            pass  # _FakeAtoms can't carry the parse all the way through — only dispatch matters here

    assert calls[0] == ("mmcif", str(cif_path))
    assert calls[1] == ("pdb", str(pdb_path))


def test_featurize_na_backbone_marks_every_residue_designable(tmp_path: Path) -> None:
    pdb = tmp_path / "mixed.pdb"
    _write_protein_dna_pdb(pdb)

    backbone = parse_pdb_na(pdb)
    feats = featurize_na_backbone(backbone, torch.device("cpu"))
    fd = feats.feature_dict

    assert fd["mask"].shape == (1, 2)
    assert torch.equal(fd["mask"], torch.ones(1, 2))
    assert torch.equal(fd["chain_mask"], fd["mask"])
    assert fd["bias"].shape == (1, 2, 33)
    assert feats.chain_letters == ["A", "B"]


def test_na_mpnn_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    from zevals.oracles.inverse_folders.na_mpnn import NAMPNNFolder

    checkpoint = tmp_path / "s_19137.pt"
    checkpoint.write_bytes(b"not the pinned checkpoint")
    folder = NAMPNNFolder(
        checkpoint_path=str(checkpoint),
        expected_checkpoint_sha256="0" * 64,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        folder._get_model()
