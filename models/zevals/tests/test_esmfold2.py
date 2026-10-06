"""Unit tests for the ESMFold2 structure-predictor oracle.

These exercise the oracle's own logic — FASTA parsing, pLDDT normalisation,
mmCIF writing, the `fold()` list-vs-single return unwrap, and `None`
confidence handling — WITHOUT importing the heavy `esm` / `transformers`
stack or loading any weights. The model + input builder are faked and
injected via the oracle's lazy-load cache (`_model`/`_builder`/`_input_types`),
so `_get_model()` short-circuits and the real `predict()` code path runs.

Only `torch` is required (CPU is fine); `esm`/`transformers` need not be
installed for these tests to run.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from zevals.oracles.base import StructurePrediction  # noqa: E402
from zevals.oracles.structure_predictors.esmfold2 import (  # noqa: E402
    ESMFold2Predictor,
    _mean_plddt,
    _read_fasta,
)

# ---------------------------------------------------------------------------
# Fakes for the esm input builder / model result
# ---------------------------------------------------------------------------


class _FakeComplex:
    def __init__(self, mmcif: str = "data_fake\n_atom_site.group_PDB\nATOM\n"):
        self._mmcif = mmcif

    def to_mmcif(self) -> str:
        return self._mmcif


class _FakeResult:
    """Mimics esm's MolecularComplexResult (the fields the oracle reads)."""

    def __init__(
        self,
        plddt,
        ptm,
        mmcif="data_fake\n_atom_site\nATOM\n",
        iptm=None,
        pair_chains_iptm=None,
    ):
        self.complex = _FakeComplex(mmcif)
        self.plddt = plddt
        self.ptm = ptm
        self.iptm = iptm
        self.pair_chains_iptm = pair_chains_iptm


class _FakeBuilder:
    """Records fold() calls; returns a preset result (optionally list-wrapped)."""

    def __init__(self, result: _FakeResult, as_list: bool = False):
        self._result = result
        self._as_list = as_list
        self.calls: list[dict] = []

    def fold(self, model, spi, **kwargs):
        self.calls.append({"spi": spi, **kwargs})
        return [self._result] if self._as_list else self._result


# Fake esm input types — the oracle only constructs and passes these through.
def _fake_protein_input(id, sequence):
    return {"id": id, "sequence": sequence}


def _fake_spi(sequences):
    return {"sequences": sequences}


def _make_predictor(builder: _FakeBuilder) -> ESMFold2Predictor:
    p = ESMFold2Predictor(
        num_loops=2, num_sampling_steps=5, num_samples=1, device="cpu"
    )
    # Prime the lazy-load cache so _get_model() short-circuits to the fakes.
    p._model = object()
    p._builder = builder
    p._input_types = {
        "protein": _fake_protein_input,
        "rna": _fake_protein_input,
        "dna": _fake_protein_input,
        "structure": _fake_spi,
    }
    return p


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_read_fasta_parses_multiple_records(tmp_path: Path):
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">seqA desc\nMKT\nLLL\n>seqB\nGGG\n")
    records = _read_fasta(fasta)
    assert records == [("seqA", "MKTLLL"), ("seqB", "GGG")]


def test_mean_plddt_rescales_0_1_to_0_100():
    t = torch.tensor([0.9, 0.8, 1.0])
    assert _mean_plddt(t) == pytest.approx(90.0, abs=1e-4)


def test_mean_plddt_passes_through_0_100_scale():
    t = torch.tensor([90.0, 80.0, 100.0])
    assert _mean_plddt(t) == pytest.approx(90.0, abs=1e-4)


def test_mean_plddt_handles_none_and_empty():
    assert math.isnan(_mean_plddt(None))
    assert math.isnan(_mean_plddt(torch.tensor([])))


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_writes_cif_and_returns_prediction(tmp_path: Path):
    builder = _FakeBuilder(_FakeResult(plddt=torch.tensor([0.95, 0.85]), ptm=0.77))
    predictor = _make_predictor(builder)

    fasta = tmp_path / "in.fasta"
    fasta.write_text(">design0\nMKT\n")
    out = tmp_path / "structures"

    preds = predictor.predict(fasta, out)

    assert len(preds) == 1
    pred = preds[0]
    assert isinstance(pred, StructurePrediction)
    # mmCIF written under seq id, and referenced by the prediction.
    assert pred.pdb_path == out / "design0.cif"
    assert pred.pdb_path.exists()
    assert "_atom_site" in pred.pdb_path.read_text()
    # 0-1 plddt rescaled to 0-100; ptm passed through as float.
    assert pred.plddt == pytest.approx(90.0, abs=1e-4)
    assert pred.ptm == pytest.approx(0.77, abs=1e-6)
    assert pred.sequence == "MKT"
    # fold() called once with the configured knobs.
    assert len(builder.calls) == 1
    assert builder.calls[0]["num_loops"] == 2
    assert builder.calls[0]["num_sampling_steps"] == 5


def test_predict_unwraps_list_return(tmp_path: Path):
    # `fold` is typed to return either a single result or a list — the oracle
    # must unwrap a length-1 list transparently.
    builder = _FakeBuilder(
        _FakeResult(plddt=torch.tensor([1.0]), ptm=0.5), as_list=True
    )
    predictor = _make_predictor(builder)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s\nAAA\n")

    preds = predictor.predict(fasta, tmp_path / "out")

    assert len(preds) == 1
    assert preds[0].plddt == pytest.approx(100.0, abs=1e-4)


def test_predict_ptm_none_becomes_nan(tmp_path: Path):
    builder = _FakeBuilder(_FakeResult(plddt=torch.tensor([0.9]), ptm=None))
    predictor = _make_predictor(builder)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s\nAAA\n")

    preds = predictor.predict(fasta, tmp_path / "out")

    assert math.isnan(preds[0].ptm)


def test_predict_multiple_samples_uses_suffixed_filenames(tmp_path: Path):
    builder = _FakeBuilder(_FakeResult(plddt=torch.tensor([0.9]), ptm=0.6))
    predictor = _make_predictor(builder)
    predictor.num_samples = 3
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">bb\nAAA\n")
    out = tmp_path / "out"

    preds = predictor.predict(fasta, out)

    assert len(preds) == 3
    names = sorted(p.pdb_path.name for p in preds)
    assert names == ["bb_sample0.cif", "bb_sample1.cif", "bb_sample2.cif"]
    # Distinct seeds per sample (seed + i).
    seeds = [c["seed"] for c in builder.calls]
    assert seeds == [0, 1, 2]


def test_rna_predictor_uses_typed_rna_input(tmp_path: Path) -> None:
    builder = _FakeBuilder(_FakeResult(plddt=torch.tensor([0.8]), ptm=0.6))
    predictor = ESMFold2Predictor(entity_type="rna", device="cpu", seed=7)
    predictor._model = object()
    predictor._builder = builder
    predictor._input_types = {
        "protein": lambda **kwargs: pytest.fail("used ProteinInput for RNA"),
        "rna": _fake_protein_input,
        "dna": _fake_protein_input,
        "structure": _fake_spi,
    }
    fasta = tmp_path / "rna.fasta"
    fasta.write_text(">rna_1\nACGU\n")

    predictions = predictor.predict(fasta, tmp_path / "out")

    assert predictions[0].sequence == "ACGU"
    assert builder.calls[0]["spi"] == {"sequences": [{"id": "A", "sequence": "ACGU"}]}
    assert builder.calls[0]["seed"] == 7


def test_rna_predictor_rejects_protein_letters(tmp_path: Path) -> None:
    predictor = ESMFold2Predictor(entity_type="rna", device="cpu")
    predictor._model = object()
    predictor._builder = _FakeBuilder(_FakeResult(plddt=None, ptm=None))
    predictor._input_types = {"rna": _fake_protein_input, "structure": _fake_spi}
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">bad\nACGL\n")

    with pytest.raises(ValueError, match="unsupported symbols"):
        predictor.predict(fasta, tmp_path / "out")


def test_invalid_entity_type_fails() -> None:
    with pytest.raises(ValueError, match="protein, rna, or dna"):
        ESMFold2Predictor(entity_type="ligand", device="cpu")


def test_predict_complex_passes_native_multimodal_input_unchanged(
    tmp_path: Path,
) -> None:
    class FakeStructureInput:
        pass

    native_input = FakeStructureInput()
    pair_iptm = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
    builder = _FakeBuilder(
        _FakeResult(
            plddt=torch.tensor([0.9, 0.8]),
            ptm=0.72,
            iptm=0.68,
            pair_chains_iptm=pair_iptm,
        )
    )
    predictor = _make_predictor(builder)
    predictor._input_types["structure"] = FakeStructureInput

    predictions = predictor.predict_complex(
        native_input, tmp_path / "complex_outputs", name="rna_ligand"
    )

    assert builder.calls[0]["spi"] is native_input
    assert predictions[0].structure_path.name == "rna_ligand.cif"
    assert predictions[0].iptm == pytest.approx(0.68)
    assert torch.equal(predictions[0].extra["pair_chains_iptm"], pair_iptm)


def test_predict_complex_rejects_non_native_input(tmp_path: Path) -> None:
    builder = _FakeBuilder(_FakeResult(plddt=None, ptm=None))
    predictor = _make_predictor(builder)
    predictor._input_types["structure"] = dict

    with pytest.raises(TypeError, match="StructurePredictionInput"):
        predictor.predict_complex(object(), tmp_path / "out")
