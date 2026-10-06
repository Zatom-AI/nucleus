"""Unit tests for the RNA metric modules touched this session:
`rna/validity.py::_tm_score` (was a `NotImplementedError` stub, now a real
TMalign subprocess call), and `rna/diversity.py`/`rna/novelty.py` after
their refactor from a bare `qtmclust_bin`/FoldSeek-typed parameter to a
generic injected `BaseStructureSearcher` (USalign in practice).

`FakeStructureSearcher` mirrors `test_proteins_metrics.py`'s
`FakeFoldSeekSearcher` — same shape, since both modules' metric functions
only depend on the `BaseStructureSearcher` ABC, not a concrete tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.oracles.base import (  # noqa: E402
    InverseFoldResult,
    SearchResult,
    StructurePrediction,
)
from zevals.proteins import diversity as protein_diversity  # noqa: E402
from zevals.rna import validity as rna_validity  # noqa: E402
from zevals.rna.diversity import (  # noqa: E402
    compute_sequence_diversity,
    compute_structural_diversity,
)
from zevals.rna.novelty import compute_novelty  # noqa: E402


class FakeStructureSearcher:
    def __init__(
        self,
        cluster_results: list[SearchResult] | None = None,
        search_results: list[SearchResult] | None = None,
    ):
        self._cluster_results = cluster_results or []
        self._search_results = search_results or []
        self.tmp_dir = Path("/tmp/fake_usalign")

    def cluster(self, structure_dir, output_tsv, min_tm=0.45):
        return self._cluster_results

    def search(self, query_dir, target_db, output_tsv):
        return self._search_results


# --------------------------------------------------------------------------
# validity.py — _tm_score
# --------------------------------------------------------------------------


def test_tm_score_parses_tm1_normalized_by_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_cmd = []

    class FakeResult:
        stdout = (
            "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\n"
            "ref.pdb\tpred.pdb\t0.8123\t0.7000\t1.20\n"
        )

    def fake_run_subprocess(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return FakeResult()

    monkeypatch.setattr(rna_validity, "run_subprocess", fake_run_subprocess)

    score = rna_validity._tm_score(Path("ref.pdb"), Path("pred.pdb"))

    assert score == 0.8123  # TM1, normalized by ref_pdb (chain1)
    assert "-mol" in captured_cmd and "RNA" in captured_cmd
    assert "-atom" in captured_cmd and " C1'" in captured_cmd
    assert Path(captured_cmd[0]).name == "TMalign"


def test_tm_score_returns_zero_when_no_data_row(monkeypatch: pytest.MonkeyPatch):
    class FakeResult:
        stdout = "#PDBchain1\tPDBchain2\tTM1\tTM2\n"

    monkeypatch.setattr(
        rna_validity, "run_subprocess", lambda cmd, **kwargs: FakeResult()
    )

    assert rna_validity._tm_score(Path("ref.pdb"), Path("pred.pdb")) == 0.0


def test_standardize_single_rna_chain_translates_na_mpnn_alphabet():
    assert rna_validity._standardize_single_rna_chain("bdhu") == "ACGU"


def test_standardize_single_rna_chain_rejects_multichain_design():
    with pytest.raises(ValueError, match="one generated RNA chain"):
        rna_validity._standardize_single_rna_chain("bd/hu")


def test_compute_validity_reports_coverage_and_does_not_count_oracle_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    backbones = [tmp_path / "ok.cif", tmp_path / "fail.cif"]
    for backbone in backbones:
        backbone.write_text("")

    class FakeInverseFolder:
        def design(self, pdb_path, output_dir):
            return InverseFoldResult(sequences=["bdhu"], scores=[0.0])

    class FakePredictor:
        def predict(self, fasta_path, output_dir):
            assert "ACGU" in fasta_path.read_text()
            if "fail" in str(output_dir):
                raise RuntimeError("oracle unavailable for this sample")
            output_dir.mkdir(parents=True, exist_ok=True)
            pdb_path = output_dir / "pred.pdb"
            pdb_path.write_text("")
            return [
                StructurePrediction(
                    sequence="ACGU",
                    pdb_path=pdb_path,
                    plddt=80.0,
                    ptm=float("nan"),
                )
            ]

    monkeypatch.setattr(rna_validity, "_tm_score", lambda *args, **kwargs: 0.8)

    aggregate, details, sequences = rna_validity.compute_validity_detailed(
        backbones,
        FakePredictor(),
        FakeInverseFolder(),
        tmp_path / "validity",
    )

    assert aggregate["validity"] == 1.0
    assert aggregate["evaluation_coverage"] == 0.5
    assert aggregate["evaluated_backbones"] == 1
    assert aggregate["failed_backbones"] == 1
    assert [detail["status"] for detail in details] == ["complete", "failed"]
    assert sequences == ["ACGU", "ACGU"]


# --------------------------------------------------------------------------
# diversity.py — compute_structural_diversity
# --------------------------------------------------------------------------


def test_compute_structural_diversity_counts_unique_clusters(tmp_path: Path):
    pdbs = [tmp_path / f"s{i}.pdb" for i in range(4)]
    for p in pdbs:
        p.write_text("")

    # s0/s1 share a cluster (repr s0); s2, s3 are singletons -> 3 clusters.
    cluster_results = [
        SearchResult(query_id="s0", hits=[], cluster_id="s0"),
        SearchResult(query_id="s1", hits=[], cluster_id="s0"),
        SearchResult(query_id="s2", hits=[], cluster_id="s2"),
        SearchResult(query_id="s3", hits=[], cluster_id="s3"),
    ]
    searcher = FakeStructureSearcher(cluster_results=cluster_results)

    result = compute_structural_diversity(pdbs, searcher, tmp_path / "out")

    assert result == {"structural_diversity": 0.75, "n_clusters": 3}


def test_compute_structural_diversity_empty_input_returns_zero(tmp_path: Path):
    result = compute_structural_diversity([], FakeStructureSearcher(), tmp_path)
    assert result == {"structural_diversity": 0.0, "n_clusters": 0}


def test_compute_structural_diversity_ignores_clusters_outside_wanted_set(
    tmp_path: Path,
):
    pdbs = [tmp_path / "s0.pdb"]
    pdbs[0].write_text("")

    # structure_searcher clustered an extra PDB (e.g. left over from a prior
    # call sharing the same parent dir) that isn't in `backbone_pdbs`.
    cluster_results = [
        SearchResult(query_id="s0", hits=[], cluster_id="s0"),
        SearchResult(query_id="stale", hits=[], cluster_id="stale"),
    ]
    searcher = FakeStructureSearcher(cluster_results=cluster_results)

    result = compute_structural_diversity(pdbs, searcher, tmp_path / "out")

    assert result == {"structural_diversity": 1.0, "n_clusters": 1}


# --------------------------------------------------------------------------
# diversity.py — compute_sequence_diversity (real nucleotide mode)
# --------------------------------------------------------------------------


def _fake_mmseqs_run(captured_cmds: list) -> callable:
    """Records every subprocess.run() call; on `createtsv`, writes a
    singleton-cluster clusters.tsv (one row per FASTA record — the real
    `sequences.fasta` this reads back was written for real, unmocked,
    before any subprocess call) so the real `tsv_path.read_text()` parsing
    step downstream has something valid to read, without needing to mock
    `Path.read_text` globally (which would affect every Path in the
    process, not just this one file)."""

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        if cmd[1] == "createtsv":
            tsv_path = Path(cmd[5])
            fasta_path = tsv_path.with_name("sequences.fasta")
            ids = [
                line[1:]
                for line in fasta_path.read_text().splitlines()
                if line.startswith(">")
            ]
            tsv_path.write_text("".join(f"{i}\t{i}\n" for i in ids))

    return fake_run


def test_compute_sequence_diversity_forces_nucleotide_dbtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured_cmds: list = []
    monkeypatch.setattr(
        protein_diversity.subprocess, "run", _fake_mmseqs_run(captured_cmds)
    )

    # NA-MPNN's own alphabet (lowercase b/d/h/u/y for RNA).
    compute_sequence_diversity(["hbdbdh"], tmp_path)

    createdb_cmd = captured_cmds[0]
    assert "--dbtype" in createdb_cmd
    assert createdb_cmd[createdb_cmd.index("--dbtype") + 1] == "2"


def test_compute_sequence_diversity_translates_alphabet_and_splits_chains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(protein_diversity.subprocess, "run", _fake_mmseqs_run([]))

    # A "/"-joined 2-chain design (e.g. a duplex) in NA-MPNN's own alphabet.
    compute_sequence_diversity(["bdh/act"], tmp_path)

    fasta_text = (tmp_path / "sequences.fasta").read_text()
    # Split into two separate FASTA records, both translated to standard
    # IUPAC nucleotide codes — no literal "/" (not a valid sequence char)
    # and no lowercase NA-MPNN-internal letters reach MMseqs2.
    assert "/" not in fasta_text
    assert "ACG" in fasta_text  # bdh -> ACG
    assert "ACT" in fasta_text  # act -> ACT
    assert fasta_text.count(">") == 2


def test_compute_sequence_diversity_caps_threads_for_tiny_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured_cmds: list = []
    monkeypatch.setattr(
        protein_diversity.subprocess, "run", _fake_mmseqs_run(captured_cmds)
    )
    monkeypatch.setattr(protein_diversity.os, "cpu_count", lambda: 244)

    compute_sequence_diversity(["bdh", "act"], tmp_path)

    cluster_cmd = captured_cmds[1]
    assert "--threads" in cluster_cmd
    threads = int(cluster_cmd[cluster_cmd.index("--threads") + 1])
    assert 1 <= threads <= 2  # capped to len(sequences), not the 244 visible cores


# --------------------------------------------------------------------------
# novelty.py — compute_novelty
# --------------------------------------------------------------------------


def test_compute_novelty_uses_best_hit_per_query(tmp_path: Path):
    pdbs = [tmp_path / "s0.pdb", tmp_path / "s1.pdb"]
    for p in pdbs:
        p.write_text("")

    search_results = [
        SearchResult(
            query_id="s0",
            hits=[
                {"target": "ref1", "tm_score": 0.9},
                {"target": "ref2", "tm_score": 0.3},
            ],
        ),
        SearchResult(query_id="s1", hits=[{"target": "ref1", "tm_score": 0.1}]),
    ]
    searcher = FakeStructureSearcher(search_results=search_results)

    result = compute_novelty(
        pdbs, searcher, tmp_path / "target_db", tmp_path / "out", tm_threshold=0.45
    )

    # s0's best hit (0.9) >= threshold -> not novel; s1's best hit (0.1) < threshold -> novel.
    assert result["novelty"] == pytest.approx((0.9 + 0.1) / 2)
    assert result["novel_rate"] == 0.5


def test_compute_novelty_empty_input_returns_nan_mean(tmp_path: Path):
    result = compute_novelty(
        [], FakeStructureSearcher(), tmp_path / "target_db", tmp_path / "out"
    )
    assert result["novel_rate"] == 0.0
    assert result["novelty"] != result["novelty"]  # nan


# --------------------------------------------------------------------------
# validity.py — configurable self-consistency atom (USalign -atom field)
# --------------------------------------------------------------------------


def test_usalign_atom_formats_pdb_field():
    """`_usalign_atom` must produce the 4-char PDB atom-name field USalign
    expects. The C1' default must stay byte-identical to the previously
    hardcoded ' C1'' so existing scTM numbers don't shift."""
    fmt = rna_validity._usalign_atom
    assert fmt("C1'") == " C1'"  # byte-identical to the old hardcoded default
    assert fmt("C4'") == " C4'"
    assert fmt("CA") == " CA "
    assert fmt("P") == " P  "
    assert rna_validity._DEFAULT_SC_ATOM == "C1'"


def test_compute_validity_resumes_complete_oracle_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    backbone = tmp_path / "sample.cif"
    backbone.write_text("")
    sample_out = tmp_path / "validity" / "sample"
    sample_out.mkdir(parents=True)
    (sample_out / "designed.fasta").write_text(">seq_0\nACGU\n")
    predicted = sample_out / "structures" / "0000" / "unrelaxed_model.pdb"
    predicted.parent.mkdir(parents=True)
    predicted.write_text("")

    class NoInverseFolder:
        num_sequences = 1

        def design(self, *args, **kwargs):
            pytest.fail("complete designed FASTA should be resumed")

    class NoPredictor:
        def predict(self, *args, **kwargs):
            pytest.fail("complete refold should be resumed")

    inverse_folder = NoInverseFolder()
    structure_predictor = NoPredictor()
    provenance = rna_validity._new_provenance(
        rna_validity._artifact_configuration(
            inverse_folder, structure_predictor, backbone
        )
    )
    provenance["sequences_complete"] = True
    provenance["predictions_complete"] = True
    rna_validity._write_json_atomic(sample_out / "provenance.json", provenance)

    monkeypatch.setattr(rna_validity, "_tm_score", lambda *args, **kwargs: 0.7)
    aggregate, details, sequences = rna_validity.compute_validity_detailed(
        [backbone], structure_predictor, inverse_folder, tmp_path / "validity"
    )

    assert aggregate["validity"] == 1.0
    assert aggregate["evaluation_coverage"] == 1.0
    assert details[0]["resumed_sequences"] is True
    assert details[0]["resumed_predictions"] is True
    assert sequences == ["ACGU"]


def test_compute_validity_recomputes_artifacts_when_provenance_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    backbone = tmp_path / "sample.cif"
    backbone.write_text("")
    sample_out = tmp_path / "validity" / "sample"
    sample_out.mkdir(parents=True)
    (sample_out / "designed.fasta").write_text(">seq_0\nACGU\n")
    predicted = sample_out / "structures" / "0000" / "unrelaxed_model.pdb"
    predicted.parent.mkdir(parents=True)
    predicted.write_text("")

    class TrackingInverseFolder:
        num_sequences = 1
        temperature = 0.1

        def __init__(self, seed):
            self.seed = seed
            self.calls = 0

        def design(self, *args, **kwargs):
            self.calls += 1
            return InverseFoldResult(sequences=["bdhu"], scores=[0.0])

    class TrackingPredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, fasta_path, output_dir):
            self.calls += 1
            output = output_dir / "0000" / "unrelaxed_model.pdb"
            output.parent.mkdir(parents=True)
            output.write_text("")
            return [
                StructurePrediction(
                    sequence="ACGU",
                    pdb_path=output,
                    plddt=80.0,
                    ptm=float("nan"),
                )
            ]

    stale_inverse = TrackingInverseFolder(seed=1)
    stale_predictor = TrackingPredictor()
    provenance = rna_validity._new_provenance(
        rna_validity._artifact_configuration(stale_inverse, stale_predictor, backbone)
    )
    provenance["sequences_complete"] = True
    provenance["predictions_complete"] = True
    rna_validity._write_json_atomic(sample_out / "provenance.json", provenance)

    current_inverse = TrackingInverseFolder(seed=2)
    current_predictor = TrackingPredictor()
    monkeypatch.setattr(rna_validity, "_tm_score", lambda *args, **kwargs: 0.7)
    _, details, _ = rna_validity.compute_validity_detailed(
        [backbone], current_predictor, current_inverse, tmp_path / "validity"
    )

    assert current_inverse.calls == 1
    assert current_predictor.calls == 1
    assert details[0]["resumed_sequences"] is False
    assert details[0]["resumed_predictions"] is False


def test_compute_novelty_rejects_missing_query_results(tmp_path: Path):
    pdbs = [tmp_path / "s0.pdb", tmp_path / "s1.pdb"]
    for path in pdbs:
        path.write_text("")
    searcher = FakeStructureSearcher(
        search_results=[
            SearchResult(query_id="s0", hits=[{"target": "ref", "tm_score": 0.2}])
        ]
    )

    with pytest.raises(RuntimeError, match="no result"):
        compute_novelty(pdbs, searcher, tmp_path / "db", tmp_path / "out")


def test_na_mpnn_backbone_seed_is_stable_and_sample_specific():
    from zevals.oracles.inverse_folders.na_mpnn import NAMPNNFolder

    folder = NAMPNNFolder(device="cpu", seed=123)
    first = folder._seed_for_backbone(Path("same_sample.cif"))
    assert first == folder._seed_for_backbone(Path("same_sample.cif"))
    assert first != folder._seed_for_backbone(Path("different_sample.cif"))
    assert first != NAMPNNFolder(device="cpu", seed=124)._seed_for_backbone(
        Path("same_sample.cif")
    )
