"""Unit tests for the protein metric modules touched this session:
designability (Kabsch RMSD + per-sample error isolation + designable-subset
return), co-designability (pLDDT averaging + error isolation), novelty (the
mean_best_tm/novel_rate key split), structural diversity (the cluster-
membership-filtering bug fix), and secondary structure content.

Fake oracles (`FakeInverseFolder`, `FakeStructurePredictor`,
`FakeFoldSeekSearcher`) follow the same shape used for manual smoke testing
throughout this session's development, formalized into pytest fixtures.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.oracles.base import (  # noqa: E402
    InverseFoldResult,
    SearchResult,
    StructurePrediction,
)
from zevals.proteins.co_designability import compute_co_designability  # noqa: E402
from zevals.proteins.designability import (  # noqa: E402
    _atom_coords,
    _atom_rmsd,
    _kabsch_rmsd,
    compute_designability,
)
from zevals.proteins.diversity import (  # noqa: E402
    compute_structural_diversity,
    compute_tmalign_complete_linkage_diversity,
)
from zevals.proteins.novelty import compute_novelty  # noqa: E402
from zevals.proteins.secondary_structure import (  # noqa: E402
    compute_secondary_structure,
)


def _write_ca_pdb(path: Path, coords) -> None:
    """Write a minimal valid CA-only PDB via Biopython's own writer, so the
    fixed-width column format is guaranteed correct (no hand-rolled PDB text)."""
    from Bio.PDB import StructureBuilder
    from Bio.PDB.PDBIO import PDBIO

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("s")
    sb.init_model(0)
    sb.init_chain("A")
    sb.init_seg(" ")
    for i, (x, y, z) in enumerate(coords):
        sb.init_residue("ALA", " ", i + 1, " ")
        sb.init_atom("CA", (x, y, z), 0.0, 1.0, " ", "CA", i, element="C")
    structure = sb.get_structure()
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _write_ca_cif(path: Path, coords) -> None:
    """Same as `_write_ca_pdb`, but writes mmCIF via Biopython's `MMCIFIO`
    — used to test `_atom_coords`'s CIF-vs-PDB format dispatch."""
    from Bio.PDB import StructureBuilder
    from Bio.PDB.mmcifio import MMCIFIO

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("s")
    sb.init_model(0)
    sb.init_chain("A")
    sb.init_seg(" ")
    for i, (x, y, z) in enumerate(coords):
        sb.init_residue("ALA", " ", i + 1, " ")
        sb.init_atom("CA", (x, y, z), 0.0, 1.0, " ", "CA", i, element="C")
    structure = sb.get_structure()
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(path))


class FakeInverseFolder:
    """Returns a fixed sequence per backbone stem; can simulate a crash."""

    num_sequences = 1
    temperature = 0.1

    def __init__(
        self, seq_by_stem: dict[str, list[str]], raise_for: set[str] = frozenset()
    ):
        self._seq_by_stem = seq_by_stem
        self._raise_for = raise_for

    def design(self, pdb_path: Path, output_dir: Path) -> InverseFoldResult:
        if pdb_path.stem in self._raise_for:
            raise RuntimeError(f"simulated inverse-folding failure for {pdb_path.stem}")
        output_dir.mkdir(parents=True, exist_ok=True)
        seqs = self._seq_by_stem.get(pdb_path.stem, ["A"])
        return InverseFoldResult(sequences=seqs, scores=[0.1] * len(seqs))


class FakeStructurePredictor:
    """Each predict() call "refolds" every sequence in the batch to a copy
    of the next PDB in `refold_pdbs`, consumed in call order."""

    num_samples = 1

    def __init__(self, refold_pdbs: list[Path], plddt: float = 80.0):
        self._refold_pdbs = refold_pdbs
        self._call_idx = 0
        self._plddt = plddt

    def predict(self, fasta_path: Path, output_dir: Path) -> list[StructurePrediction]:
        output_dir.mkdir(parents=True, exist_ok=True)
        refold_pdb = self._refold_pdbs[self._call_idx]
        self._call_idx += 1
        seq_ids = [
            line[1:]
            for line in fasta_path.read_text().splitlines()
            if line.startswith(">")
        ]
        preds = []
        for seq_id in seq_ids:
            dst = output_dir / f"{seq_id}.pdb"
            shutil.copy(refold_pdb, dst)
            preds.append(
                StructurePrediction(
                    sequence="A", pdb_path=dst, plddt=self._plddt, ptm=0.8
                )
            )
        return preds


class FakeFoldSeekSearcher:
    def __init__(
        self,
        cluster_results: list[SearchResult] | None = None,
        search_results: list[SearchResult] | None = None,
    ):
        self._cluster_results = cluster_results or []
        self._search_results = search_results or []
        self.tmp_dir = Path("/tmp/fake_foldseek")

    def cluster(self, structure_dir, output_tsv, min_tm=0.5):
        return self._cluster_results

    def search(self, query_dir, target_db, output_tsv):
        return self._search_results


# --------------------------------------------------------------------------
# designability.py — Kabsch math
# --------------------------------------------------------------------------


def test_kabsch_rmsd_is_near_zero_after_exact_rotation():
    rng = np.random.default_rng(0)
    mobile = rng.normal(size=(20, 3))

    # Random proper rotation via QR.
    a = rng.normal(size=(3, 3))
    rot, _ = np.linalg.qr(a)
    if np.linalg.det(rot) < 0:
        rot[:, -1] *= -1
    target = mobile @ rot.T + rng.normal(size=(3,))  # rotate + translate

    assert _kabsch_rmsd(mobile, target) == pytest.approx(0.0, abs=1e-8)


def test_kabsch_rmsd_is_translation_invariant():
    rng = np.random.default_rng(1)
    mobile = rng.normal(size=(10, 3))
    target = mobile + np.array([100.0, -50.0, 3.0])  # pure translation, no rotation

    assert _kabsch_rmsd(mobile, target) == pytest.approx(0.0, abs=1e-8)


def test_kabsch_rmsd_is_positive_for_genuinely_different_shapes():
    rng = np.random.default_rng(2)
    mobile = rng.normal(size=(15, 3))
    target = rng.normal(size=(15, 3))  # unrelated point cloud

    assert _kabsch_rmsd(mobile, target) > 0.5


def test_kabsch_rmsd_returns_inf_for_nonfinite_coordinates():
    mobile = np.zeros((4, 3), dtype=float)
    mobile[0, 0] = np.nan
    target = np.zeros((4, 3), dtype=float)

    assert _kabsch_rmsd(mobile, target) == float("inf")


# --------------------------------------------------------------------------
# designability.py — PDB I/O
# --------------------------------------------------------------------------


def test_atom_rmsd_self_comparison_is_near_zero(tmp_path: Path):
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(8, 3))
    pdb = tmp_path / "structure.pdb"
    _write_ca_pdb(pdb, coords)

    assert _atom_rmsd(pdb, pdb) == pytest.approx(0.0, abs=1e-6)


def test_atom_rmsd_truncates_to_shorter_length_on_mismatch(tmp_path: Path):
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(10, 3))
    long_pdb = tmp_path / "long.pdb"
    short_pdb = tmp_path / "short.pdb"
    _write_ca_pdb(long_pdb, coords)
    _write_ca_pdb(short_pdb, coords[:6])  # first 6 residues only

    # Should not raise despite the length mismatch, and should be near-zero
    # since the first 6 residues are identical in both files.
    assert _atom_rmsd(long_pdb, short_pdb) == pytest.approx(0.0, abs=1e-6)
    assert len(_atom_coords(long_pdb)) == 10
    assert len(_atom_coords(short_pdb)) == 6


def test_atom_rmsd_returns_inf_for_empty_structure(tmp_path: Path):
    pdb = tmp_path / "empty.pdb"
    _write_ca_pdb(pdb, [])
    assert _atom_rmsd(pdb, pdb) == float("inf")


def test_atom_coords_reads_cif_same_as_pdb(tmp_path: Path):
    """`_atom_coords` must dispatch on `.suffix` — a CIF and a PDB encoding
    the exact same coordinates should read back identically."""
    rng = np.random.default_rng(5)
    coords = rng.normal(size=(8, 3))
    pdb = tmp_path / "structure.pdb"
    cif = tmp_path / "structure.cif"
    _write_ca_pdb(pdb, coords)
    _write_ca_cif(cif, coords)

    assert np.allclose(_atom_coords(pdb), _atom_coords(cif), atol=1e-2)


def test_atom_rmsd_cross_format_self_comparison_is_near_zero(tmp_path: Path):
    """The original backbone (now potentially CIF) vs. ESMFold's refold
    (always PDB) is exactly this cross-format comparison in real usage."""
    rng = np.random.default_rng(6)
    coords = rng.normal(size=(8, 3))
    cif = tmp_path / "original.cif"
    pdb = tmp_path / "refold.pdb"
    _write_ca_cif(cif, coords)
    _write_ca_pdb(pdb, coords)

    assert _atom_rmsd(cif, pdb) == pytest.approx(0.0, abs=1e-6)


def _write_two_atom_pdb(path: Path, ca_coords, alt_coords, alt_atom: str) -> None:
    """Write a PDB where each residue carries both a ``CA`` atom and a second
    ``alt_atom`` (e.g. ``C1'``) at distinct coordinates — to test that
    `_atom_coords`/`_atom_rmsd` select the requested atom."""
    from Bio.PDB import StructureBuilder
    from Bio.PDB.PDBIO import PDBIO

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("s")
    sb.init_model(0)
    sb.init_chain("A")
    sb.init_seg(" ")
    for i, (ca, alt) in enumerate(zip(ca_coords, alt_coords)):
        sb.init_residue("ALA", " ", i + 1, " ")
        sb.init_atom("CA", tuple(ca), 0.0, 1.0, " ", "CA", i * 2, element="C")
        sb.init_atom(
            alt_atom, tuple(alt), 0.0, 1.0, " ", alt_atom, i * 2 + 1, element="C"
        )
    io = PDBIO()
    io.set_structure(sb.get_structure())
    io.save(str(path))


def test_atom_coords_selects_requested_atom(tmp_path: Path):
    """`_atom_coords(atom=...)` must return the coordinates of the requested
    atom, not always CA — the knob backing configurable self-consistency."""
    rng = np.random.default_rng(11)
    ca = rng.normal(size=(6, 3))
    c1p = rng.normal(size=(6, 3))
    pdb = tmp_path / "two_atom.pdb"
    _write_two_atom_pdb(pdb, ca, c1p, "C1'")

    assert np.allclose(_atom_coords(pdb, "CA"), ca, atol=1e-2)
    assert np.allclose(_atom_coords(pdb, "C1'"), c1p, atol=1e-2)
    # Default stays CA (backwards compatible).
    assert np.allclose(_atom_coords(pdb), ca, atol=1e-2)


def test_atom_rmsd_differs_by_selected_atom(tmp_path: Path):
    """scRMSD on different atoms of the same structures gives different values,
    confirming the `atom` argument actually drives the computation."""
    rng = np.random.default_rng(12)
    ca_ref, ca_pred = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    c1_ref, c1_pred = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    ref = tmp_path / "ref.pdb"
    pred = tmp_path / "pred.pdb"
    _write_two_atom_pdb(ref, ca_ref, c1_ref, "C1'")
    _write_two_atom_pdb(pred, ca_pred, c1_pred, "C1'")

    rmsd_ca = _atom_rmsd(ref, pred, "CA")
    rmsd_c1 = _atom_rmsd(ref, pred, "C1'")
    assert rmsd_ca > 0 and rmsd_c1 > 0
    assert rmsd_ca != pytest.approx(rmsd_c1, abs=1e-6)


# --------------------------------------------------------------------------
# designability.py — compute_designability
# --------------------------------------------------------------------------


def test_compute_designability_splits_into_designable_and_not(tmp_path: Path):
    rng = np.random.default_rng(5)
    good_coords = rng.normal(size=(8, 3))
    bad_coords_source = rng.normal(size=(8, 3))  # unrelated shape

    good_pdb = tmp_path / "good.pdb"
    bad_pdb = tmp_path / "bad.pdb"
    _write_ca_pdb(good_pdb, good_coords)
    _write_ca_pdb(bad_pdb, bad_coords_source)

    # "good" refolds to an identical structure (scRMSD ~ 0, designable);
    # "bad" refolds to something unrelated (scRMSD large, not designable).
    refold_good = tmp_path / "refold_good.pdb"
    refold_bad = tmp_path / "refold_bad.pdb"
    _write_ca_pdb(refold_good, good_coords)
    _write_ca_pdb(refold_bad, rng.normal(size=(8, 3)) * 50)

    inverse_folder = FakeInverseFolder({"good": ["A"], "bad": ["A"]})
    structure_predictor = FakeStructurePredictor([refold_good, refold_bad])

    metrics, designable_pdbs = compute_designability(
        [good_pdb, bad_pdb],
        structure_predictor=structure_predictor,
        inverse_folder=inverse_folder,
        output_dir=tmp_path / "out",
        sc_rmsd_threshold=2.0,
    )

    assert designable_pdbs == [good_pdb]
    assert metrics["designability"] == pytest.approx(0.5)
    assert metrics["mean_best_plddt"] == pytest.approx(80.0)


def test_compute_designability_applies_plddt_threshold(tmp_path: Path):
    rng = np.random.default_rng(55)
    coords = rng.normal(size=(8, 3))
    pdb = tmp_path / "low_confidence.pdb"
    refold = tmp_path / "refold_low_confidence.pdb"
    _write_ca_pdb(pdb, coords)
    _write_ca_pdb(refold, coords)

    inverse_folder = FakeInverseFolder({"low_confidence": ["A"]})
    structure_predictor = FakeStructurePredictor([refold], plddt=60.0)

    metrics, designable_pdbs = compute_designability(
        [pdb],
        structure_predictor=structure_predictor,
        inverse_folder=inverse_folder,
        output_dir=tmp_path / "out",
        sc_rmsd_threshold=2.0,
        plddt_threshold=70.0,
    )

    assert designable_pdbs == []
    assert metrics["designability"] == 0.0
    assert metrics["mean_best_plddt"] == pytest.approx(60.0)


def test_compute_designability_isolates_per_backbone_crash(tmp_path: Path):
    rng = np.random.default_rng(6)
    coords = rng.normal(size=(8, 3))
    ok_pdb = tmp_path / "ok.pdb"
    crash_pdb = tmp_path / "crash.pdb"
    _write_ca_pdb(ok_pdb, coords)
    _write_ca_pdb(crash_pdb, coords)

    refold_ok = tmp_path / "refold_ok.pdb"
    _write_ca_pdb(refold_ok, coords)

    inverse_folder = FakeInverseFolder({"ok": ["A"]}, raise_for={"crash"})
    structure_predictor = FakeStructurePredictor([refold_ok])

    # Must not raise despite `crash.pdb` triggering an exception inside design().
    metrics, designable_pdbs = compute_designability(
        [crash_pdb, ok_pdb],
        structure_predictor=structure_predictor,
        inverse_folder=inverse_folder,
        output_dir=tmp_path / "out",
        sc_rmsd_threshold=2.0,
    )

    assert designable_pdbs == [ok_pdb]
    assert metrics["designability"] == pytest.approx(0.5)


def test_compute_designability_empty_sequences_counts_as_non_designable(tmp_path: Path):
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(5, 3))
    pdb = tmp_path / "backbone.pdb"
    _write_ca_pdb(pdb, coords)

    inverse_folder = FakeInverseFolder({"backbone": []})  # no sequences returned
    structure_predictor = FakeStructurePredictor([])

    metrics, designable_pdbs = compute_designability(
        [pdb],
        structure_predictor=structure_predictor,
        inverse_folder=inverse_folder,
        output_dir=tmp_path / "out",
    )

    assert designable_pdbs == []
    assert metrics["designability"] == 0.0
    assert metrics["mean_min_scRMSD"] == float("inf")


# --------------------------------------------------------------------------
# co_designability.py
# --------------------------------------------------------------------------


def test_compute_co_designability_averages_plddt_over_passing_backbones_only(
    tmp_path: Path,
):
    rng = np.random.default_rng(8)
    coords = rng.normal(size=(6, 3))
    pass_pdb = tmp_path / "pass.pdb"
    fail_pdb = tmp_path / "fail.pdb"
    _write_ca_pdb(pass_pdb, coords)
    _write_ca_pdb(fail_pdb, coords)

    refold_pass = tmp_path / "refold_pass.pdb"
    _write_ca_pdb(refold_pass, coords)  # near-identical -> struct_ok
    refold_fail = tmp_path / "refold_fail.pdb"
    _write_ca_pdb(refold_fail, rng.normal(size=(6, 3)) * 50)  # far off -> not struct_ok

    inverse_folder = FakeInverseFolder({"pass": ["A"], "fail": ["A"]})
    structure_predictor = FakeStructurePredictor([refold_pass, refold_fail], plddt=90.0)

    metrics = compute_co_designability(
        [pass_pdb, fail_pdb],
        structure_predictor=structure_predictor,
        inverse_folder=inverse_folder,
        output_dir=tmp_path / "out",
        sc_rmsd_threshold=2.0,
    )

    assert metrics["co_designability"] == pytest.approx(0.5)
    # pLDDT is only averaged over the co-designable backbone ("pass"), even
    # though both fake predictions report the same plddt=90 — the "fail"
    # backbone must not contribute to (or dilute) the average.
    assert metrics["mean_plddt_co_designable"] == pytest.approx(90.0)


def test_compute_co_designability_isolates_per_backbone_crash(tmp_path: Path):
    rng = np.random.default_rng(9)
    coords = rng.normal(size=(6, 3))
    ok_pdb = tmp_path / "ok.pdb"
    crash_pdb = tmp_path / "crash.pdb"
    _write_ca_pdb(ok_pdb, coords)
    _write_ca_pdb(crash_pdb, coords)

    refold_ok = tmp_path / "refold_ok.pdb"
    _write_ca_pdb(refold_ok, coords)

    inverse_folder = FakeInverseFolder({"ok": ["A"]}, raise_for={"crash"})
    structure_predictor = FakeStructurePredictor([refold_ok])

    metrics = compute_co_designability(
        [crash_pdb, ok_pdb],
        structure_predictor=structure_predictor,
        inverse_folder=inverse_folder,
        output_dir=tmp_path / "out",
    )

    assert metrics["co_designability"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# novelty.py — mean_best_tm ("novelty") vs threshold fraction ("novel_rate")
# --------------------------------------------------------------------------


def test_compute_novelty_splits_primary_and_secondary_keys(tmp_path: Path):
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    a.touch()
    b.touch()

    searcher = FakeFoldSeekSearcher(
        search_results=[
            SearchResult(
                query_id="a", hits=[{"target": "x", "tm_score": 0.9, "evalue": 0.0}]
            ),
            SearchResult(
                query_id="b", hits=[{"target": "y", "tm_score": 0.1, "evalue": 0.0}]
            ),
        ]
    )

    metrics = compute_novelty(
        [a, b],
        foldseek=searcher,
        target_db=tmp_path / "db",
        output_dir=tmp_path / "out",
        tm_threshold=0.5,
    )

    assert metrics["novelty"] == pytest.approx((0.9 + 0.1) / 2)
    assert metrics["novel_rate"] == pytest.approx(0.5)  # only "b" is below threshold


def test_compute_novelty_missing_hits_default_to_zero_tm(tmp_path: Path):
    a = tmp_path / "a.pdb"
    a.touch()
    searcher = FakeFoldSeekSearcher(search_results=[])  # no hits found for "a" at all

    metrics = compute_novelty(
        [a], foldseek=searcher, target_db=tmp_path / "db", output_dir=tmp_path / "out"
    )

    assert metrics["novelty"] == pytest.approx(0.0)
    assert metrics["novel_rate"] == pytest.approx(1.0)


def test_compute_novelty_empty_input_returns_nan_novelty():
    metrics = compute_novelty(
        [],
        foldseek=FakeFoldSeekSearcher(),
        target_db=Path("db"),
        output_dir=Path("out"),
    )
    assert metrics["novelty"] != metrics["novelty"]  # NaN != NaN
    assert metrics["novel_rate"] == 0.0


# --------------------------------------------------------------------------
# diversity.py — cluster-membership filtering bug fix
# --------------------------------------------------------------------------


def test_structural_diversity_only_counts_clusters_among_designable_subset(
    tmp_path: Path,
):
    """Regression test: FoldSeek clusters every PDB in the shared parent
    directory, which may include non-designable backbones — n_clusters must
    be filtered down to `backbone_pdbs` before counting, or the numerator
    (clusters) and denominator (len(backbone_pdbs)) describe different
    populations."""
    designable_a = tmp_path / "designable_a.pdb"
    designable_b = tmp_path / "designable_b.pdb"
    not_designable = tmp_path / "not_designable.pdb"
    for p in (designable_a, designable_b, not_designable):
        p.touch()

    # FoldSeek clustered the whole directory (3 files -> 3 singleton
    # clusters), but only 2 of those files are in the designable subset
    # we're asking about.
    searcher = FakeFoldSeekSearcher(
        cluster_results=[
            SearchResult(query_id="designable_a", hits=[], cluster_id="designable_a"),
            SearchResult(query_id="designable_b", hits=[], cluster_id="designable_b"),
            SearchResult(
                query_id="not_designable", hits=[], cluster_id="not_designable"
            ),
        ]
    )

    metrics = compute_structural_diversity(
        [designable_a, designable_b], foldseek=searcher, output_dir=tmp_path / "out"
    )

    assert metrics["n_clusters"] == 2
    assert metrics["structural_diversity"] == pytest.approx(
        1.0
    )  # 2 clusters / 2 samples


def test_structural_diversity_empty_input():
    metrics = compute_structural_diversity(
        [], foldseek=FakeFoldSeekSearcher(), output_dir=Path("out")
    )
    assert metrics == {"structural_diversity": 0.0, "n_clusters": 0}


def test_tmalign_complete_linkage_diversity_uses_min_symmetrized_tm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    c = tmp_path / "c.pdb"
    for path in (a, b, c):
        path.touch()

    def fake_run_subprocess(cmd, **kwargs):
        assert cmd[0] == "TMalign"
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        stdout = "\n".join(
            [
                "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD",
                "sample_00000_a.pdb\tsample_00001_b.pdb\t0.90\t0.80\t1.0",
                "sample_00000_a.pdb\tsample_00002_c.pdb\t0.40\t0.30\t2.0",
                "sample_00001_b.pdb\tsample_00002_c.pdb\t0.70\t0.20\t3.0",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("zevals.proteins.diversity.run_subprocess", fake_run_subprocess)

    metrics = compute_tmalign_complete_linkage_diversity(
        [a, b, c],
        output_dir=tmp_path / "out",
        tm_thresholds=[0.0, 0.5, 0.85, 1.0],
    )

    # Min-symmetrized pair TMs are AB=0.8, AC=0.3, BC=0.2. Complete linkage
    # merges AB first, then C joins that cluster only at TM <= 0.2.
    assert metrics["tmalign_n_clusters_at_0.5"] == pytest.approx(2.0)
    assert metrics["tmalign_cluster_fraction_at_0.5"] == pytest.approx(2 / 3)
    assert metrics["tmalign_mean_pairwise_tm"] == pytest.approx((0.8 + 0.3 + 0.2) / 3)
    assert metrics["tmalign_median_pairwise_tm"] == pytest.approx(0.3)
    assert metrics["tmalign_diversity_auc"] == pytest.approx(0.6916666666666667)
    assert metrics["tmalign_n_missing_pair_scores"] == pytest.approx(0.0)

    curve = (tmp_path / "out" / "tmalign_complete_linkage_curve.csv").read_text()
    assert "0.5,2,0.666667" in curve


def test_tmalign_complete_linkage_diversity_empty_input(tmp_path: Path):
    metrics = compute_tmalign_complete_linkage_diversity([], output_dir=tmp_path)

    assert metrics["tmalign_diversity_auc"] == 0.0
    assert metrics["tmalign_n_structures"] == 0.0


# --------------------------------------------------------------------------
# secondary_structure.py
# --------------------------------------------------------------------------


def test_secondary_structure_fractions_sum_to_one(tmp_path: Path):
    """Regression test: fractions must be normalized by classified (a+b+c)
    residues, not total residue count — P-SEA leaves chain-break/non-peptide
    residues unclassified ('')."""
    rng = np.random.default_rng(10)
    # A helix-like structure needs more than a handful of CA atoms to get
    # any P-SEA classification at all; a plain random cloud may classify
    # everything as coil, which is fine — we're testing normalization, not
    # the P-SEA algorithm itself.
    coords = np.cumsum(rng.normal(scale=3.8, size=(30, 3)), axis=0)
    pdb = tmp_path / "backbone.pdb"
    _write_ca_pdb(pdb, coords)

    metrics = compute_secondary_structure([pdb])

    total = metrics["alpha_frac"] + metrics["beta_frac"] + metrics["coil_frac"]
    assert total == pytest.approx(1.0)


def test_secondary_structure_skips_unparseable_files_without_raising(tmp_path: Path):
    bogus = tmp_path / "not_a_pdb.pdb"
    bogus.write_text("this is not valid PDB content")

    metrics = compute_secondary_structure([bogus])

    assert metrics["alpha_frac"] != metrics["alpha_frac"]  # NaN
