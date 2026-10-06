"""Unit tests for `oracles/structure_search/usalign.py` — output parsing
(pure functions, no subprocess) plus `search()`/`cluster()` with
`run_subprocess` mocked. Real binary verification (compiling TMalign/
qTMclust from source and running them against real RNA PDBs) was done
manually this session; not repeated here since it needs `g++` + network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.oracles.structure_search import usalign  # noqa: E402
from zevals.oracles.structure_search.usalign import USalignSearcher  # noqa: E402

_TMALIGN_STDOUT = """#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali
1EHZ.pdb\t4OQU.pdb\t0.2471\t0.2058\t3.86\t0.145\t0.093\t0.265\t62\t97\t34
1EHZ.pdb\t1EHZ_copy.pdb\t1.0000\t1.0000\t0.00\t1.000\t1.000\t1.000\t62\t62\t62
#Total CPU time is  0.01 seconds
"""

_QTMCLUST_OUTPUT = "4OQU.pdb\n1EHZ.pdb\tsimilar.pdb\n"


def test_parse_tmalign_output_groups_by_query_uses_tm2():
    results = usalign._parse_tmalign_output(_TMALIGN_STDOUT)
    by_id = {r.query_id: r for r in results}

    assert set(by_id) == {"1EHZ"}
    hits = {h["target"]: h["tm_score"] for h in by_id["1EHZ"].hits}
    # TM2 (normalized by the *target*'s length), not TM1.
    assert hits == {"4OQU": 0.2058, "1EHZ_copy": 1.0}
    first_hit = by_id["1EHZ"].hits[0]
    assert first_hit["query_tm_score"] == 0.2471
    assert first_hit["target_tm_score"] == 0.2058


def test_parse_tmalign_output_ignores_comment_and_blank_lines():
    stdout = "# header\n\n#Total CPU time is 0.01 seconds\n"
    assert usalign._parse_tmalign_output(stdout) == []


def test_parse_qtmclust_output_one_line_per_cluster(tmp_path: Path):
    output_path = tmp_path / "clusters.txt"
    output_path.write_text(_QTMCLUST_OUTPUT)

    results = usalign._parse_qtmclust_output(output_path)
    by_query = {r.query_id: r.cluster_id for r in results}

    # "4OQU.pdb" is its own (singleton) cluster; "1EHZ.pdb" and "similar.pdb"
    # share a cluster represented by 1EHZ.
    assert by_query == {"4OQU": "4OQU", "1EHZ": "1EHZ", "similar": "1EHZ"}


def test_parse_qtmclust_output_missing_file_returns_empty(tmp_path: Path):
    assert usalign._parse_qtmclust_output(tmp_path / "does_not_exist.txt") == []


def test_binary_in_dir_resolves_relative_to_binary_path_directory():
    assert Path(usalign._binary_in_dir(None, "TMalign")).name == "TMalign"
    assert usalign._binary_in_dir("/opt/usalign", "TMalign") == "/opt/usalign/TMalign"
    assert usalign._binary_in_dir("/opt/usalign", "qTMclust") == "/opt/usalign/qTMclust"


def test_search_hardcodes_rna_and_c1prime_atom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    query_dir = tmp_path / "query"
    target_dir = tmp_path / "target"
    query_dir.mkdir()
    target_dir.mkdir()
    (query_dir / "a.pdb").write_text("")
    (target_dir / "b.pdb").write_text("")

    captured_cmd = []

    class FakeResult:
        stdout = _TMALIGN_STDOUT

    def fake_run_subprocess(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return FakeResult()

    monkeypatch.setattr(usalign, "run_subprocess", fake_run_subprocess)

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    searcher.search(query_dir, target_dir, tmp_path / "out.tsv")

    assert "-mol" in captured_cmd and "RNA" in captured_cmd
    assert "-atom" in captured_cmd and " C1'" in captured_cmd
    assert Path(captured_cmd[0]).name == "TMalign"


def test_cluster_hardcodes_rna_and_c1prime_atom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    (structure_dir / "a.pdb").write_text("")

    captured_cmd = []

    def fake_run_subprocess(cmd, **kwargs):
        captured_cmd.extend(cmd)

    monkeypatch.setattr(usalign, "run_subprocess", fake_run_subprocess)

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    searcher.cluster(structure_dir, tmp_path / "out.txt", min_tm=0.45)

    assert "-mol" in captured_cmd and "RNA" in captured_cmd
    assert "-atom" in captured_cmd and " C1'" in captured_cmd
    assert Path(captured_cmd[0]).name == "qTMclust"


def test_search_empty_dirs_returns_empty_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []
    monkeypatch.setattr(
        usalign, "run_subprocess", lambda cmd, **kwargs: calls.append(cmd)
    )

    query_dir = tmp_path / "query"
    target_dir = tmp_path / "target"
    query_dir.mkdir()
    target_dir.mkdir()

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    result = searcher.search(query_dir, target_dir, tmp_path / "out.tsv")

    assert result == []
    assert calls == []


def test_search_globs_cif_files_alongside_pdb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """search() must pick up .cif query/target structures too, not just
    .pdb — it dispatches through glob_structures(), same as loaders.py."""
    query_dir = tmp_path / "query"
    target_dir = tmp_path / "target"
    query_dir.mkdir()
    target_dir.mkdir()
    (query_dir / "a.cif").write_text("")
    (target_dir / "b.cif").write_text("")

    class FakeResult:
        stdout = _TMALIGN_STDOUT

    def fake_run_subprocess(cmd, **kwargs):
        return FakeResult()

    monkeypatch.setattr(usalign, "run_subprocess", fake_run_subprocess)

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    searcher.search(query_dir, target_dir, tmp_path / "out.tsv")

    query_list = (tmp_path / "tmp" / "query_list_000.txt").read_text()
    target_list = (tmp_path / "tmp" / "target_list.txt").read_text()
    assert query_list == "a.cif\n"
    assert target_list == "b.cif\n"


def test_search_prefers_cif_over_pdb_for_same_stem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    query_dir = tmp_path / "query"
    target_dir = tmp_path / "target"
    query_dir.mkdir()
    target_dir.mkdir()
    (query_dir / "a.pdb").write_text("")
    (query_dir / "a.cif").write_text("")
    (target_dir / "b.pdb").write_text("")

    class FakeResult:
        stdout = _TMALIGN_STDOUT

    monkeypatch.setattr(usalign, "run_subprocess", lambda cmd, **kwargs: FakeResult())

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    searcher.search(query_dir, target_dir, tmp_path / "out.tsv")

    query_list = (tmp_path / "tmp" / "query_list_000.txt").read_text()
    assert query_list == "a.cif\n"


def test_cluster_globs_cif_files_alongside_pdb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    (structure_dir / "a.cif").write_text("")

    monkeypatch.setattr(usalign, "run_subprocess", lambda cmd, **kwargs: None)

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    searcher.cluster(structure_dir, tmp_path / "out.txt", min_tm=0.45)

    chain_list = (tmp_path / "tmp" / "chain_list.txt").read_text()
    assert chain_list == "a.cif\n"


def test_cluster_empty_dir_returns_empty_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []
    monkeypatch.setattr(
        usalign, "run_subprocess", lambda cmd, **kwargs: calls.append(cmd)
    )

    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()

    searcher = USalignSearcher(tmp_dir=str(tmp_path / "tmp"))
    result = searcher.cluster(structure_dir, tmp_path / "out.txt", min_tm=0.45)

    assert result == []
    assert calls == []
