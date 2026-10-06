"""Unit tests for `zevals/cli.py` — argument parsing and subcommand
dispatch. `ZevalSuite`/`zevals.setup.install` are always mocked here;
`ZevalSuite` itself is tested in `test_suite.py`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals import cli  # noqa: E402


def test_split_kv_splits_on_first_occurrence_only():
    assert cli._split_kv("proteins=outputs/backbones", sep="=") == (
        "proteins",
        "outputs/backbones",
    )
    # A value containing the separator must not be truncated.
    assert cli._split_kv("proteins=outputs/run=1/backbones", sep="=") == (
        "proteins",
        "outputs/run=1/backbones",
    )


def test_split_kv_raises_when_separator_missing():
    with pytest.raises(ValueError, match="Expected"):
        cli._split_kv("no_separator_here", sep="=")


def test_evaluate_subcommand_builds_samples_dirs_and_calls_suite(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    class FakeSuite:
        def __init__(self, modality_overrides=None, results_dir=None):
            captured["modality_overrides"] = modality_overrides
            captured["results_dir"] = results_dir

        def evaluate(self, samples_dirs):
            captured["samples_dirs"] = samples_dirs

    monkeypatch.setattr(cli, "ZevalSuite", FakeSuite)

    cli.main(
        [
            "evaluate",
            "--modality",
            "proteins=outputs/backbones",
            "--modality",
            "molecules=outputs/ligands",
            "--override",
            "proteins:oracle/inverse_folder=soluble_mpnn",
            "--results-dir",
            "outputs/eval",
        ]
    )

    assert captured["samples_dirs"] == {
        "proteins": "outputs/backbones",
        "molecules": "outputs/ligands",
    }
    assert captured["modality_overrides"] == {
        "proteins": ["oracle/inverse_folder=soluble_mpnn"]
    }
    assert captured["results_dir"] == "outputs/eval"


def test_evaluate_subcommand_requires_at_least_one_modality(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(SystemExit):
        cli.main(["evaluate"])


def test_evaluate_subcommand_defaults_have_no_overrides_or_results_dir(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    class FakeSuite:
        def __init__(self, modality_overrides=None, results_dir=None):
            captured["modality_overrides"] = modality_overrides
            captured["results_dir"] = results_dir

        def evaluate(self, samples_dirs):
            captured["samples_dirs"] = samples_dirs

    monkeypatch.setattr(cli, "ZevalSuite", FakeSuite)

    cli.main(["evaluate", "--modality", "proteins=outputs/backbones"])

    assert captured["modality_overrides"] == {}
    assert captured["results_dir"] is None


def test_evaluate_subcommand_passes_ground_truth_dirs(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    class FakeSuite:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, samples_dirs, ground_truth_dirs=None):
            captured["samples_dirs"] = samples_dirs
            captured["ground_truth_dirs"] = ground_truth_dirs

    monkeypatch.setattr(cli, "ZevalSuite", FakeSuite)

    cli.main(
        [
            "evaluate",
            "--modality",
            "molecules=outputs/conformers",
            "--ground-truth",
            "molecules=references/conformers",
        ]
    )

    assert captured == {
        "samples_dirs": {"molecules": "outputs/conformers"},
        "ground_truth_dirs": {"molecules": "references/conformers"},
    }


def test_setup_subcommand_calls_install_with_parsed_args(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_install(tools=None, install_dir=None, force=False):
        captured["tools"] = tools
        captured["install_dir"] = install_dir
        captured["force"] = force

    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", fake_install)

    cli.main(["setup", "--tools", "foldseek,mmseqs", "--install-dir", "/x", "--force"])

    assert captured == {
        "tools": ["foldseek", "mmseqs"],
        "install_dir": "/x",
        "force": True,
    }


def test_setup_subcommand_defaults_to_no_explicit_tools(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_install(tools=None, install_dir=None, force=False):
        captured["tools"] = tools

    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", fake_install)

    cli.main(["setup"])

    assert captured["tools"] is None


def test_no_command_exits_with_error():
    with pytest.raises(SystemExit):
        cli.main([])


def test_setup_subcommand_does_not_build_db_by_default(monkeypatch: pytest.MonkeyPatch):
    build_db_calls = []

    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", lambda **kw: None)
    monkeypatch.setattr(
        zevals_setup_module, "build_database", lambda *a, **kw: build_db_calls.append(a)
    )

    cli.main(["setup"])

    assert build_db_calls == []


def test_setup_subcommand_builds_db_after_install_when_requested(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", lambda **kw: None)

    def fake_build_database(db_name, db_dir, force=False):
        captured["db_name"] = db_name
        captured["db_dir"] = db_dir
        captured["force"] = force
        return f"{db_dir}/{db_name}/db"

    monkeypatch.setattr(zevals_setup_module, "build_database", fake_build_database)

    cli.main(
        [
            "setup",
            "--build-db",
            "PDB",
            "--db-dir",
            "/persistent/foldseek_dbs",
            "--force",
        ]
    )

    assert captured == {
        "db_name": "PDB",
        "db_dir": "/persistent/foldseek_dbs",
        "force": True,
    }


def test_setup_subcommand_build_db_without_db_dir_exits_with_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", lambda **kw: None)
    monkeypatch.setattr(zevals_setup_module, "build_database", lambda *a, **kw: None)

    with pytest.raises(SystemExit):
        cli.main(["setup", "--build-db", "PDB"])


def test_setup_subcommand_does_not_download_checkpoints_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_calls = []

    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", lambda **kw: None)
    monkeypatch.setattr(
        zevals_setup_module,
        "download_checkpoint",
        lambda *a, **kw: checkpoint_calls.append(a),
    )

    cli.main(["setup"])

    assert checkpoint_calls == []


def test_setup_subcommand_downloads_requested_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = []

    import zevals.setup as zevals_setup_module

    monkeypatch.setattr(zevals_setup_module, "install", lambda **kw: None)

    def fake_download_checkpoint(name, output_dir=None, force=False):
        captured.append((name, output_dir, force))
        return f"/fake/{name}"

    monkeypatch.setattr(
        zevals_setup_module, "download_checkpoint", fake_download_checkpoint
    )

    cli.main(
        [
            "setup",
            "--checkpoints",
            "protein_mpnn_ca,ligand_mpnn",
            "--checkpoint-dir",
            "/custom/checkpoints",
            "--force",
        ]
    )

    assert captured == [
        ("protein_mpnn_ca", "/custom/checkpoints", True),
        ("ligand_mpnn", "/custom/checkpoints", True),
    ]
