"""Unit tests for `zevals/setup.py` — the `zevals setup` backend.
`subprocess.run` is always mocked here; these tests never actually download
a binary (see `scripts/install_foldseek.sh`/`install_mmseqs.sh`/
`install_usalign.sh` themselves, which were verified by actually running
them this session)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals import setup as zevals_setup  # noqa: E402


def test_resolve_scripts_dir_finds_real_scripts_directory():
    """Not mocked — this should resolve to the actual repo's scripts/ dir
    (source-checkout branch) since that's the environment tests run in."""
    scripts_dir = zevals_setup._resolve_scripts_dir()
    assert (scripts_dir / "install_foldseek.sh").is_file()
    assert (scripts_dir / "install_mmseqs.sh").is_file()
    assert (scripts_dir / "install_usalign.sh").is_file()


def test_install_defaults_to_all_three_tools(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd) or None)

    zevals_setup.install()

    assert len(calls) == 3
    assert any("install_foldseek.sh" in c[1] for c in calls)
    assert any("install_mmseqs.sh" in c[1] for c in calls)
    assert any("install_usalign.sh" in c[1] for c in calls)
    assert all(c[0] == "bash" for c in calls)


def test_install_respects_check_true(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs = {}

    def fake_run(cmd, check):
        captured_kwargs["check"] = check
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    zevals_setup.install(tools=["foldseek"])
    assert captured_kwargs["check"] is True


def test_install_can_select_single_tool(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))

    zevals_setup.install(tools=["mmseqs"])

    assert len(calls) == 1
    assert "install_mmseqs.sh" in calls[0][1]


def test_install_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        zevals_setup.install(tools=["not_a_real_tool"])


def test_install_passes_install_dir_and_force(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))

    zevals_setup.install(tools=["foldseek"], install_dir="/custom/dir", force=True)

    assert calls[0] == [
        "bash",
        calls[0][1],
        "/custom/dir",
        "--force",
    ]


def test_install_force_without_install_dir_still_passes_force(
    monkeypatch: pytest.MonkeyPatch,
):
    """Force needs an explicit install_dir positional before it in the
    script's argv convention — verify the fallback default is supplied."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))

    zevals_setup.install(tools=["foldseek"], force=True)

    assert calls[0][2] == str(Path.home() / ".local" / "bin")
    assert calls[0][3] == "--force"


def test_resolve_scripts_dir_finds_build_foldseek_db_script():
    """Not mocked — the new database-build script must ship alongside the
    existing install scripts."""
    scripts_dir = zevals_setup._resolve_scripts_dir()
    assert (scripts_dir / "build_foldseek_db.sh").is_file()


def test_build_database_invokes_script_with_db_name_and_output_dir(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))

    result = zevals_setup.build_database("PDB", "/persistent/foldseek_dbs")

    assert calls[0][0] == "bash"
    assert "build_foldseek_db.sh" in calls[0][1]
    assert calls[0][2:] == ["PDB", "/persistent/foldseek_dbs"]
    assert result == "/persistent/foldseek_dbs/PDB/db"


def test_build_database_passes_force_flag(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))

    zevals_setup.build_database("PDB", "/persistent/foldseek_dbs", force=True)

    assert calls[0][-1] == "--force"


def test_build_database_slugifies_db_name_with_slashes_and_spaces(
    monkeypatch: pytest.MonkeyPatch,
):
    """Matches build_foldseek_db.sh's `tr '/ ' '__'` slugification exactly —
    both `/` and ` ` become `_`."""
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: None)

    result = zevals_setup.build_database(
        "Alphafold/UniProt50-minimal", "/persistent/foldseek_dbs"
    )

    assert result == "/persistent/foldseek_dbs/Alphafold_UniProt50-minimal/db"


def test_default_checkpoint_dir_respects_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")
    assert zevals_setup.default_checkpoint_dir() == Path(
        "/custom/cache/zevals/checkpoints"
    )


def test_default_checkpoint_dir_falls_back_to_home_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert zevals_setup.default_checkpoint_dir() == (
        Path.home() / ".cache" / "zevals" / "checkpoints"
    )


def _fake_urlretrieve(calls: list) -> callable:
    """Records (url, dest) and, like the real `urlretrieve`, actually
    creates a file at `dest` — `download_checkpoint` now downloads to a
    `.part` sibling and renames it into place on success (see its own
    docstring), so a mock that doesn't create a real file would make that
    rename fail."""

    def fake(url, dest):
        calls.append((url, dest))
        Path(dest).write_bytes(b"fake checkpoint bytes")

    return fake


def test_download_checkpoint_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        zevals_setup.download_checkpoint("not_a_real_checkpoint")


def test_download_checkpoint_fetches_to_expected_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list = []
    monkeypatch.setattr(
        zevals_setup.urllib.request, "urlretrieve", _fake_urlretrieve(calls)
    )

    result = zevals_setup.download_checkpoint(
        "protein_mpnn_ca", output_dir=str(tmp_path)
    )

    assert result == str(tmp_path / "protein_mpnn" / "ca_model_weights")
    assert len(calls) == 1
    url, dest = calls[0]
    assert (
        url
        == "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/ca_model_weights/v_48_020.pt"
    )
    # Downloaded to a .part sibling, not the final path directly.
    assert dest == tmp_path / "protein_mpnn" / "ca_model_weights" / "v_48_020.pt.part"
    final_path = tmp_path / "protein_mpnn" / "ca_model_weights" / "v_48_020.pt"
    assert final_path.is_file()
    assert not final_path.with_suffix(final_path.suffix + ".part").exists()


def test_download_checkpoint_ligand_mpnn_returns_direct_file_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Unlike the ProteinMPNN-family checkpoints (a weights_dir), LigandMPNN's
    config field is a direct file path — the returned path must match."""
    monkeypatch.setattr(
        zevals_setup.urllib.request, "urlretrieve", _fake_urlretrieve([])
    )

    result = zevals_setup.download_checkpoint("ligand_mpnn", output_dir=str(tmp_path))

    assert result == str(tmp_path / "ligand_mpnn" / "ligandmpnn_v_32_010_25.pt")
    assert Path(result).is_file()


def test_download_checkpoint_na_mpnn_design_returns_direct_file_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """NAMPNNFolder's `checkpoint_path` field is also a direct file path
    (like LigandMPNN, unlike the ProteinMPNN-family weights_dir style)."""
    monkeypatch.setattr(
        zevals_setup.urllib.request, "urlretrieve", _fake_urlretrieve([])
    )

    result = zevals_setup.download_checkpoint(
        "na_mpnn_design", output_dir=str(tmp_path)
    )

    assert result == str(tmp_path / "na_mpnn" / "s_19137.pt")
    assert Path(result).is_file()


def test_download_checkpoint_rhofold_returns_direct_file_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        zevals_setup.urllib.request, "urlretrieve", _fake_urlretrieve([])
    )

    result = zevals_setup.download_checkpoint("rhofold", output_dir=str(tmp_path))

    assert result == str(tmp_path / "rhofold" / "RhoFold_pretrained.pt")
    assert Path(result).is_file()


def test_download_checkpoint_skips_existing_file_unless_forced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list = []
    monkeypatch.setattr(
        zevals_setup.urllib.request, "urlretrieve", _fake_urlretrieve(calls)
    )
    dest_file = tmp_path / "protein_mpnn" / "ca_model_weights" / "v_48_020.pt"
    dest_file.parent.mkdir(parents=True)
    dest_file.write_bytes(b"already here")

    zevals_setup.download_checkpoint("protein_mpnn_ca", output_dir=str(tmp_path))
    assert calls == []  # skipped, already exists

    zevals_setup.download_checkpoint(
        "protein_mpnn_ca", output_dir=str(tmp_path), force=True
    )
    assert len(calls) == 1  # force re-downloads
    assert calls[0][1] == dest_file.with_suffix(dest_file.suffix + ".part")


def test_download_checkpoint_cleans_up_part_file_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def failing_urlretrieve(url, dest):
        Path(dest).write_bytes(b"truncated")
        raise ConnectionError("simulated network drop")

    monkeypatch.setattr(zevals_setup.urllib.request, "urlretrieve", failing_urlretrieve)

    with pytest.raises(ConnectionError):
        zevals_setup.download_checkpoint("protein_mpnn_ca", output_dir=str(tmp_path))

    dest_dir = tmp_path / "protein_mpnn" / "ca_model_weights"
    # Neither the corrupt .part file nor a (nonexistent) final file should
    # be left behind — a retry must see a clean slate, not a fake "already
    # downloaded" state.
    assert not dest_dir.exists() or list(dest_dir.iterdir()) == []


def test_download_checkpoint_defaults_to_default_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(zevals_setup, "default_checkpoint_dir", lambda: tmp_path)
    monkeypatch.setattr(
        zevals_setup.urllib.request, "urlretrieve", _fake_urlretrieve([])
    )

    result = zevals_setup.download_checkpoint("ligand_mpnn")

    assert result == str(tmp_path / "ligand_mpnn" / "ligandmpnn_v_32_010_25.pt")
