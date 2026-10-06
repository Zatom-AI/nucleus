"""Tests for the current paper checkpoint registry and Hub downloads."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from nucleus.inference_engines.checkpoint_registry import (
    REGISTERED_CHECKPOINTS,
    RegisteredCheckpoint,
)
from nucleus_cli import download_checkpoints

MANIFEST_PATH = Path(__file__).parent / "fixtures" / "zatom2_hub_checkpoints.json"


def test_paper_registry_matches_verified_hub_inventory() -> None:
    """Cover every backed-up file, using metadata captured from the Hub."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    paper = {
        name: checkpoint
        for name, checkpoint in REGISTERED_CHECKPOINTS.items()
        if checkpoint.repo_id == manifest["repo_id"] and name != "zatom2"
    }
    assert len(paper) == len(manifest["checkpoints"]) == 27
    assert {checkpoint.repo_filename for checkpoint in paper.values()} == set(
        manifest["checkpoints"]
    )
    for name, checkpoint in paper.items():
        remote = manifest["checkpoints"][checkpoint.repo_filename]
        assert checkpoint.sha256 == remote["sha256"]
        assert checkpoint.url is None
        assert re.fullmatch(r"[0-9a-f]{64}", checkpoint.sha256)
        # CLI identifiers and Hub directories must use the same table/figure.
        assert (
            name.split("-")[1]
            == Path(checkpoint.repo_filename).parent.name.split("-")[1]
        )

    filenames = [checkpoint.filename for checkpoint in paper.values()]
    assert len(filenames) == len(set(filenames))
    assert all(Path(filename).name == filename for filename in filenames)
    assert REGISTERED_CHECKPOINTS["zatom2"] is REGISTERED_CHECKPOINTS[
        "zatom2-table1-zatom2-l-joint-properties-ablation"
    ]
    assert "zatom2p" not in REGISTERED_CHECKPOINTS
    assert "zatom2n" not in REGISTERED_CHECKPOINTS


@pytest.mark.parametrize(
    ("name", "step"),
    [
        ("zatom2-table3-scope-no-pretraining", 25000),
        ("zatom2-table3-scope-from-joint-force-conditioned", 120000),
        ("zatom2-table3-scope-from-joint-properties-force-conditioned", 130000),
        ("zatom2-table3-scope-from-joint-without-force-conditioning", 125000),
    ],
)
def test_protein_identifiers_install_selected_checkpoints(
    name: str, step: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI must fetch each final protein selection, including scratch at 25k."""
    payload = b"paper-ready-protein-checkpoint"
    checkpoint = REGISTERED_CHECKPOINTS[name]
    remote_filename = (
        f"paper-ready-checkpoints/paper-{name.removeprefix('zatom2-')}/"
        f"step-{step:08d}.ckpt"
    )
    calls = []

    def fake_hf_hub_download(**kwargs) -> str:
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)

    monkeypatch.setattr(checkpoint, "sha256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        download_checkpoints, "_resolve_checkpoint_dirs", lambda _: [tmp_path]
    )
    result = CliRunner().invoke(download_checkpoints.app, ["install", name])

    assert result.exit_code == 0, result.output
    assert (tmp_path / checkpoint.filename).read_bytes() == payload
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "Zatom-AI/Zatom-2"
    assert calls[0]["filename"] == remote_filename


@pytest.mark.parametrize("selection", ["all", "base-models"])
def test_install_groups_resolve_current_registry(
    selection: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Group installs should retain base shortcuts and include new paper entries."""
    calls = []
    monkeypatch.setattr(
        download_checkpoints, "_resolve_checkpoint_dirs", lambda _: [tmp_path]
    )
    monkeypatch.setattr(
        download_checkpoints,
        "install_model",
        lambda name, directory, force: calls.append((name, directory, force)),
    )
    result = CliRunner().invoke(download_checkpoints.app, ["install", selection])
    assert result.exit_code == 0, result.output
    expected = (
        [name for name in REGISTERED_CHECKPOINTS if name != "zatom2"]
        if selection == "all"
        else ["zatom2"]
    )
    assert calls == [(name, tmp_path, False) for name in expected]


def test_list_available_and_examples_use_current_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Help and discovery must advertise installable IDs after renumbering."""
    monkeypatch.setattr(
        download_checkpoints, "console", Console(width=240, color_system=None)
    )
    result = CliRunner().invoke(download_checkpoints.app, ["list-available"])
    assert result.exit_code == 0, result.output
    for name in REGISTERED_CHECKPOINTS:
        assert name in result.output
    assert "zatom2-table4" not in result.output
    assert "zatom2-figure1" not in result.output
    for name in re.findall(
        r"nucleus install ([^\s]+)", download_checkpoints.install.__doc__
    ):
        assert name in REGISTERED_CHECKPOINTS or name in {"all", "base-models"}


def test_deleted_scratch_checkpoint_is_not_silently_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The removed epoch-1 ID must not silently return different trained weights."""
    monkeypatch.setattr(
        download_checkpoints, "_resolve_checkpoint_dirs", lambda _: [tmp_path]
    )
    result = CliRunner().invoke(
        download_checkpoints.app,
        ["install", "zatom2-table4-scope-no-pretraining-epoch1"],
    )
    assert result.exit_code == 1
    assert "Unknown model" in result.output
    assert not list(tmp_path.iterdir())


def test_download_checkpoint_uses_hub_client_and_verifies_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hub downloads should use the configured repo and validate their payload."""
    payload = b"paper-ready-checkpoint"
    checkpoint = RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-omol25-omat24-joint-force-conditioned-"
            "epoch80.ckpt"
        ),
        description="test checkpoint",
        repo_id="Zatom-AI/Zatom-2",
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-"
            "force-conditioned/step-00097500.ckpt"
        ),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    calls = []

    def fake_hf_hub_download(**kwargs) -> str:
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)

    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hf_hub_download)
    destination = tmp_path / checkpoint.filename

    download_checkpoints.download_checkpoint(checkpoint, destination, force=True)

    assert destination.read_bytes() == payload
    assert calls == [
        {
            "repo_id": "Zatom-AI/Zatom-2",
            "filename": (
                "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-"
                "force-conditioned/step-00097500.ckpt"
            ),
            "local_dir": tmp_path,
            "force_download": True,
        }
    ]


def test_download_checkpoint_removes_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupt Hub download should not remain in the checkpoint directory."""
    checkpoint = RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-omol25-omat24-joint-force-conditioned-"
            "epoch80.ckpt"
        ),
        description="test checkpoint",
        repo_id="Zatom-AI/Zatom-2",
        sha256="0" * 64,
    )

    def fake_hf_hub_download(**kwargs) -> str:
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"corrupt")
        return str(destination)

    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hf_hub_download)
    destination = tmp_path / checkpoint.filename

    with pytest.raises(ValueError, match="Hash mismatch"):
        download_checkpoints.download_checkpoint(checkpoint, destination)

    assert not destination.exists()
