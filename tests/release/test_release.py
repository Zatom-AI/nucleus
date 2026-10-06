"""Lightweight release contracts, independent of GPU/training dependencies."""

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from install_reference_data import install
from link_scope_structures import link_structures
from train_paper import command


def args(experiment, devices=4, nodes=1, **kwargs):
    return argparse.Namespace(
        experiment=experiment,
        devices=devices,
        nodes=nodes,
        checkpoint=kwargs.get("checkpoint"),
        bootstrap=kwargs.get("bootstrap", False),
    )


@pytest.mark.parametrize(
    ("experiment", "devices", "expected"),
    [
        ("paper-table3-joint-properties-force-conditioned", 4, 16),
        ("paper-table3-zatom2-l-joint-properties-force-conditioned", 4, 32),
        ("paper-table2-qm9-mp20-unconditional-without-force-conditioning", 4, 32),
        ("paper-table4-scope", 4, 4),
    ],
)
def test_paper_batch_contract(experiment, devices, expected):
    cmd = command(args(experiment, devices=devices), [])
    assert f"trainer.grad_accum_steps={expected}" in cmd
    if experiment == "paper-table4-scope":
        assert "trainer.n_examples_per_epoch=3200000" in cmd
        assert "trainer.limit_train_batches=200000" in cmd


def test_invalid_world_size_cannot_change_paper_batch():
    with pytest.raises(ValueError, match="divisor"):
        command(args("paper-table3-joint-properties-force-conditioned", devices=3), [])


def test_scope_bootstrap_preserves_absolute_epoch_counter(tmp_path):
    checkpoint = tmp_path / "epoch-0080.ckpt"
    checkpoint.touch()
    cmd = command(args("paper-table4-scope", checkpoint=checkpoint, bootstrap=True), [])
    assert "trainer.max_epochs=82" in cmd
    assert "reset_optimizer_on_resume=true" in cmd
    assert "trainer.allow_world_size_change_on_resume=true" in cmd


@pytest.mark.parametrize(
    "experiment",
    sorted(
        p.stem for p in (ROOT / "models/zatom2/configs/experiment").glob("paper-*.yaml")
    )
    + ["train-geom-drugs"],
)
def test_paper_launch_commands_compose_without_private_environment(
    experiment, monkeypatch
):
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("NUCLEUS_DATA_PATH", str(ROOT / "data"))
    monkeypatch.setenv("ATOMWORKS_DATA_PATH", str(ROOT / "data"))
    cmd = command(args(experiment), [])
    with initialize_config_dir(
        config_dir=str(ROOT / "models/zatom2/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=cmd[3:])
    assert cfg.trainer.num_nodes == 1
    assert "wandb" not in cfg.logger
    assert cfg.model.net.diffusion_module.f_pred == "flow_matching"


def test_scope_links_use_frozen_domain_ids_and_do_not_overwrite(tmp_path):
    original = tmp_path / "original" / "ab" / "d1abc_1.ent"
    original.parent.mkdir(parents=True)
    original.write_text("ATOM")
    root = tmp_path / "rungs"
    csv = root / "seed0/scope_2000.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("domain_id,path\nd1abc_1,d1abc_1\n")
    assert link_structures(tmp_path / "original", root) == 1
    target = root / "seed0/pdb_2000/d1abc_1.pdb"
    assert target.resolve() == original.resolve()
    assert link_structures(tmp_path / "original", root) == 0
    target.unlink()
    target.write_text("keep")
    with pytest.raises(FileExistsError):
        link_structures(tmp_path / "original", root)
    assert target.read_text() == "keep"


def reference_archive(
    tmp_path, name="omol25/reference.pt", content=b"original", file_hash=None
):
    archive = tmp_path / "reference.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": {
                    name: {
                        "sha256": file_hash or hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                    }
                },
            }
        )
    )
    return archive, manifest


def test_reference_installer_verifies_and_preserves_existing_files(tmp_path):
    archive, manifest = reference_archive(tmp_path)
    target = tmp_path / "data"
    install(archive, target, manifest)
    install(archive, target, manifest)
    reference = target / "omol25/reference.pt"
    assert reference.read_bytes() == b"original"
    reference.write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        install(archive, target, manifest)
    assert reference.read_bytes() == b"user data"


@pytest.mark.parametrize(
    ("name", "file_hash"), [("../escape", None), ("omol25/reference.pt", "0" * 64)]
)
def test_reference_installer_rejects_invalid_payload_before_publishing(
    tmp_path, name, file_hash
):
    archive, manifest = reference_archive(tmp_path, name=name, file_hash=file_hash)
    target = tmp_path / "data"
    with pytest.raises(ValueError):
        install(archive, target, manifest)
    assert list(target.iterdir()) == []
    assert not (tmp_path / "escape").exists()
