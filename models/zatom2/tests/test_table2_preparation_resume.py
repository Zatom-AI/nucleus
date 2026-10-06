"""Preparation can restart only its own interrupted, identical protocol request."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import zatom2.table2_evaluation as table2
import zatom2.table2_resume as resume


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    data_dirs = {}
    for domain in table2.DOMAIN_ORDER:
        data_dir = tmp_path / f"{domain}_data"
        data_dir.mkdir()
        (data_dir / "data0000.aselmdb").write_bytes(domain.encode())
        (data_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": domain,
                    "num_examples": 3,
                    "shards": ["data0000.aselmdb"],
                }
            )
        )
        data_dirs[domain] = data_dir
    cfg = OmegaConf.create(
        {
            "diffusion_batch_size": 2,
            "n_batches": 1,
            "evaluation": {
                "protocol_dir": str(tmp_path / "protocol"),
                "resume": True,
                "overwrite": False,
                "seed": 42,
                **{
                    domain: {
                        "num_samples": 5,
                        "training_num_examples": 3,
                        "training_data_dir": str(data_dirs[domain]),
                    }
                    for domain in table2.DOMAIN_ORDER
                },
            },
        }
    )
    monkeypatch.setattr(
        table2, "_scan_qm9_training_data", lambda _: ([3, 3, 3], ["C"], 0)
    )
    monkeypatch.setattr(table2, "_scan_mp20_training_data", lambda _: [2, 4, 4])
    return cfg


def _interrupt_after_first_asset(monkeypatch, cfg):
    write = table2._write_json_gz

    def interrupted_write(path, value):
        write(path, value)
        raise resume.EvaluationPaused("simulated allocation expiration")

    with monkeypatch.context() as patch:
        patch.setattr(table2, "_write_json_gz", interrupted_write)
        with pytest.raises(resume.EvaluationPaused, match="simulated"):
            table2.prepare_protocol(cfg)


def test_interrupted_preparation_recomputes_and_matches_fresh(monkeypatch, preparation):
    cfg = preparation
    _interrupt_after_first_asset(monkeypatch, cfg)
    output = Path(cfg.evaluation.protocol_dir)
    paths = table2._protocol_paths(output)
    assert paths["qm9_counts"].is_file()
    assert not paths["manifest"].exists()
    request_path = output / ".preparation_request.json"
    request_bytes = request_path.read_bytes()
    assert json.loads(request_bytes) == table2._protocol_request(cfg)

    # Incomplete artifacts are deliberately regenerated, not trusted.
    paths["qm9_counts"].write_bytes(b"partial data")
    actual = table2.prepare_protocol(cfg)
    assert request_path.read_bytes() == request_bytes
    verified, _ = table2.load_protocol(cfg)
    assert verified == actual
    cfg.evaluation.protocol_dir = str(output.with_name("fresh"))
    expected = table2.prepare_protocol(cfg)
    assert actual == expected


@pytest.mark.parametrize("change", ["seed", "dataset", "sample_count"])
def test_changed_request_rejected_before_recomputation(
    monkeypatch, preparation, change
):
    cfg = preparation
    _interrupt_after_first_asset(monkeypatch, cfg)
    output = Path(cfg.evaluation.protocol_dir)
    request_path = output / ".preparation_request.json"
    before = request_path.read_bytes()
    if change == "seed":
        cfg.evaluation.seed = 43
    elif change == "dataset":
        shard = Path(cfg.evaluation.qm9.training_data_dir) / "data0000.aselmdb"
        shard.write_bytes(b"changed training dataset")
    elif change == "sample_count":
        cfg.evaluation.qm9.num_samples = 6

    def unexpected_scan(_):
        pytest.fail("Changed request must be rejected before recomputing assets")

    monkeypatch.setattr(table2, "_scan_qm9_training_data", unexpected_scan)
    with pytest.raises(ValueError, match="preparation request changed"):
        table2.prepare_protocol(cfg)
    assert request_path.read_bytes() == before


def test_incomplete_preparation_requires_resume(monkeypatch, preparation):
    cfg = preparation
    _interrupt_after_first_asset(monkeypatch, cfg)
    cfg.evaluation.resume = False
    with pytest.raises(FileExistsError, match="evaluation.resume=true"):
        table2.prepare_protocol(cfg)


def test_unknown_incomplete_directory_remains_protected(preparation):
    cfg = preparation
    output = Path(cfg.evaluation.protocol_dir)
    output.mkdir()
    unknown = output / "qm9_node_counts.json.gz"
    unknown.write_bytes(b"preexisting unbound data")
    with pytest.raises(FileExistsError, match="no matching preparation request"):
        table2.prepare_protocol(cfg)
    assert unknown.read_bytes() == b"preexisting unbound data"
    assert not (output / ".preparation_request.json").exists()


def test_interruption_before_request_publication_can_restart(preparation):
    cfg = preparation
    output = Path(cfg.evaluation.protocol_dir)
    output.mkdir()
    (output / "..preparation_request.json.interrupted.tmp").write_bytes(b'{"partial":')
    table2.prepare_protocol(cfg)
    assert table2._protocol_paths(output)["manifest"].is_file()


def test_protocol_lock_protects_shared_directory_and_preserves_deadline(
    monkeypatch, preparation, tmp_path
):
    cfg = preparation
    output = Path(cfg.evaluation.protocol_dir)
    output.mkdir()
    with (output / ".preparation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another evaluation is preparing"):
            table2.prepare_protocol(cfg)
    monkeypatch.setattr(resume.time, "monotonic", lambda: 100.0)
    with resume.evaluation_allocation(tmp_path / "evaluation", 60):
        table2.prepare_protocol(cfg)
        assert resume.remaining_seconds() == 60


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_allocation_and_stage_budgets_must_be_finite_positive(tmp_path, value):
    with pytest.raises(ValueError, match="must be positive"):
        with resume.evaluation_allocation(tmp_path, value):
            pytest.fail("Invalid allocation budget accepted")
    with pytest.raises(ValueError, match="must be positive"):
        resume.remaining_seconds(value)
