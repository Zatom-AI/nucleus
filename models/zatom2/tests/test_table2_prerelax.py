"""Table-2 pre-relaxation routing, provenance, and pause behavior."""

from __future__ import annotations

import gzip
from dataclasses import replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from zevals.materials import chgnet_prerelax as chgnet

import zatom2.table2_evaluation as table2


@pytest.fixture
def prepared(tmp_path):
    cfg = OmegaConf.create(
        {
            "out_dir": str(tmp_path),
            "evaluation": {
                "seed": 42,
                "resume": True,
                "mp20": {
                    "prerelax": {
                        "enabled": True,
                        "runner_command": ["/isolated/python"],
                        "fmax": 0.1,
                        "steps": 1500,
                        "relax_cell": True,
                        "device": "cuda",
                        "seed": 42,
                    }
                },
            },
        }
    )
    input_dir = tmp_path / "mp20/samples"
    input_dir.mkdir(parents=True)
    rows = []
    for index in range(2):
        path = input_dir / f"mp20-{index:06d}.cif.gz"
        path.write_bytes(gzip.compress(f"raw-{index}".encode(), mtime=0))
        rows.append({"resolved_cif_path": str(path)})
    output_dir = tmp_path / "mp20/chgnet_prerelax"
    (output_dir / "cifs").mkdir(parents=True)
    (output_dir / "records").mkdir()
    contract = chgnet._contract(
        input_dir,
        chgnet._configuration(
            fmax=0.1, steps=1500, relax_cell=True, device="cuda", seed=42
        ),
    )
    runtime = {
        "packages": {"chgnet": "0.3.1", "ase": "3.22.1"},
        "actual_device": "cuda:0",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "checkpoint": {
            "model": "0.3.0",
            "sha256": chgnet.CHGNET_CHECKPOINT_SHA256,
            "wheel_record_sha256": chgnet.CHGNET_CHECKPOINT_SHA256,
        },
    }
    contract_digest, runtime_digest = (
        chgnet._json_digest(contract),
        chgnet._json_digest(runtime),
    )
    records = []
    for item in contract["inputs"]:
        path = output_dir / "cifs" / item["name"]
        path.write_bytes(gzip.compress(b"relaxed geometry", mtime=0))
        record = {
            "name": item["name"],
            "input_sha256": item["sha256"],
            "output_sha256": chgnet._sha256(path),
            "contract_sha256": contract_digest,
            "runtime_sha256": runtime_digest,
            "status": "relaxed",
            "converged": True,
        }
        chgnet._atomic_json(output_dir / "records" / f"{item['name']}.json", record)
        records.append(record)
    manifest = {
        "contract": contract,
        "contract_sha256": contract_digest,
        "runtime": runtime,
        "runtime_sha256": runtime_digest,
        "complete": True,
        "attempted_count": 2,
        "completed_count": 2,
        "converged_count": 2,
        "records_sha256": chgnet._json_digest(records),
        "status_counts": {
            "relaxed": 2,
            "fallback_parse_error": 0,
            "fallback_relaxation_error": 0,
        },
    }
    chgnet._atomic_json(output_dir / chgnet._MANIFEST, manifest)
    result = chgnet.validate_chgnet_prerelaxation(input_dir, output_dir=output_dir)
    return cfg, rows, result


def test_disabled_prerelax_scores_original_cifs(prepared, monkeypatch):
    cfg, rows, _ = prepared
    cfg.evaluation.mp20.prerelax.enabled = False

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled pre-relaxation must not invoke CHGNet")

    monkeypatch.setattr(chgnet, "run_chgnet_prerelaxation", forbidden)
    directory, inputs, provenance = table2._material_scoring_input(cfg, rows, run=True)
    assert directory == Path(cfg.out_dir) / "mp20/samples"
    assert inputs == table2._expected_lemat_source_inputs(rows)
    assert provenance is None


def test_enabled_prerelax_maps_canonical_gzip_names(prepared, monkeypatch):
    cfg, rows, result = prepared
    calls = []

    def run(input_dir, **kwargs):
        calls.append((input_dir, kwargs))
        return result

    monkeypatch.setattr(chgnet, "run_chgnet_prerelaxation", run)
    monkeypatch.setattr(table2, "remaining_seconds", lambda: 123)
    directory, inputs, provenance = table2._material_scoring_input(
        cfg, list(reversed(rows)), run=True
    )
    assert directory == result.cif_dir
    assert [Path(item["source_path"]).name for item in inputs] == [
        "mp20-000000.cif.gz",
        "mp20-000001.cif.gz",
    ]
    assert all(Path(item["source_path"]).parent == result.cif_dir for item in inputs)
    assert provenance["variant"] == "chgnet-flowmm"
    assert provenance["manifest_sha256"] == table2._sha256_file(result.manifest_path)
    assert provenance["records_sha256"] == result.manifest["records_sha256"]
    assert calls[0][1]["resume"] is True
    assert calls[0][1]["timeout_seconds"] == 123
    assert calls[0][1]["runner_command"] == ["/isolated/python"]


def test_read_only_material_validation_does_not_start_worker(prepared, monkeypatch):
    cfg, rows, result = prepared

    def forbidden(*args, **kwargs):
        pytest.fail("Report validation must not run a relaxation worker")

    monkeypatch.setattr(chgnet, "run_chgnet_prerelaxation", forbidden)
    directory, _, provenance = table2._material_scoring_input(cfg, rows, run=False)
    assert directory == result.cif_dir
    assert provenance["runtime_sha256"] == result.manifest["runtime_sha256"]


def test_incomplete_relaxation_pauses_before_scoring(prepared, monkeypatch):
    cfg, rows, result = prepared
    monkeypatch.setattr(
        chgnet,
        "run_chgnet_prerelaxation",
        lambda *args, **kwargs: replace(result, complete=False),
    )
    with pytest.raises(table2.EvaluationPaused, match="partial progress"):
        table2._material_scoring_input(cfg, rows, run=True)


def test_attempt_count_mismatch_is_rejected(prepared, monkeypatch):
    cfg, rows, result = prepared
    manifest = dict(result.manifest, records=result.manifest["records"][:1])
    monkeypatch.setattr(
        chgnet,
        "run_chgnet_prerelaxation",
        lambda *args, **kwargs: replace(result, manifest=manifest),
    )
    with pytest.raises(ValueError, match="attempted sample count"):
        table2._material_scoring_input(cfg, rows, run=True)


@pytest.mark.parametrize("source", ["raw", "relaxed"])
def test_material_validation_rejects_stale_hashes(prepared, source):
    cfg, rows, result = prepared
    path = (
        Path(rows[0]["resolved_cif_path"])
        if source == "raw"
        else result.cif_dir / "mp20-000000.cif.gz"
    )
    path.write_bytes(gzip.compress(b"changed", mtime=0))
    with pytest.raises(ValueError):
        table2._material_scoring_input(cfg, rows, run=False)


def test_raw_and_prerelaxed_scores_have_distinct_run_names(prepared):
    cfg, rows, _ = prepared
    _, _, provenance = table2._material_scoring_input(cfg, rows, run=False)
    raw_hash = "f" * 64
    relaxed_hash = table2._material_run_hash(raw_hash, provenance)
    assert table2._material_run_hash(raw_hash, None) == raw_hash
    assert relaxed_hash != raw_hash
    assert table2._material_run_hash(raw_hash, provenance) == relaxed_hash
    names = {
        table2._effective_lemat_run_name(
            "table2",
            domain_dir=Path(cfg.out_dir) / "mp20",
            sample_manifest_sha256="e" * 64,
            sample_artifacts_sha256=digest,
        )
        for digest in (raw_hash, relaxed_hash)
    }
    assert len(names) == 2


def test_prerelax_config_is_optional_and_freezes_flowmm_protocol(tmp_path, monkeypatch):
    project_root = Path(__file__).parents[3]
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    config_dir = project_root / "models/zatom2/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="table2_evaluation",
            overrides=["ckpt_path=/tmp/checkpoint.ckpt", f"out_dir={tmp_path}"],
        )
    assert cfg.evaluation.mp20.prerelax.enabled is False
    cfg.evaluation.mp20.prerelax.enabled = True
    assert table2._validate_protocol_conformance(cfg) == {}
    for name, changed in (
        ("fmax", 0.2),
        ("steps", 100),
        ("relax_cell", False),
        ("device", "cpu"),
        ("seed", 43),
    ):
        original = cfg.evaluation.mp20.prerelax[name]
        cfg.evaluation.mp20.prerelax[name] = changed
        with pytest.raises(ValueError, match="frozen manuscript protocol"):
            table2._validate_protocol_conformance(cfg)
        cfg.evaluation.mp20.prerelax[name] = original
