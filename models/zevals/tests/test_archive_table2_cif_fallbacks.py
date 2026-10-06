"""The narrow CIF-fallback recovery command preserves valid evaluation work."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "archive_table2_cif_fallbacks.py"
SPEC = importlib.util.spec_from_file_location("archive_table2_cif_fallbacks", SCRIPT)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def affected_run(tmp_path):
    root = tmp_path / "evaluation"
    protocol_path = root / "protocol" / "protocol_manifest.json"
    _write_json(protocol_path, {"protocol_version": 3})
    run = {
        "protocol": {
            "name": recovery.PROTOCOL,
            "version": 3,
            "manifest": str(protocol_path),
            "manifest_sha256": recovery._sha256(protocol_path),
        },
        "domains": {},
    }
    for domain in ("qm9", "mp20"):
        aggregate = hashlib.sha256()
        rows = []
        for index in range(2):
            sample_id = f"{domain}-{index:06d}"
            row = {
                "sample_id": sample_id,
                "sample_index": index,
                "cif_path": f"samples/{sample_id}.cif.gz",
                "metadata_path": f"samples/{sample_id}.json",
            }
            metadata = root / domain / row["metadata_path"]
            _write_json(metadata, {"table2_sampling": {"sample_id": sample_id}})
            cif = root / domain / row["cif_path"]
            cif.write_bytes(
                gzip.compress(
                    b"data_generated\n_cell.length_a 3.0\n"
                    b"_atom_site.label_atom_id Al1\n",
                    mtime=0,
                )
            )
            aggregate.update(sample_id.encode())
            aggregate.update(recovery._sha256(cif).encode())
            aggregate.update(recovery._sha256(metadata).encode())
            rows.append(row)
        manifest_path = root / domain / "sample_manifest.json"
        _write_json(manifest_path, rows)
        run["domains"][domain] = {
            "num_samples": len(rows),
            "sample_manifest_sha256": recovery._sha256(manifest_path),
            "sample_artifacts_sha256": aggregate.hexdigest(),
        }
    _write_json(root / "run_manifest.json", run)
    _write_json(
        root / "qm9" / "metrics.json",
        {
            "evaluation": {
                "protocol": recovery.PROTOCOL,
                "protocol_version": 3,
                "num_attempted": 2,
                **run["domains"]["qm9"],
            }
        },
    )
    inputs = [
        {"name": path.name, "sha256": recovery._sha256(path)}
        for path in sorted((root / "mp20" / "samples").glob("*.cif.gz"))
    ]
    contract = {
        "protocol_version": 1,
        "input_dir": str(root / "mp20/samples"),
        "inputs": inputs,
    }
    stage = root / "mp20/chgnet_prerelax"
    (stage / "cifs").mkdir(parents=True)
    records = []
    for item in inputs:
        record = {
            "name": item["name"],
            "input_sha256": item["sha256"],
            "output_sha256": item["sha256"],
            "status": "fallback_parse_error",
            "error": recovery.PARSE_ERROR,
            "converged": False,
            "contract_sha256": recovery._digest(contract),
            "runtime_sha256": recovery._digest({}),
        }
        _write_json(stage / "records" / f"{item['name']}.json", record)
        (stage / "cifs" / item["name"]).write_bytes(
            (root / "mp20/samples" / item["name"]).read_bytes()
        )
        records.append(record)
    _write_json(
        stage / "chgnet_prerelax.manifest.json",
        {
            "contract": contract,
            "contract_sha256": recovery._digest(contract),
            "runtime": {},
            "runtime_sha256": recovery._digest({}),
            "complete": True,
            "attempted_count": 2,
            "completed_count": 2,
            "converged_count": 0,
            "status_counts": {
                "fallback_parse_error": 2,
                "fallback_relaxation_error": 0,
                "relaxed": 0,
            },
            "records_sha256": recovery._digest(records),
        },
    )
    _write_json(root / "mp20/lematgenbench/resume_state/request.json", {})
    for relative in recovery.LOCK_PATHS:
        (root / relative).touch()
    # These unrelated artifacts must survive even though recovery does not need
    # to load or hash a multi-GB preserved checkpoint.
    _write_json(root / "launcher/request.json", {"requested": "unchanged"})
    _write_json(root / "launcher/checkpoint/model.ckpt", {"checkpoint": "untouched"})
    return root


def test_dry_run_changes_nothing(affected_run, capsys):
    before = _snapshot(affected_run)
    assert recovery.archive_fallbacks(affected_run) is None
    assert _snapshot(affected_run) == before
    assert "DRY RUN: no files changed" in capsys.readouterr().out
    assert not list(affected_run.glob("cif-fallback-recovery-*"))


def test_apply_archives_only_failed_stages(affected_run):
    before = _snapshot(affected_run)
    archive = recovery.archive_fallbacks(affected_run, apply=True)
    assert archive.parent == affected_run
    for relative, content in before.items():
        stage = next(
            (
                stage
                for stage in recovery.STAGE_PATHS
                if relative.startswith(stage + "/")
            ),
            None,
        )
        if stage:
            assert not (affected_run / relative).exists()
            archived = archive / Path(stage).name / Path(relative).relative_to(stage)
            assert archived.read_bytes() == content
        else:
            assert (affected_run / relative).read_bytes() == content
    assert json.loads((archive / "recovery.json").read_text())["fallback_count"] == 2


@pytest.mark.parametrize("relative", recovery.LOCK_PATHS)
def test_refuses_active_run_at_every_lock(affected_run, relative):
    before = _snapshot(affected_run)
    with (affected_run / relative).open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Run is active"):
            recovery.archive_fallbacks(affected_run, apply=True)
    assert _snapshot(affected_run) == before


@pytest.mark.parametrize(
    "relative",
    [
        "mp20/metrics.json",
        "table2_metrics.json",
        "table2_metrics.csv",
        "mp20/lematgenbench/lematgenbench_raw.json",
        "mp20/lematgenbench/resume_state/completed.json",
        "mp20/lematgenbench/resume_state/oracle-cache.json",
    ],
)
def test_refuses_completed_or_additional_work(affected_run, relative):
    _write_json(affected_run / relative, {})
    before = _snapshot(affected_run)
    with pytest.raises(ValueError):
        recovery.archive_fallbacks(affected_run, apply=True)
    assert _snapshot(affected_run) == before


@pytest.mark.parametrize("domain", ["qm9", "mp20"])
def test_refuses_changed_raw_samples(affected_run, domain):
    source = affected_run / domain / "samples" / f"{domain}-000000.cif.gz"
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="sample contents changed"):
        recovery.archive_fallbacks(affected_run, apply=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "relaxed"),
        ("error", "ValueError: a different parse problem"),
        ("output_sha256", "wrong"),
        ("converged", True),
    ],
)
def test_refuses_non_matching_fallbacks(affected_run, key, value):
    path = affected_run / "mp20/chgnet_prerelax/records/mp20-000000.cif.gz.json"
    record = json.loads(path.read_text())
    record[key] = value
    _write_json(path, record)
    with pytest.raises(ValueError, match="known parse fallback"):
        recovery.archive_fallbacks(affected_run, apply=True)


def test_refuses_symlink_in_archive_target(affected_run):
    stage = affected_run / "mp20/lematgenbench"
    (stage / "input_cifs").symlink_to(affected_run / "mp20/samples")
    with pytest.raises(ValueError, match="symlinks in archive target"):
        recovery.archive_fallbacks(affected_run, apply=True)


def test_refuses_partial_chgnet_run(affected_run):
    path = affected_run / "mp20/chgnet_prerelax/chgnet_prerelax.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="completed pre-conversion run"):
        recovery.archive_fallbacks(affected_run, apply=True)


def test_refuses_changed_qm9_metric_binding(affected_run):
    path = affected_run / "qm9/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["evaluation"]["sample_artifacts_sha256"] = "changed"
    _write_json(path, metrics)
    with pytest.raises(ValueError, match="QM9 metrics do not match"):
        recovery.archive_fallbacks(affected_run, apply=True)


def test_refuses_unsafe_manifest_paths(affected_run):
    path = affected_run / "mp20/sample_manifest.json"
    rows = json.loads(path.read_text())
    rows[0]["cif_path"] = "../qm9/samples/qm9-000000.cif.gz"
    _write_json(path, rows)
    run_path = affected_run / "run_manifest.json"
    run = json.loads(run_path.read_text())
    run["domains"]["mp20"]["sample_manifest_sha256"] = recovery._sha256(path)
    _write_json(run_path, run)
    with pytest.raises(ValueError, match="Unexpected mp20 sample path"):
        recovery.archive_fallbacks(affected_run, apply=True)


def test_rolls_back_first_move_on_second_move_failure(affected_run, monkeypatch):
    before = _snapshot(affected_run)
    rename = Path.rename

    def fail_second(path, target):
        if path == affected_run / "mp20/lematgenbench":
            raise OSError("simulated rename failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="simulated"):
        recovery.archive_fallbacks(affected_run, apply=True)
    after = _snapshot(affected_run)
    assert all(after[path] == content for path, content in before.items())
    assert len(list(affected_run.glob("cif-fallback-recovery-*/recovery.json"))) == 1
