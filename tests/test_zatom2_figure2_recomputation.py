"""Validate sample manifests without importing model or GPU evaluators."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def recomputation(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "scripts"))
    generation_module = types.ModuleType("zatom2.generation_evaluation")
    generation_module._load_input_specifications = lambda path: json.loads(
        Path(path).read_text()
    )
    monkeypatch.setitem(sys.modules, "zatom2.generation_evaluation", generation_module)
    path = repo / "scripts/recompute_zatom2_figure2_metrics.py"
    spec = importlib.util.spec_from_file_location("figure2_recompute_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_job(recomputation, root, job_name, target_ids, *, seed=42, payload=b"CIF"):
    fixture = root / "fixture.json"
    fixture.write_text(json.dumps({name: {} for name in ("a", "b", "c", "d", "e")}))
    job = root / "train" / recomputation.EXPERIMENTS["base_full"] / job_name
    dataset_dir = job / "val_structures/predictions/epoch_15/omol25_generation"
    dataset_dir.mkdir(parents=True)
    config_dir = job / ".hydra"
    config_dir.mkdir()
    config = {
        "paths": {"root_dir": str(root)},
        "datasets": {
            "val": {
                "omol25_generation": {
                    "dataset": {
                        "data": str(fixture),
                        "eval_every_n": 1,
                        "subset_to_keys": None,
                        "bond_sidecar_path": None,
                    }
                }
            }
        },
        "trainer": {"num_nodes": 1, "devices_per_node": 2},
        "callbacks": {
            "log_molecule_afd_validation_metrics_callback": {
                "_target_": "fake.AFDCallback",
                "seed": seed,
            }
        },
    }
    OmegaConf.save(OmegaConf.create(config), config_dir / "config.yaml")
    for target in target_ids:
        path = dataset_dir / f"{target}_model_0.cif.gz"
        path.write_bytes(gzip.compress(payload, mtime=0))
    return dataset_dir, config_dir / "config.yaml"


def discover(recomputation, root):
    return recomputation.discover(
        root / "train", "base_full", 15, "omol25_generation", root
    )


def test_complementary_jobs_preserve_original_rank_major_order(recomputation, tmp_path):
    create_job(recomputation, tmp_path, "job_a", ["a", "c", "e"])
    create_job(recomputation, tmp_path, "job_b", ["b", "d"])

    bundle = discover(recomputation, tmp_path)

    assert [row["target_id"] for row in bundle["manifest"]] == [
        "a",
        "c",
        "e",
        "b",
        "d",
    ]
    assert bundle["expected_targets"] == 5
    assert bundle["missing_targets"] == []
    # DistributedSampler would pad rank 1 to three items. The manifest excludes
    # that extra target, as the original metric callback excludes padding.
    assert len(bundle["manifest"]) == 5
    assert len(bundle["configurations"]) == 2


def test_missing_samples_stay_missing_and_duplicates_are_counted_once(
    recomputation, tmp_path
):
    create_job(recomputation, tmp_path, "job_a", ["a", "e"])
    create_job(recomputation, tmp_path, "job_b", ["a", "b", "d"])

    bundle = discover(recomputation, tmp_path)

    assert [row["target_id"] for row in bundle["manifest"]] == ["a", "e", "b", "d"]
    assert bundle["missing_targets"] == ["c"]
    assert len(bundle["manifest"][0]["duplicate_paths"]) == 1
    assert bundle["expected_targets"] == 5


def test_conflicting_predictions_for_one_target_are_rejected(recomputation, tmp_path):
    create_job(recomputation, tmp_path, "job_a", ["a"], payload=b"first prediction")
    create_job(recomputation, tmp_path, "job_b", ["a"], payload=b"other prediction")

    with pytest.raises(ValueError, match="Conflicting duplicate prediction for a"):
        discover(recomputation, tmp_path)


def test_changed_evaluation_settings_cannot_mix_job_samples(recomputation, tmp_path):
    create_job(recomputation, tmp_path, "job_a", ["a"], seed=42)
    create_job(recomputation, tmp_path, "job_b", ["b"], seed=99)

    with pytest.raises(ValueError, match="Inconsistent evaluation settings"):
        discover(recomputation, tmp_path)


def test_unknown_target_is_rejected_instead_of_silently_ignored(
    recomputation, tmp_path
):
    create_job(recomputation, tmp_path, "job_a", ["unknown_target"])

    with pytest.raises(ValueError, match="Prediction absent from validation fixture"):
        discover(recomputation, tmp_path)


def test_multiple_candidate_protocol_is_rejected(recomputation, tmp_path):
    directory, _ = create_job(recomputation, tmp_path, "job_a", ["a"])
    (directory / "a_model_1.cif.gz").write_bytes(gzip.compress(b"CIF", mtime=0))

    with pytest.raises(ValueError, match="Expected one candidate per target"):
        discover(recomputation, tmp_path)


def test_species_protocol_requires_successful_historical_bug_control(
    recomputation, tmp_path
):
    path = tmp_path / "base_full_epoch15_omol25_stale_carbon_medium.json"
    control = {
        "status": "diagnostic_stale_atomic_number",
        "n_samples": 1000,
        "historical_logged_value": 0.27736391915461867,
        "value": 0.27736064073354205,
    }
    path.write_text(json.dumps(control))
    result = recomputation.validate_legacy_afd_control("omol25", tmp_path)
    assert result["passed"]
    assert result["absolute_difference"] < result["absolute_tolerance"]

    control["value"] = 0.281313146726466
    path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match="stale-carbon replay failed"):
        recomputation.validate_legacy_afd_control("omol25", tmp_path)


def test_structure_recovery_needs_matching_complete_control(recomputation, tmp_path):
    missing = recomputation.validate_structure_control(
        "large_full", 20, "omat_match", 0.074, tmp_path
    )
    assert missing["passed"] is False
    control = recomputation.validate_structure_control(
        "base_full", 15, "omat_match", 0.072, tmp_path
    )
    assert control["passed"] is True
    changed = recomputation.validate_structure_control(
        "base_full", 15, "omat_match", 0.074, tmp_path
    )
    assert changed["passed"] is False


@pytest.mark.parametrize(
    "setting", ["key_to_balance", "subset_to_keys", "eval_every_n"]
)
def test_sampler_changes_require_an_explicit_order_manifest(
    recomputation, tmp_path, setting
):
    _, config_path = create_job(recomputation, tmp_path, "job_a", ["a"])
    config = OmegaConf.load(config_path)
    dataset = config.datasets.val.omol25_generation
    if setting == "key_to_balance":
        dataset.key_to_balance = "atom_count"
    elif setting == "subset_to_keys":
        dataset.dataset.subset_to_keys = ["a"]
    else:
        dataset.dataset.eval_every_n = 2
    OmegaConf.save(config, config_path)

    with pytest.raises(ValueError, match="samplers|fixtures|eval_every_n"):
        discover(recomputation, tmp_path)
