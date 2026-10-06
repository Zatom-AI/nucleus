from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import zatom2.table2_evaluation as table2_evaluation
from zatom2.generation_evaluation import GeneratedSample
from zatom2.table2_evaluation import (
    CANONICAL_TRAINING_DATASETS,
    _dataset_manifest,
    _effective_lemat_run_name,
    _normalize_stages,
    _read_json_gz,
    _retain_planned_samples,
    _sample_artifacts_sha256,
    _sampling_plan_metadata,
    _sha256_file,
    _validate_lemat_manifest_contract,
    _validate_protocol_conformance,
    _validate_saved_sample_manifest,
    _verify_protocol_request,
    _verify_qm9_report_binding,
    _verify_report_bindings,
    _write_json_gz,
    build_input_manifest,
    load_protocol,
    prepare_protocol,
    run_table2_evaluation,
    sample_node_counts,
    score_mp20,
)


def _write_saved_table2_samples(
    output_dir: Path,
    checkpoint: Path,
    *,
    domain: str,
    count: int,
) -> Path:
    domain_dir = output_dir / domain
    samples_dir = domain_dir / "samples"
    samples_dir.mkdir(parents=True)
    rows = []
    for index in range(count):
        sample_id = f"{domain}-{index:06d}"
        group_id = f"{domain}-group-{index:06d}"
        plan_sha256 = f"{index:064x}"
        cif_path = samples_dir / f"{sample_id}.cif.gz"
        metadata_path = samples_dir / f"{sample_id}.json"
        # Deliberately not a parseable CIF: manifest validation must not erase
        # failed generations from the attempted denominator.
        cif_path.write_bytes(f"invalid-{sample_id}".encode())
        metadata_path.write_text(
            json.dumps(
                {
                    "ckpt_path": str(checkpoint),
                    "table2_sampling": {
                        "canonical_sample_id": sample_id,
                        "canonical_sample_index": index,
                        "sampling_group_id": group_id,
                        "sampling_group_model_index": 0,
                        "sampling_plan_sha256": plan_sha256,
                    },
                }
            )
            + "\n"
        )
        rows.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "cif_path": str(cif_path.relative_to(domain_dir)),
                "metadata_path": str(metadata_path.relative_to(domain_dir)),
                "sampling_group_id": group_id,
                "sampling_group_model_index": 0,
                "sampling_plan_sha256": plan_sha256,
            }
        )
    (domain_dir / "sample_manifest.json").write_text(json.dumps(rows) + "\n")
    return domain_dir


def _minimal_lemat_cfg(output_dir: Path, checkpoint: Path, *, reuse_raw_results: bool):
    return OmegaConf.create(
        {
            "ckpt_path": str(checkpoint),
            "out_dir": str(output_dir),
            "evaluation": {
                "mp20": {
                    "num_samples": 2,
                    "lematgenbench": {
                        "repo_path": "/unused/pinned-checkout",
                        "run_name": "zatom2-table2",
                        "config_name": "comprehensive_multi_mlip_hull",
                        "families": [
                            "distribution",
                            "diversity",
                            "novelty",
                            "uniqueness",
                            "stability",
                            "sun",
                        ],
                        "fingerprint_method": "structure-matcher",
                        "runner_command": ["uv", "run", "--frozen", "python"],
                        "expected_commit": "commit",
                        "expected_lock_sha256": "lock",
                        "verify_mace_checkpoint": True,
                        "expected_mace_checkpoint_sha256": "mace",
                        "verify_orb_checkpoint": True,
                        "expected_orb_checkpoint_sha256": "orb",
                        "verify_uma_checkpoint": True,
                        "expected_uma_checkpoint_sha256": "uma",
                        "enforce_reference_revisions": True,
                        "runtime_device": "cuda",
                        "cuda_visible_devices": None,
                        "parse_failures_as_invalid": True,
                        "require_complete_oracle_coverage": True,
                        "require_clean_repo": True,
                        "timeout_seconds": None,
                        "fail_on_family_error": True,
                        "reuse_raw_results": reuse_raw_results,
                        "distribution_key": "distribution/JSDistance",
                        "diversity_key": "diversity/element_diversity",
                    },
                }
            },
        }
    )


def test_sample_node_counts_is_deterministic_and_empirical() -> None:
    counts = [3, 3, 4, 7]

    first = sample_node_counts(counts, num_samples=20, seed=42)
    second = sample_node_counts(counts, num_samples=20, seed=42)

    assert first == second
    assert set(first) <= {3, 4, 7}
    assert first != sample_node_counts(counts, num_samples=20, seed=43)


def test_input_manifests_group_equal_counts_into_diffusion_replicas() -> None:
    qm9 = build_input_manifest(
        "qm9", [3], num_samples=5, seed=42, diffusion_batch_size=2
    )
    mp20 = build_input_manifest(
        "mp20", [8], num_samples=5, seed=42, diffusion_batch_size=2
    )

    assert qm9["global_args"] == {"total_charge": 0, "spin_multiplicity": 1}
    qm9_group_ids = list(qm9)[1:]
    mp20_group_ids = list(mp20)
    assert len(qm9_group_ids) == len(mp20_group_ids) == 3
    assert qm9[qm9_group_ids[0]]["contig"] == "3m"
    assert mp20[mp20_group_ids[0]]["contig"] == "8M"
    assert mp20[mp20_group_ids[0]]["extra"]["table2_sample_indices"] == [0, 1]
    assert mp20[mp20_group_ids[-1]]["extra"]["table2_sample_indices"] == [4]
    assert mp20[mp20_group_ids[-1]]["extra"]["table2_retained_model_indices"] == [0]


def test_grouped_manifest_preserves_exact_sampled_node_count_multiset() -> None:
    node_counts = [3, 3, 4, 7]
    num_samples = 41
    diffusion_batch_size = 4
    sampled_counts = sample_node_counts(node_counts, num_samples=num_samples, seed=42)
    manifest = build_input_manifest(
        "mp20",
        node_counts,
        num_samples=num_samples,
        seed=42,
        diffusion_batch_size=diffusion_batch_size,
    )
    plan = _sampling_plan_metadata(
        "mp20",
        manifest,
        diffusion_batch_size=diffusion_batch_size,
        expected_num_samples=num_samples,
    )

    reconstructed_counts = [None] * num_samples
    for specification in manifest.values():
        extra = specification["extra"]
        for sample_index in extra["table2_sample_indices"]:
            reconstructed_counts[sample_index] = extra["table2_node_count"]

    histogram = Counter(sampled_counts)
    expected_groups = sum(
        (count + diffusion_batch_size - 1) // diffusion_batch_size
        for count in histogram.values()
    )
    assert reconstructed_counts == sampled_counts
    assert plan["num_input_specifications"] == expected_groups
    assert plan["num_generated_outputs"] == expected_groups * diffusion_batch_size
    assert plan["num_retained_outputs"] == num_samples
    assert plan["num_discarded_padding_outputs"] == (
        expected_groups * diffusion_batch_size - num_samples
    )
    assert plan["node_count_histogram"] == {
        str(node_count): count for node_count, count in sorted(histogram.items())
    }
    assert plan == _sampling_plan_metadata(
        "mp20",
        manifest,
        diffusion_batch_size=diffusion_batch_size,
        expected_num_samples=num_samples,
    )


def test_grouped_outputs_are_trimmed_and_reindexed_canonically() -> None:
    diffusion_batch_size = 2
    manifest = build_input_manifest(
        "mp20",
        [5],
        num_samples=5,
        seed=42,
        diffusion_batch_size=diffusion_batch_size,
    )
    generated = []
    for group_id, specification in manifest.items():
        for model_index in range(diffusion_batch_size):
            source_id = f"{group_id}_0_model_{model_index}"
            output = SimpleNamespace(
                example_id=source_id,
                metadata={"preserved": True},
                atom_array=None,
            )
            generated.append(
                GeneratedSample(
                    sample_index=len(generated),
                    sample_id=source_id,
                    target_id=group_id,
                    batch_index=0,
                    model_index=model_index,
                    specification=dict(specification),
                    output=output,
                )
            )

    retained, stats = _retain_planned_samples(
        "mp20",
        generated,
        manifest,
        diffusion_batch_size=diffusion_batch_size,
        expected_num_samples=5,
    )

    assert [sample.sample_index for sample in retained] == list(range(5))
    assert [sample.sample_id for sample in retained] == [
        f"mp20-{index:06d}" for index in range(5)
    ]
    assert [sample.model_index for sample in retained] == [0, 1, 0, 1, 0]
    assert [sample.output.example_id for sample in retained] == [
        sample.sample_id for sample in retained
    ]
    assert stats["num_generated_outputs"] == 6
    assert stats["num_retained_outputs"] == 5
    assert stats["num_discarded_padding_outputs"] == 1
    for sample in retained:
        sampling_metadata = sample.output.metadata["table2_sampling"]
        assert sampling_metadata["canonical_sample_id"] == sample.sample_id
        assert sampling_metadata["sampling_plan_sha256"] == stats["sha256"]
        assert sample.output.metadata["specification"] == sample.specification
        assert sample.specification["extra"]["table2_sample_index"] == (
            sample.sample_index
        )
        assert sample.specification["extra"]["table2_sampling_group_id"] == (
            sample.target_id
        )
        assert sample.output.metadata["preserved"]


def test_sampling_resets_group_rng_after_lazy_engine_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import zatom2.run_inference as run_inference

    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    input_paths = {}
    specifications_by_domain = {}
    sampling_plans = {}
    for domain, node_count in (("qm9", 5), ("mp20", 7)):
        raw_manifest = build_input_manifest(
            domain,
            [node_count],
            num_samples=1,
            seed=42,
            diffusion_batch_size=1,
        )
        input_path = protocol_dir / f"{domain}_inputs.json"
        input_path.write_text(json.dumps(raw_manifest) + "\n")
        input_paths[f"{domain}_inputs"] = input_path
        specifications = {
            key: value for key, value in raw_manifest.items() if key != "global_args"
        }
        if domain == "qm9":
            specifications = {
                key: value | raw_manifest["global_args"]
                for key, value in specifications.items()
            }
        specifications_by_domain[domain] = specifications
        sampling_plans[domain] = _sampling_plan_metadata(
            domain,
            specifications,
            diffusion_batch_size=1,
            expected_num_samples=1,
        )
    protocol_manifest_path = protocol_dir / "protocol_manifest.json"
    protocol_manifest_path.write_text("{}\n")
    input_paths["manifest"] = protocol_manifest_path
    protocol_manifest = {
        "request": {domain: {"num_samples": 1} for domain in ("qm9", "mp20")},
        "sampling_plans": sampling_plans,
    }

    captured_draws = []

    class FakeFabric:
        global_rank = 0
        world_size = 1

        @staticmethod
        def barrier() -> None:
            return None

    class FakeEngine:
        def __init__(self) -> None:
            self.initialized = False

        def initialize(self):
            if not self.initialized:
                # Simulate checkpoint/trainer construction consuming RNG.
                torch.rand(17)
                self.trainer = SimpleNamespace(fabric=FakeFabric())
                self.initialized = True

        def run(self, *, inputs, n_batches, out_dir):
            self.initialize()
            captured_draws.append(float(torch.rand(())))
            domain = Path(inputs).stem.removesuffix("_inputs")
            generated = []
            for group_id, specification in specifications_by_domain[domain].items():
                output = SimpleNamespace(
                    example_id=f"{group_id}_0_model_0",
                    metadata={},
                    atom_array=None,
                )
                generated.append(
                    GeneratedSample(
                        sample_index=0,
                        sample_id=output.example_id,
                        target_id=group_id,
                        batch_index=0,
                        model_index=0,
                        specification=dict(specification),
                        output=output,
                    )
                )
            return generated

    def fake_dump(samples, domain_dir):
        samples_dir = domain_dir / "samples"
        samples_dir.mkdir(parents=True)
        rows = []
        for sample in samples:
            cif_path = samples_dir / f"{sample.sample_id}.cif.gz"
            metadata_path = samples_dir / f"{sample.sample_id}.json"
            cif_path.write_bytes(b"sample")
            metadata_path.write_text(
                json.dumps(
                    {
                        **sample.output.metadata,
                        "ckpt_path": str(checkpoint.resolve()),
                    }
                )
                + "\n"
            )
            rows.append(
                {
                    "sample_index": sample.sample_index,
                    "sample_id": sample.sample_id,
                    "target_id": sample.target_id,
                    "batch_index": sample.batch_index,
                    "model_index": sample.model_index,
                    "cif_path": str(cif_path.relative_to(domain_dir)),
                    "metadata_path": str(metadata_path.relative_to(domain_dir)),
                }
            )
        return rows

    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(checkpoint),
            "out_dir": str(tmp_path / "output"),
            "inputs": None,
            "json_keys_subset": None,
            "diffusion_batch_size": 1,
            "inference_sampler": {},
            "use_ema_weights": True,
            "seed": 42,
            "evaluation": {
                "overwrite": False,
                "seed": 42,
                "hash_checkpoint": True,
                "allow_noncanonical_protocol": True,
            },
        }
    )
    monkeypatch.setattr(
        run_inference, "build_inference_engine", lambda _cfg: FakeEngine()
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_flatten_outputs",
        lambda outputs, *_args, **_kwargs: outputs,
    )
    monkeypatch.setattr(table2_evaluation, "_dump_samples", fake_dump)
    monkeypatch.setattr(
        table2_evaluation,
        "_checkpoint_provenance",
        lambda _cfg: {"path": str(checkpoint.resolve())},
    )
    monkeypatch.setattr(table2_evaluation, "_repository_provenance", lambda: {})
    monkeypatch.setattr(table2_evaluation, "_software_provenance", lambda: {})

    _, is_rank_zero = table2_evaluation.sample_checkpoint(
        cfg,
        protocol_manifest,
        input_paths,
    )

    expected_draws = []
    for domain, specifications in specifications_by_domain.items():
        group_id = next(iter(specifications))
        expected_generator = torch.Generator(device="cpu").manual_seed(
            table2_evaluation.group_seed(42, domain, group_id)
        )
        expected_draws.append(float(torch.rand((), generator=expected_generator)))
    assert is_rank_zero
    assert captured_draws == pytest.approx(expected_draws)


def test_prepare_protocol_freezes_and_verifies_grouped_plans(
    monkeypatch, tmp_path: Path
) -> None:
    def make_training_data(domain: str, count: int) -> Path:
        data_dir = tmp_path / f"{domain}_data"
        data_dir.mkdir()
        shard_name = "data0000.aselmdb"
        (data_dir / shard_name).write_bytes(domain.encode("ascii"))
        (data_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": domain,
                    "num_examples": count,
                    "shards": [shard_name],
                }
            )
            + "\n"
        )
        return data_dir

    qm9_data_dir = make_training_data("qm9", 4)
    mp20_data_dir = make_training_data("mp20", 3)
    monkeypatch.setattr(
        table2_evaluation,
        "_scan_qm9_training_data",
        lambda _: ([3, 3, 3, 3], ["C"], 0),
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_scan_mp20_training_data",
        lambda _: [2, 4, 4],
    )
    cfg = OmegaConf.create(
        {
            "diffusion_batch_size": 2,
            "n_batches": 1,
            "evaluation": {
                "protocol_dir": str(tmp_path / "protocol"),
                "overwrite": False,
                "seed": 42,
                "qm9": {
                    "num_samples": 5,
                    "training_num_examples": 4,
                    "training_data_dir": str(qm9_data_dir),
                },
                "mp20": {
                    "num_samples": 3,
                    "training_num_examples": 3,
                    "training_data_dir": str(mp20_data_dir),
                },
            },
        }
    )

    manifest = prepare_protocol(cfg)
    loaded, paths = load_protocol(cfg)

    assert loaded == manifest
    assert manifest["request"]["protocol_version"] == table2_evaluation.PROTOCOL_VERSION
    assert manifest["request"]["inference_batching"] == {
        "diffusion_batch_size": 2,
        "n_batches": 1,
        "strategy": "group_equal_node_counts_and_discard_tail_padding",
    }
    assert manifest["sampling_plans"]["qm9"]["num_generated_outputs"] == 6
    assert manifest["sampling_plans"]["qm9"]["num_retained_outputs"] == 5
    assert manifest["sampling_plans"]["qm9"]["num_discarded_padding_outputs"] == 1
    assert (
        _sha256_file(paths["qm9_inputs"])
        == manifest["artifacts"]["qm9_inputs"]["sha256"]
    )


def test_compressed_protocol_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    _write_json_gz(first, [3, 4, 5])
    _write_json_gz(second, [3, 4, 5])

    assert _sha256_file(first) == _sha256_file(second)
    assert _read_json_gz(first) == [3, 4, 5]


def test_saved_sample_validation_keeps_malformed_cif_as_an_attempt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    domain_dir = _write_saved_table2_samples(
        tmp_path / "output", checkpoint, domain="mp20", count=2
    )

    rows = _validate_saved_sample_manifest(
        domain_dir,
        domain="mp20",
        expected_num_samples=2,
        expected_checkpoint=checkpoint,
    )

    assert len(rows) == 2
    assert Path(rows[0]["resolved_cif_path"]).read_bytes().startswith(b"invalid-")


def test_effective_lemat_run_name_is_stable_and_output_unique(tmp_path: Path) -> None:
    kwargs = {
        "sample_manifest_sha256": "a" * 64,
        "sample_artifacts_sha256": "b" * 64,
    }
    first = _effective_lemat_run_name(
        "zatom2-table2", domain_dir=tmp_path / "first", **kwargs
    )
    repeated = _effective_lemat_run_name(
        "zatom2-table2", domain_dir=tmp_path / "first", **kwargs
    )
    second = _effective_lemat_run_name(
        "zatom2-table2", domain_dir=tmp_path / "second", **kwargs
    )

    assert first == repeated
    assert first != second
    assert first.startswith("zatom2-table2-")


def test_lemat_manifest_contract_rejects_noncanonical_reuse(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "output"
    domain_dir = _write_saved_table2_samples(
        output_dir, checkpoint, domain="mp20", count=2
    )
    cfg = _minimal_lemat_cfg(output_dir, checkpoint, reuse_raw_results=True)
    validated_rows = _validate_saved_sample_manifest(
        domain_dir,
        domain="mp20",
        expected_num_samples=2,
        expected_checkpoint=checkpoint,
    )
    source_records = sorted(
        (
            {
                "source_path": row["resolved_cif_path"],
                "sha256": _sha256_file(Path(row["resolved_cif_path"])),
            }
            for row in validated_rows
        ),
        key=lambda record: record["source_path"],
    )
    lemat_dir = domain_dir / "lematgenbench"
    staged_dir = lemat_dir / "input_cifs"
    staged_dir.mkdir(parents=True)
    input_cifs = []
    for index, source in enumerate(source_records):
        staged_path = staged_dir / f"{index:06d}.cif.gz"
        staged_path.write_bytes(Path(source["source_path"]).read_bytes())
        input_cifs.append(
            {
                "path": str(staged_path),
                "source_path": source["source_path"],
                "sha256": source["sha256"],
                "post_run_sha256": source["sha256"],
            }
        )
    reference_revisions = {"reference": "revision"}
    oracle_identities = {"orb": {"model": "orb"}}
    manifest = {
        "adapter": "zevals.materials.lematgenbench",
        "upstream_commit": "commit",
        "expected_commit": "commit",
        "upstream_uv_lock_sha256": "lock",
        "expected_uv_lock_sha256": "lock",
        "config_name": "comprehensive_multi_mlip_hull",
        "families": list(cfg.evaluation.mp20.lematgenbench.families),
        "fingerprint_method": "structure-matcher",
        "parse_failures_as_invalid": True,
        "require_clean_repo": True,
        "run_name": "effective-run",
        "family_errors": {},
        "input_integrity_errors": {},
        "local_reference_changes_during_run": {},
        "local_reference_integrity_errors": {},
        "input_count": {
            "expected_cifs": 2,
            "upstream_loaded_structures": 1,
            "dropped_during_cif_parsing": 1,
            "exact_match": False,
            "accepted_as_invalid_attempts": True,
            "error": "one CIF failed to parse",
        },
        "input_cifs": input_cifs,
        "runtime": {
            "device_policy": "cuda",
            "requested_cuda_visible_devices": None,
            "inherited_pythonpath_sanitized": True,
            "provenance": {
                "requested_device_policy": "cuda",
                "cuda_available": True,
                "cuda_device_count": 1,
            },
        },
        "mace_checkpoint": {
            "verification_enabled": True,
            "expected_sha256": "mace",
            "sha256": "mace",
            "post_run_sha256": "mace",
            "post_run_error": None,
            "fallback_to_small_detected": False,
        },
        "oracle_protocol": {
            "identities": oracle_identities,
            "verify_orb_checkpoint": True,
            "expected_orb_checkpoint_sha256": "orb",
            "verify_uma_checkpoint": True,
            "expected_uma_checkpoint_sha256": "uma",
            "runtime_checkpoint_verification_effective": {
                "orb": True,
                "mace": True,
                "uma": True,
            },
            "require_complete_coverage": True,
            "coverage": {
                "valid_structures": 1,
                "status": "complete",
                "all_three_complete": True,
            },
            "coverage_error": None,
        },
        "reference_revision_pins": {
            "enforced": True,
            "revisions": reference_revisions,
            "expected_oracle_digests": {
                "mace": "mace",
                "orb": "orb",
                "uma": "uma",
            },
            "error": None,
            "runtime_access": {
                "verification": "enabled",
                "missing_expected_repositories": [],
                "missing_oracle_artifacts": [],
                "mismatched_oracle_artifacts": [],
            },
        },
        "local_reference_artifacts": {
            "data/convex_hulls/orb.parquet": {"exists": False}
        },
        "local_reference_artifacts_post_run": {
            "data/convex_hulls/orb.parquet": {"exists": False}
        },
    }

    accounting = _validate_lemat_manifest_contract(
        manifest,
        lemat_dir=lemat_dir,
        lemat_cfg=cfg.evaluation.mp20.lematgenbench,
        effective_run_name="effective-run",
        expected_inputs=source_records,
        reference_revisions=reference_revisions,
        oracle_identities=oracle_identities,
    )
    assert accounting == {"attempted": 2, "loaded": 1, "dropped": 1}

    manifest["reference_revision_pins"]["enforced"] = False
    with pytest.raises(ValueError, match="reference revisions"):
        _validate_lemat_manifest_contract(
            manifest,
            lemat_dir=lemat_dir,
            lemat_cfg=cfg.evaluation.mp20.lematgenbench,
            effective_run_name="effective-run",
            expected_inputs=source_records,
            reference_revisions=reference_revisions,
            oracle_identities=oracle_identities,
        )


def test_score_mp20_passes_persisted_cifs_and_keeps_parse_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import zevals.materials.lematgenbench as lematgenbench

    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "output"
    domain_dir = _write_saved_table2_samples(
        output_dir, checkpoint, domain="mp20", count=2
    )
    cfg = _minimal_lemat_cfg(output_dir, checkpoint, reuse_raw_results=False)
    protocol = {"request": {"mp20": {"num_samples": 2}}}
    lemat_dir = domain_dir / "lematgenbench"
    captured = {}
    flat_metrics = {
        "attempt_normalized/overall_valid_rate": 0.5,
        "attempt_normalized/unique_rate": 0.5,
        "attempt_normalized/novel_rate": 0.5,
        "attempt_normalized/stable_rate": 0.5,
        "attempt_normalized/sun_rate": 0.5,
        "attempt_normalized/metasun_rate": 0.5,
        "distribution/JSDistance": 0.1,
        "diversity/element_diversity": 0.2,
    }

    def fake_run(generated_structures, **kwargs):
        captured["input"] = generated_structures
        captured["kwargs"] = kwargs
        lemat_dir.mkdir()
        manifest_path = lemat_dir / "lematgenbench_manifest.json"
        manifest_path.write_text("{}\n")
        raw_path = lemat_dir / "lematgenbench_raw.json"
        raw_path.write_text("{}\n")
        return SimpleNamespace(
            manifest_path=manifest_path,
            raw_result_path=raw_path,
            metrics=flat_metrics,
            artifact_sha256={},
            provenance={},
        )

    monkeypatch.setattr(lematgenbench, "run_lematgenbench", fake_run)
    monkeypatch.setattr(
        table2_evaluation,
        "_validate_lemat_manifest_contract",
        lambda *_args, **_kwargs: {"attempted": 2, "loaded": 1, "dropped": 1},
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_validate_lemat_artifact_hashes",
        lambda *_args, **_kwargs: {"raw": "raw-hash", "manifest": "manifest-hash"},
    )

    report = score_mp20(cfg, protocol)

    assert captured["input"] == domain_dir / "samples"
    assert captured["kwargs"]["parse_failures_as_invalid"] is True
    assert captured["kwargs"]["require_complete_oracle_coverage"] is True
    assert captured["kwargs"]["run_name"].startswith("zatom2-table2-")
    assert report["evaluation"]["num_attempted"] == 2
    assert report["evaluation"]["num_cif_parse_failures"] == 1
    assert report["table2"]["overall_valid"] == 0.5


def test_score_mp20_reuse_recomputes_attempt_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "output"
    domain_dir = _write_saved_table2_samples(
        output_dir, checkpoint, domain="mp20", count=2
    )
    cfg = _minimal_lemat_cfg(output_dir, checkpoint, reuse_raw_results=True)
    protocol = {"request": {"mp20": {"num_samples": 2}}}
    lemat_dir = domain_dir / "lematgenbench"
    lemat_dir.mkdir()
    raw_result = {
        "run_info": {"n_structures": 1},
        "validity_filtering": {
            "total_input_structures": 1,
            "valid_structures": 1,
        },
        "results": {
            "validity": {
                "final_scores": {
                    "total_structures": 1,
                    "overall_validity_count": 1,
                }
            },
            "distribution": {"final_scores": {"JSDistance": 0.1}},
            "diversity": {"final_scores": {"element_diversity": 0.2}},
            "uniqueness": {"final_scores": {"unique_structures_count": 1}},
            "novelty": {"final_scores": {"novel_structures_count": 1}},
            "stability": {"final_scores": {"stable_count": 1}},
            "sun": {"final_scores": {"sun_count": 1, "msun_count": 1}},
        },
    }
    raw_path = lemat_dir / "lematgenbench_raw.json"
    raw_path.write_text(json.dumps(raw_result) + "\n")
    manifest_path = lemat_dir / "lematgenbench_manifest.json"
    manifest_path.write_text("{}\n")
    (lemat_dir / "lematgenbench_metrics.json").write_text("{}\n")
    manifest_hash = _sha256_file(domain_dir / "sample_manifest.json")
    artifacts_hash = _sample_artifacts_sha256(domain_dir)
    (domain_dir / "metrics.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "sample_manifest_sha256": manifest_hash,
                    "sample_artifacts_sha256": artifacts_hash,
                },
                "lematgenbench": {
                    "artifact_sha256": {},
                    "adapter_manifest_sha256": _sha256_file(manifest_path),
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_validate_lemat_manifest_contract",
        lambda *_args, **_kwargs: {"attempted": 2, "loaded": 1, "dropped": 1},
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_validate_lemat_artifact_hashes",
        lambda *_args, **_kwargs: {
            "raw": _sha256_file(raw_path),
            "manifest": _sha256_file(manifest_path),
        },
    )

    report = score_mp20(cfg, protocol)

    assert report["table2"]["overall_valid"] == 0.5
    assert report["table2"]["unique"] == 0.5
    assert report["table2"]["novel"] == 0.5
    assert report["evaluation"]["num_loaded_by_lematgenbench"] == 1
    assert report["evaluation"]["num_cif_parse_failures"] == 1


def test_score_qm9_records_the_enforced_attempt_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output"
    domain_dir = output_dir / "qm9"
    domain_dir.mkdir(parents=True)
    (domain_dir / "sample_manifest.json").write_text("[]\n")
    reference_path = tmp_path / "qm9_reference.json.gz"
    _write_json_gz(reference_path, [])
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(tmp_path / "epoch.ckpt"),
            "out_dir": str(output_dir),
        }
    )
    core_metrics = {
        "protocol": {"name": "zatom1-table2-qm9"},
        "counts": {"generated": 1},
        "denominators": {"valid": 1},
        "zero_denominator_metrics": [],
        "rates": {"valid": 1.0},
        "percent": {"valid": 100.0},
        "digests": {"valid_smiles_sha256": "digest"},
    }
    monkeypatch.setattr(
        table2_evaluation,
        "_compute_qm9_saved_metrics",
        lambda *_args, **_kwargs: json.loads(json.dumps(core_metrics)),
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_sample_artifacts_sha256",
        lambda *_args, **_kwargs: "sample-artifacts",
    )
    monkeypatch.setattr(table2_evaluation, "_qm9_scoring_provenance", lambda: {})

    report = table2_evaluation.score_qm9(
        cfg,
        {"request": {"qm9": {"num_samples": 1}}},
        {"qm9_reference": reference_path},
    )

    assert report["evaluation"]["num_attempted"] == 1
    assert report["evaluation"]["qm9_scoring_revision"] == 1
    assert "reference_artifact_sha256" not in report["evaluation"]
    assert (domain_dir / "metrics.json").is_file()


def test_qm9_report_binding_recomputes_publication_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output"
    domain_dir = output_dir / "qm9"
    samples_dir = domain_dir / "samples"
    samples_dir.mkdir(parents=True)
    cif_path = samples_dir / "qm9-000000.cif.gz"
    metadata_path = samples_dir / "qm9-000000.json"
    cif_path.write_bytes(b"sample")
    metadata_path.write_text("{}\n")
    (domain_dir / "sample_manifest.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "qm9-000000",
                    "cif_path": "samples/qm9-000000.cif.gz",
                    "metadata_path": "samples/qm9-000000.json",
                }
            ]
        )
        + "\n"
    )
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    reference_path = protocol_dir / "qm9_train_canonical_smiles.json.gz"
    _write_json_gz(reference_path, ["[H]C([H])([H])[H]"])
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(tmp_path / "epoch.ckpt"),
            "out_dir": str(output_dir),
            "evaluation": {
                "protocol_dir": str(protocol_dir),
                "qm9": {"num_samples": 1},
            },
        }
    )
    core_metrics = {
        "protocol": {"name": "zatom1-table2-qm9"},
        "counts": {"generated": 1, "valid": 1},
        "denominators": {"valid": 1},
        "zero_denominator_metrics": [],
        "rates": {"valid": 1.0},
        "percent": {"valid": 100.0},
        "digests": {"valid_smiles_sha256": "digest"},
    }
    monkeypatch.setattr(
        table2_evaluation,
        "_compute_qm9_saved_metrics",
        lambda *_args, **_kwargs: json.loads(json.dumps(core_metrics)),
    )
    scoring_provenance = {"modules": {"zevals.molecules.qm9_zatom1": "digest"}}
    monkeypatch.setattr(
        table2_evaluation,
        "_qm9_scoring_provenance",
        lambda *_args, **_kwargs: json.loads(json.dumps(scoring_provenance)),
    )
    metrics = json.loads(json.dumps(core_metrics))
    metrics["evaluation"] = {
        "protocol": table2_evaluation.PROTOCOL_NAME,
        "protocol_version": table2_evaluation.PROTOCOL_VERSION,
        "qm9_scoring_revision": table2_evaluation.QM9_SCORING_REVISION,
        "num_attempted": 1,
        "sample_manifest_sha256": _sha256_file(domain_dir / "sample_manifest.json"),
        "sample_artifacts_sha256": _sample_artifacts_sha256(domain_dir),
        "scoring_provenance": json.loads(json.dumps(scoring_provenance)),
    }

    _verify_qm9_report_binding(cfg, metrics)
    metrics["percent"]["valid"] = 99.0
    with pytest.raises(ValueError, match="fresh recomputation"):
        _verify_qm9_report_binding(cfg, metrics)
    metrics["percent"]["valid"] = 100.0
    metrics["evaluation"]["scoring_provenance"]["modules"] = {}
    with pytest.raises(ValueError, match="metric provenance"):
        _verify_qm9_report_binding(cfg, metrics)


def test_qm9_report_rejects_edm_metrics_before_rescoring(monkeypatch) -> None:
    def unexpected_recomputation(*args, **kwargs):
        pytest.fail("Old metric revisions must be rejected before chemistry scoring")

    monkeypatch.setattr(
        table2_evaluation, "_compute_qm9_saved_metrics", unexpected_recomputation
    )
    with pytest.raises(ValueError, match="evaluation.stages=score_qm9.*resume=false"):
        _verify_qm9_report_binding(
            OmegaConf.create({}),
            {
                "protocol": {"name": "edm-qm9-with-h"},
                "evaluation": {"protocol_version": 3},
            },
        )


def test_qm9_provenance_binds_posebusters_helpers_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posebusters = pytest.importorskip("posebusters")
    pytest.importorskip("openbabel.openbabel")
    root = tmp_path / "posebusters"
    root.mkdir()
    (root / "__init__.py").write_text("# package\n")
    (root / "energy_ratio.py").write_text("# energy implementation\n")
    config = root / "mol.yml"
    config.write_text("threshold_energy_ratio: 100\n")
    monkeypatch.setattr(posebusters, "__file__", str(root / "__init__.py"))
    monkeypatch.setattr(
        table2_evaluation,
        "_scoring_provenance",
        lambda _: {"modules": {}, "packages": {}},
    )
    original = table2_evaluation._qm9_scoring_provenance()
    assert original["posebusters_files"]["energy_ratio.py"] == _sha256_file(
        root / "energy_ratio.py"
    )
    assert original["posebusters_files"]["mol.yml"] == _sha256_file(config)
    assert original["packages"]["posebusters"]
    config.write_text("threshold_energy_ratio: 7\n")
    assert table2_evaluation._qm9_scoring_provenance() != original


def test_qm9_saved_scoring_preserves_failed_attempts_and_explicit_hydrogens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import nucleus.inference_input_parsing as parsing
    import zevals.molecules.qm9_zatom1 as qm9_zatom1

    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "epoch.ckpt"
    _write_saved_table2_samples(output_dir, checkpoint, domain="qm9", count=2)
    cfg = OmegaConf.create({"ckpt_path": str(checkpoint), "out_dir": str(output_dir)})
    atoms = SimpleNamespace(
        coord=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
        element=np.array(["C", "H"]),
    )

    def load_cif(path):
        if "000001" in path.name:
            raise ValueError("malformed generated coordinates")
        return {"atom_array": atoms}

    def score_samples(samples):
        assert len(samples) == 2
        np.testing.assert_array_equal(samples[0][0], [6, 1])
        np.testing.assert_array_equal(samples[0][1], atoms.coord)
        assert samples[1] is None
        return {"counts": {"generated": 2}}

    monkeypatch.setattr(parsing, "inference_load_preserving_hydrogens", load_cif)
    monkeypatch.setattr(
        qm9_zatom1, "compute_qm9_zatom1_generation_metrics", score_samples
    )
    assert table2_evaluation._compute_qm9_saved_metrics(
        cfg, expected_num_samples=2
    ) == {"counts": {"generated": 2}}

    def inaccessible_cif(path):
        raise PermissionError("CIF cannot be read")

    monkeypatch.setattr(
        parsing, "inference_load_preserving_hydrogens", inaccessible_cif
    )
    with pytest.raises(PermissionError, match="cannot be read"):
        table2_evaluation._compute_qm9_saved_metrics(cfg, expected_num_samples=2)


def test_qm9_csv_reports_zatom1_table2_checks(tmp_path: Path) -> None:
    checks = (
        "valid",
        "unique",
        "all_atoms_connected",
        "bond_angles",
        "bond_lengths",
        "aromatic_ring_flatness",
        "double_bond_flatness",
        "internal_energy",
        "internal_steric_clash",
    )
    percentages = {name: 12.5 * index for index, name in enumerate(checks)}
    output = tmp_path / "table2.csv"
    table2_evaluation._write_table_csv(
        output,
        {"percent": percentages},
        {"table2": {"distribution": 0.3}, "table2_percent": {"overall_valid": 80.0}},
    )
    with output.open() as handle:
        row = next(csv.DictReader(handle))
    assert {key for key in row if key.startswith("qm9_")} == {
        f"qm9_{name}_percent" for name in checks
    }
    assert float(row["qm9_internal_energy_percent"]) == 87.5
    assert float(row["mp20_distribution"]) == 0.3


def test_mp20_report_binding_rejects_auxiliary_and_provenance_edits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from zevals.materials.lematgenbench import (
        derive_attempt_normalized_rates,
        extract_lematgenbench_metrics,
        select_table2_metrics,
    )

    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "output"
    domain_dir = _write_saved_table2_samples(
        output_dir, checkpoint, domain="mp20", count=2
    )
    cfg = _minimal_lemat_cfg(output_dir, checkpoint, reuse_raw_results=False)
    lemat_dir = domain_dir / "lematgenbench"
    lemat_dir.mkdir()
    raw_result = {
        "run_info": {"n_structures": 1},
        "validity_filtering": {
            "total_input_structures": 1,
            "valid_structures": 1,
        },
        "results": {
            "validity": {
                "final_scores": {
                    "total_structures": 1,
                    "overall_validity_count": 1,
                }
            },
            "distribution": {"final_scores": {"JSDistance": 0.1}},
            "diversity": {"final_scores": {"element_diversity": 0.2}},
            "uniqueness": {"final_scores": {"unique_structures_count": 1}},
            "novelty": {"final_scores": {"novel_structures_count": 1}},
            "stability": {"final_scores": {"stable_count": 1}},
            "sun": {"final_scores": {"sun_count": 1, "msun_count": 1}},
        },
    }
    raw_path = lemat_dir / "lematgenbench_raw.json"
    raw_path.write_text(json.dumps(raw_result) + "\n")
    flat_metrics = extract_lematgenbench_metrics(raw_result)
    flat_metrics.update(derive_attempt_normalized_rates(flat_metrics, total_attempts=2))
    flat_metrics.update(
        {
            "input_accounting/attempted_structures": 2.0,
            "input_accounting/upstream_loaded_structures": 1.0,
            "input_accounting/cif_parse_failure_count": 1.0,
        }
    )
    flat_metrics = dict(sorted(flat_metrics.items()))
    metrics_path = lemat_dir / "lematgenbench_metrics.json"
    metrics_path.write_text(json.dumps(flat_metrics) + "\n")
    runtime_provenance = {"requested_device_policy": "cuda"}
    oracle_identities = {"orb": {"model": "orb-v3"}}
    oracle_coverage = {
        "valid_structures": 1,
        "status": "complete",
        "all_three_complete": True,
    }
    reference_revisions = {"LeMaterial/LeMat-Bulk": "revision"}
    adapter_manifest = {
        "upstream_commit": "commit",
        "upstream_uv_lock_sha256": "lock",
        "archived_uv_lock_path": str(lemat_dir / "lematgenbench.uv.lock"),
        "upstream_result_path": str(lemat_dir / "upstream.json"),
        "command": ["uv", "run", "scripts/run_benchmarks.py"],
        "family_errors": {},
        "runtime": {
            "provenance": runtime_provenance,
            "provenance_path": str(lemat_dir / "lematgenbench_runtime.json"),
            "provenance_sha256": "runtime-sha256",
        },
        "mace_checkpoint": {
            "model": "medium-mpa-0",
            "path": "/cache/mace-medium.model",
            "sha256": "mace",
        },
        "oracle_protocol": {
            "identities": oracle_identities,
            "coverage": oracle_coverage,
        },
        "reference_revision_pins": {
            "enforced": True,
            "revisions": reference_revisions,
            "shim_path": str(lemat_dir / "shim.py"),
            "shim_sha256": "shim-sha256",
            "audit_path": str(lemat_dir / "audit.jsonl"),
        },
    }
    manifest_path = lemat_dir / "lematgenbench_manifest.json"
    manifest_path.write_text(json.dumps(adapter_manifest) + "\n")
    artifact_sha256 = {
        "raw": _sha256_file(raw_path),
        "manifest": _sha256_file(manifest_path),
    }
    monkeypatch.setattr(
        table2_evaluation,
        "_validate_lemat_manifest_contract",
        lambda *_args, **_kwargs: {"attempted": 2, "loaded": 1, "dropped": 1},
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_validate_lemat_artifact_hashes",
        lambda *_args, **_kwargs: artifact_sha256,
    )
    scoring_provenance = {"modules": {"lematgenbench": "digest"}}
    monkeypatch.setattr(
        table2_evaluation,
        "_scoring_provenance",
        lambda *_args, **_kwargs: scoring_provenance,
    )
    sample_manifest_sha256 = _sha256_file(domain_dir / "sample_manifest.json")
    sample_artifacts_sha256 = _sample_artifacts_sha256(domain_dir)
    effective_run_name = _effective_lemat_run_name(
        "zatom2-table2",
        domain_dir=domain_dir,
        sample_manifest_sha256=sample_manifest_sha256,
        sample_artifacts_sha256=sample_artifacts_sha256,
    )
    provenance = {
        "upstream_commit": "commit",
        "upstream_uv_lock_sha256": "lock",
        "raw_result_path": str(raw_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "stdout_path": str(lemat_dir / "lematgenbench.stdout.log"),
        "stderr_path": str(lemat_dir / "lematgenbench.stderr.log"),
        "uv_lock_path": str(lemat_dir / "lematgenbench.uv.lock"),
        "upstream_result_path": str(lemat_dir / "upstream.json"),
        "command": ["uv", "run", "scripts/run_benchmarks.py"],
        "family_errors": {},
        "expected_input_count": 2,
        "upstream_loaded_count": 1,
        "parse_failure_count": 1,
        "mace_model": "medium-mpa-0",
        "mace_checkpoint_path": "/cache/mace-medium.model",
        "mace_checkpoint_sha256": "mace",
        "reference_revisions_enforced": True,
        "reference_revisions": reference_revisions,
        "reference_pin_shim_path": str(lemat_dir / "shim.py"),
        "reference_pin_shim_sha256": "shim-sha256",
        "reference_pin_audit_path": str(lemat_dir / "audit.jsonl"),
        "runtime_environment": runtime_provenance,
        "runtime_environment_path": str(lemat_dir / "lematgenbench_runtime.json"),
        "runtime_environment_sha256": "runtime-sha256",
        "artifact_sha256": artifact_sha256,
        "oracle_identities": oracle_identities,
        "oracle_coverage": oracle_coverage,
        "reused_raw_results": False,
        "raw_result_sha256": artifact_sha256["raw"],
        "adapter_manifest_sha256": artifact_sha256["manifest"],
        "effective_run_name": effective_run_name,
    }
    table2 = select_table2_metrics(
        flat_metrics,
        distribution_key="distribution/JSDistance",
        diversity_key="diversity/element_diversity",
    )
    report = {
        "evaluation": {
            "protocol": table2_evaluation.PROTOCOL_NAME,
            "protocol_version": table2_evaluation.PROTOCOL_VERSION,
            "num_attempted": 2,
            "num_loaded_by_lematgenbench": 1,
            "num_cif_parse_failures": 1,
            "sample_manifest_sha256": sample_manifest_sha256,
            "sample_artifacts_sha256": sample_artifacts_sha256,
            "table_projection": {
                "rate_denominator": (
                    "all attempted structures (Zatom-1 Table convention)"
                ),
                "distribution_key": "distribution/JSDistance",
                "diversity_key": "diversity/element_diversity",
            },
            "scoring_provenance": scoring_provenance,
        },
        "table2": table2,
        "table2_percent": {
            key: 100.0 * value
            for key, value in table2.items()
            if key in {"overall_valid", "unique", "novel", "stable", "sun", "metasun"}
        },
        "native_lemat_rates": table2_evaluation._native_lemat_rates(flat_metrics),
        "distribution": table2_evaluation._lemat_component_group(
            flat_metrics, "distribution"
        ),
        "diversity": table2_evaluation._lemat_component_group(
            flat_metrics, "diversity"
        ),
        "flat_lemat_metrics": flat_metrics,
        "lematgenbench": provenance,
    }

    table2_evaluation._verify_mp20_lemat_report_binding(cfg, report)
    report["distribution"]["JSDistance"] = 0.9
    with pytest.raises(ValueError, match="auxiliary metrics"):
        table2_evaluation._verify_mp20_lemat_report_binding(cfg, report)
    report["distribution"]["JSDistance"] = 0.1
    report["lematgenbench"]["upstream_commit"] = "edited"
    with pytest.raises(ValueError, match="not bound"):
        table2_evaluation._verify_mp20_lemat_report_binding(cfg, report)


@pytest.mark.parametrize("domain", ["qm9", "mp20"])
def test_dataset_manifest_requires_exact_checkpoint_selection(
    domain: str, monkeypatch, tmp_path: Path
) -> None:
    contract = {
        **CANONICAL_TRAINING_DATASETS[domain],
        "shards": list(CANONICAL_TRAINING_DATASETS[domain]["shards"]),
        "shard_sha256": {},
    }
    data_dir = tmp_path / domain
    data_dir.mkdir()
    for shard_name in contract["shards"]:
        (data_dir / shard_name).write_bytes(shard_name.encode("ascii"))
        contract["shard_sha256"][shard_name] = _sha256_file(data_dir / shard_name)
    monkeypatch.setitem(CANONICAL_TRAINING_DATASETS, domain, contract)
    manifest = {"dataset": domain, **contract}
    (data_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")

    frozen = _dataset_manifest(data_dir, domain, int(contract["num_examples"]))
    assert (
        frozen["manifest"]["selected_indices_sha256"]
        == contract["selected_indices_sha256"]
    )

    first_shard = data_dir / contract["shards"][0]
    original_shard = first_shard.read_bytes()
    first_shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard does not match"):
        _dataset_manifest(data_dir, domain, int(contract["num_examples"]))
    first_shard.write_bytes(original_shard)

    manifest["selected_indices_sha256"] = "0" * 64
    (data_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="exact Table-2 checkpoint selection"):
        _dataset_manifest(data_dir, domain, int(contract["num_examples"]))


def test_stage_normalization_is_ordered() -> None:
    assert _normalize_stages(["report", "sample", "score_qm9"]) == (
        "sample",
        "score_qm9",
        "report",
    )
    assert _normalize_stages("all") == (
        "prepare",
        "sample",
        "score_qm9",
        "score_mp20",
        "report",
    )


def test_table2_hydra_config_freezes_checkpoint_sampler(
    monkeypatch, tmp_path: Path
) -> None:
    project_root = Path(__file__).parents[3]
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    config_dir = project_root / "models" / "zatom2" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="table2_evaluation",
            overrides=[
                "ckpt_path=/tmp/checkpoint.ckpt",
                f"out_dir={tmp_path}",
            ],
        )

    assert cfg.seed == 42
    assert cfg.use_ema_weights
    assert cfg.diffusion_batch_size == 8
    assert cfg.inference_sampler.solver == "af3"
    assert cfg.inference_sampler.num_timesteps == 100
    assert cfg.inference_sampler.gamma_0 == 0.8
    assert cfg.inference_sampler.n_recycle == 2
    assert not cfg.inference_sampler.use_classifier_free_guidance
    assert cfg.evaluation.qm9.num_samples == 10_000
    assert cfg.evaluation.mp20.num_samples == 2_500
    assert cfg.evaluation.mp20.lematgenbench.parse_failures_as_invalid
    assert cfg.evaluation.mp20.lematgenbench.runtime_device == "cuda"
    assert cfg.evaluation.mp20.lematgenbench.enforce_reference_revisions
    assert cfg.evaluation.mp20.lematgenbench.require_complete_oracle_coverage
    assert _validate_protocol_conformance(cfg) == {}

    OmegaConf.update(
        cfg,
        "specification.extra",
        {"mean_force_norm": 0.25},
        force_add=True,
    )
    with pytest.raises(ValueError, match="frozen manuscript protocol"):
        _validate_protocol_conformance(cfg)
    cfg.specification = {}

    OmegaConf.update(
        cfg,
        "transform_overrides.use_element_map",
        True,
        force_add=True,
    )
    with pytest.raises(ValueError, match="frozen manuscript protocol"):
        _validate_protocol_conformance(cfg)
    cfg.transform_overrides = {"remove_hydrogens": None}

    OmegaConf.update(
        cfg,
        "inference_sampler.unexpected_toggle",
        True,
        force_add=True,
    )
    with pytest.raises(ValueError, match="frozen manuscript protocol"):
        _validate_protocol_conformance(cfg)
    cfg.inference_sampler = OmegaConf.create(table2_evaluation.FROZEN_INFERENCE_SAMPLER)

    cfg.inference_sampler.num_timesteps = 99
    with pytest.raises(ValueError, match="frozen manuscript protocol"):
        _validate_protocol_conformance(cfg)
    cfg.evaluation.allow_noncanonical_protocol = True
    mismatch = _validate_protocol_conformance(cfg)["inference_sampler"]
    assert mismatch["configured"]["num_timesteps"] == 99
    assert mismatch["expected"]["num_timesteps"] == 100
    with pytest.raises(ValueError, match="Refusing to publish"):
        table2_evaluation.write_report(cfg)


def test_input_manifest_is_plain_json(tmp_path: Path) -> None:
    manifest = build_input_manifest(
        "mp20", [5], num_samples=1, seed=42, diffusion_batch_size=8
    )
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest))

    loaded = json.loads(path.read_text())
    assert loaded == manifest
    group_id = next(iter(loaded))
    assert torch.tensor([int(loaded[group_id]["contig"][:-1])]).item() == 5


def test_table2_rejects_divergent_generation_and_protocol_seeds() -> None:
    cfg = OmegaConf.create({"seed": 1, "evaluation": {"seed": 2}})

    with pytest.raises(ValueError, match="must match"):
        run_table2_evaluation(cfg)


def test_frozen_protocol_rejects_changed_staged_config(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "diffusion_batch_size": 8,
            "n_batches": 1,
            "evaluation": {
                "seed": 42,
                "qm9": {
                    "num_samples": 10_000,
                    "training_num_examples": 100_000,
                    "training_data_dir": str(tmp_path / "qm9"),
                },
                "mp20": {
                    "num_samples": 2_500,
                    "training_num_examples": 27_138,
                    "training_data_dir": str(tmp_path / "mp20"),
                },
            },
        }
    )
    manifest = {
        "request": {
            "protocol": "zatom2-paper-table2-qm9-mp20",
            # Version 3 freezes independent RNG streams for resumable groups.
            "protocol_version": table2_evaluation.PROTOCOL_VERSION,
            "seed": 42,
            "inference_batching": {
                "diffusion_batch_size": 8,
                "n_batches": 1,
                "strategy": "group_equal_node_counts_and_discard_tail_padding",
            },
            "node_count_sampling": {"source": "exact_training_partition"},
            "qm9_metrics": {
                "novelty_reference": ("EDM full-graph exact 100k training geometries")
            },
            "qm9": {
                "num_samples": 10_000,
                "training_num_examples": 100_000,
                "training_data": {"path": str((tmp_path / "qm9").resolve())},
            },
            "mp20": {
                "num_samples": 2_500,
                "training_num_examples": 27_138,
                "training_data": {"path": str((tmp_path / "mp20").resolve())},
            },
        }
    }

    _verify_protocol_request(cfg, manifest)
    cfg.evaluation.mp20.num_samples = 2_501
    with pytest.raises(ValueError, match="does not match"):
        _verify_protocol_request(cfg, manifest)


def test_report_bindings_detect_sample_artifact_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        table2_evaluation,
        "_verify_qm9_report_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        table2_evaluation,
        "_verify_mp20_lemat_report_binding",
        lambda *_args, **_kwargs: None,
    )
    checkpoint = tmp_path / "epoch.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    protocol_manifest = protocol_dir / "protocol_manifest.json"
    protocol_manifest.write_text("{}\n")
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(checkpoint),
            "out_dir": str(tmp_path / "output"),
            "evaluation": {"protocol_dir": str(protocol_dir)},
        }
    )
    run_domains = {}
    metrics = {}
    for domain in ("qm9", "mp20"):
        domain_dir = Path(cfg.out_dir) / domain
        samples_dir = domain_dir / "samples"
        samples_dir.mkdir(parents=True)
        (samples_dir / "sample.cif.gz").write_bytes(b"cif")
        (samples_dir / "sample.json").write_text("{}\n")
        sample_manifest = [
            {
                "sample_id": f"{domain}-000000",
                "cif_path": "samples/sample.cif.gz",
                "metadata_path": "samples/sample.json",
            }
        ]
        manifest_path = domain_dir / "sample_manifest.json"
        manifest_path.write_text(json.dumps(sample_manifest) + "\n")
        manifest_hash = _sha256_file(manifest_path)
        artifacts_hash = _sample_artifacts_sha256(domain_dir)
        run_domains[domain] = {
            "sample_manifest_sha256": manifest_hash,
            "sample_artifacts_sha256": artifacts_hash,
        }
        metrics[domain] = {
            "evaluation": {
                "sample_manifest_sha256": manifest_hash,
                "sample_artifacts_sha256": artifacts_hash,
            }
        }
    run_manifest = {
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
        },
        "protocol": {
            "name": table2_evaluation.PROTOCOL_NAME,
            "version": table2_evaluation.PROTOCOL_VERSION,
            "manifest_sha256": _sha256_file(protocol_manifest),
            "canonical_runtime": True,
            "runtime_mismatches": {},
        },
        "domains": run_domains,
    }

    _verify_report_bindings(cfg, run_manifest=run_manifest, domain_metrics=metrics)
    run_manifest["protocol"]["canonical_runtime"] = False
    run_manifest["protocol"]["runtime_mismatches"] = {"seed": {"configured": 1}}
    with pytest.raises(ValueError, match="noncanonical"):
        _verify_report_bindings(cfg, run_manifest=run_manifest, domain_metrics=metrics)
    run_manifest["protocol"]["canonical_runtime"] = True
    run_manifest["protocol"]["runtime_mismatches"] = {}

    checkpoint.write_bytes(b"checkpoinu")
    with pytest.raises(ValueError, match="contents changed"):
        _verify_report_bindings(cfg, run_manifest=run_manifest, domain_metrics=metrics)
    checkpoint.write_bytes(b"checkpoint")

    (Path(cfg.out_dir) / "mp20" / "samples" / "sample.cif.gz").write_bytes(b"changed")
    # Simulate rescoring the changed artifact. The fresh metric hash must still
    # be rejected because the immutable generation manifest binds the old CIF.
    metrics["mp20"]["evaluation"]["sample_artifacts_sha256"] = _sample_artifacts_sha256(
        Path(cfg.out_dir) / "mp20"
    )
    with pytest.raises(ValueError, match="not hash-bound"):
        _verify_report_bindings(cfg, run_manifest=run_manifest, domain_metrics=metrics)
