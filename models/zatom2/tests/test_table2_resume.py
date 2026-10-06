"""Resume sampling across interrupted allocations without changing RNG streams."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import zatom2.table2_evaluation as table2
import zatom2.table2_resume as resume
from zatom2.generation_evaluation import GeneratedSample, _load_input_specifications


@pytest.fixture
def sampling_case(monkeypatch, tmp_path):
    import zatom2.run_inference as run_inference

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    paths = {"manifest": protocol_dir / "protocol_manifest.json"}
    paths["manifest"].write_text("{}\n")
    plans = {}
    for domain in table2.DOMAIN_ORDER:
        inputs = table2.build_input_manifest(
            domain, [5], num_samples=5, seed=42, diffusion_batch_size=2
        )
        paths[f"{domain}_inputs"] = protocol_dir / f"{domain}_inputs.json"
        paths[f"{domain}_inputs"].write_text(json.dumps(inputs))
        plans[domain] = table2._sampling_plan_metadata(
            domain,
            _load_input_specifications(paths[f"{domain}_inputs"]),
            diffusion_batch_size=2,
            expected_num_samples=5,
        )
    protocol = {
        "sampling_plans": plans,
        "request": {domain: {"num_samples": 5} for domain in table2.DOMAIN_ORDER},
    }
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(checkpoint),
            "out_dir": str(tmp_path / "interrupted"),
            "inputs": None,
            "json_keys_subset": None,
            "num_nodes": 1,
            "devices_per_node": 1,
            "diffusion_batch_size": 2,
            "inference_sampler": {"num_timesteps": 100},
            "use_ema_weights": True,
            "seed": 42,
            "evaluation": {
                "resume": True,
                "overwrite": False,
                "seed": 42,
                "hash_checkpoint": True,
                "allow_noncanonical_protocol": True,
            },
        }
    )
    events = SimpleNamespace(initializations=0, calls=[], interrupt_dump=None)

    class FakeEngine:
        def __init__(self, engine_cfg):
            self.output_name = Path(engine_cfg.out_dir).name
            self.trainer = SimpleNamespace(fabric=SimpleNamespace(world_size=1))

        def initialize(self):
            events.initializations += 1
            # Checkpoint construction consumes the global RNG on every restart.
            torch.rand(17)

        def run(self, *, inputs, n_batches, out_dir):
            assert n_batches == 1 and out_dir is None
            domain = Path(inputs).stem.removesuffix("_inputs")
            specifications = _load_input_specifications(inputs)
            assert len(self.json_keys_subset) == 1
            group_id = self.json_keys_subset[0]
            events.calls.append((self.output_name, domain, group_id))
            outputs = []
            for model_index in range(2):
                example_id = f"{group_id}_0_model_{model_index}"
                output = SimpleNamespace(
                    example_id=example_id,
                    metadata={"draw": float(torch.rand(()))},
                    atom_array=None,
                )
                outputs.append(
                    GeneratedSample(
                        sample_index=model_index,
                        sample_id=example_id,
                        target_id=group_id,
                        batch_index=0,
                        model_index=model_index,
                        specification=specifications[group_id],
                        output=output,
                    )
                )
            return outputs

    def dump_samples(samples, domain_dir):
        samples_dir = domain_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for sample in samples:
            cif_path = samples_dir / f"{sample.sample_id}.cif.gz"
            metadata_path = samples_dir / f"{sample.sample_id}.json"
            cif_path.write_bytes(str(sample.output.metadata["draw"]).encode())
            event = (domain_dir.parent.name, domain_dir.name, sample.target_id)
            if events.interrupt_dump == event:
                events.interrupt_dump = None
                # Simulate the allocation disappearing after the CIF but before
                # metadata and the group's completion marker are durable.
                raise resume.EvaluationPaused("simulated interruption during dump")
            metadata_path.write_text(
                json.dumps(
                    {**sample.output.metadata, "ckpt_path": str(checkpoint)},
                    sort_keys=True,
                )
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

    monkeypatch.setattr(run_inference, "build_inference_engine", FakeEngine)
    monkeypatch.setattr(table2, "_dump_samples", dump_samples)
    monkeypatch.setattr(table2, "_flatten_outputs", lambda outputs, *_a, **_k: outputs)
    monkeypatch.setattr(
        table2,
        "_checkpoint_provenance",
        lambda config: {
            "path": str(checkpoint),
            "sha256": table2._sha256_file(Path(config.ckpt_path)),
        },
    )
    monkeypatch.setattr(table2, "_repository_provenance", lambda: {})
    monkeypatch.setattr(table2, "_scoring_provenance", lambda *_a: {})
    monkeypatch.setattr(
        table2, "sampling_source_hashes", lambda: {"nucleus": "original"}
    )
    monkeypatch.setattr(
        table2,
        "_software_provenance",
        lambda: {"cuda_devices": ["NVIDIA A100-SXM4-80GB"], "hostname": "nid00001"},
    )
    return SimpleNamespace(
        cfg=cfg,
        protocol=protocol,
        paths=paths,
        checkpoint=checkpoint,
        events=events,
        run_inference=run_inference,
    )


def test_partial_group_recomputed_with_identical_fresh_samples(sampling_case):
    case = sampling_case
    groups = list(_load_input_specifications(case.paths["qm9_inputs"]))
    case.events.interrupt_dump = ("interrupted", "qm9", groups[1])
    with pytest.raises(resume.EvaluationPaused, match="simulated interruption"):
        table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    interrupted = Path(case.cfg.out_dir)
    markers = list((interrupted / "qm9/.sampling_progress").glob("*.json"))
    assert len(markers) == 1
    assert not (interrupted / "run_manifest.json").exists()
    assert len(list((interrupted / "qm9/samples").glob("*.cif.gz"))) == 3

    table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    assert case.events.calls.count(("interrupted", "qm9", groups[0])) == 1
    assert case.events.calls.count(("interrupted", "qm9", groups[1])) == 2
    fresh = interrupted.with_name("fresh")
    case.cfg.out_dir = str(fresh)
    table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    for domain in table2.DOMAIN_ORDER:
        manifest = json.loads((fresh / domain / "sample_manifest.json").read_text())
        assert len(manifest) == 5  # D2 tail padding is discarded in both runs.
        assert manifest == json.loads(
            (interrupted / domain / "sample_manifest.json").read_text()
        )
        for artifact in (fresh / domain / "samples").iterdir():
            assert (
                artifact.read_bytes()
                == (interrupted / domain / "samples" / artifact.name).read_bytes()
            )


def test_completed_sampling_skips_engine_initialization(monkeypatch, sampling_case):
    case = sampling_case
    first, _ = table2.sample_checkpoint(case.cfg, case.protocol, case.paths)

    def cannot_initialize(_cfg):
        pytest.fail("Completed sampling must not construct the inference engine")

    monkeypatch.setattr(case.run_inference, "build_inference_engine", cannot_initialize)
    second, is_rank_zero = table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    assert first == second
    assert is_rank_zero
    assert case.events.initializations == 1


def test_resume_allows_new_node_hostname(monkeypatch, sampling_case):
    case = sampling_case
    table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    monkeypatch.setattr(
        table2,
        "_software_provenance",
        lambda: {"cuda_devices": ["NVIDIA A100-SXM4-80GB"], "hostname": "nid99999"},
    )
    table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    assert case.events.initializations == 1


@pytest.mark.parametrize(
    "change",
    ["checkpoint", "input", "sampler", "seed", "diffusion_seed", "gpu", "source"],
)
def test_changed_resume_contract_rejected(monkeypatch, sampling_case, change):
    case = sampling_case
    table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    if change == "checkpoint":
        case.checkpoint.write_bytes(b"different checkpoint")
    elif change == "input":
        with case.paths["qm9_inputs"].open("a") as stream:
            stream.write("\n")
    elif change == "sampler":
        case.cfg.inference_sampler.num_timesteps = 200
    elif change == "seed":
        case.cfg.seed = case.cfg.evaluation.seed = 43
    elif change == "diffusion_seed":
        case.cfg.evaluation.seed = 43
    elif change == "gpu":
        monkeypatch.setattr(
            table2, "_software_provenance", lambda: {"cuda_devices": ["other GPU"]}
        )
    elif change == "source":
        monkeypatch.setattr(
            table2, "sampling_source_hashes", lambda: {"nucleus": "changed"}
        )
    with pytest.raises(ValueError, match="resume contract changed"):
        table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    assert case.events.initializations == 1


def test_corrupt_committed_sample_rejected(sampling_case):
    case = sampling_case
    table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    artifact = Path(case.cfg.out_dir) / "qm9/samples/qm9-000000.cif.gz"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Committed sample artifacts changed"):
        table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    assert case.events.initializations == 1


def test_walltime_exhaustion_pauses_before_next_group(monkeypatch, sampling_case):
    case = sampling_case
    now = [100.0]
    monkeypatch.setattr(resume.time, "monotonic", lambda: now[0])
    original_commit = table2.commit_group

    def expire_after_commit(*args, **kwargs):
        original_commit(*args, **kwargs)
        now[0] = 161.0

    monkeypatch.setattr(table2, "commit_group", expire_after_commit)
    output = Path(case.cfg.out_dir)
    with resume.evaluation_allocation(output, 60):
        with pytest.raises(resume.EvaluationPaused, match="budget exhausted"):
            table2.sample_checkpoint(case.cfg, case.protocol, case.paths)
    assert len(case.events.calls) == 1
    assert len(list((output / "qm9/.sampling_progress").glob("*.json"))) == 1
    assert not (output / "run_manifest.json").exists()
    assert resume.remaining_seconds() is None


def test_remaining_budget_caps_stage_timeout_and_resets(monkeypatch, tmp_path):
    now = [100.0]
    monkeypatch.setattr(resume.time, "monotonic", lambda: now[0])
    with resume.evaluation_allocation(tmp_path, 60):
        assert resume.remaining_seconds() == 60
        assert resume.remaining_seconds(30) == 30
        now[0] += 40
        assert resume.remaining_seconds(30) == 20
        now[0] += 20
        with pytest.raises(resume.EvaluationPaused):
            resume.remaining_seconds()
    assert resume.remaining_seconds(30) == 30


def test_allocation_lock_released_after_exception(tmp_path):
    with pytest.raises(RuntimeError, match="simulated failure"):
        with resume.evaluation_allocation(tmp_path, None):
            with pytest.raises(RuntimeError, match="Another evaluation"):
                with resume.evaluation_allocation(tmp_path, None):
                    pytest.fail("Concurrent writer acquired the lock")
            raise RuntimeError("simulated failure")
    with resume.evaluation_allocation(tmp_path, None):
        assert resume.remaining_seconds() is None


def test_atomic_json_never_replaces_valid_marker_with_invalid_json(tmp_path):
    path = tmp_path / "marker.json"
    resume.atomic_json(path, {"committed": True})
    with pytest.raises(ValueError):
        resume.atomic_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"committed": True}


def test_group_seeds_separate_domains_and_groups():
    values = {
        resume.group_seed(42, domain, f"group-{index}")
        for domain in ("qm9", "mp20")
        for index in range(10)
    }
    assert len(values) == 20
    assert resume.group_seed(42, "qm9", "group-0") == resume.group_seed(
        42, "qm9", "group-0"
    )
