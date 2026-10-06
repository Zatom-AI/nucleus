from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.materials import _lemat_resume as resume  # noqa: E402
from zevals.materials import _lemat_resume_runtime as runtime  # noqa: E402
from zevals.materials import lematgenbench as lemat  # noqa: E402


@pytest.fixture
def cache_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEVALS_LEMAT_RESUME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ZEVALS_LEMAT_RESUME_CONTRACT", "test-contract")
    return tmp_path / "cache"


def test_oracle_cache_retries_interrupted_and_partial_computations(cache_environment):
    calls = []

    def interrupted():
        calls.append("interrupted")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runtime.cached_call("oracle-orb", {"structure": 1}, interrupted)
    assert not list(cache_environment.rglob("*.json"))
    for _ in range(2):
        runtime.cached_call(
            "oracle-orb",
            {"structure": 1},
            lambda: calls.append("partial"),
            valid=lambda result: result is not None,
        )
    assert not list(cache_environment.rglob("*.json"))
    value = runtime.cached_call(
        "oracle-orb",
        {"structure": 1},
        lambda: calls.append("complete") or {"energy": -2},
    )
    assert value == {"energy": -2}
    assert runtime.cached_call("oracle-orb", {"structure": 1}, interrupted) == value
    assert calls == ["interrupted", "partial", "partial", "complete"]
    # Each oracle has independent progress for the same structure.
    with pytest.raises(KeyboardInterrupt):
        runtime.cached_call("oracle-mace", {"structure": 1}, interrupted)


def test_cache_checksum_and_identity_are_verified(cache_environment, monkeypatch):
    runtime.cached_call("family", [1, 2], lambda: {"count": 2})
    payload = next(cache_environment.rglob("*.pickle"))
    payload.write_bytes(b"corrupted")
    with pytest.raises(runtime.CacheIntegrityError, match="checksum mismatch"):
        runtime.cached_call("family", [1, 2], lambda: None)
    monkeypatch.setenv("ZEVALS_LEMAT_RESUME_CONTRACT", "different-runtime")
    assert runtime.cached_call("family", [1, 2], lambda: {"count": 3}) == {"count": 3}


def test_upstream_validity_wall_clock_does_not_invalidate_cache(cache_environment):
    first = {"properties": {"validity_timestamp": "100", "overall_valid": True}}
    second = {"properties": {"validity_timestamp": "200", "overall_valid": True}}
    assert runtime.cached_call("oracle", first, lambda: 1) == 1
    assert runtime.cached_call("oracle", second, lambda: 2) == 1
    second["properties"]["overall_valid"] = False
    assert runtime.cached_call("oracle", second, lambda: 3) == 3


def test_interrupted_row_publication_preserves_previous_commit(tmp_path, monkeypatch):
    record = tmp_path / "row.json"
    runtime._publish(record, "row", "contract", {0: True})
    write = runtime._atomic_bytes

    def interrupted(path, payload):
        if path == record:
            raise KeyboardInterrupt
        write(path, payload)

    monkeypatch.setattr(runtime, "_atomic_bytes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runtime._publish(record, "row", "contract", {0: True, 1: False})
    assert runtime._cached_locked(
        record, "row", "contract", "rows", lambda: None, None
    ) == {0: True}


def test_runtime_hooks_cache_whole_family_not_chunk_metrics(
    cache_environment, monkeypatch
):
    monkeypatch.setenv("ZEVALS_LEMAT_INTERPRETER_ROLE", "benchmark")
    calls = []

    class Structure:
        def __init__(self, value):
            self.value = value

        def as_dict(self):
            return {"value": self.value}

    class Benchmark:
        def evaluate(self, structures):
            calls.append([structure.value for structure in structures])
            return {"unique_count": len({structure.value for structure in structures})}

    modules = {
        "distribution_benchmark": "DistributionBenchmark",
        "diversity_benchmark": "DiversityBenchmark",
        "hhi_benchmark": "HHIBenchmark",
        "multi_mlip_stability_benchmark": "StabilityBenchmark",
        "novelty_benchmark": "NoveltyBenchmark",
        "sun_benchmark": "SUNBenchmark",
        "uniqueness_benchmark": "UniquenessBenchmark",
    }
    for name in (
        "lemat_genbench",
        "lemat_genbench.preprocess",
        "lemat_genbench.benchmarks",
        "lemat_genbench.metrics",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    multi = ModuleType("lemat_genbench.preprocess.multi_mlip_preprocess")
    oracle_calls = []

    def oracle(
        structure,
        calculator,
        mlip_name,
        relax_structures,
        relaxation_config,
        calculate_formation_energy,
        calculate_energy_above_hull,
        extract_embeddings,
    ):
        oracle_calls.append(mlip_name)
        return {
            "energy": float("nan") if len(oracle_calls) == 1 else -2.0,
            "forces": [0.0],
        }

    multi._process_single_mlip = oracle
    monkeypatch.setitem(sys.modules, multi.__name__, multi)
    classes = {}
    for module_name, class_name in modules.items():
        module = ModuleType(f"lemat_genbench.benchmarks.{module_name}")
        classes[class_name] = type(class_name, (Benchmark,), {})
        setattr(module, class_name, classes[class_name])
        monkeypatch.setitem(sys.modules, module.__name__, module)
    novelty_calls = []

    class Novelty:
        def __init__(self, reference="all"):
            self.config = SimpleNamespace(reference=reference)

        def _load_reference_dataset(self):
            return {"dataset": SimpleNamespace(_fingerprint="frozen-reference")}

        @staticmethod
        def compute_structure(structure, information, fingerprinter, verbose=False):
            novelty_calls.append(structure.value)
            return float(structure.value != 1)

    class Unique:
        def compute(self, structures):
            return None

    class SUN:
        def _compute_uniqueness_within_set(self, structures):
            return None

    for name, cls_name, cls in (
        ("novelty_metric", "NoveltyMetric", Novelty),
        ("uniqueness_metric", "UniquenessMetric", Unique),
        ("sun_metric", "SUNMetric", SUN),
    ):
        module = ModuleType(f"lemat_genbench.metrics.{name}")
        setattr(module, cls_name, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime.install()
    oracle_arguments = (Structure(1), None, "orb", False, {}, False, False, False)
    assert str(multi._process_single_mlip(*oracle_arguments)["energy"]) == "nan"
    assert multi._process_single_mlip(*oracle_arguments)["energy"] == -2
    assert multi._process_single_mlip(*oracle_arguments)["energy"] == -2
    assert oracle_calls == ["orb", "orb"]
    benchmark = classes["UniquenessBenchmark"]()
    duplicate_set = [Structure(1), Structure(1)]
    assert benchmark.evaluate(duplicate_set) == {"unique_count": 1}
    assert benchmark.evaluate(duplicate_set) == {"unique_count": 1}
    assert benchmark.evaluate([Structure(1), Structure(2)]) == {"unique_count": 2}
    assert calls == [[1, 1], [1, 2]]
    novelty = Novelty()
    info = novelty._load_reference_dataset()
    assert novelty.compute_structure(Structure(1), info, None) == 0
    assert novelty.compute_structure(Structure(1), info, None) == 0
    changed = Novelty("subset")._load_reference_dataset()
    assert novelty.compute_structure(Structure(1), changed, None) == 0
    assert novelty_calls == [1, 1]


def test_uniqueness_rows_preserve_order_and_survive_interrupted_row(cache_environment):
    class Structure:
        def __init__(self, value):
            self.value = value

        def as_dict(self):
            return {"value": self.value}

    class Matcher:
        def __init__(self, interrupt=None):
            self.calls = []
            self.interrupt = interrupt

        def is_equivalent(self, first, second):
            if len(self.calls) == self.interrupt:
                raise KeyboardInterrupt
            self.calls.append((first.value, second.value))
            return first.value == second.value

    structures = [Structure(value) for value in [1, 2, 1, 3, 2]]

    # These are the pinned upstream's distinct representative conventions:
    # ordinary uniqueness compares every later item; SUN retains first matches.
    def unique(matcher):
        return [
            i
            for i, first in enumerate(structures)
            if not any(
                [
                    matcher.is_equivalent(first, second)
                    for j, second in enumerate(structures)
                    if i < j
                ]
            )
        ]

    def sun_unique(matcher):
        indices = []
        for i, first in enumerate(structures):
            if not any(matcher.is_equivalent(first, structures[j]) for j in range(i)):
                indices.append(i)
        return indices

    for namespace, algorithm, expected in (
        ("unique", unique, [2, 3, 4]),
        ("sun", sun_unique, [0, 1, 3]),
    ):
        baseline = Matcher()
        assert algorithm(baseline) == expected
        initial = Matcher(interrupt=5)
        proxy = runtime._ComparisonRows(initial, structures, namespace, {})
        with pytest.raises(KeyboardInterrupt):
            algorithm(proxy)
        # Simulate a hard allocation kill: the active row never gets flushed.
        resumed = Matcher()
        proxy = runtime._ComparisonRows(resumed, structures, namespace, {})
        assert algorithm(proxy) == expected
        proxy.flush()
        assert len(resumed.calls) < len(baseline.calls)
        # The cache never changes the relative order of uncached comparisons.
        assert resumed.calls == baseline.calls[-len(resumed.calls) :]
        warm = Matcher()
        proxy = runtime._ComparisonRows(warm, structures, namespace, {})
        assert algorithm(proxy) == expected
        proxy.flush()
        assert warm.calls == []
        assert len(list((cache_environment / namespace).glob("*.json"))) <= len(
            structures
        )


def test_runtime_contract_allows_reallocated_identical_gpu_only():
    original = {
        "effective_cuda_visible_devices": "GPU-first",
        "torch_version": "2",
        "cuda_devices": [{"name": "A100", "uuid": "first", "total_memory_bytes": 80}],
    }
    other = {
        "effective_cuda_visible_devices": "GPU-next",
        "torch_version": "2",
        "cuda_devices": [{"name": "A100", "uuid": "next", "total_memory_bytes": 80}],
    }
    assert resume.runtime_contract(original) == resume.runtime_contract(other)
    other["cuda_devices"][0]["total_memory_bytes"] = 40
    assert resume.runtime_contract(original) != resume.runtime_contract(other)


def test_archive_can_resume_a_kill_during_moves(tmp_path, monkeypatch):
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    first.write_text("first")
    second.write_text("second")
    original = Path.rename
    calls = 0

    def rename(path, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(path, destination)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(KeyboardInterrupt):
        resume.archive_attempt(tmp_path, {"a": first, "b": second})
    monkeypatch.setattr(Path, "rename", original)
    history = resume.archive_attempt(tmp_path, {"a": first, "b": second})
    assert len(history) == 1
    assert history[0]["status"] == "archived"
    assert (Path(history[0]["directory"]) / "a.log").read_text() == "first"
    assert (Path(history[0]["directory"]) / "b.log").read_text() == "second"


def test_timeout_preserves_logs_and_stops_descendant_processes(tmp_path):
    pid_path = tmp_path / "child.pid"
    source = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        f"Path({str(pid_path)!r}).write_text(str(p.pid))\n"
        "print('progress persisted',flush=True)\n"
        "time.sleep(30)\n"
    )
    stdout, stderr = tmp_path / "stdout", tmp_path / "stderr"
    with pytest.raises(resume.LeMatResumeTimeout):
        resume.run_streamed(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            environment=os.environ,
            stdout_path=stdout,
            stderr_path=stderr,
            timeout_seconds=0.7,
        )
    assert "progress persisted" in stdout.read_text()
    pid = int(pid_path.read_text())
    status = Path(f"/proc/{pid}/stat")
    for _ in range(20):
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("LeMat grandchild survived process-group timeout")


def test_user_interrupt_is_cancellation_not_resumable_timeout(tmp_path):
    source = (
        "import os,signal,time\n"
        "print('ready',flush=True)\n"
        "os.kill(os.getppid(),signal.SIGINT)\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(KeyboardInterrupt):
        resume.run_streamed(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            environment=os.environ,
            stdout_path=tmp_path / "stdout",
            stderr_path=tmp_path / "stderr",
            timeout_seconds=10,
        )
    assert (tmp_path / "stdout").read_text() == "ready\n"


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_adapter_resumes_completed_result_and_rejects_changed_inputs(
    tmp_path, monkeypatch, interrupt_first
):
    repo, output = tmp_path / "upstream", tmp_path / "output"
    (repo / "results_final").mkdir(parents=True)
    cif = tmp_path / "input.cif"
    cif.write_text("a deliberately opaque fixture; parser runs upstream")
    monkeypatch.setattr(
        lemat,
        "_inspect_checkout",
        lambda *args, **kwargs: SimpleNamespace(
            commit="commit", lock_sha256=None, pyproject_sha256="project"
        ),
    )
    monkeypatch.setattr(lemat, "_canonical_reference_inventory", lambda *args: {})
    monkeypatch.setattr(
        lemat, "_verify_reference_pin_shim", lambda **kwargs: ("python", "probe")
    )
    monkeypatch.setattr(
        lemat,
        "_reference_pin_audit_summary",
        lambda *args, **kwargs: ({"verification": "enabled"}, None),
    )

    def probe(**kwargs):
        payload = {"python_version": "test", "cuda_devices": []}
        kwargs["output_path"].write_text(json.dumps(payload))
        return payload, ("python", "probe"), lemat._sha256_file(kwargs["output_path"])

    monkeypatch.setattr(lemat, "_probe_runtime_environment", probe)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if interrupt_first and len(calls) == 1:
            kwargs["stdout_path"].write_text("first allocation progress\n")
            cache = output / "resume_state" / "cache" / "completed-oracle"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("durable oracle progress")
            raise resume.LeMatResumeTimeout("allocation ended")
        result_path = repo / "results_final" / "example_diagnostic_0.json"
        result_path.write_text(
            json.dumps(
                {
                    "run_info": {"n_structures": 1},
                    "validity_filtering": {
                        "total_input_structures": 1,
                        "valid_structures": 1,
                    },
                    "results": {
                        "validity": {
                            "final_scores": {
                                "overall_validity_count": 1,
                                "total_structures": 1,
                            }
                        }
                    },
                }
            )
        )
        return subprocess.CompletedProcess(
            command, 0, stdout=f"Results saved to: {result_path}\n", stderr=""
        )

    monkeypatch.setattr(lemat, "run_streamed", run)
    kwargs = dict(
        repo_path=repo,
        output_dir=output,
        run_name="example",
        config_name="diagnostic",
        families=("validity",),
        runner_command=("python",),
        verify_mace_checkpoint=False,
        runtime_device="cpu",
        resume=True,
    )
    if interrupt_first:
        with pytest.raises(lemat.LeMatGenBenchTimeout):
            lemat.run_lematgenbench(cif, **kwargs)
    result = lemat.run_lematgenbench(cif, **kwargs)
    again = lemat.run_lematgenbench(cif, **kwargs)
    assert len(calls) == 1 + int(interrupt_first)
    assert again.metrics == result.metrics
    assert again.artifact_sha256 == result.artifact_sha256
    assert "resume_runtime" in result.artifact_sha256
    normalizer_source = lemat._material_cif_source()
    with monkeypatch.context() as changed:
        changed.setattr(
            lemat, "_material_cif_source", lambda: normalizer_source + "\n# changed\n"
        )
        with pytest.raises(ValueError, match="contract changed"):
            lemat.run_lematgenbench(cif, **kwargs)
    assert len(calls) == 1 + int(interrupt_first)
    if interrupt_first:
        manifest = json.loads(result.manifest_path.read_text())
        assert len(manifest["resume"]["attempt_history"]) == 1
        archive = Path(manifest["resume"]["attempt_history"][0]["directory"])
        assert (
            archive / "lematgenbench.stdout.log"
        ).read_text() == "first allocation progress\n"
        assert (
            output / "resume_state" / "cache" / "completed-oracle"
        ).read_text() == "durable oracle progress"
    cif.write_text("changed generated material")
    with pytest.raises(ValueError, match="contract changed"):
        lemat.run_lematgenbench(cif, **kwargs)
    assert len(calls) == 1 + int(interrupt_first)
