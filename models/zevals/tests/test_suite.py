"""Unit + integration tests for `pipeline/suite.py::ZevalSuite`.

Two tiers, matching the module's own design:
  - Fast/mocked: `ZevalSuite._get_pipeline` is monkeypatched to return a fake
    pipeline, isolating `evaluate()`'s dispatch/merge/error-isolation/caching
    logic from real Hydra composition and oracle construction.
  - Real: a couple of tests exercise the actual `_resolve_config_dir()` +
    `hydra.compose()` + `instantiate()` path against the `molecules` config,
    which (unlike proteins/rna/ligands) has zero unimplemented-oracle
    blockers, so it can complete a full run cheaply and for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.pipeline.base import EvalResults  # noqa: E402
from zevals.pipeline.suite import ZevalSuite, _resolve_config_dir  # noqa: E402


class FakePipeline:
    """A minimal EvalPipeline stand-in for mocked ZevalSuite tests."""

    def __init__(self, aggregate: dict[str, float], raise_on_run: bool = False):
        self._aggregate = aggregate
        self._raise_on_run = raise_on_run
        self.run_calls: list[object] = []

    def run(self, samples):
        self.run_calls.append(samples)
        if self._raise_on_run:
            raise RuntimeError("simulated pipeline failure")
        return EvalResults(aggregate=dict(self._aggregate))


# --------------------------------------------------------------------------
# Real (unmocked) config resolution + end-to-end molecules run
# --------------------------------------------------------------------------


def test_resolve_config_dir_finds_real_configs_directory():
    config_dir = _resolve_config_dir()
    assert (config_dir / "proteins.yaml").is_file()
    assert (config_dir / "molecules.yaml").is_file()


def test_resolve_config_dir_ignores_empty_placeholder_directory():
    """Regression test for the bug found this session: a source checkout
    also has an empty src/zevals/configs/.gitkeep placeholder at the
    same relative offset as the installed-wheel layout — the resolver must
    not be fooled by its mere existence as a directory."""
    placeholder = ROOT / "zevals" / "configs"
    assert placeholder.is_dir()
    assert not (placeholder / "proteins.yaml").is_file()

    config_dir = _resolve_config_dir()
    assert config_dir != placeholder


def test_evaluate_real_molecules_config_end_to_end(tmp_path: Path):
    """molecules.yaml has no unimplemented-oracle blockers, so this exercises
    the real compose()/instantiate() path (not just the mocked-pipeline
    tests below) without needing any downloaded checkpoints."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "gen.smi").write_text("CCO\nc1ccccc1\n")

    suite = ZevalSuite()
    metrics = suite.evaluate({"molecules": samples_dir}, print_results=False)

    assert "molecules/valid_rate" in metrics
    assert metrics["molecules/valid_rate"] == pytest.approx(1.0)


def test_evaluate_reuses_cached_pipeline_across_real_calls(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "gen.smi").write_text("CCO\n")

    suite = ZevalSuite()
    suite.evaluate({"molecules": samples_dir}, print_results=False)
    first_pipeline = suite._pipelines["molecules"]

    suite.evaluate({"molecules": samples_dir}, print_results=False)
    second_pipeline = suite._pipelines["molecules"]

    assert first_pipeline is second_pipeline


# --------------------------------------------------------------------------
# Mocked-pipeline tests: dispatch / merge / error isolation / caching logic
# --------------------------------------------------------------------------


def test_evaluate_namespaces_keys_by_modality(monkeypatch: pytest.MonkeyPatch):
    suite = ZevalSuite()
    fake = FakePipeline({"designability": 0.8, "novelty": 0.3})
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fake)
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {"proteins": lambda samples_dir, gt: ["a.pdb", "b.pdb"]},
    )

    metrics = suite.evaluate({"proteins": "unused"}, print_results=False)

    assert metrics == {"proteins/designability": 0.8, "proteins/novelty": 0.3}
    assert fake.run_calls == [["a.pdb", "b.pdb"]]


def test_evaluate_merges_multiple_modalities(monkeypatch: pytest.MonkeyPatch):
    suite = ZevalSuite()
    fakes = {
        "proteins": FakePipeline({"designability": 0.8}),
        "molecules": FakePipeline({"valid_rate": 0.9}),
    }
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fakes[modality])
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {
            "proteins": lambda samples_dir, gt: ["a.pdb"],
            "molecules": lambda samples_dir, gt: ["CCO"],
        },
    )

    metrics = suite.evaluate(
        {"proteins": "dir1", "molecules": "dir2"}, print_results=False
    )

    assert metrics == {"proteins/designability": 0.8, "molecules/valid_rate": 0.9}


def test_evaluate_isolates_one_modality_failure_from_another(
    monkeypatch: pytest.MonkeyPatch,
):
    suite = ZevalSuite()
    fakes = {
        "proteins": FakePipeline({}, raise_on_run=True),
        "molecules": FakePipeline({"valid_rate": 0.9}),
    }
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fakes[modality])
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {
            "proteins": lambda samples_dir, gt: ["a.pdb"],
            "molecules": lambda samples_dir, gt: ["CCO"],
        },
    )

    # Must not raise, despite the proteins pipeline crashing.
    metrics = suite.evaluate(
        {"proteins": "dir1", "molecules": "dir2"}, print_results=False
    )

    assert metrics == {"molecules/valid_rate": 0.9}
    assert suite.last_metadata["proteins"]["pipeline_status"]["status"] == "failed"
    assert (
        "simulated pipeline failure"
        in suite.last_metadata["proteins"]["pipeline_status"]["error"]
    )


def test_evaluate_isolates_unknown_modality(monkeypatch: pytest.MonkeyPatch):
    suite = ZevalSuite()
    fake = FakePipeline({"valid_rate": 0.9})
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fake)
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {"molecules": lambda samples_dir, gt: ["CCO"]},
    )

    metrics = suite.evaluate(
        {"not_a_real_modality": "dir1", "molecules": "dir2"}, print_results=False
    )

    assert metrics == {"molecules/valid_rate": 0.9}


def test_evaluate_caches_pipeline_across_calls_with_mocked_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    suite = ZevalSuite()
    build_calls = []

    def fake_get_pipeline(modality):
        build_calls.append(modality)
        return FakePipeline({"x": 1.0})

    monkeypatch.setattr(suite, "_get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {"proteins": lambda samples_dir, gt: []},
    )

    suite.evaluate({"proteins": "dir1"}, print_results=False)
    suite.evaluate({"proteins": "dir1"}, print_results=False)

    # _get_pipeline is called every time (it does its own internal caching
    # via self._pipelines) — the point of this test is that ZevalSuite
    # doesn't add a *second*, redundant caching layer on top.
    assert build_calls == ["proteins", "proteins"]


def test_get_pipeline_only_builds_once_per_modality(monkeypatch: pytest.MonkeyPatch):
    """The real (non-monkeypatched) `_get_pipeline` caches on `self._pipelines`."""
    suite = ZevalSuite()
    instantiate_calls = []

    class FakeCfg:
        pipeline = object()

    monkeypatch.setattr(
        "zevals.pipeline.suite.compose", lambda config_name, overrides: FakeCfg()
    )

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "zevals.pipeline.suite.initialize_config_dir",
        lambda config_dir, version_base: _NullContext(),
    )

    def fake_instantiate(pipeline_cfg):
        instantiate_calls.append(pipeline_cfg)
        return FakePipeline({})

    monkeypatch.setattr("zevals.pipeline.suite.instantiate", fake_instantiate)

    p1 = suite._get_pipeline("proteins")
    p2 = suite._get_pipeline("proteins")

    assert p1 is p2
    assert len(instantiate_calls) == 1


def test_modality_overrides_forwarded_to_compose(monkeypatch: pytest.MonkeyPatch):
    suite = ZevalSuite(
        modality_overrides={"proteins": ["oracle/inverse_folder=soluble_mpnn"]}
    )
    captured = {}

    def fake_compose(config_name, overrides):
        captured["config_name"] = config_name
        captured["overrides"] = overrides
        return type("Cfg", (), {"pipeline": object()})()

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("zevals.pipeline.suite.compose", fake_compose)
    monkeypatch.setattr(
        "zevals.pipeline.suite.initialize_config_dir",
        lambda config_dir, version_base: _NullContext(),
    )
    monkeypatch.setattr(
        "zevals.pipeline.suite.instantiate", lambda pipeline_cfg: FakePipeline({})
    )

    suite._get_pipeline("proteins")

    assert captured == {
        "config_name": "proteins",
        "overrides": ["oracle/inverse_folder=soluble_mpnn"],
    }


def test_evaluate_writes_results_json_when_results_dir_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    suite = ZevalSuite(results_dir=tmp_path)
    fake = FakePipeline({"designability": 0.8})
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fake)
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {"proteins": lambda samples_dir, gt: []},
    )

    suite.evaluate({"proteins": "dir1"}, print_results=False)

    results_path = tmp_path / "results.json"
    assert results_path.is_file()
    assert json.loads(results_path.read_text()) == {"proteins/designability": 0.8}
    status_path = tmp_path / "metric_status.json"
    assert status_path.is_file()
    assert json.loads(status_path.read_text())["proteins"]["pipeline_status"] == {
        "status": "complete"
    }


def test_evaluate_does_not_write_json_when_results_dir_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    suite = ZevalSuite()  # results_dir defaults to None
    fake = FakePipeline({"designability": 0.8})
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fake)
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {"proteins": lambda samples_dir, gt: []},
    )

    suite.evaluate({"proteins": "dir1"}, print_results=False)

    assert not (tmp_path / "results.json").exists()


def test_evaluate_print_results_flag_controls_report_printing(
    monkeypatch: pytest.MonkeyPatch,
):
    suite = ZevalSuite()
    fake = FakePipeline({"designability": 0.8})
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fake)
    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY",
        {"proteins": lambda samples_dir, gt: []},
    )

    print_calls = []
    monkeypatch.setattr(
        "zevals.pipeline.suite.print_merged_report",
        lambda merged, title=None: print_calls.append(merged),
    )

    suite.evaluate({"proteins": "dir1"}, print_results=False)
    assert print_calls == []

    suite.evaluate({"proteins": "dir1"}, print_results=True)
    assert print_calls == [{"proteins/designability": 0.8}]


def test_ground_truth_dirs_forwarded_to_loader(monkeypatch: pytest.MonkeyPatch):
    suite = ZevalSuite()
    fake = FakePipeline({"x": 1.0})
    monkeypatch.setattr(suite, "_get_pipeline", lambda modality: fake)

    captured = {}

    def fake_loader(samples_dir, gt_dir):
        captured["samples_dir"] = samples_dir
        captured["gt_dir"] = gt_dir
        return []

    monkeypatch.setattr(
        "zevals.pipeline.suite.LOADER_REGISTRY", {"ligands": fake_loader}
    )

    suite.evaluate(
        {"ligands": "samples/dir"},
        ground_truth_dirs={"ligands": "gt/dir"},
        print_results=False,
    )

    assert captured["samples_dir"] == Path("samples/dir")
    assert captured["gt_dir"] == Path("gt/dir")
