from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyCrystal:
    """A simple crystal class for testing validity."""

    def __init__(self, valid: bool, comp_valid: bool = True, struct_valid: bool = True):
        self.valid = valid
        self.comp_valid = comp_valid
        self.struct_valid = struct_valid


class FakeStructure:
    """A simple structure class for testing"""

    def __init__(self, label: str, num_sites: int = 1):
        self.label = label
        self._num_sites = num_sites
        self.composition = SimpleNamespace(reduced_formula=label)

    def __len__(self) -> int:
        return self._num_sites


def _install_fake_matminer(
    monkeypatch: pytest.MonkeyPatch,
    call_counter: dict[str, int] | None = None,
) -> None:
    """Install a deterministic matminer stub for diversity testing."""
    matminer = ModuleType("matminer")
    matminer.__path__ = []  # type: ignore[attr-defined]

    featurizers = ModuleType("matminer.featurizers")
    featurizers.__path__ = []  # type: ignore[attr-defined]

    site = ModuleType("matminer.featurizers.site")
    site.__path__ = []  # type: ignore[attr-defined]

    fingerprint = ModuleType("matminer.featurizers.site.fingerprint")

    class FakeCrystalNNFingerprint:
        """A fake CrystalNNFingerprint that returns deterministic vectors based on the structure label."""

        @classmethod
        def from_preset(cls, preset: str):
            """Ignore the preset and return an instance of the fake fingerprint."""
            return cls()

        def featurize(self, structure: FakeStructure, site_index: int):
            """Return a deterministic vector based on the structure label and site index."""
            if call_counter is not None:
                call_counter["featurize"] = call_counter.get("featurize", 0) + 1
            vectors = {
                "A": np.array([1.0, 0.0]),
                "B": np.array([0.0, 1.0]),
            }
            return vectors.get(structure.label, np.array([0.5, 0.5]))

    fingerprint.CrystalNNFingerprint = FakeCrystalNNFingerprint
    site.fingerprint = fingerprint
    featurizers.site = site
    matminer.featurizers = featurizers

    monkeypatch.setitem(sys.modules, "matminer", matminer)
    monkeypatch.setitem(sys.modules, "matminer.featurizers", featurizers)
    monkeypatch.setitem(sys.modules, "matminer.featurizers.site", site)
    monkeypatch.setitem(
        sys.modules, "matminer.featurizers.site.fingerprint", fingerprint
    )


def test_material_metrics_match_expected_definitions(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the validity, novelty, and diversity metrics compute expected values on controlled inputs."""
    from zevals.materials import diversity, novelty, uniqueness, validity

    monkeypatch.setattr(validity, "Crystal", DummyCrystal)

    samples = [
        DummyCrystal(True, True, True),
        DummyCrystal(False, True, False),
        DummyCrystal(False, False, True),
    ]
    assert validity.compute_validity(samples) == {
        "valid_rate": pytest.approx(1 / 3),
        "comp_valid_rate": pytest.approx(2 / 3),
        "struct_valid_rate": pytest.approx(2 / 3),
    }

    with pytest.raises(TypeError, match="Crystal objects"):
        validity.is_valid_structure({"valid": True})

    call_counter: dict[str, int] = {}
    _install_fake_matminer(monkeypatch, call_counter)

    class UnexpectedStructureMatcher:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "StructureMatcher should not be used in fingerprint mode"
            )

    monkeypatch.setattr(novelty, "StructureMatcher", UnexpectedStructureMatcher)

    generated = [FakeStructure("A"), FakeStructure("B"), FakeStructure("B")]
    reference = [FakeStructure("A"), FakeStructure("C")]
    assert novelty.compute_novelty(generated, reference) == {
        "novel_rate": pytest.approx(2 / 3),
    }
    assert call_counter["featurize"] == 5

    assert diversity.compute_diversity(
        [FakeStructure("A"), FakeStructure("A"), FakeStructure("B")]
    ) == {
        "diversity_rate": pytest.approx(2 / 3),
    }

    class FakeStructureMatcher:
        """A fake StructureMatcher that considers structures with the same label as a match."""

        def __init__(self, *args, **kwargs):
            pass

        def group_structures(self, structures):
            """Group structures by their label, returning a list of groups. Each group is a list of structures that have the same label."""
            groups = []
            seen = set()
            for structure in structures:
                if structure.label in seen:
                    continue
                seen.add(structure.label)
                groups.append([structure])
            return groups

    monkeypatch.setattr(uniqueness, "StructureMatcher", FakeStructureMatcher)
    assert uniqueness.compute_uniqueness(
        [FakeStructure("A"), FakeStructure("A"), FakeStructure("B")]
    ) == {
        "num_unique": 2,
        "unique_rate": pytest.approx(2 / 3),
    }
    unique_result = uniqueness.compute_uniqueness(
        [FakeStructure("A"), FakeStructure("A"), FakeStructure("B")],
        return_unique_structures=True,
    )
    assert unique_result["num_unique"] == 2
    assert unique_result["unique_rate"] == pytest.approx(2 / 3)
    assert [structure.label for structure in unique_result["unique_structures"]] == [
        "A",
        "B",
    ]


def test_material_novelty_can_fall_back_to_structure_matcher(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the novelty metric can fall back to using a StructureMatcher when use_fingerprint is False, and that it correctly handles errors raised by the StructureMatcher."""
    from zevals.materials import novelty

    class FakeStructureMatcher:
        """A fake StructureMatcher that considers structures with the same label as a match, and raises an error if either structure is labeled 'bad'."""

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, left: FakeStructure, right: FakeStructure) -> bool:
            """Return True if the labels match, False otherwise. Raise ValueError if either structure is labeled 'bad'."""
            if left.label == "bad" or right.label == "bad":
                raise ValueError("bad structure")
            return left.label == right.label

    monkeypatch.setattr(novelty, "StructureMatcher", FakeStructureMatcher)

    generated = [FakeStructure("A"), FakeStructure("B"), FakeStructure("bad")]
    reference = [FakeStructure("A")]
    assert novelty.compute_novelty(
        generated,
        reference,
        use_fingerprint=False,
    ) == {
        "novel_rate": pytest.approx(0.5),
    }


def test_material_novelty_accepts_precomputed_reference_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the novelty metric can accept precomputed reference fingerprints to avoid redundant computations when evaluating multiple sets of generated structures against the same reference set."""
    from zevals.materials import novelty

    call_counter: dict[str, int] = {}
    _install_fake_matminer(monkeypatch, call_counter)

    generated = [FakeStructure("A"), FakeStructure("B"), FakeStructure("B")]
    reference = [FakeStructure("A"), FakeStructure("C")]

    cached_reference_fingerprints = novelty.compute_structure_fingerprints(reference)
    assert call_counter["featurize"] == 2

    call_counter["featurize"] = 0
    assert novelty.compute_novelty(
        generated,
        reference,
        reference_fingerprints=cached_reference_fingerprints,
    ) == {
        "novel_rate": pytest.approx(2 / 3),
    }
    assert call_counter["featurize"] == 3


def test_material_pipeline_aggregates_metrics_without_lematgenbench(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the MaterialEvalPipeline correctly aggregates metrics when run_lematgenbench is False."""
    from zevals.pipeline import materials as pipeline

    monkeypatch.setattr(
        pipeline,
        "crystal_from_structure",
        lambda structure, idx: SimpleNamespace(
            valid=True, structure=f"crystal-{structure}"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_validity",
        lambda crystals: {
            "valid_rate": 0.5,
            "comp_valid_rate": 0.75,
            "struct_valid_rate": 0.25,
        },
    )
    monkeypatch.setattr(
        pipeline, "compute_diversity", lambda samples: {"diversity_rate": 0.25}
    )
    uniqueness_calls = {}

    def fake_uniqueness(samples, return_unique_structures=False):
        uniqueness_calls["samples"] = list(samples)
        return {
            "num_unique": 1,
            "unique_rate": 0.5,
            "unique_structures": [samples[0]]
            if return_unique_structures and samples
            else [],
        }

    monkeypatch.setattr(
        pipeline,
        "compute_uniqueness",
        fake_uniqueness,
    )
    novelty_calls = {}

    def fake_novelty(samples, refs):
        novelty_calls["samples"] = list(samples)
        return {"novelty_rate": 0.75}

    monkeypatch.setattr(
        pipeline,
        "compute_novelty",
        fake_novelty,
    )

    def _should_not_run(*args, **kwargs):
        """This function should not be called during this test, since run_lematgenbench is False. If it is called, raise an error to fail the test."""
        raise AssertionError("run_lematgenbench should not be called")

    monkeypatch.setattr(pipeline, "run_lematgenbench", _should_not_run)

    results = pipeline.MaterialEvalPipeline(
        reference_structures=[object()],
        run_lematgenbench=False,
    ).run(["s1", "s2"])

    assert results.aggregate == {
        "valid_rate": 0.5,
        "comp_valid_rate": 0.75,
        "struct_valid_rate": 0.25,
        "diversity_rate": 0.25,
        "unique_rate": 0.5,
        "novelty_rate": 0.75,
    }
    assert uniqueness_calls["samples"] == ["crystal-s1", "crystal-s2"]
    assert novelty_calls["samples"] == ["crystal-s1"]
    assert results.per_sample == [
        {"sample_idx": 0, "structure": "s1"},
        {"sample_idx": 1, "structure": "s2"},
    ]
