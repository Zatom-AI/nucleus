from __future__ import annotations

import numpy as np
import pytest


def test_compute_material_generation_metrics_wires_material_metrics(monkeypatch):
    """Test that compute_material_generation_metrics correctly wires up the calls to the underlying material metrics functions, and correctly processes their outputs."""
    from nucleus.metrics import material_generation as mg

    monkeypatch.setattr(mg, "_as_atom_array_list", lambda x: list(x))
    monkeypatch.setattr(
        mg,
        "_atom_array_to_structure",
        lambda atom_array: None if atom_array == "bad" else f"struct-{atom_array}",
    )

    validity_calls = {}
    diversity_calls = {}
    uniqueness_calls = {}
    novelty_calls = {}

    def fake_validity(crystals):
        validity_calls["crystals"] = list(crystals)
        return {"valid_rate": 0.5, "comp_valid_rate": 0.75, "struct_valid_rate": 0.25}

    def fake_diversity(structures):
        diversity_calls["structures"] = list(structures)
        return {"diversity_rate": 0.25}

    def fake_uniqueness(structures, *, return_unique_structures=False):
        uniqueness_calls["structures"] = list(structures)
        result = {"num_unique": 1, "unique_rate": 0.5}
        if return_unique_structures:
            result["unique_structures"] = [structures[0]] if structures else []
        return result

    def fake_novelty(
        structures,
        reference_structures,
        *,
        use_fingerprint,
        reference_fingerprints=None,
    ):
        novelty_calls["structures"] = list(structures)
        novelty_calls["reference_structures"] = list(reference_structures)
        novelty_calls["use_fingerprint"] = use_fingerprint
        novelty_calls["reference_fingerprints"] = reference_fingerprints
        return {"novel_rate": 0.75}

    monkeypatch.setattr(mg, "compute_material_validity", fake_validity)
    monkeypatch.setattr(mg, "compute_material_diversity", fake_diversity)
    monkeypatch.setattr(mg, "compute_material_uniqueness", fake_uniqueness)
    monkeypatch.setattr(mg, "compute_material_novelty", fake_novelty)
    monkeypatch.setattr(
        mg,
        "crystal_from_structure",
        lambda structure: type(
            "FakeCrystal",
            (),
            {
                "valid": True,
                "comp_valid": True,
                "struct_valid": True,
                "structure": f"crystal-{structure}",
            },
        )(),
    )

    result = mg.compute_material_generation_metrics(
        predicted_atom_array_stack=["a", "bad", "b"],
        reference_structures=["ref-1", "ref-2"],
        reference_fingerprints=[np.array([1.0, 0.0])],
    )

    assert result == {
        "valid_rate": pytest.approx(2 / 3),
        "comp_valid_rate": 0.75,
        "struct_valid_rate": 0.25,
        "diversity_rate": 0.25,
        "unique_rate": 0.5,
        "novelty_rate": 0.75,
    }
    assert validity_calls["crystals"][0].structure == "crystal-struct-a"
    assert isinstance(validity_calls["crystals"][1], mg.Crystal)
    assert validity_calls["crystals"][1].valid is False
    assert validity_calls["crystals"][1].comp_valid is False
    assert validity_calls["crystals"][1].struct_valid is False
    assert validity_calls["crystals"][2].structure == "crystal-struct-b"
    assert diversity_calls["structures"] == ["crystal-struct-a", "crystal-struct-b"]
    assert uniqueness_calls["structures"] == ["crystal-struct-a", "crystal-struct-b"]
    assert novelty_calls["structures"] == ["crystal-struct-a"]
    assert novelty_calls["reference_structures"] == ["ref-1", "ref-2"]
    assert novelty_calls["use_fingerprint"] is True
    assert len(novelty_calls["reference_fingerprints"]) == 1
    np.testing.assert_array_equal(
        novelty_calls["reference_fingerprints"][0], np.array([1.0, 0.0])
    )


def test_material_generation_metric_class_uses_foundry_kwargs_mapping(monkeypatch):
    """Test that the MaterialGenerationMetrics class correctly uses the Foundry kwargs mapping to pass the right arguments to compute_material_generation_metrics."""
    from nucleus.metrics import material_generation as mg

    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return {
            "valid_rate": 1.0,
            "comp_valid_rate": 1.0,
            "struct_valid_rate": 1.0,
            "diversity_rate": 1.0,
            "unique_rate": 1.0,
            "novelty_rate": 1.0,
        }

    monkeypatch.setattr(mg, "compute_material_generation_metrics", fake_compute)

    metric = mg.MaterialGenerationMetrics(
        reference_structures=["ref"],
        reference_fingerprints=[np.array([0.5, 0.5])],
        use_ground_truth_as_reference=False,
        novelty_use_fingerprint=False,
    )

    result = metric.compute_from_kwargs(
        predicted_atom_array_stack=["pred"],
        ground_truth_atom_array_stack=["gt"],
        extra_info={"foo": "bar"},
    )

    assert result == {
        "valid_rate": 1.0,
        "comp_valid_rate": 1.0,
        "struct_valid_rate": 1.0,
        "diversity_rate": 1.0,
        "unique_rate": 1.0,
        "novelty_rate": 1.0,
    }
    assert captured["predicted_atom_array_stack"] == ["pred"]
    assert captured["ground_truth_atom_array_stack"] == ["gt"]
    assert captured["extra_info"] == {"foo": "bar"}
    assert captured["reference_structures"] == ["ref"]
    assert len(captured["reference_fingerprints"]) == 1
    np.testing.assert_array_equal(
        captured["reference_fingerprints"][0], np.array([0.5, 0.5])
    )
    assert captured["use_ground_truth_as_reference"] is False
    assert captured["novelty_use_fingerprint"] is False


def test_material_generation_metric_loads_reference_fingerprints_from_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the metric loads a precomputed fingerprint cache from disk."""
    from nucleus.metrics import material_generation as mg

    monkeypatch.setattr(mg, "_as_atom_array_list", lambda x: list(x))
    monkeypatch.setattr(
        mg,
        "_atom_array_to_structure",
        lambda atom_array: f"struct-{atom_array}",
    )
    monkeypatch.setattr(
        mg,
        "crystal_from_structure",
        lambda structure: type(
            "FakeCrystal",
            (),
            {
                "valid": True,
                "comp_valid": True,
                "struct_valid": True,
                "structure": structure,
            },
        )(),
    )
    monkeypatch.setattr(
        mg,
        "compute_material_validity",
        lambda crystals: {
            "valid_rate": 1.0,
            "comp_valid_rate": 1.0,
            "struct_valid_rate": 1.0,
        },
    )
    monkeypatch.setattr(
        mg,
        "compute_material_diversity",
        lambda structures: {"diversity_rate": 1.0},
    )
    monkeypatch.setattr(
        mg,
        "compute_material_uniqueness",
        lambda structures, return_unique_structures=False: {
            "num_unique": 1,
            "unique_rate": 1.0,
            "unique_structures": [structures[0]]
            if return_unique_structures and structures
            else [],
        },
    )

    fingerprint_cache = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    cache_path = tmp_path / "reference_fingerprints.npy"
    np.save(cache_path, fingerprint_cache)

    novelty_calls = {}

    def fake_novelty(
        structures,
        reference_structures,
        *,
        use_fingerprint,
        reference_fingerprints=None,
    ):
        novelty_calls["structures"] = list(structures)
        novelty_calls["reference_structures"] = list(reference_structures)
        novelty_calls["use_fingerprint"] = use_fingerprint
        novelty_calls["reference_fingerprints"] = reference_fingerprints
        return {"novel_rate": 0.5}

    monkeypatch.setattr(mg, "compute_material_novelty", fake_novelty)

    metric = mg.MaterialGenerationMetrics(
        reference_fingerprints_path=cache_path,
        use_ground_truth_as_reference=False,
    )

    result = metric.compute_from_kwargs(
        predicted_atom_array_stack=["a"],
        ground_truth_atom_array_stack=["gt"],
        extra_info={"example_id": "example-1"},
    )

    assert result == {
        "valid_rate": 1.0,
        "comp_valid_rate": 1.0,
        "struct_valid_rate": 1.0,
        "diversity_rate": 1.0,
        "unique_rate": 1.0,
        "novelty_rate": 0.5,
    }
    assert novelty_calls["structures"] == ["struct-a"]
    assert novelty_calls["reference_structures"] == []
    assert novelty_calls["use_fingerprint"] is True
    np.testing.assert_array_equal(
        novelty_calls["reference_fingerprints"], fingerprint_cache
    )
