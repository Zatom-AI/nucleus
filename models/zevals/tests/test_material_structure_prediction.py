"""Tests for material structure-prediction Match Rate."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_material_match_rate_uses_any_candidate_per_target(
    monkeypatch: pytest.MonkeyPatch,
):
    """MR is target-wise: one matching candidate is sufficient."""
    from zevals.materials import structure_prediction

    created_with = {}

    class FakeMatcher:
        def __init__(self, **kwargs):
            created_with.update(kwargs)

        def get_rms_dist(self, candidate, reference):
            return (0.25, 0.5) if candidate == reference else None

    monkeypatch.setattr(structure_prediction, "StructureMatcher", FakeMatcher)
    result = structure_prediction.compute_material_match_rate(
        {
            "matched": ["wrong", "reference-a"],
            "unmatched": ["wrong"],
        },
        {
            "matched": ["reference-a"],
            "unmatched": ["reference-b"],
        },
    )

    assert result == {"material_match_rate": 0.5}
    assert created_with == {
        "ltol": 0.3,
        "stol": 0.5,
        "angle_tol": 10.0,
        "scale": True,
    }


def test_material_match_rate_rejects_mismatched_target_ids():
    """Candidates must never be compared to a different target's reference."""
    from zevals.materials.structure_prediction import compute_material_match_rate

    with pytest.raises(ValueError, match="target IDs must match"):
        compute_material_match_rate(
            {"generated": [object()]},
            {"reference": [object()]},
        )


def test_material_metrics_report_best_matched_rms_and_true_target_aggregates(
    monkeypatch: pytest.MonkeyPatch,
):
    """Matched RMS is minimized per target, then mean/median aggregated."""
    from zevals.materials import structure_prediction

    distances = {
        ("a-bad", "a-ref"): None,
        ("a-good", "a-ref"): (0.3, 0.6),
        ("a-best", "a-ref"): (0.1, 0.2),
        ("b-good", "b-ref"): (0.4, 0.8),
        ("c-bad", "c-ref"): None,
    }
    created_with = {}

    class FakeMatcher:
        def __init__(self, **kwargs):
            created_with.update(kwargs)

        def get_rms_dist(self, candidate, reference):
            return distances[(candidate, reference)]

    monkeypatch.setattr(structure_prediction, "StructureMatcher", FakeMatcher)
    result = structure_prediction.compute_material_structure_prediction_metrics(
        {
            "a": ["a-bad", "a-good", "a-best"],
            "b": ["b-good"],
            "c": ["c-bad"],
        },
        {"a": ["a-ref"], "b": ["b-ref"], "c": ["c-ref"]},
        scale=True,
    )

    assert result == pytest.approx(
        {
            "material_match_rate": 2 / 3,
            "material_normalized_rms_dist_mean": 0.25,
            "material_normalized_rms_dist_median": 0.25,
            # Only "b" matches on its first candidate; "a" needs its second.
            "material_match_rate_top1": 1 / 3,
            "material_normalized_rms_dist_mean_top1": 0.4,
            "material_normalized_rms_dist_median_top1": 0.4,
            "material_matched_target_count_top1": 1.0,
            "material_matched_target_count": 2.0,
            "material_unmatched_target_count": 1.0,
            "material_candidate_count": 5.0,
            "material_candidate_count_min": 1.0,
            "material_invalid_candidate_count": 0.0,
            "material_matcher_error_count": 0.0,
        }
    )
    assert created_with["scale"] is True


def test_material_metrics_match_periodic_translation_and_atom_permutation():
    """Real StructureMatcher behavior is invariant to periodic images/order."""
    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics,
    )

    lattice = Lattice.from_parameters(4.2, 5.1, 6.4, 76.0, 82.0, 69.0)
    reference = Structure(
        lattice,
        ["Na", "Cl", "Na"],
        [[0.1, 0.2, 0.3], [0.45, 0.55, 0.65], [0.8, 0.15, 0.9]],
    )
    candidate = Structure(
        lattice,
        ["Na", "Na", "Cl"],
        [[1.8, 0.15, 0.9], [1.1, 0.2, 0.3], [1.45, 0.55, 0.65]],
    )

    result = compute_material_structure_prediction_metrics(
        {"target": [candidate]},
        {"target": [reference]},
        scale=False,
    )

    assert result["material_match_rate"] == 1.0
    assert result["material_normalized_rms_dist_mean"] == pytest.approx(0.0, abs=1e-8)


def test_material_metrics_expose_lattice_scaling_policy():
    """Scale=True permits uniform cell rescaling; Scale=False can reject it."""
    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics,
    )

    reference = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    candidate = Structure(
        Lattice.cubic(4.4),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    scaled = compute_material_structure_prediction_metrics(
        {"target": [candidate]},
        {"target": [reference]},
        ltol=0.01,
        scale=True,
    )
    fixed = compute_material_structure_prediction_metrics(
        {"target": [candidate]},
        {"target": [reference]},
        ltol=0.01,
        scale=False,
    )

    assert scaled["material_match_rate"] == 1.0
    assert fixed["material_match_rate"] == 0.0


def test_material_metrics_report_invalid_candidates_and_matcher_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    """Conversion failures and matcher exceptions must remain visible."""
    from zevals.materials import structure_prediction

    class FakeMatcher:
        def __init__(self, **kwargs):
            pass

        def get_rms_dist(self, candidate, reference):
            if candidate == "error":
                raise RuntimeError("bad candidate")
            return (0.2, 0.3) if candidate == reference else None

    monkeypatch.setattr(structure_prediction, "StructureMatcher", FakeMatcher)
    result = structure_prediction.compute_material_structure_prediction_metrics(
        {"target": [None, "error", "reference"]},
        {"target": ["reference"]},
    )

    assert result["material_match_rate"] == 1.0
    assert result["material_candidate_count"] == 3.0
    assert result["material_invalid_candidate_count"] == 2.0
    assert result["material_matcher_error_count"] == 1.0


def test_material_match_rate_at_fixed_budget_uses_the_first_k_candidates(
    monkeypatch: pytest.MonkeyPatch,
):
    """MR@k truncates in generation order rather than taking the oracle best."""
    from zevals.materials import structure_prediction

    distances = {
        ("a-bad", "a-ref"): None,
        ("a-good", "a-ref"): (0.3, 0.6),
        ("a-best", "a-ref"): (0.1, 0.2),
        ("b-good", "b-ref"): (0.4, 0.8),
        ("c-bad", "c-ref"): None,
    }

    class FakeMatcher:
        def __init__(self, **kwargs):
            pass

        def get_rms_dist(self, candidate, reference):
            return distances[(candidate, reference)]

    monkeypatch.setattr(structure_prediction, "StructureMatcher", FakeMatcher)
    result = structure_prediction.compute_material_structure_prediction_metrics(
        {
            "a": ["a-bad", "a-good", "a-best"],
            "b": ["b-good", "b-good"],
            "c": ["c-bad", "c-bad"],
        },
        {"a": ["a-ref"], "b": ["b-ref"], "c": ["c-ref"]},
        top_k=(1, 2),
    )

    assert result["material_match_rate_top1"] == pytest.approx(1 / 3)
    assert result["material_normalized_rms_dist_mean_top1"] == pytest.approx(0.4)
    assert result["material_match_rate_top2"] == pytest.approx(2 / 3)
    assert result["material_normalized_rms_dist_mean_top2"] == pytest.approx(0.35)
    # "a"'s best candidate is third, so it counts only at the full budget.
    assert result["material_match_rate"] == pytest.approx(2 / 3)
    assert result["material_normalized_rms_dist_mean"] == pytest.approx(0.25)
    assert result["material_candidate_count_min"] == 2.0


def test_material_match_rate_at_1_reports_nan_rms_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unmatched budget reports a zero rate with NaN distances, not a crash."""
    import math

    from zevals.materials import structure_prediction

    class FakeMatcher:
        def __init__(self, **kwargs):
            pass

        def get_rms_dist(self, candidate, reference):
            return None

    monkeypatch.setattr(structure_prediction, "StructureMatcher", FakeMatcher)
    result = structure_prediction.compute_material_structure_prediction_metrics(
        {"a": ["candidate"]},
        {"a": ["reference"]},
        top_k=(1,),
    )

    assert result["material_match_rate_top1"] == 0.0
    assert math.isnan(result["material_normalized_rms_dist_mean_top1"])


def test_material_top_k_rejects_non_positive_budgets():
    """A zero or negative budget is a config error, not a silent no-op."""
    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics,
    )

    with pytest.raises(ValueError, match="top_k budgets must be positive"):
        compute_material_structure_prediction_metrics(
            {"a": [object()]},
            {"a": [object()]},
            top_k=(0,),
        )

    with pytest.raises(ValueError, match="top-2 requires at least 2"):
        compute_material_structure_prediction_metrics(
            {"a": [object()]},
            {"a": [object()]},
            top_k=(2,),
        )


def test_material_metrics_require_one_reference_per_target():
    """The online CSP contract is one ground-truth structure per target."""
    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics,
    )

    with pytest.raises(ValueError, match="exactly one valid reference"):
        compute_material_structure_prediction_metrics(
            {"composition": [object()]},
            {"composition": [object(), object()]},
        )


def test_zevals_loads_atomworks_style_cif_from_fractional_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Zatom CIFs remain readable even without pymatgen's expected label field."""
    from pymatgen.core import Structure as PymatgenStructure

    from zevals.utils.io import load_structure_cif

    path = tmp_path / "zatom.cif"
    path.write_text(
        """data_zatom
_cell.length_a 4.0
_cell.length_b 5.0
_cell.length_c 6.0
_cell.angle_alpha 75.0
_cell.angle_beta 80.0
_cell.angle_gamma 70.0
loop_
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.fract_x
_atom_site.fract_y
_atom_site.fract_z
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
Na Na1 0.1 0.2 0.3 0.0 0.0 0.0
Cl Cl1 0.6 0.7 0.8 0.0 0.0 0.0
""",
        encoding="ascii",
    )
    monkeypatch.setattr(
        PymatgenStructure,
        "from_file",
        classmethod(lambda cls, filename: (_ for _ in ()).throw(ValueError())),
    )

    structure = load_structure_cif(path)

    assert structure.composition.reduced_formula == "NaCl"
    assert structure.lattice.angles == pytest.approx((75.0, 80.0, 70.0))
    np.testing.assert_allclose(
        structure.frac_coords,
        [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
    )
    assert not (structure.cart_coords == 0.0).all()


def test_load_materials_groups_cif_candidates_by_target(tmp_path: Path):
    """Passing references activates explicit target-based CIF loading."""
    from pymatgen.core import Lattice, Structure

    from zevals.pipeline.loaders import MaterialStructureCandidates, load_materials

    def write_cif(path: Path) -> None:
        Structure(
            Lattice.cubic(4.0),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ).to(filename=path)

    generated_dir = tmp_path / "generated" / "target-a"
    reference_dir = tmp_path / "reference" / "target-a"
    generated_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    write_cif(generated_dir / "candidate.cif")
    write_cif(reference_dir / "reference.cif")

    samples = load_materials(generated_dir.parent, reference_dir.parent)

    assert isinstance(samples, MaterialStructureCandidates)
    assert list(samples.generated_by_target) == ["target-a"]
    assert list(samples.reference_by_target) == ["target-a"]
    assert len(samples.structures) == 1


def test_material_pipeline_adds_match_rate_for_grouped_candidates(
    monkeypatch: pytest.MonkeyPatch,
):
    """The material pipeline preserves existing metrics and adds MR on demand."""
    from zevals.pipeline import materials as pipeline
    from zevals.pipeline.loaders import MaterialStructureCandidates

    monkeypatch.setattr(
        pipeline,
        "crystal_from_structure",
        lambda structure, _idx: SimpleNamespace(valid=True, structure=structure),
    )
    monkeypatch.setattr(
        pipeline, "compute_validity", lambda _crystals: {"valid_rate": 1.0}
    )
    monkeypatch.setattr(
        pipeline, "compute_diversity", lambda _structures: {"diversity_rate": 1.0}
    )
    monkeypatch.setattr(
        pipeline,
        "compute_uniqueness",
        lambda structures, return_unique_structures: {
            "unique_rate": 1.0,
            "unique_structures": structures if return_unique_structures else [],
        },
    )
    match_kwargs = {}

    def fake_structure_prediction_metrics(generated, reference, **kwargs):
        match_kwargs.update(kwargs)
        assert generated == {"target-a": ["candidate"]}
        assert reference == {"target-a": ["reference"]}
        return {
            "material_match_rate": 1.0,
            "material_normalized_rms_dist_mean": 0.1,
            "material_normalized_rms_dist_median": 0.1,
        }

    monkeypatch.setattr(
        pipeline,
        "compute_material_structure_prediction_metrics",
        fake_structure_prediction_metrics,
    )
    result = pipeline.MaterialEvalPipeline(
        compute_structure_prediction_metrics=True
    ).run(
        MaterialStructureCandidates(
            structures=["candidate"],
            generated_by_target={"target-a": ["candidate"]},
            reference_by_target={"target-a": ["reference"]},
        )
    )

    assert result.aggregate["material_match_rate"] == 1.0
    assert result.aggregate["material_normalized_rms_dist_mean"] == 0.1
    assert match_kwargs == {
        "ltol": 0.3,
        "stol": 0.5,
        "angle_tol": 10.0,
        "scale": True,
        "top_k": (1,),
    }
