"""Guard against mixing incompatible or imputed scaling observations."""

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def figure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("plot_zatom2_figure2_epochs")


def test_dataset_epochs_use_completed_epochs_and_condition_support(figure):
    frame = pd.DataFrame(
        {
            "condition": ["base_500k", "large_500k", "base_full", "large_full"],
            "epoch": [15, 39, 15, 79],
            "value": [0.02, 0.017, 0.021, 0.015],
        }
    )
    original = frame.copy(deep=True)
    assert figure.dataset_epochs(frame).tolist() == [80, 200, 16, 80]
    pd.testing.assert_frame_equal(frame, original)


def test_only_displayed_small_data_generation_structure_endpoints_align(figure):
    frame = pd.DataFrame(
        [
            ("base_full", "omol_afd", 76, True),
            ("base_500k", "omol_afd", 80, True),
            ("large_full", "omat_match", 76, True),
            ("large_500k", "omat_match", 80, True),
            ("base_500k", "omol_force_mae", 80, True),
            ("large_500k", "omat_force_mae", 80, True),
            ("base_500k", "omol_afd", 200, False),
        ],
        columns=["condition", "metric", "dataset_epochs", "displayed"],
    )
    original = frame.copy(deep=True)
    assert figure.plot_epochs(frame).tolist() == [76, 76, 76, 76, 80, 80, 200]
    pd.testing.assert_frame_equal(frame, original)


def test_backfill_preserves_mid_epoch_identity_and_rejects_incomplete_scores(
    figure, tmp_path
):
    job = dict(
        condition="base_500k",
        label="step3000",
        epoch=2,
        global_step=3000,
        checkpoint_kind="step",
        dataset_epochs=12.3,
        epoch_coordinate_kind="approximate rank-local progress",
    )
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": [job]}))
    path = tmp_path / "base_500k/step3000/scores.json"
    path.parent.mkdir(parents=True)
    result = dict(
        **job,
        values={"omol_afd": 0.1, "omat_match": 2.0},
        counts={"omol_afd": 1000, "omat_match": 500},
        afd={"selected_reference_features_sha256": "same"},
    )
    path.write_text(json.dumps(result))
    frame = pd.DataFrame(
        [
            dict(
                condition="base_full",
                epoch=75,
                metric="omol_afd",
                value=0.02,
                dataset_epochs=76,
            )
        ]
    )
    provenance = {"inputs": {}, "selected_reference_features_sha256": "same"}
    combined = figure.add_backfill(frame, tmp_path, provenance)
    added = combined[combined.condition.eq("base_500k")]
    assert added.epoch.eq(2).all()
    assert added.checkpoint_global_step.eq(3000).all()
    assert added.plot_epoch.eq(12.3).all()
    assert added.configured_training_draws.isna().all()
    assert len(combined) == 3
    result["counts"]["omol_afd"] = 999
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Incomplete"):
        figure.add_backfill(frame, tmp_path, provenance)


def test_selection_retains_regressions_and_excludes_incomplete_or_imputed(figure):
    history = pd.DataFrame(
        [
            (15, 0.020, 1000, "recomputed_saved_complete"),
            (20, 0.019, 1000, "recomputed_saved_complete"),
            (25, 0.021, 1000, "recomputed_saved_complete"),
            (30, 0.018, 999, "recomputed_saved_partial"),
            (35, 0.017, 1000, "interpolated"),
            (40, 0.016, 999, "recomputed_saved_complete"),
        ],
        columns=["epoch", "value", "n_samples", "status"],
    ).assign(condition="base_full", metric="omol_afd", afd_model_name=figure.ENCODER)
    selected = figure.select_history(history)
    assert selected.epoch.tolist() == [20, 25]
    assert selected.value.tolist() == [0.019, 0.021]
    with pytest.raises(ValueError, match="Duplicate"):
        figure.select_history(pd.concat([selected, selected]))
    with pytest.raises(ValueError, match="encoder"):
        figure.select_history(history.assign(afd_model_name="Ty-Perez/ct-scd-geom10"))


def test_early_small_data_history_is_retained_without_duplicate_epoch15(figure):
    history = pd.DataFrame(
        [("base_500k", 1), ("large_500k", 14), ("base_500k", 15), ("base_full", 1)],
        columns=["condition", "epoch"],
    ).assign(
        metric="omol_force_mae",
        value=0.5,
        n_samples=500,
        status="measured_wandb",
        afd_model_name=None,
    )
    selected = figure.select_history(history)
    assert selected.epoch.tolist() == [1, 14]
    assert figure.dataset_epochs(selected).tolist() == [10, 75]


@pytest.mark.parametrize(
    "changed",
    [
        {"model_name": "Ty-Perez/ct-scd-geom10"},
        {"reference_sampling_seed": 42},
        {"selected_reference_features_sha256": "different"},
    ],
)
def test_afd_protocol_changes_fail(figure, changed):
    scoring = {
        "model_name": figure.ENCODER,
        "reference_sampling_seed": 0,
        "selected_reference_features_sha256": "same",
    }
    figure.check_afd(scoring, "same")
    with pytest.raises(ValueError, match="protocol"):
        figure.check_afd(scoring | changed, "same")
