"""Check provenance and interpolation rules without loading scientific models."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_zatom2_figure2.py"
    spec = importlib.util.spec_from_file_location("zatom2_figure2_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measurement(epoch, value, status="measured_wandb", source=None):
    return {
        "condition": "base_full",
        "epoch": epoch,
        "metric": "omol_rmsd",
        "value": value,
        "status": status,
        "source_file": source or f"epoch_{epoch}.csv",
        "n_samples": 493
        if isinstance(status, str) and status.endswith("partial")
        else 500,
        "notes": "test measurement",
    }


def write_history(root, rows):
    path = root / "wandb/base_full/history.jsonl.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as handle:
        for row in reversed(rows):
            handle.write(json.dumps(row) + "\n")


def test_sparse_history_aligns_training_and_validation_after_restart(
    analysis, tmp_path, monkeypatch
):
    monkeypatch.setattr(analysis, "CONDITIONS", {"base_full": None})
    metric_key = analysis.METRICS["omol_rmsd"][2]
    train_key = analysis.TRAIN_METRICS["total_loss"]
    count_key = "val/omol25_structure_prediction/zevals_epoch.num_targets"
    rows = [
        {"_step": 1, "trainer/global_step": 100, train_key: 4.0},
        {"_step": 2, "trainer/global_step": 100, "trainer/epoch": 4},
        {"_step": 3, "trainer/global_step": 100, metric_key: 1.8},
        {"_step": 4, "trainer/global_step": 100, count_key: 500},
        # A forward fill would incorrectly associate this loss with epoch 4.
        {"_step": 5, "trainer/global_step": 200, train_key: 3.0},
        {"_step": 6, "trainer/global_step": 200, "trainer/epoch": 5},
        {"_step": 7, "trainer/global_step": 200, metric_key: 1.7},
        # A restart logs the completed epoch and validation again.
        {"_step": 8, "trainer/global_step": 200, "trainer/epoch": 5},
        {"_step": 9, "trainer/global_step": 200, metric_key: 1.6},
        {"_step": 10, "trainer/global_step": 200, count_key: 500},
    ]
    write_history(tmp_path, rows)

    validation, training, coverage = analysis.extract_history(tmp_path)

    assert list(training.epoch) == [4, 5]
    assert list(training.value) == [4.0, 3.0]
    assert list(validation.epoch) == [4, 5, 5]
    assert list(validation.n_samples) == [500, 500, 500]
    combined = analysis.combine_measurements(validation, pd.DataFrame())
    assert list(combined.epoch) == [4, 5]
    assert list(combined.value) == [1.8, 1.6]
    assert coverage[0]["last_logged_epoch"] == 5
    assert coverage[0]["history_rows"] == len(rows)


def test_validation_epoch_and_optimizer_step_mismatch_raises(
    analysis, tmp_path, monkeypatch
):
    monkeypatch.setattr(analysis, "CONDITIONS", {"base_full": None})
    metric_key = analysis.METRICS["omol_rmsd"][2]
    write_history(
        tmp_path,
        [
            {"_step": 1, "trainer/global_step": 100, "trainer/epoch": 4},
            {"_step": 2, "trainer/global_step": 200, "trainer/epoch": 5},
            {"_step": 3, "trainer/global_step": 100, metric_key: 1.8},
        ],
    )

    with pytest.raises(ValueError, match="validation epoch/step mismatch"):
        analysis.extract_history(tmp_path)


@pytest.mark.parametrize(
    ("metric", "count_key", "sample_count", "status"),
    [
        (
            "omol_rmsd",
            "val/omol25_structure_prediction/zevals_epoch.num_targets",
            493,
            "measured_wandb_partial",
        ),
        (
            "omol_afd",
            "val/omol25_generation/afd_num_samples",
            985,
            "measured_wandb_partial",
        ),
        (
            "omat_force_mae",
            "val/omat24_mlip/atomistic_property_metrics.num_energy_systems",
            492,
            "measured_wandb_partial",
        ),
        (
            "omat_force_mae",
            "val/omat24_mlip/atomistic_property_metrics.num_energy_systems",
            500,
            "measured_wandb",
        ),
    ],
)
def test_remote_sample_counts_are_recovered_from_separate_history_rows(
    analysis, tmp_path, monkeypatch, metric, count_key, sample_count, status
):
    monkeypatch.setattr(analysis, "CONDITIONS", {"base_full": None})
    metric_key = analysis.METRICS[metric][2]
    write_history(
        tmp_path,
        [
            {"_step": 1, "trainer/global_step": 100, "trainer/epoch": 19},
            {"_step": 2, "trainer/global_step": 100, metric_key: 1.8},
            {"_step": 3, "trainer/global_step": 100, count_key: sample_count},
        ],
    )

    validation, _, _ = analysis.extract_history(tmp_path)

    assert validation.iloc[0].n_samples == sample_count
    assert validation.iloc[0].status == status


def test_null_count_does_not_erase_known_partial_coverage(
    analysis, tmp_path, monkeypatch
):
    monkeypatch.setattr(analysis, "CONDITIONS", {"base_full": None})
    metric_key = analysis.METRICS["omol_rmsd"][2]
    count_key = "val/omol25_structure_prediction/zevals_epoch.num_targets"
    write_history(
        tmp_path,
        [
            {"_step": 1, "trainer/global_step": 100, "trainer/epoch": 19},
            {"_step": 2, "trainer/global_step": 100, metric_key: 1.8},
            {"_step": 3, "trainer/global_step": 100, count_key: 493},
            {"_step": 4, "trainer/global_step": 100, count_key: None},
        ],
    )

    validation, _, _ = analysis.extract_history(tmp_path)

    assert validation.iloc[0].n_samples == 493
    assert validation.iloc[0].status == "measured_wandb_partial"


def test_complete_sources_take_priority_over_partial_measurements(analysis):
    wandb = pd.DataFrame(
        [measurement(19, 0.4), measurement(20, np.nan), measurement(21, np.inf)]
    )
    local = pd.DataFrame(
        [
            measurement(19, 0.5, "measured_local_complete"),
            measurement(19, 0.6, "measured_local_partial"),
            measurement(20, 0.7, "measured_local_complete"),
            measurement(20, 0.8, "measured_local_partial"),
            measurement(21, 0.9, "measured_local_partial"),
        ]
    )

    result = analysis.combine_measurements(wandb, local).set_index("epoch")

    assert result.loc[19, "value"] == 0.4
    assert result.loc[19, "status"] == "measured_wandb"
    assert result.loc[20, "value"] == 0.7
    assert result.loc[20, "status"] == "measured_local_complete"
    assert result.loc[21, "value"] == 0.9
    assert result.loc[21, "status"] == "measured_local_partial"


def test_local_complete_precedes_partial_remote_measurement(analysis):
    wandb = pd.DataFrame([measurement(19, 0.4, "measured_wandb_partial")])
    local = pd.DataFrame([measurement(19, 0.5, "measured_local_complete")])

    result = analysis.combine_measurements(wandb, local).iloc[0]

    assert result.value == 0.5
    assert result.status == "measured_local_complete"


def test_exact_partial_measurement_is_preserved_over_interpolation(analysis):
    series = pd.DataFrame(
        [
            measurement(15, 1.5),
            measurement(19, 1.75, "measured_local_partial", "partial_19.csv"),
            measurement(30, 3.0),
        ]
    )

    result = analysis.estimate_at_epoch(series, 19)

    assert result["value"] == 1.75
    assert result["status"] == "measured_local_partial"
    assert result["source_file"] == "partial_19.csv"
    assert result["n_samples"] == 493
    assert result["left_epoch"] == result["right_epoch"] == 19
    assert result["right_weight"] == 0


def test_linear_interpolation_uses_complete_bracketing_epochs(analysis):
    series = pd.DataFrame(
        [
            measurement(30, 3.0, source="right.csv"),
            measurement(18, 99.0, "measured_local_partial"),
            measurement(20, 99.0, "measured_local_partial"),
            measurement(15, 1.5, source="left.csv"),
        ]
    )

    result = analysis.estimate_at_epoch(series, 19)

    assert result["value"] == pytest.approx(1.9)
    assert result["status"] == "interpolated"
    assert result["left_epoch"] == 15
    assert result["right_epoch"] == 30
    assert result["right_weight"] == pytest.approx(4 / 15)
    assert result["source_file"] == "left.csv|right.csv"
    assert result["n_samples"] is None


@pytest.mark.parametrize("target_epoch", [5, 35])
def test_outside_observed_range_is_missing(analysis, target_epoch):
    series = pd.DataFrame([measurement(15, 1.5), measurement(30, 3.0)])

    result = analysis.estimate_at_epoch(series, target_epoch)

    assert result["status"] == "missing"
    assert np.isnan(result["value"])


def test_entirely_missing_metric_is_retained_as_gap(analysis):
    measurements = pd.DataFrame([measurement(19, 1.7)])

    result = analysis.target_table(measurements, 19)

    assert len(result) == len(analysis.CONDITIONS) * len(analysis.METRICS)
    assert (result.status == "missing").sum() == len(result) - 1
    missing_afd = result[
        (result.condition == "base_500k") & (result.metric == "omol_afd")
    ].iloc[0]
    assert missing_afd.status == "missing"
    assert np.isnan(missing_afd.value)


def test_partial_interpolation_is_explicit_and_preserves_both_sources(analysis):
    series = pd.DataFrame(
        [
            measurement(15, 1.5, "measured_local_partial", "partial_left.csv"),
            measurement(25, 2.5, "measured_local_complete", "complete_right.csv"),
        ]
    )

    result = analysis.estimate_at_epoch(series, 19)

    assert result["value"] == pytest.approx(1.9)
    assert result["status"] == "interpolated_partial"
    assert result["left_epoch"] == 15
    assert result["right_epoch"] == 25
    assert result["right_weight"] == pytest.approx(0.4)
    assert result["source_file"] == "partial_left.csv|complete_right.csv"
    assert result["n_samples"] is None
    assert "1.5" in result["notes"] and "2.5" in result["notes"]


def afd_target_table(analysis):
    rows = []
    for condition, value in (
        ("base_full", 0.8),
        ("large_full", 0.5),
        ("large_500k", 0.4),
    ):
        row = measurement(19, value)
        row.update(condition=condition, metric="omol_afd")
        rows.append(row)
    return analysis.target_table(pd.DataFrame(rows), 19)


def test_afd_scenarios_are_separate_assumption_based_estimates(analysis):
    target = afd_target_table(analysis)
    original = target.copy(deep=True)

    result = analysis.afd_imputation_scenarios(target)

    assert len(result) == 2
    assert set(result.condition) == {"base_500k"}
    assert set(result.metric) == {"omol_afd"}
    assert set(result.status) == {"assumption_based_imputation"}
    values = result.set_index("assumption").value
    assert values["no interaction in raw AFD"] == pytest.approx(0.7)
    assert values["no interaction in log AFD"] == pytest.approx(0.64)
    assert set(result.source_statuses) == {
        "measured_wandb|measured_wandb|measured_wandb"
    }
    pd.testing.assert_frame_equal(target, original)


def test_afd_scenarios_never_replace_observed_base_500k(analysis):
    target = afd_target_table(analysis)
    selection = (target.condition == "base_500k") & (target.metric == "omol_afd")
    target.loc[selection, "value"] = 0.9
    target.loc[selection, "status"] = "measured_wandb"

    result = analysis.afd_imputation_scenarios(target)

    assert result.empty


@pytest.mark.parametrize("missing_anchor", ["base_full", "large_full", "large_500k"])
def test_afd_scenarios_require_all_three_observed_anchors(analysis, missing_anchor):
    target = afd_target_table(analysis)
    selection = (target.condition == missing_anchor) & (target.metric == "omol_afd")
    target.loc[selection, "value"] = np.nan
    target.loc[selection, "status"] = "missing"

    result = analysis.afd_imputation_scenarios(target)

    assert result.empty


def test_recomputed_source_precedence_and_sample_origin(analysis):
    wandb = pd.DataFrame(
        [
            measurement(19, 1.0),
            measurement(21, 3.0, "measured_wandb_partial"),
            measurement(22, 4.0, "measured_wandb_partial"),
        ]
    )
    local = pd.DataFrame([measurement(20, 2.0, "measured_local_complete")])
    recomputed = pd.DataFrame(
        [
            measurement(19, 10.0, "recomputed_saved_complete"),
            measurement(20, 20.0, "recomputed_checkpoint_complete"),
            measurement(21, 30.0, "recomputed_checkpoint_complete"),
            measurement(21, 31.0, "recomputed_saved_complete"),
            measurement(22, 40.0, "recomputed_checkpoint_complete"),
        ]
    )

    result = analysis.combine_measurements(wandb, local, recomputed).set_index("epoch")

    assert list(result.value) == [1.0, 2.0, 31.0, 40.0]
    assert result.loc[21, "sample_origin"] == "rescored_original_samples"
    assert result.loc[22, "sample_origin"] == "independent_checkpoint_resampling"


@pytest.mark.parametrize(
    "status",
    [
        "recomputed_control_failed",
        "recomputed_saved_failed",
        "recomputed_checkpoint_partial",
        "recomputed_checkpoint_shard_complete",
        "recomputed_checkpoint_smoke",
        "samples_complete",
        "unknown",
        None,
        "interpolated",
        "assumption_based_imputation",
    ],
)
def test_unknown_failed_smoke_and_shard_rows_are_never_measurements(analysis, status):
    candidate = pd.DataFrame([measurement(19, 1.0, status)])

    result = analysis.combine_measurements(pd.DataFrame(), pd.DataFrame(), candidate)

    assert result.empty


@pytest.mark.parametrize("count", [None, 0, 493, 499.5, 501, 1000])
def test_complete_recomputation_requires_exact_fixture_coverage(analysis, count):
    row = measurement(19, 1.0, "recomputed_checkpoint_complete")
    row["n_samples"] = count

    result = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame([row])
    )

    assert result.empty


def test_validated_saved_partial_with_more_targets_precedes_smaller_original_partial(
    analysis,
):
    original = measurement(20, 0.2, "measured_wandb_partial")
    original["n_samples"] = 492
    rescored = measurement(20, 0.25, "recomputed_saved_partial")
    rescored["n_samples"] = 497

    result = analysis.combine_measurements(
        pd.DataFrame([original]), pd.DataFrame(), pd.DataFrame([rescored])
    ).iloc[0]

    assert result.value == 0.25
    assert result.n_samples == 497
    assert result.status == "recomputed_saved_partial"
    assert result.sample_origin == "rescored_original_samples"


def afd_measurement(epoch, value, status, *, count=1000, corrected=False):
    row = measurement(epoch, value, status)
    row.update(metric="omol_afd", n_samples=count)
    if corrected:
        row["metric_protocol"] = "species_consistent_afd_v1"
        row["control_validation_passed"] = True
        row["species_consistency_passed"] = True
    return row


def test_corrected_afd_excludes_historical_scores_without_fallback(analysis):
    original = pd.DataFrame(
        [
            afd_measurement(19, 0.2, "measured_wandb"),
            measurement(19, 1.7),
        ]
    )

    result = analysis.combine_measurements(original, pd.DataFrame(), corrected_afd=True)
    target = analysis.target_table(result, 19)

    assert list(result.metric) == ["omol_rmsd"]
    afd = target[
        (target.condition == "base_full") & (target.metric == "omol_afd")
    ].iloc[0]
    assert afd.status == "missing"
    assert np.isnan(afd.value)


def test_corrected_afd_requires_explicit_protocol_and_supersedes_original(analysis):
    original = pd.DataFrame([afd_measurement(19, 0.2, "measured_wandb")])
    recomputed = pd.DataFrame(
        [
            afd_measurement(19, 0.3, "recomputed_saved_complete", corrected=True),
            afd_measurement(20, 0.4, "recomputed_saved_complete"),
        ]
    )

    result = analysis.combine_measurements(
        original, pd.DataFrame(), recomputed, corrected_afd=True
    )

    assert list(result.value) == [0.3]
    assert list(result.status) == ["recomputed_saved_complete"]
    legacy_policy = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), recomputed.iloc[:1]
    )
    assert legacy_policy.empty


def test_partial_corrected_afd_stays_in_audit_without_affecting_primary_interpolation(
    analysis,
):
    recomputed = pd.DataFrame(
        [
            afd_measurement(15, 1.5, "recomputed_saved_complete", corrected=True),
            afd_measurement(
                19, 99.0, "recomputed_saved_partial", count=985, corrected=True
            ),
            afd_measurement(30, 3.0, "recomputed_saved_complete", corrected=True),
        ]
    )

    primary = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), recomputed, corrected_afd=True
    )
    audit = analysis.combine_measurements(
        pd.DataFrame(),
        pd.DataFrame(),
        recomputed,
        corrected_afd=True,
        include_partial_afd=True,
    )
    estimate = analysis.estimate_at_epoch(primary, 19)

    assert list(primary.epoch) == [15, 30]
    assert list(audit.epoch) == [15, 19, 30]
    assert audit.loc[audit.epoch == 19, "n_samples"].item() == 985
    assert estimate["value"] == pytest.approx(1.9)
    assert estimate["left_status"] == "recomputed_saved_complete"


def test_source_control_failure_vetoes_a_stale_complete_csv(analysis, tmp_path):
    source = tmp_path / "recomputed.json"
    source.write_text(
        json.dumps(
            {
                "metric_protocol": "species_consistent_afd_v1",
                "control_validation": {"passed": False},
                "species_consistency": {"passed": True},
            }
        )
    )
    row = afd_measurement(19, 1.0, "recomputed_saved_complete")
    row["source_file"] = source.name
    csv = tmp_path / "recomputed.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)

    candidates = analysis.read_recomputed_measurements(tmp_path, [csv])
    result = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), candidates, corrected_afd=True
    )

    assert candidates.iloc[0].metric_protocol == "species_consistent_afd_v1"
    assert not candidates.iloc[0].control_validation_passed
    assert result.empty


@pytest.mark.parametrize(
    "missing", ["control_validation_passed", "species_consistency_passed"]
)
@pytest.mark.parametrize("absent_value", [None, float("nan"), "", 1])
def test_corrected_afd_requires_affirmative_validation_proofs(
    analysis, missing, absent_value
):
    row = afd_measurement(19, 0.3, "recomputed_saved_complete", corrected=True)
    row[missing] = absent_value
    candidates = pd.DataFrame([row])

    reasons = analysis.measurement_exclusion_reasons(candidates, corrected_afd=True)

    assert reasons.iloc[0] == f"unverified_{missing}"


def test_corrected_afd_protocol_without_proof_columns_is_rejected(analysis):
    row = afd_measurement(19, 0.3, "recomputed_saved_complete")
    row["metric_protocol"] = "species_consistent_afd_v1"

    result = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame([row]), corrected_afd=True
    )

    assert result.empty


@pytest.mark.parametrize("top_level_species", [None, {"passed": False}])
def test_saved_afd_loads_nested_species_proof_and_honors_explicit_failure(
    analysis, tmp_path, top_level_species
):
    source = tmp_path / "recomputed.json"
    metadata = {
        "metric_protocol": "species_consistent_afd_v1",
        "control_validation": {"passed": True},
        "scoring": {"species_consistency": {"passed": True}},
    }
    if top_level_species is not None:
        metadata["species_consistency"] = top_level_species
    source.write_text(json.dumps(metadata))
    row = afd_measurement(19, 1.0, "recomputed_saved_complete")
    row["source_file"] = source.name
    csv = tmp_path / "recomputed.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)

    candidates = analysis.read_recomputed_measurements(tmp_path, [csv])
    result = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), candidates, corrected_afd=True
    )

    assert candidates.iloc[0].control_validation_passed
    assert bool(candidates.iloc[0].species_consistency_passed) == (
        top_level_species is None
    )
    assert result.empty == (top_level_species is not None)


def test_repeated_csv_sources_are_loaded_once_and_missing_explicit_files_raise(
    analysis, tmp_path
):
    paths = [tmp_path / "saved.csv", tmp_path / "checkpoint.csv"]
    for index, path in enumerate(paths):
        pd.DataFrame(
            [measurement(19 + index, 1.0, "recomputed_saved_complete")]
        ).to_csv(path, index=False)

    result = analysis.read_recomputed_measurements(tmp_path, [*paths, paths[0]])

    assert list(result.epoch) == [19, 20]
    assert set(result.recomputed_csv) == {str(path) for path in paths}
    with pytest.raises(FileNotFoundError):
        analysis.read_recomputed_measurements(tmp_path, [tmp_path / "missing.csv"])


def common_diagnostic(tmp_path, entries):
    path = tmp_path / "common.json"
    path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "condition": condition,
                        "dataset": dataset,
                        "epoch": 19,
                        "common_targets": 493,
                        "value": value,
                    }
                    for condition, dataset, value in entries
                ]
            }
        )
    )
    return path


def test_completed_500_target_endpoints_ignore_stale_493_diagnostic(analysis, tmp_path):
    rows = []
    for condition, status, value in [
        ("base_500k", "recomputed_checkpoint_complete", 1.2),
        ("large_500k", "measured_wandb", 1.1),
    ]:
        row = measurement(19, value, status)
        row["condition"] = condition
        rows.append(row)
    target = analysis.target_table(pd.DataFrame(rows), 19)
    diagnostic = common_diagnostic(
        tmp_path,
        [
            ("base_500k", "omol25_structure_prediction", 9.0),
            ("large_500k", "omol25_structure_prediction", 8.0),
        ],
    )

    matched = analysis.matched_target_table(target, diagnostic)

    pd.testing.assert_frame_equal(matched, target)


def test_common_targets_never_replace_independent_checkpoint_samples(
    analysis, tmp_path
):
    checkpoint = measurement(19, 1.2, "recomputed_checkpoint_complete")
    checkpoint["condition"] = "base_500k"
    original = measurement(19, 1.1, "measured_local_partial")
    original["condition"] = "large_500k"
    target = analysis.target_table(pd.DataFrame([checkpoint, original]), 19)
    diagnostic = common_diagnostic(
        tmp_path,
        [
            ("base_500k", "omol25_structure_prediction", 9.0),
            ("large_500k", "omol25_structure_prediction", 8.0),
        ],
    )

    matched = analysis.matched_target_table(target, diagnostic)
    checkpoint_row = matched[
        (matched.condition == "base_500k") & (matched.metric == "omol_rmsd")
    ].iloc[0]
    original_row = matched[
        (matched.condition == "large_500k") & (matched.metric == "omol_rmsd")
    ].iloc[0]

    assert checkpoint_row.value == 1.2
    assert checkpoint_row.status == "recomputed_checkpoint_complete"
    assert checkpoint_row.n_samples == 500
    assert original_row.value == 8.0
    assert original_row.status == "measured_common_targets"
    assert original_row.original_value == 1.1
    assert original_row.original_status == "measured_local_partial"


def test_common_target_restriction_is_gated_per_metric(analysis, tmp_path):
    rows = []
    for metric in ["omol_rmsd", "omol_force_mae"]:
        for condition in ["base_500k", "large_500k"]:
            status = (
                "measured_local_partial"
                if metric == "omol_force_mae" and condition == "base_500k"
                else "measured_wandb"
            )
            row = measurement(19, 1.0, status)
            row.update(condition=condition, metric=metric)
            rows.append(row)
    target = analysis.target_table(pd.DataFrame(rows), 19)
    diagnostic = common_diagnostic(
        tmp_path,
        [
            (condition, dataset, 9.0)
            for condition in ["base_500k", "large_500k"]
            for dataset in ["omol25_structure_prediction", "omol25_mlip"]
        ],
    )

    matched = analysis.matched_target_table(target, diagnostic)

    structure = matched[
        (matched.metric == "omol_rmsd")
        & matched.condition.isin(["base_500k", "large_500k"])
    ]
    force = matched[
        (matched.metric == "omol_force_mae")
        & matched.condition.isin(["base_500k", "large_500k"])
    ]
    assert list(structure.value) == [1.0, 1.0]
    assert list(structure.n_samples) == [500, 500]
    assert set(force.status) == {"measured_common_targets"}
    assert list(force.n_samples) == [493, 493]


def encoded_afd(epoch, value, model, *, count=1000, metric="omol_afd"):
    row = afd_measurement(
        epoch,
        value,
        "recomputed_saved_complete" if count == 1000 else "recomputed_saved_partial",
        count=count,
        corrected=True,
    )
    row.update(metric=metric, afd_model_name=model)
    return row


@pytest.mark.parametrize("source_model", ["Ty-Perez/ct-scd-geom10", None])
def test_encoder_identity_is_loaded_from_source_not_csv(
    analysis, tmp_path, source_model
):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "metric_protocol": analysis.CORRECTED_AFD_PROTOCOL,
                "control_validation": {"passed": True},
                "scoring": {
                    "model_name": source_model,
                    "species_consistency": {"passed": True},
                },
            }
        )
    )
    row = encoded_afd(19, 0.3, analysis.DEFAULT_AFD_MODELS["omol_afd"])
    row["source_file"] = str(source)
    path = tmp_path / "recomputed.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    loaded = analysis.read_recomputed_measurements(tmp_path, [path])
    result = analysis.combine_measurements(
        pd.DataFrame(),
        pd.DataFrame(),
        loaded,
        corrected_afd=True,
        expected_afd_models=analysis.DEFAULT_AFD_MODELS,
    )

    assert result.empty
    assert loaded.iloc[0].afd_model_name == source_model


def test_default_discovery_includes_versioned_molecular_rescoring(analysis, tmp_path):
    directory = tmp_path / "recomputed/omol25_encoder_update"
    directory.mkdir(parents=True)
    row = encoded_afd(19, 0.3, analysis.DEFAULT_AFD_MODELS["omol_afd"])
    pd.DataFrame([row]).to_csv(directory / "recomputed_metrics.csv", index=False)

    result = analysis.read_recomputed_measurements(tmp_path)

    assert len(result) == 1
    assert result.iloc[0].recomputed_csv == str(directory / "recomputed_metrics.csv")


@pytest.mark.parametrize("metric", ["omol_afd", "omat_afd"])
@pytest.mark.parametrize("count", [1000, 769])
def test_expected_encoder_filters_complete_and_partial_before_source_precedence(
    analysis, metric, count
):
    model = analysis.DEFAULT_AFD_MODELS[metric]
    candidates = pd.DataFrame(
        [
            encoded_afd(19, 0.3, model, count=count, metric=metric),
            encoded_afd(19, 0.1, "Ty-Perez/ct-scd-geom10", count=count, metric=metric),
            encoded_afd(20, 0.2, None, count=count, metric=metric),
            measurement(19, 1.7),
        ]
    )

    result = analysis.combine_measurements(
        pd.DataFrame(),
        pd.DataFrame(),
        candidates,
        corrected_afd=True,
        include_partial_afd=True,
        expected_afd_models=analysis.DEFAULT_AFD_MODELS,
    )
    reasons = analysis.measurement_exclusion_reasons(
        candidates,
        corrected_afd=True,
        include_partial_afd=True,
        expected_afd_models=analysis.DEFAULT_AFD_MODELS,
    )

    selected = result[result.metric.eq(metric)].iloc[0]
    assert selected.value == 0.3
    assert selected.n_samples == count
    assert selected.afd_model_name == model
    assert len(result) == 2  # Non-AFD measurements retain their existing policy.
    assert list(reasons) == [
        "",
        "incompatible_afd_model_name",
        "unverified_afd_model_name",
        "",
    ]


def test_explicit_geom10_policy_reproduces_old_complete_and_partial_selection(analysis):
    old, new = "Ty-Perez/ct-scd-geom10", analysis.DEFAULT_AFD_MODELS["omol_afd"]
    candidates = pd.DataFrame(
        [
            encoded_afd(15, 0.2, old),
            encoded_afd(15, 0.3, new),
            encoded_afd(20, 0.4, old, count=769),
            encoded_afd(20, 0.5, new, count=769),
        ]
    )

    result = analysis.combine_measurements(
        pd.DataFrame(),
        pd.DataFrame(),
        candidates,
        corrected_afd=True,
        include_partial_afd=True,
        expected_afd_models={**analysis.DEFAULT_AFD_MODELS, "omol_afd": old},
    )

    assert list(result.value) == [0.2, 0.4]
    assert set(result.afd_model_name) == {old}


@pytest.mark.parametrize("right_model", ["Ty-Perez/ct-scd-geom10", None])
def test_interpolation_rejects_different_or_unverified_encoder_identity(
    analysis, right_model
):
    series = pd.DataFrame(
        [
            encoded_afd(15, 1.5, analysis.DEFAULT_AFD_MODELS["omol_afd"]),
            encoded_afd(25, 2.5, right_model),
        ]
    )

    with pytest.raises(ValueError, match="different or unverified AFD models"):
        analysis.estimate_at_epoch(series, 19)


def test_expected_encoder_cannot_fall_back_to_an_exact_old_encoder_result(analysis):
    model = analysis.DEFAULT_AFD_MODELS["omol_afd"]
    series = pd.DataFrame(
        [
            encoded_afd(15, 1.5, model),
            encoded_afd(19, 99, "Ty-Perez/ct-scd-geom10"),
            encoded_afd(25, 2.5, model),
        ]
    )

    target = analysis.target_table(
        series, 19, expected_afd_models=analysis.DEFAULT_AFD_MODELS
    )
    row = target[target.condition.eq("base_full") & target.metric.eq("omol_afd")].iloc[
        0
    ]

    assert row.status == "interpolated"
    assert row.value == pytest.approx(1.9)
    assert (
        row.afd_model_name
        == row.left_afd_model_name
        == row.right_afd_model_name
        == model
    )
    absent = target[
        target.condition.eq("base_500k") & target.metric.eq("omol_afd")
    ].iloc[0]
    assert absent.status == "missing" and absent.afd_model_name == model


def test_sensitivity_keeps_only_the_selected_encoders_partial_pool(analysis):
    model = analysis.DEFAULT_AFD_MODELS["omol_afd"]
    candidates = pd.DataFrame(
        [
            encoded_afd(15, 1.5, model),
            encoded_afd(20, 4, model, count=769),
            encoded_afd(25, 2.5, model),
            encoded_afd(20, 99, "Ty-Perez/ct-scd-geom10", count=769),
        ]
    )
    options = dict(corrected_afd=True, expected_afd_models=analysis.DEFAULT_AFD_MODELS)
    primary = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), candidates, **options
    )
    audit = analysis.combine_measurements(
        pd.DataFrame(), pd.DataFrame(), candidates, include_partial_afd=True, **options
    )
    target = analysis.target_table(primary, 19)

    result = analysis.interpolation_sensitivity(audit, target).iloc[0]

    assert result.afd_model_name == model
    assert result.nearest_right_n == 769
    assert result.nearest_bracket_value == pytest.approx(3.5)
    audit.loc[audit.epoch.eq(20), "afd_model_name"] = "Ty-Perez/ct-scd-geom10"
    with pytest.raises(ValueError, match="different or unverified AFD models"):
        analysis.interpolation_sensitivity(audit, target)


@pytest.mark.parametrize("override", [None, "Ty-Perez/ct-scd-geom10"])
def test_cli_selects_new_molecular_encoder_unless_explicitly_overridden(
    analysis, monkeypatch, override
):
    parse_args = analysis.argparse.ArgumentParser.parse_args
    arguments = [] if override is None else ["--omol-afd-model", override]

    def parse_and_stop(parser):
        parsed = parse_args(parser, arguments)
        assert parsed.omol_afd_model == (override or "Ty-Perez/ct-scd-omol25")
        assert analysis.DEFAULT_AFD_MODELS["omat_afd"] == "Ty-Perez/ct-scd-amp20"
        raise StopIteration  # Stop before opening or writing any analysis artifacts.

    monkeypatch.setattr(analysis.argparse.ArgumentParser, "parse_args", parse_and_stop)
    with pytest.raises(StopIteration):
        analysis.main()
