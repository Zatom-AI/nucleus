from types import SimpleNamespace

import pandas as pd
import pytest

from nucleus.callbacks.metrics_logging import LogCompositeValidationMetricsCallback


class _RecordingFabric:
    def __init__(self, *, is_global_zero: bool = True) -> None:
        self.is_global_zero = is_global_zero
        self.logged: list[tuple[dict[str, float], int]] = []

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((metrics, step))


def _task_metrics():
    return {
        "generation": [
            {
                "dataset": "generation",
                "metric": "quality",
                "direction": "higher",
                "bad": 0.0,
                "good": 1.0,
            }
        ],
        "structure_prediction": [
            {
                "dataset": "structure_prediction",
                "metric": "error",
                "direction": "lower",
                "bad": 4.0,
                "good": 1.0,
            }
        ],
        "mlip": [
            {
                "dataset": "mlip",
                "metric": "error",
                "direction": "lower",
                "bad": 100.0,
                "good": 1.0,
            }
        ],
    }


def _afd_task_metrics():
    return {
        "generation": [
            {
                "dataset": "generation",
                "metric": "afd_score",
                "direction": "lower",
                "bad": 0.35,
                "good": 0.25,
            }
        ]
    }


def _trainer(tmp_path, rows, frequencies, validation_datasets=None):
    validation_results_path = tmp_path / "validation_output_all_epochs.csv"
    pd.DataFrame(rows).to_csv(validation_results_path, index=False)
    fabric = _RecordingFabric()
    train_conditions = {
        condition: {"frequency": frequency}
        for condition, frequency in frequencies.items()
    }
    return SimpleNamespace(
        fabric=fabric,
        validation_results_path=validation_results_path,
        state={
            "current_epoch": 3,
            "global_step": 12_000,
            "train_cfg": {
                "datasets": {
                    "global_transform_args": {
                        "train_conditions": train_conditions,
                    },
                    **(
                        {"val": {dataset: {} for dataset in validation_datasets}}
                        if validation_datasets is not None
                        else {}
                    ),
                }
            },
        },
    )


def test_composite_validation_metrics_log_full_balanced_score(tmp_path):
    trainer = _trainer(
        tmp_path,
        rows=[
            {"epoch": 3, "dataset": "generation", "quality": 0.5},
            {"epoch": 3, "dataset": "structure_prediction", "error": 2.0},
            {"epoch": 3, "dataset": "mlip", "error": 10.0},
        ],
        frequencies={
            "unconditional": 0.34,
            "structure_prediction": 0.33,
            "mlip": 0.33,
        },
    )

    LogCompositeValidationMetricsCallback(
        task_metrics=_task_metrics()
    ).on_validation_epoch_end(trainer)

    assert len(trainer.fabric.logged) == 1
    metrics, step = trainer.fabric.logged[0]
    assert step == 12_000
    assert metrics["val/composite/task/generation"] == pytest.approx(0.5)
    assert metrics["val/composite/task/structure_prediction"] == pytest.approx(0.5)
    assert metrics["val/composite/task/mlip"] == pytest.approx(0.5)
    assert metrics["val/composite/available"] == pytest.approx(0.5)
    assert metrics["val/composite/full"] == pytest.approx(0.5)
    assert metrics["val/composite/worst_available"] == pytest.approx(0.5)
    assert metrics["val/composite/coverage"] == pytest.approx(1.0)


@pytest.mark.parametrize("num_samples", [None, float("nan"), 0.0, 1.0])
def test_composite_validation_metrics_exclude_afd_with_too_few_samples(
    tmp_path, num_samples
):
    trainer = _trainer(
        tmp_path,
        rows=[
            {
                "epoch": 3,
                "dataset": "generation",
                "afd_score": 0.0,
                "afd_num_samples": num_samples,
            }
        ],
        frequencies={"unconditional": 1.0},
    )

    LogCompositeValidationMetricsCallback(
        task_metrics=_afd_task_metrics()
    ).on_validation_epoch_end(trainer)

    metrics, _ = trainer.fabric.logged[0]
    assert metrics["val/composite/coverage"] == 0.0
    assert metrics["val/composite/available_task_count"] == 0.0
    assert "val/composite/task/generation" not in metrics
    assert "val/composite/available" not in metrics
    assert "val/composite/full" not in metrics


def test_composite_validation_metrics_allow_valid_zero_afd_score(tmp_path):
    trainer = _trainer(
        tmp_path,
        rows=[
            {
                "epoch": 3,
                "dataset": "generation",
                "afd_score": 0.0,
                "afd_num_samples": 2.0,
            }
        ],
        frequencies={"unconditional": 1.0},
    )

    LogCompositeValidationMetricsCallback(
        task_metrics=_afd_task_metrics()
    ).on_validation_epoch_end(trainer)

    metrics, _ = trainer.fabric.logged[0]
    assert metrics["val/composite/coverage"] == 1.0
    assert metrics["val/composite/task/generation"] == 1.0
    assert metrics["val/composite/available"] == 1.0
    assert metrics["val/composite/full"] == 1.0


def test_composite_validation_metrics_report_partial_active_task_coverage(tmp_path):
    trainer = _trainer(
        tmp_path,
        rows=[{"epoch": 3, "dataset": "mlip", "error": 10.0}],
        frequencies={
            "unconditional": 0.34,
            "structure_prediction": 0.33,
            "mlip": 0.33,
        },
    )

    LogCompositeValidationMetricsCallback(
        task_metrics=_task_metrics()
    ).on_validation_epoch_end(trainer)

    metrics, _ = trainer.fabric.logged[0]
    assert metrics["val/composite/available"] == pytest.approx(0.5)
    assert metrics["val/composite/coverage"] == pytest.approx(1 / 3)
    assert metrics["val/composite/task_coverage/generation"] == 0.0
    assert metrics["val/composite/task_coverage/structure_prediction"] == 0.0
    assert metrics["val/composite/task_coverage/mlip"] == 1.0
    assert "val/composite/full" not in metrics


def test_composite_validation_metrics_support_mlip_only_training(tmp_path):
    trainer = _trainer(
        tmp_path,
        rows=[{"epoch": 3, "dataset": "mlip", "error": 10.0}],
        frequencies={
            "unconditional": 0.0,
            "structure_prediction": 0.0,
            "mlip": 1.0,
        },
    )

    LogCompositeValidationMetricsCallback(
        task_metrics=_task_metrics()
    ).on_validation_epoch_end(trainer)

    metrics, _ = trainer.fabric.logged[0]
    assert metrics["val/composite/active_task_count"] == 1.0
    assert metrics["val/composite/available_task_count"] == 1.0
    assert metrics["val/composite/coverage"] == 1.0
    assert metrics["val/composite/full"] == pytest.approx(0.5)


def test_composite_validation_metrics_ignore_unconfigured_domain_metrics(tmp_path):
    trainer = _trainer(
        tmp_path,
        rows=[{"epoch": 3, "dataset": "omol_generation", "quality": 0.5}],
        frequencies={"unconditional": 1.0},
        validation_datasets=["omol_generation"],
    )
    task_metrics = {
        "generation": [
            {
                "dataset": dataset,
                "metric": "quality",
                "direction": "higher",
                "bad": 0.0,
                "good": 1.0,
            }
            for dataset in ("omol_generation", "omat_generation")
        ]
    }

    LogCompositeValidationMetricsCallback(
        task_metrics=task_metrics
    ).on_validation_epoch_end(trainer)

    metrics, _ = trainer.fabric.logged[0]
    assert metrics["val/composite/coverage"] == 1.0
    assert metrics["val/composite/full"] == pytest.approx(0.5)
