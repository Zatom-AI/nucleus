from types import SimpleNamespace

import numpy as np
import pytest
import rfd3na.metrics.hbonds_hbplus_metrics as upstream_hbond_metrics
from biotite.structure import AtomArray
from rfd3na.constants import association_schemes
from rfd3na.metrics.hbonds_hbplus_metrics import (
    HbondMetrics as RFD3NAHbondMetrics,
)
from rfd3na.metrics.hbonds_hbplus_metrics import (
    simplified_processing_atom_array,
)

from nucleus.metrics.hbonds import HbondMetrics


def _padded_rna_output() -> AtomArray:
    atom_names = association_schemes["atom23"]["A"]
    atom_array = AtomArray(len(atom_names))
    atom_array.coord = np.arange(len(atom_names) * 3, dtype=float).reshape(-1, 3)
    atom_array.atom_name = np.asarray(
        [f"V{index}" for index in range(len(atom_names))], dtype=str
    )
    atom_array.element = np.full(len(atom_names), "VX")
    atom_array.res_name = np.full(len(atom_names), "A")
    atom_array.chain_id = np.full(len(atom_names), "A")
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.set_annotation("chain_iid", np.full(len(atom_names), "A_1"))
    atom_array.set_annotation(
        "gt_atom_name",
        np.asarray(
            [name.strip() if name is not None else "VX" for name in atom_names],
            dtype=str,
        ),
    )
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_coord",
        np.ones(len(atom_names), dtype=bool),
    )
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_seq",
        np.zeros(len(atom_names), dtype=bool),
    )
    atom_array.set_annotation(
        "is_motif_atom_unindexed",
        np.zeros(len(atom_names), dtype=bool),
    )
    return atom_array


def _rna_reference() -> AtomArray:
    atom_names = [
        name.strip() for name in association_schemes["atom23"]["A"] if name is not None
    ]
    atom_array = AtomArray(len(atom_names))
    atom_array.coord = np.arange(len(atom_names) * 3, dtype=float).reshape(-1, 3)
    atom_array.atom_name = np.asarray(atom_names, dtype=str)
    atom_array.element = np.full(len(atom_names), "C")
    atom_array.res_name = np.full(len(atom_names), "A")
    atom_array.chain_id = np.full(len(atom_names), "A")
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.set_annotation("chain_iid", np.full(len(atom_names), "A_1"))
    atom_array.set_annotation("gt_atom_name", np.asarray(atom_names, dtype=str))
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_coord",
        np.ones(len(atom_names), dtype=bool),
    )
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_seq",
        np.zeros(len(atom_names), dtype=bool),
    )
    atom_array.set_annotation(
        "is_motif_atom_unindexed",
        np.zeros(len(atom_names), dtype=bool),
    )
    atom_array.set_annotation("active_donor", atom_array.atom_name == "N6")
    atom_array.set_annotation("active_acceptor", np.zeros(len(atom_names), dtype=bool))
    return atom_array


def test_upstream_hbond_metric_cleanup_crashes_on_padded_rna() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        simplified_processing_atom_array([_padded_rna_output()])


def test_hbond_metrics_clean_padded_rna_before_upstream_compute(monkeypatch) -> None:
    predicted = _padded_rna_output()
    captured = {}

    def fake_compute(
        self,
        *,
        ground_truth_atom_array_stack,
        predicted_atom_array_stack,
    ):
        captured["ground_truth"] = ground_truth_atom_array_stack
        captured["predicted"] = predicted_atom_array_stack
        return {"mean_num_hbonds": 2.0}

    monkeypatch.setattr(RFD3NAHbondMetrics, "compute", fake_compute)
    metric = HbondMetrics(association_scheme="atom23")

    result = metric.compute(
        ground_truth_atom_array_stack=["ground-truth"],
        predicted_atom_array_stack=[predicted],
    )

    cleaned = captured["predicted"][0]
    assert result == {"mean_num_hbonds": 2.0}
    assert captured["ground_truth"] == ["ground-truth"]
    assert len(cleaned) == 22
    assert "C1'" in cleaned.atom_name
    assert not np.any(cleaned.element == "VX")
    assert len(predicted) == 23
    assert np.all(predicted.element == "VX")


def test_hbond_metrics_compute_for_padded_rna(monkeypatch) -> None:
    def fake_calculate_hbonds(atom_array, **kwargs):
        del kwargs
        atom_array.set_annotation("active_donor", atom_array.atom_name == "N6")
        atom_array.set_annotation(
            "active_acceptor", np.zeros(len(atom_array), dtype=bool)
        )
        return atom_array, [], 1

    monkeypatch.setattr(
        upstream_hbond_metrics,
        "calculate_hbonds",
        fake_calculate_hbonds,
    )

    result = HbondMetrics(association_scheme="atom23").compute(
        ground_truth_atom_array_stack=[_rna_reference()],
        predicted_atom_array_stack=[_padded_rna_output()],
    )

    assert result == {
        "mean_correct_donors_percent": 1.0,
        "mean_correct_acceptors_percent": 1.0,
        "mean_num_hbonds": 1.0,
    }


class _RecordingFabric:
    is_global_zero = True

    def __init__(self) -> None:
        self.logged = []

    @staticmethod
    def to_device(value):
        return value

    @staticmethod
    def all_reduce(value, reduce_op):
        assert reduce_op == "sum"
        return value

    def log_dict(self, metrics, step):
        self.logged.append((metrics, step))


def test_hbond_conditioning_callback_reports_effective_training_signal() -> None:
    from nucleus.callbacks.train_logging import LogHbondConditioningCallback

    fabric = _RecordingFabric()
    trainer = SimpleNamespace(fabric=fabric, state={"global_step": 17})
    callback = LogHbondConditioningCallback()
    callback.on_fit_start(trainer)
    callback.on_train_batch_start(
        trainer,
        batch=[
            {"log_dict": {}},
            {
                "log_dict": {
                    "hbplus_available": 1,
                    "hbond_total_count": 4,
                    "hbond_subsample_atoms": 2,
                }
            },
            {
                "log_dict": {
                    "hbplus_available": 1,
                    "hbond_total_count": 0,
                    "hbond_subsample_atoms": 0,
                }
            },
            {
                "log_dict": {
                    "hbplus_available": 0,
                    "hbplus_unavailable_runtime_error": 1,
                }
            },
        ],
        batch_idx=0,
    )
    callback.on_train_epoch_end(trainer)

    metrics, step = fabric.logged[0]
    assert step == 17
    assert metrics["train/hbond_conditioning/attempt_fraction"] == pytest.approx(0.75)
    assert metrics["train/hbond_conditioning/availability_fraction"] == pytest.approx(
        2 / 3
    )
    assert metrics["train/hbond_conditioning/nonzero_fraction"] == pytest.approx(0.5)
    assert metrics["train/hbond_conditioning/mean_detected_atoms"] == pytest.approx(2.0)
    assert metrics["train/hbond_conditioning/mean_provided_atoms"] == pytest.approx(1.0)
    assert metrics[
        "train/hbond_conditioning/unavailable/runtime_error"
    ] == pytest.approx(1 / 3)
    assert callback._counts == {}
