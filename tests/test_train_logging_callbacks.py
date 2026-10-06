import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torchmetrics.aggregation import MeanMetric

from nucleus.callbacks.train_logging import (
    LogAF3TrainingLossesCallback,
    LogConditioningGradientNormsCallback,
    LogModelHealthCallback,
    _epoch_progress_percentage,
)


class _RecordingFabric:
    def __init__(self, is_global_zero: bool = True) -> None:
        self.is_global_zero = is_global_zero
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.global_rank = 0

    def log_dict(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((metrics, step))

    @staticmethod
    def all_reduce(value, reduce_op: str):
        assert reduce_op == "sum"
        return value

    @staticmethod
    def to_device(value):
        return value


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 1, bias=False)
        self.token_head = nn.Linear(1, 1, bias=False)
        self.atom_type_head = nn.Linear(1, 1, bias=False)
        # Mirrors the detached shadow parameters in the trainer's EMA wrapper.
        self.ema_weight = nn.Parameter(torch.tensor([1000.0]), requires_grad=False)


def _make_trainer(step: int, is_global_zero: bool = True) -> SimpleNamespace:
    model = _ToyModel()
    model.backbone.weight.data.copy_(torch.tensor([[3.0, 4.0]]))
    model.token_head.weight.data.fill_(12.0)
    model.atom_type_head.weight.data.fill_(5.0)
    model.backbone.weight.grad = torch.tensor([[6.0, 8.0]])
    model.atom_type_head.weight.grad = torch.zeros_like(model.atom_type_head.weight)

    return SimpleNamespace(
        fabric=_RecordingFabric(is_global_zero=is_global_zero),
        state={"global_step": step, "model": model},
    )


def test_model_health_callback_logs_compact_aggregates() -> None:
    callback = LogModelHealthCallback(
        log_freq=100,
        parameter_groups={
            "backbone": ["backbone"],
            "token_head": ["token_head"],
            "atom_type_head": ["atom_type_head"],
        },
    )
    trainer = _make_trainer(step=100)

    callback.on_before_optimizer_step(trainer, optimizer=None)

    assert len(trainer.fabric.logged) == 1
    metrics, step = trainer.fabric.logged[0]
    assert step == 100
    assert len(metrics) == 22
    torch.testing.assert_close(
        metrics["train/health/weights/l2_norm"], torch.tensor(194.0).sqrt()
    )
    torch.testing.assert_close(
        metrics["train/health/grads/l2_norm"], torch.tensor(10.0)
    )
    torch.testing.assert_close(
        metrics["train/health/gradient_element_coverage"], torch.tensor(0.75)
    )
    torch.testing.assert_close(
        metrics["train/health/groups/token_head/grads/rms"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        metrics["train/health/groups/token_head/gradient_element_coverage"],
        torch.tensor(0.0),
    )


@pytest.mark.parametrize("step", [1, 99, 101])
def test_model_health_callback_respects_log_frequency(step: int) -> None:
    callback = LogModelHealthCallback(log_freq=100)
    trainer = _make_trainer(step=step)

    callback.on_before_optimizer_step(trainer, optimizer=None)

    assert trainer.fabric.logged == []


def test_model_health_callback_only_logs_on_global_zero() -> None:
    callback = LogModelHealthCallback(log_freq=100)
    trainer = _make_trainer(step=100, is_global_zero=False)

    callback.on_before_optimizer_step(trainer, optimizer=None)

    assert trainer.fabric.logged == []


def test_model_health_callback_counts_nonfinite_values() -> None:
    callback = LogModelHealthCallback(log_freq=100)
    trainer = _make_trainer(step=100)
    trainer.state["model"].backbone.weight.data[0, 0] = torch.nan
    trainer.state["model"].backbone.weight.grad[0, 0] = torch.inf

    callback.on_before_optimizer_step(trainer, optimizer=None)

    metrics, _ = trainer.fabric.logged[0]
    assert metrics["train/health/weights/nonfinite_count"].item() == 1
    assert metrics["train/health/grads/nonfinite_count"].item() == 1
    torch.testing.assert_close(
        metrics["train/health/weights/l2_norm"], torch.tensor(185.0).sqrt()
    )
    torch.testing.assert_close(metrics["train/health/grads/l2_norm"], torch.tensor(8.0))


def test_model_health_callback_rejects_nonpositive_frequency() -> None:
    with pytest.raises(ValueError, match="log_freq must be positive"):
        LogModelHealthCallback(log_freq=0)


class _ToyConditioningModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.force = nn.Linear(2, 2, bias=False)
        self.graph = nn.Linear(2, 2, bias=False)
        self.atom = nn.Linear(2, 2, bias=False)
        for module in (self.force, self.graph, self.atom):
            module.weight.data.copy_(torch.eye(2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        force_embedding = self.force(features)
        graph_features = self.graph(features)
        atom_features = self.atom(features)
        return (
            2.0 * force_embedding.sum()
            + 4.0 * graph_features.sum()
            + 8.0 * atom_features.sum()
        )


def test_conditioning_gradient_callback_logs_normalized_feature_sensitivity() -> None:
    callback = LogConditioningGradientNormsCallback(
        enabled=True,
        log_every_n=10,
        module_groups={
            "force_embedding": ["force"],
            "graph_features": ["graph"],
            "atom_features": ["atom"],
        },
    )
    trainer = SimpleNamespace(
        fabric=_RecordingFabric(),
        state={"global_step": 10, "model": _ToyConditioningModel()},
    )
    callback.on_fit_start(trainer)
    callback.on_train_batch_start(trainer, batch=None, batch_idx=0)

    features = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    trainer.state["model"](features).backward()
    callback.on_before_optimizer_step(trainer, optimizer=None)

    assert len(trainer.fabric.logged) == 1
    metrics, step = trainer.fabric.logged[0]
    assert step == 10
    assert len(metrics) == 17
    torch.testing.assert_close(
        metrics["train/conditioning_gradients/force_embedding/gradient_rms"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        metrics["train/conditioning_gradients/graph_features/gradient_rms"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        metrics["train/conditioning_gradients/atom_features/gradient_rms"],
        torch.tensor(8.0),
    )
    torch.testing.assert_close(
        metrics["train/conditioning_gradients/force_to_graph_features_gradient_rms"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        metrics["train/conditioning_gradients/force_to_atom_features_gradient_rms"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        metrics["train/conditioning_gradients/force_embedding/activation_rms"],
        torch.tensor(2.5),
    )

    callback.on_fit_end(trainer)
    assert callback._hook_handles == []


def test_conditioning_gradient_callback_is_noop_when_disabled() -> None:
    callback = LogConditioningGradientNormsCallback(enabled=False)
    trainer = SimpleNamespace(
        fabric=_RecordingFabric(),
        state={"global_step": 0, "model": _ToyConditioningModel()},
    )

    callback.on_fit_start(trainer)
    callback.on_train_batch_start(trainer, batch=None, batch_idx=0)
    trainer.state["model"](torch.ones((1, 2))).backward()
    callback.on_before_optimizer_step(trainer, optimizer=None)

    assert trainer.fabric.logged == []
    assert callback._hook_handles == []


def test_conditioning_gradient_callback_rejects_missing_module_group() -> None:
    callback = LogConditioningGradientNormsCallback(
        enabled=True,
        module_groups={"force_embedding": ["not_present"]},
    )
    trainer = SimpleNamespace(
        fabric=_RecordingFabric(),
        state={"global_step": 0, "model": _ToyConditioningModel()},
    )

    with pytest.raises(ValueError, match="force_embedding"):
        callback.on_fit_start(trainer)


def test_epoch_losses_use_global_step_and_do_not_overwrite_it_with_epoch() -> None:
    callback = LogAF3TrainingLossesCallback(log_every_n=25)
    tracker = MeanMetric()
    tracker.update(torch.tensor(2.0))
    tracker.update(torch.tensor(4.0))
    callback.loss_trackers = {"total_loss": tracker}
    callback.start_time = time.time()
    trainer = SimpleNamespace(
        fabric=_RecordingFabric(),
        state={
            "current_epoch": 7,
            "global_step": 29_310,
            "batches_completed_in_epoch": 2,
            "callback_states": {},
        },
    )

    callback.on_train_epoch_end(trainer)

    assert trainer.fabric.logged == [({"train/per_epoch_total_loss": 3.0}, 29_310)]


def test_batch_loss_aggregation_does_not_materialize_on_nonzero_rank(
    monkeypatch,
) -> None:
    """Ordinary batch accounting must not synchronize CUDA tensors via item()."""
    callback = LogAF3TrainingLossesCallback(log_every_n=25)
    trainer = _make_trainer(step=1, is_global_zero=False)

    def fail_item(self):
        raise AssertionError("Tensor.item() must not run during batch aggregation")

    with monkeypatch.context() as item_guard:
        item_guard.setattr(torch.Tensor, "item", fail_item)
        callback.on_train_batch_end(
            trainer,
            outputs={
                "loss_dict": {
                    "total_loss": torch.tensor([2.0, 4.0]),
                    "aux_loss": torch.tensor([1.0, 3.0]),
                }
            },
            batch=[{"example_id": "example"}],
            batch_idx=1,
        )

    torch.testing.assert_close(
        callback.loss_trackers["total_loss"].compute(), torch.tensor(3.0)
    )
    torch.testing.assert_close(
        callback.loss_trackers["aux_loss"].compute(), torch.tensor(2.0)
    )
    assert trainer.fabric.logged == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_scalar_materialization_supports_mixed_cpu_and_cuda_devices() -> None:
    """CPU diagnostics and CUDA losses may coexist in one loss dictionary."""
    values = LogAF3TrainingLossesCallback._materialize_scalar_tensors(
        {
            "cpu_metric": torch.tensor(1.25),
            "cuda_loss": torch.tensor(2.5, device="cuda"),
            "second_cuda_loss": torch.tensor(3.75, device="cuda"),
        }
    )

    assert values == {
        "cpu_metric": 1.25,
        "cuda_loss": 2.5,
        "second_cuda_loss": 3.75,
    }


def test_epoch_loss_aggregation_does_not_materialize_on_nonzero_rank(
    monkeypatch,
) -> None:
    """Only rank zero should transfer epoch loss summaries to the CPU."""
    callback = LogAF3TrainingLossesCallback(log_every_n=25)
    tracker = MeanMetric()
    tracker.update(torch.tensor(3.0))
    callback.loss_trackers = {"total_loss": tracker}
    callback.start_time = time.time()
    trainer = SimpleNamespace(
        fabric=_RecordingFabric(is_global_zero=False),
        state={
            "current_epoch": 0,
            "global_step": 1,
            "callback_states": {},
        },
    )

    def fail_item(self):
        raise AssertionError("Tensor.item() must not run on a nonzero rank")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    callback.on_train_epoch_end(trainer)

    assert trainer.fabric.logged == []


def test_epoch_loss_accumulators_survive_step_checkpoint_resume() -> None:
    callback = LogAF3TrainingLossesCallback(log_every_n=25)
    tracker = MeanMetric()
    tracker.update(torch.tensor(2.0))
    tracker.update(torch.tensor(4.0))
    callback.loss_trackers = {"total_loss": tracker}
    callback.start_time = time.time()
    state = {"callback_states": {}, "batches_completed_in_epoch": 2}
    trainer = SimpleNamespace(
        fabric=_RecordingFabric(),
        state=state,
        _gather_rank_states=lambda local_state: [local_state],
    )

    callback.on_save_checkpoint(trainer, state)
    resumed = LogAF3TrainingLossesCallback(log_every_n=25)
    resumed.on_fit_start(trainer)

    assert resumed.loss_trackers["total_loss"].update_count == 2
    torch.testing.assert_close(
        resumed.loss_trackers["total_loss"].compute(), torch.tensor(3.0)
    )


def test_token_budget_progress_uses_examples_in_packed_batches() -> None:
    trainer = SimpleNamespace(
        state={"examples_completed_in_epoch": 80},
        n_examples_per_rank=200,
        n_batches_per_epoch=200,
    )

    progress = _epoch_progress_percentage(trainer, [{} for _ in range(20)])

    assert progress == 50.0


def test_legacy_resume_does_not_report_inexact_epoch_progress() -> None:
    trainer = SimpleNamespace(
        state={
            "examples_completed_in_epoch": 20,
            "epoch_progress_is_exact": False,
        },
        n_examples_per_rank=200,
        n_batches_per_epoch=200,
    )

    assert _epoch_progress_percentage(trainer, [{}]) is None
