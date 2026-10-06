from types import SimpleNamespace

import torch

from nucleus.callbacks.train_logging import LogModelDiagnosticsCallback
from nucleus.metrics.tensor_diagnostics import (
    summarize_logit_scale,
    summarize_tensor_scale,
)


class IdentityFabric:
    is_global_zero = True
    global_rank = 0

    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []

    @staticmethod
    def all_reduce(value: torch.Tensor, reduce_op: str) -> torch.Tensor:
        assert reduce_op in {"sum", "max"}
        return value

    @staticmethod
    def to_device(value):
        return value

    def log_dict(self, metrics: dict, step: int | None = None) -> None:
        self.logged.append((metrics, step))


def _diagnostics(value: torch.Tensor) -> dict[str, dict]:
    summary = summarize_tensor_scale(value)
    return {
        "rms": {"representation_rms": summary["rms"]},
        "max": {"representation_absmax": summary["absmax"]},
    }


def test_tensor_scale_summary_is_detached_and_exact() -> None:
    value = torch.tensor([3.0, -4.0], requires_grad=True)

    summary = summarize_tensor_scale(value)

    torch.testing.assert_close(summary["rms"]["sum_squares"], torch.tensor(25.0))
    assert summary["rms"]["count"].item() == 2
    torch.testing.assert_close(summary["absmax"], torch.tensor(4.0))
    assert not summary["rms"]["sum_squares"].requires_grad
    assert not summary["absmax"].requires_grad


def test_logit_scale_summary_reports_outliers_and_span() -> None:
    logits = torch.tensor([[-2.0, 1.0, 5.0], [-7.0, -3.0, -4.0]])

    summary = summarize_logit_scale(logits)

    torch.testing.assert_close(summary["absmax"], torch.tensor(7.0))
    torch.testing.assert_close(summary["span_max"], torch.tensor(7.0))


def test_callback_logs_batch_and_exact_epoch_statistics() -> None:
    fabric = IdentityFabric()
    trainer = SimpleNamespace(fabric=fabric, state={"global_step": 12})
    callback = LogModelDiagnosticsCallback(log_every_n=2)
    callback.on_train_epoch_start(trainer)

    callback.on_train_batch_end(
        trainer,
        outputs={"model_diagnostics": _diagnostics(torch.tensor([3.0, 4.0]))},
        batch=None,
        batch_idx=0,
    )
    callback.on_train_batch_end(
        trainer,
        outputs={"model_diagnostics": _diagnostics(torch.tensor([0.0, 12.0]))},
        batch=None,
        batch_idx=1,
    )
    callback.on_train_epoch_end(trainer)

    assert len(fabric.logged) == 2
    batch_metrics, batch_step = fabric.logged[0]
    epoch_metrics, epoch_step = fabric.logged[1]
    torch.testing.assert_close(
        batch_metrics["train/activation_scale/batch/representation_rms"],
        torch.tensor(12.5).sqrt(),
    )
    torch.testing.assert_close(
        batch_metrics["train/activation_scale/batch/representation_absmax"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        epoch_metrics["train/activation_scale/epoch/representation_rms"],
        torch.tensor(42.25).sqrt(),
    )
    torch.testing.assert_close(
        epoch_metrics["train/activation_scale/epoch/representation_absmax"],
        torch.tensor(12.0),
    )
    assert batch_step == epoch_step == 12


def test_callback_combines_uneven_distributed_counts_exactly() -> None:
    class UnevenDistributedFabric(IdentityFabric):
        def __init__(self) -> None:
            super().__init__()
            self.remote_sums = iter([75.0, 3])
            self.remote_maxima = iter([10.0])

        def all_reduce(self, value: torch.Tensor, reduce_op: str) -> torch.Tensor:
            remote_values = (
                self.remote_sums if reduce_op == "sum" else self.remote_maxima
            )
            remote = torch.tensor(next(remote_values), dtype=value.dtype)
            if reduce_op == "sum":
                return value + remote
            return torch.maximum(value, remote)

    fabric = UnevenDistributedFabric()
    trainer = SimpleNamespace(fabric=fabric, state={"global_step": 0})
    callback = LogModelDiagnosticsCallback()

    metrics = callback._reduce_and_format(
        trainer,
        _diagnostics(torch.tensor([3.0, 4.0])),
        scope="batch",
    )

    # Local: sum_sq=25, count=2. Remote: sum_sq=75, count=3.
    torch.testing.assert_close(
        metrics["train/activation_scale/batch/representation_rms"],
        torch.tensor(20.0).sqrt(),
    )
    torch.testing.assert_close(
        metrics["train/activation_scale/batch/representation_absmax"],
        torch.tensor(10.0),
    )


def test_callback_is_noop_when_model_diagnostics_are_disabled() -> None:
    fabric = IdentityFabric()
    trainer = SimpleNamespace(fabric=fabric, state={"global_step": 0})
    callback = LogModelDiagnosticsCallback()

    callback.on_train_batch_end(
        trainer,
        outputs={"loss_dict": {}},
        batch=None,
        batch_idx=0,
    )
    callback.on_train_epoch_end(trainer)

    assert fabric.logged == []


def test_epoch_diagnostics_survive_step_checkpoint_resume() -> None:
    fabric = IdentityFabric()
    state = {
        "global_step": 12,
        "batches_completed_in_epoch": 1,
        "callback_states": {},
    }
    trainer = SimpleNamespace(
        fabric=fabric,
        state=state,
        _gather_rank_states=lambda local_state: [local_state],
    )
    callback = LogModelDiagnosticsCallback(log_every_n=100)
    callback.on_train_epoch_start(trainer)
    callback.on_train_batch_end(
        trainer,
        outputs={"model_diagnostics": _diagnostics(torch.tensor([3.0, 4.0]))},
        batch=None,
        batch_idx=1,
    )
    callback.on_save_checkpoint(trainer, state)

    resumed = LogModelDiagnosticsCallback(log_every_n=100)
    resumed.on_fit_start(trainer)
    resumed.on_train_epoch_start(trainer)
    resumed.on_train_batch_end(
        trainer,
        outputs={"model_diagnostics": _diagnostics(torch.tensor([0.0, 12.0]))},
        batch=None,
        batch_idx=2,
    )
    resumed.on_train_epoch_end(trainer)

    epoch_metrics, epoch_step = fabric.logged[-1]
    torch.testing.assert_close(
        epoch_metrics["train/activation_scale/epoch/representation_rms"],
        torch.tensor(42.25).sqrt(),
    )
    assert epoch_step == 12
