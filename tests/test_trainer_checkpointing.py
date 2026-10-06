import multiprocessing
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from foundry.utils.weights import CheckpointConfig
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Subset, TensorDataset, WeightedRandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from worker_rng_dataset import WorkerRandomDataset

from nucleus.trainers.checkpointing import (
    checkpoint_completes_epoch,
    format_checkpoint_name,
    get_checkpoint_kind,
    get_checkpoint_path,
    get_latest_checkpoint,
    prune_checkpoints,
)
from nucleus.trainers.fabric import FabricTrainer
from nucleus.utils.datasets import StatefulSampler, assemble_distributed_loader


class _RecordingFabric:
    """A mock Fabric object that records calls to its hooks and saves."""

    def __init__(self) -> None:
        self.saved_paths = []
        self.calls = []
        self.logged = []
        self.is_global_zero = True

    def call(self, hook_name: str, **kwargs) -> None:
        """Record a call to a hook."""
        self.calls.append((hook_name, kwargs))

    def save(self, checkpoint_path: Path, state: dict) -> None:
        """Simulate saving a checkpoint."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("checkpoint")
        self.saved_paths.append(checkpoint_path)

    def log_dict(self, metrics: dict, step: int) -> None:
        """Record metrics and their optimizer-step coordinate."""
        self.logged.append((metrics, step))


class _ConcreteTrainer(FabricTrainer):
    """A concrete subclass of FabricTrainer for testing purposes."""

    def training_step(self, batch, batch_idx: int, is_accumulating: bool) -> None:
        """Simulate a training step."""
        raise NotImplementedError

    def validation_step(
        self,
        batch,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        """Simulate a validation step."""
        raise NotImplementedError


class _LinearTrainer(FabricTrainer):
    """Tiny real-Fabric trainer for interruption/resume integration tests."""

    def __init__(
        self,
        output_dir: Path,
        *,
        checkpoint_every_n_steps: int | None,
        checkpoint_every_n_epochs: int | None = None,
        stop_after_step_checkpoint: bool = False,
        grad_accum_steps: int = 1,
        max_epochs: int = 1,
        allow_world_size_change_on_resume: bool = False,
    ) -> None:
        super().__init__(
            accelerator="cpu",
            devices_per_node=1,
            precision="32-true",
            max_epochs=max_epochs,
            grad_accum_steps=grad_accum_steps,
            n_examples_per_epoch=4,
            output_dir=output_dir,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            allow_world_size_change_on_resume=allow_world_size_change_on_resume,
        )
        self.stop_after_step_checkpoint = stop_after_step_checkpoint
        self.training_calls: list[tuple[int, int]] = []

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        self.initialize_or_update_trainer_state(
            {
                "model": model,
                "optimizer": optimizer,
                "train_cfg": {},
            }
        )

    def training_step(self, batch, batch_idx: int, is_accumulating: bool) -> None:
        """Fit a deterministic scalar linear model and record data order."""
        inputs = batch[0]
        self.training_calls.extend((int(value), batch_idx) for value in inputs[:, 0])
        predictions = self.state["model"](inputs)
        loss = torch.mean((predictions - 2 * inputs) ** 2)
        self._backward_training_loss(loss)
        self._current_train_return = {"loss": loss.detach()}

    def validation_step(
        self,
        batch,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        """Validation is unused in these tests."""
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_kind="epoch") -> None:
        """Optionally emulate a job ending immediately after a step checkpoint."""
        super().save_checkpoint(checkpoint_kind)
        if checkpoint_kind == "step" and self.stop_after_step_checkpoint:
            self.should_stop = True


class _RNGLinearTrainer(_LinearTrainer):
    """Linear trainer whose model-side loss consumes all process RNG streams."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng_draws: list[tuple[float, float, float]] = []

    def training_step(self, batch, batch_idx: int, is_accumulating: bool) -> None:
        """Fit using Python, NumPy, and torch draws and record them exactly."""
        inputs = batch[0]
        self.training_calls.extend((int(value), batch_idx) for value in inputs[:, 0])

        python_draw = random.random()
        numpy_draw = float(np.random.random())
        torch_draw = torch.rand(())
        self.rng_draws.append((python_draw, numpy_draw, float(torch_draw)))

        predictions = self.state["model"](inputs)
        target_scale = 2 + python_draw + numpy_draw + torch_draw
        loss = torch.mean((predictions - target_scale * inputs) ** 2)
        self._backward_training_loss(loss)
        self._current_train_return = {"loss": loss.detach()}


class _ValidationLinearTrainer(_LinearTrainer):
    """Linear trainer that records validation calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.validation_calls = 0

    def validation_step(
        self,
        batch,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        """Record a cheap validation batch."""
        del batch, batch_idx, val_loader_name
        self.validation_calls += 1
        return {}


def _configure_exponential_scheduler(trainer, interval=None) -> None:
    """Construct a real scheduler through the trainer's configuration path."""
    model_config = {
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.ExponentialLR",
            "gamma": 0.998,
        }
    }
    if interval is not None:
        model_config["lr_scheduler_interval"] = interval
    trainer.state["train_cfg"] = OmegaConf.create({"model": model_config})
    trainer.construct_scheduler()


@pytest.mark.parametrize("interval", [None, "step", "epoch"])
def test_scheduler_uses_configured_interval(interval) -> None:
    """Existing schedules default to optimizer steps; epoch schedules opt in."""
    trainer = object.__new__(_ConcreteTrainer)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    trainer.initialize_or_update_trainer_state({"optimizer": optimizer})
    _configure_exponential_scheduler(trainer, interval)

    expected_interval = interval or "step"
    assert trainer.state["scheduler_cfg"].interval == expected_interval
    optimizer.step()
    other_interval = "epoch" if expected_interval == "step" else "step"
    trainer.step_scheduler(level=other_interval, current_value=0)
    assert optimizer.param_groups[0]["lr"] == 0.05
    trainer.step_scheduler(level=expected_interval, current_value=0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05 * 0.998)


def test_scheduler_rejects_invalid_interval() -> None:
    """A misspelled interval must fail instead of silently disabling decay."""
    trainer = object.__new__(_ConcreteTrainer)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    trainer.initialize_or_update_trainer_state({"optimizer": optimizer})
    with pytest.raises(ValueError, match="lr_scheduler_interval"):
        _configure_exponential_scheduler(trainer, "batch")


def _linear_loader() -> StatefulDataLoader:
    """Return four deterministic one-example batches."""
    inputs = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    return StatefulDataLoader(TensorDataset(inputs), batch_size=1, shuffle=False)


def _linear_loader_with_batch_size(
    *, num_examples: int, batch_size: int
) -> StatefulDataLoader:
    """Return deterministic scalar inputs with the requested batching."""
    inputs = torch.arange(num_examples, dtype=torch.float32).reshape(-1, 1)
    return StatefulDataLoader(
        TensorDataset(inputs), batch_size=batch_size, shuffle=False
    )


def _linear_loader_with_batch_sizes(
    batch_sizes: tuple[int, ...],
) -> StatefulDataLoader:
    """Return deterministic inputs grouped into explicitly sized batches."""
    num_examples = sum(batch_sizes)
    inputs = torch.arange(num_examples, dtype=torch.float32).reshape(-1, 1)
    batch_sampler = []
    start = 0
    for batch_size in batch_sizes:
        batch_sampler.append(list(range(start, start + batch_size)))
        start += batch_size
    return StatefulDataLoader(
        TensorDataset(inputs),
        batch_sampler=batch_sampler,
    )


def _weighted_linear_loader() -> StatefulDataLoader:
    """Return a weighted loader with a dedicated sampler generator."""
    inputs = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    return _weighted_loader(TensorDataset(inputs))


def _linear_weight(trainer: _LinearTrainer) -> torch.Tensor:
    """Return the unwrapped scalar model weight."""
    model = getattr(trainer.state["model"], "module", trainer.state["model"])
    return model.weight.detach().clone()


@pytest.mark.parametrize("grad_accum_steps", [0, -1, True, 1.5])
def test_gradient_accumulation_steps_must_be_a_positive_integer(
    tmp_path: Path,
    grad_accum_steps,
) -> None:
    """Reject invalid accumulation windows before entering the train loop."""
    with pytest.raises(ValueError, match="positive integer"):
        _LinearTrainer(
            tmp_path,
            checkpoint_every_n_steps=None,
            grad_accum_steps=grad_accum_steps,
        )


def _seed_test_rngs(seed: int) -> None:
    """Seed every process RNG exercised by the checkpoint integration tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _set_checkpoint_rng_world_size(checkpoint_path: Path, world_size: int) -> None:
    """Rewrite the recorded RNG world size to emulate a topology change."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["rng_state"]["world_size"] = world_size
    torch.save(checkpoint, checkpoint_path)


class _AccessRecordingDataset(Dataset):
    """Map dataset that records which indices reach ``__getitem__``."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.accesses: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        self.accesses.append(index)
        return index


def _weighted_loader(dataset: Dataset) -> StatefulDataLoader:
    """Return a loader whose sampler has its own advancing generator."""
    generator = torch.Generator().manual_seed(1234)
    sampler = WeightedRandomSampler(
        torch.ones(len(dataset)),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return StatefulDataLoader(
        dataset,
        batch_size=1,
        sampler=StatefulSampler(sampler),
    )


def _multiprocess_random_loader() -> StatefulDataLoader:
    """Return a persistent two-worker loader with worker-RNG checkpointing."""
    multiprocessing_context = None
    # macOS defaults to spawn, which spends seconds rebuilding this test process
    # in every worker. Forkserver avoids that overhead without unsafely forking
    # the multithreaded pytest process; other platforms retain their defaults.
    if (
        sys.platform == "darwin"
        and "forkserver" in multiprocessing.get_all_start_methods()
    ):
        multiprocessing_context = "forkserver"
    return assemble_distributed_loader(
        dataset=Subset(WorkerRandomDataset(), range(4)),
        sampler=None,
        rank=0,
        world_size=1,
        n_examples_per_epoch=6,
        loader_cfg={
            "dataloader_params": {
                "batch_size": 1,
                "num_workers": 2,
                "persistent_workers": True,
                "multiprocessing_context": multiprocessing_context,
            }
        },
        stateful=True,
    )


def _touch(path: Path, mtime_ns: int) -> Path:
    """Create a file at the given path and set its modification time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.name)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_checkpoint_names_encode_epoch_or_global_step(tmp_path: Path) -> None:
    """Test that checkpoint names are formatted correctly based on epoch or global step."""
    assert (
        format_checkpoint_name("epoch", current_epoch=7, global_step=123)
        == "epoch-0007.ckpt"
    )
    assert (
        format_checkpoint_name("step", current_epoch=7, global_step=123)
        == "step-00000123.ckpt"
    )
    assert get_checkpoint_path(
        tmp_path,
        "step",
        current_epoch=7,
        global_step=123,
    ) == (tmp_path / "ckpt" / "step-00000123.ckpt")
    assert get_checkpoint_kind("epoch-0007.ckpt") == "epoch"
    assert get_checkpoint_kind("step-00000123.ckpt") == "step"
    assert checkpoint_completes_epoch("epoch-0007.ckpt")
    assert not checkpoint_completes_epoch("step-00000123.ckpt")


def test_latest_checkpoint_searches_recursively_and_keeps_step_checkpoints(
    tmp_path: Path,
) -> None:
    """Test that the latest checkpoint is found recursively and that step checkpoints are preferred over epoch checkpoints."""
    _touch(tmp_path / "ckpt" / "epoch-0003.ckpt", mtime_ns=1_000_000)
    step_checkpoint = _touch(
        tmp_path / "ckpt" / "step-00000042.ckpt",
        mtime_ns=2_000_000,
    )
    _touch(tmp_path / "other" / "not-a-checkpoint.txt", mtime_ns=3_000_000)

    assert get_latest_checkpoint(tmp_path) == step_checkpoint
    assert get_latest_checkpoint(tmp_path / "missing") is None


def test_prune_checkpoints_retains_newest_and_ignores_unknown_files(
    tmp_path: Path,
) -> None:
    """Retention must be recursive and limited to trainer-named checkpoints."""
    oldest = _touch(
        tmp_path / "run-a" / "ckpt" / "step-00000001.ckpt",
        mtime_ns=1_000_000,
    )
    middle = _touch(
        tmp_path / "run-b" / "ckpt" / "epoch-0001.ckpt",
        mtime_ns=2_000_000,
    )
    newest = _touch(
        tmp_path / "run-b" / "ckpt" / "step-00000002.ckpt",
        mtime_ns=3_000_000,
    )
    unknown = _touch(
        tmp_path / "run-a" / "ckpt" / "manual.ckpt",
        mtime_ns=500_000,
    )

    deleted = prune_checkpoints(tmp_path, keep_last_n=2)

    assert deleted == [oldest]
    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()
    assert unknown.exists()


def test_validation_loader_intervals_select_only_due_loaders() -> None:
    """Expensive validation loaders may run less often than cheap loaders."""
    trainer = object.__new__(_ConcreteTrainer)
    trainer.state = {"current_epoch": 3}
    trainer.validate_first_epoch = False
    trainer.validate_every_n_epochs = 1
    trainer.validate_loader_every_n_epochs = {"generation": 4}
    loaders = {"generation": object(), "mlip": object()}

    assert trainer._scheduled_validation_loaders(loaders) == {"mlip": loaders["mlip"]}

    trainer.state["current_epoch"] = 4
    assert trainer._scheduled_validation_loaders(loaders) == loaders


def test_validation_loader_intervals_ignore_disabled_null_entries() -> None:
    """Null intervals disable inherited validation-loader schedules."""
    trainer = _ConcreteTrainer(
        accelerator="cpu",
        devices_per_node=1,
        precision="32-true",
        validate_loader_every_n_epochs={
            "omat24_generation": 5,
            "omol25_generation": None,
            "omat24_mlip": None,
        },
    )

    assert trainer.validate_loader_every_n_epochs == {"omat24_generation": 5}


def test_epoch_progress_is_logged_on_global_step_axis() -> None:
    """Epoch must be a value, never a replacement for the W&B step coordinate."""
    trainer = object.__new__(_ConcreteTrainer)
    trainer.fabric = _RecordingFabric()
    trainer.state = {"current_epoch": 7, "global_step": 29_310}

    trainer._log_epoch_progress()

    assert trainer.fabric.logged == [({"trainer/epoch": 7}, 29_310)]


def test_fabric_save_checkpoint_uses_requested_checkpoint_kind(
    tmp_path: Path,
) -> None:
    """Test that the FabricTrainer saves checkpoints with the requested kind (epoch or step)."""
    trainer = object.__new__(_ConcreteTrainer)
    trainer.output_dir = tmp_path
    trainer.fabric = _RecordingFabric()
    trainer.state = {
        "current_epoch": 5,
        "global_step": 12,
    }

    trainer.save_checkpoint("epoch")
    trainer.save_checkpoint("step")

    assert trainer.fabric.saved_paths == [
        tmp_path / "ckpt" / "epoch-0005.ckpt",
        tmp_path / "ckpt" / "step-00000012.ckpt",
    ]


def test_step_checkpoint_resumes_at_next_batch(tmp_path: Path) -> None:
    """An interrupted run should match uninterrupted data order and parameters."""
    uninterrupted = _LinearTrainer(
        tmp_path / "uninterrupted",
        checkpoint_every_n_steps=None,
    )
    uninterrupted.fit(_linear_loader())

    interrupted = _LinearTrainer(
        tmp_path / "interrupted",
        checkpoint_every_n_steps=2,
        stop_after_step_checkpoint=True,
    )
    interrupted.fit(_linear_loader())
    checkpoint_path = tmp_path / "interrupted" / "ckpt" / "step-00000002.ckpt"

    resumed = _LinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
    )
    resumed.fit(
        _linear_loader(),
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert interrupted.training_calls + resumed.training_calls == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    torch.testing.assert_close(
        _linear_weight(resumed),
        _linear_weight(uninterrupted),
        rtol=0,
        atol=0,
    )


def test_step_checkpoint_restores_model_side_rng_exactly(tmp_path: Path) -> None:
    """A resumed model should continue every process RNG stream bit-for-bit."""
    _seed_test_rngs(42)
    uninterrupted = _RNGLinearTrainer(
        tmp_path / "uninterrupted",
        checkpoint_every_n_steps=None,
    )
    uninterrupted.fit(_linear_loader())

    _seed_test_rngs(42)
    interrupted = _RNGLinearTrainer(
        tmp_path / "interrupted",
        checkpoint_every_n_steps=2,
        stop_after_step_checkpoint=True,
    )
    interrupted.fit(_linear_loader())
    checkpoint_path = tmp_path / "interrupted" / "ckpt" / "step-00000002.ckpt"

    # Deliberately perturb every stream before constructing the resumed trainer.
    _seed_test_rngs(999)
    resumed = _RNGLinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
    )
    resumed.fit(
        _linear_loader(),
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert interrupted.rng_draws + resumed.rng_draws == uninterrupted.rng_draws
    torch.testing.assert_close(
        _linear_weight(resumed),
        _linear_weight(uninterrupted),
        rtol=0,
        atol=0,
    )


def test_step_checkpoint_resumes_on_gradient_accumulation_boundary(
    tmp_path: Path,
) -> None:
    """A checkpoint after two accumulated batches should resume without replay."""
    uninterrupted = _LinearTrainer(
        tmp_path / "uninterrupted",
        checkpoint_every_n_steps=None,
        grad_accum_steps=2,
    )
    uninterrupted.fit(_linear_loader())

    interrupted = _LinearTrainer(
        tmp_path / "interrupted",
        checkpoint_every_n_steps=1,
        stop_after_step_checkpoint=True,
        grad_accum_steps=2,
    )
    interrupted.fit(_linear_loader())
    checkpoint_path = tmp_path / "interrupted" / "ckpt" / "step-00000001.ckpt"

    resumed = _LinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
        grad_accum_steps=2,
    )
    resumed.fit(
        _linear_loader(),
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert interrupted.training_calls == [(0, 0), (1, 1)]
    assert resumed.training_calls == [(2, 2), (3, 3)]
    assert uninterrupted.state["global_step"] == resumed.state["global_step"] == 2
    torch.testing.assert_close(
        _linear_weight(resumed),
        _linear_weight(uninterrupted),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_examples", [3, 4])
def test_gradient_accumulation_matches_larger_mean_batches(
    tmp_path: Path,
    num_examples: int,
) -> None:
    """Full and partial accumulation windows should average microbatch losses."""
    reference = _LinearTrainer(
        tmp_path / f"reference-{num_examples}",
        checkpoint_every_n_steps=None,
        grad_accum_steps=1,
    )
    reference.fit(
        _linear_loader_with_batch_size(num_examples=num_examples, batch_size=2)
    )

    accumulated = _LinearTrainer(
        tmp_path / f"accumulated-{num_examples}",
        checkpoint_every_n_steps=None,
        grad_accum_steps=2,
    )
    accumulated.fit(
        _linear_loader_with_batch_size(num_examples=num_examples, batch_size=1)
    )

    assert accumulated.state["global_step"] == reference.state["global_step"]
    torch.testing.assert_close(
        _linear_weight(accumulated),
        _linear_weight(reference),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("reference_batch_sizes", "accumulated_batch_sizes"),
    [
        ((4,), (1, 3)),
        ((4, 2), (1, 3, 2)),
    ],
)
def test_gradient_accumulation_weights_variable_microbatches_by_example(
    tmp_path: Path,
    reference_batch_sizes: tuple[int, ...],
    accumulated_batch_sizes: tuple[int, ...],
) -> None:
    """Uneven microbatches, including a partial window, match larger batches."""
    reference = _LinearTrainer(
        tmp_path / "reference-variable",
        checkpoint_every_n_steps=None,
        grad_accum_steps=1,
    )
    reference.fit(_linear_loader_with_batch_sizes(reference_batch_sizes))

    accumulated = _LinearTrainer(
        tmp_path / "accumulated-variable",
        checkpoint_every_n_steps=None,
        grad_accum_steps=2,
    )
    accumulated.fit(_linear_loader_with_batch_sizes(accumulated_batch_sizes))

    assert accumulated.state["global_step"] == reference.state["global_step"]
    torch.testing.assert_close(
        _linear_weight(accumulated),
        _linear_weight(reference),
        rtol=0,
        atol=1e-7,
    )


def test_gradient_normalization_uses_global_ddp_example_count() -> None:
    """DDP's rank mean is converted to a true global example mean."""

    class _UnevenCountFabric:
        world_size = 2

        @staticmethod
        def all_reduce(value: torch.Tensor, reduce_op: str) -> torch.Tensor:
            assert reduce_op == "sum"
            # This rank contributes one example; the other rank contributes three.
            return value + 3

    parameter = torch.nn.Parameter(torch.zeros(()))
    # DDP averages the two rank-local weighted gradient sums: (2 + 18) / 2.
    parameter.grad = torch.tensor(10.0)
    trainer = object.__new__(_ConcreteTrainer)
    trainer.fabric = _UnevenCountFabric()
    trainer.state = {"model": torch.nn.ParameterList([parameter])}

    trainer._normalize_accumulated_gradients(local_num_examples=1)

    # The desired global example mean is (2 + 18) / (1 + 3).
    torch.testing.assert_close(parameter.grad, torch.tensor(5.0))


def test_epoch_checkpoint_restores_next_epoch_rng_and_sampler_state(
    tmp_path: Path,
) -> None:
    """Epoch resume should preserve model RNG and a dedicated sampler stream."""
    _seed_test_rngs(42)
    uninterrupted = _RNGLinearTrainer(
        tmp_path / "uninterrupted",
        checkpoint_every_n_steps=None,
        checkpoint_every_n_epochs=1,
        max_epochs=2,
    )
    uninterrupted.fit(_weighted_linear_loader())

    _seed_test_rngs(42)
    first_epoch = _RNGLinearTrainer(
        tmp_path / "first_epoch",
        checkpoint_every_n_steps=None,
        checkpoint_every_n_epochs=1,
    )
    first_epoch.fit(_weighted_linear_loader())

    _seed_test_rngs(999)
    resumed = _RNGLinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
        checkpoint_every_n_epochs=1,
        max_epochs=2,
    )
    resumed.fit(
        _weighted_linear_loader(),
        ckpt_config=CheckpointConfig(
            path=tmp_path / "first_epoch" / "ckpt" / "epoch-0000.ckpt"
        ),
    )

    assert first_epoch.training_calls + resumed.training_calls == (
        uninterrupted.training_calls
    )
    assert first_epoch.rng_draws + resumed.rng_draws == uninterrupted.rng_draws
    torch.testing.assert_close(
        _linear_weight(resumed),
        _linear_weight(uninterrupted),
        rtol=0,
        atol=0,
    )


def test_epoch_checkpoint_rejects_world_size_change_by_default(
    tmp_path: Path,
) -> None:
    """Rank-local epoch RNG state must remain strict unless explicitly relaxed."""
    first_epoch = _LinearTrainer(
        tmp_path / "first",
        checkpoint_every_n_steps=None,
        checkpoint_every_n_epochs=1,
    )
    first_epoch.fit(_linear_loader())
    checkpoint_path = tmp_path / "first" / "ckpt" / "epoch-0000.ckpt"
    _set_checkpoint_rng_world_size(checkpoint_path, world_size=2)

    resumed = _LinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
        max_epochs=2,
    )
    with pytest.raises(
        ValueError,
        match="Cannot restore rank-local RNG state with a different world size",
    ):
        resumed.fit(
            _linear_loader(),
            ckpt_config=CheckpointConfig(path=checkpoint_path),
        )


def test_epoch_checkpoint_allows_opt_in_world_size_change(tmp_path: Path) -> None:
    """An opted-in epoch resume must drop incompatible rank-local RNG state."""
    first_epoch = _LinearTrainer(
        tmp_path / "first",
        checkpoint_every_n_steps=None,
        checkpoint_every_n_epochs=1,
    )
    first_epoch.fit(_linear_loader())
    checkpoint_path = tmp_path / "first" / "ckpt" / "epoch-0000.ckpt"
    _set_checkpoint_rng_world_size(checkpoint_path, world_size=2)

    resumed = _LinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
        max_epochs=2,
        allow_world_size_change_on_resume=True,
    )
    resumed.fit(
        _linear_loader(),
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert resumed.state["current_epoch"] == 2
    assert resumed.state["global_step"] == 8
    assert resumed._loaded_rng_state is None


def test_world_size_change_option_rejects_step_checkpoint(tmp_path: Path) -> None:
    """Mid-epoch dataloader state cannot cross physical world sizes."""
    interrupted = _LinearTrainer(
        tmp_path / "interrupted",
        checkpoint_every_n_steps=2,
        stop_after_step_checkpoint=True,
    )
    interrupted.fit(_linear_loader())
    checkpoint_path = tmp_path / "interrupted" / "ckpt" / "step-00000002.ckpt"
    _set_checkpoint_rng_world_size(checkpoint_path, world_size=2)

    resumed = _LinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
        allow_world_size_change_on_resume=True,
    )
    with pytest.raises(
        ValueError,
        match="allow_world_size_change_on_resume only supports epoch checkpoints",
    ):
        resumed.fit(
            _linear_loader(),
            ckpt_config=CheckpointConfig(path=checkpoint_path),
        )


def test_pre_validation_checkpoint_prevents_training_and_validation_replay(
    tmp_path: Path,
) -> None:
    """An interruption during validation must resume at the next epoch."""
    first_run = _ValidationLinearTrainer(
        tmp_path / "first",
        checkpoint_every_n_steps=None,
    )
    first_run.checkpoint_before_validation = True
    first_run.fit(_linear_loader(), val_loaders={"cheap": _linear_loader()})
    checkpoint_path = tmp_path / "first" / "ckpt" / "epoch-0000.ckpt"

    resumed = _ValidationLinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
    )
    resumed.checkpoint_before_validation = True
    resumed.fit(
        _linear_loader(),
        val_loaders={"cheap": _linear_loader()},
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert checkpoint_path.exists()
    assert first_run.validation_calls == 4
    assert resumed.training_calls == []
    assert resumed.validation_calls == 0


def test_epoch_scheduler_resume_after_interrupted_validation_matches_uninterrupted(
    monkeypatch, tmp_path: Path
) -> None:
    """Pre-validation checkpoints must preserve completed epoch LR decay."""
    uninterrupted = _ValidationLinearTrainer(
        tmp_path / "uninterrupted",
        checkpoint_every_n_steps=None,
        max_epochs=3,
    )
    _configure_exponential_scheduler(uninterrupted, "epoch")
    uninterrupted.fit(_linear_loader(), val_loaders={"cheap": _linear_loader()})

    interrupted = _ValidationLinearTrainer(
        tmp_path / "interrupted",
        checkpoint_every_n_steps=None,
        max_epochs=3,
    )
    _configure_exponential_scheduler(interrupted, "epoch")
    interrupted.checkpoint_before_validation = True

    def interrupt_validation(*args, **kwargs):
        raise RuntimeError("validation interrupted")

    monkeypatch.setattr(interrupted, "validation_step", interrupt_validation)
    with pytest.raises(RuntimeError, match="validation interrupted"):
        interrupted.fit(_linear_loader(), val_loaders={"cheap": _linear_loader()})

    checkpoint_path = tmp_path / "interrupted" / "ckpt" / "epoch-0000.ckpt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["scheduler_cfg"]["scheduler"]["last_epoch"] == 1
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(
        0.05 * 0.998
    )

    resumed = _ValidationLinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
        max_epochs=3,
    )
    _configure_exponential_scheduler(resumed, "epoch")
    resumed.fit(
        _linear_loader(),
        val_loaders={"cheap": _linear_loader()},
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert interrupted.training_calls + resumed.training_calls == (
        uninterrupted.training_calls
    )
    assert resumed.state["optimizer"].param_groups[0]["lr"] == pytest.approx(
        0.05 * 0.998**3
    )
    assert resumed.state["scheduler_cfg"].state_dict() == (
        uninterrupted.state["scheduler_cfg"].state_dict()
    )
    torch.testing.assert_close(
        _linear_weight(resumed),
        _linear_weight(uninterrupted),
        rtol=0,
        atol=0,
    )


def test_mid_epoch_stop_skips_epoch_checkpoint_and_bookkeeping(
    tmp_path: Path,
) -> None:
    """A graceful stop after a step checkpoint must leave the epoch unfinished."""
    trainer = _LinearTrainer(
        tmp_path,
        checkpoint_every_n_steps=2,
        stop_after_step_checkpoint=True,
    )
    trainer.checkpoint_every_n_epochs = 1

    trainer.fit(_linear_loader())

    assert trainer.training_calls == [(0, 0), (1, 1)]
    assert trainer.state["current_epoch"] == 0
    assert trainer.state["batches_completed_in_epoch"] == 2
    assert sorted(path.name for path in (tmp_path / "ckpt").glob("*.ckpt")) == [
        "step-00000002.ckpt"
    ]


def test_final_batch_step_checkpoint_does_not_replay_epoch(tmp_path: Path) -> None:
    """A step checkpoint on an epoch's final batch should resume with no replay."""
    interrupted = _LinearTrainer(
        tmp_path / "interrupted",
        checkpoint_every_n_steps=4,
        stop_after_step_checkpoint=True,
    )
    interrupted.fit(_linear_loader())
    checkpoint_path = tmp_path / "interrupted" / "ckpt" / "step-00000004.ckpt"

    resumed = _LinearTrainer(
        tmp_path / "resumed",
        checkpoint_every_n_steps=None,
    )
    resumed.fit(
        _linear_loader(),
        ckpt_config=CheckpointConfig(path=checkpoint_path),
    )

    assert interrupted.training_calls == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert resumed.training_calls == []
    assert resumed.state["global_step"] == 4


def test_stateful_sampler_restores_later_epoch_without_loading_prefix() -> None:
    """Sampler RNG/cursor restore should not fetch already-consumed examples."""
    original_dataset = _AccessRecordingDataset(8)
    original_loader = _weighted_loader(original_dataset)
    list(original_loader)  # Advance the dedicated sampler generator by one epoch.
    iterator = iter(original_loader)
    prefix = [int(next(iterator)) for _ in range(3)]
    state = original_loader.state_dict()
    expected_tail = [int(value) for value in iterator]
    expected_next_epoch = [int(value) for value in original_loader]

    torch.manual_seed(9999)
    resumed_dataset = _AccessRecordingDataset(8)
    resumed_loader = _weighted_loader(resumed_dataset)
    resumed_loader.load_state_dict(state)
    actual_tail = [int(value) for value in resumed_loader]
    actual_next_epoch = [int(value) for value in resumed_loader]

    assert len(prefix) == 3
    assert actual_tail == expected_tail
    assert actual_next_epoch == expected_next_epoch
    assert resumed_dataset.accesses == [*expected_tail, *expected_next_epoch]


def test_multiprocess_loader_restores_worker_rng_state() -> None:
    """Resumed stochastic transforms should match the uninterrupted worker stream."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = _multiprocess_random_loader()
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected_tail = list(iterator)
    expected_next_epoch = list(loader)
    del iterator, loader

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_loader = _multiprocess_random_loader()
    resumed_loader.load_state_dict(state)
    actual_tail = list(resumed_loader)
    actual_next_epoch = list(resumed_loader)

    assert actual_tail == expected_tail
    assert actual_next_epoch == expected_next_epoch
