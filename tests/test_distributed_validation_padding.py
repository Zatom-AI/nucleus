from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler

from nucleus.trainers.fabric import FabricTrainer


class _Model:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class _RecordingFabric:
    world_size = 2

    def __init__(self) -> None:
        self.calls = []

    def call(self, hook_name: str, **kwargs: Any) -> None:
        self.calls.append((hook_name, kwargs))


class _ValidationTrainer(FabricTrainer):
    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        del batch, batch_idx, is_accumulating

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        del val_loader_name
        self.validation_batches.append((batch_idx, batch))
        return {"batch_idx": batch_idx}


def _trainer_without_initialization() -> _ValidationTrainer:
    trainer = object.__new__(_ValidationTrainer)
    trainer.fabric = _RecordingFabric()
    trainer.state = {"model": _Model()}
    trainer.should_stop = False
    trainer.validation_batches = []
    return trainer


def test_live_validation_shape_identifies_twelve_padding_slots() -> None:
    """The 500-example, 16-rank job marks precisely its final 12 slots."""
    dataset = list(range(500))
    padding_ranks = []

    for rank in range(16):
        sampler = DistributedSampler(
            dataset,
            num_replicas=16,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(dataset, sampler=sampler, batch_size=1)
        assert not FabricTrainer._is_distributed_validation_padding(
            loader,
            batch_idx=30,
        )
        if FabricTrainer._is_distributed_validation_padding(loader, batch_idx=31):
            padding_ranks.append(rank)

    assert padding_ranks == list(range(4, 16))


def test_validation_forwards_padding_but_marks_callback_hooks() -> None:
    """Padding keeps DDP forwards aligned without entering metric aggregates."""
    dataset = list(range(3))
    sampler = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=1)
    trainer = _trainer_without_initialization()

    trainer.validation_loop(val_loaders={"validation": loader})

    assert [batch_idx for batch_idx, _ in trainer.validation_batches] == [0, 1]
    batch_end_calls = [
        kwargs
        for hook_name, kwargs in trainer.fabric.calls
        if hook_name == "on_validation_batch_end"
    ]
    assert [call["is_distributed_padding"] for call in batch_end_calls] == [
        False,
        True,
    ]
    assert trainer.state["model"].training is True


def test_validation_output_callbacks_ignore_distributed_padding(tmp_path) -> None:
    """All repository callbacks that persist or aggregate outputs skip fillers."""
    from nucleus.callbacks.dump_validation_structures import (
        DumpValidationStructuresCallback,
    )
    from nucleus.callbacks.metrics_logging import (
        LogGroupedZevalsStructurePredictionMetricsCallback,
        LogMoleculeAFDValidationMetricsCallback,
        StoreValidationMetricsInDFCallback,
    )

    store_callback = StoreValidationMetricsInDFCallback(save_dir=tmp_path)
    store_callback.on_validation_epoch_start(trainer=None)
    store_callback.on_validation_batch_end(
        trainer=None,
        outputs={},
        batch=None,
        batch_idx=0,
        num_batches=1,
        is_distributed_padding=True,
    )
    assert store_callback.per_gpu_outputs_df.empty

    afd_callback = LogMoleculeAFDValidationMetricsCallback(enabled=True)
    afd_callback.on_validation_epoch_start(trainer=None)
    afd_callback.on_validation_batch_end(
        trainer=None,
        outputs={},
        batch=None,
        batch_idx=0,
        num_batches=1,
        is_distributed_padding=True,
    )
    assert not afd_callback._records_by_dataset
    assert not afd_callback._total_samples_by_dataset

    grouped_callback = LogGroupedZevalsStructurePredictionMetricsCallback(enabled=True)
    grouped_callback.on_validation_batch_end(
        trainer=None,
        outputs={},
        batch=None,
        batch_idx=0,
        num_batches=1,
        is_distributed_padding=True,
    )
    assert not grouped_callback._records_by_dataset

    dump_callback = object.__new__(DumpValidationStructuresCallback)
    dump_callback.on_validation_batch_end(
        trainer=SimpleNamespace(),
        outputs={},
        batch=None,
        dataset_name="validation",
        is_distributed_padding=True,
    )
