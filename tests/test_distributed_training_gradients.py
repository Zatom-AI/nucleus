"""Distributed regression tests for training gradient normalization."""

import multiprocessing
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nucleus.trainers.fabric import FabricTrainer

_RANK_BATCHES = {
    0: (
        torch.tensor([[1.0]]),
        torch.tensor([[2.0], [3.0]]),
        torch.tensor([[4.0]]),
    ),
    1: (
        torch.tensor([[5.0], [6.0], [7.0]]),
        torch.tensor([[8.0]]),
        torch.tensor([[9.0], [10.0]]),
    ),
}


class _DistributedFabric:
    """Minimal Fabric-compatible adapter backed by a real Gloo process group."""

    def __init__(self, rank: int, world_size: int) -> None:
        self.device = torch.device("cpu")
        self.global_rank = rank
        self.world_size = world_size
        self.is_global_zero = rank == 0

    @staticmethod
    def call(*args, **kwargs) -> None:
        del args, kwargs

    @staticmethod
    def backward(loss: torch.Tensor, **kwargs) -> None:
        loss.backward(**kwargs)

    @staticmethod
    def no_backward_sync(model, *, enabled: bool):
        return model.no_sync() if enabled else nullcontext()

    @staticmethod
    def all_reduce(value: torch.Tensor, *, reduce_op: str) -> torch.Tensor:
        assert reduce_op == "sum"
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value


class _DistributedLinearTrainer(FabricTrainer):
    """Small trainer that exercises the production accumulation loop with DDP."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        grad_accum_steps: int,
    ) -> None:
        # Constructing Lightning Fabric would attempt to launch processes. This
        # test already owns the Gloo workers, so install only the state used by
        # FabricTrainer.train_loop().
        self.fabric = _DistributedFabric(rank, world_size)
        self.grad_accum_steps = grad_accum_steps
        self.should_stop = False
        self.checkpoint_every_n_steps = None
        self.limit_train_batches = float("inf")

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.initialize_or_update_trainer_state(
            {
                "model": model,
                "optimizer": optimizer,
                "train_cfg": {},
            }
        )

    def training_step(self, batch, batch_idx: int, is_accumulating: bool) -> None:
        del batch_idx
        inputs = batch[0]
        with self.fabric.no_backward_sync(self.state["model"], enabled=is_accumulating):
            predictions = self.state["model"](inputs)
            loss = torch.mean((predictions - 2 * inputs) ** 2)
            self._backward_training_loss(loss)
        self._current_train_return = {"loss": loss.detach()}

    def validation_step(self, batch, batch_idx: int, val_loader_name=None) -> dict:
        del batch, batch_idx, val_loader_name
        raise NotImplementedError

    def step_optimizer(self) -> None:
        optimizer = self.state["optimizer"]
        optimizer.step()
        optimizer.zero_grad()


def _run_distributed_training_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue,
) -> None:
    """Run no-accumulation and accumulated cases in one process group."""
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        results = {}
        for grad_accum_steps in (1, 2):
            trainer = _DistributedLinearTrainer(
                rank=rank,
                world_size=world_size,
                grad_accum_steps=grad_accum_steps,
            )
            local_loader = [(batch,) for batch in _RANK_BATCHES[rank]]
            assert trainer.train_loop(train_loader=local_loader)
            results[grad_accum_steps] = {
                "weight": float(trainer.state["model"].module.weight.item()),
                "global_step": trainer.state["global_step"],
            }
            dist.barrier()
        result_queue.put(("ok", rank, results))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _reference_weight(grad_accum_steps: int) -> tuple[float, int]:
    """Train on each optimizer window's exact global union of examples."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_microbatches = len(_RANK_BATCHES[0])
    global_step = 0
    for window_start in range(0, num_microbatches, grad_accum_steps):
        window_end = min(window_start + grad_accum_steps, num_microbatches)
        inputs = torch.cat(
            [
                _RANK_BATCHES[rank][batch_idx]
                for batch_idx in range(window_start, window_end)
                for rank in range(2)
            ]
        )
        loss = torch.mean((model(inputs) - 2 * inputs) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    return float(model.weight.item()), global_step


@pytest.mark.skipif(sys.platform == "win32", reason="Gloo file rendezvous differs")
def test_real_ddp_matches_global_batches_with_and_without_accumulation(
    tmp_path: Path,
) -> None:
    """DDP/no_sync must match global mean batches, including a partial window."""
    world_size = 2
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "ddp-gradient-rendezvous")
    workers = [
        context.Process(
            target=_run_distributed_training_worker,
            args=(rank, world_size, init_file, result_queue),
        )
        for rank in range(world_size)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=90)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)

    messages = [result_queue.get(timeout=10) for _ in range(world_size)]
    errors = [payload for status, _, payload in messages if status == "error"]
    assert not errors, "\n".join(errors)
    assert all(worker.exitcode == 0 for worker in workers)

    rank_results = [payload for status, _, payload in messages if status == "ok"]
    assert len(rank_results) == world_size
    for grad_accum_steps in (1, 2):
        reference_weight, reference_steps = _reference_weight(grad_accum_steps)
        for results in rank_results:
            assert results[grad_accum_steps]["global_step"] == reference_steps
            assert results[grad_accum_steps]["weight"] == pytest.approx(
                reference_weight,
                rel=0,
                abs=1e-6,
            )
