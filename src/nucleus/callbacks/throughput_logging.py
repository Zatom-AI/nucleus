"""Callbacks for logging training throughput."""

import time
from collections import deque
from typing import Any

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger

global_logger = RankedLogger(__name__, rank_zero_only=True)


def _batch_counts(batch: Any) -> tuple[int, int, int, int]:
    """Return (n_samples, n_tokens, n_atoms, diffusion_batch_size) for a batch.

    The training loop hands callbacks the *pre-collation* list of per-example
    dicts (``SparseCollate`` runs inside the training step), so sizes are counted
    from the examples themselves. A collated dict is also accepted, since the
    validation path and future refactors may hand one over instead.

    Returns zeros for any quantity that cannot be determined.
    """
    # ... pre-collation: a list of per-example dicts.
    if isinstance(batch, (list, tuple)):
        if not batch:
            return 0, 0, 0, 0
        n_tokens = 0
        n_atoms = 0
        for example in batch:
            gt = example.get("ground_truth") if hasattr(example, "get") else None
            if not isinstance(gt, dict):
                continue
            if (seq := gt.get("sequence_gt_I")) is not None:
                n_tokens += int(seq.shape[0])
            if (coord := gt.get("coord_atom_lvl")) is not None:
                n_atoms += int(coord.shape[0])
        # `t` carries a leading diffusion axis of size D.
        t = batch[0].get("t") if hasattr(batch[0], "get") else None
        d = int(t.shape[0]) if t is not None and getattr(t, "ndim", 0) >= 1 else 1
        return len(batch), n_tokens, n_atoms, d

    # ... post-collation: SparseCollate records the sizes directly.
    if isinstance(batch, dict):
        feats = batch.get("feats")
        if isinstance(feats, dict):
            n_samples = int(feats.get("batch_size", 0) or 0)
            d = int(feats.get("diffusion_batch_size", 1) or 1)
            expanded = int(feats.get("batch_num_tokens_expanded", 0) or 0)
            n_tokens = expanded // d if d else 0
            batch_atom_lvl = feats.get("batch_atom_lvl")
            n_atoms = int(batch_atom_lvl.shape[0]) if batch_atom_lvl is not None else 0
            return n_samples, n_tokens, n_atoms, d

    return 0, 0, 0, 0


class LogThroughputCallback(BaseCallback):
    """Log training throughput and batch composition to the active logger.

    Reports iterations/second, samples (examples)/second, and token/atom rates
    over a trailing window of batches, alongside the batch sizes behind them.

    Rates are measured between consecutive ``on_train_batch_end`` calls, so they
    reflect the full per-batch cost (data loading, forward, backward, and the
    optimizer step on accumulation boundaries) as the training loop sees it.

    Sizes are counted from the batch contents rather than from the configured
    ``batch_size``, so they stay correct on a short final batch and when the
    loader's fallback-retry logic drops examples.

    Args:
        log_every_n: Log once every this many training batches.
        window: Number of recent batches to average rates over. A trailing window
            keeps the reported rate responsive rather than letting slow startup
            batches dominate.
        enabled: Set False to disable without removing the callback.
    """

    def __init__(
        self, log_every_n: int = 25, window: int = 50, enabled: bool = True
    ) -> None:
        self.log_every_n = max(1, int(log_every_n))
        self.window = max(2, int(window))
        self.enabled = bool(enabled)
        self._dt: deque[float] = deque(maxlen=self.window)
        self._samples: deque[int] = deque(maxlen=self.window)
        self._tokens: deque[int] = deque(maxlen=self.window)
        self._atoms: deque[int] = deque(maxlen=self.window)
        self._last_t: float | None = None
        self._warned_no_counts = False

    def _reset(self) -> None:
        self._last_t = None
        self._dt.clear()
        self._samples.clear()
        self._tokens.clear()
        self._atoms.clear()

    def on_train_epoch_start(self, trainer: Any) -> None:
        # Validation and checkpointing sit between epochs; their cost must not be
        # attributed to the first training batch of the next epoch.
        self._reset()

    def on_train_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        n_samples, n_tokens, n_atoms, diffusion_bs = _batch_counts(batch)

        if n_samples == 0 and not self._warned_no_counts:
            self._warned_no_counts = True
            global_logger.warning(
                "LogThroughputCallback could not determine the batch size from a "
                f"{type(batch).__name__}; sample/token rates will be reported as 0. "
                "Timing metrics are unaffected."
            )

        # The first batch of an epoch has no predecessor to measure against.
        if self._last_t is not None:
            self._dt.append(now - self._last_t)
            self._samples.append(n_samples)
            self._tokens.append(n_tokens)
            self._atoms.append(n_atoms)
        self._last_t = now

        if (batch_idx + 1) % self.log_every_n or not self._dt:
            return
        if not trainer.fabric.is_global_zero:
            return

        elapsed = sum(self._dt)
        if elapsed <= 0.0:
            return

        n_batches = len(self._dt)
        it_per_s = n_batches / elapsed
        samples_per_s = sum(self._samples) / elapsed
        world_size = int(getattr(trainer.fabric, "world_size", 1) or 1)
        grad_accum = int(getattr(trainer, "grad_accum_steps", 1) or 1)

        metrics = {
            # Throughput as observed by this rank.
            "throughput/it_per_s": it_per_s,
            "throughput/s_per_it": elapsed / n_batches,
            "throughput/samples_per_s": samples_per_s,
            "throughput/tokens_per_s": sum(self._tokens) / elapsed,
            "throughput/atoms_per_s": sum(self._atoms) / elapsed,
            "throughput/optimizer_steps_per_s": it_per_s / grad_accum,
            # Summed across ranks, so DDP runs stay comparable to single-GPU ones.
            "throughput/samples_per_s_all_ranks": samples_per_s * world_size,
            # Batch composition behind the rates above.
            "batch/size": float(n_samples),
            "batch/diffusion_batch_size": float(diffusion_bs),
            "batch/num_tokens": float(n_tokens),
            "batch/num_tokens_expanded": float(n_tokens * diffusion_bs),
            "batch/grad_accum_steps": float(grad_accum),
            "batch/effective_size": float(n_samples * grad_accum * world_size),
        }
        trainer.fabric.log_dict(metrics, step=trainer.state["global_step"])
