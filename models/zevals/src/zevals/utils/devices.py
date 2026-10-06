"""GPU device assignment for oracle models.

zevals never queries `torch.distributed` (or any other distributed-training
framework's rank API) — see `oracles/base.py`'s execution-model docstring
for the analogous in-process/subprocess split this complements. Callers
decide when to invoke evaluation (e.g. `if is_main_process():
suite.evaluate(...)`); this module only decides which *physical GPU* each
oracle model lands on within that one process.
"""

from __future__ import annotations

import itertools

import torch

_counter = itertools.count()


def resolve_device(device: str | None, reserve_gpu0: bool = True) -> str:
    """Resolve an oracle's `device` config value to a concrete device string.

    An explicit value (`"cuda:2"`, `"cpu"`, ...) passes through unchanged —
    full manual control is always available. `None` (the default) or
    `"auto"` round-robins across visible GPUs, skipping index 0 (assumed to
    hold the model being trained) whenever more than one GPU is visible;
    falls back to GPU 0 if it's the only one, or `"cpu"` if none are
    visible.

    Respects `CUDA_VISIBLE_DEVICES` automatically (torch remaps visible
    devices to local indices), so this composes correctly with the common
    one-GPU-per-rank distributed training setup — round-robin degrades to a
    no-op there since only one device is ever visible to this function in
    the first place.

    Args:
        device: An explicit device string, `"auto"`, or `None`.
        reserve_gpu0: If True (default), skip GPU 0 for round-robin
            assignment whenever another GPU is available.

    Returns:
        A concrete device string suitable for `torch.device(...)`.
    """
    if device not in (None, "auto"):
        return device
    if not torch.cuda.is_available():
        return "cpu"

    num_gpus = torch.cuda.device_count()
    candidates = [i for i in range(num_gpus) if not (reserve_gpu0 and i == 0)]
    if not candidates:
        candidates = list(range(num_gpus))

    idx = candidates[next(_counter) % len(candidates)]
    return f"cuda:{idx}"
