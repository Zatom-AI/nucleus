"""Tests for the per-example and HBPLUS wall-clock bounds in the data path.

A single example that never returns stalls its dataloader worker, therefore its
DDP rank, and finally the whole job when the remaining ranks time out inside a
collective.  These tests pin the two bounds that keep one bad structure from
doing that.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from atomworks.ml.datasets.concat_dataset import FallbackDatasetWrapper
from biotite.structure import AtomArray
from rfd3na.transforms import hbonds_hbplus, na_geom_utils
from torch.utils.data import Dataset

from nucleus.transforms.hbplus import (
    SafeCalculateHbondsPlus,
    patch_rfd3na_hbplus_subprocess_timeout,
)
from nucleus.utils.datasets import (
    ExampleLoadTimeoutError,
    TimeBoundedDataset,
    patch_atomworks_cif_cache_atomic_writes,
)


class _SleepingDataset(Dataset):
    """Dataset whose first index sleeps far longer than any real example."""

    def __init__(self, slow_index: int = 0, sleep_seconds: float = 30.0) -> None:
        self.slow_index = slow_index
        self.sleep_seconds = sleep_seconds
        self.loaded: list[int] = []

    def __len__(self) -> int:
        return 4

    def idx_to_id(self, index: int) -> str:
        return f"example-{index}"

    def __getitem__(self, index: int):
        if index == self.slow_index:
            time.sleep(self.sleep_seconds)
        self.loaded.append(index)
        return {"index": index}


def test_time_bounded_dataset_interrupts_an_over_budget_example():
    """An example past its budget raises instead of blocking the worker."""
    dataset = TimeBoundedDataset(_SleepingDataset(), timeout_seconds=0.25)

    started = time.monotonic()
    with pytest.raises(ExampleLoadTimeoutError, match="example-0"):
        dataset[0]

    assert time.monotonic() - started < 5.0


def test_time_bounded_dataset_leaves_fast_examples_untouched():
    """Examples inside the budget are returned unchanged, with no timer left."""
    dataset = TimeBoundedDataset(_SleepingDataset(), timeout_seconds=5.0)

    assert dataset[1] == {"index": 1}

    import signal

    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_time_bounded_dataset_falls_back_to_another_example():
    """The existing fallback wrapper retries a stalled example's neighbour."""
    inner = _SleepingDataset()
    bounded = TimeBoundedDataset(inner, timeout_seconds=0.25)
    wrapper = FallbackDatasetWrapper(bounded, fallback_dataset=bounded)

    assert wrapper[(0, 2)] == {"index": 2}
    assert inner.loaded == [2]


def test_time_bounded_dataset_survives_a_swallowed_deadline():
    """Library code that catches broad exceptions must not absorb the bound.

    W&B's cleanup finalizer wraps its blocking call in
    ``contextlib.suppress(Exception)``, which ate the first deadline in
    production.
    """

    class _SwallowingDataset(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            with contextlib.suppress(Exception):  # absorbs the first deadline
                time.sleep(20)
            time.sleep(20)  # and the load goes on waiting afterwards
            return {"index": index}

    dataset = TimeBoundedDataset(
        _SwallowingDataset(), timeout_seconds=0.25, retry_seconds=0.25
    )

    started = time.monotonic()
    with pytest.raises(ExampleLoadTimeoutError):
        dataset[0]

    assert time.monotonic() - started < 10.0


def test_time_bounded_dataset_delegates_dataset_attributes():
    """Sampler-facing dataset attributes remain reachable through the wrapper."""
    inner = _SleepingDataset()
    bounded = TimeBoundedDataset(inner, timeout_seconds=1.0)

    assert len(bounded) == len(inner)
    assert bounded.idx_to_id(3) == "example-3"


def test_time_bounded_dataset_rejects_a_non_positive_budget():
    """A misconfigured budget must fail at construction, not at load time."""
    with pytest.raises(ValueError, match="must be positive"):
        TimeBoundedDataset(_SleepingDataset(), timeout_seconds=0)


def test_cif_cache_entries_survive_a_write_read_round_trip(tmp_path):
    """Cache entries must stay readable: pandas infers gzip from the suffix."""
    import pandas as pd
    from atomworks.io import parser

    patch_atomworks_cif_cache_atomic_writes()
    target = tmp_path / "1abc_assembly_1.pkl.gz"

    parser.pd.to_pickle({"asym_unit": [1, 2, 3]}, target)

    assert pd.read_pickle(target) == {"asym_unit": [1, 2, 3]}
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_cif_cache_writes_are_published_atomically(tmp_path):
    """Concurrent ranks must never observe a half-written cache entry."""
    from atomworks.io import parser

    patch_atomworks_cif_cache_atomic_writes()
    target = tmp_path / "entry.pkl.gz"
    observed: list[list[str]] = []

    class _Recording:
        def to_pickle(self, obj, path, *args, **kwargs):
            observed.append(sorted(p.name for p in tmp_path.iterdir()))
            Path(path).write_bytes(b"payload")

    original, parser.pd._pandas = parser.pd._pandas, _Recording()
    try:
        parser.pd.to_pickle({"asym_unit": 1}, target)
    finally:
        parser.pd._pandas = original

    assert target.read_bytes() == b"payload"
    assert observed == [[]]  # the final name only appears after the rename
    assert [p.name for p in tmp_path.iterdir()] == ["entry.pkl.gz"]


def test_hbplus_patch_bounds_the_external_process(monkeypatch):
    """A hanging HBPLUS process is killed rather than waited on forever."""
    monkeypatch.setenv("NUCLEUS_HBPLUS_TIMEOUT_SECONDS", "0.5")
    patch_rfd3na_hbplus_subprocess_timeout()

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        hbonds_hbplus.subprocess.call(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    assert time.monotonic() - started < 10.0


def test_hbplus_patch_applies_to_the_geometry_helper(monkeypatch):
    """Nucleic-acid geometry features use a second HBPLUS call site."""
    patch_rfd3na_hbplus_subprocess_timeout()

    assert na_geom_utils._nucleus_hbplus_timeout_patched
    assert na_geom_utils.subprocess.DEVNULL == subprocess.DEVNULL


def test_hbplus_patch_closes_stdin(monkeypatch):
    """HBPLUS must never be able to block prompting on an inherited stdin."""
    monkeypatch.setenv("NUCLEUS_HBPLUS_TIMEOUT_SECONDS", "10")
    patch_rfd3na_hbplus_subprocess_timeout()

    status = hbonds_hbplus.subprocess.call(
        [sys.executable, "-c", "import sys; sys.exit(1 if sys.stdin.read() else 0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    assert status == 0


def test_safe_hbplus_reports_a_timeout_as_unavailable(monkeypatch):
    """A timed-out HBPLUS degrades the crop's features instead of failing it."""
    atom_array = AtomArray(1)
    atom_array.atom_name = np.full(1, "CA")
    atom_array.res_name = np.full(1, "ALA")
    atom_array.chain_id = np.full(1, "A")
    atom_array.res_id = np.arange(1, 2)
    atom_array.coord = np.zeros((1, 3), dtype=np.float32)
    atom_array.set_annotation("chain_iid", np.asarray(["chain_0"]))
    monkeypatch.setenv("HBPLUS_PATH", sys.executable)

    def timing_out_forward(self, data):
        raise subprocess.TimeoutExpired(cmd="hbplus", timeout=120)

    monkeypatch.setattr(
        "rfd3na.transforms.hbonds_hbplus.CalculateHbondsPlus.forward",
        timing_out_forward,
    )
    data = SafeCalculateHbondsPlus()({"atom_array": atom_array})

    assert data["log_dict"] == {
        "hbplus_available": 0,
        "hbplus_unavailable_timeout": 1,
    }
