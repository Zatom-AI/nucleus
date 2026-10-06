"""Unit tests for `utils/devices.py::resolve_device`."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.utils import devices  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_round_robin_counter():
    """`resolve_device`'s round-robin state is a module-level counter that
    persists across calls (by design, within one process) — reset it before
    each test so test order doesn't leak into results."""
    devices._counter = itertools.count()
    yield


def test_explicit_device_string_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    """An explicit device value is never touched, regardless of GPU availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert devices.resolve_device("cuda:0") == "cuda:0"
    assert devices.resolve_device("cuda:3") == "cuda:3"
    assert devices.resolve_device("cpu") == "cpu"


def test_no_gpu_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch):
    """None/"auto" resolve to "cpu" when no GPU is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert devices.resolve_device(None) == "cpu"
    assert devices.resolve_device("auto") == "cpu"


def test_single_gpu_uses_it_despite_reserve_gpu0(monkeypatch: pytest.MonkeyPatch):
    """Can't reserve away your only GPU — single-GPU visibility always uses it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    for _ in range(3):
        assert devices.resolve_device(None) == "cuda:0"


def test_multi_gpu_round_robins_skipping_gpu0(monkeypatch: pytest.MonkeyPatch):
    """With >1 GPU visible, None/"auto" round-robins across GPUs 1..N-1."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assigned = [devices.resolve_device(None) for _ in range(6)]
    assert assigned == ["cuda:1", "cuda:2", "cuda:3", "cuda:1", "cuda:2", "cuda:3"]
    # GPU 0 (assumed to hold the model being trained) is never auto-assigned.
    assert "cuda:0" not in assigned


def test_reserve_gpu0_false_includes_gpu0_in_rotation(monkeypatch: pytest.MonkeyPatch):
    """`reserve_gpu0=False` opts back into using GPU 0 in the round-robin."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assigned = [devices.resolve_device(None, reserve_gpu0=False) for _ in range(4)]
    assert assigned == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


def test_explicit_and_auto_calls_do_not_share_counter_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """Explicit device requests must not advance the round-robin counter —
    only unresolved (None/"auto") calls should."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert devices.resolve_device("cuda:0") == "cuda:0"
    assert devices.resolve_device(None) == "cuda:1"
    assert devices.resolve_device("cuda:2") == "cuda:2"
    assert devices.resolve_device(None) == "cuda:2"
