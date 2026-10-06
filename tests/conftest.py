"""Test fixtures and utilities for nucleus tests."""

import os

os.environ["DEBUG"] = "false"
os.environ["TRITON_INTERPRET"] = "1"  # Support unit testing on MPS systems

from foundry.testing import configure_pytest, get_test_data_dir  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Configure pytest for nucleus tests."""
    configure_pytest(config, __file__)


import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_release_paths(monkeypatch):
    """Keep tests independent of the caller's dataset/checkpoint environment."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.setenv("NUCLEUS_DATA_PATH", str(root / "data"))
    for key in ("QM9_DATA_DIR", "MP20_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
