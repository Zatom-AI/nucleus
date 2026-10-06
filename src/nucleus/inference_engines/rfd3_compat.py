"""Compatibility helpers for loading `rfd3.engine` against the local nucleus base."""

from __future__ import annotations

import importlib

from nucleus.inference_engines.base import (
    BaseInferenceEngine as NucleusBaseInferenceEngine,
)


def load_rfd3_engine():
    """
    Import or reload `rfd3.engine` after swapping in the local nucleus base class.

    `rfd3.engine` binds `BaseInferenceEngine` at import time, so we patch the
    vendored `foundry` module first and then reload `rfd3.engine` to ensure
    `RFD3InferenceEngine` inherits from `nucleus.inference_engines.base.BaseInferenceEngine`.
    """
    foundry_base = importlib.import_module("foundry.inference_engines.base")
    foundry_base.BaseInferenceEngine = NucleusBaseInferenceEngine

    rfd3_engine = importlib.import_module("rfd3.engine")
    return importlib.reload(rfd3_engine)
