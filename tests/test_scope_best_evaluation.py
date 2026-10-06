"""Cohort accounting and sampling-unit checks for the SCOPe evaluation."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "summarize_scope_best",
    Path(__file__).resolve().parents[1] / "scripts/summarize_scope_best.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_bootstrap_preserves_fixed_length_mixture():
    # Variation is entirely between lengths, not between draws at each length.
    # An unstratified bootstrap would spuriously add length-mixture uncertainty.
    result = module.stratified_bootstrap(
        np.array([0, 0, 0, 1, 1, 1]), np.array([50, 50, 50, 128, 128, 128])
    )
    assert result == (0.5, 0.5, 0.5)


def test_bootstrap_rejects_missing_measurements():
    with pytest.raises(ValueError, match="Nonfinite|nonfinite"):
        module.stratified_bootstrap(np.array([0, np.nan]), np.array([50, 50]))


def test_zero_success_rate_retains_nonzero_upper_uncertainty():
    lo, hi = module.wilson_interval(0, 1027)
    assert lo == pytest.approx(0, abs=1e-15)
    assert 0.003 < hi < 0.004
    all_lo, all_hi = module.wilson_interval(1027, 1027)
    assert all_lo == pytest.approx(1 - hi)
    assert all_hi == pytest.approx(1)


@pytest.fixture
def cohort():
    name = "scope_len050_rep00_0_model_0"
    return {
        "per_backbone": [
            dict(sample=name, min_scRMSD=1.5, max_scTM=0.9, designable=False)
        ],
        "per_refold": [
            dict(sample=name, sequence_index=i, scRMSD=1.5 + i, scTM=0.9 - i * 0.01)
            for i in range(8)
        ],
    }


def test_audit_keeps_strict_threshold(cohort):
    rows = module.audit_cohort(cohort, {"scope_len050_rep00": {"length": 50}})
    assert rows[0]["length"] == 50
    assert not rows[0]["designable"]


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing_refold", "missing_tm", "wrong_best"]
)
def test_audit_refuses_incomplete_or_misaggregated_results(cohort, corruption):
    payload = copy.deepcopy(cohort)
    if corruption == "duplicate":
        payload["per_backbone"] *= 2
    elif corruption == "missing_refold":
        payload["per_refold"].pop()
    elif corruption == "missing_tm":
        payload["per_refold"][0]["scTM"] = None
    else:
        payload["per_backbone"][0]["min_scRMSD"] = 0.5
    with pytest.raises(ValueError):
        module.audit_cohort(payload, {"scope_len050_rep00": {"length": 50}})
