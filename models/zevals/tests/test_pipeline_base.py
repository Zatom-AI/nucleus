"""Unit tests for `pipeline/base.py` — `EvalResults`, `EvalPipeline.report`,
and the `safe_update` graceful-degradation helper."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.pipeline.base import EvalResults, safe_update  # noqa: E402


def test_safe_update_merges_successful_result():
    agg: dict[str, float] = {"existing": 1.0}
    safe_update(agg, "widgets", lambda: {"widget_rate": 0.5})
    assert agg == {"existing": 1.0, "widget_rate": 0.5}


def test_safe_update_passes_args_and_kwargs_through():
    captured = {}

    def fn(a, b, *, c):
        captured["args"] = (a, b)
        captured["kwargs"] = {"c": c}
        return {"ok": 1.0}

    agg: dict[str, float] = {}
    safe_update(agg, "label", fn, "x", "y", c="z")
    assert captured == {"args": ("x", "y"), "kwargs": {"c": "z"}}
    assert agg == {"ok": 1.0}


def test_safe_update_swallows_exception_and_leaves_agg_untouched(
    caplog: pytest.LogCaptureFixture,
):
    """A failing metric function must not raise, and must not add any keys —
    this is the core graceful-degradation contract other pipelines rely on."""
    agg: dict[str, float] = {"existing": 1.0}

    def failing():
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        safe_update(agg, "widgets", failing)

    assert agg == {"existing": 1.0}
    assert any("widgets" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_safe_update_does_not_partially_merge_on_failure():
    """If `fn` raises partway through building its return dict, nothing from
    it should land in `agg` — the merge is all-or-nothing per call."""

    def failing():
        d = {"a": 1.0}
        d.update(
            {"b": 1.0 / 0}
        )  # raises ZeroDivisionError while constructing return value
        return d

    agg: dict[str, float] = {}
    safe_update(agg, "label", failing)
    assert agg == {}


def test_eval_results_defaults_are_independent_between_instances():
    """Regression guard: EvalResults uses dataclass field(default_factory=...),
    not mutable default args — two instances must not share state."""
    a = EvalResults()
    b = EvalResults()
    a.per_sample.append({"x": 1})
    a.aggregate["y"] = 1.0
    assert b.per_sample == []
    assert b.aggregate == {}


def test_report_formats_floats_and_non_floats_differently():
    from zevals.pipeline.base import EvalPipeline

    class ConcretePipeline(EvalPipeline):
        def run(self, samples):
            raise NotImplementedError

        def aggregate(self, per_sample):
            raise NotImplementedError

    results = EvalResults(aggregate={"float_metric": 0.5, "int_metric": 3})
    text = ConcretePipeline().report(results)
    assert "float_metric: 0.5000" in text
    assert "int_metric: 3" in text
