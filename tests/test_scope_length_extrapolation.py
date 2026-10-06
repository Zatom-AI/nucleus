"""Checks for extrapolation cohort integrity and length-stratified contrasts."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[1] / "models/zevals/src"))
from evaluate_scope_length_extrapolation import (
    audit_lengths,
    bootstrap,
    cluster_metrics,
)


def test_requires_every_prespecified_length_and_stress_replicate():
    rows = [{"length": length} for length in range(129, 257) for _ in range(8)]
    rows += [{"length": length} for length in [384, 512] for _ in range(64)]
    audit_lengths(rows)
    with pytest.raises(ValueError):
        audit_lengths(rows[:-1])
    rows[-1] = {"length": 128}
    with pytest.raises(ValueError):
        audit_lengths(rows)


def test_bootstrap_keeps_paired_endpoints_and_fixed_length_weights():
    values = np.array([[0.0, 0.2], [0.0, 0.2], [1.0, 0.8], [1.0, 0.8]])
    draws = bootstrap(values, np.array([129, 129, 256, 256]), seed=5, draws=100)
    np.testing.assert_allclose(draws, np.full((100, 2), 0.5))


def test_complete_linkage_rejects_single_linkage_bridge():
    matrix = np.array([[1.0, 0.8, 0.4], [0.8, 1.0, 0.7], [0.4, 0.7, 1.0]])
    result = cluster_metrics(matrix, np.array([True, True, True]))
    assert result["distinct_designable_clusters"] == 2
    assert 0 <= result["diversity_auc"] <= 1
    assert (
        cluster_metrics(matrix, np.array([False, False, False]))[
            "distinct_designable_clusters"
        ]
        == 0
    )
    assert (
        cluster_metrics(matrix, np.array([True, False, False]))[
            "distinct_designable_clusters"
        ]
        == 1
    )
