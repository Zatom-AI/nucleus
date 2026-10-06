"""Regression checks for the atom-level Appendix C.4 neighborhood diagnostic."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("plot_zatom2_element_embeddings")


def test_neighbors_exclude_entire_query_structure(analysis):
    # Duplicate atoms of one structure must never count as its own neighbors.
    values = np.array([[0.0], [0.0], [1.0], [3.0]])
    structures = np.array([0, 0, 1, 2])
    labels = np.array(["a", "a", "a", "b"])
    purity, chance = analysis.neighbor_scores(values, labels, structures, k=1)
    np.testing.assert_array_equal(purity, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(chance, [0.5, 2 / 3, 0.0], rtol=0, atol=1e-15)


def test_structure_weighting_does_not_favor_large_systems(analysis):
    unique, means = analysis.structure_means(
        np.array([1.0, 1.0, 1.0, 0.0]), np.array([5, 5, 5, 9])
    )
    np.testing.assert_array_equal(unique, [5, 9])
    assert means.mean() == 0.5


def test_requires_enough_external_neighbors(analysis):
    with pytest.raises(ValueError, match="Insufficient"):
        analysis.neighbor_scores(
            np.zeros((3, 2)), np.array(["a"] * 3), np.array([0, 0, 1]), k=2
        )


def test_paired_interval_preserves_constant_improvement(analysis):
    np.testing.assert_array_equal(
        analysis.paired_interval(np.full(20, 0.25)), [0.25, 0.25]
    )
