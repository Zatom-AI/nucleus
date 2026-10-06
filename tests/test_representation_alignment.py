"""Numerical checks for the manuscript's representation alignment metrics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "representation_alignment",
    Path(__file__).resolve().parents[1] / "scripts/zatom2_representation_alignment.py",
)
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)


def test_relative_coordinates_match_cosines_and_ignore_orthogonal_basis():
    values = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    anchors = values[[0, 2]]
    expected = np.array([[1.0, 0.0], [2**-0.5, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        alignment.relative_embeddings(values, anchors), expected, atol=1e-15
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        alignment.relative_embeddings(values @ rotation, anchors @ rotation),
        expected,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="nonzero"):
        alignment.relative_embeddings(np.zeros((1, 3)), anchors)


def test_alignment_invariant_to_translation_scale_and_rotation():
    values = np.array([[0.0, 1.0], [2.0, 3.0], [5.0, -2.0], [10.0, 4.0], [-7.0, 9.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    transformed = 3 * values @ rotation + 12
    np.testing.assert_allclose(
        alignment.alignment_scores(values, transformed, k=2),
        np.ones(3),
        atol=1e-14,
    )


def test_knn_is_jaccard_and_excludes_self():
    # A neighbors: [1, 0, 1, 2]; B neighbors: [2, 2, 0, 1].
    first = np.array([[0.0], [1.0], [3.0], [7.0]])
    second = np.array([[0.0], [3.0], [1.0], [7.0]])
    assert alignment.alignment_scores(first, second, k=1)[2] == 0
    # Every two-neighbor set intersects in one point: Jaccard=1/(2+2-1).
    third = np.array([[0.0], [7.0], [1.0], [3.0]])
    assert alignment.alignment_scores(first, third, k=2)[2] == pytest.approx(1 / 3)


def test_scores_reject_invalid_sample_pairing():
    with pytest.raises(ValueError, match="paired samples"):
        alignment.alignment_scores(np.ones((5, 3)), np.ones((4, 3)), k=1)
    with pytest.raises(ValueError, match="nonconstant"):
        alignment.alignment_scores(np.ones((5, 3)), np.ones((5, 3)), k=1)


def test_numerically_tied_neighbors_use_sample_order():
    # An element-only representation has multiple equivalent sites. A tiny
    # roundoff perturbation must not change which shared sample IDs break ties.
    first = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    second = first.copy()
    second[1, 1] = 1e-9
    assert alignment.alignment_scores(first, second, k=1)[2] == 1
