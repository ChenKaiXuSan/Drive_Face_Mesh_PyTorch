#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Tests for FusedViewComparator"""

import numpy as np
import pytest

from head3D_fuse.fuse.compare_fused import FusedViewComparator


def _make_sequence(values):
    values = np.asarray(values, dtype=float)
    return np.stack([values, np.zeros_like(values), np.zeros_like(values)], axis=1)[
        :, None, :
    ]


def test_compute_metrics_basic():
    fused = _make_sequence([0.0, 1.0, 0.0, 1.0])
    view_front = _make_sequence([0.0, 1.0, 0.0, 1.0])
    view_left = _make_sequence([1.0, 2.0, 1.0, 2.0])

    comparator = FusedViewComparator(
        fused, {"front": view_front, "left": view_left}
    )

    metrics = comparator.compute_metrics()

    assert np.isclose(metrics["mean_distance_to_views"]["front"], 0.0)
    assert np.isclose(metrics["mean_distance_to_views"]["left"], 1.0)
    assert np.isclose(
        metrics["view_consistency"]["front_vs_left"]["mean"], 1.0
    )
    assert np.isclose(metrics["fused_jitter"]["mean"], 2.0)
    assert np.isclose(metrics["view_jitters"]["front"]["mean"], 2.0)
    assert np.isclose(metrics["view_jitters"]["left"]["mean"], 2.0)


def test_compute_euclidean_distances_subset():
    fused = np.zeros((3, 2, 3), dtype=float)
    view_front = np.zeros((3, 2, 3), dtype=float)
    view_left = np.ones((3, 2, 3), dtype=float)

    comparator = FusedViewComparator(
        fused, {"front": view_front, "left": view_left}
    )

    distances = comparator.compute_euclidean_distances(keypoint_indices=[1])

    assert distances["front"].shape == (3, 1)
    assert distances["left"].shape == (3, 1)
    assert np.allclose(distances["front"], 0.0)
    assert np.allclose(distances["left"], np.sqrt(3.0))


def test_validate_data_shape_mismatch():
    fused = np.zeros((2, 1, 3), dtype=float)
    view_front = np.zeros((3, 1, 3), dtype=float)

    with pytest.raises(ValueError):
        FusedViewComparator(fused, {"front": view_front})
