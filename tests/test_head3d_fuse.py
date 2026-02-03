import numpy as np

from head3D_fuse.fuse import fuse_3view_keypoints


def test_fuse_3view_keypoints_mean():
    keypoints = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    keypoints_by_view = {
        "front": keypoints,
        "left": keypoints * 2,
        "right": keypoints * 3,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view, method="mean"
    )

    assert np.allclose(fused[0], np.array([2.0, 4.0, 6.0]))
    assert np.isnan(fused[1]).all()
    assert fused_mask.tolist() == [True, False]
    assert n_valid.tolist() == [3, 0]


def test_fuse_3view_keypoints_first():
    keypoints_by_view = {
        "front": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "left": np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
        "right": np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view, method="first"
    )

    assert np.allclose(fused[0], np.array([2.0, 2.0, 2.0]))
    assert np.allclose(fused[1], np.array([1.0, 1.0, 1.0]))
    assert fused_mask.tolist() == [True, True]
    assert n_valid.tolist() == [2, 2]


def test_fuse_3view_keypoints_median_and_nan():
    keypoints_by_view = {
        "front": np.array([[1.0, 1.0, 1.0], [np.nan, 0.0, 0.0]]),
        "left": np.array([[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]),
        "right": np.array([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]]),
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view, method="median"
    )

    assert np.allclose(fused[0], np.array([3.0, 3.0, 3.0]))
    assert np.allclose(fused[1], np.array([2.0, 2.0, 2.0]))
    assert fused_mask.tolist() == [True, True]
    assert n_valid.tolist() == [3, 1]


def test_fuse_3view_keypoints_with_extrinsic_alignment():
    keypoints_by_view = {
        "front": np.array([[1.0, 1.0, 1.0]]),
        "left": np.array([[2.0, 1.0, 1.0]]),
        "right": np.array([[0.0, 1.0, 1.0]]),
    }
    view_transforms = {
        "front": {"R": np.eye(3), "t_wc": np.zeros(3)},
        "left": {"R": np.eye(3), "t_wc": np.array([1.0, 0.0, 0.0])},
        "right": {"R": np.eye(3), "t_wc": np.array([-1.0, 0.0, 0.0])},
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        view_transforms=view_transforms,
        transform_mode="world_to_camera",
    )

    assert np.allclose(fused[0], np.array([1.0, 1.0, 1.0]))
    assert fused_mask.tolist() == [True]
    assert n_valid.tolist() == [3]


def test_fuse_3view_keypoints_with_procrustes_alignment():
    ref = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector = np.array([2.0, -1.0, 0.0])
    transformed = (ref @ rotation_matrix) + translation_vector
    keypoints_by_view = {
        "front": ref,
        "left": transformed,
        "right": transformed,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes",
        alignment_reference="front",
        alignment_scale=False,
    )

    assert np.allclose(fused, ref, atol=1e-6)
    assert fused_mask.tolist() == [True, True, True]
    assert n_valid.tolist() == [3, 3, 3]


def test_fuse_3view_keypoints_with_trimmed_alignment():
    ref = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector = np.array([2.0, -1.0, 0.0])
    transformed = (ref @ rotation_matrix) + translation_vector
    transformed[0] = np.array([10.0, 10.0, 0.0])  # inject a strong outlier
    keypoints_by_view = {
        "front": ref,
        "left": transformed,
        "right": transformed,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes_trimmed",
        alignment_reference="front",
        alignment_scale=False,
        alignment_trim_ratio=0.2,
        alignment_max_iters=5,
    )

    assert np.allclose(fused, ref, atol=1e-6)
    assert fused_mask.tolist() == [True, True, True, True, True]
    assert n_valid.tolist() == [3, 3, 3, 3, 3]


def test_fuse_3view_keypoints_trimmed_alignment_fallback():
    ref = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    keypoints_by_view = {
        "front": ref,
        "left": ref + 1.0,
        "right": ref - 1.0,
    }

    fused_base, fused_mask_base, n_valid_base = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
    )

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes_trimmed",
        alignment_reference="front",
        alignment_scale=False,
        alignment_trim_ratio=0.99,
        alignment_max_iters=5,
    )

    assert np.allclose(fused, fused_base)
    assert fused_mask.tolist() == [True, True, True]
    assert n_valid.tolist() == [3, 3, 3]


def test_fuse_3view_keypoints_trimmed_alignment_moderate():
    ref = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector = np.array([2.0, -1.0, 0.0])
    transformed = (ref @ rotation_matrix) + translation_vector
    transformed[0] = np.array([8.0, 8.0, 0.0])  # inject a moderate outlier
    keypoints_by_view = {
        "front": ref,
        "left": transformed,
        "right": transformed,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes_trimmed",
        alignment_reference="front",
        alignment_scale=False,
        alignment_trim_ratio=0.5,
        alignment_max_iters=5,
    )

    assert np.allclose(fused, ref, atol=1e-6)
    assert fused_mask.tolist() == [True, True, True, True, True]
    assert n_valid.tolist() == [3, 3, 3, 3, 3]
