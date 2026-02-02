import numpy as np

from head3D_fuse.fuse import fuse_3view_keypoints


def test_fuse_3view_keypoints_mean():
    kpt = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    keypoints_by_view = {
        "front": kpt,
        "left": kpt * 2,
        "right": kpt * 3,
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
