#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Temporal smoothing and optimization for 3D keypoints sequences.

This module provides functions to optimize 3D keypoints along the temporal
dimension using various smoothing and filtering techniques.

Author: Kaixu Chen
Last Modified: February 2026
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


class TemporalKeypointOptimizer:
    """Optimize 3D keypoints along temporal dimension."""

    def __init__(self, method: str = "gaussian", **kwargs):
        """
        Initialize temporal optimizer.

        Args:
            method: Smoothing method. Options:
                - "gaussian": Gaussian smoothing
                - "savgol": Savitzky-Golay filter
                - "kalman": Kalman filter
                - "bilateral": Bilateral filter (preserves edges)
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.params = kwargs

    def optimize(
        self,
        keypoints: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        **method_kwargs,
    ) -> np.ndarray:
        """
        Optimize temporal sequence of 3D keypoints.

        Args:
            keypoints: (T, N, 3) array of 3D keypoints where:
                - T: number of frames
                - N: number of keypoints
                - 3: x, y, z coordinates
            visibility: (T, N) boolean array indicating valid keypoints
                If None, assumes all keypoints are valid.
            **method_kwargs: Override parameters from __init__

        Returns:
            (T, N, 3) array of smoothed 3D keypoints
        """
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError(
                f"Expected keypoints shape (T, N, 3), got {keypoints.shape}"
            )

        T, N, _ = keypoints.shape

        # Merge parameters
        params = {**self.params, **method_kwargs}

        if self.method == "gaussian":
            return self._gaussian_smooth(keypoints, visibility, **params)
        elif self.method == "savgol":
            return self._savgol_smooth(keypoints, visibility, **params)
        elif self.method == "kalman":
            return self._kalman_smooth(keypoints, visibility, **params)
        elif self.method == "bilateral":
            return self._bilateral_smooth(keypoints, visibility, **params)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _gaussian_smooth(
        self,
        keypoints: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Smooth keypoints using Gaussian filter.

        Args:
            keypoints: (T, N, 3) keypoints array
            visibility: (T, N) optional validity mask
            sigma: standard deviation of Gaussian kernel
        """
        T, N, _ = keypoints.shape
        smoothed = np.zeros_like(keypoints)

        for n in range(N):
            for d in range(3):  # xyz dimensions
                seq = keypoints[:, n, d]

                if visibility is not None:
                    # Only smooth valid points
                    mask = visibility[:, n]
                    if mask.sum() < 2:
                        smoothed[:, n, d] = seq
                        continue
                    seq = seq.copy()
                    seq[~mask] = np.nan

                # Apply 1D Gaussian filter
                smoothed[:, n, d] = gaussian_filter1d(
                    seq, sigma=sigma, mode="nearest"
                )

        return smoothed

    def _savgol_smooth(
        self,
        keypoints: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        window_length: int = 5,
        polyorder: int = 2,
    ) -> np.ndarray:
        """
        Smooth keypoints using Savitzky-Golay filter.

        Args:
            keypoints: (T, N, 3) keypoints array
            visibility: (T, N) optional validity mask
            window_length: length of the filter window (must be odd)
            polyorder: order of polynomial fit
        """
        T, N, _ = keypoints.shape

        # Ensure window_length is odd and valid
        window_length = min(window_length, T)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(3, window_length)

        if polyorder >= window_length:
            polyorder = window_length - 1

        smoothed = np.zeros_like(keypoints)

        for n in range(N):
            for d in range(3):
                seq = keypoints[:, n, d]

                if visibility is not None:
                    mask = visibility[:, n]
                    if mask.sum() < window_length:
                        smoothed[:, n, d] = seq
                        continue

                try:
                    smoothed[:, n, d] = signal.savgol_filter(
                        seq, window_length=window_length, polyorder=polyorder
                    )
                except ValueError:
                    logger.warning(
                        f"Savgol filter failed for keypoint {n}, dim {d}; using original"
                    )
                    smoothed[:, n, d] = seq

        return smoothed

    def _kalman_smooth(
        self,
        keypoints: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
    ) -> np.ndarray:
        """
        Smooth keypoints using Kalman filter.

        Args:
            keypoints: (T, N, 3) keypoints array
            visibility: (T, N) optional validity mask
            process_variance: process noise variance
            measurement_variance: measurement noise variance
        """
        T, N, _ = keypoints.shape
        smoothed = np.zeros_like(keypoints)

        for n in range(N):
            for d in range(3):
                seq = keypoints[:, n, d].copy()
                if visibility is not None:
                    mask = visibility[:, n]
                    if mask.sum() < 2:
                        smoothed[:, n, d] = seq
                        continue
                else:
                    mask = np.ones(T, dtype=bool)

                # Forward pass (filtering)
                x_hat = np.zeros(T)
                p = np.zeros(T)
                x_hat[0] = seq[0]
                p[0] = 1.0

                for t in range(1, T):
                    if not mask[t]:
                        continue

                    # Predict
                    x_pred = x_hat[t - 1]
                    p_pred = p[t - 1] + process_variance

                    # Update
                    K = p_pred / (p_pred + measurement_variance)
                    x_hat[t] = x_pred + K * (seq[t] - x_pred)
                    p[t] = (1 - K) * p_pred

                # Backward pass (smoothing)
                smoothed_seq = x_hat.copy()
                for t in range(T - 2, -1, -1):
                    if mask[t]:
                        smoothed_seq[t] = (
                            x_hat[t]
                            + (p[t] / (p[t] + process_variance))
                            * (smoothed_seq[t + 1] - x_hat[t])
                        )

                smoothed[:, n, d] = smoothed_seq

        return smoothed

    def _bilateral_smooth(
        self,
        keypoints: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        sigma_space: float = 1.0,
        sigma_range: float = 0.1,
    ) -> np.ndarray:
        """
        Smooth keypoints using bilateral filter (edge-preserving).

        Args:
            keypoints: (T, N, 3) keypoints array
            visibility: (T, N) optional validity mask
            sigma_space: spatial (temporal) standard deviation
            sigma_range: range (value) standard deviation
        """
        T, N, _ = keypoints.shape
        smoothed = np.zeros_like(keypoints)

        # Bilateral filter window radius
        radius = int(np.ceil(3 * sigma_space))

        for n in range(N):
            for d in range(3):
                seq = keypoints[:, n, d]

                if visibility is not None:
                    mask = visibility[:, n]
                    if mask.sum() < 2:
                        smoothed[:, n, d] = seq
                        continue
                else:
                    mask = np.ones(T, dtype=bool)

                for t in range(T):
                    if not mask[t]:
                        smoothed[t, n, d] = seq[t]
                        continue

                    # Collect neighboring points
                    t_start = max(0, t - radius)
                    t_end = min(T, t + radius + 1)

                    neighbors = seq[t_start:t_end]
                    valid_neighbors = mask[t_start:t_end]

                    if valid_neighbors.sum() == 0:
                        smoothed[t, n, d] = seq[t]
                        continue

                    # Spatial weights (temporal distance)
                    spatial_dist = np.arange(t_start, t_end) - t
                    spatial_weights = np.exp(-(spatial_dist**2) / (2 * sigma_space**2))

                    # Range weights (value difference)
                    value_diff = np.abs(neighbors - seq[t])
                    range_weights = np.exp(-(value_diff**2) / (2 * sigma_range**2))

                    # Combined weights
                    weights = spatial_weights * range_weights
                    weights[~valid_neighbors] = 0

                    # Weighted average
                    if weights.sum() > 0:
                        smoothed[t, n, d] = np.sum(neighbors * weights) / weights.sum()
                    else:
                        smoothed[t, n, d] = seq[t]

        return smoothed


def smooth_keypoints_sequence(
    keypoints: np.ndarray,
    method: str = "gaussian",
    visibility: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to smooth 3D keypoints sequence.

    Args:
        keypoints: (T, N, 3) array of 3D keypoints
        method: Smoothing method ("gaussian", "savgol", "kalman", "bilateral")
        visibility: (T, N) optional boolean array of valid keypoints
        **kwargs: Method-specific parameters

    Returns:
        (T, N, 3) smoothed keypoints array

    Example:
        >>> # Load your temporal keypoints
        >>> kpts = np.random.randn(100, 17, 3)  # 100 frames, 17 keypoints
        >>> 
        >>> # Gaussian smoothing
        >>> smoothed = smooth_keypoints_sequence(
        ...     kpts, method="gaussian", sigma=2.0
        ... )
        >>> 
        >>> # Savitzky-Golay smoothing
        >>> smoothed = smooth_keypoints_sequence(
        ...     kpts, method="savgol", window_length=11, polyorder=3
        ... )
    """
    optimizer = TemporalKeypointOptimizer(method=method, **kwargs)
    return optimizer.optimize(keypoints, visibility=visibility)


def optimize_keypoints_with_constraints(
    keypoints: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    bone_constraints: Optional[Dict[Tuple[int, int], float]] = None,
    smoothness_weight: float = 1.0,
    constraint_weight: float = 1.0,
) -> np.ndarray:
    """
    Optimize keypoints with temporal smoothness and structural constraints.

    Args:
        keypoints: (T, N, 3) array of 3D keypoints
        visibility: (T, N) optional validity mask
        bone_constraints: Dict mapping (idx_i, idx_j) -> expected_bone_length
            Example: {(0, 1): 0.5, (1, 2): 0.4}
        smoothness_weight: Weight of temporal smoothness term
        constraint_weight: Weight of structural constraints

    Returns:
        (T, N, 3) optimized keypoints

    Example:
        >>> # Define bone lengths (known structure)
        >>> bone_constraints = {
        ...     (0, 1): 0.5,  # bone between keypoint 0 and 1 has length 0.5
        ...     (1, 2): 0.4,
        ... }
        >>> 
        >>> optimized = optimize_keypoints_with_constraints(
        ...     kpts,
        ...     bone_constraints=bone_constraints,
        ...     smoothness_weight=2.0,
        ...     constraint_weight=1.0
        ... )
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    T, N, _ = keypoints.shape

    if visibility is None:
        visibility = np.ones((T, N), dtype=bool)

    # Flatten optimization variable
    x0 = keypoints.reshape(-1)

    def objective(x):
        kpts = x.reshape(T, N, 3)
        loss = 0.0

        # Temporal smoothness: penalize large accelerations
        if T >= 3:
            # Second derivative (acceleration)
            accel = kpts[2:] - 2 * kpts[1:-1] + kpts[:-2]
            valid_accel = visibility[1:-1][:, :, None]
            loss += smoothness_weight * np.sum((accel * valid_accel) ** 2)

        # Measurement fit: how far from original observation
        diff = kpts - keypoints
        valid_diff = visibility[:, :, None]
        loss += np.sum((diff * valid_diff) ** 2)

        # Structural constraints: maintain bone lengths
        if bone_constraints:
            for (idx_i, idx_j), target_length in bone_constraints.items():
                bone_vecs = kpts[:, idx_j] - kpts[:, idx_i]
                bone_lengths = np.linalg.norm(bone_vecs, axis=-1)
                length_diff = bone_lengths - target_length
                valid_bones = visibility[:, idx_i] & visibility[:, idx_j]
                loss += constraint_weight * np.sum((length_diff[valid_bones]) ** 2)

        return loss

    # Optimize
    result = least_squares(
        lambda x: np.sqrt(objective(x) / len(x)),
        x0,
        max_nfev=100,
    )

    optimized = result.x.reshape(T, N, 3)
    return optimized


def estimate_velocity(
    keypoints: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate velocity from keypoint sequence.

    Args:
        keypoints: (T, N, 3) keypoints array
        visibility: (T, N) optional validity mask

    Returns:
        (T-1, N, 3) velocity array
    """
    T, N, _ = keypoints.shape
    velocity = keypoints[1:] - keypoints[:-1]

    if visibility is not None:
        # Zero out velocity where keypoints are invalid
        valid_vel = visibility[:-1] & visibility[1:]
        velocity[~valid_vel] = 0

    return velocity


def estimate_acceleration(
    keypoints: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate acceleration from keypoint sequence.

    Args:
        keypoints: (T, N, 3) keypoints array
        visibility: (T, N) optional validity mask

    Returns:
        (T-2, N, 3) acceleration array
    """
    if keypoints.shape[0] < 3:
        raise ValueError("At least 3 frames required to estimate acceleration")

    velocity = estimate_velocity(keypoints, visibility)
    acceleration = velocity[1:] - velocity[:-1]

    if visibility is not None:
        valid_accel = visibility[:-2] & visibility[1:-1] & visibility[2:]
        acceleration[~valid_accel] = 0

    return acceleration
