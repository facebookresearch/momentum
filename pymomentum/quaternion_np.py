# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Quaternion Utilities (NumPy)
============================

NumPy implementation of quaternion utilities for PyMomentum.

This module mirrors :mod:`quaternion` but uses NumPy arrays instead of PyTorch tensors.
See :mod:`quaternion` for full documentation on the quaternion format and operations.

Quaternion Format
-----------------
This module uses the (x, y, z, w) format where:

- **(x, y, z)**: Vector part representing the rotation axis scaled by sin(θ/2)
- **w**: Scalar part representing cos(θ/2), where θ is the rotation angle

The identity quaternion is (0, 0, 0, 1), representing no rotation.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# pyre-strict


def check(q: NDArray) -> None:
    """
    Check if an array represents a quaternion.

    :parameter q: An array representing a quaternion.
    """
    assert q.shape[-1] == 4, "Quaternion should have last dimension equal to 4."


def split(q: NDArray) -> tuple[NDArray, NDArray]:
    """
    Split a quaternion into its scalar and vector parts.

    :parameter q: An array representing a quaternion.
    :return: The scalar and vector parts of the quaternion.
    """
    check(q)
    return q[..., 3:4], q[..., 0:3]


def normalize(q: NDArray) -> NDArray:
    """
    Normalize a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The normalized quaternion.
    """
    check(q)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return q / norms


def multiply_assume_normalized(q1: NDArray, q2: NDArray) -> NDArray:
    """
    Multiply two quaternions together, assuming they are already normalized.

    This is a performance-optimized version of :func:`multiply` that skips
    normalization of the input quaternions. Use this only when you are certain
    both quaternions are already normalized.

    :param q1: A normalized quaternion ((x, y, z), w)).
    :param q2: A normalized quaternion ((x, y, z), w)).
    :return: The product q1*q2.
    """
    check(q1)
    check(q2)

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.stack((x, y, z, w), axis=-1)


def multiply(q1: NDArray, q2: NDArray) -> NDArray:
    """
    Multiply two quaternions together.

    Normalizes input quaternions before multiplication for numerical stability.
    For performance-critical code where quaternions are guaranteed to be normalized,
    use :func:`multiply_assume_normalized`.

    :param q1: A quaternion ((x, y, z), w)).
    :param q2: A quaternion ((x, y, z), w)).
    :return: The normalized product q1*q2.
    """
    return multiply_assume_normalized(normalize(q1), normalize(q2))


def conjugate(q: NDArray) -> NDArray:
    """
    Conjugate a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The conjugate.
    """
    check(q)
    scalar, vec = split(q)
    return np.concatenate((-vec, scalar), axis=-1)


def inverse(q: NDArray) -> NDArray:
    """
    Compute the inverse of a quaternion.

    Uses numerical clamping to avoid division by very small numbers,
    improving numerical stability for near-zero quaternions.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The inverse.
    """
    check(q)
    return conjugate(q) / np.maximum((q * q).sum(axis=-1, keepdims=True), 1e-7)


def _get_nonzero_denominator(d: NDArray, eps: float) -> NDArray:
    near_zeros = np.abs(d) < eps
    d = d * (~near_zeros)
    d = d + np.sign(d) * (near_zeros * eps)
    return d


def quaternion_to_xyz_euler(q: NDArray, eps: float = 1e-6) -> NDArray:
    """
    :param eps: a small number to avoid calling asin(1) or asin(-1).
        Should not be smaller than 1e-6 as this can cause numerical issues.
    """
    check(q)
    q = normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    denom = _get_nonzero_denominator(1 - 2 * (np.square(x) + np.square(y)), eps=eps)
    rx = np.arctan2(2 * (w * x + y * z), denom)
    ry = np.arcsin(np.clip(2 * (w * y - z * x), -1 + eps, 1 - eps))

    denom = _get_nonzero_denominator(1 - 2 * (np.square(y) + np.square(z)), eps=eps)
    rz = np.arctan2(2 * (w * z + x * y), denom)
    return np.stack([rx, ry, rz], axis=-1)


def rotate_vector_assume_normalized(q: NDArray, v: NDArray) -> NDArray:
    """
    Rotate a vector by a quaternion, assuming the quaternion is already normalized.

    This is a performance-optimized version of :func:`rotate_vector` that skips
    normalization of the input quaternion. Use this only when you are certain
    the quaternion is already normalized.

    :param q: (nBatch x k x 4) array with normalized quaternions in ((x, y, z), w) format.
    :param v: (nBatch x k x 3) vector.
    :return: (nBatch x k x 3) rotated vectors.
    """
    check(q)
    r, axis = split(q)
    av = np.cross(axis, v, axis=-1)
    aav = np.cross(axis, av, axis=-1)
    return v + 2 * (av * r + aav)


def rotate_vector(q: NDArray, v: NDArray) -> NDArray:
    """
    Rotate a vector by a quaternion.

    Normalizes the input quaternion before rotation for numerical stability.
    For performance-critical code where quaternions are guaranteed to be normalized,
    use :func:`rotate_vector_assume_normalized`.

    :param q: (nBatch x k x 4) array with the quaternions in ((x, y, z), w) format.
    :param v: (nBatch x k x 3) vector.
    :return: (nBatch x k x 3) rotated vectors.
    """
    return rotate_vector_assume_normalized(normalize(q), v)


def to_rotation_matrix_assume_normalized(q: NDArray) -> NDArray:
    """
    Convert quaternions to 3x3 rotation matrices.

    :parameter q: (nBatch x k x 4) array with the quaternions in ((x, y, z), w) format.
    :return: (nBatch x k x 3 x 3) array with 3x3 rotation matrices.
    """
    check(q)
    qx = q[..., 0:1]
    qy = q[..., 1:2]
    qz = q[..., 2:3]
    qw = q[..., 3:4]
    qx2 = np.square(qx)
    qy2 = np.square(qy)
    qz2 = np.square(qz)
    qxqy = qx * qy
    qxqz = qx * qz
    qxqw = qx * qw
    qyqz = qy * qz
    qyqw = qy * qw
    qzqw = qz * qw
    one = np.ones_like(qx)
    result = np.concatenate(
        [
            one - 2 * (qy2 + qz2),
            2 * (qxqy - qzqw),
            2 * (qxqz + qyqw),
            2 * (qxqy + qzqw),
            one - 2 * (qx2 + qz2),
            2 * (qyqz - qxqw),
            2 * (qxqz - qyqw),
            2 * (qyqz + qxqw),
            one - 2 * (qx2 + qy2),
        ],
        axis=-1,
    )
    return result.reshape(list(q.shape[:-1]) + [3, 3])


def to_rotation_matrix(q: NDArray) -> NDArray:
    """
    Convert quaternions to 3x3 rotation matrices.

    :parameter q: (nBatch x k x 4) array with the quaternions in ((x, y, z), w) format.
    :return: (nBatch x k x 3 x 3) array with 3x3 rotation matrices.
    """
    return to_rotation_matrix_assume_normalized(normalize(q))


def identity(
    size: Sequence[int] | None = None,
    dtype: np.dtype = np.float32,
) -> NDArray:
    """
    Create a quaternion identity array.

    :parameter size: A tuple of integers representing the size of the quaternion array.
    :parameter dtype: The data type for the array.
    :return: A quaternion identity array with the specified size and dtype.
    """
    size = size or ()
    return np.concatenate(
        [
            np.zeros((*size, 3), dtype=dtype),
            np.ones((*size, 1), dtype=dtype),
        ],
        axis=-1,
    )


def from_axis_angle(axis_angle: NDArray) -> NDArray:
    """
    Convert an axis-angle array to a quaternion.

    :parameter axis_angle: An array of shape (..., 3) representing the axis-angle.
    :return: An array of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    normed_axes = axis_angle / np.maximum(angles, 1e-8)
    sin_half_angles = np.sin(angles / 2)
    cos_half_angles = np.cos(angles / 2)

    return np.concatenate([normed_axes * sin_half_angles, cos_half_angles], axis=-1)


def euler_xyz_to_quaternion(euler_xyz: NDArray) -> NDArray:
    """
    Convert Euler XYZ angles to a quaternion.

    This function converts XYZ Euler angles to quaternions.
    The rotation order is X-Y-Z, meaning first rotate around X-axis, then Y-axis,
    then Z-axis.

    :parameter euler_xyz: An array of shape (..., 3) representing the Euler XYZ angles
                         in order [roll, pitch, yaw].
    :return: An array of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    roll, pitch, yaw = euler_xyz[..., 0], euler_xyz[..., 1], euler_xyz[..., 2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.stack((x, y, z, w), axis=-1)


def euler_zyx_to_quaternion(euler_zyx: NDArray) -> NDArray:
    """
    Convert Euler ZYX angles to a quaternion.

    This function converts ZYX Euler angles (yaw-pitch-roll convention) to quaternions.
    The rotation order is Z-Y-X, meaning first rotate around Z-axis (yaw), then Y-axis (pitch),
    then X-axis (roll).

    :parameter euler_zyx: An array of shape (..., 3) representing the Euler ZYX angles
                         in order [yaw, pitch, roll].
    :return: An array of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    yaw, pitch, roll = euler_zyx[..., 0], euler_zyx[..., 1], euler_zyx[..., 2]

    # Compute half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Compute quaternion components for ZYX convention
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    w = cr * cp * cy - sr * sp * sy

    return np.stack((x, y, z, w), axis=-1)


def from_rotation_matrix(matrices: NDArray, eta: float = 1e-6) -> NDArray:
    """
    Convert a rotation matrix to a quaternion using numerically stable method.

    This implementation uses the robust algorithm that computes all four quaternion
    component candidates and selects the best-conditioned one, ensuring numerical
    stability across all rotation matrix configurations.

    :parameter matrices: An array of shape (..., 3, 3) representing the rotation matrices.
    :parameter eta: Numerical precision threshold (unused, kept for compatibility).
    :return: An array of shape (..., 4) representing the quaternions in ((x, y, z), w) format.
    """
    m = matrices
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # Compute the absolute values of all four quaternion components
    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,  # w component
                    1.0 + m00 - m11 - m22,  # x component
                    1.0 - m00 + m11 - m22,  # y component
                    1.0 - m00 - m11 + m22,  # z component
                ],
                axis=-1,
            ),
            1e-15,
        )
    )

    # We produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = np.stack(
        [
            np.stack(
                [m21 - m12, m02 - m20, m10 - m01, np.square(q_abs[..., 0])], axis=-1
            ),
            np.stack(
                [np.square(q_abs[..., 1]), m10 + m01, m02 + m20, m21 - m12], axis=-1
            ),
            np.stack(
                [m10 + m01, np.square(q_abs[..., 2]), m12 + m21, m02 - m20], axis=-1
            ),
            np.stack(
                [m20 + m02, m21 + m12, np.square(q_abs[..., 3]), m10 - m01], axis=-1
            ),
        ],
        axis=-2,
    )

    # We floor here at 0.01 to avoid divide-by-zero but the exact level is not important;
    # if q_abs is small, the candidate won't be picked.
    flr = 0.01
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

    # If not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    result = quat_candidates[..., 0, :]

    # Select the best candidate by picking the one with the largest denominator.
    result = np.where(
        q_abs[..., 1, None] > q_abs[..., 0, None], quat_candidates[..., 1, :], result
    )
    result = np.where(
        np.logical_and(
            q_abs[..., 2, None] > q_abs[..., 0, None],
            q_abs[..., 2, None] > q_abs[..., 1, None],
        ),
        quat_candidates[..., 2, :],
        result,
    )
    result = np.where(
        np.logical_and(
            np.logical_and(
                q_abs[..., 3, None] > q_abs[..., 0, None],
                q_abs[..., 3, None] > q_abs[..., 1, None],
            ),
            q_abs[..., 3, None] > q_abs[..., 2, None],
        ),
        quat_candidates[..., 3, :],
        result,
    )
    return normalize(result)


def check_and_normalize_weights(
    quaternions: NDArray, weights_in: NDArray | None = None
) -> NDArray:
    """
    Check and normalize the weights for blending quaternions.

    :parameter quaternions: An array of shape (..., k, 4) representing the quaternions to blend.
    :parameter weights_in: An optional array of shape (..., k) representing the weights for each quaternion.
                       If not provided, all quaternions will be weighted equally.
    :return: An array of shape (..., k) representing the normalized weights.
    """
    if weights_in is not None:
        weights = weights_in
    else:
        weights = np.ones_like(quaternions[..., 0])

    if weights.ndim == quaternions.ndim:
        weights = weights[..., 0:1].reshape(weights.shape[:-1])

    if weights.ndim + 1 != quaternions.ndim:
        raise ValueError(
            f"Expected weights vector to match quaternion vector in all dimensions except the last; "
            f"got weights={weights.shape} and quaternions={quaternions.shape}"
        )

    for i in range(weights.ndim):
        if weights.shape[i] != quaternions.shape[i]:
            raise ValueError(
                f"Expected weights vector to match quaternion vector in all dimensions except the last; "
                f"got weights={weights.shape} and quaternions={quaternions.shape}"
            )

    # Normalize the weights
    weights = np.maximum(weights, 0)
    weight_sum = weights.sum(axis=-1, keepdims=True)
    return weights / np.broadcast_to(weight_sum, weights.shape)


def blend(quaternions: NDArray, weights_in: NDArray | None = None) -> NDArray:
    """
    Blend multiple quaternions together using the method described in
    https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    and http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf.

    :parameter quaternions: An array of shape (..., k, 4) representing the quaternions to blend.
    :parameter weights_in: An optional array of shape (..., k) representing the weights for each quaternion.
                       If not provided, all quaternions will be weighted equally.
    :return: An array of shape (..., 4) representing the blended quaternion.
    """
    # If no weights, then assume evenly weighted:
    weights = check_and_normalize_weights(quaternions, weights_in)

    # Find average rotation by means described in the references above
    check(quaternions)
    outer_prod = np.einsum("...i,...k->...ik", quaternions, quaternions)
    QtQ = (weights[..., None, None] * outer_prod).sum(axis=-3)
    eigenvalues, eigenvectors = np.linalg.eigh(QtQ)
    result = eigenvectors[..., :, 3]
    return result


def slerp(q0: NDArray, q1: NDArray, t: NDArray | float) -> NDArray:
    """
    Perform spherical linear interpolation (slerp) between two quaternions.

    :parameter q0: The starting quaternion.
    :parameter q1: The ending quaternion.
    :parameter t: The interpolation parameter, where 0 <= t <= 1.  t=0 corresponds to q0, t=1 corresponds to q1.
    :return: The interpolated quaternion.
    """
    check(q0)
    check(q1)

    t = np.asarray(t)

    # Compute the cosine of the angle between the two quaternions
    cos_theta = np.einsum("...x,...x", q0, q1)[..., None]
    # Clamp for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # If the dot product is negative, the quaternions have opposite handed-ness
    # and slerp won't take the shorter path. Fix by reversing one quaternion.
    q1 = np.where(cos_theta < 0, -q1, q1)
    cos_theta = np.abs(cos_theta)

    # Use linear interpolation for very close quaternions to avoid division by zero
    lerp_result = normalize(q0 + t * (q1 - q0))

    # Calculate the angle and the sin of the angle
    eps = 1e-4
    theta = np.arccos(np.clip(cos_theta, 0, 1.0 - eps))
    inv_sin_theta = np.reciprocal(np.sin(theta))
    c0 = np.sin((1 - t) * theta) * inv_sin_theta
    c1 = np.sin(t * theta) * inv_sin_theta

    slerp_result = normalize(c0 * q0 + c1 * q1)

    return np.where(cos_theta > 0.9995, lerp_result, slerp_result)


def from_two_vectors(v1: NDArray, v2: NDArray) -> NDArray:
    """
    Construct a quaternion that rotates one vector into another.

    :parameter v1: The initial vector.
    :parameter v2: The target vector.
    :return: A quaternion representing the rotation from v1 to v2.
    """
    # Ensure both vectors are unit vectors
    v1 = v1 / np.maximum(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-12)
    v2 = v2 / np.maximum(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-12)

    scalar = np.sum(v1 * v2, axis=-1, keepdims=True) + 1
    vec = np.cross(v1, v2, axis=-1)

    # handle the anti-parallel case, we need a vector which is perpendicular to
    # both v1 and v2 which we can obtain using the SVD:
    m = np.stack([v1, v2], axis=-2)
    _, _, vh = np.linalg.svd(m, full_matrices=True)
    axis = vh[..., 2, :]

    vec = np.where(scalar <= 0, axis, vec)
    return normalize(np.concatenate((vec, scalar), axis=-1))
