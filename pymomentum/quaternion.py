# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Quaternion Utilities
===================

This module provides comprehensive utilities for working with quaternions in PyMomentum.

Quaternions are a mathematical representation of rotations in 3D space that offer several
advantages over other rotation representations like Euler angles or rotation matrices:

- **No gimbal lock**: Unlike Euler angles, quaternions don't suffer from singularities
- **Compact representation**: Only 4 components vs 9 for rotation matrices
- **Efficient composition**: Quaternion multiplication is faster than matrix multiplication
- **Smooth interpolation**: SLERP provides natural rotation interpolation

Quaternion Format
-----------------
This module uses the (x, y, z, w) format where:

- **(x, y, z)**: Vector part representing the rotation axis scaled by sin(θ/2)
- **w**: Scalar part representing cos(θ/2), where θ is the rotation angle

The identity quaternion is (0, 0, 0, 1), representing no rotation.

Core Operations
---------------
The module provides functions for:

- **Basic operations**: :func:`multiply`, :func:`conjugate`, :func:`inverse`, :func:`normalize`
- **Conversions**: :func:`from_axis_angle`, :func:`euler_xyz_to_quaternion`,
  :func:`from_rotation_matrix`, :func:`to_rotation_matrix`
- **Vector operations**: :func:`rotate_vector`, :func:`from_two_vectors`
- **Interpolation**: :func:`slerp`, :func:`blend`
- **Utilities**: :func:`check`, :func:`split`, :func:`identity`

Example:
    Basic quaternion operations::

        import torch
        from pymomentum import quaternion

        # Create identity quaternion
        q_identity = quaternion.identity()

        # Create quaternion from axis-angle
        axis_angle = torch.tensor([0.0, 0.0, 1.57])  # 90° rotation around Z
        q_rot = quaternion.from_axis_angle(axis_angle)

        # Rotate a vector
        vector = torch.tensor([1.0, 0.0, 0.0])
        rotated = quaternion.rotate_vector(q_rot, vector)

        # Interpolate between quaternions
        q_interp = quaternion.slerp(q_identity, q_rot, 0.5)

Note:
    All functions expect quaternions as PyTorch tensors with the last dimension
    having size 4, following the (x, y, z, w) format. Most functions support
    batched operations for efficient processing of multiple quaternions.
"""

from typing import Sequence, Tuple

import torch

# pyre-strict


def check(q: torch.Tensor) -> None:
    """
    Check if a tensor represents a quaternion.

    :parameter q: A tensor representing a quaternion.
    """
    assert q.size(-1) == 4, "Quaternion should have last dimension equal to 4."


def split(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a quaternion into its scalar and vector parts.

    :parameter q: A tensor representing a quaternion.
    :return: The scalar and vector parts of the quaternion.
    """
    check(q)
    return q.narrow(-1, 3, 1), q.narrow(-1, 0, 3)


def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def multiply_assume_normalized(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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

    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack((x, y, z, w), dim=-1)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The normalized quaternion.
    """
    check(q)
    return torch.nn.functional.normalize(q, dim=-1)


def conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Conjugate a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The conjugate.
    """
    check(q)
    scalar, vec = split(q)
    return torch.cat((-vec, scalar), -1)


def inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a quaternion.

    Uses numerical clamping to avoid division by very small numbers,
    improving numerical stability for near-zero quaternions.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The inverse.
    """
    return conjugate(q) / torch.clamp((q * q).sum(-1, keepdim=True), min=1e-7)


def _get_nonzero_denominator(d: torch.Tensor, eps: float) -> torch.Tensor:
    near_zeros = torch.abs(d) < eps
    d = d * (near_zeros.logical_not())
    d = d + torch.sign(d) * (near_zeros * eps)
    return d


def quaternion_to_xyz_euler(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    :param eps: a small number to avoid calling asin(1) or asin(-1).
        Should not be smaller than 1e-6 as this can cause NaN gradients for some models.
    """
    q = normalize(q)
    x, y, z, w = q.unbind(-1)

    denom = _get_nonzero_denominator(
        1 - 2 * (torch.square(x) + torch.square(y)), eps=eps
    )
    rx = torch.atan2(2 * (w * x + y * z), denom)
    ry = torch.asin(torch.clamp(2 * (w * y - z * x), -1 + eps, 1 - eps))

    denom = _get_nonzero_denominator(
        1 - 2 * (torch.square(y) + torch.square(z)), eps=eps
    )
    rz = torch.atan2(2 * (w * z + x * y), denom)
    return torch.stack([rx, ry, rz], -1)


def rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion.

    Normalizes the input quaternion before rotation for numerical stability.
    For performance-critical code where quaternions are guaranteed to be normalized,
    use :func:`rotate_vector_assume_normalized`.

    :param q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
    :param v: (nBatch x k x 3) vector.
    :return: (nBatch x k x 3) rotated vectors.
    """
    return rotate_vector_assume_normalized(normalize(q), v)


def rotate_vector_assume_normalized(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion, assuming the quaternion is already normalized.

    This is a performance-optimized version of :func:`rotate_vector` that skips
    normalization of the input quaternion. Use this only when you are certain
    the quaternion is already normalized.

    :param q: (nBatch x k x 4) tensor with normalized quaternions in ((x, y, z), w) format.
    :param v: (nBatch x k x 3) vector.
    :return: (nBatch x k x 3) rotated vectors.
    """
    r, axis = split(q)
    av = torch.cross(axis, v, -1)
    aav = torch.cross(axis, av, -1)
    return v + 2 * (av * r + aav)


def to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to 3x3 rotation matrices.

    :parameter q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
    :return: (nBatch x k x 3 x 3) tensor with 3x3 rotation matrices.
    """
    assert q.size(-1) == 4, "Expected quaternion tensor (last dimension=4)."
    qx = q.select(-1, 0).unsqueeze(-1)
    qy = q.select(-1, 1).unsqueeze(-1)
    qz = q.select(-1, 2).unsqueeze(-1)
    qw = q.select(-1, 3).unsqueeze(-1)
    qx2 = torch.square(qx)
    qy2 = torch.square(qy)
    qz2 = torch.square(qz)
    qxqy = qx * qy
    qxqz = qx * qz
    qxqw = qx * qw
    qyqz = qy * qz
    qyqw = qy * qw
    qzqw = qz * qw
    one = torch.ones_like(qx)
    result = torch.cat(
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
        -1,
    )
    return result.reshape(*q.shape[:-1], 3, 3)


def identity(
    size: Sequence[int] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a quaternion identity tensor.

    :parameter sizes: A tuple of integers representing the size of the quaternion tensor.
    :parameter device: The device on which to create the tensor.
    :return: A quaternion identity tensor with the specified sizes and device.
    """
    size = size or ()
    return torch.cat(
        [
            torch.zeros(*size, 3, device=device, dtype=dtype),
            torch.ones(*size, 1, device=device, dtype=dtype),
        ],
        dim=-1,
    )


def from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert an axis-angle tensor to a quaternion.

    :parameter axis_angle: A tensor of shape (..., 3) representing the axis-angle.
    :return: A tensor of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    angles = axis_angle.norm(dim=-1, keepdim=True)
    normed_axes = axis_angle / angles.clamp(min=1e-8)
    sin_half_angles = torch.sin(angles / 2)
    cos_half_angles = torch.cos(angles / 2)

    return torch.cat([normed_axes * sin_half_angles, cos_half_angles], dim=-1)


def euler_xyz_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler XYZ angles to a quaternion.

    :parameter euler_xyz: A tensor of shape (..., 3) representing the Euler XYZ angles.
    :return: A tensor of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    x_angles = euler_xyz[..., 0]
    y_angles = euler_xyz[..., 1]
    z_angles = euler_xyz[..., 2]
    # Create rotation axes and angles
    x_axes = torch.zeros_like(euler_xyz)
    x_axes[..., 0] = 1
    y_axes = torch.zeros_like(euler_xyz)
    y_axes[..., 1] = 1
    z_axes = torch.zeros_like(euler_xyz)
    z_axes[..., 2] = 1
    # Apply rotations in ZYX order (since we're converting from XYZ Euler angles)
    rot_z = from_axis_angle(z_axes * z_angles.unsqueeze(-1))
    rot_y = from_axis_angle(y_axes * y_angles.unsqueeze(-1))
    rot_x = from_axis_angle(x_axes * x_angles.unsqueeze(-1))
    # Multiply quaternions in reverse order (since we applied them in ZYX order)
    q = multiply(rot_z, multiply(rot_y, rot_x))
    return q


def euler_zyx_to_quaternion(euler_zyx: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler ZYX angles to a quaternion.

    This function converts ZYX Euler angles (yaw-pitch-roll convention) to quaternions.
    The rotation order is Z-Y-X, meaning first rotate around Z-axis (yaw), then Y-axis (pitch),
    then X-axis (roll).

    :parameter euler_zyx: A tensor of shape (..., 3) representing the Euler ZYX angles
                         in order [yaw, pitch, roll].
    :return: A tensor of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    yaw, pitch, roll = euler_zyx.unbind(-1)

    # Compute half angles
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    # Compute quaternion components for ZYX convention
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    w = cr * cp * cy - sr * sp * sy

    return torch.stack((x, y, z, w), dim=-1)


def from_rotation_matrix(matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to a quaternion.

    :parameter matrices: A tensor of shape (..., 3, 3) representing the rotation matrices.
    :return: A tensor of shape (..., 4) representing the quaternions in ((x, y, z), w) format.
    """
    # Convert a rotation matrix to a quaternion using the method described here:
    # https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    # Assumes that the input is a rotation matrix, will return the wrong result
    # if not.
    eigenvalues, eigenvectors = torch.linalg.eig(matrices)
    # Angle can be read off the trace, as described in the SO post:
    trace_m = matrices[..., 0, 0] + matrices[..., 1, 1] + matrices[..., 2, 2]
    cos_theta = (trace_m - 1) / 2
    # For quaternion, we need cos(theta/2) and sin(theta/2):
    cos_half_theta = torch.sqrt(torch.clamp((1 + cos_theta) / 2.0, min=0))
    sin_half_theta = torch.sqrt(torch.clamp((1 - cos_theta) / 2.0, min=0))
    # There is one vector for which R * v = v; that vector must be the axis of
    # rotation and has an eigenvalue of 1.  Because the eigenvalues aren't sorted
    # we'll need to find it by looking for eigenvalue with the largest real
    # component (which should be 1).
    max_eig, max_eig_ind = torch.max(eigenvalues.real, dim=-1)
    # Extract the eigenvector matching the real eigenvalue 1.  This requires a
    # torch.gather because we're extracting a different eigenvector for each
    # input (the eigenvalues are not sorted and anyway they all have norm 1).
    qv = (
        torch.gather(
            eigenvectors,
            dim=-1,
            index=max_eig_ind.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*max_eig_ind.shape, 3, 1),
        )
        .squeeze(-1)
        .real
    )
    qv = sin_half_theta.unsqueeze(-1).expand_as(qv) * qv
    qx = qv[..., 0].unsqueeze(-1)
    qy = qv[..., 1].unsqueeze(-1)
    qz = qv[..., 2].unsqueeze(-1)
    qw = cos_half_theta.unsqueeze(-1)
    # Because of the way we derived qw, we don't know if we have the sign correct
    # (this depends also on which way v is pointing).  To make sure we use the
    # correct one, reconstruct the upper triangular part of the matrix from the
    # quaternion and pick the sign that produces the target matrix.
    #
    # Quaternion-to-matrix is:
    #   [1 0 0]     [-qy^2-qz^2    qx*qy       qx*qz     ]        [ 0  -qz  qy]
    #   [0 1 0] + 2*[qx*qy        -qx^2-qz^2   qy*qz     ] + 2*qw*[ qz  0  -qx]
    #   [0 0 1]     [qx*qz         qy*qz       -qx^2-qy^2]        [-qy  qx  0 ]
    #   identity    symmetric matrix                            skew-symmetric
    #   matrix
    # We can ignore the diagonal entries since they don't depend on w and focus
    # on the upper off-diagonal entries.
    #     2*qx*qy - 2*w*qz = m12
    #     2*qx*qz + 2*w*qy = m13
    #     2*qy*qz - 2*w*qx = m23
    symmetric_part_diff = torch.stack(
        [
            2 * qx * qy - matrices[..., 0, 1].unsqueeze(-1),
            2 * qx * qz - matrices[..., 0, 2].unsqueeze(-1),
            2 * qy * qz - matrices[..., 1, 2].unsqueeze(-1),
        ],
        dim=-1,
    )
    skew_symmetric_part = torch.stack([-2 * qz, 2 * qy, -2 * qx], dim=-1)
    diff_w_positive = (
        symmetric_part_diff
        + qw.unsqueeze(-1).expand_as(skew_symmetric_part) * skew_symmetric_part
    )
    diff_w_negative = (
        symmetric_part_diff
        - qw.unsqueeze(-1).expand_as(skew_symmetric_part) * skew_symmetric_part
    )
    qv = torch.where(
        torch.norm(diff_w_positive, dim=-1) < torch.norm(diff_w_negative, dim=-1),
        qv,
        -qv,
    )
    return torch.cat([qv, qw], dim=-1)


def check_and_normalize_weights(
    quaternions: torch.Tensor, weights_in: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Check and normalize the weights for blending quaternions.

    :parameter quaternions: A tensor of shape (..., k, 4) representing the quaternions to blend.
    :parameter weights_in: An optional tensor of shape (..., k) representing the weights for each quaternion.
                       If not provided, all quaternions will be weighted equally.
    :return: A tensor of shape (..., k) representing the normalized weights.
    """
    if weights_in is not None:
        weights = weights_in
    else:
        weights = torch.ones_like(quaternions.select(-1, 0))

    if weights.dim() == quaternions.dim():
        weights = weights.squeeze(-1)

    if weights.dim() + 1 != quaternions.dim():
        raise ValueError(
            f"Expected weights vector to match quaternion vector in all dimensions except the last; "
            f"got weights={weights.size()} and quaternions={quaternions.size()}"
        )

    for i in range(weights.dim()):
        if weights.size(i) != quaternions.size(i):
            raise ValueError(
                f"Expected weights vector to match quaternion vector in all dimensions except the last; "
                f"got weights={weights.size()} and quaternions={quaternions.size()}"
            )

    # Normalize the weights
    weights = weights.clamp(min=0)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    return weights / weight_sum.expand_as(weights)


def blend(
    quaternions: torch.Tensor, weights_in: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Blend multiple quaternions together using the method described in
    https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    and http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf.

    :parameter quaternions: A tensor of shape (..., k, 4) representing the quaternions to blend.
    :parameter weights_in: An optional tensor of shape (..., k) representing the weights for each quaternion.
                       If not provided, all quaternions will be weighted equally.
    :return: A tensor of shape (..., 4) representing the blended quaternion.
    """
    # If no weights, then assume evenly weighted:
    weights = check_and_normalize_weights(quaternions, weights_in)

    # Find average rotation by means described in the references above
    check(quaternions)
    outer_prod = torch.einsum("...i,...k->...ik", [quaternions, quaternions])
    QtQ = (weights.unsqueeze(-1).unsqueeze(-1) * outer_prod).sum(dim=-3)
    eigenvalues, eigenvectors = torch.linalg.eigh(QtQ)
    result = eigenvectors.select(dim=-1, index=3)
    return result


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Perform spherical linear interpolation (slerp) between two quaternions.

    :parameter q0: The starting quaternion.
    :parameter q1: The ending quaternion.
    :parameter t: The interpolation parameter, where 0 <= t <= 1.  t=0 corresponds to q0, t=1 corresponds to q1.
    :return: The interpolated quaternion.
    """
    check(q0)
    check(q1)

    # Compute the cosine of the angle between the two quaternions
    cos_theta = torch.einsum("...x,...x", q0, q1)[..., None]
    # Clamp for numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # If the dot product is negative, the quaternions have opposite handed-ness
    # and slerp won't take the shorter path. Fix by reversing one quaternion.
    q1 = torch.where(cos_theta < 0, -q1, q1)
    cos_theta = torch.abs(cos_theta)

    # Use linear interpolation for very close quaternions to avoid division by zero
    lerp_result = normalize(q0 + t * (q1 - q0))

    # Calculate the angle and the sin of the angle
    eps = 1e-4
    theta = torch.acos(torch.clamp(cos_theta, 0, 1.0 - eps))
    inv_sin_theta = torch.reciprocal(torch.sin(theta))
    c0 = torch.sin((1 - t) * theta) * inv_sin_theta
    c1 = torch.sin(t * theta) * inv_sin_theta

    slerp_result = normalize(c0 * q0 + c1 * q1)

    return torch.where(cos_theta > 0.9995, lerp_result, slerp_result)


def from_two_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Construct a quaternion that rotates one vector into another.

    :parameter v1: The initial vector.
    :parameter v2: The target vector.
    :return: A quaternion representing the rotation from v1 to v2.
    """
    # Ensure both vectors are unit vectors
    v1 = torch.nn.functional.normalize(v1, dim=-1)
    v2 = torch.nn.functional.normalize(v2, dim=-1)

    scalar = torch.sum(v1 * v2, dim=-1, keepdim=True) + 1
    vec = torch.cross(v1, v2, dim=-1)

    # handle the anti-parallel case, we need a vector which is perpendicular to
    # both v1 and v2 which we can obtain using the SVD:
    m = torch.stack([v1, v2], dim=-2)
    u, s, vh = torch.svd(m, compute_uv=True, some=False)
    axis = vh[..., :, 2]

    vec = torch.where(scalar <= 0, axis, vec)
    return normalize(torch.cat((vec, scalar), dim=-1))


def normalize_backprop(q: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Custom backpropagation for quaternion normalization.

    This function computes gradients for quaternion normalization in a numerically
    stable way, avoiding potential issues with automatic differentiation when
    quaternions are near zero norm.

    :param q: The input quaternion tensor of shape (..., 4).
    :param grad: The gradient from the output of shape (..., 4).
    :return: The gradient with respect to the input quaternion q.
    """
    with torch.no_grad():
        s = torch.linalg.norm(q, dim=-1, keepdim=True)
        g = s * s * grad - q * (torch.sum(q * grad, dim=-1, keepdim=True))
        g = g / (s * s * s)
    return g


def rotate_vector_backprop(
    q: torch.Tensor, v: torch.Tensor, grad: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom backpropagation for quaternion vector rotation.

    Computes gradients for the quaternion rotation operation using the
    Euler-Rodrigues formula.

    This version normalizes the input quaternion. For performance-critical code
    where quaternions are guaranteed to be normalized, use
    :func:`rotate_vector_backprop_assume_normalized`.

    :param q: The quaternion tensor of shape (..., 4).
    :param v: The vector tensor of shape (..., 3).
    :param grad: The gradient from the output of shape (..., 3).
    :return: A tuple of (grad_q, grad_v) representing gradients with respect
             to the quaternion and vector respectively.
    """
    q_normalized = normalize(q)
    grad_q_normalized, grad_v = rotate_vector_backprop_assume_normalized(
        q_normalized, v, grad
    )
    # Convert gradient from normalized quaternion back to original quaternion
    grad_q = normalize_backprop(q, grad_q_normalized)
    return grad_q, grad_v


def rotate_vector_backprop_assume_normalized(
    q: torch.Tensor, v: torch.Tensor, grad: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom backpropagation for quaternion vector rotation assuming unit quaternions.

    This is a performance-optimized version of :func:`rotate_vector_backprop` that
    assumes the input quaternion is already normalized. Use this only when you are
    certain the quaternion is normalized to avoid numerical issues.

    :param q: The normalized quaternion tensor of shape (..., 4).
    :param v: The vector tensor of shape (..., 3).
    :param grad: The gradient from the output of shape (..., 3).
    :return: A tuple of (grad_q, grad_v) representing gradients with respect
             to the quaternion and vector respectively.
    """
    with torch.no_grad():
        # Split quaternion into axis and scalar parts
        a = q[..., :3]  # axis
        w = q[..., 3:]  # scalar

        # Compute cross products needed for gradients
        av = torch.cross(a, v, dim=-1)
        ag = torch.cross(a, grad, dim=-1)
        aag = torch.cross(a, ag, dim=-1)
        gv = torch.cross(grad, v, dim=-1)

        # Compute dot products needed for gradients
        adv = (a * v).sum(dim=-1, keepdim=True)
        adg = (a * grad).sum(dim=-1, keepdim=True)
        vdg = (v * grad).sum(dim=-1, keepdim=True)
        avdg = (av * grad).sum(dim=-1, keepdim=True)

        # Calculate gradients
        grad_v = grad - 2 * w * ag + 2 * aag
        grad_w = 2 * avdg
        grad_a = -2 * gv * w + 2 * (adv * grad + v * adg - 2 * a * vdg)

        grad_q = torch.cat([grad_a, grad_w], dim=-1)
        # For unit quaternions, project gradient to tangent space
        grad_q = grad_q - q * torch.sum(q * grad_q, dim=-1, keepdim=True)

    return grad_q, grad_v


def multiply_backprop(
    q1: torch.Tensor, q2: torch.Tensor, grad_q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom backpropagation for quaternion multiplication.

    Computes gradients for quaternion multiplication with proper handling of
    normalization.

    This version normalizes the input quaternions. For performance-critical code
    where quaternions are guaranteed to be normalized, use
    :func:`multiply_backprop_assume_normalized`.

    :param q1: The first quaternion tensor of shape (..., 4).
    :param q2: The second quaternion tensor of shape (..., 4).
    :param grad_q: The gradient from the output of shape (..., 4).
    :return: A tuple of (grad_q1, grad_q2) representing gradients with respect
             to the first and second quaternions respectively.
    """
    q1_normalized = normalize(q1)
    q2_normalized = normalize(q2)
    grad_q1_normalized, grad_q2_normalized = multiply_backprop_assume_normalized(
        q1_normalized, q2_normalized, grad_q
    )
    # Convert gradients from normalized quaternions back to original quaternions
    grad_q1 = normalize_backprop(q1, grad_q1_normalized)
    grad_q2 = normalize_backprop(q2, grad_q2_normalized)
    return grad_q1, grad_q2


def multiply_backprop_assume_normalized(
    q1: torch.Tensor, q2: torch.Tensor, grad_q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom backpropagation for quaternion multiplication assuming unit quaternions.

    Computes gradients for quaternion multiplication when both input quaternions
    are assumed to be normalized. This is more efficient than the general case
    but should only be used when quaternions are guaranteed to be unit quaternions.

    :param q1: The first normalized quaternion tensor of shape (..., 4).
    :param q2: The second normalized quaternion tensor of shape (..., 4).
    :param grad_q: The gradient from the output of shape (..., 4).
    :return: A tuple of (grad_q1, grad_q2) representing gradients with respect
             to the first and second quaternions respectively.
    """
    with torch.no_grad():
        # Use quaternion multiplication properties for gradient computation
        grad_q1 = multiply_assume_normalized(grad_q, conjugate(q2))
        grad_q2 = multiply_assume_normalized(conjugate(q1), grad_q)

        # For unit quaternions, project gradients to tangent space
        q_result = multiply_assume_normalized(q1, q2)
        grad_q1 = grad_q1 - q1 * torch.sum(q_result * grad_q, dim=-1, keepdim=True)
        grad_q2 = grad_q2 - q2 * torch.sum(q_result * grad_q, dim=-1, keepdim=True)

    return grad_q1, grad_q2
