# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

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

    :parameter q1: A quaternion ((x, y, z), w)).
    :parameter q2: A quaternion ((x, y, z), w)).
    :return: The product q1*q2.
    """
    check(q1)
    check(q2)
    r1, v1 = split(q1)
    r2, v2 = split(q2)
    assert q1.size() == q2.size(), "Expected matching quaternion dimensions."
    r_res = r1 * r2 - (v1 * v2).sum(-1, keepdim=True)
    v_res = r1.expand_as(v2) * v2 + r2.expand_as(v1) * v1 + torch.cross(v1, v2, -1)
    return torch.cat((v_res, r_res), -1)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The normalized quaternion.
    """
    check(q)
    return q / q.norm(2, dim=-1, keepdim=True)


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

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The inverse.
    """
    return conjugate(q) / (q * q).sum(-1, keepdim=True)


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

    :parameter q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
    :parameter v: (nBatch x k x 3) vector.
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
