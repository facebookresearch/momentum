# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TRS (Translation-Rotation-Scale) Utilities
===========================================

This module provides utilities for working with TRS (Translation-Rotation-Scale)
transforms in PyMomentum.

A TRS transform is represented as a tuple of three separate tensors:

- **Translation (t)**: tensor of shape [..., 3] - 3D position offset
- **Rotation (r)**: tensor of shape [..., 3, 3] - 3x3 rotation matrix
- **Scale (s)**: tensor of shape [..., 1] - uniform scale factor

This representation provides efficient operations since rotation matrices can be
directly used for transformations and inverted via transpose, avoiding quaternion
to matrix conversions.  It is also preferable to working with 4x4 matrices, because
extracting the scale from a fully general 4x4 matrix is expensive.

Note that internally, momentum mostly uses skel_states which use quaternions rather
than rotation matrices.  However, many use cases (particularly for ML) prefer rotation
matrices (such as the widely-used 6D rotation representation), so this library provides
useful functionality for converting between the two.

Key features:

- Creating TRS transforms from individual components (:func:`from_translation`,
  :func:`from_rotation_matrix`, :func:`from_scale`)
- Converting between TRS transforms and 4x4 transformation matrices
  (:func:`to_matrix`, :func:`from_matrix`)
- Performing transformations and operations (:func:`multiply`, :func:`inverse`,
  :func:`transform_points`)
- Interoperability with skeleton states (:func:`from_skeleton_state`,
  :func:`to_skeleton_state`)
- Fast inverse computation using rotation matrix transpose

Example:
    Creating and using a TRS transform::

        import torch
        from pymomentum import trs

        # Create identity transform
        t, r, s = trs.identity()

        # Create from translation
        translation = torch.tensor([1.0, 2.0, 3.0])
        t, r, s = trs.from_translation(translation)

        # Transform points
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        transformed = trs.transform_points((t, r, s), points)

Note:
    All functions in this module work with TRS transforms represented as tuples
    of (translation, rotation_matrix, scale) tensors.
"""

from typing import Sequence

import torch
from pymomentum import quaternion

# pyre-strict

# Type alias for TRS transform tuple
TRSTransform = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def from_translation(translation: torch.Tensor) -> TRSTransform:
    """
    Create a TRS transform from translation.

    :parameter translation: The translation component of shape [..., 3].
    :type translation: torch.Tensor
    :return: TRS transform tuple (translation, identity_rotation, unit_scale).
    :rtype: TRSTransform
    """
    batch_shape = translation.shape[:-1]
    device = translation.device
    dtype = translation.dtype

    # Identity rotation matrix
    rotation = (
        torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).contiguous()
    )

    # Unit scale
    scale = torch.ones(*batch_shape, 1, device=device, dtype=dtype)

    return translation, rotation, scale


def from_rotation_matrix(rotation_matrix: torch.Tensor) -> TRSTransform:
    """
    Create a TRS transform from rotation matrix.

    :parameter rotation_matrix: The rotation matrix component of shape [..., 3, 3].
    :type rotation_matrix: torch.Tensor
    :return: TRS transform tuple (zero_translation, rotation_matrix, unit_scale).
    :rtype: TRSTransform
    """
    batch_shape = rotation_matrix.shape[:-2]
    device = rotation_matrix.device
    dtype = rotation_matrix.dtype

    # Zero translation
    translation = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)

    # Unit scale
    scale = torch.ones(*batch_shape, 1, device=device, dtype=dtype)

    return translation, rotation_matrix, scale


def from_scale(scale: torch.Tensor) -> TRSTransform:
    """
    Create a TRS transform from scale.

    :parameter scale: The scale component of shape [..., 1].
    :type scale: torch.Tensor
    :return: TRS transform tuple (zero_translation, identity_rotation, scale).
    :rtype: TRSTransform
    """
    batch_shape = scale.shape[:-1]
    device = scale.device
    dtype = scale.dtype

    # Zero translation
    translation = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)

    # Identity rotation matrix
    rotation = (
        torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).contiguous()
    )

    return translation, rotation, scale


def identity(
    size: Sequence[int] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TRSTransform:
    """
    Returns a TRS transform representing the identity transform.

    :parameter size: The size of each batch dimension in the output tensors.
                     Defaults to None, which means the output will have no batch dimensions.
    :type size: Sequence[int], optional
    :parameter device: The device on which to create the tensors. Defaults to None.
    :type device: torch.device, optional
    :parameter dtype: The data type of the tensors. Defaults to None (float32).
    :type dtype: torch.dtype, optional
    :return: Identity TRS transform tuple.
    :rtype: TRSTransform
    """
    if dtype is None:
        dtype = torch.float32

    if size is None:
        size = []

    # Zero translation
    translation = torch.zeros(*size, 3, device=device, dtype=dtype)

    # Identity rotation matrix
    rotation = torch.eye(3, device=device, dtype=dtype).expand(*size, 3, 3).contiguous()

    # Unit scale
    scale = torch.ones(*size, 1, device=device, dtype=dtype)

    return translation, rotation, scale


def multiply(trs1: TRSTransform, trs2: TRSTransform) -> TRSTransform:
    """
    Multiply (compose) two TRS transforms.

    :parameter trs1: The first TRS transform tuple (t1, r1, s1).
    :type trs1: TRSTransform
    :parameter trs2: The second TRS transform tuple (t2, r2, s2).
    :type trs2: TRSTransform
    :return: The composed TRS transform tuple representing trs1 * trs2.
    :rtype: TRSTransform
    """
    t1, r1, s1 = trs1
    t2, r2, s2 = trs2

    # Composed rotation: r1 @ r2
    r_result = rotmat_multiply(r1, r2)

    # Composed scale: s1 * s2
    s_result = s1 * s2

    # Composed translation: t1 + r1 @ (s1 * t2)
    # First scale t2 by s1, then rotate by r1, then add t1
    scaled_t2 = s1 * t2  # [..., 3, 1]
    rotated_scaled_t2 = rotmat_rotate_vector(r1, scaled_t2)
    t_result = t1 + rotated_scaled_t2

    return t_result, r_result, s_result


def inverse(trs: TRSTransform) -> TRSTransform:
    """
    Compute the inverse of a TRS transform. This is efficient since rotation
    matrix inverse is just transpose.

    :parameter trs: The TRS transform tuple to invert.
    :type trs: TRSTransform
    :return: The inverted TRS transform tuple.
    :rtype: TRSTransform
    """
    t, r, s = trs

    # Inverse rotation: transpose of rotation matrix
    r_inv = r.transpose(-2, -1)

    # Inverse scale: reciprocal
    s_inv = torch.reciprocal(s)

    # Inverse translation: -R^T * (t / s)
    # First scale translation by inverse scale, then apply inverse rotation
    scaled_t = s_inv.unsqueeze(-1) * t.unsqueeze(-1)  # [..., 3, 1]
    t_inv = -rotmat_multiply(r_inv, scaled_t).view_as(t)

    return t_inv, r_inv, s_inv


def transform_points(trs: TRSTransform, points: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D points by the TRS transform.

    :parameter trs: The TRS transform tuple to use for transformation.
    :type trs: TRSTransform
    :parameter points: The points to transform of shape [..., 3].
    :type points: torch.Tensor
    :return: The transformed points of shape [..., 3].
    :rtype: torch.Tensor
    """
    if points.dim() < 1 or points.shape[-1] != 3:
        raise ValueError("Points tensor should have last dimension 3.")

    t, r, s = trs

    # Apply scale, then rotation, then translation: t + r @ (s * points)
    transformed_points = t + rotmat_rotate_vector(r, s * points)

    return transformed_points


def to_matrix(trs: TRSTransform) -> torch.Tensor:
    """
    Convert TRS transform to 4x4 transformation matrix.

    :parameter trs: The TRS transform tuple to convert.
    :type trs: TRSTransform
    :return: A tensor containing 4x4 transformation matrices of shape [..., 4, 4].
    :rtype: torch.Tensor
    """
    t, r, s = trs

    # Scale the rotation matrix
    linear = r * s.unsqueeze(-2).expand_as(r)

    # Construct the affine part [linear | translation]
    affine = torch.cat((linear, t.unsqueeze(-1)), -1)

    # Add the homogeneous row [0, 0, 0, 1]
    batch_shape = t.shape[:-1]
    last_row = torch.tensor([0, 0, 0, 1], device=t.device, dtype=t.dtype)
    last_row = last_row.expand(*batch_shape, 1, 4)

    # Combine affine and homogeneous row
    matrix = torch.cat((affine, last_row), -2)

    return matrix


def from_matrix(matrices: torch.Tensor) -> TRSTransform:
    """
    Convert 4x4 transformation matrices to TRS transforms. Assumes uniform scaling.

    :parameter matrices: A tensor of 4x4 matrices of shape [..., 4, 4].
    :type matrices: torch.Tensor
    :return: TRS transform tuple (translation, rotation_matrix, scale).
    :rtype: TRSTransform
    """
    if matrices.dim() < 2 or matrices.shape[-1] != 4 or matrices.shape[-2] != 4:
        raise ValueError("Expected a tensor of 4x4 matrices")

    initial_shape = matrices.shape
    if matrices.dim() == 2:
        matrices = matrices.unsqueeze(0)
    else:
        matrices = matrices.flatten(0, -3)

    # Extract linear part and translation
    linear = matrices[..., :3, :3]
    translation = matrices[..., :3, 3]

    # Use SVD to decompose linear part: linear = U * S * V^T
    # where U and V^T are rotations and S contains scales
    U, S, Vt = torch.linalg.svd(linear)

    # Extract scale (assuming uniform scaling, take first singular value)
    scale = S[..., :1]

    # Extract rotation matrix: R = U * V^T
    rotation_matrix = torch.bmm(U, Vt)

    # Reshape back to original batch dimensions
    result_shape_t = list(initial_shape[:-2]) + [3]
    result_shape_r = list(initial_shape[:-2]) + [3, 3]
    result_shape_s = list(initial_shape[:-2]) + [1]

    translation = translation.reshape(result_shape_t)
    rotation_matrix = rotation_matrix.reshape(result_shape_r)
    scale = scale.reshape(result_shape_s)

    return translation, rotation_matrix, scale


def from_skeleton_state(skeleton_state: torch.Tensor) -> TRSTransform:
    """
    Convert skeleton state to TRS transform.

    :parameter skeleton_state: The skeleton state tensor of shape [..., 8] containing
                               (tx, ty, tz, rx, ry, rz, rw, s).
    :type skeleton_state: torch.Tensor
    :return: TRS transform tuple (translation, rotation_matrix, scale).
    :rtype: TRSTransform
    """
    if skeleton_state.shape[-1] != 8:
        raise ValueError("Expected skeleton state to have last dimension 8")

    # Extract translation, quaternion, and scale
    translation = skeleton_state[..., :3]
    quaternion_xyzw = skeleton_state[..., 3:7]
    scale = skeleton_state[..., 7:]

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion.to_rotation_matrix(quaternion_xyzw)

    return translation, rotation_matrix, scale


def to_skeleton_state(trs: TRSTransform) -> torch.Tensor:
    """
    Convert TRS transform to skeleton state.

    :parameter trs: The TRS transform tuple to convert.
    :type trs: TRSTransform
    :return: Skeleton state tensor of shape [..., 8] containing (tx, ty, tz, rx, ry, rz, rw, s).
    :rtype: torch.Tensor
    """
    t, r, s = trs

    # Convert rotation matrix to quaternion
    quaternion_xyzw = quaternion.from_rotation_matrix(r)

    # Combine into skeleton state format
    skeleton_state = torch.cat([t, quaternion_xyzw, s], dim=-1)

    return skeleton_state


def slerp(trs0: TRSTransform, trs1: TRSTransform, t: torch.Tensor) -> TRSTransform:
    """
    Spherical linear interpolation between two TRS transforms.

    :parameter trs0: The first TRS transform tuple.
    :type trs0: TRSTransform
    :parameter trs1: The second TRS transform tuple.
    :type trs1: TRSTransform
    :parameter t: The interpolation factor where 0 <= t <= 1. t=0 corresponds to trs0, t=1 corresponds to trs1.
    :type t: torch.Tensor
    :return: The interpolated TRS transform.
    :rtype: TRSTransform
    """
    t0, r0, s0 = trs0
    t1, r1, s1 = trs1

    # Linear interpolation for translation and scale
    t_interp = (1 - t).unsqueeze(-1) * t0 + t.unsqueeze(-1) * t1
    s_interp = (1 - t).unsqueeze(-1) * s0 + t.unsqueeze(-1) * s1

    # Spherical interpolation for rotation matrices via quaternions
    q0 = quaternion.from_rotation_matrix(r0)
    q1 = quaternion.from_rotation_matrix(r1)
    q_interp = quaternion.slerp(q0, q1, t)
    r_interp = quaternion.to_rotation_matrix(q_interp)

    return t_interp, r_interp, s_interp


def blend(
    trs_transforms: Sequence[TRSTransform], weights: torch.Tensor | None = None
) -> TRSTransform:
    """
    Blend multiple TRS transforms with the given weights.

    :parameter trs_transforms: A sequence of TRS transform tuples to blend.
    :type trs_transforms: Sequence[TRSTransform]
    :parameter weights: The weights to use for blending. If not provided, equal weights are used.
                        Should have shape [num_transforms] or [..., num_transforms].
    :type weights: torch.Tensor, optional
    :return: The blended TRS transform.
    :rtype: TRSTransform
    """
    if len(trs_transforms) == 0:
        raise ValueError("Cannot blend empty list of transforms")

    if len(trs_transforms) == 1:
        return trs_transforms[0]

    # Stack all transforms
    translations = torch.stack([trs[0] for trs in trs_transforms], dim=-2)
    rotations = torch.stack([trs[1] for trs in trs_transforms], dim=-3)
    scales = torch.stack([trs[2] for trs in trs_transforms], dim=-2)

    # Handle weights
    if weights is None:
        num_transforms = len(trs_transforms)
        device = translations.device
        dtype = translations.dtype
        weights = (
            torch.ones(num_transforms, device=device, dtype=dtype) / num_transforms
        )

    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Blend translation and scale with linear interpolation
    t_blend = (weights.unsqueeze(-1) * translations).sum(dim=-2)
    s_blend = (weights.unsqueeze(-1) * scales).sum(dim=-2)

    # Blend rotations via quaternions
    quaternions = torch.stack(
        [quaternion.from_rotation_matrix(r) for r in rotations], dim=-2
    )
    q_blend = quaternion.blend(quaternions, weights)
    r_blend = quaternion.to_rotation_matrix(q_blend)

    return t_blend, r_blend, s_blend


# ======================================================================
# Rotation Matrix Utilities
# ======================================================================


def rotmat_inverse(r: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of rotation matrices.

    For orthogonal rotation matrices, the inverse is simply the transpose.

    :parameter r: Rotation matrices of shape [..., 3, 3].
    :type r: torch.Tensor
    :return: Inverse rotation matrices of shape [..., 3, 3].
    :rtype: torch.Tensor
    """
    return r.transpose(-2, -1)


def rotmat_multiply(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two rotation matrices.

    :parameter r1: First rotation matrices of shape [..., 3, 3].
    :type r1: torch.Tensor
    :parameter r2: Second rotation matrices of shape [..., 3, 3].
    :type r2: torch.Tensor
    :return: Product r1 @ r2 of shape [..., 3, 3].
    :rtype: torch.Tensor
    """
    return torch.einsum("...ij,...jk->...ik", r1, r2)


def rotmat_rotate_vector(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation matrices to vectors.

    :parameter r: Rotation matrices of shape [..., 3, 3].
    :type r: torch.Tensor
    :parameter v: Vectors of shape [..., 3].
    :type v: torch.Tensor
    :return: Rotated vectors r @ v of shape [..., 3].
    :rtype: torch.Tensor
    """
    return torch.einsum("...ij,...j->...i", r, v)


def rotmat_from_euler_xyz(euler: torch.Tensor) -> torch.Tensor:
    """
    Create rotation matrices from XYZ Euler angles.

    This function converts XYZ Euler angles to rotation matrices using the
    ZYX rotation order (also known as Tait-Bryan angles). The rotation is
    applied as follows: first around the X-axis, then Y-axis, then Z-axis.

    :parameter euler: Euler angles of shape [..., 3] where the last dimension
                      contains [rx, ry, rz] rotations in radians.
    :type euler: torch.Tensor
    :return: Rotation matrices of shape [..., 3, 3].
    :rtype: torch.Tensor
    """
    cos_angles = torch.cos(euler)
    sin_angles = torch.sin(euler)

    cx, cy, cz = cos_angles.unbind(-1)
    sx, sy, sz = sin_angles.unbind(-1)

    result = torch.stack(
        [
            cy * cz,
            -cx * sz + sx * sy * cz,
            sx * sz + cx * sy * cz,
            cy * sz,
            cx * cz + sx * sy * sz,
            -sx * cz + cx * sy * sz,
            -sy,
            sx * cy,
            cx * cy,
        ],
        dim=-1,
    )
    result_shape = result.shape[:-1] + (3, 3)
    return result.view(result_shape)
