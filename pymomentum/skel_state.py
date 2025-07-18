# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from pymomentum import quaternion

# pyre-strict


def check(skel_state: torch.Tensor) -> None:
    """
    Check if the skeleton state has the correct shape.

    :parameter skel_state: The skeleton state to check.
    :type skel_state: torch.Tensor
    :raises ValueError: If the skeleton state does not have the correct shape.
    """
    if skel_state.shape[-1] != 8:
        raise ValueError(
            "Expected skeleton state to have last dimension 8 (tx, ty, tz, rx, ry, rz, rw, s)"
        )


def split(skel_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a skeleton state into translation, rotation, and scale components.

    :parameter skel_state: The skeleton state to split.
    :type skel_state: torch.Tensor
    :return: A tuple of tensors (translation, rotation, scale).
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    check(skel_state)
    return skel_state[..., :3], skel_state[..., 3:7], skel_state[..., 7:]


def from_translation(translation: torch.Tensor) -> torch.Tensor:
    """
    Create a skeleton state from translation.

    :parameter translation: The translation component.
    :type translation: torch.Tensor
    :return: The skeleton state.
    :rtype: torch.Tensor
    """
    return torch.cat(
        (
            translation,
            quaternion.identity(
                size=translation.shape[:-1],
                device=translation.device,
                dtype=translation.dtype,
            ),
            torch.ones(
                *translation.shape[:-1],
                1,
                dtype=translation.dtype,
                device=translation.device,
            ),
        ),
        -1,
    )


def from_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """
    Create a skeleton state from rotation.

    :parameter rotation: The rotation component.
    :type rotation: torch.Tensor
    :return: The skeleton state.
    :rtype: torch.Tensor
    """
    return torch.cat(
        (
            torch.zeros(
                *rotation.shape[:-1],
                3,
                dtype=rotation.dtype,
                device=rotation.device,
            ),
            rotation,
            torch.ones(
                *rotation.shape[:-1], 1, dtype=rotation.dtype, device=rotation.device
            ),
        ),
        -1,
    )


def from_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Create a skeleton state from scale.

    :parameter scale: The scale component.
    :type scale: torch.Tensor
    :return: The skeleton state.
    :rtype: torch.Tensor
    """
    return torch.cat(
        (
            torch.zeros(
                *scale.shape[:-1],
                3,
                dtype=scale.dtype,
                device=scale.device,
            ),
            quaternion.identity(
                size=scale.shape[:-1],
                device=scale.device,
                dtype=scale.dtype,
            ),
            scale,
        ),
        -1,
    )


def to_matrix(skeleton_state: torch.Tensor) -> torch.Tensor:
    """
    Convert skeleton state to a tensor of 4x4 matrices. The matrix represents the transform from a local joint space to the world space.

    :parameter skeleton_state: The skeleton state to convert.
    :type skeleton_state: torch.Tensor
    :return: A tensor containing 4x4 matrix transforms.
    :rtype: torch.Tensor
    """
    check(skeleton_state)
    t, q, s = split(skeleton_state)

    # Assuming quaternionToRotationMatrix is implemented elsewhere
    rot_mat = quaternion.to_rotation_matrix(q)
    linear = rot_mat * s.unsqueeze(-2).expand_as(rot_mat)
    affine = torch.cat((linear, t.unsqueeze(-1)), -1)

    last_row = (
        torch.tensor([0, 0, 0, 1], device=skeleton_state.device)
        .unsqueeze(0)
        .expand(*affine.shape[:-2], 1, 4)
    )
    result = torch.cat((affine, last_row), -2)
    return result


def match_leading_dimensions(
    t_left: torch.Tensor, t_right: torch.Tensor
) -> torch.Tensor:
    """
    Match the leading dimensions of two tensors.

    :parameter t_left: The first tensor.
    :type t_left: torch.Tensor
    :parameter t_right: The second tensor.
    :type t_right: torch.Tensor
    :return: The first tensor with its leading dimensions matched to the second tensor.
    :rtype: torch.Tensor
    """
    if t_right.dim() < t_left.dim():
        raise ValueError(
            "First tensor can't have larger dimensionality than the second."
        )

    while t_left.dim() < t_right.dim():
        t_left = t_left.unsqueeze(0)

    t_left = t_left.expand(*t_right.shape[:-1], t_left.shape[-1])
    return t_left


def multiply(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two skeleton states.

    :parameter s1: The first skeleton state.
    :type s1: torch.Tensor
    :parameter s2: The second skeleton state.
    :type s2: torch.Tensor
    :return: The product of the two skeleton states.
    :rtype: torch.Tensor
    """
    check(s1)
    check(s2)
    s1 = match_leading_dimensions(s1, s2)

    t1, q1, s1_ = split(s1)
    t2, q2, s2_ = split(s2)

    # Assuming quaternionRotateVector and quaternionMultiply are implemented elsewhere
    t_res = t1 + quaternion.rotate_vector(q1, s1_.expand_as(t2) * t2)
    s_res = s1_ * s2_
    q_res = quaternion.multiply(q1, q2)

    return torch.cat((t_res, q_res, s_res), -1)


def inverse(skeleton_states: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a skeleton state.

    :parameter skeleton_states: The skeleton state to invert.
    :type skeleton_states: torch.Tensor
    :return: The inverted skeleton state.
    :rtype: torch.Tensor
    """
    t, q, s = split(skeleton_states)
    q_inv = quaternion.inverse(q)
    s_inv = torch.reciprocal(s)

    return torch.cat((-s_inv * quaternion.rotate_vector(q_inv, t), q_inv, s_inv), -1)


def transform_points(skel_state: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Transform 3d points by the transform represented by the skeleton state.

    :parameter skel_state: The skeleton state to use for transformation.
    :type skel_state: torch.Tensor
    :parameter points: The points to transform.
    :type points: torch.Tensor
    :return: The transformed points.
    :rtype: torch.Tensor
    """
    check(skel_state)
    if points.dim() < 1 or points.shape[-1] != 3:
        raise ValueError("Points tensor should have last dimension 3.")
    skel_state = match_leading_dimensions(skel_state, points)

    t, q, s = split(skel_state)
    return t + quaternion.rotate_vector(q, s.expand_as(points) * points)


def identity(
    size: Sequence[int] | None = None, device: torch.device | None = None
) -> torch.Tensor:
    """
    Returns a skeleton state representing the identity transform.

    :parameter sizes: The size of each dimension in the output tensor. Defaults to None, which means the output will be a 1D tensor with 8 elements.
    :type sizes: list[int], optional
    :parameter device: The device on which to create the tensor. Defaults to None, which means the tensor will be created on the default device.
    :type device: torch.device, optional
    :return: The identity skeleton state.
    :rtype: torch.Tensor
    """
    zeros = (
        torch.zeros(*size, 3, device=device) if size else torch.zeros(3, device=device)
    )
    ones = torch.ones(*size, 1, device=device) if size else torch.ones(1, device=device)
    q_identity = quaternion.identity(size=size, device=device)

    return torch.cat((zeros, q_identity, ones), -1)


def blend(
    skel_states: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Blend k skeleton states with the passed-in weights.

    :parameter skel_states: The skeleton states to blend.
    :type skel_states: torch.Tensor
    :parameter weights: The weights to use, if not provided, weights are assumed to be all 1s
    :return: The blended skeleton state.
    """
    t, q, s = split(skel_states)
    weights = quaternion.check_and_normalize_weights(q, weights)
    t_blend = (weights.unsqueeze(-1).expand_as(t) * t).sum(-2)
    q_blend = quaternion.blend(q, weights)
    s_blend = (weights.unsqueeze(-1).expand_as(s) * s).sum(-2)
    return torch.cat((t_blend, q_blend, s_blend), -1)


def slerp(s0: torch.Tensor, s1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two skeleton states.

    :parameter s0: The first skeleton state.
    :parameter s1: The second skeleton state.
    :parameter t: The interpolation factor, where 0 <= t <= 1.  t=0 corresponds to s0, t=1 corresponds to s1.
    :return: The interpolated skeleton state.
    """
    check(s0)
    check(s1)

    t = match_leading_dimensions(t, s0)
    t0, q0, s0 = split(s0)
    t1, q1, s1 = split(s1)

    s = (1 - t) * s0 + t * s1
    q = quaternion.slerp(q0, q1, t)
    t = (1 - t) * t0 + t * t1
    return torch.cat((t, q, s), -1)


def from_matrix(matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert 4x4 matrices to skeleton states.  Assumes that the scale is uniform.

    Args:
        matrices (torch.Tensor): A tensor of 4x4 matrices.

    Returns:
        torch.Tensor: The corresponding skeleton states.
    """
    if matrices.dim() < 2 or matrices.shape[-1] != 4 or matrices.shape[-2] != 4:
        raise ValueError("Expected a tensor of 4x4 matrices")
    initial_shape = matrices.shape
    if matrices.dim() == 2:
        matrices = matrices.unsqueeze(0)
    else:
        matrices = matrices.flatten(0, -3)
    linear = matrices[..., :3, :3]
    translations = matrices[..., :3, 3]
    U, S, Vt = torch.linalg.svd(linear)
    scales = S[..., :1]
    rotation_matrices = torch.bmm(U, Vt)
    quaternions = quaternion.from_rotation_matrix(rotation_matrices)
    result = torch.cat((translations, quaternions, scales), -1)
    result_shape = list(initial_shape[:-2]) + [8]
    return result.reshape(result_shape)
