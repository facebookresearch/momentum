# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Skeleton State Utilities (NumPy)
================================

NumPy implementation of skeleton state utilities for PyMomentum.

This module mirrors :mod:`skel_state` but uses NumPy arrays instead of PyTorch tensors.
See :mod:`skel_state` for full documentation on the skeleton state format and operations.

A skeleton state is a compact 8-dimensional representation of a 3D transformation
that encodes translation, rotation, and uniform scale. The 8 components are organized as:

- **Translation (3 components)**: tx, ty, tz - 3D position offset
- **Rotation (4 components)**: rx, ry, rz, rw - quaternion rotation (x, y, z, w)
- **Scale (1 component)**: s - uniform scale factor
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pymomentum import quaternion_np as quaternion

# pyre-strict


def check(skel_state: NDArray) -> None:
    """
    Check if the skeleton state has the correct shape.

    :parameter skel_state: The skeleton state to check.
    :raises ValueError: If the skeleton state does not have the correct shape.
    """
    if skel_state.shape[-1] != 8:
        raise ValueError(
            "Expected skeleton state to have last dimension 8 (tx, ty, tz, rx, ry, rz, rw, s)"
        )


def split(skel_state: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    Split a skeleton state into translation, rotation, and scale components.

    :parameter skel_state: The skeleton state to split.
    :return: A tuple of arrays (translation, rotation, scale).
    """
    check(skel_state)
    return skel_state[..., :3], skel_state[..., 3:7], skel_state[..., 7:]


def from_translation(translation: NDArray) -> NDArray:
    """
    Create a skeleton state from translation.

    :parameter translation: The translation component.
    :return: The skeleton state.
    """
    return np.concatenate(
        (
            translation,
            quaternion.identity(
                size=translation.shape[:-1],
                dtype=translation.dtype,
            ),
            np.ones((*translation.shape[:-1], 1), dtype=translation.dtype),
        ),
        axis=-1,
    )


def from_quaternion(rotation: NDArray) -> NDArray:
    """
    Create a skeleton state from rotation.

    :parameter rotation: The rotation component.
    :return: The skeleton state.
    """
    return np.concatenate(
        (
            np.zeros((*rotation.shape[:-1], 3), dtype=rotation.dtype),
            rotation,
            np.ones((*rotation.shape[:-1], 1), dtype=rotation.dtype),
        ),
        axis=-1,
    )


def from_scale(scale: NDArray) -> NDArray:
    """
    Create a skeleton state from scale.

    :parameter scale: The scale component.
    :return: The skeleton state.
    """
    return np.concatenate(
        (
            np.zeros((*scale.shape[:-1], 3), dtype=scale.dtype),
            quaternion.identity(
                size=scale.shape[:-1],
                dtype=scale.dtype,
            ),
            scale,
        ),
        axis=-1,
    )


def to_matrix(skeleton_state: NDArray) -> NDArray:
    """
    Convert skeleton state to an array of 4x4 matrices. The matrix represents the transform from a local joint space to the world space.

    :parameter skeleton_state: The skeleton state to convert.
    :return: An array containing 4x4 matrix transforms.
    """
    check(skeleton_state)
    t, q, s = split(skeleton_state)

    rot_mat = quaternion.to_rotation_matrix(q)
    linear = rot_mat * s[..., None, :]
    affine = np.concatenate((linear, t[..., None]), axis=-1)

    last_row = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=skeleton_state.dtype).reshape(
            *([1] * (affine.ndim - 2)), 1, 4
        ),
        (*affine.shape[:-2], 1, 4),
    )
    result = np.concatenate((affine, last_row), axis=-2)
    return result


def _normalize_split_skel_state(
    components: tuple[NDArray, NDArray, NDArray],
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Normalize the quaternion component of a split skeleton state.

    :parameter components: Tuple of (translation, rotation, scale) components.
    :return: Tuple of (translation, normalized_rotation, scale) components.
    """
    t, q, s = components
    return t, quaternion.normalize(q), s


def _multiply_split_skel_states(
    skel_state1: tuple[NDArray, NDArray, NDArray],
    skel_state2: tuple[NDArray, NDArray, NDArray],
) -> NDArray:
    """
    Helper function to multiply two skeleton states from their split components.
    Assumes quaternions are already normalized.

    :parameter skel_state1: Tuple of (translation, rotation, scale) for first skeleton state.
    :parameter skel_state2: Tuple of (translation, rotation, scale) for second skeleton state.
    :return: The product of the two skeleton states.
    """
    t1, q1, s1 = skel_state1
    t2, q2, s2 = skel_state2

    t_res = t1 + s1 * quaternion.rotate_vector_assume_normalized(q1, t2)
    s_res = s1 * s2
    q_res = quaternion.multiply_assume_normalized(q1, q2)

    return np.concatenate((t_res, q_res, s_res), axis=-1)


def multiply(s1: NDArray, s2: NDArray) -> NDArray:
    """
    Multiply two skeleton states.

    :parameter s1: The first skeleton state.
    :parameter s2: The second skeleton state.
    :return: The product of the two skeleton states.
    """
    check(s1)
    check(s2)
    while s1.ndim < s2.ndim:
        s1 = np.expand_dims(s1, 0)

    return _multiply_split_skel_states(
        _normalize_split_skel_state(split(s1)), _normalize_split_skel_state(split(s2))
    )


def multiply_assume_normalized(s1: NDArray, s2: NDArray) -> NDArray:
    """
    Multiply two skeleton states.
    This is an optimized version that assumes both skeleton states are already properly
    formatted and the quaternions are normalized, skipping validation checks.

    :parameter s1: The first skeleton state.
    :parameter s2: The second skeleton state.
    :return: The product of the two skeleton states.
    """
    while s1.ndim < s2.ndim:
        s1 = np.expand_dims(s1, 0)

    return _multiply_split_skel_states(split(s1), split(s2))


def inverse(skeleton_states: NDArray) -> NDArray:
    """
    Compute the inverse of a skeleton state.

    :parameter skeleton_states: The skeleton state to invert.
    :return: The inverted skeleton state.
    """
    t, q, s = split(skeleton_states)
    q_inv = quaternion.inverse(q)
    s_inv = np.reciprocal(s)

    return np.concatenate(
        (-s_inv * quaternion.rotate_vector(q_inv, t), q_inv, s_inv), axis=-1
    )


def _transform_points_split_skel_state(
    skel_state: tuple[NDArray, NDArray, NDArray],
    points: NDArray,
) -> NDArray:
    """
    Helper function to transform points using skeleton state components.
    Assumes quaternion is already normalized.

    :parameter skel_state: Tuple of (translation, rotation, scale) components.
    :parameter points: The points to transform.
    :return: The transformed points.
    """
    t, q, s = skel_state
    return t + quaternion.rotate_vector(q, s * points)


def transform_points(skel_state: NDArray, points: NDArray) -> NDArray:
    """
    Transform 3d points by the transform represented by the skeleton state.

    :parameter skel_state: The skeleton state to use for transformation.
    :parameter points: The points to transform.
    :return: The transformed points.
    """
    check(skel_state)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("Points tensor should have last dimension 3.")

    # allow transforming multiple points with a single skel_state.
    while skel_state.ndim < points.ndim:
        skel_state = np.expand_dims(skel_state, 0)

    return _transform_points_split_skel_state(
        _normalize_split_skel_state(split(skel_state)), points
    )


def transform_points_assume_normalized(skel_state: NDArray, points: NDArray) -> NDArray:
    """
    Transform 3d points by the transform represented by the skeleton state.
    This is an optimized version that assumes the skeleton state is already properly
    formatted and the quaternion is normalized, skipping validation checks.

    :parameter skel_state: The skeleton state to use for transformation.
    :parameter points: The points to transform.
    :return: The transformed points.
    """
    # when multiplying by points, we'll allow a single skel_state to transform multiple points.
    if skel_state.ndim < points.ndim:
        skel_state = np.expand_dims(skel_state, 0)
    return _transform_points_split_skel_state(split(skel_state), points)


def identity(
    size: Sequence[int] | None = None,
    dtype: np.dtype = np.float32,
) -> NDArray:
    """
    Returns a skeleton state representing the identity transform.

    :parameter size: The size of each dimension in the output array. Defaults to None, which means the output will be a 1D array with 8 elements.
    :parameter dtype: The data type for the array.
    :return: The identity skeleton state.
    """
    shape = size or ()
    zeros = np.zeros((*shape, 3), dtype=dtype)
    ones = np.ones((*shape, 1), dtype=dtype)
    q_identity = quaternion.identity(size=size, dtype=dtype)

    return np.concatenate((zeros, q_identity, ones), axis=-1)


def blend(skel_states: NDArray, weights: NDArray | None = None) -> NDArray:
    """
    Blend k skeleton states with the passed-in weights.

    :parameter skel_states: The skeleton states to blend.
    :parameter weights: The weights to use, if not provided, weights are assumed to be all 1s
    :return: The blended skeleton state.
    """
    t, q, s = split(skel_states)
    weights = quaternion.check_and_normalize_weights(q, weights)
    t_blend = (np.broadcast_to(weights[..., None], t.shape) * t).sum(axis=-2)
    q_blend = quaternion.blend(q, weights)
    s_blend = (np.broadcast_to(weights[..., None], s.shape) * s).sum(axis=-2)
    return np.concatenate((t_blend, q_blend, s_blend), axis=-1)


def slerp(s0: NDArray, s1: NDArray, t: NDArray | float) -> NDArray:
    """
    Spherical linear interpolation between two skeleton states.

    :parameter s0: The first skeleton state.
    :parameter s1: The second skeleton state.
    :parameter t: The interpolation factor, where 0 <= t <= 1.  t=0 corresponds to s0, t=1 corresponds to s1.
    :return: The interpolated skeleton state.
    """
    check(s0)
    check(s1)

    t = np.asarray(t)
    if t.ndim < s0.ndim:
        t = np.expand_dims(t, 0)

    t0, q0, sc0 = split(s0)
    t1, q1, sc1 = split(s1)

    s = (1 - t) * sc0 + t * sc1
    q = quaternion.slerp(q0, q1, t)
    tr = (1 - t) * t0 + t * t1
    return np.concatenate((tr, q, s), axis=-1)


def from_matrix(matrices: NDArray) -> NDArray:
    """
    Convert 4x4 matrices to skeleton states.  Assumes that the scale is uniform.

    :parameter matrices: An array of 4x4 matrices.
    :return: The corresponding skeleton states.
    """
    if matrices.ndim < 2 or matrices.shape[-1] != 4 or matrices.shape[-2] != 4:
        raise ValueError("Expected an array of 4x4 matrices")
    initial_shape = matrices.shape
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    else:
        matrices = matrices.reshape(-1, 4, 4)
    linear = matrices[..., :3, :3]
    translations = matrices[..., :3, 3]
    U, S, Vt = np.linalg.svd(linear)
    scales = S[..., :1]
    rotation_matrices = np.einsum("bij,bjk->bik", U, Vt)
    quaternions = quaternion.from_rotation_matrix(rotation_matrices)
    result = np.concatenate((translations, quaternions, scales), axis=-1)
    result_shape = list(initial_shape[:-2]) + [8]
    return result.reshape(result_shape)
