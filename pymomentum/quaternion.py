# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Quaternion Utilities
====================

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

from collections.abc import Sequence

import torch
from pymomentum.backend import torch_quaternion
from pymomentum.backend.selection import resolve_backend

# pyre-strict


def _get_triton_quaternion():  # type: ignore[no-untyped-def]
    """Lazy import to avoid circular dependency with pymomentum.backend."""
    from pymomentum.backend import triton_quaternion

    return triton_quaternion


def check(q: torch.Tensor) -> None:
    """
    Check if a tensor represents a quaternion.

    :parameter q: A tensor representing a quaternion.
    """
    return torch_quaternion.check(q)


def split(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a quaternion into its scalar and vector parts.

    :parameter q: A tensor representing a quaternion.
    :return: The scalar and vector parts of the quaternion.
    """
    return torch_quaternion.split(q)


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
    return torch_quaternion.multiply(q1, q2)


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
    return torch_quaternion.multiply_assume_normalized(q1, q2)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The normalized quaternion.
    """
    return torch_quaternion.normalize(q)


def conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Conjugate a quaternion.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The conjugate.
    """
    return torch_quaternion.conjugate(q)


def inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a quaternion.

    Uses numerical clamping to avoid division by very small numbers,
    improving numerical stability for near-zero quaternions.

    :parameter q: A quaternion ((x, y, z), w)).
    :return: The inverse.
    """
    return torch_quaternion.inverse(q)


def quaternion_to_xyz_euler(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    :param eps: a small number to avoid calling asin(1) or asin(-1).
        Should not be smaller than 1e-6 as this can cause NaN gradients for some models.
    """
    return torch_quaternion.quaternion_to_xyz_euler(q, eps)


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
    return torch_quaternion.rotate_vector(q, v)


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
    return torch_quaternion.rotate_vector_assume_normalized(q, v)


def to_rotation_matrix_assume_normalized(
    q: torch.Tensor, backend: str = "torch"
) -> torch.Tensor:
    """
    Convert quaternions to 3x3 rotation matrices.

    :parameter q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
    :parameter backend: Backend to use. "auto" opts into the experimental Triton backend on CUDA float32, else torch.
    :return: (nBatch x k x 3 x 3) tensor with 3x3 rotation matrices.
    """
    backend = resolve_backend(
        backend,
        q,
        use_double_precision=q.dtype == torch.float64,
    )
    if backend == "triton":
        return _get_triton_quaternion().to_rotation_matrix_assume_normalized(q)
    if backend != "torch":
        raise ValueError(f"Unsupported quaternion backend: {backend}")
    return torch_quaternion.to_rotation_matrix_assume_normalized(q)


def to_rotation_matrix(
    q: torch.Tensor, backend: str = "torch", eps: float = 1e-12
) -> torch.Tensor:
    """
    Convert quaternions to 3x3 rotation matrices.

    :parameter q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
    :parameter backend: Backend to use. "auto" opts into the experimental Triton backend on CUDA float32, else torch.
    :parameter eps: Minimum quaternion norm used during normalization.
    :return: (nBatch x k x 3 x 3) tensor with 3x3 rotation matrices.
    """
    backend = resolve_backend(
        backend,
        q,
        use_double_precision=q.dtype == torch.float64,
    )
    if backend == "triton":
        return _get_triton_quaternion().to_rotation_matrix(q, eps=eps)
    if backend != "torch":
        raise ValueError(f"Unsupported quaternion backend: {backend}")
    return torch_quaternion.to_rotation_matrix(q, eps=eps)


def identity(
    size: Sequence[int] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a quaternion identity tensor.

    :parameter size: A tuple of integers representing the size of the quaternion tensor.
    :parameter device: The device on which to create the tensor.
    :return: A quaternion identity tensor with the specified sizes and device.
    """
    return torch_quaternion.identity(size, device, dtype)


def from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert an axis-angle tensor to a quaternion.

    :parameter axis_angle: A tensor of shape (..., 3) representing the axis-angle.
    :return: A tensor of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    return torch_quaternion.from_axis_angle(axis_angle)


def euler_xyz_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler XYZ angles to a quaternion.

    This function converts XYZ Euler angles to quaternions.
    The rotation order is X-Y-Z, meaning first rotate around X-axis, then Y-axis,
    then Z-axis.

    :parameter euler_xyz: A tensor of shape (..., 3) representing the Euler XYZ angles
                         in order [roll, pitch, yaw].
    :return: A tensor of shape (..., 4) representing the quaternion in ((x, y, z), w) format.
    """
    return torch_quaternion.euler_xyz_to_quaternion(euler_xyz)


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
    return torch_quaternion.euler_zyx_to_quaternion(euler_zyx)


def from_rotation_matrix(
    matrices: torch.Tensor, eta: float = 1e-6, backend: str = "torch"
) -> torch.Tensor:
    """
    Convert a rotation matrix to a quaternion using numerically stable method.

    This implementation uses the robust algorithm that computes all four quaternion
    component candidates and selects the best-conditioned one, ensuring numerical
    stability across all rotation matrix configurations.

    :parameter matrices: A tensor of shape (..., 3, 3) representing the rotation matrices.
    :parameter eta: Numerical precision threshold (unused, kept for compatibility).
    :parameter backend: Backend to use. "auto" opts into the experimental Triton backend on CUDA float32, else torch.
    :return: A tensor of shape (..., 4) representing the quaternions in ((x, y, z), w) format.
    """
    backend = resolve_backend(
        backend,
        matrices,
        use_double_precision=matrices.dtype == torch.float64,
    )
    if backend == "triton":
        return _get_triton_quaternion().from_rotation_matrix(matrices)
    if backend != "torch":
        raise ValueError(f"Unsupported quaternion backend: {backend}")
    return torch_quaternion.from_rotation_matrix(matrices, eta)


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
    return torch_quaternion.check_and_normalize_weights(quaternions, weights_in)


def blend(
    quaternions: torch.Tensor, weights_in: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Blend multiple quaternions together using the method described in
    https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    and http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf.

    Uses SVD with a custom backward pass that only differentiates through the
    top singular triplet, providing stable gradients for training.

    :parameter quaternions: A tensor of shape (..., k, 4) representing the quaternions to blend.
    :parameter weights_in: An optional tensor of shape (..., k) representing the weights for each quaternion.
                       If not provided, all quaternions will be weighted equally.
    :return: A tensor of shape (..., 4) representing the blended quaternion.
    """
    return torch_quaternion.blend(quaternions, weights_in)


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Perform spherical linear interpolation (slerp) between two quaternions.

    :parameter q0: The starting quaternion.
    :parameter q1: The ending quaternion.
    :parameter t: The interpolation parameter, where 0 <= t <= 1.  t=0 corresponds to q0, t=1 corresponds to q1.
    :return: The interpolated quaternion.
    """
    return torch_quaternion.slerp(q0, q1, t)


def from_two_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Construct a quaternion that rotates one vector into another.

    :parameter v1: The initial vector.
    :parameter v2: The target vector.
    :return: A quaternion representing the rotation from v1 to v2.
    """
    return torch_quaternion.from_two_vectors(v1, v2)


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
    return torch_quaternion.normalize_backprop(q, grad)


def rotate_vector_backprop(
    q: torch.Tensor, v: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return torch_quaternion.rotate_vector_backprop(q, v, grad)


def rotate_vector_backprop_assume_normalized(
    q: torch.Tensor, v: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return torch_quaternion.rotate_vector_backprop_assume_normalized(q, v, grad)


def multiply_backprop(
    q1: torch.Tensor, q2: torch.Tensor, grad_q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return torch_quaternion.multiply_backprop(q1, q2, grad_q)


def multiply_backprop_assume_normalized(
    q1: torch.Tensor, q2: torch.Tensor, grad_q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return torch_quaternion.multiply_backprop_assume_normalized(q1, q2, grad_q)
