# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
Skeleton State Backend for PyMomentum

This module provides efficient forward kinematics and skinning operations using
the skeleton state representation (8 parameters per joint: translation, quaternion, scale).

The skeleton state representation is more compact than the TRS backend and uses
quaternions for rotation, making it suitable for applications requiring smooth
interpolation and fewer parameters.

Performance Notes:
This backend matches the behavior of the C++ Momentum code, ensuring consistency
with the reference implementation. However, the TRS backend may be 25-50% faster
in PyTorch due to not requiring quaternion normalization operations, though the
exact performance difference may vary depending on the specific use case.

Key Functions:
- global_skel_state_from_local_skel_state: Forward kinematics from local to global joint states
- skin_points_from_skel_state: Linear blend skinning using skeleton states
- local_skel_state_from_joint_params: Convert joint parameters to local states

Related Modules:
- trs_backend: Alternative backend using separate translation/rotation/scale tensors
- skel_state: Core skeleton state operations and utilities
- quaternion: Quaternion math operations used by this backend
"""

from typing import List, Tuple

import torch as th

from pymomentum import quaternion, skel_state, trs
from pymomentum.backend.trs_backend import unpose_from_global_joint_state


@th.jit.script
def local_skel_state_from_joint_params(
    joint_params: th.Tensor,
    joint_offset: th.Tensor,
    joint_quat_rotation: th.Tensor,
) -> th.Tensor:
    """
    Convert joint parameters to local skeleton state representation.

    This function transforms 7-parameter joint representation (translation, euler angles, log-scale)
    into the 8-parameter skeleton state format (translation, quaternion, scale). The skeleton state
    representation uses quaternions for rotation and linear scale factors.

    Parameter Transformation:
    - Translation: joint_params[:,:,:3] + joint_offset -> direct translation
    - Rotation: euler_xyz angles -> quaternion, composed with joint pre-rotation
    - Scale: log2(scale) -> linear scale via exp2() transformation

    Args:
        joint_params: Joint parameters, shape (batch_size, num_joints, 7).
            Each joint has [tx, ty, tz, euler_x, euler_y, euler_z, log2_scale] parameters.
        joint_offset: Per-joint translation offset, shape (num_joints, 3).
            Static offset applied to each joint's local translation.
        joint_quat_rotation: Per-joint rotation offset as quaternions, shape (num_joints, 4).
            Static rotation [qx, qy, qz, qw] applied to each joint's local rotation.

    Returns:
        local_skel_state: Local skeleton state, shape (batch_size, num_joints, 8).
            Each joint contains [tx, ty, tz, qx, qy, qz, qw, s] parameters.

    Note:
        The quaternion format follows [qx, qy, qz, qw] convention (vector-first).
        Scale values are exponentiated from log2 space to linear space.
    """
    t = joint_offset[None, :] + joint_params[:, :, :3]
    q = quaternion.multiply(
        joint_quat_rotation[None],
        quaternion.euler_xyz_to_quaternion(joint_params[:, :, 3:6]),
    )
    s = th.exp2(joint_params[:, :, 6:])

    return th.cat([t, q, s], dim=-1)


@th.jit.script
def global_skel_state_from_local_skel_state_impl(
    local_skel_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    save_intermediate_results: bool = True,
    use_double_precision: bool = True,
) -> Tuple[th.Tensor, List[th.Tensor]]:
    """
    Compute global skeleton state from local joint transformations using forward kinematics.

    This function implements forward kinematics (FK) using prefix multiplication for efficient
    parallel computation of joint transformations. Each joint's local transformation is composed
    with its parent's global transformation to produce the joint's global transformation.

    The skeleton state uses an 8-parameter representation per joint: [translation, quaternion, scale]
    - translation (3D): local translation vector [tx, ty, tz]
    - quaternion (4D): rotation quaternion [qx, qy, qz, qw] (normalized)
    - scale (1D): uniform scale factor [s]

    Forward Kinematics Algorithm:
    For each joint j with parent p in the kinematic hierarchy:
        global_state_j = global_state_p ⊙ local_state_j

    Where ⊙ represents similarity transformation composition:
    - t_global = t_parent + s_parent * rotate_by_quaternion(q_parent, t_local)
    - q_global = quaternion_multiply(q_parent, q_local)
    - s_global = s_parent * s_local

    Args:
        local_skel_state: Local joint transformations, shape (batch_size, num_joints, 8).
            Each joint state contains [tx, ty, tz, qx, qy, qz, qw, s] parameters.
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs that define
            the traversal order for the kinematic tree. This ordering enables efficient
            parallel computation while respecting parent-child dependencies.
        save_intermediate_results: If True, saves intermediate joint states during the
            forward pass for use in backpropagation. Set to False for inference-only
            computations to reduce memory usage.
        use_double_precision: If True, performs computations in float64 for improved
            numerical stability. Recommended for deep kinematic chains to minimize
            accumulated floating-point errors.

    Returns:
        global_skel_state: Global joint transformations, shape (batch_size, num_joints, 8).
            Each joint contains the composed transformation from root to that joint.
        intermediate_results: List of intermediate joint states from the forward pass.
            Required for efficient gradient computation during backpropagation.
            Empty if save_intermediate_results=False.

    Note:
        This function is JIT-compiled for performance. The prefix multiplication approach
        allows vectorized batch computation while maintaining kinematic chain dependencies.

    See Also:
        :func:`global_skel_state_from_local_skel_state`: User-facing wrapper function
        :func:`local_skel_state_from_joint_params`: Convert joint parameters to local states
    """
    dtype = local_skel_state.dtype
    intermediate_results: List[th.Tensor] = []
    if use_double_precision:
        global_skel_state = local_skel_state.clone().double()
    else:
        global_skel_state = local_skel_state.clone()
    for prefix_mul_index in prefix_mul_indices:
        source = prefix_mul_index[0]
        target = prefix_mul_index[1]

        state1 = global_skel_state.index_select(-2, target)
        state2 = global_skel_state.index_select(-2, source)

        if save_intermediate_results:
            intermediate_results.append(state2.clone())

        global_skel_state.index_copy_(-2, source, skel_state.multiply(state1, state2))

    return (global_skel_state.to(dtype), intermediate_results)


def global_skel_state_from_local_skel_state_no_grad(
    local_skel_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    save_intermediate_results: bool = True,
    use_double_precision: bool = True,
) -> Tuple[th.Tensor, List[th.Tensor]]:
    """
    Compute global skeleton state without gradient tracking.

    This is a convenience wrapper around global_skel_state_from_local_skel_state_impl
    that explicitly disables gradient computation using torch.no_grad(). Useful for
    inference-only forward passes to reduce memory usage.

    Args:
        local_skel_state: Local joint transformations, shape (batch_size, num_joints, 8)
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs
        save_intermediate_results: Whether to save intermediate states for backprop
        use_double_precision: Whether to use float64 for numerical stability

    Returns:
        global_skel_state: Global joint transformations, shape (batch_size, num_joints, 8)
        intermediate_results: List of intermediate joint states from forward pass

    See Also:
        :func:`global_skel_state_from_local_skel_state_impl`: Implementation function
    """
    with th.no_grad():
        outputs = global_skel_state_from_local_skel_state_impl(
            local_skel_state,
            prefix_mul_indices,
            save_intermediate_results=save_intermediate_results,
            use_double_precision=use_double_precision,
        )
    return outputs


# @th.jit.script
def global_skel_state_from_local_skel_state_backprop(
    global_skel_state: th.Tensor,
    grad_global_skel_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    intermediate_results: List[th.Tensor],
    use_double_precision: bool = True,
) -> th.Tensor:
    """
    Compute gradients for local skeleton state through backpropagation.

    This function implements the backward pass for forward kinematics, computing
    gradients of the loss with respect to local joint states given gradients
    with respect to global joint states.

    The backpropagation uses the intermediate results saved during the forward
    pass to efficiently compute gradients without recomputing the full forward
    kinematics chain.

    Gradient Flow:
    For each joint j with parent p, the backward pass computes:
    ∂L/∂local_j = ∂L/∂global_j * ∂global_j/∂local_j

    Where the Jacobian ∂global_j/∂local_j depends on the parent's global state
    and is computed using the chain rule through similarity transformation composition.

    Args:
        global_skel_state: Global joint states from forward pass, shape (batch_size, num_joints, 8).
            These states are modified in-place during backpropagation.
        grad_global_skel_state: Gradients w.r.t. global joint states, shape (batch_size, num_joints, 8).
            Input gradients from downstream computations (e.g., skinning loss).
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs that defined
            the forward pass traversal order. Backprop processes these in reverse.
        intermediate_results: List of intermediate joint states saved during forward pass.
            Required to compute accurate gradients without numerical drift.
        use_double_precision: If True, performs computations in float64 for numerical stability.
            Recommended for deep kinematic chains and precise gradient computation.

    Returns:
        grad_local_skel_state: Gradients w.r.t. local joint states, shape (batch_size, num_joints, 8).
            These gradients can be used to update joint parameters via optimization.

    Note:
        This function processes the kinematic chain in reverse order of the forward pass,
        accumulating gradients from children to parents while reconstructing intermediate states.

    See Also:
        :func:`global_skel_state_from_local_skel_state_impl`: Forward pass implementation
    """
    dtype = global_skel_state.dtype
    if use_double_precision:
        grad_local_skel_state = grad_global_skel_state.clone().double()
        global_skel_state = global_skel_state.clone().double()
    else:
        grad_local_skel_state = grad_global_skel_state.clone()
        global_skel_state = global_skel_state.clone()

    for prefix_mul_index, state_source_original in list(
        zip(prefix_mul_indices, intermediate_results)
    )[::-1]:
        source = prefix_mul_index[0]
        target = prefix_mul_index[1]

        state_target = global_skel_state.index_select(-2, target)

        # forward
        # This is for checking that the numerical error is in the safe
        # range, but it's not completely abnormal to see large numerical
        # error during backpropagation at the beginning of training.
        # state_source = global_skel_state[:, source]
        # state_source_expect = sim3_multiplication(
        #     state_target, state_source_original
        # )
        # assert th.allclose(
        #     state_source, state_source_expect, atol=1e-5, rtol=1e-5
        # )

        # backward
        grad_target_accum, grad_source_original = skel_state.multiply_backprop(
            state1=state_target,
            state2=state_source_original,
            grad_state=grad_local_skel_state.index_select(-2, source),
        )

        # setup the reduced gradients
        grad_local_skel_state.index_copy_(-2, source, grad_source_original)
        grad_local_skel_state.index_add_(-2, target, grad_target_accum)
        # setup the reduced KC
        global_skel_state.index_copy_(-2, source, state_source_original)

    return grad_local_skel_state.to(dtype)


class GlobalSkelStateFromLocalSkelStateJIT(th.autograd.Function):
    """
    PyTorch autograd function for differentiable forward kinematics using skeleton states.

    This class implements automatic differentiation for the forward kinematics operation,
    allowing gradients to flow from global joint states back to local joint states.
    The forward pass computes global states efficiently while saving intermediate results
    for the backward pass.

    Note:
        This class is used internally by global_skel_state_from_local_skel_state when
        not in JIT mode. It provides gradient computation capabilities that are not
        available in pure JIT-compiled functions.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(
        local_skel_state: th.Tensor,
        prefix_mul_indices: List[th.Tensor],
    ) -> Tuple[th.Tensor, List[th.Tensor]]:
        """
        Compute forward pass for differentiable forward kinematics.

        Args:
            local_skel_state: Local joint transformations, shape (batch_size, num_joints, 8)
            prefix_mul_indices: List of [child_index, parent_index] tensor pairs

        Returns:
            Tuple of (global_skel_state, intermediate_results)
        """
        return global_skel_state_from_local_skel_state_no_grad(
            local_skel_state,
            prefix_mul_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    # pyre-ignore[2]
    def setup_context(ctx, inputs, outputs) -> None:
        """
        Save context for backward pass.

        Args:
            ctx: Context object for saving tensors and data
            inputs: Tuple of (local_skel_state, prefix_mul_indices)
            outputs: Tuple of (global_skel_state, intermediate_results)
        """
        (
            _,
            prefix_mul_indices,
        ) = inputs
        (
            global_skel_state,
            intermediate_results,
        ) = outputs
        # need to clone as it's modified in-place
        ctx.save_for_backward(global_skel_state.clone())
        ctx.intermediate_results = intermediate_results
        ctx.prefix_mul_indices = prefix_mul_indices

    @staticmethod
    # pyre-ignore[14]
    def backward(
        # pyre-ignore[2]
        ctx,
        grad_global_skel_state: th.Tensor,
        _0,
    ) -> Tuple[th.Tensor, None]:
        (global_skel_state,) = ctx.saved_tensors

        intermediate_results = ctx.intermediate_results
        prefix_mul_indices = ctx.prefix_mul_indices

        grad_local_state = global_skel_state_from_local_skel_state_backprop(
            global_skel_state,
            grad_global_skel_state,
            prefix_mul_indices,
            intermediate_results,
        )
        return grad_local_state, None


def global_skel_state_from_local_skel_state(
    local_skel_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
) -> th.Tensor:
    """
    Compute global skeleton state from local joint transformations (user-facing wrapper).

    This is the main entry point for forward kinematics using skeleton states. It automatically
    selects between JIT-compiled and autograd-enabled implementations based on the execution context.

    Args:
        local_skel_state: Local joint transformations, shape (batch_size, num_joints, 8).
                         Each joint contains [tx, ty, tz, qx, qy, qz, qw, s] parameters.
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs defining
                           the kinematic hierarchy traversal order.

    Returns:
        global_skel_state: Global joint transformations, shape (batch_size, num_joints, 8).
                          Each joint contains the composed transformation from root to that joint.

    Note:
        When called within torch.jit.script or torch.jit.trace context, uses the JIT-compiled
        implementation for maximum performance. Otherwise, uses the autograd-enabled version
        for gradient computation.

    See Also:
        :func:`global_skel_state_from_local_skel_state_impl`: JIT implementation
        :func:`local_skel_state_from_joint_params`: Convert joint parameters to local states
    """

    if th.jit.is_tracing() or th.jit.is_scripting():
        global_skel_state, _ = global_skel_state_from_local_skel_state_impl(
            local_skel_state, prefix_mul_indices
        )
        return global_skel_state
    else:
        global_skel_state, _ = GlobalSkelStateFromLocalSkelStateJIT.apply(
            local_skel_state,
            prefix_mul_indices,
        )
        return global_skel_state


@th.jit.script
def skin_points_from_skel_state(
    template: th.Tensor,
    global_skel_state: th.Tensor,
    binded_skel_state_inv: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
) -> th.Tensor:
    """
    Apply linear blend skinning (LBS) to points using skeleton state transformations.

    This function deforms template points by blending joint transformations according
    to skinning weights. Each point is influenced by multiple joints, with the final
    position computed as a weighted average of joint-transformed positions.

    Linear Blend Skinning Formula:
    For each vertex v_i with position p_i in template:
        skinned_p_i = Σ_j w_ij * T_j * (T_bind_j^(-1) * p_i)

    Where:
    - w_ij: skinning weight of joint j on vertex i (sum to 1 per vertex)
    - T_j: global transformation of joint j
    - T_bind_j^(-1): inverse bind pose transformation of joint j

    The bind pose represents the joint configuration when skinning weights were defined.
    The inverse bind pose transformation removes the original joint influence before
    applying the current joint transformation.

    Args:
        template: Template vertex positions, shape (batch_size, num_vertices, 3).
            These are the rest pose positions before deformation.
        global_skel_state: Global joint transformations, shape (batch_size, num_joints, 8).
            Each joint contains [tx, ty, tz, qx, qy, qz, qw, s] parameters.
        binded_skel_state_inv: Inverse bind pose transformations, shape (num_joints, 8).
            Static transformations that represent the joint configuration during binding.
        skin_indices_flattened: Joint indices for skinning, shape (num_influences,).
            Flattened list of joint indices that influence vertices.
        skin_weights_flattened: Skinning weights, shape (num_influences,).
            Corresponding weights for each joint influence (normalized per vertex).
        vert_indices_flattened: Vertex indices for skinning, shape (num_influences,).
            Flattened list of vertex indices corresponding to joint influences.

    Returns:
        skinned_points: Deformed vertex positions, shape (batch_size, num_vertices, 3).
            Points transformed according to current joint poses and skinning weights.

    Note:
        This function is JIT-compiled for performance. The flattened indices allow
        efficient vectorized computation of skinning influences across all vertices.

    See Also:
        :func:`skin_oriented_points_from_skel_state`: Skinning for oriented points (gaussians)
        :func:`global_skel_state_from_local_skel_state`: Forward kinematics for joint states
    """
    assert template.shape[-1] == 3
    while template.ndim < global_skel_state.ndim:
        template = template.unsqueeze(0)

    template = template.expand(
        list(global_skel_state.shape[:-2]) + list(template.shape[-2:])
    )

    joint_state = skel_state.multiply(
        global_skel_state,
        binded_skel_state_inv,
    )

    skinned = th.zeros_like(template)
    skinned = skinned.index_add(
        -2,
        vert_indices_flattened,
        skel_state.transform_points(
            th.index_select(joint_state, -2, skin_indices_flattened),
            th.index_select(template, -2, vert_indices_flattened),
        )
        * skin_weights_flattened[None, :, None],
    )
    return skinned


@th.jit.script
def skin_oriented_points_from_skel_state(
    means: th.Tensor,
    quaternions: th.Tensor,
    global_skel_state: th.Tensor,
    binded_skel_state_inv: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """
    Apply linear blend skinning to oriented points (Gaussian splatting) using skeleton states.

    This function extends standard LBS to handle oriented points, which have both position
    and orientation. Each oriented point is influenced by multiple joints, with both its
    position and orientation blended according to skinning weights.

    For orientations, rotation matrices are linearly interpolated (LERP) and then converted
    back to quaternions. This is different from spherical linear interpolation (SLERP) but
    provides smoother blending for skinning applications.

    Args:
        means: Point positions, shape (batch_size, num_points, 3).
            Center positions of the oriented points (e.g., Gaussian means).
        quaternions: Point orientations, shape (batch_size, num_points, 4).
            Orientations as quaternions in [qx, qy, qz, qw] format.
        global_skel_state: Global joint transformations, shape (batch_size, num_joints, 8).
            Each joint contains [tx, ty, tz, qx, qy, qz, qw, s] parameters.
        binded_skel_state_inv: Inverse bind pose transformations, shape (num_joints, 8).
            Static transformations representing the joint configuration during binding.
        skin_indices_flattened: Joint indices for skinning, shape (num_influences,).
            Flattened list of joint indices that influence points.
        skin_weights_flattened: Skinning weights, shape (num_influences,).
            Corresponding weights for each joint influence (normalized per point).
        vert_indices_flattened: Point indices for skinning, shape (num_influences,).
            Flattened list of point indices corresponding to joint influences.

    Returns:
        skinned_means: Deformed point positions, shape (batch_size, num_points, 3).
            Transformed positions according to current joint poses and skinning weights.
        skinned_quaternions: Blended point orientations, shape (batch_size, num_points, 4).
            Orientations blended using rotation matrix interpolation and converted back to quaternions.

    Note:
        This function is commonly used for Gaussian splatting applications where each
        Gaussian has both a mean position and orientation that need to be skinned.
        The rotation blending uses linear interpolation in matrix space for stability.

    See Also:
        :func:`skin_points_from_skel_state`: Standard LBS for points without orientation
        :func:`global_skel_state_from_local_skel_state`: Forward kinematics for joint states
    """
    assert means.shape[-1] == 3
    assert quaternions.shape[-1] == 4
    while means.ndim < global_skel_state.ndim:
        means = means.unsqueeze(0)

    means = means.expand(list(global_skel_state.shape[:-2]) + list(means.shape[-2:]))

    joint_state = skel_state.multiply(
        global_skel_state,
        binded_skel_state_inv,
    )

    sim3_transforms = th.index_select(joint_state, -2, skin_indices_flattened)
    t, q, s = skel_state.split(sim3_transforms)
    r = quaternion.to_rotation_matrix(q)

    means_flattened = th.index_select(means, -2, vert_indices_flattened)

    skinned_means = th.zeros_like(means)
    skinned_means = skinned_means.index_add(
        -2,
        vert_indices_flattened,
        (s * quaternion.rotate_vector(q, means_flattened) + t)
        * skin_weights_flattened[None, :, None],
    )

    lerp_rotations = th.zeros(
        quaternions.shape[:-1] + (3, 3),
        dtype=quaternions.dtype,
        device=quaternions.device,
    )
    lerp_rotations = lerp_rotations.index_add(
        -3,
        vert_indices_flattened,
        r * skin_weights_flattened[None, :, None, None],
    )
    lerp_quaternions = quaternion.from_rotation_matrix(lerp_rotations)
    lerp_quaternions = quaternion.normalize(lerp_quaternions)
    skinned_quaternions = quaternion.multiply_assume_normalized(
        quaternions, lerp_quaternions
    )
    return skinned_means, skinned_quaternions


def unpose_from_momentum_global_joint_state(
    verts: th.Tensor,
    global_joint_state: th.Tensor,
    binded_joint_state_inv: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
    with_high_precision: bool = True,
) -> th.Tensor:
    """
    The inverse function of skinning().
    WARNING: the precision is low...

    Args:
        verts: [batch_size, num_verts, 3]
        global_joint_state (th.Tensor): (B, J, 8)
        binded_joint_state (th.Tensor): (J, 8)
        skin_indices_flattened: (N, ) LBS skinning nbr joint indices
        skin_weights_flattened: (N, ) LBS skinning nbr joint weights
        vert_indices_flattened: (N, ) LBS skinning nbr corresponding vertex indices
        with_high_precision: if True, use high precision solver (LDLT), but requires a cuda device sync
    """
    t, r, s = trs.from_skeleton_state(global_joint_state)
    t0, r0, _ = trs.from_skeleton_state(binded_joint_state_inv)

    return unpose_from_global_joint_state(
        verts,
        t,
        r,
        s,
        t0,
        r0,
        skin_indices_flattened,
        skin_weights_flattened,
        vert_indices_flattened,
        with_high_precision,
    )
