# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
TRS Backend for PyMomentum

This module provides efficient forward kinematics and skinning operations using
the TRS (Translation-Rotation-Scale) representation where each transformation
component is stored separately.

The TRS representation uses separate tensors for translation (3D), rotation matrices (3x3),
and scale factors (1D), making it suitable for applications that need explicit access
to individual transformation components.

Performance Notes:
This backend is typically 25-50% faster than the skeleton state backend in PyTorch,
likely due to not requiring quaternion normalization operations. While it doesn't
match the C++ reference implementation exactly (use skel_state_backend for that),
it provides excellent performance for PyTorch-based applications.

Key Functions:
- global_trs_state_from_local_trs_state: Forward kinematics from local to global joint states
- skin_points_from_trs_state: Linear blend skinning using TRS transformations
- local_trs_state_from_joint_params: Convert joint parameters to local TRS states

Related Modules:
- skel_state_backend: Alternative backend using compact 8-parameter skeleton states
- trs: Core TRS transformation operations and utilities
"""

from typing import List, Tuple

import torch as th
from pymomentum import trs


@th.jit.script
def global_trs_state_from_local_trs_state_impl(
    local_state_t: th.Tensor,
    local_state_r: th.Tensor,
    local_state_s: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    save_intermediate_results: bool = True,
    use_double_precision: bool = True,
) -> Tuple[
    th.Tensor, th.Tensor, th.Tensor, List[Tuple[th.Tensor, th.Tensor, th.Tensor]]
]:
    """
    Compute global TRS state from local joint transformations using forward kinematics.

    This function implements forward kinematics (FK) using prefix multiplication for efficient
    parallel computation. Each joint's local TRS transformation is composed with its parent's
    global transformation to produce the joint's global transformation.

    The TRS representation uses separate tensors for each transformation component:
    - Translation (3D): translation vector [tx, ty, tz]
    - Rotation (3x3): rotation matrix
    - Scale (1D): uniform scale factor [s]

    Forward Kinematics Formula:
    For each joint j with parent p in the kinematic hierarchy:
        s_global_j = s_parent * s_local_j
        R_global_j = R_parent * R_local_j
        t_global_j = t_parent + s_parent * R_parent * t_local_j

    This corresponds to the similarity transformation: y = s * R * x + t

    Args:
        local_state_t: Local joint translations, shape (batch_size, num_joints, 3).
        local_state_r: Local joint rotations, shape (batch_size, num_joints, 3, 3).
        local_state_s: Local joint scales, shape (batch_size, num_joints, 1).
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
        global_state_t: Global joint translations, shape (batch_size, num_joints, 3).
        global_state_r: Global joint rotations, shape (batch_size, num_joints, 3, 3).
        global_state_s: Global joint scales, shape (batch_size, num_joints, 1).
        intermediate_results: List of (t, r, s) tuples from the forward pass.
            Required for efficient gradient computation during backpropagation.
            Empty if save_intermediate_results=False.

    Note:
        This function is JIT-compiled for performance. The prefix multiplication approach
        allows vectorized batch computation while maintaining kinematic chain dependencies.
        The function is not differentiable by itself - use the wrapper function for gradients.

    See Also:
        :func:`global_trs_state_from_local_trs_state`: User-facing wrapper with autodiff
        :func:`local_trs_state_from_joint_params`: Convert joint parameters to local states
    """
    dtype = local_state_t.dtype
    with th.no_grad():
        if use_double_precision:
            joint_state_t = local_state_t.clone().double()
            joint_state_r = local_state_r.clone().double()
            joint_state_s = local_state_s.clone().double()
        else:
            joint_state_t = local_state_t.clone()
            joint_state_r = local_state_r.clone()
            joint_state_s = local_state_s.clone()

    intermediate_results: List[Tuple[th.Tensor, th.Tensor, th.Tensor]] = []

    for prefix_mul_index in prefix_mul_indices:
        source = prefix_mul_index[0]
        target = prefix_mul_index[1]

        s1 = joint_state_s[:, target]
        r1 = joint_state_r[:, target]
        t1 = joint_state_t[:, target]

        s2 = joint_state_s[:, source]
        r2 = joint_state_r[:, source]
        t2 = joint_state_t[:, source]

        if save_intermediate_results:
            intermediate_results.append(
                (
                    t2.clone(),
                    r2.clone(),
                    s2.clone(),
                )
            )

        t3, r3, s3 = trs.multiply((t1, r1, s1), (t2, r2, s2))

        joint_state_s[:, source] = s3
        joint_state_r[:, source] = r3
        joint_state_t[:, source] = t3

    return (
        joint_state_t.to(dtype),
        joint_state_r.to(dtype),
        joint_state_s.to(dtype),
        intermediate_results,
    )


@th.jit.script
def global_trs_state_from_local_trs_state_no_grad(
    local_state_t: th.Tensor,
    local_state_r: th.Tensor,
    local_state_s: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    save_intermediate_results: bool = True,
    use_double_precision: bool = True,
) -> Tuple[
    th.Tensor, th.Tensor, th.Tensor, List[Tuple[th.Tensor, th.Tensor, th.Tensor]]
]:
    """
    Compute global TRS state without gradient tracking.

    This is a convenience wrapper around global_trs_state_from_local_trs_state_impl
    that explicitly disables gradient computation using torch.no_grad(). Useful for
    inference-only forward passes to reduce memory usage.

    Args:
        local_state_t: Local joint translations, shape (batch_size, num_joints, 3)
        local_state_r: Local joint rotations, shape (batch_size, num_joints, 3, 3)
        local_state_s: Local joint scales, shape (batch_size, num_joints, 1)
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs
        save_intermediate_results: Whether to save intermediate states for backprop
        use_double_precision: Whether to use float64 for numerical stability

    Returns:
        global_state_t: Global joint translations, shape (batch_size, num_joints, 3)
        global_state_r: Global joint rotations, shape (batch_size, num_joints, 3, 3)
        global_state_s: Global joint scales, shape (batch_size, num_joints, 1)
        intermediate_results: List of (t, r, s) tuples from forward pass

    See Also:
        :func:`global_trs_state_from_local_trs_state_impl`: Implementation function
    """
    with th.no_grad():
        outputs = global_trs_state_from_local_trs_state_impl(
            local_state_t,
            local_state_r,
            local_state_s,
            prefix_mul_indices,
            save_intermediate_results=save_intermediate_results,
            use_double_precision=use_double_precision,
        )
    return outputs


@th.jit.script
def ik_from_global_state(
    global_state_t: th.Tensor,
    global_state_r: th.Tensor,
    global_state_s: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    use_double_precision: bool = True,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    dtype = global_state_t.dtype

    if use_double_precision:
        local_state_t = global_state_t.clone().double()
        local_state_r = global_state_r.clone().double()
        local_state_s = global_state_s.clone().double()
    else:
        local_state_t = global_state_t.clone()
        local_state_r = global_state_r.clone()
        local_state_s = global_state_s.clone()

    # Compose the inverse of the FK transforms, in reverse order.
    for prefix_mul_index in prefix_mul_indices[::-1]:
        joint = prefix_mul_index[0]
        parent = prefix_mul_index[1]

        s1 = local_state_s[:, parent].reciprocal()
        r1 = trs.rotmat_inverse(local_state_r[:, parent])
        t1 = local_state_t[:, parent]

        s2 = local_state_s[:, joint]
        r2 = local_state_r[:, joint]
        t2 = local_state_t[:, joint]

        local_state_s[:, joint] = s1 * s2
        local_state_r[:, joint] = trs.rotmat_multiply(r1, r2)
        local_state_t[:, joint] = trs.rotmat_rotate_vector(r1, (t2 - t1) * s1)

    return (
        local_state_t.to(dtype),
        local_state_r.to(dtype),
        local_state_s.to(dtype),
    )


@th.jit.script
def global_trs_state_from_local_trs_state_backprop(
    joint_state_t: th.Tensor,
    joint_state_r: th.Tensor,
    joint_state_s: th.Tensor,
    grad_joint_state_t: th.Tensor,
    grad_joint_state_r: th.Tensor,
    grad_joint_state_s: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    intermediate_results: List[Tuple[th.Tensor, th.Tensor, th.Tensor]],
    use_double_precision: bool = True,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    r"""
    The backward pass of fk_from_local_state_no_grad.

    during backprop, we have
    \partial L/\partial tl_i = \sum_j \partial L/\partial tg_j * \partial tg_j/\partial tl_i
    \partial L/\partial sl_i = \sum_j \partial L/\partial tg_j * \partial tg_j/\partial sl_i + \sum_j \partial L/\partial sg_j * \partial tg_j/\partial sl_i
    \partial L/\partial rl_i = \sum_j \partial L/\partial tg_j * \partial tg_j/\partial rl_i + \sum_j \partial L/\partial rg_j * \partial tg_j/\partial rl_i

    however, if we naively do this, sum_j is very expensive (and the jacobian is very sparse).

    consider how we do prefix multiplication during forward:
    assume the chain order is [0, 1, 2, 3]

    forward (source <- target)

    level 0: [1, 3] <- [0, 2]
    now the chain is [0, 01, 2, 23]

    level 1: [2, 3] <- [1, 1]
    now the chain is [0, 01, 012, 0123]

    now consider backward, for level 0 we need to cast
    {
        g(s1), g(s1s2), g(s1s2s3), g(s1s2s3s4);
        g(R1), g(R1R2), g(R1R2R3), g(R1R2R3R4);
        g(t1), g(t1+s1R1t2), g(s1+s1R2t2+s1R1s2R2t3), ...
    }
    into
    {
        g(s1), g(s1s2), g(s3), g(s3s4);
        g(R1), g(R1R2), g(R3), g(R3R4);
        g(t1), g(t1+s1R1t2), g(t3), g(t3+s3R3t4)
    }
    which is actually in reverse order of forward levels.
    """
    dtype = joint_state_t.dtype
    if use_double_precision:
        grad_local_state_t = grad_joint_state_t.clone().double()
        grad_local_state_r = grad_joint_state_r.clone().double()
        grad_local_state_s = grad_joint_state_s.clone().double()
        joint_state_t = joint_state_t.clone().double()
        joint_state_r = joint_state_r.clone().double()
        joint_state_s = joint_state_s.clone().double()
    else:
        grad_local_state_t = grad_joint_state_t.clone()
        grad_local_state_r = grad_joint_state_r.clone()
        grad_local_state_s = grad_joint_state_s.clone()
        joint_state_t = joint_state_t.clone()
        joint_state_r = joint_state_r.clone()
        joint_state_s = joint_state_s.clone()

    # instead of calculating the original s, r and t from global state
    # we just load them via forward intermediate results
    for prefix_mul_index, (t, r, s) in list(
        zip(prefix_mul_indices, intermediate_results)
    )[::-1]:
        source = prefix_mul_index[0]
        target = prefix_mul_index[1]

        grad_s2 = grad_local_state_s[:, source]
        grad_r2 = grad_local_state_r[:, source]
        grad_t2 = grad_local_state_t[:, source]

        # the corresponding global state
        sg1 = joint_state_s[:, target]
        rg1 = joint_state_r[:, target]

        # TODO: maybe we should better formulate this function
        # similar to pymomentum_state.py

        # backward accumulation on the reduced child state (source)
        # (translate torch.einsum as explicit summations to improve speed)
        grad_s = sg1 * grad_s2
        # original: grad_t = sg1 * th.einsum("bjyx,bjy->bjx", rg1, grad_t2)
        grad_t = sg1 * (rg1 * grad_t2[..., None]).sum(dim=2)
        # original: grad_r = th.einsum("bjyx,bjyz->bjxz", rg1, grad_r2)
        grad_r = (rg1[:, :, :, :, None] * grad_r2[:, :, :, None, :]).sum(dim=2)

        # backward accumulation on the ancestor state (target)
        # original: grad_s1_accum = th.einsum("bjxy,bjy,bjx->bj", rg1, t, grad_t2)[
        #     :, :, None
        # ]
        grad_s1_accum = (
            (rg1 * t[:, :, None, :] * grad_t2[:, :, :, None])
            .sum(dim=3)
            .sum(dim=2, keepdim=True)
        )
        grad_s1_accum = grad_s1_accum + s * grad_s2

        # original: grad_r1_accum = th.einsum("bjxy,bjzy->bjxz", grad_r2, r)
        # original: grad_r1_accum = grad_r1_accum + th.einsum(
        #     "bj,bjx,bjy->bjxy",
        #     sg1[:, :, 0],
        #     grad_t2,
        #     t,
        # )
        grad_r1_accum = (grad_r2[:, :, :, None, :] * r[:, :, None, :, :]).sum(dim=4)
        grad_r1_accum = (
            grad_r1_accum + (sg1 * grad_t2)[:, :, :, None] * t[:, :, None, :]
        )

        grad_t1_accum = grad_t2

        # setup the reduced gradients
        grad_local_state_t[:, source] = grad_t
        grad_local_state_r[:, source] = grad_r
        grad_local_state_s[:, source] = grad_s

        grad_local_state_t.index_add_(1, target, grad_t1_accum)
        grad_local_state_r.index_add_(1, target, grad_r1_accum)
        grad_local_state_s.index_add_(1, target, grad_s1_accum)

        # setup the reduced KC
        joint_state_t[:, source] = t
        joint_state_r[:, source] = r
        joint_state_s[:, source] = s

    return (
        grad_local_state_t.to(dtype),
        grad_local_state_r.to(dtype),
        grad_local_state_s.to(dtype),
    )


class ForwardKinematicsFromLocalTransformationJIT(th.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        local_state_t: th.Tensor,
        local_state_r: th.Tensor,
        local_state_s: th.Tensor,
        prefix_mul_indices: List[th.Tensor],
    ) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor, List[Tuple[th.Tensor, th.Tensor, th.Tensor]]
    ]:
        """
        Compute forward pass for differentiable forward kinematics using TRS representation.

        Args:
            local_state_t: Local joint translations, shape (batch_size, num_joints, 3)
            local_state_r: Local joint rotations, shape (batch_size, num_joints, 3, 3)
            local_state_s: Local joint scales, shape (batch_size, num_joints, 1)
            prefix_mul_indices: List of [child_index, parent_index] tensor pairs

        Returns:
            Tuple of (global_state_t, global_state_r, global_state_s, intermediate_results)
        """
        return global_trs_state_from_local_trs_state_no_grad(
            local_state_t,
            local_state_r,
            local_state_s,
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
            inputs: Tuple of (local_state_t, local_state_r, local_state_s, prefix_mul_indices)
            outputs: Tuple of (joint_state_t, joint_state_r, joint_state_s, intermediate_results)
        """
        (
            _,
            _,
            _,
            prefix_mul_indices,
        ) = inputs
        (
            joint_state_t,
            joint_state_r,
            joint_state_s,
            intermediate_results,
        ) = outputs
        # need to clone as it's modified in-place
        ctx.save_for_backward(
            joint_state_t.clone(),
            joint_state_r.clone(),
            joint_state_s.clone(),
        )
        ctx.intermediate_results = intermediate_results
        ctx.prefix_mul_indices = prefix_mul_indices

    @staticmethod
    # pyre-ignore[14]
    def backward(
        # pyre-ignore[2]
        ctx,
        grad_joint_state_t: th.Tensor,
        grad_joint_state_r: th.Tensor,
        grad_joint_state_s: th.Tensor,
        _0,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, None]:
        (
            joint_state_t,
            joint_state_r,
            joint_state_s,
        ) = ctx.saved_tensors

        intermediate_results = ctx.intermediate_results
        prefix_mul_indices = ctx.prefix_mul_indices

        (
            grad_local_state_t,
            grad_local_state_r,
            grad_local_state_s,
        ) = global_trs_state_from_local_trs_state_backprop(
            joint_state_t,
            joint_state_r,
            joint_state_s,
            grad_joint_state_t,
            grad_joint_state_r,
            grad_joint_state_s,
            prefix_mul_indices,
            intermediate_results,
        )
        return (grad_local_state_t, grad_local_state_r, grad_local_state_s, None)


def global_trs_state_from_local_trs_state(
    local_state_t: th.Tensor,
    local_state_r: th.Tensor,
    local_state_s: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Compute global TRS state from local joint transformations (user-facing wrapper).

    This is the main entry point for forward kinematics using TRS states. It automatically
    selects between JIT-compiled and autograd-enabled implementations based on the execution context.

    Args:
        local_state_t: Local joint translations, shape (batch_size, num_joints, 3).
        local_state_r: Local joint rotations, shape (batch_size, num_joints, 3, 3).
        local_state_s: Local joint scales, shape (batch_size, num_joints, 1).
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs defining the kinematic hierarchy traversal order.

    Returns:
        global_state_t: Global joint translations, shape (batch_size, num_joints, 3).
        global_state_r: Global joint rotations, shape (batch_size, num_joints, 3, 3).
        global_state_s: Global joint scales, shape (batch_size, num_joints, 1).

    Note:
        When called within torch.jit.script or torch.jit.trace context, uses the JIT-compiled
        implementation for maximum performance. Otherwise, uses the autograd-enabled version
        for gradient computation.

    See Also:
        :func:`global_trs_state_from_local_trs_state_impl`: JIT implementation
        :func:`local_trs_state_from_joint_params`: Convert joint parameters to local states
    """
    if th.jit.is_tracing() or th.jit.is_scripting():
        (
            joint_state_t,
            joint_state_r,
            joint_state_s,
            _,
        ) = global_trs_state_from_local_trs_state_impl(
            local_state_t,
            local_state_r,
            local_state_s,
            prefix_mul_indices,
        )
    else:
        (
            joint_state_t,
            joint_state_r,
            joint_state_s,
            _,
        ) = ForwardKinematicsFromLocalTransformationJIT.apply(
            local_state_t,
            local_state_r,
            local_state_s,
            prefix_mul_indices,
        )
    return (
        joint_state_t,
        joint_state_r,
        joint_state_s,
    )


def global_trs_state_from_local_trs_state_forward_only(
    local_state_t: th.Tensor,
    local_state_r: th.Tensor,
    local_state_s: th.Tensor,
    prefix_mul_indices: list[th.Tensor],
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Compute global TRS state from local joint transformations (forward-only wrapper).

    This is a forward-only version that bypasses autograd completely, used when
    gradients are not needed and maximum performance is required.

    Args:
        local_state_t: Local joint translations, shape (batch_size, num_joints, 3).
        local_state_r: Local joint rotations, shape (batch_size, num_joints, 3, 3).
        local_state_s: Local joint scales, shape (batch_size, num_joints, 1).
        prefix_mul_indices: List of [child_index, parent_index] tensor pairs.

    Returns:
        global_state_t: Global joint translations, shape (batch_size, num_joints, 3).
        global_state_r: Global joint rotations, shape (batch_size, num_joints, 3, 3).
        global_state_s: Global joint scales, shape (batch_size, num_joints, 1).

    See Also:
        :func:`global_trs_state_from_local_trs_state`: Main user-facing function with autograd
    """
    (
        joint_state_t,
        joint_state_r,
        joint_state_s,
        _,
    ) = ForwardKinematicsFromLocalTransformationJIT.forward(
        local_state_t,
        local_state_r,
        local_state_s,
        prefix_mul_indices,
    )
    return (
        joint_state_t,
        joint_state_r,
        joint_state_s,
    )


@th.jit.script
def skinning(
    template: th.Tensor,
    t: th.Tensor,
    r: th.Tensor,
    s: th.Tensor,
    t0: th.Tensor,
    r0: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
) -> th.Tensor:
    r"""
    LBS skinning formula as is in lbs_pytorch:
    https://ghe.oculus-rep.com/ydong142857/lbs_pytorch

    TODO: we might want to change skinning to double precision
    with current float32 formulation the numerical error is bigger than 1e-3 level
    (but smaller than 1e-2 level)

    Basically,
    y_i = \sum_j w_ij (s_j * r_j * (r0_j * x_i + t0_j) + t_j)
    where \sum_j w_ij = 1, \forall i

    Args:
        template: (B, V, 3) LBS template
        t: (B, J, 3) Translation of the joints
        r: (B, J, 3, 3) Rotation of the joints
        s: (B, J, 1) Scale of the joints
        t0: (J, 3) Translation of inverse bind pose
        r0: (J, 3, 3) Rotation of inverse bind pose
        (for our setting, s0 == 1)
        skin_indices_flattened: (N, ) LBS skinning nbr joint indices
        skin_weights_flattened: (N, ) LBS skinning nbr joint weights
        vert_indices_flattened: (N, ) LBS skinning nbr corresponding vertex indices

    Returns:
        skinned: (B, V, 3) Skinned mesh
    """
    batch_size = t.shape[0]
    if template.shape[0] != batch_size:
        template = template[None, ...].expand(batch_size, -1, -1)

    sr = s[:, :, :, None] * r
    A = trs.rotmat_multiply(sr, r0[None])
    b = trs.rotmat_rotate_vector(sr, t0[None]) + t

    skinned = th.zeros_like(template)
    skinned = skinned.index_add(
        1,
        vert_indices_flattened,
        (
            trs.rotmat_rotate_vector(
                th.index_select(A, 1, skin_indices_flattened),
                th.index_select(template, 1, vert_indices_flattened),
            )
            + th.index_select(b, 1, skin_indices_flattened)
        )
        * skin_weights_flattened[None, :, None],
    )
    return skinned


@th.jit.script
def multi_topology_skinning(
    template: th.Tensor,
    t: th.Tensor,
    r: th.Tensor,
    s: th.Tensor,
    t0: th.Tensor,
    r0: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
) -> th.Tensor:
    r"""
    LBS skinning formula as is in lbs_pytorch:
    https://ghe.oculus-rep.com/ydong142857/lbs_pytorch

    The difference here is that we assume that the flattened indices are for multiple
    topologies. So vert_indices_flattened needs to flattened with the batch dimension.

    TODO: we might want to change skinning to double precision
    with current float32 formulation the numerical error is bigger than 1e-3 level
    (but smaller than 1e-2 level)

    Basically,
    y_i = \sum_j w_ij (s_j * r_j * (r0_j * x_i + t0_j) + t_j)
    where \sum_j w_ij = 1, \forall i

    Args:
        template: (B, V, 3) LBS template
        t: (B, J, 3) Translation of the joints
        r: (B, J, 3, 3) Rotation of the joints
        s: (B, J, 1) Scale of the joints
        t0: (J, 3) Translation of inverse bind pose
        r0: (J, 3, 3) Rotation of inverse bind pose
        (for our setting, s0 == 1)
        skin_indices_flattened: (N, ) LBS skinning nbr joint indices
        skin_weights_flattened: (N, ) LBS skinning nbr joint weights
        vert_indices_flattened: (N, ) LBS skinning nbr corresponding vertex indices

    Returns:
        skinned: (B, V, 3) Skinned mesh
    """
    batch_size = t.shape[0]
    if template.shape[0] != batch_size:
        template = template[None, ...].expand(batch_size, -1, -1)

    sr = s[:, :, :, None] * r
    A = trs.rotmat_multiply(sr, r0[None])
    b = trs.rotmat_rotate_vector(sr, t0[None]) + t

    # If multi_topology is True, then index on the 0th dimension of A and b
    # because we assume that the skin indices are flattened to index into different
    # vertex indices in each sample of the batch.

    skinning_A = th.index_select(
        A.view(A.shape[0] * A.shape[1], A.shape[2], A.shape[3]),
        0,
        skin_indices_flattened,
    )

    skinning_b = th.index_select(
        b.view(b.shape[0] * b.shape[1], b.shape[2]), 0, skin_indices_flattened
    )

    skinning_verts = th.index_select(
        template.view(template.shape[0] * template.shape[1], template.shape[2]),
        0,
        vert_indices_flattened,
    )

    skinned = th.zeros_like(template).view(
        template.shape[0] * template.shape[1], template.shape[2]
    )
    skinned = skinned.index_add(
        0,
        vert_indices_flattened,
        (trs.rotmat_rotate_vector(skinning_A, skinning_verts) + skinning_b)
        * skin_weights_flattened[..., None],
    )
    return skinned.view(template.shape[0], template.shape[1], template.shape[2])


def unpose_from_global_joint_state(
    verts: th.Tensor,
    t: th.Tensor,
    r: th.Tensor,
    s: th.Tensor,
    t0: th.Tensor,
    r0: th.Tensor,
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
        t: (B, J, 3) Translation of the joints
        r: (B, J, 3, 3) Rotation of the joints
        s: (B, J, 1) Scale of the joints
        t0: (J, 3) Translation of inverse bind pose
        r0: (J, 3, 3) Rotation of inverse bind pose
        skin_indices_flattened: (N, ) LBS skinning nbr joint indices
        skin_weights_flattened: (N, ) LBS skinning nbr joint weights
        vert_indices_flattened: (N, ) LBS skinning nbr corresponding vertex indices
        with_high_precision: if True, use high precision solver (LDLT), but requires a cuda device sync
    """
    dtype = verts.dtype
    device = verts.device

    sr = s[:, :, :, None] * r
    A = trs.rotmat_multiply(sr, r0[None])
    b = trs.rotmat_rotate_vector(sr, t0[None]) + t

    fused_A = th.zeros(verts.shape + (3,), dtype=dtype, device=device)
    fused_b = th.zeros(verts.shape, dtype=dtype, device=device)
    fused_A = fused_A.index_add_(
        1,
        vert_indices_flattened,
        th.index_select(
            A,
            1,
            skin_indices_flattened,
        )
        * skin_weights_flattened[None, :, None, None],
    )
    fused_b = fused_b.index_add_(
        1,
        vert_indices_flattened,
        th.index_select(
            b,
            1,
            skin_indices_flattened,
        )
        * skin_weights_flattened[None, :, None],
    )

    if with_high_precision:
        # th.linalg.solve is not aware of the condition number
        # let's use LDLT decomposition
        ATA = th.einsum("bvyx,bvyz->bvxz", fused_A, fused_A)
        ATb = th.einsum("bvyx,bvy->bvx", fused_A, verts - fused_b)

        # ldl_factor_ex is very slow on GPU
        LD, pivots, _ = th.linalg.ldl_factor_ex(ATA.cpu())
        unposed_mesh = th.linalg.ldl_solve(LD, pivots, ATb[..., None].cpu())[..., 0]

        unposed_mesh = unposed_mesh.to(ATA.device)
    else:
        unposed_mesh = th.linalg.solve(fused_A, verts - fused_b)

    return unposed_mesh


@th.jit.script
def get_local_state_from_joint_params(
    joint_params: th.Tensor,
    joint_offset: th.Tensor,
    joint_rotation: th.Tensor,
    joint_parents: th.Tensor | None = None,
    allow_inverse_kinematic_chain: bool = False,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    calculate local joint state from joint parameters.

    Args:
        joint_params: [batch_size, num_joints, 7] or [batch_size, num_joints * 7]
        joint_offset: [num_joints, 3]
        joint_rotation: [num_joints, 3, 3]
        allow_inverse_kinematic_chain: if set to True, this hints that the kinematic
            chain might be reversed (e.g. from wrist to root). This leads to a few
            changes in assumption. One of the major difference is that the root joint
            always has identity [0, I, 1] transformation.

    Returns:
        local_state_t: [batch_size, num_joints, 3]
        local_state_r: [batch_size, num_joints, 3, 3]
        local_state_s: [batch_size, num_joints, 1]
    """
    if len(joint_params.shape) == 2:
        # reshape joint_params as (batch_size, num_joints, 7)
        joint_params = joint_params.view(joint_params.shape[0], -1, 7)

    # the vanilla conversion
    local_state_t = joint_params[:, :, :3] + joint_offset[None, :]
    local_state_r = trs.rotmat_multiply(
        joint_rotation[None], trs.rotmat_from_euler_xyz(joint_params[:, :, 3:6])
    )
    local_state_s = th.exp2(joint_params[:, :, 6:])

    if allow_inverse_kinematic_chain:
        assert joint_parents is not None
        assert len(joint_parents.shape) == 1
        device = joint_parents.device
        root_joint = th.where(joint_parents == -1)[0]
        inversed_joints = th.where(
            joint_parents
            > th.arange(0, len(joint_parents), dtype=th.long, device=device)
        )[0]
        inversed_joint_parents = joint_parents[inversed_joints]

        # create a new node so the autograd does not fail
        (
            _local_state_t,
            _local_state_r,
            _local_state_s,
        ) = (
            local_state_t.clone(),
            local_state_r.clone(),
            local_state_s.clone(),
        )

        # for the inverse joints
        # the order needs to be inversed
        (
            _local_state_t[:, inversed_joints],
            _local_state_r[:, inversed_joints],
            _local_state_s[:, inversed_joints],
        ) = trs.inverse(
            (
                local_state_t[:, inversed_joint_parents],
                local_state_r[:, inversed_joint_parents],
                local_state_s[:, inversed_joint_parents],
            )
        )

        # set new root joint to identity
        _local_state_t[:, root_joint] = 0
        _local_state_r[:, root_joint] = th.eye(3, device=device)[None]
        _local_state_s[:, root_joint] = 1

        (
            local_state_t,
            local_state_r,
            local_state_s,
        ) = (
            _local_state_t,
            _local_state_r,
            _local_state_s,
        )

    return local_state_t, local_state_r, local_state_s
