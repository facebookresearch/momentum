# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import List, Tuple

import torch as th

from pymomentum import quaternion, skel_state, trs
from pymomentum.backend.trs_backend import unpose_from_global_joint_state


@th.jit.script
def get_local_momentum_state_from_joint_params(
    joint_params: th.Tensor,
    joint_offset: th.Tensor,
    joint_quat_rotation: th.Tensor,
) -> th.Tensor:
    """
    calculate local momentum joint state from joint parameters.

    Args:
        joint_params: (B, J, 7), the momentum per-joint params
        joint_offset: (J, 3) per-joint pre-offset
        joint_quat_rotation: (J, 4) per-joint pre-rotation

    Returns:
        local_joint_state: (B, J, 8)
    """
    t = joint_offset[None, :] + joint_params[:, :, :3]
    q = quaternion.multiply(
        joint_quat_rotation[None],
        quaternion.euler_xyz_to_quaternion(joint_params[:, :, 3:6]),
    )
    s = th.exp2(joint_params[:, :, 6:])

    return th.cat([t, q, s], dim=-1)


@th.jit.script
def fk_from_local_momentum_state_impl(
    local_joint_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    save_intermediate_results: bool = True,
    use_double_precision: bool = True,
) -> Tuple[th.Tensor, List[th.Tensor]]:
    """
    same as fk_from_local_state, but for momentum

    TODO: need to re-verify if double precision is needed

    Args:
        local_joint_state (th.Tensor): (B, J, 8)
        prefix_mul_indices (List[th.Tensor]): the prefix multiplication indices
        save_intermediate_results (bool, optional): save intermediate results for backward.
            Defaults to True.

    Returns:
        global_joint_state (th.Tensor): (B, J, 8)
        intermediate_results (List[th.Tensor]): intermediate results for backward
    """
    dtype = local_joint_state.dtype
    intermediate_results: List[th.Tensor] = []
    if use_double_precision:
        global_joint_state = local_joint_state.clone().double()
    else:
        global_joint_state = local_joint_state.clone()
    for prefix_mul_index in prefix_mul_indices:
        source = prefix_mul_index[0]
        target = prefix_mul_index[1]

        state1 = global_joint_state.index_select(-2, target)
        state2 = global_joint_state.index_select(-2, source)

        if save_intermediate_results:
            intermediate_results.append(state2.clone())

        global_joint_state.index_copy_(-2, source, skel_state.multiply(state1, state2))

    return (global_joint_state.to(dtype), intermediate_results)


def fk_from_local_momentum_state_no_grad(
    local_joint_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    save_intermediate_results: bool = True,
    use_double_precision: bool = True,
) -> Tuple[th.Tensor, List[th.Tensor]]:
    """just for compatibility"""
    with th.no_grad():
        outputs = fk_from_local_momentum_state_impl(
            local_joint_state,
            prefix_mul_indices,
            save_intermediate_results=save_intermediate_results,
            use_double_precision=use_double_precision,
        )
    return outputs


# @th.jit.script
def fk_from_local_momentum_state_backprop(
    global_joint_state: th.Tensor,
    grad_global_joint_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
    intermediate_results: List[th.Tensor],
    use_double_precision: bool = True,
) -> th.Tensor:
    """
    same as fk_from_local_state_backprop, but for momentum

    TODO: need to re-verify if double precision is needed

    Args:
        global_joint_state (th.Tensor): (B, J, 8)
        grad_global_joint_state (th.Tensor): (B, J, 8)
        prefix_mul_indices (List[th.Tensor]): the prefix multiplication indices
        intermediate_results (List[th.Tensor]): intermediate results from fk forward

    Returns:
        grad_local_joint_state (th.Tensor): (B, J, 8)
    """
    dtype = global_joint_state.dtype
    if use_double_precision:
        grad_local_joint_state = grad_global_joint_state.clone().double()
        global_joint_state = global_joint_state.clone().double()
    else:
        grad_local_joint_state = grad_global_joint_state.clone()
        global_joint_state = global_joint_state.clone()

    for prefix_mul_index, state_source_original in list(
        zip(prefix_mul_indices, intermediate_results)
    )[::-1]:
        source = prefix_mul_index[0]
        target = prefix_mul_index[1]

        state_target = global_joint_state.index_select(-2, target)

        # forward
        # This is for checking that the numerical error is in the safe
        # range, but it's not completely abnormal to see large numerical
        # error during backpropagation at the beginning of training.
        # state_source = global_joint_state[:, source]
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
            grad_state=grad_local_joint_state.index_select(-2, source),
        )

        # setup the reduced gradients
        grad_local_joint_state.index_copy_(-2, source, grad_source_original)
        grad_local_joint_state.index_add_(-2, target, grad_target_accum)
        # setup the reduced KC
        global_joint_state.index_copy_(-2, source, state_source_original)

    return grad_local_joint_state.to(dtype)


class ForwardKinematicsFromLocalMomentumStateJIT(th.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        local_joint_state: th.Tensor,
        prefix_mul_indices: List[th.Tensor],
    ) -> Tuple[th.Tensor, List[th.Tensor]]:
        return fk_from_local_momentum_state_no_grad(
            local_joint_state,
            prefix_mul_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    # pyre-ignore[2]
    def setup_context(ctx, inputs, outputs) -> None:
        (
            _,
            prefix_mul_indices,
        ) = inputs
        (
            global_joint_state,
            intermediate_results,
        ) = outputs
        # need to clone as it's modified in-place
        ctx.save_for_backward(global_joint_state.clone())
        ctx.intermediate_results = intermediate_results
        ctx.prefix_mul_indices = prefix_mul_indices

    @staticmethod
    # pyre-ignore[14]
    def backward(
        # pyre-ignore[2]
        ctx,
        grad_global_joint_state: th.Tensor,
        _0,
    ) -> Tuple[th.Tensor, None]:
        (global_joint_state,) = ctx.saved_tensors

        intermediate_results = ctx.intermediate_results
        prefix_mul_indices = ctx.prefix_mul_indices

        grad_local_state = fk_from_local_momentum_state_backprop(
            global_joint_state,
            grad_global_joint_state,
            prefix_mul_indices,
            intermediate_results,
        )
        return grad_local_state, None


def fk_from_local_momentum_state(
    local_joint_state: th.Tensor,
    prefix_mul_indices: List[th.Tensor],
) -> th.Tensor:
    """
    Wrap FK as a function.
    Args:
        local_joint_state (th.Tensor): (B, J, 8) local joint state
        prefix_mul_indices (List[th.Tensor]): the prefix multiplication indices
    """

    if th.jit.is_tracing() or th.jit.is_scripting():
        global_joint_state, _ = fk_from_local_momentum_state_impl(
            local_joint_state, prefix_mul_indices
        )
        return global_joint_state
    else:
        global_joint_state, _ = ForwardKinematicsFromLocalMomentumStateJIT.apply(
            local_joint_state,
            prefix_mul_indices,
        )
        return global_joint_state


@th.jit.script
def skinning_from_momentum_state(
    template: th.Tensor,
    global_joint_state: th.Tensor,
    binded_joint_state_inv: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
) -> th.Tensor:
    """
    same as skinning(), but for momentum
    Args:
        template (th.Tensor): (B, V, 3)
        global_joint_state (th.Tensor): (B, J, 8)
        binded_joint_state (th.Tensor): (B, J, 8)
        skin_indices_flattened: (N, ) LBS skinning nbr joint indices
        skin_weights_flattened: (N, ) LBS skinning nbr joint weights
        vert_indices_flattened: (N, ) LBS skinning nbr corresponding vertex indices
    Returns:
        skinned (th.Tensor): (B, V, 3)
    """
    assert template.shape[-1] == 3
    while template.ndim < global_joint_state.ndim:
        template = template.unsqueeze(0)

    template = template.expand(
        list(global_joint_state.shape[:-2]) + list(template.shape[-2:])
    )

    joint_state = skel_state.multiply(
        global_joint_state,
        binded_joint_state_inv,
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
def skinning_oriented_points_from_momentum_state(
    means: th.Tensor,
    quaternions: th.Tensor,
    global_joint_state: th.Tensor,
    binded_joint_state_inv: th.Tensor,
    skin_indices_flattened: th.Tensor,
    skin_weights_flattened: th.Tensor,
    vert_indices_flattened: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """
    same as skinning_from_momentum_state(), but for gaussians
    for the orientation we use lerp of rotation matrices to match
    Args:
        means (th.Tensor): (B, V, 3)
        quaternions (th.Tensor): (B, V, 4)
        global_joint_state (th.Tensor): (B, J, 8)
        binded_joint_state (th.Tensor): (B, J, 8)
        skin_indices_flattened: (N, ) LBS skinning nbr joint indices
        skin_weights_flattened: (N, ) LBS skinning nbr joint weights
        vert_indices_flattened: (N, ) LBS skinning nbr corresponding vertex indices
    Returns:
        skinned_means (th.Tensor): (B, V, 3)
        skinned_quaternions (th.Tensor): (B, V, 4)
    """
    assert means.shape[-1] == 3
    assert quaternions.shape[-1] == 4
    while means.ndim < global_joint_state.ndim:
        means = means.unsqueeze(0)

    means = means.expand(list(global_joint_state.shape[:-2]) + list(means.shape[-2:]))

    joint_state = skel_state.multiply(
        global_joint_state,
        binded_joint_state_inv,
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
        with_high_precision: if True, use high precision solver (LDLT),
            but requires a cuda device sync
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
