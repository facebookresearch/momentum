# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_BLOCK_INFLUENCES = 256
_BLOCK_VERTICES = 32

if triton is not None and tl is not None:
    from pymomentum.backend.triton_common import (  # @manual=:backend_triton
        _quat_multiply,
        _quat_multiply_backward,
        _rotate_vector,
        _rotate_vector_backward,
    )

    @triton.jit
    def _load_skel_state(data, batch, joint, n_joints, mask):
        base = data + (batch * n_joints + joint) * 8
        return (
            tl.load(base + 0, mask=mask, other=0.0),
            tl.load(base + 1, mask=mask, other=0.0),
            tl.load(base + 2, mask=mask, other=0.0),
            tl.load(base + 3, mask=mask, other=0.0),
            tl.load(base + 4, mask=mask, other=0.0),
            tl.load(base + 5, mask=mask, other=0.0),
            tl.load(base + 6, mask=mask, other=1.0),
            tl.load(base + 7, mask=mask, other=1.0),
        )

    @triton.jit
    def _batch_offset(batch, batch_dim1: tl.constexpr, n_batch_dims: tl.constexpr):
        if n_batch_dims == 0:
            return 0, 0
        if n_batch_dims == 1:
            return batch, 0
        return batch // batch_dim1, batch % batch_dim1

    @triton.jit
    def _load_points(
        data,
        batch,
        vertex,
        stride_batch0: tl.constexpr,
        stride_batch1: tl.constexpr,
        stride_vertex: tl.constexpr,
        stride_coord: tl.constexpr,
        batch_dim1: tl.constexpr,
        n_batch_dims: tl.constexpr,
        mask,
    ):
        batch0, batch1 = _batch_offset(batch, batch_dim1, n_batch_dims)
        base = (
            data
            + batch0 * stride_batch0
            + batch1 * stride_batch1
            + vertex * stride_vertex
        )
        return (
            tl.load(base + 0 * stride_coord, mask=mask, other=0.0),
            tl.load(base + 1 * stride_coord, mask=mask, other=0.0),
            tl.load(base + 2 * stride_coord, mask=mask, other=0.0),
        )

    @triton.jit
    def _add_point_grads(
        data,
        batch,
        vertex,
        gx,
        gy,
        gz,
        stride_batch0: tl.constexpr,
        stride_batch1: tl.constexpr,
        stride_vertex: tl.constexpr,
        stride_coord: tl.constexpr,
        batch_dim1: tl.constexpr,
        n_batch_dims: tl.constexpr,
        mask,
    ):
        batch0, batch1 = _batch_offset(batch, batch_dim1, n_batch_dims)
        base = (
            data
            + batch0 * stride_batch0
            + batch1 * stride_batch1
            + vertex * stride_vertex
        )
        tl.atomic_add(base + 0 * stride_coord, gx, sem="relaxed", mask=mask)
        tl.atomic_add(base + 1 * stride_coord, gy, sem="relaxed", mask=mask)
        tl.atomic_add(base + 2 * stride_coord, gz, sem="relaxed", mask=mask)

    @triton.jit
    def _skin_vertices(
        template,
        global_skel_state,
        binded_skel_state_inv,
        skin_indices,
        skin_weights,
        output,
        batch,
        vertices,
        n_joints,
        n_vertices,
        n_skin_influences,
        template_stride_batch0: tl.constexpr,
        template_stride_batch1: tl.constexpr,
        template_stride_vertex: tl.constexpr,
        template_stride_coord: tl.constexpr,
        batch_dim1: tl.constexpr,
        n_batch_dims: tl.constexpr,
        vertex_mask,
    ):
        influence = tl.arange(0, n_skin_influences)
        influence_offsets = vertices[:, None] * n_skin_influences + influence[None, :]
        influence_mask = vertex_mask[:, None]
        joint = tl.load(skin_indices + influence_offsets, mask=influence_mask, other=0)
        weight = tl.load(
            skin_weights + influence_offsets, mask=influence_mask, other=0.0
        )

        gtx, gty, gtz, gqx, gqy, gqz, gqw, gs = _load_skel_state(
            global_skel_state, batch, joint, n_joints, influence_mask
        )
        btx, bty, btz, bqx, bqy, bqz, bqw, bs = _load_skel_state(
            binded_skel_state_inv, batch, joint, n_joints, influence_mask
        )
        vx, vy, vz = _load_points(
            template,
            batch,
            vertices,
            template_stride_batch0,
            template_stride_batch1,
            template_stride_vertex,
            template_stride_coord,
            batch_dim1,
            n_batch_dims,
            vertex_mask,
        )

        rbtx, rbty, rbtz = _rotate_vector(gqx, gqy, gqz, gqw, btx, bty, btz)
        ctx = gtx + gs * rbtx
        cty = gty + gs * rbty
        ctz = gtz + gs * rbtz
        cqx, cqy, cqz, cqw = _quat_multiply(gqx, gqy, gqz, gqw, bqx, bqy, bqz, bqw)
        cs = gs * bs

        rvx, rvy, rvz = _rotate_vector(
            cqx, cqy, cqz, cqw, vx[:, None], vy[:, None], vz[:, None]
        )
        out_base = output + (batch * n_vertices + vertices) * 3
        tl.store(
            out_base + 0,
            tl.sum(weight * (ctx + cs * rvx), axis=1),
            mask=vertex_mask,
        )
        tl.store(
            out_base + 1,
            tl.sum(weight * (cty + cs * rvy), axis=1),
            mask=vertex_mask,
        )
        tl.store(
            out_base + 2,
            tl.sum(weight * (ctz + cs * rvz), axis=1),
            mask=vertex_mask,
        )

    @triton.jit
    def _forward_kernel(
        template,
        global_skel_state,
        binded_skel_state_inv,
        skin_indices,
        skin_weights,
        output,
        n_joints,
        n_vertices,
        n_skin_influences: tl.constexpr,
        template_stride_batch0: tl.constexpr,
        template_stride_batch1: tl.constexpr,
        template_stride_vertex: tl.constexpr,
        template_stride_coord: tl.constexpr,
        batch_dim1: tl.constexpr,
        n_batch_dims: tl.constexpr,
        BLOCK_VERTICES: tl.constexpr,
    ):
        batch = tl.program_id(0)
        vertices = tl.program_id(1) * BLOCK_VERTICES + tl.arange(0, BLOCK_VERTICES)
        vertex_mask = vertices < n_vertices
        _skin_vertices(
            template,
            global_skel_state,
            binded_skel_state_inv,
            skin_indices,
            skin_weights,
            output,
            batch,
            vertices,
            n_joints,
            n_vertices,
            n_skin_influences,
            template_stride_batch0,
            template_stride_batch1,
            template_stride_vertex,
            template_stride_coord,
            batch_dim1,
            n_batch_dims,
            vertex_mask,
        )

    @triton.jit
    def _backward_kernel(
        template,
        global_skel_state,
        binded_skel_state_inv,
        skin_indices_flattened,
        skin_weights_flattened,
        vert_indices_flattened,
        grad_output,
        grad_template,
        grad_global_skel_state,
        n_joints,
        n_vertices,
        n_influences,
        template_stride_batch0: tl.constexpr,
        template_stride_batch1: tl.constexpr,
        template_stride_vertex: tl.constexpr,
        template_stride_coord: tl.constexpr,
        grad_template_stride_batch0: tl.constexpr,
        grad_template_stride_batch1: tl.constexpr,
        grad_template_stride_vertex: tl.constexpr,
        grad_template_stride_coord: tl.constexpr,
        batch_dim1: tl.constexpr,
        n_batch_dims: tl.constexpr,
        BLOCK_INFLUENCES: tl.constexpr,
    ):
        batch = tl.program_id(0)
        influence = tl.program_id(1) * BLOCK_INFLUENCES + tl.arange(0, BLOCK_INFLUENCES)
        mask = influence < n_influences

        joint = tl.load(skin_indices_flattened + influence, mask=mask, other=0)
        vertex = tl.load(vert_indices_flattened + influence, mask=mask, other=0)
        weight = tl.load(skin_weights_flattened + influence, mask=mask, other=0.0)

        gtx, gty, gtz, gqx, gqy, gqz, gqw, gs = _load_skel_state(
            global_skel_state, batch, joint, n_joints, mask
        )
        btx, bty, btz, bqx, bqy, bqz, bqw, bs = _load_skel_state(
            binded_skel_state_inv, batch, joint, n_joints, mask
        )
        vx, vy, vz = _load_points(
            template,
            batch,
            vertex,
            template_stride_batch0,
            template_stride_batch1,
            template_stride_vertex,
            template_stride_coord,
            batch_dim1,
            n_batch_dims,
            mask,
        )

        gout_base = grad_output + (batch * n_vertices + vertex) * 3
        gyx = weight * tl.load(gout_base + 0, mask=mask, other=0.0)
        gyy = weight * tl.load(gout_base + 1, mask=mask, other=0.0)
        gyz = weight * tl.load(gout_base + 2, mask=mask, other=0.0)

        rbtx, rbty, rbtz = _rotate_vector(gqx, gqy, gqz, gqw, btx, bty, btz)
        cqx, cqy, cqz, cqw = _quat_multiply(gqx, gqy, gqz, gqw, bqx, bqy, bqz, bqw)
        cs = gs * bs

        rvx, rvy, rvz = _rotate_vector(cqx, cqy, cqz, cqw, vx, vy, vz)
        grad_cs = gyx * rvx + gyy * rvy + gyz * rvz
        grad_rvx = cs * gyx
        grad_rvy = cs * gyy
        grad_rvz = cs * gyz

        (
            grad_cqx,
            grad_cqy,
            grad_cqz,
            grad_cqw,
            grad_vx,
            grad_vy,
            grad_vz,
        ) = _rotate_vector_backward(
            cqx, cqy, cqz, cqw, vx, vy, vz, grad_rvx, grad_rvy, grad_rvz
        )

        (
            grad_gqx_from_quat,
            grad_gqy_from_quat,
            grad_gqz_from_quat,
            grad_gqw_from_quat,
            _,
            _,
            _,
            _,
        ) = _quat_multiply_backward(
            gqx,
            gqy,
            gqz,
            gqw,
            bqx,
            bqy,
            bqz,
            bqw,
            grad_cqx,
            grad_cqy,
            grad_cqz,
            grad_cqw,
        )

        grad_rbtx = gs * gyx
        grad_rbty = gs * gyy
        grad_rbtz = gs * gyz
        (
            grad_gqx_from_t,
            grad_gqy_from_t,
            grad_gqz_from_t,
            grad_gqw_from_t,
            _,
            _,
            _,
        ) = _rotate_vector_backward(
            gqx,
            gqy,
            gqz,
            gqw,
            btx,
            bty,
            btz,
            grad_rbtx,
            grad_rbty,
            grad_rbtz,
        )

        grad_gs = gyx * rbtx + gyy * rbty + gyz * rbtz + grad_cs * bs

        _add_point_grads(
            grad_template,
            batch,
            vertex,
            grad_vx,
            grad_vy,
            grad_vz,
            grad_template_stride_batch0,
            grad_template_stride_batch1,
            grad_template_stride_vertex,
            grad_template_stride_coord,
            batch_dim1,
            n_batch_dims,
            mask,
        )

        grad_joint_base = grad_global_skel_state + (batch * n_joints + joint) * 8
        tl.atomic_add(grad_joint_base + 0, gyx, sem="relaxed", mask=mask)
        tl.atomic_add(grad_joint_base + 1, gyy, sem="relaxed", mask=mask)
        tl.atomic_add(grad_joint_base + 2, gyz, sem="relaxed", mask=mask)
        tl.atomic_add(
            grad_joint_base + 3,
            grad_gqx_from_t + grad_gqx_from_quat,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            grad_joint_base + 4,
            grad_gqy_from_t + grad_gqy_from_quat,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            grad_joint_base + 5,
            grad_gqz_from_t + grad_gqz_from_quat,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            grad_joint_base + 6,
            grad_gqw_from_t + grad_gqw_from_quat,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(grad_joint_base + 7, grad_gs, sem="relaxed", mask=mask)


class _SkinPointsFromSkelState(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        template: torch.Tensor,
        global_skel_state: torch.Tensor,
        binded_skel_state_inv: torch.Tensor,
        skin_indices: torch.Tensor,
        skin_weights: torch.Tensor,
        skin_indices_flattened: torch.Tensor,
        skin_weights_flattened: torch.Tensor,
        vert_indices_flattened: torch.Tensor,
        batch_dim1: int,
        n_batch_dims: int,
    ) -> torch.Tensor:
        batch_size = global_skel_state.shape[0]
        n_joints = global_skel_state.shape[1]
        n_vertices = template.shape[-2]
        n_skin_influences = skin_indices.shape[1]
        output = torch.empty(
            (batch_size, n_vertices, 3),
            device=template.device,
            dtype=template.dtype,
        )
        grid = (batch_size, triton.cdiv(n_vertices, _BLOCK_VERTICES))
        _forward_kernel[grid](
            template,
            global_skel_state,
            binded_skel_state_inv,
            skin_indices,
            skin_weights,
            output,
            n_joints,
            n_vertices,
            n_skin_influences,
            template.stride(0) if n_batch_dims > 0 else 0,
            template.stride(1) if n_batch_dims > 1 else 0,
            template.stride(-2),
            template.stride(-1),
            batch_dim1,
            n_batch_dims,
            BLOCK_VERTICES=_BLOCK_VERTICES,
        )
        ctx.save_for_backward(
            template,
            global_skel_state,
            binded_skel_state_inv,
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        )
        ctx.batch_dim1 = batch_dim1
        ctx.n_batch_dims = n_batch_dims
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            template,
            global_skel_state,
            binded_skel_state_inv,
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
        ) = ctx.saved_tensors
        batch_size = global_skel_state.shape[0]
        n_joints = global_skel_state.shape[1]
        n_vertices = template.shape[-2]
        n_influences = skin_indices_flattened.numel()

        grad_template = torch.zeros_like(template)
        grad_global_skel_state = torch.zeros_like(global_skel_state)
        grid = (
            batch_size,
            triton.cdiv(n_influences, _BLOCK_INFLUENCES),
        )
        _backward_kernel[grid](
            template,
            global_skel_state,
            binded_skel_state_inv,
            skin_indices_flattened,
            skin_weights_flattened,
            vert_indices_flattened,
            grad_output.contiguous(),
            grad_template,
            grad_global_skel_state,
            n_joints,
            n_vertices,
            n_influences,
            template.stride(0) if ctx.n_batch_dims > 0 else 0,
            template.stride(1) if ctx.n_batch_dims > 1 else 0,
            template.stride(-2),
            template.stride(-1),
            grad_template.stride(0) if ctx.n_batch_dims > 0 else 0,
            grad_template.stride(1) if ctx.n_batch_dims > 1 else 0,
            grad_template.stride(-2),
            grad_template.stride(-1),
            ctx.batch_dim1,
            ctx.n_batch_dims,
            BLOCK_INFLUENCES=_BLOCK_INFLUENCES,
        )
        return (
            grad_template,
            grad_global_skel_state,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def skin_points_from_skel_state(
    template: torch.Tensor,
    global_skel_state: torch.Tensor,
    binded_skel_state_inv: torch.Tensor,
    skin_indices: torch.Tensor,
    skin_weights: torch.Tensor,
    skin_indices_flattened: torch.Tensor,
    skin_weights_flattened: torch.Tensor,
    vert_indices_flattened: torch.Tensor,
) -> torch.Tensor:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if not global_skel_state.is_cuda:
        raise RuntimeError("input is on CPU, but Triton skinning requires CUDA")
    if global_skel_state.dtype != torch.float32:
        raise RuntimeError("Triton skinning only supports float32 inputs")
    if template.shape[-1] != 3:
        raise ValueError("template must contain 3D points")

    batch_shape = global_skel_state.shape[:-2]
    if template.ndim == global_skel_state.ndim - 1 and tuple(
        template.shape[:-2]
    ) == tuple(global_skel_state.shape[:-3]):
        template = template.unsqueeze(-3)
    while template.ndim < global_skel_state.ndim:
        template = template.unsqueeze(0)
    template = template.expand(list(batch_shape) + list(template.shape[-2:]))
    if len(batch_shape) > 2:
        template = template.contiguous()

    while binded_skel_state_inv.ndim < global_skel_state.ndim:
        binded_skel_state_inv = binded_skel_state_inv.unsqueeze(0)
    binded_skel_state_inv = binded_skel_state_inv.expand_as(
        global_skel_state
    ).contiguous()

    global_shape = global_skel_state.shape
    template_shape = template.shape
    if len(batch_shape) > 2:
        template_for_kernel = template.view(-1, template_shape[-2], 3)
        n_batch_dims = 1
        batch_dim1 = 1
    else:
        template_for_kernel = template
        n_batch_dims = len(batch_shape)
        batch_dim1 = template_shape[1] if n_batch_dims > 1 else 1
    flat_global_skel_state = global_skel_state.contiguous().view(
        -1, global_shape[-2], 8
    )
    flat_binded_skel_state_inv = binded_skel_state_inv.view(-1, global_shape[-2], 8)

    result = _SkinPointsFromSkelState.apply(
        template_for_kernel,
        flat_global_skel_state,
        flat_binded_skel_state_inv,
        skin_indices.contiguous(),
        skin_weights.contiguous(),
        skin_indices_flattened.contiguous(),
        skin_weights_flattened.contiguous(),
        vert_indices_flattened.contiguous(),
        batch_dim1,
        n_batch_dims,
    )
    return result.view(template_shape)
