# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    try:
        # @manual=//triton:triton
        import triton.language.extra.libdevice as libdevice
    except ImportError:
        # @manual=//triton:triton
        import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


_BLOCK_BATCH = 128
_BLOCK_SIZE = 256
_LN2 = tl.constexpr(math.log(2.0)) if tl is not None else math.log(2.0)
_NORM_EPS = tl.constexpr(1e-12) if tl is not None else 1e-12
_EULER_EPS = tl.constexpr(1e-6) if tl is not None else 1e-6

if triton is not None and tl is not None:  # noqa: C901
    from pymomentum.backend.triton_common import (  # @manual=:backend_triton
        _quat_multiply,
        _quat_multiply_backward,
        _rotate_vector,
        _rotate_vector_backward,
    )

    @triton.jit
    def _normalize_quaternion(x, y, z, w):
        norm2 = x * x + y * y + z * z + w * w
        inv_norm = tl.rsqrt(tl.maximum(norm2, _NORM_EPS * _NORM_EPS))
        return x * inv_norm, y * inv_norm, z * inv_norm, w * inv_norm

    @triton.jit
    def _normalize_quaternion_backward(x, y, z, w, gx, gy, gz, gw):
        nx, ny, nz, nw = _normalize_quaternion(x, y, z, w)
        norm2 = x * x + y * y + z * z + w * w
        inv_norm = tl.rsqrt(tl.maximum(norm2, _NORM_EPS * _NORM_EPS))
        dot = gx * nx + gy * ny + gz * nz + gw * nw
        unclamped_gx = (gx - nx * dot) * inv_norm
        unclamped_gy = (gy - ny * dot) * inv_norm
        unclamped_gz = (gz - nz * dot) * inv_norm
        unclamped_gw = (gw - nw * dot) * inv_norm
        clamped_gx = gx * inv_norm
        clamped_gy = gy * inv_norm
        clamped_gz = gz * inv_norm
        clamped_gw = gw * inv_norm
        is_unclamped = norm2 >= _NORM_EPS * _NORM_EPS
        return (
            tl.where(is_unclamped, unclamped_gx, clamped_gx),
            tl.where(is_unclamped, unclamped_gy, clamped_gy),
            tl.where(is_unclamped, unclamped_gz, clamped_gz),
            tl.where(is_unclamped, unclamped_gw, clamped_gw),
        )

    @triton.jit
    def _nonzero_denominator(x):
        return tl.where(
            tl.abs(x) < _EULER_EPS,
            tl.where(x < 0.0, -_EULER_EPS, _EULER_EPS),
            x,
        )

    @triton.jit
    def _quaternion_to_xyz_euler(x, y, z, w):
        x, y, z, w = _normalize_quaternion(x, y, z, w)
        rx_num = 2.0 * (w * x + y * z)
        rx_den = _nonzero_denominator(1.0 - 2.0 * (x * x + y * y))
        ry_arg = tl.minimum(
            tl.maximum(2.0 * (w * y - z * x), -1.0 + _EULER_EPS),
            1.0 - _EULER_EPS,
        )
        rz_num = 2.0 * (w * z + x * y)
        rz_den = _nonzero_denominator(1.0 - 2.0 * (y * y + z * z))
        return (
            libdevice.atan2(rx_num, rx_den),
            libdevice.asin(ry_arg),
            libdevice.atan2(rz_num, rz_den),
        )

    @triton.jit
    def _quaternion_to_xyz_euler_backward(x, y, z, w, grx, gry, grz):
        nx, ny, nz, nw = _normalize_quaternion(x, y, z, w)
        rx_num = 2.0 * (nw * nx + ny * nz)
        rx_den_raw = 1.0 - 2.0 * (nx * nx + ny * ny)
        rx_den = _nonzero_denominator(rx_den_raw)
        rx_inv = 1.0 / (rx_num * rx_num + rx_den * rx_den)
        grad_rx_num = grx * rx_den * rx_inv
        grad_rx_den = -grx * rx_num * rx_inv
        use_rx_den = tl.abs(rx_den_raw) >= _EULER_EPS

        ry_arg_raw = 2.0 * (nw * ny - nz * nx)
        ry_arg = tl.minimum(tl.maximum(ry_arg_raw, -1.0 + _EULER_EPS), 1.0 - _EULER_EPS)
        use_ry = (ry_arg_raw >= -1.0 + _EULER_EPS) & (ry_arg_raw <= 1.0 - _EULER_EPS)
        grad_ry_arg = tl.where(use_ry, gry / tl.sqrt(1.0 - ry_arg * ry_arg), 0.0)

        rz_num = 2.0 * (nw * nz + nx * ny)
        rz_den_raw = 1.0 - 2.0 * (ny * ny + nz * nz)
        rz_den = _nonzero_denominator(rz_den_raw)
        rz_inv = 1.0 / (rz_num * rz_num + rz_den * rz_den)
        grad_rz_num = grz * rz_den * rz_inv
        grad_rz_den = -grz * rz_num * rz_inv
        use_rz_den = tl.abs(rz_den_raw) >= _EULER_EPS

        gx = (
            2.0 * nw * grad_rx_num
            + tl.where(use_rx_den, -4.0 * nx * grad_rx_den, 0.0)
            - 2.0 * nz * grad_ry_arg
            + 2.0 * ny * grad_rz_num
        )
        gy = (
            2.0 * nz * grad_rx_num
            + tl.where(use_rx_den, -4.0 * ny * grad_rx_den, 0.0)
            + 2.0 * nw * grad_ry_arg
            + 2.0 * nx * grad_rz_num
            + tl.where(use_rz_den, -4.0 * ny * grad_rz_den, 0.0)
        )
        gz = (
            2.0 * ny * grad_rx_num
            - 2.0 * nx * grad_ry_arg
            + 2.0 * nw * grad_rz_num
            + tl.where(use_rz_den, -4.0 * nz * grad_rz_den, 0.0)
        )
        gw = 2.0 * nx * grad_rx_num + 2.0 * ny * grad_ry_arg + 2.0 * nz * grad_rz_num
        return _normalize_quaternion_backward(x, y, z, w, gx, gy, gz, gw)

    @triton.jit
    def _load_state(data, batch, joint, n_joints, mask):
        base = data + batch * n_joints * 8 + joint * 8
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
    def _load_flat_state(data, offset, mask):
        base = data + offset * 8
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
    def _store_flat_state(output, offset, state, mask):
        base = output + offset * 8
        tl.store(base + 0, state[0], mask=mask)
        tl.store(base + 1, state[1], mask=mask)
        tl.store(base + 2, state[2], mask=mask)
        tl.store(base + 3, state[3], mask=mask)
        tl.store(base + 4, state[4], mask=mask)
        tl.store(base + 5, state[5], mask=mask)
        tl.store(base + 6, state[6], mask=mask)
        tl.store(base + 7, state[7], mask=mask)

    @triton.jit
    def _multiply_state(state1, state2, NORMALIZE: tl.constexpr):
        t1x, t1y, t1z, q1x, q1y, q1z, q1w, s1 = state1
        t2x, t2y, t2z, q2x, q2y, q2z, q2w, s2 = state2
        if NORMALIZE:
            q1x, q1y, q1z, q1w = _normalize_quaternion(q1x, q1y, q1z, q1w)
            q2x, q2y, q2z, q2w = _normalize_quaternion(q2x, q2y, q2z, q2w)
        rtx, rty, rtz = _rotate_vector(q1x, q1y, q1z, q1w, t2x, t2y, t2z)
        qx, qy, qz, qw = _quat_multiply(q1x, q1y, q1z, q1w, q2x, q2y, q2z, q2w)
        return (
            t1x + s1 * rtx,
            t1y + s1 * rty,
            t1z + s1 * rtz,
            qx,
            qy,
            qz,
            qw,
            s1 * s2,
        )

    @triton.jit
    def _multiply_kernel(
        state1,
        state2,
        output,
        n_states,
        NORMALIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_states
        _store_flat_state(
            output,
            offsets,
            _multiply_state(
                _load_flat_state(state1, offsets, mask),
                _load_flat_state(state2, offsets, mask),
                NORMALIZE,
            ),
            mask,
        )

    @triton.jit
    def _inverse_kernel(state, output, n_states, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_states
        tx, ty, tz, qx, qy, qz, qw, s = _load_flat_state(state, offsets, mask)
        q_norm2 = tl.maximum(qx * qx + qy * qy + qz * qz + qw * qw, 1.0e-7)
        iqx = -qx / q_norm2
        iqy = -qy / q_norm2
        iqz = -qz / q_norm2
        iqw = qw / q_norm2
        niqx, niqy, niqz, niqw = _normalize_quaternion(iqx, iqy, iqz, iqw)
        rtx, rty, rtz = _rotate_vector(niqx, niqy, niqz, niqw, tx, ty, tz)
        s_inv = 1.0 / s
        _store_flat_state(
            output,
            offsets,
            (
                -s_inv * rtx,
                -s_inv * rty,
                -s_inv * rtz,
                iqx,
                iqy,
                iqz,
                iqw,
                s_inv,
            ),
            mask,
        )

    @triton.jit
    def _multiply_backward_kernel(
        state1,
        state2,
        grad_output,
        grad_state1,
        grad_state2,
        n_states,
        NORMALIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_states
        t1x, t1y, t1z, q1x, q1y, q1z, q1w, s1 = _load_flat_state(state1, offsets, mask)
        t2x, t2y, t2z, q2x, q2y, q2z, q2w, s2 = _load_flat_state(state2, offsets, mask)
        gtx, gty, gtz, gqx, gqy, gqz, gqw, gs = _load_flat_state(
            grad_output, offsets, mask
        )

        nq1x = q1x
        nq1y = q1y
        nq1z = q1z
        nq1w = q1w
        nq2x = q2x
        nq2y = q2y
        nq2z = q2z
        nq2w = q2w
        if NORMALIZE:
            nq1x, nq1y, nq1z, nq1w = _normalize_quaternion(q1x, q1y, q1z, q1w)
            nq2x, nq2y, nq2z, nq2w = _normalize_quaternion(q2x, q2y, q2z, q2w)

        rtx, rty, rtz = _rotate_vector(nq1x, nq1y, nq1z, nq1w, t2x, t2y, t2z)
        (
            gq1x_from_t,
            gq1y_from_t,
            gq1z_from_t,
            gq1w_from_t,
            gt2x,
            gt2y,
            gt2z,
        ) = _rotate_vector_backward(
            nq1x,
            nq1y,
            nq1z,
            nq1w,
            t2x,
            t2y,
            t2z,
            s1 * gtx,
            s1 * gty,
            s1 * gtz,
        )
        (
            gq1x_from_q,
            gq1y_from_q,
            gq1z_from_q,
            gq1w_from_q,
            gq2x,
            gq2y,
            gq2z,
            gq2w,
        ) = _quat_multiply_backward(
            nq1x, nq1y, nq1z, nq1w, nq2x, nq2y, nq2z, nq2w, gqx, gqy, gqz, gqw
        )

        gq1x = gq1x_from_t + gq1x_from_q
        gq1y = gq1y_from_t + gq1y_from_q
        gq1z = gq1z_from_t + gq1z_from_q
        gq1w = gq1w_from_t + gq1w_from_q
        if NORMALIZE:
            gq1x, gq1y, gq1z, gq1w = _normalize_quaternion_backward(
                q1x, q1y, q1z, q1w, gq1x, gq1y, gq1z, gq1w
            )
            gq2x, gq2y, gq2z, gq2w = _normalize_quaternion_backward(
                q2x, q2y, q2z, q2w, gq2x, gq2y, gq2z, gq2w
            )

        _store_flat_state(
            grad_state1,
            offsets,
            (
                gtx,
                gty,
                gtz,
                gq1x,
                gq1y,
                gq1z,
                gq1w,
                rtx * gtx + rty * gty + rtz * gtz + gs * s2,
            ),
            mask,
        )
        _store_flat_state(
            grad_state2,
            offsets,
            (gt2x, gt2y, gt2z, gq2x, gq2y, gq2z, gq2w, gs * s1),
            mask,
        )

    @triton.jit
    def _inverse_backward_kernel(
        state,
        grad_output,
        grad_state,
        n_states,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_states
        tx, ty, tz, qx, qy, qz, qw, s = _load_flat_state(state, offsets, mask)
        gtx, gty, gtz, gqx_out, gqy_out, gqz_out, gqw_out, gs_out = _load_flat_state(
            grad_output, offsets, mask
        )

        norm2 = qx * qx + qy * qy + qz * qz + qw * qw
        denom = tl.maximum(norm2, 1.0e-7)
        iqx = -qx / denom
        iqy = -qy / denom
        iqz = -qz / denom
        iqw = qw / denom
        niqx, niqy, niqz, niqw = _normalize_quaternion(iqx, iqy, iqz, iqw)
        rtx, rty, rtz = _rotate_vector(niqx, niqy, niqz, niqw, tx, ty, tz)
        s_inv = 1.0 / s

        (
            giqx_from_t,
            giqy_from_t,
            giqz_from_t,
            giqw_from_t,
            grad_tx,
            grad_ty,
            grad_tz,
        ) = _rotate_vector_backward(
            niqx,
            niqy,
            niqz,
            niqw,
            tx,
            ty,
            tz,
            -s_inv * gtx,
            -s_inv * gty,
            -s_inv * gtz,
        )
        giqx_from_t, giqy_from_t, giqz_from_t, giqw_from_t = (
            _normalize_quaternion_backward(
                iqx, iqy, iqz, iqw, giqx_from_t, giqy_from_t, giqz_from_t, giqw_from_t
            )
        )
        giqx = gqx_out + giqx_from_t
        giqy = gqy_out + giqy_from_t
        giqz = gqz_out + giqz_from_t
        giqw = gqw_out + giqw_from_t

        inv_denom = 1.0 / denom
        grad_qx = -giqx * inv_denom
        grad_qy = -giqy * inv_denom
        grad_qz = -giqz * inv_denom
        grad_qw = giqw * inv_denom
        grad_denom = -(giqx * (-qx) + giqy * (-qy) + giqz * (-qz) + giqw * qw) * (
            inv_denom * inv_denom
        )
        use_denom_grad = norm2 >= 1.0e-7
        grad_qx += tl.where(use_denom_grad, grad_denom * 2.0 * qx, 0.0)
        grad_qy += tl.where(use_denom_grad, grad_denom * 2.0 * qy, 0.0)
        grad_qz += tl.where(use_denom_grad, grad_denom * 2.0 * qz, 0.0)
        grad_qw += tl.where(use_denom_grad, grad_denom * 2.0 * qw, 0.0)

        grad_s_inv = -(rtx * gtx + rty * gty + rtz * gtz) + gs_out
        grad_s = -grad_s_inv * s_inv * s_inv
        _store_flat_state(
            grad_state,
            offsets,
            (grad_tx, grad_ty, grad_tz, grad_qx, grad_qy, grad_qz, grad_qw, grad_s),
            mask,
        )

    @triton.jit
    def _compute_local_state(global_state, parents, batch, joint, n_joints, mask):
        ctx, cty, ctz, cqx, cqy, cqz, cqw, cs = _load_state(
            global_state, batch, joint, n_joints, mask
        )
        parent = tl.load(parents + joint)
        has_parent = parent >= 0
        parent = tl.maximum(parent, 0)
        ptx, pty, ptz, pqx, pqy, pqz, pqw, ps = _load_state(
            global_state, batch, parent, n_joints, mask & has_parent
        )
        qix = -pqx
        qiy = -pqy
        qiz = -pqz
        qiw = pqw
        nqix, nqiy, nqiz, nqiw = _normalize_quaternion(qix, qiy, qiz, qiw)
        ncqx, ncqy, ncqz, ncqw = _normalize_quaternion(cqx, cqy, cqz, cqw)
        dx = ctx - ptx
        dy = cty - pty
        dz = ctz - ptz
        rdx, rdy, rdz = _rotate_vector(nqix, nqiy, nqiz, nqiw, dx, dy, dz)
        inv_ps = 1.0 / ps
        ltx = tl.where(has_parent, rdx * inv_ps, ctx)
        lty = tl.where(has_parent, rdy * inv_ps, cty)
        ltz = tl.where(has_parent, rdz * inv_ps, ctz)
        lqx, lqy, lqz, lqw = _quat_multiply(
            nqix, nqiy, nqiz, nqiw, ncqx, ncqy, ncqz, ncqw
        )
        lqx = tl.where(has_parent, lqx, cqx)
        lqy = tl.where(has_parent, lqy, cqy)
        lqz = tl.where(has_parent, lqz, cqz)
        lqw = tl.where(has_parent, lqw, cqw)
        ls = tl.where(has_parent, cs * inv_ps, cs)
        return ltx, lty, ltz, lqx, lqy, lqz, lqw, ls

    @triton.jit
    def _local_state_to_joint_params(
        local_tx,
        local_ty,
        local_tz,
        local_qx,
        local_qy,
        local_qz,
        local_qw,
        local_s,
        offsets,
        prerotations,
        joint,
    ):
        offset_base = offsets + joint * 3
        pre_base = prerotations + joint * 4
        pre_x = tl.load(pre_base + 0)
        pre_y = tl.load(pre_base + 1)
        pre_z = tl.load(pre_base + 2)
        pre_w = tl.load(pre_base + 3)
        local_qx, local_qy, local_qz, local_qw = _normalize_quaternion(
            local_qx, local_qy, local_qz, local_qw
        )
        adj_x, adj_y, adj_z, adj_w = _quat_multiply(
            -pre_x,
            -pre_y,
            -pre_z,
            pre_w,
            local_qx,
            local_qy,
            local_qz,
            local_qw,
        )
        rx, ry, rz = _quaternion_to_xyz_euler(adj_x, adj_y, adj_z, adj_w)
        return (
            local_tx - tl.load(offset_base + 0),
            local_ty - tl.load(offset_base + 1),
            local_tz - tl.load(offset_base + 2),
            rx,
            ry,
            rz,
            tl.log2(local_s),
        )

    @triton.jit
    def _forward_kernel(
        global_state,
        offsets,
        prerotations,
        parents,
        output,
        batch_size,
        n_joints,
        BLOCK_BATCH: tl.constexpr,
    ):
        batch = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = batch < batch_size
        joint = 0
        while joint < n_joints:
            local = _compute_local_state(
                global_state, parents, batch, joint, n_joints, mask
            )
            params = _local_state_to_joint_params(
                local[0],
                local[1],
                local[2],
                local[3],
                local[4],
                local[5],
                local[6],
                local[7],
                offsets,
                prerotations,
                joint,
            )
            out_base = output + batch * n_joints * 7 + joint * 7
            tl.store(out_base + 0, params[0], mask=mask)
            tl.store(out_base + 1, params[1], mask=mask)
            tl.store(out_base + 2, params[2], mask=mask)
            tl.store(out_base + 3, params[3], mask=mask)
            tl.store(out_base + 4, params[4], mask=mask)
            tl.store(out_base + 5, params[5], mask=mask)
            tl.store(out_base + 6, params[6], mask=mask)
            joint += 1

    @triton.jit
    def _add_grad(grad_global_state, batch, joint, n_joints, values, mask):
        base = grad_global_state + batch * n_joints * 8 + joint * 8
        tl.store(
            base + 0, tl.load(base + 0, mask=mask, other=0.0) + values[0], mask=mask
        )
        tl.store(
            base + 1, tl.load(base + 1, mask=mask, other=0.0) + values[1], mask=mask
        )
        tl.store(
            base + 2, tl.load(base + 2, mask=mask, other=0.0) + values[2], mask=mask
        )
        tl.store(
            base + 3, tl.load(base + 3, mask=mask, other=0.0) + values[3], mask=mask
        )
        tl.store(
            base + 4, tl.load(base + 4, mask=mask, other=0.0) + values[4], mask=mask
        )
        tl.store(
            base + 5, tl.load(base + 5, mask=mask, other=0.0) + values[5], mask=mask
        )
        tl.store(
            base + 6, tl.load(base + 6, mask=mask, other=0.0) + values[6], mask=mask
        )
        tl.store(
            base + 7, tl.load(base + 7, mask=mask, other=0.0) + values[7], mask=mask
        )

    @triton.jit
    def _backward_one_joint(
        global_state,
        offsets,
        prerotations,
        parents,
        grad_output,
        grad_global_state,
        batch,
        joint,
        n_joints,
        mask,
    ):
        local = _compute_local_state(
            global_state, parents, batch, joint, n_joints, mask
        )
        ltx, lty, ltz, lqx, lqy, lqz, lqw, ls = local
        grad_base = grad_output + batch * n_joints * 7 + joint * 7
        gltx = tl.load(grad_base + 0, mask=mask, other=0.0)
        glty = tl.load(grad_base + 1, mask=mask, other=0.0)
        gltz = tl.load(grad_base + 2, mask=mask, other=0.0)
        grx = tl.load(grad_base + 3, mask=mask, other=0.0)
        gry = tl.load(grad_base + 4, mask=mask, other=0.0)
        grz = tl.load(grad_base + 5, mask=mask, other=0.0)
        gls = tl.load(grad_base + 6, mask=mask, other=0.0) / (ls * _LN2)

        pre_base = prerotations + joint * 4
        pre_x = tl.load(pre_base + 0)
        pre_y = tl.load(pre_base + 1)
        pre_z = tl.load(pre_base + 2)
        pre_w = tl.load(pre_base + 3)
        nlqx, nlqy, nlqz, nlqw = _normalize_quaternion(lqx, lqy, lqz, lqw)
        adj_x, adj_y, adj_z, adj_w = _quat_multiply(
            -pre_x, -pre_y, -pre_z, pre_w, nlqx, nlqy, nlqz, nlqw
        )
        gadj_x, gadj_y, gadj_z, gadj_w = _quaternion_to_xyz_euler_backward(
            adj_x, adj_y, adj_z, adj_w, grx, gry, grz
        )
        _, _, _, _, glqx, glqy, glqz, glqw = _quat_multiply_backward(
            -pre_x,
            -pre_y,
            -pre_z,
            pre_w,
            lqx,
            lqy,
            lqz,
            lqw,
            gadj_x,
            gadj_y,
            gadj_z,
            gadj_w,
        )
        glqx, glqy, glqz, glqw = _normalize_quaternion_backward(
            lqx, lqy, lqz, lqw, glqx, glqy, glqz, glqw
        )

        parent = tl.load(parents + joint)
        has_parent = parent >= 0
        parent = tl.maximum(parent, 0)
        ctx, cty, ctz, cqx, cqy, cqz, cqw, cs = _load_state(
            global_state, batch, joint, n_joints, mask
        )
        ptx, pty, ptz, pqx, pqy, pqz, pqw, ps = _load_state(
            global_state, batch, parent, n_joints, mask & has_parent
        )
        qix = -pqx
        qiy = -pqy
        qiz = -pqz
        qiw = pqw
        nqix, nqiy, nqiz, nqiw = _normalize_quaternion(qix, qiy, qiz, qiw)
        ncqx, ncqy, ncqz, ncqw = _normalize_quaternion(cqx, cqy, cqz, cqw)
        dx = ctx - ptx
        dy = cty - pty
        dz = ctz - ptz

        rdx, rdy, rdz = _rotate_vector(nqix, nqiy, nqiz, nqiw, dx, dy, dz)
        inv_ps = 1.0 / ps
        grdx = gltx * inv_ps
        grdy = glty * inv_ps
        grdz = gltz * inv_ps
        gnqix_t, gnqiy_t, gnqiz_t, gnqiw_t, gdx, gdy, gdz = _rotate_vector_backward(
            nqix, nqiy, nqiz, nqiw, dx, dy, dz, grdx, grdy, grdz
        )
        gps = -(gltx * rdx + glty * rdy + gltz * rdz) * inv_ps * inv_ps
        gnqix_q, gnqiy_q, gnqiz_q, gnqiw_q, gncqx, gncqy, gncqz, gncqw = (
            _quat_multiply_backward(
                nqix,
                nqiy,
                nqiz,
                nqiw,
                ncqx,
                ncqy,
                ncqz,
                ncqw,
                glqx,
                glqy,
                glqz,
                glqw,
            )
        )
        gcqx, gcqy, gcqz, gcqw = _normalize_quaternion_backward(
            cqx, cqy, cqz, cqw, gncqx, gncqy, gncqz, gncqw
        )
        gqix, gqiy, gqiz, gqiw = _normalize_quaternion_backward(
            qix,
            qiy,
            qiz,
            qiw,
            gnqix_t + gnqix_q,
            gnqiy_t + gnqiy_q,
            gnqiz_t + gnqiz_q,
            gnqiw_t + gnqiw_q,
        )
        gcs = gls * inv_ps
        gps += -gls * cs * inv_ps * inv_ps

        child_values = (
            tl.where(has_parent, gdx, gltx),
            tl.where(has_parent, gdy, glty),
            tl.where(has_parent, gdz, gltz),
            tl.where(has_parent, gcqx, glqx),
            tl.where(has_parent, gcqy, glqy),
            tl.where(has_parent, gcqz, glqz),
            tl.where(has_parent, gcqw, glqw),
            tl.where(has_parent, gcs, gls),
        )
        parent_values = (
            -gdx,
            -gdy,
            -gdz,
            -gqix,
            -gqiy,
            -gqiz,
            gqiw,
            gps,
        )
        _add_grad(grad_global_state, batch, joint, n_joints, child_values, mask)
        _add_grad(
            grad_global_state,
            batch,
            parent,
            n_joints,
            parent_values,
            mask & has_parent,
        )

    @triton.jit
    def _backward_kernel(
        global_state,
        offsets,
        prerotations,
        parents,
        grad_output,
        grad_global_state,
        batch_size,
        n_joints,
        BLOCK_BATCH: tl.constexpr,
    ):
        batch = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = batch < batch_size
        joint = n_joints - 1
        while joint >= 0:
            _backward_one_joint(
                global_state,
                offsets,
                prerotations,
                parents,
                grad_output,
                grad_global_state,
                batch,
                joint,
                n_joints,
                mask,
            )
            joint -= 1


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError(
            "Triton skeleton-state backend requested, but Triton is not available "
            "in this build."
        )


def _validate_skel_state(state: torch.Tensor) -> None:
    if not state.is_cuda:
        raise RuntimeError("Triton skel_state backend requested, but input is on CPU.")
    if state.dtype != torch.float32:
        raise RuntimeError("Triton skel_state backend currently only supports float32.")
    if state.ndim == 0 or state.shape[-1] != 8:
        raise RuntimeError("Skeleton state tensor must have shape [..., 8].")


def _launch_multiply(
    state1: torch.Tensor,
    state2: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    _validate_skel_state(state1)
    _validate_skel_state(state2)
    _require_triton()
    state1, state2 = torch.broadcast_tensors(state1, state2)
    shape = state2.shape
    state1 = state1.contiguous().reshape(-1, 8)
    state2 = state2.contiguous().reshape(-1, 8)
    output = torch.empty_like(state2)
    n_states = state2.shape[0]
    grid = (triton.cdiv(n_states, _BLOCK_SIZE),)
    _multiply_kernel[grid](
        state1,
        state2,
        output,
        n_states,
        NORMALIZE=normalize,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output.reshape(shape)


def _launch_inverse(state: torch.Tensor) -> torch.Tensor:
    _validate_skel_state(state)
    _require_triton()
    shape = state.shape
    state = state.contiguous().reshape(-1, 8)
    output = torch.empty_like(state)
    n_states = state.shape[0]
    grid = (triton.cdiv(n_states, _BLOCK_SIZE),)
    _inverse_kernel[grid](state, output, n_states, BLOCK_SIZE=_BLOCK_SIZE)
    return output.reshape(shape)


def _launch_multiply_backward(
    state1: torch.Tensor,
    state2: torch.Tensor,
    grad_output: torch.Tensor,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    state1 = state1.contiguous().reshape(-1, 8)
    state2 = state2.contiguous().reshape(-1, 8)
    grad_output = grad_output.contiguous().reshape(-1, 8)
    grad_state1 = torch.empty_like(state1)
    grad_state2 = torch.empty_like(state2)
    n_states = state2.shape[0]
    grid = (triton.cdiv(n_states, _BLOCK_SIZE),)
    _multiply_backward_kernel[grid](
        state1,
        state2,
        grad_output,
        grad_state1,
        grad_state2,
        n_states,
        NORMALIZE=normalize,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return grad_state1, grad_state2


def _launch_inverse_backward(
    state: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    state = state.contiguous().reshape(-1, 8)
    grad_output = grad_output.contiguous().reshape(-1, 8)
    grad_state = torch.empty_like(state)
    n_states = state.shape[0]
    grid = (triton.cdiv(n_states, _BLOCK_SIZE),)
    _inverse_backward_kernel[grid](
        state,
        grad_output,
        grad_state,
        n_states,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return grad_state


class _Multiply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        state1: torch.Tensor,
        state2: torch.Tensor,
        normalize: bool,
    ) -> torch.Tensor:
        shape = state2.shape
        ctx.normalize = normalize
        ctx.shape = shape
        ctx.save_for_backward(state1, state2)
        return _launch_multiply(state1, state2, normalize)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        state1, state2 = ctx.saved_tensors
        grad_state1, grad_state2 = _launch_multiply_backward(
            state1,
            state2,
            grad_output,
            ctx.normalize,
        )
        return grad_state1.reshape(ctx.shape), grad_state2.reshape(ctx.shape), None


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        state: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(state)
        return _launch_inverse(state)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        (state,) = ctx.saved_tensors
        grad_state = _launch_inverse_backward(
            state,
            grad_output,
        )
        return (grad_state.reshape(state.shape),)


def multiply(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    state1, state2 = torch.broadcast_tensors(state1, state2)
    return _Multiply.apply(state1, state2, True)


def multiply_assume_normalized(
    state1: torch.Tensor, state2: torch.Tensor
) -> torch.Tensor:
    state1, state2 = torch.broadcast_tensors(state1, state2)
    return _Multiply.apply(state1, state2, False)


def inverse(state: torch.Tensor) -> torch.Tensor:
    return _Inverse.apply(state)


def _validate_inputs(
    skel_state: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
) -> None:
    if not skel_state.is_cuda:
        raise RuntimeError(
            "Triton skeleton-state backend requested, but input is on CPU."
        )
    if skel_state.dtype != torch.float32:
        raise RuntimeError("Triton skeleton-state backend only supports float32.")
    if skel_state.ndim < 2 or skel_state.shape[-1] != 8:
        raise RuntimeError("skel_state must have shape [..., num_joints, 8].")
    if joint_translation_offsets.device != skel_state.device:
        raise RuntimeError("joint_translation_offsets must be on the same device.")
    if joint_prerotations.device != skel_state.device:
        raise RuntimeError("joint_prerotations must be on the same device.")
    if joint_parents.device != skel_state.device:
        raise RuntimeError("joint_parents must be on the same device.")
    if joint_translation_offsets.dtype != torch.float32:
        raise RuntimeError("joint_translation_offsets must be float32.")
    if joint_prerotations.dtype != torch.float32:
        raise RuntimeError("joint_prerotations must be float32.")
    if joint_parents.dtype != torch.int32:
        raise RuntimeError("joint_parents must be int32.")


def _launch_forward(
    skel_state: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    _validate_inputs(
        skel_state,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
    )
    num_joints = skel_state.shape[-2]
    batch_size = skel_state.numel() // (num_joints * 8)
    output = torch.empty(
        (*skel_state.shape[:-2], num_joints * 7),
        device=skel_state.device,
        dtype=skel_state.dtype,
    )
    grid = (triton.cdiv(batch_size, _BLOCK_BATCH),)
    _forward_kernel[grid](
        skel_state,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        output,
        batch_size,
        num_joints,
        BLOCK_BATCH=_BLOCK_BATCH,
    )
    return output


def _launch_backward(
    skel_state: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    grad_output = grad_output.contiguous()
    grad_skel_state = torch.zeros_like(skel_state)
    num_joints = skel_state.shape[-2]
    batch_size = skel_state.numel() // (num_joints * 8)
    grid = (triton.cdiv(batch_size, _BLOCK_BATCH),)
    _backward_kernel[grid](
        skel_state,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        grad_output,
        grad_skel_state,
        batch_size,
        num_joints,
        BLOCK_BATCH=_BLOCK_BATCH,
    )
    return grad_skel_state


class _SkeletonStateToJointParameters(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        skel_state: torch.Tensor,
        joint_translation_offsets: torch.Tensor,
        joint_prerotations: torch.Tensor,
        joint_parents: torch.Tensor,
    ) -> torch.Tensor:
        skel_state = skel_state.contiguous()
        joint_translation_offsets = joint_translation_offsets.contiguous()
        joint_prerotations = joint_prerotations.contiguous()
        joint_parents = joint_parents.contiguous()
        output = _launch_forward(
            skel_state,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
        )
        ctx.save_for_backward(
            skel_state,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        (
            skel_state,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
        ) = ctx.saved_tensors
        grad_skel_state = _launch_backward(
            skel_state,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            grad_output,
        )
        return grad_skel_state, None, None, None


def skeleton_state_to_joint_parameters(
    skel_state: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
) -> torch.Tensor:
    return _SkeletonStateToJointParameters.apply(
        skel_state,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
    )
