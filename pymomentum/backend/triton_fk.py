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
except ImportError:
    triton = None
    tl = None


_BLOCK_BATCH = 128
_LN2 = tl.constexpr(math.log(2.0)) if tl is not None else math.log(2.0)
if triton is not None and tl is not None:  # noqa: C901
    from pymomentum.backend.triton_common import (  # @manual=:backend_triton
        _quat_multiply,
        _quat_multiply_backward,
        _rotate_vector,
        _rotate_vector_backward,
    )

    @triton.jit
    def _euler_to_quaternion(rx, ry, rz):
        cy = tl.cos(rz * 0.5)
        sy = tl.sin(rz * 0.5)
        cp = tl.cos(ry * 0.5)
        sp = tl.sin(ry * 0.5)
        cr = tl.cos(rx * 0.5)
        sr = tl.sin(rx * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    @triton.jit
    def _euler_backward(rx, ry, rz, gx, gy, gz, gw):
        cy = tl.cos(rz * 0.5)
        sy = tl.sin(rz * 0.5)
        cp = tl.cos(ry * 0.5)
        sp = tl.sin(ry * 0.5)
        cr = tl.cos(rx * 0.5)
        sr = tl.sin(rx * 0.5)

        grad_rx = (
            gx * (0.5 * cr * cp * cy + 0.5 * sr * sp * sy)
            + gy * (-0.5 * sr * sp * cy + 0.5 * cr * cp * sy)
            + gz * (-0.5 * sr * cp * sy - 0.5 * cr * sp * cy)
            + gw * (-0.5 * sr * cp * cy + 0.5 * cr * sp * sy)
        )
        grad_ry = (
            gx * (-0.5 * sr * sp * cy - 0.5 * cr * cp * sy)
            + gy * (0.5 * cr * cp * cy - 0.5 * sr * sp * sy)
            + gz * (-0.5 * cr * sp * sy - 0.5 * sr * cp * cy)
            + gw * (-0.5 * cr * sp * cy + 0.5 * sr * cp * sy)
        )
        grad_rz = (
            gx * (-0.5 * sr * cp * sy - 0.5 * cr * sp * cy)
            + gy * (-0.5 * cr * sp * sy + 0.5 * sr * cp * cy)
            + gz * (0.5 * cr * cp * cy + 0.5 * sr * sp * sy)
            + gw * (-0.5 * cr * cp * sy + 0.5 * sr * sp * cy)
        )
        return grad_rx, grad_ry, grad_rz

    @triton.jit
    def _rotate_vector_backward_parent(qx, qy, qz, qw, vx, vy, vz, gx, gy, gz):
        grad_qx = (
            (2.0 * qy * vy + 2.0 * qz * vz) * gx
            + (2.0 * qy * vx - 4.0 * qx * vy - 2.0 * qw * vz) * gy
            + (2.0 * qz * vx + 2.0 * qw * vy - 4.0 * qx * vz) * gz
        )
        grad_qy = (
            (-4.0 * qy * vx + 2.0 * qx * vy + 2.0 * qw * vz) * gx
            + (2.0 * qx * vx + 2.0 * qz * vz) * gy
            + (-2.0 * qw * vx + 2.0 * qz * vy - 4.0 * qy * vz) * gz
        )
        grad_qz = (
            (-4.0 * qz * vx - 2.0 * qw * vy + 2.0 * qx * vz) * gx
            + (2.0 * qw * vx - 4.0 * qz * vy + 2.0 * qy * vz) * gy
            + (2.0 * qx * vx + 2.0 * qy * vy) * gz
        )
        grad_qw = (
            (-2.0 * qz * vy + 2.0 * qy * vz) * gx
            + (2.0 * qz * vx - 2.0 * qx * vz) * gy
            + (-2.0 * qy * vx + 2.0 * qx * vy) * gz
        )
        return grad_qx, grad_qy, grad_qz, grad_qw

    @triton.jit
    def _quat_multiply_backward_parent(x1, y1, z1, w1, x2, y2, z2, w2, gx, gy, gz, gw):
        return (
            gx * w2 - gy * z2 + gz * y2 - gw * x2,
            gx * z2 + gy * w2 - gz * x2 - gw * y2,
            -gx * y2 + gy * x2 + gz * w2 - gw * z2,
            gx * x2 + gy * y2 + gz * z2 + gw * w2,
        )

    @triton.jit
    def _load_local_state(
        joint_params, offsets, prerotations, batch, joint, n_joints, mask
    ):
        param_base = joint_params + batch * n_joints * 7 + joint * 7
        offset_base = offsets + joint * 3
        pre_base = prerotations + joint * 4

        tx = tl.load(offset_base + 0) + tl.load(param_base + 0, mask=mask, other=0.0)
        ty = tl.load(offset_base + 1) + tl.load(param_base + 1, mask=mask, other=0.0)
        tz = tl.load(offset_base + 2) + tl.load(param_base + 2, mask=mask, other=0.0)

        rx = tl.load(param_base + 3, mask=mask, other=0.0)
        ry = tl.load(param_base + 4, mask=mask, other=0.0)
        rz = tl.load(param_base + 5, mask=mask, other=0.0)
        eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)

        pqx = tl.load(pre_base + 0)
        pqy = tl.load(pre_base + 1)
        pqz = tl.load(pre_base + 2)
        pqw = tl.load(pre_base + 3)
        qx, qy, qz, qw = _quat_multiply(pqx, pqy, pqz, pqw, eqx, eqy, eqz, eqw)
        scale = tl.exp2(tl.load(param_base + 6, mask=mask, other=0.0))
        return tx, ty, tz, qx, qy, qz, qw, scale

    @triton.jit
    def _forward_one_joint(
        joint_params,
        offsets,
        prerotations,
        parents,
        output,
        batch,
        joint,
        n_joints,
        mask,
    ):
        ltx, lty, ltz, lqx, lqy, lqz, lqw, ls = _load_local_state(
            joint_params, offsets, prerotations, batch, joint, n_joints, mask
        )

        parent = tl.load(parents + joint)
        has_parent = parent >= 0
        parent = tl.maximum(parent, 0)
        parent_base = output + batch * n_joints * 8 + parent * 8
        parent_mask = mask & has_parent
        ptx = tl.load(parent_base + 0, mask=parent_mask, other=0.0)
        pty = tl.load(parent_base + 1, mask=parent_mask, other=0.0)
        ptz = tl.load(parent_base + 2, mask=parent_mask, other=0.0)
        pqx = tl.load(parent_base + 3, mask=parent_mask, other=0.0)
        pqy = tl.load(parent_base + 4, mask=parent_mask, other=0.0)
        pqz = tl.load(parent_base + 5, mask=parent_mask, other=0.0)
        pqw = tl.load(parent_base + 6, mask=parent_mask, other=1.0)
        ps = tl.load(parent_base + 7, mask=parent_mask, other=1.0)

        rtx, rty, rtz = _rotate_vector(pqx, pqy, pqz, pqw, ltx, lty, ltz)
        gtx = ptx + ps * rtx
        gty = pty + ps * rty
        gtz = ptz + ps * rtz
        gqx, gqy, gqz, gqw = _quat_multiply(pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw)
        gs = ps * ls

        out_base = output + batch * n_joints * 8 + joint * 8
        tl.store(out_base + 0, gtx, mask=mask)
        tl.store(out_base + 1, gty, mask=mask)
        tl.store(out_base + 2, gtz, mask=mask)
        tl.store(out_base + 3, gqx, mask=mask)
        tl.store(out_base + 4, gqy, mask=mask)
        tl.store(out_base + 5, gqz, mask=mask)
        tl.store(out_base + 6, gqw, mask=mask)
        tl.store(out_base + 7, gs, mask=mask)

    @triton.jit
    def _forward_kernel(
        joint_params,
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
            _forward_one_joint(
                joint_params,
                offsets,
                prerotations,
                parents,
                output,
                batch,
                joint,
                n_joints,
                mask,
            )
            joint += 1

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
            tl.load(base + 6, mask=mask, other=0.0),
            tl.load(base + 7, mask=mask, other=0.0),
        )

    @triton.jit
    def _load_parent_state(parents, global_state, batch, joint, n_joints, mask):
        parent = tl.load(parents + joint)
        has_parent = parent >= 0
        parent = tl.maximum(parent, 0)
        parent_mask = mask & has_parent
        parent_base = global_state + batch * n_joints * 8 + parent * 8
        return (
            parent,
            parent_mask,
            tl.load(parent_base + 3, mask=parent_mask, other=0.0),
            tl.load(parent_base + 4, mask=parent_mask, other=0.0),
            tl.load(parent_base + 5, mask=parent_mask, other=0.0),
            tl.load(parent_base + 6, mask=parent_mask, other=1.0),
            tl.load(parent_base + 7, mask=parent_mask, other=1.0),
        )

    @triton.jit
    def _compute_active_joint_gradients(
        pqx,
        pqy,
        pqz,
        pqw,
        ltx,
        lty,
        ltz,
        lqx,
        lqy,
        lqz,
        lqw,
        gqx,
        gqy,
        gqz,
        gqw,
        scaled_gtx,
        scaled_gty,
        scaled_gtz,
    ):
        (
            grad_parent_qx_t,
            grad_parent_qy_t,
            grad_parent_qz_t,
            grad_parent_qw_t,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
        ) = _rotate_vector_backward(
            pqx, pqy, pqz, pqw, ltx, lty, ltz, scaled_gtx, scaled_gty, scaled_gtz
        )
        (
            grad_parent_qx_q,
            grad_parent_qy_q,
            grad_parent_qz_q,
            grad_parent_qw_q,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
        ) = _quat_multiply_backward(
            pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw, gqx, gqy, gqz, gqw
        )
        return (
            grad_parent_qx_t + grad_parent_qx_q,
            grad_parent_qy_t + grad_parent_qy_q,
            grad_parent_qz_t + grad_parent_qz_q,
            grad_parent_qw_t + grad_parent_qw_q,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
        )

    @triton.jit
    def _compute_inactive_joint_gradients(
        pqx,
        pqy,
        pqz,
        pqw,
        ltx,
        lty,
        ltz,
        lqx,
        lqy,
        lqz,
        lqw,
        gqx,
        gqy,
        gqz,
        gqw,
        scaled_gtx,
        scaled_gty,
        scaled_gtz,
    ):
        grad_parent_qx_t, grad_parent_qy_t, grad_parent_qz_t, grad_parent_qw_t = (
            _rotate_vector_backward_parent(
                pqx, pqy, pqz, pqw, ltx, lty, ltz, scaled_gtx, scaled_gty, scaled_gtz
            )
        )
        grad_parent_qx_q, grad_parent_qy_q, grad_parent_qz_q, grad_parent_qw_q = (
            _quat_multiply_backward_parent(
                pqx, pqy, pqz, pqw, lqx, lqy, lqz, lqw, gqx, gqy, gqz, gqw
            )
        )
        return (
            grad_parent_qx_t + grad_parent_qx_q,
            grad_parent_qy_t + grad_parent_qy_q,
            grad_parent_qz_t + grad_parent_qz_q,
            grad_parent_qw_t + grad_parent_qw_q,
            scaled_gtx * 0.0,
            scaled_gty * 0.0,
            scaled_gtz * 0.0,
            gqx * 0.0,
            gqy * 0.0,
            gqz * 0.0,
            gqw * 0.0,
        )

    @triton.jit
    def _compute_backward_gradients(
        pqx,
        pqy,
        pqz,
        pqw,
        ps,
        ltx,
        lty,
        ltz,
        lqx,
        lqy,
        lqz,
        lqw,
        ls,
        gtx,
        gty,
        gtz,
        gqx,
        gqy,
        gqz,
        gqw,
        gs,
        compute_joint_grad,
    ):
        rtx, rty, rtz = _rotate_vector(pqx, pqy, pqz, pqw, ltx, lty, ltz)
        scaled_gtx = ps * gtx
        scaled_gty = ps * gty
        scaled_gtz = ps * gtz
        if compute_joint_grad:
            (
                grad_parent_qx,
                grad_parent_qy,
                grad_parent_qz,
                grad_parent_qw,
                grad_local_tx,
                grad_local_ty,
                grad_local_tz,
                grad_local_qx,
                grad_local_qy,
                grad_local_qz,
                grad_local_qw,
            ) = _compute_active_joint_gradients(
                pqx,
                pqy,
                pqz,
                pqw,
                ltx,
                lty,
                ltz,
                lqx,
                lqy,
                lqz,
                lqw,
                gqx,
                gqy,
                gqz,
                gqw,
                scaled_gtx,
                scaled_gty,
                scaled_gtz,
            )
        else:
            (
                grad_parent_qx,
                grad_parent_qy,
                grad_parent_qz,
                grad_parent_qw,
                grad_local_tx,
                grad_local_ty,
                grad_local_tz,
                grad_local_qx,
                grad_local_qy,
                grad_local_qz,
                grad_local_qw,
            ) = _compute_inactive_joint_gradients(
                pqx,
                pqy,
                pqz,
                pqw,
                ltx,
                lty,
                ltz,
                lqx,
                lqy,
                lqz,
                lqw,
                gqx,
                gqy,
                gqz,
                gqw,
                scaled_gtx,
                scaled_gty,
                scaled_gtz,
            )
        return (
            gtx,
            gty,
            gtz,
            grad_parent_qx,
            grad_parent_qy,
            grad_parent_qz,
            grad_parent_qw,
            rtx * gtx + rty * gty + rtz * gtz + ls * gs,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
            ps * gs,
        )

    @triton.jit
    def _accumulate_parent_gradients(
        grad_work,
        batch,
        parent,
        n_joints,
        parent_mask,
        grad_parent_tx,
        grad_parent_ty,
        grad_parent_tz,
        grad_parent_qx,
        grad_parent_qy,
        grad_parent_qz,
        grad_parent_qw,
        grad_parent_s,
    ):
        parent_grad_base = grad_work + batch * n_joints * 8 + parent * 8
        tl.store(
            parent_grad_base + 0,
            tl.load(parent_grad_base + 0, mask=parent_mask, other=0.0) + grad_parent_tx,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 1,
            tl.load(parent_grad_base + 1, mask=parent_mask, other=0.0) + grad_parent_ty,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 2,
            tl.load(parent_grad_base + 2, mask=parent_mask, other=0.0) + grad_parent_tz,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 3,
            tl.load(parent_grad_base + 3, mask=parent_mask, other=0.0) + grad_parent_qx,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 4,
            tl.load(parent_grad_base + 4, mask=parent_mask, other=0.0) + grad_parent_qy,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 5,
            tl.load(parent_grad_base + 5, mask=parent_mask, other=0.0) + grad_parent_qz,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 6,
            tl.load(parent_grad_base + 6, mask=parent_mask, other=0.0) + grad_parent_qw,
            mask=parent_mask,
        )
        tl.store(
            parent_grad_base + 7,
            tl.load(parent_grad_base + 7, mask=parent_mask, other=0.0) + grad_parent_s,
            mask=parent_mask,
        )

    @triton.jit
    def _store_joint_param_gradients(
        joint_params,
        prerotations,
        grad_joint_params,
        batch,
        joint,
        n_joints,
        mask,
        compute_joint_grad,
        grad_local_tx,
        grad_local_ty,
        grad_local_tz,
        grad_local_qx,
        grad_local_qy,
        grad_local_qz,
        grad_local_qw,
        grad_local_s,
        ls,
    ):
        if compute_joint_grad:
            param_base = joint_params + batch * n_joints * 7 + joint * 7
            rx = tl.load(param_base + 3, mask=mask, other=0.0)
            ry = tl.load(param_base + 4, mask=mask, other=0.0)
            rz = tl.load(param_base + 5, mask=mask, other=0.0)
            eqx, eqy, eqz, eqw = _euler_to_quaternion(rx, ry, rz)

            pre_base = prerotations + joint * 4
            pqx_pre = tl.load(pre_base + 0)
            pqy_pre = tl.load(pre_base + 1)
            pqz_pre = tl.load(pre_base + 2)
            pqw_pre = tl.load(pre_base + 3)
            _, _, _, _, grad_eqx, grad_eqy, grad_eqz, grad_eqw = (
                _quat_multiply_backward(
                    pqx_pre,
                    pqy_pre,
                    pqz_pre,
                    pqw_pre,
                    eqx,
                    eqy,
                    eqz,
                    eqw,
                    grad_local_qx,
                    grad_local_qy,
                    grad_local_qz,
                    grad_local_qw,
                )
            )
            grad_rx, grad_ry, grad_rz = _euler_backward(
                rx, ry, rz, grad_eqx, grad_eqy, grad_eqz, grad_eqw
            )

            grad_base = grad_joint_params + batch * n_joints * 7 + joint * 7
            tl.store(grad_base + 0, grad_local_tx, mask=mask)
            tl.store(grad_base + 1, grad_local_ty, mask=mask)
            tl.store(grad_base + 2, grad_local_tz, mask=mask)
            tl.store(grad_base + 3, grad_rx, mask=mask)
            tl.store(grad_base + 4, grad_ry, mask=mask)
            tl.store(grad_base + 5, grad_rz, mask=mask)
            tl.store(grad_base + 6, grad_local_s * ls * _LN2, mask=mask)

    @triton.jit
    def _backward_one_joint(
        joint_params,
        offsets,
        prerotations,
        parents,
        active_joints,
        global_state,
        grad_work,
        grad_joint_params,
        batch,
        joint,
        n_joints,
        mask,
        HAS_ACTIVE_JOINTS: tl.constexpr,
    ):
        ltx, lty, ltz, lqx, lqy, lqz, lqw, ls = _load_local_state(
            joint_params, offsets, prerotations, batch, joint, n_joints, mask
        )
        gtx, gty, gtz, gqx, gqy, gqz, gqw, gs = _load_state(
            grad_work, batch, joint, n_joints, mask
        )

        parent, parent_mask, pqx, pqy, pqz, pqw, ps = _load_parent_state(
            parents, global_state, batch, joint, n_joints, mask
        )
        compute_joint_grad = True
        if HAS_ACTIVE_JOINTS:
            compute_joint_grad = tl.load(active_joints + joint)

        (
            grad_parent_tx,
            grad_parent_ty,
            grad_parent_tz,
            grad_parent_qx,
            grad_parent_qy,
            grad_parent_qz,
            grad_parent_qw,
            grad_parent_s,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
            grad_local_s,
        ) = _compute_backward_gradients(
            pqx,
            pqy,
            pqz,
            pqw,
            ps,
            ltx,
            lty,
            ltz,
            lqx,
            lqy,
            lqz,
            lqw,
            ls,
            gtx,
            gty,
            gtz,
            gqx,
            gqy,
            gqz,
            gqw,
            gs,
            compute_joint_grad,
        )
        _accumulate_parent_gradients(
            grad_work,
            batch,
            parent,
            n_joints,
            parent_mask,
            grad_parent_tx,
            grad_parent_ty,
            grad_parent_tz,
            grad_parent_qx,
            grad_parent_qy,
            grad_parent_qz,
            grad_parent_qw,
            grad_parent_s,
        )
        _store_joint_param_gradients(
            joint_params,
            prerotations,
            grad_joint_params,
            batch,
            joint,
            n_joints,
            mask,
            compute_joint_grad,
            grad_local_tx,
            grad_local_ty,
            grad_local_tz,
            grad_local_qx,
            grad_local_qy,
            grad_local_qz,
            grad_local_qw,
            grad_local_s,
            ls,
        )

    @triton.jit
    def _backward_kernel(
        joint_params,
        offsets,
        prerotations,
        parents,
        active_joints,
        global_state,
        grad_work,
        grad_joint_params,
        batch_size,
        n_joints,
        HAS_ACTIVE_JOINTS: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
    ):
        batch = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = batch < batch_size
        joint = n_joints - 1
        while joint >= 0:
            _backward_one_joint(
                joint_params,
                offsets,
                prerotations,
                parents,
                active_joints,
                global_state,
                grad_work,
                grad_joint_params,
                batch,
                joint,
                n_joints,
                mask,
                HAS_ACTIVE_JOINTS,
            )
            joint -= 1

    @triton.jit
    def _local_forward_kernel(
        joint_params,
        offsets,
        prerotations,
        output,
        batch_size,
        n_joints,
        BLOCK_BATCH: tl.constexpr,
    ):
        # Local skeleton state: the per-joint state from _load_local_state, WITHOUT
        # the parent composition that _forward_one_joint applies (i.e. the global FK
        # before the prefix product up the kinematic tree).
        batch = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = batch < batch_size
        joint = 0
        while joint < n_joints:
            tx, ty, tz, qx, qy, qz, qw, s = _load_local_state(
                joint_params, offsets, prerotations, batch, joint, n_joints, mask
            )
            out_base = output + batch * n_joints * 8 + joint * 8
            tl.store(out_base + 0, tx, mask=mask)
            tl.store(out_base + 1, ty, mask=mask)
            tl.store(out_base + 2, tz, mask=mask)
            tl.store(out_base + 3, qx, mask=mask)
            tl.store(out_base + 4, qy, mask=mask)
            tl.store(out_base + 5, qz, mask=mask)
            tl.store(out_base + 6, qw, mask=mask)
            tl.store(out_base + 7, s, mask=mask)
            joint += 1

    @triton.jit
    def _local_backward_kernel(
        joint_params,
        offsets,
        prerotations,
        grad_output,
        grad_joint_params,
        batch_size,
        n_joints,
        BLOCK_BATCH: tl.constexpr,
    ):
        # Backward of the local FK: there is no tree, so grad w.r.t. the local
        # skeleton state IS grad_output per joint. _store_joint_param_gradients
        # propagates it to joint_params (translation identity, rotation through the
        # prerotation*euler quaternion multiply + euler, scale through exp2).
        batch = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        mask = batch < batch_size
        joint = 0
        while joint < n_joints:
            _, _, _, _, _, _, _, ls = _load_local_state(
                joint_params, offsets, prerotations, batch, joint, n_joints, mask
            )
            gtx, gty, gtz, gqx, gqy, gqz, gqw, gs = _load_state(
                grad_output, batch, joint, n_joints, mask
            )
            _store_joint_param_gradients(
                joint_params,
                prerotations,
                grad_joint_params,
                batch,
                joint,
                n_joints,
                mask,
                True,
                gtx,
                gty,
                gtz,
                gqx,
                gqy,
                gqz,
                gqw,
                gs,
                ls,
            )
            joint += 1


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError(
            "Triton FK requested, but Triton is not available in this build."
        )


def _validate_inputs(  # noqa: C901
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
) -> None:
    if not joint_parameters.is_cuda:
        raise RuntimeError("Triton FK requested, but joint_parameters is on CPU.")
    if joint_parameters.dtype != torch.float32:
        raise RuntimeError(
            "Triton FK currently only supports float32 joint parameters."
        )
    if joint_parameters.ndim < 2 or joint_parameters.shape[-1] != 7:
        raise RuntimeError("joint_parameters must have shape [..., num_joints, 7].")
    if joint_translation_offsets.device != joint_parameters.device:
        raise RuntimeError("joint_translation_offsets must be on the same device.")
    if joint_prerotations.device != joint_parameters.device:
        raise RuntimeError("joint_prerotations must be on the same device.")
    if joint_parents.device != joint_parameters.device:
        raise RuntimeError("joint_parents must be on the same device.")
    if joint_translation_offsets.dtype != torch.float32:
        raise RuntimeError("joint_translation_offsets must be float32.")
    if joint_prerotations.dtype != torch.float32:
        raise RuntimeError("joint_prerotations must be float32.")
    if joint_parents.dtype != torch.int32:
        raise RuntimeError("joint_parents must be int32.")
    num_joints = joint_parameters.shape[-2]
    if joint_translation_offsets.shape != (num_joints, 3):
        raise RuntimeError("joint_translation_offsets must have shape [num_joints, 3].")
    if joint_prerotations.shape != (num_joints, 4):
        raise RuntimeError("joint_prerotations must have shape [num_joints, 4].")
    if joint_parents.shape != (num_joints,):
        raise RuntimeError("joint_parents must have shape [num_joints].")


def _launch_forward(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
) -> torch.Tensor:
    _validate_inputs(
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
    )
    _require_triton()

    num_joints = joint_parameters.shape[-2]
    batch_size = joint_parameters.numel() // (num_joints * 7)
    output = torch.empty(
        (*joint_parameters.shape[:-1], 8),
        device=joint_parameters.device,
        dtype=joint_parameters.dtype,
    )
    grid = (triton.cdiv(batch_size, _BLOCK_BATCH),)
    _forward_kernel[grid](
        joint_parameters,
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
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    global_state: torch.Tensor,
    grad_global_state: torch.Tensor,
    active_joints: torch.Tensor | None,
) -> torch.Tensor:
    _require_triton()
    grad_global_state = grad_global_state.contiguous()
    grad_work = grad_global_state.clone()
    has_active_joints = active_joints is not None
    if active_joints is None:
        active_joints = torch.empty(
            (0,),
            device=joint_parameters.device,
            dtype=torch.bool,
        )
        grad_joint_parameters = torch.empty_like(joint_parameters)
    else:
        active_joints = active_joints.contiguous()
        grad_joint_parameters = torch.zeros_like(joint_parameters)

    num_joints = joint_parameters.shape[-2]
    batch_size = joint_parameters.numel() // (num_joints * 7)
    grid = (triton.cdiv(batch_size, _BLOCK_BATCH),)
    _backward_kernel[grid](
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        active_joints,
        global_state,
        grad_work,
        grad_joint_parameters,
        batch_size,
        num_joints,
        HAS_ACTIVE_JOINTS=has_active_joints,
        BLOCK_BATCH=_BLOCK_BATCH,
    )
    return grad_joint_parameters


class _JointParametersToSkeletonState(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        joint_parameters: torch.Tensor,
        joint_translation_offsets: torch.Tensor,
        joint_prerotations: torch.Tensor,
        joint_parents: torch.Tensor,
        active_joints: torch.Tensor | None,
    ) -> torch.Tensor:
        joint_parameters = joint_parameters.contiguous()
        joint_translation_offsets = joint_translation_offsets.contiguous()
        joint_prerotations = joint_prerotations.contiguous()
        joint_parents = joint_parents.contiguous()
        output = _launch_forward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
        )
        ctx.save_for_backward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            output,
        )
        ctx.active_joints = (
            active_joints.contiguous() if active_joints is not None else None
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None, None, None]:
        (
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            global_state,
        ) = ctx.saved_tensors
        grad_joint_parameters = _launch_backward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            joint_parents,
            global_state,
            grad_output,
            ctx.active_joints,
        )
        return grad_joint_parameters, None, None, None, None


def joint_parameters_to_skeleton_state(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    joint_parents: torch.Tensor,
    active_joints: torch.Tensor | None = None,
) -> torch.Tensor:
    return _JointParametersToSkeletonState.apply(
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        joint_parents,
        active_joints,
    )


def _validate_local_inputs(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
) -> None:
    # Same as _validate_inputs but the local FK needs no kinematic tree (parents).
    if not joint_parameters.is_cuda:
        raise RuntimeError("Triton local FK requested, but joint_parameters is on CPU.")
    if joint_parameters.dtype != torch.float32:
        raise RuntimeError(
            "Triton local FK currently only supports float32 joint parameters."
        )
    if joint_parameters.ndim < 2 or joint_parameters.shape[-1] != 7:
        raise RuntimeError("joint_parameters must have shape [..., num_joints, 7].")
    if joint_translation_offsets.device != joint_parameters.device:
        raise RuntimeError("joint_translation_offsets must be on the same device.")
    if joint_prerotations.device != joint_parameters.device:
        raise RuntimeError("joint_prerotations must be on the same device.")
    if joint_translation_offsets.dtype != torch.float32:
        raise RuntimeError("joint_translation_offsets must be float32.")
    if joint_prerotations.dtype != torch.float32:
        raise RuntimeError("joint_prerotations must be float32.")
    num_joints = joint_parameters.shape[-2]
    if joint_translation_offsets.shape != (num_joints, 3):
        raise RuntimeError("joint_translation_offsets must have shape [num_joints, 3].")
    if joint_prerotations.shape != (num_joints, 4):
        raise RuntimeError("joint_prerotations must have shape [num_joints, 4].")


def _launch_local_forward(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
) -> torch.Tensor:
    _validate_local_inputs(
        joint_parameters, joint_translation_offsets, joint_prerotations
    )
    _require_triton()
    num_joints = joint_parameters.shape[-2]
    batch_size = joint_parameters.numel() // (num_joints * 7)
    output = torch.empty(
        (*joint_parameters.shape[:-1], 8),
        device=joint_parameters.device,
        dtype=joint_parameters.dtype,
    )
    grid = (triton.cdiv(batch_size, _BLOCK_BATCH),)
    _local_forward_kernel[grid](
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        output,
        batch_size,
        num_joints,
        BLOCK_BATCH=_BLOCK_BATCH,
    )
    return output


def _launch_local_backward(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    grad_output = grad_output.contiguous()
    grad_joint_parameters = torch.empty_like(joint_parameters)
    num_joints = joint_parameters.shape[-2]
    batch_size = joint_parameters.numel() // (num_joints * 7)
    grid = (triton.cdiv(batch_size, _BLOCK_BATCH),)
    _local_backward_kernel[grid](
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
        grad_output,
        grad_joint_parameters,
        batch_size,
        num_joints,
        BLOCK_BATCH=_BLOCK_BATCH,
    )
    return grad_joint_parameters


class _JointParametersToLocalSkeletonState(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        joint_parameters: torch.Tensor,
        joint_translation_offsets: torch.Tensor,
        joint_prerotations: torch.Tensor,
    ) -> torch.Tensor:
        joint_parameters = joint_parameters.contiguous()
        joint_translation_offsets = joint_translation_offsets.contiguous()
        joint_prerotations = joint_prerotations.contiguous()
        output = _launch_local_forward(
            joint_parameters, joint_translation_offsets, joint_prerotations
        )
        ctx.save_for_backward(
            joint_parameters, joint_translation_offsets, joint_prerotations
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        joint_parameters, joint_translation_offsets, joint_prerotations = (
            ctx.saved_tensors
        )
        grad_joint_parameters = _launch_local_backward(
            joint_parameters,
            joint_translation_offsets,
            joint_prerotations,
            grad_output,
        )
        return grad_joint_parameters, None, None


def joint_parameters_to_local_skeleton_state(
    joint_parameters: torch.Tensor,
    joint_translation_offsets: torch.Tensor,
    joint_prerotations: torch.Tensor,
) -> torch.Tensor:
    return _JointParametersToLocalSkeletonState.apply(
        joint_parameters,
        joint_translation_offsets,
        joint_prerotations,
    )
