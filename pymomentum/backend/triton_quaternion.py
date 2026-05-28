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


_BLOCK_SIZE = 256
if triton is not None and tl is not None:

    @triton.jit
    def _normalize_quaternion(x, y, z, w, eps, NORMALIZE: tl.constexpr):
        if NORMALIZE:
            norm2 = x * x + y * y + z * z + w * w
            eps2 = eps * eps
            inv_norm = tl.rsqrt(tl.maximum(norm2, eps2))
            return x * inv_norm, y * inv_norm, z * inv_norm, w * inv_norm
        return x, y, z, w

    @triton.jit
    def _to_rotation_matrix_kernel(
        q,
        output,
        n_quaternions,
        eps,
        NORMALIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_quaternions
        q_base = q + offsets * 4

        x = tl.load(q_base + 0, mask=mask, other=0.0)
        y = tl.load(q_base + 1, mask=mask, other=0.0)
        z = tl.load(q_base + 2, mask=mask, other=0.0)
        w = tl.load(q_base + 3, mask=mask, other=1.0)
        x, y, z, w = _normalize_quaternion(x, y, z, w, eps, NORMALIZE)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        xw = x * w
        yz = y * z
        yw = y * w
        zw = z * w

        out_base = output + offsets * 9
        tl.store(out_base + 0, 1.0 - 2.0 * (yy + zz), mask=mask)
        tl.store(out_base + 1, 2.0 * (xy - zw), mask=mask)
        tl.store(out_base + 2, 2.0 * (xz + yw), mask=mask)
        tl.store(out_base + 3, 2.0 * (xy + zw), mask=mask)
        tl.store(out_base + 4, 1.0 - 2.0 * (xx + zz), mask=mask)
        tl.store(out_base + 5, 2.0 * (yz - xw), mask=mask)
        tl.store(out_base + 6, 2.0 * (xz - yw), mask=mask)
        tl.store(out_base + 7, 2.0 * (yz + xw), mask=mask)
        tl.store(out_base + 8, 1.0 - 2.0 * (xx + yy), mask=mask)

    @triton.jit
    def _to_rotation_matrix_backward_kernel(
        q,
        grad_output,
        grad_q,
        n_quaternions,
        eps,
        NORMALIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_quaternions
        q_base = q + offsets * 4
        grad_base = grad_output + offsets * 9

        raw_x = tl.load(q_base + 0, mask=mask, other=0.0)
        raw_y = tl.load(q_base + 1, mask=mask, other=0.0)
        raw_z = tl.load(q_base + 2, mask=mask, other=0.0)
        raw_w = tl.load(q_base + 3, mask=mask, other=1.0)
        x, y, z, w = _normalize_quaternion(raw_x, raw_y, raw_z, raw_w, eps, NORMALIZE)

        g00 = tl.load(grad_base + 0, mask=mask, other=0.0)
        g01 = tl.load(grad_base + 1, mask=mask, other=0.0)
        g02 = tl.load(grad_base + 2, mask=mask, other=0.0)
        g10 = tl.load(grad_base + 3, mask=mask, other=0.0)
        g11 = tl.load(grad_base + 4, mask=mask, other=0.0)
        g12 = tl.load(grad_base + 5, mask=mask, other=0.0)
        g20 = tl.load(grad_base + 6, mask=mask, other=0.0)
        g21 = tl.load(grad_base + 7, mask=mask, other=0.0)
        g22 = tl.load(grad_base + 8, mask=mask, other=0.0)

        gx = (
            2.0 * y * (g01 + g10)
            + 2.0 * z * (g02 + g20)
            + 2.0 * w * (-g12 + g21)
            - 4.0 * x * (g11 + g22)
        )
        gy = (
            -4.0 * y * (g00 + g22)
            + 2.0 * x * (g01 + g10)
            + 2.0 * w * (g02 - g20)
            + 2.0 * z * (g12 + g21)
        )
        gz = (
            -4.0 * z * (g00 + g11)
            + 2.0 * x * (g02 + g20)
            + 2.0 * y * (g12 + g21)
            + 2.0 * w * (-g01 + g10)
        )
        gw = 2.0 * z * (-g01 + g10) + 2.0 * y * (g02 - g20) + 2.0 * x * (-g12 + g21)

        if NORMALIZE:
            norm2 = raw_x * raw_x + raw_y * raw_y + raw_z * raw_z + raw_w * raw_w
            eps2 = eps * eps
            inv_norm = tl.rsqrt(tl.maximum(norm2, eps2))
            dot = gx * x + gy * y + gz * z + gw * w
            unclamped_gx = (gx - x * dot) * inv_norm
            unclamped_gy = (gy - y * dot) * inv_norm
            unclamped_gz = (gz - z * dot) * inv_norm
            unclamped_gw = (gw - w * dot) * inv_norm
            clamped_gx = gx * inv_norm
            clamped_gy = gy * inv_norm
            clamped_gz = gz * inv_norm
            clamped_gw = gw * inv_norm
            is_unclamped = norm2 >= eps2
            gx = tl.where(is_unclamped, unclamped_gx, clamped_gx)
            gy = tl.where(is_unclamped, unclamped_gy, clamped_gy)
            gz = tl.where(is_unclamped, unclamped_gz, clamped_gz)
            gw = tl.where(is_unclamped, unclamped_gw, clamped_gw)

        out_base = grad_q + offsets * 4
        tl.store(out_base + 0, gx, mask=mask)
        tl.store(out_base + 1, gy, mask=mask)
        tl.store(out_base + 2, gz, mask=mask)
        tl.store(out_base + 3, gw, mask=mask)

    @triton.jit
    def _from_rotation_matrix_kernel(
        matrices,
        output,
        n_matrices,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_matrices
        matrix_base = matrices + offsets * 9

        m00 = tl.load(matrix_base + 0, mask=mask, other=1.0)
        m01 = tl.load(matrix_base + 1, mask=mask, other=0.0)
        m02 = tl.load(matrix_base + 2, mask=mask, other=0.0)
        m10 = tl.load(matrix_base + 3, mask=mask, other=0.0)
        m11 = tl.load(matrix_base + 4, mask=mask, other=1.0)
        m12 = tl.load(matrix_base + 5, mask=mask, other=0.0)
        m20 = tl.load(matrix_base + 6, mask=mask, other=0.0)
        m21 = tl.load(matrix_base + 7, mask=mask, other=0.0)
        m22 = tl.load(matrix_base + 8, mask=mask, other=1.0)

        qw_abs = tl.sqrt(tl.maximum(1.0 + m00 + m11 + m22, 1.0e-15))
        qx_abs = tl.sqrt(tl.maximum(1.0 + m00 - m11 - m22, 1.0e-15))
        qy_abs = tl.sqrt(tl.maximum(1.0 - m00 + m11 - m22, 1.0e-15))
        qz_abs = tl.sqrt(tl.maximum(1.0 - m00 - m11 + m22, 1.0e-15))

        flr = 0.01
        qw_denom = 2.0 * tl.maximum(qw_abs, flr)
        qx_denom = 2.0 * tl.maximum(qx_abs, flr)
        qy_denom = 2.0 * tl.maximum(qy_abs, flr)
        qz_denom = 2.0 * tl.maximum(qz_abs, flr)

        x = (m21 - m12) / qw_denom
        y = (m02 - m20) / qw_denom
        z = (m10 - m01) / qw_denom
        w = (qw_abs * qw_abs) / qw_denom

        use_x = qx_abs > qw_abs
        x = tl.where(use_x, (qx_abs * qx_abs) / qx_denom, x)
        y = tl.where(use_x, (m10 + m01) / qx_denom, y)
        z = tl.where(use_x, (m02 + m20) / qx_denom, z)
        w = tl.where(use_x, (m21 - m12) / qx_denom, w)

        use_y = (qy_abs > qw_abs) & (qy_abs > qx_abs)
        x = tl.where(use_y, (m10 + m01) / qy_denom, x)
        y = tl.where(use_y, (qy_abs * qy_abs) / qy_denom, y)
        z = tl.where(use_y, (m12 + m21) / qy_denom, z)
        w = tl.where(use_y, (m02 - m20) / qy_denom, w)

        use_z = (qz_abs > qw_abs) & (qz_abs > qx_abs) & (qz_abs > qy_abs)
        x = tl.where(use_z, (m20 + m02) / qz_denom, x)
        y = tl.where(use_z, (m21 + m12) / qz_denom, y)
        z = tl.where(use_z, (qz_abs * qz_abs) / qz_denom, z)
        w = tl.where(use_z, (m10 - m01) / qz_denom, w)

        norm2 = x * x + y * y + z * z + w * w
        inv_norm = tl.rsqrt(tl.maximum(norm2, 1.0e-24))

        out_base = output + offsets * 4
        tl.store(out_base + 0, x * inv_norm, mask=mask)
        tl.store(out_base + 1, y * inv_norm, mask=mask)
        tl.store(out_base + 2, z * inv_norm, mask=mask)
        tl.store(out_base + 3, w * inv_norm, mask=mask)

    @triton.jit
    def _accumulate_q_abs_grad(
        grad_a,
        a,
        s,
        sign00: tl.constexpr,
        sign11: tl.constexpr,
        sign22: tl.constexpr,
    ):
        grad_s = tl.where(s > 1.0e-15, grad_a * 0.5 / a, 0.0)
        return sign00 * grad_s, sign11 * grad_s, sign22 * grad_s

    @triton.jit
    def _grad_div_num_by_2clamped_abs(grad_value, numerator, a):
        denom = 2.0 * tl.maximum(a, 0.01)
        grad_num = grad_value / denom
        grad_a = tl.where(a >= 0.01, grad_value * (-numerator / (2.0 * a * a)), 0.0)
        return grad_num, grad_a

    @triton.jit
    def _grad_abs_square_by_2clamped_abs(grad_value, a):
        grad_a_unclamped = 0.5 * grad_value
        grad_a_clamped = grad_value * a / 0.01
        return tl.where(a >= 0.01, grad_a_unclamped, grad_a_clamped)

    @triton.jit
    def _from_rotation_matrix_backward_kernel(
        matrices,
        grad_output,
        grad_matrices,
        n_matrices,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_matrices
        matrix_base = matrices + offsets * 9

        m00 = tl.load(matrix_base + 0, mask=mask, other=1.0)
        m01 = tl.load(matrix_base + 1, mask=mask, other=0.0)
        m02 = tl.load(matrix_base + 2, mask=mask, other=0.0)
        m10 = tl.load(matrix_base + 3, mask=mask, other=0.0)
        m11 = tl.load(matrix_base + 4, mask=mask, other=1.0)
        m12 = tl.load(matrix_base + 5, mask=mask, other=0.0)
        m20 = tl.load(matrix_base + 6, mask=mask, other=0.0)
        m21 = tl.load(matrix_base + 7, mask=mask, other=0.0)
        m22 = tl.load(matrix_base + 8, mask=mask, other=1.0)

        s0 = 1.0 + m00 + m11 + m22
        s1 = 1.0 + m00 - m11 - m22
        s2 = 1.0 - m00 + m11 - m22
        s3 = 1.0 - m00 - m11 + m22
        qw_abs = tl.sqrt(tl.maximum(s0, 1.0e-15))
        qx_abs = tl.sqrt(tl.maximum(s1, 1.0e-15))
        qy_abs = tl.sqrt(tl.maximum(s2, 1.0e-15))
        qz_abs = tl.sqrt(tl.maximum(s3, 1.0e-15))

        qw_denom = 2.0 * tl.maximum(qw_abs, 0.01)
        qx_denom = 2.0 * tl.maximum(qx_abs, 0.01)
        qy_denom = 2.0 * tl.maximum(qy_abs, 0.01)
        qz_denom = 2.0 * tl.maximum(qz_abs, 0.01)

        rx = (m21 - m12) / qw_denom
        ry = (m02 - m20) / qw_denom
        rz = (m10 - m01) / qw_denom
        rw = (qw_abs * qw_abs) / qw_denom

        xx = (qx_abs * qx_abs) / qx_denom
        xy = (m10 + m01) / qx_denom
        xz = (m02 + m20) / qx_denom
        xw = (m21 - m12) / qx_denom

        yx = (m10 + m01) / qy_denom
        yy = (qy_abs * qy_abs) / qy_denom
        yz = (m12 + m21) / qy_denom
        yw = (m02 - m20) / qy_denom

        zx = (m20 + m02) / qz_denom
        zy = (m21 + m12) / qz_denom
        zz = (qz_abs * qz_abs) / qz_denom
        zw = (m10 - m01) / qz_denom

        use_x = qx_abs > qw_abs
        use_y = (qy_abs > qw_abs) & (qy_abs > qx_abs)
        use_z = (qz_abs > qw_abs) & (qz_abs > qx_abs) & (qz_abs > qy_abs)

        raw_x = tl.where(use_z, zx, tl.where(use_y, yx, tl.where(use_x, xx, rx)))
        raw_y = tl.where(use_z, zy, tl.where(use_y, yy, tl.where(use_x, xy, ry)))
        raw_z = tl.where(use_z, zz, tl.where(use_y, yz, tl.where(use_x, xz, rz)))
        raw_w = tl.where(use_z, zw, tl.where(use_y, yw, tl.where(use_x, xw, rw)))

        norm2 = raw_x * raw_x + raw_y * raw_y + raw_z * raw_z + raw_w * raw_w
        inv_norm = tl.rsqrt(tl.maximum(norm2, 1.0e-24))
        qx = raw_x * inv_norm
        qy = raw_y * inv_norm
        qz = raw_z * inv_norm
        qw = raw_w * inv_norm

        grad_base = grad_output + offsets * 4
        gx = tl.load(grad_base + 0, mask=mask, other=0.0)
        gy = tl.load(grad_base + 1, mask=mask, other=0.0)
        gz = tl.load(grad_base + 2, mask=mask, other=0.0)
        gw = tl.load(grad_base + 3, mask=mask, other=0.0)

        dot = gx * qx + gy * qy + gz * qz + gw * qw
        raw_gx = (gx - qx * dot) * inv_norm
        raw_gy = (gy - qy * dot) * inv_norm
        raw_gz = (gz - qz * dot) * inv_norm
        raw_gw = (gw - qw * dot) * inv_norm

        g00 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g01 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g02 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g10 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g11 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g12 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g20 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g21 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        g22 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gx, m21 - m12, qw_abs)
        rg21 = grad_num
        rg12 = -grad_num
        rga = grad_a
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gy, m02 - m20, qw_abs)
        rg02 = grad_num
        rg20 = -grad_num
        rga += grad_a
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gz, m10 - m01, qw_abs)
        rg10 = grad_num
        rg01 = -grad_num
        rga += grad_a
        rga += _grad_abs_square_by_2clamped_abs(raw_gw, qw_abs)
        rg00, rg11, rg22 = _accumulate_q_abs_grad(rga, qw_abs, s0, 1.0, 1.0, 1.0)

        xga = _grad_abs_square_by_2clamped_abs(raw_gx, qx_abs)
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gy, m10 + m01, qx_abs)
        xg10 = grad_num
        xg01 = grad_num
        xga += grad_a
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gz, m02 + m20, qx_abs)
        xg02 = grad_num
        xg20 = grad_num
        xga += grad_a
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gw, m21 - m12, qx_abs)
        xg21 = grad_num
        xg12 = -grad_num
        xga += grad_a
        xg00, xg11, xg22 = _accumulate_q_abs_grad(xga, qx_abs, s1, 1.0, -1.0, -1.0)

        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gx, m10 + m01, qy_abs)
        yg10 = grad_num
        yg01 = grad_num
        yga = grad_a
        yga += _grad_abs_square_by_2clamped_abs(raw_gy, qy_abs)
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gz, m12 + m21, qy_abs)
        yg12 = grad_num
        yg21 = grad_num
        yga += grad_a
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gw, m02 - m20, qy_abs)
        yg02 = grad_num
        yg20 = -grad_num
        yga += grad_a
        yg00, yg11, yg22 = _accumulate_q_abs_grad(yga, qy_abs, s2, -1.0, 1.0, -1.0)

        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gx, m20 + m02, qz_abs)
        zg20 = grad_num
        zg02 = grad_num
        zga = grad_a
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gy, m21 + m12, qz_abs)
        zg21 = grad_num
        zg12 = grad_num
        zga += grad_a
        zga += _grad_abs_square_by_2clamped_abs(raw_gz, qz_abs)
        grad_num, grad_a = _grad_div_num_by_2clamped_abs(raw_gw, m10 - m01, qz_abs)
        zg10 = grad_num
        zg01 = -grad_num
        zga += grad_a
        zg00, zg11, zg22 = _accumulate_q_abs_grad(zga, qz_abs, s3, -1.0, -1.0, 1.0)

        g00 = tl.where(use_z, zg00, tl.where(use_y, yg00, tl.where(use_x, xg00, rg00)))
        g01 = tl.where(use_z, zg01, tl.where(use_y, yg01, tl.where(use_x, xg01, rg01)))
        g02 = tl.where(use_z, zg02, tl.where(use_y, yg02, tl.where(use_x, xg02, rg02)))
        g10 = tl.where(use_z, zg10, tl.where(use_y, yg10, tl.where(use_x, xg10, rg10)))
        g11 = tl.where(use_z, zg11, tl.where(use_y, yg11, tl.where(use_x, xg11, rg11)))
        g12 = tl.where(use_z, zg12, tl.where(use_y, yg12, tl.where(use_x, xg12, rg12)))
        g20 = tl.where(use_z, zg20, tl.where(use_y, yg20, tl.where(use_x, xg20, rg20)))
        g21 = tl.where(use_z, zg21, tl.where(use_y, yg21, tl.where(use_x, xg21, rg21)))
        g22 = tl.where(use_z, zg22, tl.where(use_y, yg22, tl.where(use_x, xg22, rg22)))

        out_base = grad_matrices + offsets * 9
        tl.store(out_base + 0, g00, mask=mask)
        tl.store(out_base + 1, g01, mask=mask)
        tl.store(out_base + 2, g02, mask=mask)
        tl.store(out_base + 3, g10, mask=mask)
        tl.store(out_base + 4, g11, mask=mask)
        tl.store(out_base + 5, g12, mask=mask)
        tl.store(out_base + 6, g20, mask=mask)
        tl.store(out_base + 7, g21, mask=mask)
        tl.store(out_base + 8, g22, mask=mask)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError(
            "Triton quaternion backend requested, but Triton is not available "
            "in this build."
        )


def _validate_quaternions(q: torch.Tensor, eps: float) -> None:
    if not q.is_cuda:
        raise RuntimeError("Triton quaternion backend requested, but input is on CPU.")
    if q.dtype != torch.float32:
        raise RuntimeError("Triton quaternion backend currently only supports float32.")
    if q.ndim == 0 or q.shape[-1] != 4:
        raise RuntimeError("Quaternion tensor must have shape [..., 4].")
    if eps <= 0.0:
        raise RuntimeError("eps must be positive.")


def _validate_rotation_matrices(matrices: torch.Tensor) -> None:
    if not matrices.is_cuda:
        raise RuntimeError("Triton quaternion backend requested, but input is on CPU.")
    if matrices.dtype != torch.float32:
        raise RuntimeError("Triton quaternion backend currently only supports float32.")
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise RuntimeError("Rotation matrix tensor must have shape [..., 3, 3].")


def _launch_forward(q: torch.Tensor, normalize: bool, eps: float) -> torch.Tensor:
    _validate_quaternions(q, eps)
    _require_triton()
    n_quaternions = q.numel() // 4
    output = torch.empty(
        (*q.shape[:-1], 3, 3),
        device=q.device,
        dtype=q.dtype,
    )
    grid = (triton.cdiv(n_quaternions, _BLOCK_SIZE),)
    _to_rotation_matrix_kernel[grid](
        q,
        output,
        n_quaternions,
        eps,
        NORMALIZE=normalize,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


def _launch_from_rotation_matrix(matrices: torch.Tensor) -> torch.Tensor:
    _validate_rotation_matrices(matrices)
    _require_triton()
    n_matrices = matrices.numel() // 9
    output = torch.empty(
        (*matrices.shape[:-2], 4),
        device=matrices.device,
        dtype=matrices.dtype,
    )
    grid = (triton.cdiv(n_matrices, _BLOCK_SIZE),)
    _from_rotation_matrix_kernel[grid](
        matrices,
        output,
        n_matrices,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


def _launch_from_rotation_matrix_backward(
    matrices: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    grad_output = grad_output.contiguous()
    grad_matrices = torch.empty_like(matrices)
    n_matrices = matrices.numel() // 9
    grid = (triton.cdiv(n_matrices, _BLOCK_SIZE),)
    _from_rotation_matrix_backward_kernel[grid](
        matrices,
        grad_output,
        grad_matrices,
        n_matrices,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return grad_matrices


def _launch_backward(
    q: torch.Tensor,
    grad_output: torch.Tensor,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    _require_triton()
    grad_output = grad_output.contiguous()
    grad_q = torch.empty_like(q)
    n_quaternions = q.numel() // 4
    grid = (triton.cdiv(n_quaternions, _BLOCK_SIZE),)
    _to_rotation_matrix_backward_kernel[grid](
        q,
        grad_output,
        grad_q,
        n_quaternions,
        eps,
        NORMALIZE=normalize,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return grad_q


class _ToRotationMatrix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        normalize: bool,
        eps: float,
    ) -> torch.Tensor:
        q = q.contiguous()
        output = _launch_forward(q, normalize, eps)
        ctx.save_for_backward(q)
        ctx.normalize = normalize
        ctx.eps = eps
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (q,) = ctx.saved_tensors
        grad_q = _launch_backward(q, grad_output, ctx.normalize, ctx.eps)
        return grad_q, None, None


class _FromRotationMatrix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        matrices: torch.Tensor,
    ) -> torch.Tensor:
        matrices = matrices.contiguous()
        output = _launch_from_rotation_matrix(matrices)
        ctx.save_for_backward(matrices)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (matrices,) = ctx.saved_tensors
        grad_matrices = _launch_from_rotation_matrix_backward(matrices, grad_output)
        return (grad_matrices,)


def to_rotation_matrix(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return _ToRotationMatrix.apply(q, True, eps)


def to_rotation_matrix_assume_normalized(q: torch.Tensor) -> torch.Tensor:
    return _ToRotationMatrix.apply(q, False, 1.0)


def from_rotation_matrix(matrices: torch.Tensor) -> torch.Tensor:
    return _FromRotationMatrix.apply(matrices)
