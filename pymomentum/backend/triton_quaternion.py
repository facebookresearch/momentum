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


def to_rotation_matrix(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return _ToRotationMatrix.apply(q, True, eps)


def to_rotation_matrix_assume_normalized(q: torch.Tensor) -> torch.Tensor:
    return _ToRotationMatrix.apply(q, False, 1.0)
