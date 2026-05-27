# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None and tl is not None:

    @triton.jit
    def _quat_multiply(x1, y1, z1, w1, x2, y2, z2, w2):
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    @triton.jit
    def _quat_multiply_backward(x1, y1, z1, w1, x2, y2, z2, w2, gx, gy, gz, gw):
        return (
            gx * w2 - gy * z2 + gz * y2 - gw * x2,
            gx * z2 + gy * w2 - gz * x2 - gw * y2,
            -gx * y2 + gy * x2 + gz * w2 - gw * z2,
            gx * x2 + gy * y2 + gz * z2 + gw * w2,
            gx * w1 + gy * z1 - gz * y1 - gw * x1,
            -gx * z1 + gy * w1 + gz * x1 - gw * y1,
            gx * y1 - gy * x1 + gz * w1 - gw * z1,
            gx * x1 + gy * y1 + gz * z1 + gw * w1,
        )

    @triton.jit
    def _rotate_vector(qx, qy, qz, qw, vx, vy, vz):
        return (
            (1.0 - 2.0 * qy * qy - 2.0 * qz * qz) * vx
            + (2.0 * qx * qy - 2.0 * qz * qw) * vy
            + (2.0 * qx * qz + 2.0 * qy * qw) * vz,
            (2.0 * qx * qy + 2.0 * qz * qw) * vx
            + (1.0 - 2.0 * qx * qx - 2.0 * qz * qz) * vy
            + (2.0 * qy * qz - 2.0 * qx * qw) * vz,
            (2.0 * qx * qz - 2.0 * qy * qw) * vx
            + (2.0 * qy * qz + 2.0 * qx * qw) * vy
            + (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * vz,
        )

    @triton.jit
    def _rotate_vector_backward(qx, qy, qz, qw, vx, vy, vz, gx, gy, gz):
        grad_vx = (
            (1.0 - 2.0 * qy * qy - 2.0 * qz * qz) * gx
            + (2.0 * qx * qy + 2.0 * qz * qw) * gy
            + (2.0 * qx * qz - 2.0 * qy * qw) * gz
        )
        grad_vy = (
            (2.0 * qx * qy - 2.0 * qz * qw) * gx
            + (1.0 - 2.0 * qx * qx - 2.0 * qz * qz) * gy
            + (2.0 * qy * qz + 2.0 * qx * qw) * gz
        )
        grad_vz = (
            (2.0 * qx * qz + 2.0 * qy * qw) * gx
            + (2.0 * qy * qz - 2.0 * qx * qw) * gy
            + (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * gz
        )

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
        return grad_qx, grad_qy, grad_qz, grad_qw, grad_vx, grad_vy, grad_vz
