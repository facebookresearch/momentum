# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.quaternion as pym_quaternion
import torch
from pymomentum.backend import selection as backend_selection


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton backend tests")
    if not backend_selection._HAS_TRITON:
        raise RuntimeError("Triton is required for Triton backend tests")
    device = torch.device("cuda")
    torch.empty((), device=device)
    return device


def _generate_random_quats(size: int) -> torch.Tensor:
    return pym_quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(size, 4),
            dtype=torch.float64,
            requires_grad=True,
        )
    )


class TritonQuaternionBackendTest(unittest.TestCase):
    def test_backend_rejects_cpu(self) -> None:
        quats = _generate_random_quats(3).to(torch.float32)
        with self.assertRaisesRegex(RuntimeError, "input is on CPU"):
            pym_quaternion.to_rotation_matrix(quats, backend="triton")

        mats = pym_quaternion.to_rotation_matrix(quats)
        with self.assertRaisesRegex(RuntimeError, "input is on CPU"):
            pym_quaternion.from_rotation_matrix(mats, backend="triton")

    def test_to_rotation_matrix_matches_torch(self) -> None:
        device = _cuda_device()

        torch.manual_seed(0)
        quaternions = torch.randn((257, 4), device=device, dtype=torch.float32)

        self._assert_to_rotation_matrix_close(
            quaternions,
            assume_normalized=False,
            eps=1e-12,
        )

    def test_to_rotation_matrix_assume_normalized_matches_torch(self) -> None:
        device = _cuda_device()

        torch.manual_seed(0)
        quaternions = torch.randn((2, 3, 257, 4), device=device, dtype=torch.float32)
        quaternions = pym_quaternion.normalize(quaternions)

        self._assert_to_rotation_matrix_close(
            quaternions,
            assume_normalized=True,
            eps=1e-12,
        )

    def test_from_rotation_matrix_matches_torch(self) -> None:
        device = _cuda_device()

        torch.manual_seed(0)
        quaternions = pym_quaternion.normalize(
            torch.randn((2, 3, 257, 4), device=device, dtype=torch.float32)
        )
        matrices = pym_quaternion.to_rotation_matrix(quaternions)

        torch_matrices = matrices.detach().clone().requires_grad_()
        triton_matrices = matrices.detach().clone().requires_grad_()

        torch_result = pym_quaternion.from_rotation_matrix(torch_matrices)
        triton_result = pym_quaternion.from_rotation_matrix(
            triton_matrices,
            backend="triton",
        )

        torch.testing.assert_close(torch_result, triton_result, atol=1e-6, rtol=1e-6)

        grad_output = torch.randn_like(torch_result)
        torch_result.backward(grad_output)
        triton_result.backward(grad_output)

        torch_grad = torch_matrices.grad
        triton_grad = triton_matrices.grad
        self.assertIsNotNone(torch_grad)
        self.assertIsNotNone(triton_grad)
        if torch_grad is None or triton_grad is None:
            return
        torch.testing.assert_close(torch_grad, triton_grad, atol=1e-5, rtol=1e-5)

    def test_zero_quaternion_is_finite(self) -> None:
        device = _cuda_device()
        eps = 1e-12
        quaternions = torch.zeros((4, 4), device=device, dtype=torch.float32)
        quaternions.requires_grad_()

        result = pym_quaternion.to_rotation_matrix(
            quaternions,
            backend="triton",
            eps=eps,
        )
        result.sum().backward()

        self.assertTrue(torch.isfinite(result).all())
        self.assertIsNotNone(quaternions.grad)
        if quaternions.grad is None:
            return
        self.assertTrue(torch.isfinite(quaternions.grad).all())

    def _assert_to_rotation_matrix_close(
        self,
        quaternions: torch.Tensor,
        *,
        assume_normalized: bool,
        eps: float,
    ) -> None:
        torch_quats = quaternions.detach().clone().requires_grad_()
        triton_quats = quaternions.detach().clone().requires_grad_()

        if assume_normalized:
            torch_result = pym_quaternion.to_rotation_matrix_assume_normalized(
                torch_quats
            )
            triton_result = pym_quaternion.to_rotation_matrix_assume_normalized(
                triton_quats,
                backend="triton",
            )
        else:
            torch_result = pym_quaternion.to_rotation_matrix(torch_quats, eps=eps)
            triton_result = pym_quaternion.to_rotation_matrix(
                triton_quats,
                backend="triton",
                eps=eps,
            )

        torch.testing.assert_close(torch_result, triton_result, atol=1e-6, rtol=1e-6)

        grad_output = torch.randn_like(torch_result)
        torch_result.backward(grad_output)
        triton_result.backward(grad_output)

        torch_grad = torch_quats.grad
        triton_grad = triton_quats.grad
        self.assertIsNotNone(torch_grad)
        self.assertIsNotNone(triton_grad)
        if torch_grad is None or triton_grad is None:
            return
        torch.testing.assert_close(torch_grad, triton_grad, atol=1e-5, rtol=1e-5)
