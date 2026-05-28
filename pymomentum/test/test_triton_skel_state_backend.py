# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
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


def _generate_random_skel_state(size: int) -> torch.Tensor:
    trans = torch.normal(mean=0, std=4, size=(size, 3), dtype=torch.float64)
    rot = pym_quaternion.normalize(
        torch.normal(mean=0, std=4, size=(size, 4), dtype=torch.float64)
    )
    scale = torch.rand(size=(size, 1), dtype=torch.float64)
    return torch.cat([trans, rot, scale], -1)


def _random_cuda_skel_state(size: int) -> torch.Tensor:
    trans = torch.normal(
        mean=0,
        std=4,
        size=(size, 3),
        device="cuda",
        dtype=torch.float32,
    )
    rot = pym_quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(size, 4),
            device="cuda",
            dtype=torch.float32,
        )
    )
    scale = torch.rand(size=(size, 1), device="cuda", dtype=torch.float32) + 0.5
    return torch.cat([trans, rot, scale], -1)


class TritonSkelStateBackendTest(unittest.TestCase):
    def test_backend_rejects_cpu(self) -> None:
        state = _generate_random_skel_state(3).to(torch.float32)
        with self.assertRaisesRegex(RuntimeError, "input is on CPU"):
            pym_skel_state.multiply(state, state, backend="triton")

        with self.assertRaisesRegex(RuntimeError, "input is on CPU"):
            pym_skel_state.multiply_assume_normalized(state, state, backend="triton")

        with self.assertRaisesRegex(RuntimeError, "input is on CPU"):
            pym_skel_state.inverse(state, backend="triton")

    def test_multiply_matches_torch_forward_and_backward(self) -> None:
        _cuda_device()
        torch.manual_seed(0)
        state1 = _random_cuda_skel_state(256)
        state2 = _random_cuda_skel_state(256)
        state1_torch = state1.detach().clone().requires_grad_(True)
        state2_torch = state2.detach().clone().requires_grad_(True)
        state1_triton = state1.detach().clone().requires_grad_(True)
        state2_triton = state2.detach().clone().requires_grad_(True)

        out_torch = pym_skel_state.multiply(state1_torch, state2_torch)
        out_triton = pym_skel_state.multiply(
            state1_triton,
            state2_triton,
            backend="triton",
        )
        torch.testing.assert_close(out_triton, out_torch, atol=1e-5, rtol=1e-5)

        grad = torch.randn_like(out_torch)
        torch.autograd.backward(out_torch, grad)
        torch.autograd.backward(out_triton, grad)
        self._assert_grad_close(state1_triton, state1_torch)
        self._assert_grad_close(state2_triton, state2_torch)

    def test_multiply_assume_normalized_matches_torch_backward(self) -> None:
        _cuda_device()
        torch.manual_seed(1)
        state1 = _random_cuda_skel_state(256)
        state2 = _random_cuda_skel_state(256)
        state1_torch = state1.detach().clone().requires_grad_(True)
        state2_torch = state2.detach().clone().requires_grad_(True)
        state1_triton = state1.detach().clone().requires_grad_(True)
        state2_triton = state2.detach().clone().requires_grad_(True)

        out_torch = pym_skel_state.multiply_assume_normalized(
            state1_torch,
            state2_torch,
        )
        out_triton = pym_skel_state.multiply_assume_normalized(
            state1_triton,
            state2_triton,
            backend="triton",
        )
        torch.testing.assert_close(out_triton, out_torch, atol=1e-5, rtol=1e-5)

        grad = torch.randn_like(out_torch)
        torch.autograd.backward(out_torch, grad)
        torch.autograd.backward(out_triton, grad)
        self._assert_grad_close(state1_triton, state1_torch)
        self._assert_grad_close(state2_triton, state2_torch)

    def test_inverse_matches_torch_forward_and_backward(self) -> None:
        _cuda_device()
        torch.manual_seed(2)
        state = _random_cuda_skel_state(256)
        state_torch = state.detach().clone().requires_grad_(True)
        state_triton = state.detach().clone().requires_grad_(True)

        out_torch = pym_skel_state.inverse(state_torch)
        out_triton = pym_skel_state.inverse(state_triton, backend="triton")
        torch.testing.assert_close(out_triton, out_torch, atol=1e-5, rtol=1e-5)

        grad = torch.randn_like(out_torch)
        torch.autograd.backward(out_torch, grad)
        torch.autograd.backward(out_triton, grad)
        self._assert_grad_close(state_triton, state_torch)

    def _assert_grad_close(
        self,
        triton_tensor: torch.Tensor,
        torch_tensor: torch.Tensor,
    ) -> None:
        triton_grad = triton_tensor.grad
        torch_grad = torch_tensor.grad
        self.assertIsNotNone(triton_grad)
        self.assertIsNotNone(torch_grad)
        if triton_grad is None or torch_grad is None:
            return
        torch.testing.assert_close(triton_grad, torch_grad, atol=1e-5, rtol=1e-5)
