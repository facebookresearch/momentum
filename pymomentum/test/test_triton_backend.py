# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.geometry_test_utils as pym_test_utils  # @manual=:geometry_test_utils
import pymomentum.quaternion as pym_quaternion
import pymomentum.torch.character as pym_character
import torch
from pymomentum.backend import selection as backend_selection


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable")
    device = torch.device("cuda")
    torch.empty((), device=device)
    return device


class TritonBackendTest(unittest.TestCase):
    def test_auto_backend_selection(self) -> None:
        cpu_tensor = torch.empty((1,), dtype=torch.float32)

        self.assertEqual(backend_selection.resolve_backend("auto", cpu_tensor), "torch")
        self.assertEqual(
            backend_selection.resolve_backend(
                "auto",
                cpu_tensor.to(torch.float64),
                use_double_precision=True,
            ),
            "torch",
        )
        self.assertEqual(
            backend_selection.resolve_backend("torch", cpu_tensor), "torch"
        )
        self.assertEqual(
            backend_selection.resolve_backend("triton", cpu_tensor), "triton"
        )

        if torch.cuda.is_available() and backend_selection._HAS_TRITON:
            cuda_tensor = cpu_tensor.cuda()
            self.assertEqual(
                backend_selection.resolve_backend("auto", cuda_tensor), "triton"
            )

    def test_fk_matches_torch(self) -> None:
        device = _cuda_device()
        character_cpu = pym_test_utils.create_test_character()
        character = pym_character.Character(
            character_cpu,
            has_parameter_transform=False,
            has_skeleton=True,
            has_rest_mesh=False,
            has_skinning=False,
        ).to(device)

        torch.manual_seed(0)
        joint_params = torch.rand(
            5,
            character_cpu.skeleton.size,
            7,
            device=device,
            dtype=torch.float32,
        )
        self._assert_joint_parameter_backend_close(character, joint_params)

        batched_joint_params = torch.rand(
            2,
            3,
            character_cpu.skeleton.size,
            7,
            device=device,
            dtype=torch.float32,
        )
        self._assert_joint_parameter_backend_close(character, batched_joint_params)

    def test_fk_with_model_parameters_matches_torch(self) -> None:
        device = _cuda_device()
        character_cpu = pym_test_utils.create_test_character()
        character = pym_character.Character(
            character_cpu,
            has_parameter_transform=True,
            has_skeleton=True,
            has_rest_mesh=False,
            has_skinning=False,
        ).to(device)

        torch.manual_seed(0)
        model_params = torch.rand(
            5,
            character_cpu.parameter_transform.size,
            device=device,
            dtype=torch.float32,
        )

        torch_params = model_params.detach().clone().requires_grad_()
        triton_params = model_params.detach().clone().requires_grad_()

        torch_state = character.model_parameters_to_skeleton_state(
            torch_params,
            use_double_precision=False,
            fk_backend="torch",
        )
        triton_state = character.model_parameters_to_skeleton_state(
            triton_params,
            use_double_precision=False,
            fk_backend="triton",
        )

        torch.testing.assert_close(torch_state, triton_state, atol=1e-5, rtol=1e-5)

        grad_output = torch.randn_like(torch_state)
        torch_state.backward(grad_output)
        triton_state.backward(grad_output)

        torch_grad = torch_params.grad
        triton_grad = triton_params.grad
        self.assertIsNotNone(torch_grad)
        self.assertIsNotNone(triton_grad)
        if torch_grad is None or triton_grad is None:
            return
        torch.testing.assert_close(torch_grad, triton_grad, atol=1e-4, rtol=1e-4)

    def test_inverse_fk_matches_torch(self) -> None:
        device = _cuda_device()
        character_cpu = pym_test_utils.create_test_character()
        character = pym_character.Character(
            character_cpu,
            has_parameter_transform=False,
            has_skeleton=True,
            has_rest_mesh=False,
            has_skinning=False,
        ).to(device)

        torch.manual_seed(0)
        joint_params = torch.rand(
            5,
            character_cpu.skeleton.size,
            7,
            device=device,
            dtype=torch.float32,
        )

        skel_state = character.joint_parameters_to_skeleton_state(
            joint_params,
            use_double_precision=False,
            fk_backend="triton",
        )
        torch_state = skel_state.detach().clone().requires_grad_()
        triton_state = skel_state.detach().clone().requires_grad_()

        torch_params = character.skeleton.skeleton_state_to_joint_parameters(
            torch_state,
            backend="torch",
        )
        triton_params = character.skeleton.skeleton_state_to_joint_parameters(
            triton_state,
            backend="triton",
        )

        torch.testing.assert_close(torch_params, triton_params, atol=1e-5, rtol=1e-5)

        grad_output = torch.randn_like(torch_params)
        torch_params.backward(grad_output)
        triton_params.backward(grad_output)

        torch_grad = torch_state.grad
        triton_grad = triton_state.grad
        self.assertIsNotNone(torch_grad)
        self.assertIsNotNone(triton_grad)
        if torch_grad is None or triton_grad is None:
            return
        torch.testing.assert_close(torch_grad, triton_grad, atol=1e-4, rtol=1e-4)

    def test_quaternion_to_rotation_matrix_matches_torch(self) -> None:
        device = _cuda_device()

        torch.manual_seed(0)
        quaternions = torch.randn((257, 4), device=device, dtype=torch.float32)

        self._assert_to_rotation_matrix_close(
            quaternions,
            assume_normalized=False,
            eps=1e-12,
        )

    def test_quaternion_to_rotation_matrix_assume_normalized_matches_torch(
        self,
    ) -> None:
        device = _cuda_device()

        torch.manual_seed(0)
        quaternions = torch.randn((2, 3, 257, 4), device=device, dtype=torch.float32)
        quaternions = pym_quaternion.normalize(quaternions)

        self._assert_to_rotation_matrix_close(
            quaternions,
            assume_normalized=True,
            eps=1e-12,
        )

    def test_quaternion_from_rotation_matrix_matches_torch(self) -> None:
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

    def _assert_joint_parameter_backend_close(
        self,
        character: pym_character.Character,
        joint_params: torch.Tensor,
    ) -> None:
        torch_params = joint_params.detach().clone().requires_grad_()
        triton_params = joint_params.detach().clone().requires_grad_()

        torch_state = character.joint_parameters_to_skeleton_state(
            torch_params,
            use_double_precision=False,
            fk_backend="torch",
        )
        triton_state = character.joint_parameters_to_skeleton_state(
            triton_params,
            use_double_precision=False,
            fk_backend="triton",
        )

        torch.testing.assert_close(torch_state, triton_state, atol=1e-5, rtol=1e-5)

        grad_output = torch.randn_like(torch_state)
        torch_state.backward(grad_output)
        triton_state.backward(grad_output)

        torch_grad = torch_params.grad
        triton_grad = triton_params.grad
        self.assertIsNotNone(torch_grad)
        self.assertIsNotNone(triton_grad)
        if torch_grad is None or triton_grad is None:
            return
        torch.testing.assert_close(torch_grad, triton_grad, atol=1e-4, rtol=1e-4)

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
