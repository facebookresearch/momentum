# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.geometry_test_utils as pym_test_utils  # @manual=:geometry_test_utils
import pymomentum.torch.character as pym_character
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


class TritonFKBackendTest(unittest.TestCase):
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

        device = _cuda_device()
        self.assertEqual(
            backend_selection.resolve_backend("auto", cpu_tensor.to(device)), "triton"
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
            backend="torch",
        )
        triton_state = character.model_parameters_to_skeleton_state(
            triton_params,
            use_double_precision=False,
            backend="triton",
        )

        torch.testing.assert_close(torch_state, triton_state, atol=1e-5, rtol=1e-5)

        grad_output = torch.randn_like(torch_state)
        torch_state.backward(grad_output)
        triton_state.backward(grad_output)
        self._assert_grad_close(triton_params, torch_params)

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
            backend="triton",
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
        self._assert_grad_close(triton_state, torch_state)

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
            backend="torch",
        )
        triton_state = character.joint_parameters_to_skeleton_state(
            triton_params,
            use_double_precision=False,
            backend="triton",
        )

        torch.testing.assert_close(torch_state, triton_state, atol=1e-5, rtol=1e-5)

        grad_output = torch.randn_like(torch_state)
        torch_state.backward(grad_output)
        triton_state.backward(grad_output)
        self._assert_grad_close(triton_params, torch_params)

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
        torch.testing.assert_close(triton_grad, torch_grad, atol=1e-4, rtol=1e-4)
