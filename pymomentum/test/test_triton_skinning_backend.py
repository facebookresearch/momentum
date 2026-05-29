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


class TritonSkinningBackendTest(unittest.TestCase):
    def test_skinning_matches_torch(self) -> None:
        device = _cuda_device()
        character_cpu = pym_test_utils.create_test_character()
        character = pym_character.Character(
            character_cpu,
            has_parameter_transform=False,
            has_skeleton=True,
            has_rest_mesh=True,
            has_skinning=True,
        ).to(device)

        torch.manual_seed(0)
        joint_params = torch.rand(
            2,
            3,
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

        rest_vertices = character.mesh.rest_vertices.detach().clone()
        torch_state = skel_state.detach().clone().requires_grad_()
        triton_state = skel_state.detach().clone().requires_grad_()
        torch_vertices = rest_vertices.detach().clone().requires_grad_()
        triton_vertices = rest_vertices.detach().clone().requires_grad_()

        torch_skinned = character.skin_points(
            torch_state,
            torch_vertices,
            backend="torch",
        )
        triton_skinned = character.skin_points(
            triton_state,
            triton_vertices,
            backend="triton",
        )

        torch.testing.assert_close(
            torch_skinned,
            triton_skinned,
            atol=1e-4,
            rtol=1e-4,
        )

        grad_output = torch.randn_like(torch_skinned)
        torch_skinned.backward(grad_output)
        triton_skinned.backward(grad_output)

        self._assert_grad_close(triton_state, torch_state, atol=1e-3, rtol=1e-3)
        self._assert_grad_close(triton_vertices, torch_vertices, atol=1e-4, rtol=1e-4)

    def test_sequence_rest_vertices_are_broadcast(self) -> None:
        device = _cuda_device()
        character_cpu = pym_test_utils.create_test_character()
        character = pym_character.Character(
            character_cpu,
            has_parameter_transform=False,
            has_skeleton=True,
            has_rest_mesh=True,
            has_skinning=True,
        ).to(device)

        torch.manual_seed(0)
        joint_params = torch.rand(
            2,
            3,
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
        rest_vertices = character.mesh.rest_vertices.unsqueeze(0).expand(2, -1, -1)

        torch_skinned = character.skin_points(
            skel_state,
            rest_vertices.unsqueeze(1),
            backend="torch",
        )
        triton_skinned = character.skin_points(
            skel_state,
            rest_vertices,
            backend="triton",
        )

        self.assertEqual(triton_skinned.shape, torch_skinned.shape)
        torch.testing.assert_close(
            torch_skinned,
            triton_skinned,
            atol=1e-4,
            rtol=1e-4,
        )

    def _assert_grad_close(
        self,
        triton_tensor: torch.Tensor,
        torch_tensor: torch.Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        triton_grad = triton_tensor.grad
        torch_grad = torch_tensor.grad
        self.assertIsNotNone(triton_grad)
        self.assertIsNotNone(torch_grad)
        if triton_grad is None or torch_grad is None:
            return
        torch.testing.assert_close(triton_grad, torch_grad, atol=atol, rtol=rtol)
