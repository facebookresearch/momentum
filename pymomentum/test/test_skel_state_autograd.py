# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional

import numpy as np
import pymomentum.diff_geometry as pym_diff_geometry  # @manual=:diff_geometry
import pymomentum.geometry as pym_geometry  # @manual=:geometry
import pymomentum.geometry_test_utils as pym_test_utils  # @manual=:geometry_test_utils
import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import torch
from torch.nn import Parameter as P

# Flag to check if autograd is enabled (disabled in arvr build modes)
AUTOGRAD_ENABLED: bool = pym_diff_geometry.AUTOGRAD_ENABLED

# Tests that require autograd; excluded via load_tests() when unavailable
# to avoid "Skipping" notifications in CI.
_AUTOGRAD_TESTS = frozenset(
    {
        "test_skel_state_to_transforms",
        "test_bulk_multiplication_backward",
        "test_multiply_backprop",
        "test_blend_gradient_check",
        "test_blend_gradient_flows",
    }
)


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: Optional[str],
) -> unittest.TestSuite:
    """Custom test loader that excludes autograd tests when autograd is unavailable.

    This prevents 'Skipping' notifications in CI by not discovering
    the tests at all, rather than discovering and then skipping them.
    """
    if AUTOGRAD_ENABLED:
        return standard_tests

    filtered = unittest.TestSuite()
    for suite in standard_tests:
        for test in suite:
            if test._testMethodName not in _AUTOGRAD_TESTS:
                filtered.addTest(test)
    return filtered


class TestSkelStateAutograd(unittest.TestCase):
    def test_skel_state_to_transforms(self) -> None:
        character = pym_test_utils.create_test_character()
        nBatch = 2
        modelParams = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size),
            dtype=np.float64,
        )
        joint_params = character.parameter_transform.apply(modelParams)
        skel_state = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(character, joint_params)
        )
        skel_state_d = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(
                character, joint_params.astype(np.float64)
            )
        ).requires_grad_(True)
        inputs = [skel_state_d]
        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                pym_skel_state.to_matrix,
                inputs,
                eps=1e-3,
                atol=1e-4,
                raise_exception=True,
            )

        self.assertTrue(torch.allclose(skel_state, skel_state_d.detach(), atol=1e-3))

    def init_sim3_like_skel_state(
        self, batch_size: int, nr_joints: int
    ) -> torch.Tensor:
        """Initialize skel_state similar to how sim3 tests initialize their state"""
        t = torch.randn((batch_size, nr_joints, 3), dtype=torch.float32)
        q = torch.randn((batch_size, nr_joints, 4), dtype=torch.float32)
        s = torch.randn((batch_size, nr_joints, 1), dtype=torch.float32)

        q = pym_quaternion.normalize(q)
        s = torch.exp2(s).clamp(max=2)
        return torch.cat([t, q, s], dim=-1)

    def test_bulk_multiplication_backward(self) -> None:
        """Test multiplication backward pass using the same approach as sim3 tests"""
        torch.manual_seed(10023893)
        state1 = self.init_sim3_like_skel_state(1024, 159)
        state2 = self.init_sim3_like_skel_state(1024, 159)

        ps1 = P(state1)
        ps2 = P(state2)

        state = pym_skel_state.multiply(ps1, ps2)
        grad = torch.randn_like(state)

        ds1, ds2 = torch.autograd.grad(
            outputs=[state],
            inputs=[ps1, ps2],
            grad_outputs=[grad],
        )

        # Verify gradients are not None and have correct shape
        self.assertIsNotNone(ds1)
        self.assertIsNotNone(ds2)
        self.assertEqual(ds1.shape, state1.shape)
        self.assertEqual(ds2.shape, state2.shape)

    def test_multiply_backprop(self) -> None:
        """Test multiply_backprop function matches autograd gradients"""
        torch.manual_seed(10023893)
        state1 = self.init_sim3_like_skel_state(1024, 159)
        state2 = self.init_sim3_like_skel_state(1024, 159)

        ps1 = P(state1)
        ps2 = P(state2)

        state = pym_skel_state.multiply(ps1, ps2)
        grad = torch.randn_like(state)

        ds1_autograd, ds2_autograd = torch.autograd.grad(
            outputs=[state],
            inputs=[ps1, ps2],
            grad_outputs=[grad],
        )

        # Test our custom backprop function
        ds1_custom, ds2_custom = pym_skel_state.multiply_backprop(ps1, ps2, grad)

        # Compare gradients - use appropriate tolerance for numerical differences
        self.assertTrue(torch.allclose(ds1_autograd, ds1_custom, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(ds2_autograd, ds2_custom, atol=1e-4, rtol=1e-4))

    @staticmethod
    def _make_random_skel_states(n: int) -> torch.Tensor:
        states = []
        for _ in range(n):
            t = torch.randn(3, dtype=torch.float64)
            q = pym_quaternion.normalize(torch.randn(4, dtype=torch.float64))
            s = torch.rand(1, dtype=torch.float64) + 0.5
            states.append(torch.cat([t, q, s]))
        return torch.stack(states)

    def test_blend_gradient_check(self) -> None:
        """Verify gradcheck passes for skel_state.blend via the SVD backward."""
        torch.manual_seed(42)
        skel_states = self._make_random_skel_states(4).requires_grad_(True)
        weights = torch.tensor(
            [0.6, 0.2, 0.1, 0.1], dtype=torch.float64, requires_grad=True
        )

        def _blend_to_matrix(ss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return pym_skel_state.to_matrix(pym_skel_state.blend(ss, w))

        self.assertTrue(
            torch.autograd.gradcheck(
                _blend_to_matrix, (skel_states, weights), eps=1e-6
            ),
        )

    def test_blend_gradient_flows(self) -> None:
        """Verify gradients flow through skel_state.blend to inputs."""
        torch.manual_seed(42)
        skel_states = self._make_random_skel_states(3).requires_grad_(True)
        weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64, requires_grad=True)

        result = pym_skel_state.blend(skel_states, weights)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(skel_states.grad)
        self.assertFalse(torch.all(skel_states.grad == 0))
        self.assertIsNotNone(weights.grad)
        self.assertFalse(torch.all(weights.grad == 0))


if __name__ == "__main__":
    unittest.main()
