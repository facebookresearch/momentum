# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple

import pymomentum.geometry as pym_geometry
import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import torch
from torch.nn import Parameter as P

# Flag to check if autograd is enabled (disabled in arvr build modes)
AUTOGRAD_ENABLED: bool = pym_geometry.AUTOGRAD_ENABLED


def generate_skel_state_components(
    sz: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [sz, 3
    trans: torch.Tensor = torch.normal(
        mean=0,
        std=4,
        size=(sz, 3),
        dtype=torch.float64,
        requires_grad=AUTOGRAD_ENABLED,
    )

    rot: torch.Tensor = pym_quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(sz, 4),
            dtype=torch.float64,
            requires_grad=AUTOGRAD_ENABLED,
        )
    )

    scale: torch.Tensor = torch.rand(
        size=(sz, 1), dtype=torch.float64, requires_grad=AUTOGRAD_ENABLED
    )
    return (trans, rot, scale)


def generate_random_skel_state(sz: int) -> torch.Tensor:
    (trans, rot, scale) = generate_skel_state_components(sz)
    return torch.cat([trans, rot, scale], -1)


class TestSkelState(unittest.TestCase):
    def test_skel_state_to_transforms(self) -> None:
        character = pym_geometry.create_test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )
        joint_params = character.parameter_transform.apply(modelParams)
        skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params
        )
        skel_state_d = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params.to(torch.double)
        )
        inputs = [skel_state_d]
        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                pym_skel_state.to_matrix,
                inputs,
                eps=1e-3,
                atol=1e-4,
                raise_exception=True,
            )

        self.assertTrue(torch.allclose(skel_state, skel_state_d, atol=1e-3))

    def test_multiply(self) -> None:
        nMats = 6
        s1 = generate_random_skel_state(nMats)
        s2 = generate_random_skel_state(nMats)

        s12 = pym_skel_state.multiply(s1, s2)
        m12 = pym_skel_state.to_matrix(s12)

        m1 = pym_skel_state.to_matrix(s1)
        m2 = pym_skel_state.to_matrix(s2)

        self.assertLess(
            torch.norm(torch.bmm(m1, m2) - m12),
            1e-4,
            "Multiplication is correct",
        )

    def test_inverse(self) -> None:
        nMats = 6
        s = generate_random_skel_state(nMats)
        s_inv = pym_skel_state.inverse(s)

        m = pym_skel_state.to_matrix(s)
        m_inv = pym_skel_state.to_matrix(s_inv)
        m_inv2 = torch.linalg.inv(m)

        self.assertLess(
            torch.norm(m_inv - m_inv2),
            1e-4,
            "Inverse is correct",
        )

    def test_transform_points(self) -> None:
        nMats = 6
        s = generate_random_skel_state(nMats)
        m = pym_skel_state.to_matrix(s)
        pts = torch.rand(size=(nMats, 3), dtype=torch.float64, requires_grad=False)
        hpts = torch.cat([pts, torch.ones(nMats, 1)], -1)
        transformed1 = torch.bmm(m, hpts.unsqueeze(-1))[:, 0:3].squeeze(-1)
        transformed2 = pym_skel_state.transform_points(s, pts)

        self.assertLess(
            torch.norm(transformed1 - transformed2),
            1e-4,
            "Inverse is correct",
        )

    def test_construct(self) -> None:
        nMats = 4
        (trans, rot, scale) = generate_skel_state_components(nMats)

        s1 = torch.cat([trans, rot, scale], -1)
        s2 = pym_skel_state.multiply(
            pym_skel_state.from_translation(trans),
            pym_skel_state.multiply(
                pym_skel_state.from_quaternion(rot), pym_skel_state.from_scale(scale)
            ),
        )

        self.assertLess(
            torch.norm(s1 - s2),
            1e-4,
            "constructing from components",
        )

    def test_matrix_to_skel_state(self) -> None:
        nMats = 6
        s1 = generate_random_skel_state(nMats)
        m1 = pym_skel_state.to_matrix(s1)
        s2 = pym_skel_state.from_matrix(m1)
        m2 = pym_skel_state.to_matrix(s2)
        self.assertLess(
            torch.norm(m1 - m2),
            1e-4,
            "matrix-to-skel-state conversion:L matrices should always match",
        )

        # Translation and scale should match.  Rotation might have its sign flipped:
        self.assertLess(
            torch.norm(s1[..., 0:3] - s2[..., 0:3]),
            1e-4,
            "matrix-to-skel-state conversion: translations should match",
        )
        self.assertLess(
            torch.norm(s1[..., 7] - s2[..., 7]),
            1e-4,
            "matrix-to-skel-state conversion: scales should match",
        )

    def test_skel_state_blending(self) -> None:
        t1 = torch.Tensor([1, 0, 0])
        t2 = torch.Tensor([0, 1, 0])

        s1 = pym_skel_state.from_translation(t1)
        s2 = pym_skel_state.from_translation(t2)
        s_test = pym_skel_state.from_translation(torch.Tensor([0.25, 0.75, 0]))

        weights = torch.Tensor([0.25, 0.75])
        s_blended = pym_skel_state.blend(torch.stack([s1, s2], 0), weights)

        self.assertLess(
            torch.norm(s_blended - s_test),
            1e-4,
            "matrix-to-skel-state blending",
        )

    def test_skel_state_splitting(self) -> None:
        t1 = torch.Tensor([1, 0, 0])
        skel_state1 = pym_skel_state.from_translation(t1)
        t2, _, _ = pym_skel_state.split(skel_state1)
        self.assertTrue(torch.allclose(t1, t2))

        r3 = torch.Tensor([0, 1, 0, 0])
        skel_state3 = pym_skel_state.from_quaternion(r3)
        t4, r4, s4 = pym_skel_state.split(skel_state3)
        self.assertTrue(torch.allclose(r3, r4))

        s5 = torch.Tensor([2])
        skel_state5 = pym_skel_state.from_scale(s5)
        t6, r6, s6 = pym_skel_state.split(skel_state5)
        self.assertTrue(torch.allclose(s5, s6))

    def test_check_invalid_shape(self) -> None:
        # Test case where skel_state.shape[-1] != 8 (triggers ValueError)
        invalid_skel_state = torch.randn(2, 7)  # Wrong last dimension (7 instead of 8)

        with self.assertRaises(ValueError) as context:
            pym_skel_state.check(invalid_skel_state)

        self.assertIn(
            "Expected skeleton state to have last dimension 8", str(context.exception)
        )

    def test_identity_no_size(self) -> None:
        # Test case where size=None (triggers the else branch in identity function)
        identity_state = pym_skel_state.identity()

        # Should return a 1D tensor with 8 elements
        self.assertEqual(identity_state.shape, (8,))
        # Should have zeros for translation, identity quaternion, and ones for scale
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(identity_state, expected))

    def test_from_matrix_invalid_dimensions(self) -> None:
        # Test case where matrices don't have correct dimensions
        invalid_matrix = torch.randn(3, 3)  # Should be 4x4

        with self.assertRaises(ValueError) as context:
            pym_skel_state.from_matrix(invalid_matrix)

        self.assertIn("Expected a tensor of 4x4 matrices", str(context.exception))

    def test_from_matrix_2d_case(self) -> None:
        # Test case where matrices.dim() == 2 (triggers unsqueeze)
        # Create a single 4x4 matrix
        matrix_2d = torch.eye(4)
        matrix_2d[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Add translation

        skel_state = pym_skel_state.from_matrix(matrix_2d)

        # Should return a 1D tensor with 8 elements
        self.assertEqual(skel_state.shape, (8,))
        # Translation should match what we set
        self.assertTrue(torch.allclose(skel_state[:3], torch.tensor([1.0, 2.0, 3.0])))

    def test_transform_points_invalid_dimensions(self) -> None:
        # Test case where points.dim() < 1 or points.shape[-1] != 3
        skel_state = pym_skel_state.identity()

        # Test points with wrong last dimension
        invalid_points = torch.randn(5, 2)  # Should be (..., 3)

        with self.assertRaises(ValueError) as context:
            pym_skel_state.transform_points(skel_state, invalid_points)

        self.assertIn(
            "Points tensor should have last dimension 3", str(context.exception)
        )

        # Test points with dim() < 1 (scalar)
        scalar_points = torch.tensor(1.0)  # 0 dimensions

        with self.assertRaises(ValueError) as context:
            pym_skel_state.transform_points(skel_state, scalar_points)

        self.assertIn(
            "Points tensor should have last dimension 3", str(context.exception)
        )

    def test_slerp_extremes(self) -> None:
        nMats = 6
        s0 = generate_random_skel_state(nMats)
        s1 = generate_random_skel_state(nMats)

        # Test slerp at t=0 should return s0
        s_slerp_0 = pym_skel_state.slerp(s0, s1, torch.tensor([0.0]))
        self.assertTrue(
            torch.allclose(
                pym_skel_state.to_matrix(s_slerp_0),
                pym_skel_state.to_matrix(s0),
                atol=1e-4,
            ),
            "Expected slerp result to be s0 when t=0.",
        )

        # Test slerp at t=1 should return s1
        s_slerp_1 = pym_skel_state.slerp(s0, s1, torch.tensor([1.0]))
        self.assertTrue(
            torch.allclose(
                pym_skel_state.to_matrix(s_slerp_1),
                pym_skel_state.to_matrix(s1),
                atol=1e-4,
            ),
            "Expected slerp result to be s1 when t=1.",
        )

    def test_slerp_midpoint(self) -> None:
        nMats = 6
        s0 = generate_random_skel_state(nMats)
        s1 = generate_random_skel_state(nMats)

        # Perform slerp at t=0.5
        s_slerp_mid = pym_skel_state.slerp(s0, s1, torch.tensor([0.5]))

        # Extract components
        t0, _, s0_scale = pym_skel_state.split(s0)
        t1, _, s1_scale = pym_skel_state.split(s1)
        t_mid, _, s_mid_scale = pym_skel_state.split(s_slerp_mid)

        # Check if translation and scale are in the middle
        self.assertTrue(
            torch.allclose(t_mid, (t0 + t1) / 2, atol=1e-4),
            "Expected translation to be the midpoint when t=0.5.",
        )
        self.assertTrue(
            torch.allclose(s_mid_scale, (s0_scale + s1_scale) / 2, atol=1e-4),
            "Expected scale to be the midpoint when t=0.5.",
        )

    # Additional tests migrated from sim3 to ensure complete coverage
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

    def test_bulk_multiplication(self) -> None:
        """Test multiplication using the same approach as sim3 tests"""
        torch.manual_seed(1002389)
        state1 = self.init_sim3_like_skel_state(1024, 159)
        state2 = self.init_sim3_like_skel_state(1024, 159)
        state = pym_skel_state.multiply(state1, state2)

        # Convert to TRS format (like sim3_to_trs_state)
        t1, q1, s1 = pym_skel_state.split(state1)
        t2, q2, s2 = pym_skel_state.split(state2)
        r1 = pym_quaternion.to_rotation_matrix(q1)
        r2 = pym_quaternion.to_rotation_matrix(q2)

        # Expected multiplication results (matching sim3 formula)
        t_expected = torch.einsum("bjxy,bjy->bjx", r1, t2) * s1 + t1
        r_expected = torch.einsum("bjxy,bjyz->bjxz", r1, r2)
        s_expected = s1 * s2

        # Get actual results
        tt, qq, ss = pym_skel_state.split(state)
        rr = pym_quaternion.to_rotation_matrix(qq)

        self.assertTrue(torch.allclose(t_expected, tt, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(r_expected, rr, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(s_expected, ss, atol=1e-5, rtol=1e-5))

    def test_bulk_points_transformation(self) -> None:
        """Test point transformation using the same approach as sim3 tests"""
        torch.manual_seed(1002389)
        state = self.init_sim3_like_skel_state(1024, 159)
        points = torch.randn((1024, 159, 3))

        # Apply transformation
        p1 = pym_skel_state.transform_points(state, points)

        # Expected result (matching apply_sim3_on_points formula)
        tt, qq, ss = pym_skel_state.split(state)
        rr = pym_quaternion.to_rotation_matrix(qq)
        p2 = ss * torch.einsum("bjxy,bjy->bjx", rr, points) + tt

        self.assertTrue(torch.allclose(p1, p2, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(AUTOGRAD_ENABLED, "Autograd not available in ARVR build modes")
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

    def test_bulk_inverse(self) -> None:
        """Test inverse using the same approach as sim3 tests"""
        torch.manual_seed(1002389)
        state = self.init_sim3_like_skel_state(1024, 159)
        state_inv = pym_skel_state.inverse(state)

        # Expected identity result
        state_expected = torch.zeros_like(state)
        state_expected[..., 6] = 1  # quaternion w component = 1
        state_expected[..., 7] = 1  # scale component = 1

        # Test both directions of multiplication
        state2 = pym_skel_state.multiply(state, state_inv)
        self.assertTrue(torch.allclose(state2, state_expected, atol=1e-5, rtol=1e-5))

        state3 = pym_skel_state.multiply(state_inv, state)
        self.assertTrue(torch.allclose(state3, state_expected, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(AUTOGRAD_ENABLED, "Autograd not available in ARVR build modes")
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

    def test_multiply_broadcasting(self) -> None:
        """Test that multiplication works correctly with broadcasting for different tensor dimensions"""
        torch.manual_seed(42)

        # Create skel states with different shapes for broadcasting
        # First state: [3, 2, 8] - larger first dimension
        state1 = generate_random_skel_state(6).reshape(3, 2, 8)

        # Second state: [1, 2, 8] - smaller first dimension (will be broadcasted)
        state2 = generate_random_skel_state(2).reshape(1, 2, 8)

        # Multiply with broadcasting
        result_broadcast = pym_skel_state.multiply(state1, state2)

        # Manual expansion for reference
        state2_expanded = state2.expand(3, 2, 8)
        result_manual = pym_skel_state.multiply(state1, state2_expanded)

        # Results should be the same
        self.assertTrue(
            torch.allclose(result_broadcast, result_manual, atol=1e-6),
            "Broadcasting multiplication should match manually expanded multiplication",
        )

        # Test with different broadcasting patterns
        # Test broadcasting with compatible dimensions: [2, 1, 8] vs [2, 3, 8]
        state_narrow = generate_random_skel_state(2).reshape(2, 1, 8)
        state_wide = generate_random_skel_state(6).reshape(2, 3, 8)

        result_narrow_broadcast = pym_skel_state.multiply(state_narrow, state_wide)
        narrow_expanded = state_narrow.expand(2, 3, 8)
        result_narrow_manual = pym_skel_state.multiply(narrow_expanded, state_wide)

        self.assertTrue(
            torch.allclose(result_narrow_broadcast, result_narrow_manual, atol=1e-6),
            "Narrow state broadcasting should work correctly",
        )

        # Test reverse broadcasting: [2, 3, 8] vs [2, 1, 8]
        result_reverse_broadcast = pym_skel_state.multiply(state_wide, state_narrow)
        result_reverse_manual = pym_skel_state.multiply(state_wide, narrow_expanded)

        self.assertTrue(
            torch.allclose(result_reverse_broadcast, result_reverse_manual, atol=1e-6),
            "Reverse broadcasting should work correctly",
        )

        # Verify shapes are correct
        self.assertEqual(result_broadcast.shape, (3, 2, 8))
        self.assertEqual(result_narrow_broadcast.shape, (2, 3, 8))
        self.assertEqual(result_reverse_broadcast.shape, (2, 3, 8))

        # Test that results preserve transformation properties by converting to matrices
        # Broadcasting result should match element-wise multiplication
        for i in range(3):
            for j in range(2):
                m1 = pym_skel_state.to_matrix(state1[i, j])
                m2 = pym_skel_state.to_matrix(state2[0, j])  # Broadcasted dimension
                expected_matrix = torch.matmul(m1, m2)
                actual_matrix = pym_skel_state.to_matrix(result_broadcast[i, j])

                self.assertTrue(
                    torch.allclose(expected_matrix, actual_matrix, atol=1e-5),
                    f"Matrix multiplication should match for element [{i}, {j}]",
                )


if __name__ == "__main__":
    unittest.main()
