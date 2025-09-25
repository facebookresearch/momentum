# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import pymomentum.quaternion as quaternion
import torch
from torch.nn import Parameter as P


def generateRandomQuats(sz: int) -> torch.Tensor:
    return quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(sz, 4),
            dtype=torch.float64,
            requires_grad=True,
        )
    )


class TestQuaternion(unittest.TestCase):
    def test_euler_conversion(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nBatch = 6
        nMat = 5
        euler1 = torch.normal(
            mean=0,
            std=4,
            size=(nBatch, nMat, 3),
            dtype=torch.float64,
            requires_grad=True,
        )
        quats1 = quaternion.euler_xyz_to_quaternion(euler1)
        euler2 = quaternion.quaternion_to_xyz_euler(quats1)
        quats2 = quaternion.euler_xyz_to_quaternion(euler2)

        quaternion.inverse(quats1)

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(quats1)
                - quaternion.to_rotation_matrix(quats2)
            ),
            1e-4,
            "Expected rotation difference to be small.",
        )

        # check corner case conversion
        q_cornercase = torch.FloatTensor([0.4832, -0.5161, 0.4835, 0.5162])
        euler_cornercase = quaternion.quaternion_to_xyz_euler(q_cornercase)
        self.assertFalse(torch.isnan(euler_cornercase).any())

        # check the gradients:
        torch.autograd.gradcheck(
            quaternion.euler_xyz_to_quaternion,
            [euler1],
            raise_exception=True,
        )

        torch.autograd.gradcheck(
            quaternion.quaternion_to_xyz_euler,
            [quats1],
            raise_exception=True,
        )

    def test_matrix_to_quaternion(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nBatch = 6
        quats = generateRandomQuats(nBatch)
        mats = quaternion.to_rotation_matrix(quats)
        quats2 = quaternion.from_rotation_matrix(mats)
        mats2 = quaternion.to_rotation_matrix(quats2)
        diff = torch.minimum(
            (quats - quats).norm(dim=-1), (quats + quats2).norm(dim=-1)
        )
        self.assertLess(
            torch.norm(diff), 1e-4, "Expected quaternions to match (up to sign)"
        )
        self.assertLess(
            torch.norm(mats2 - mats), 1e-4, "Expected quaternions to match (up to sign)"
        )

    def test_multiply(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q1 = generateRandomQuats(nMat)
        q2 = generateRandomQuats(nMat)
        q12 = quaternion.multiply(q1, q2)

        m1 = quaternion.to_rotation_matrix(q1)
        m2 = quaternion.to_rotation_matrix(q2)
        m12 = torch.bmm(m1, m2)

        self.assertLess(
            torch.norm(quaternion.to_rotation_matrix(q12) - m12),
            1e-4,
            "Expected rotation difference to be small.",
        )

    def test_identity(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q1 = generateRandomQuats(nMat)
        ident = quaternion.identity().unsqueeze(0).expand_as(q1).type_as(q1)

        self.assertLess(
            torch.norm(quaternion.multiply(q1, ident) - q1),
            1e-4,
            "Identity on the right",
        )

        self.assertLess(
            torch.norm(quaternion.multiply(ident, q1) - q1),
            1e-4,
            "Identity on the right",
        )

    def test_rotate_vector(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q = generateRandomQuats(nMat)

        vec = torch.normal(
            mean=0,
            std=4,
            size=(nMat, 3),
            dtype=torch.float64,
            requires_grad=True,
        )

        rotated1 = quaternion.rotate_vector(q, vec)
        rotated2 = torch.bmm(
            quaternion.to_rotation_matrix(q), vec.unsqueeze(-1)
        ).squeeze(-1)

        self.assertLess(
            torch.norm(rotated1 - rotated2),
            1e-4,
            "Matrix rotation should match quaternion rotation",
        )

    def test_blend_same(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q = generateRandomQuats(nMat)
        q_dup = q.unsqueeze(1).expand(-1, 5, -1)
        q_blend = quaternion.blend(q_dup)

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_blend)
                - quaternion.to_rotation_matrix(q)
            ),
            1e-4,
            "quaternion blending of the same quaternion should be identity.",
        )

    def test_blend_euler(self) -> None:
        unit_x = torch.tensor([0.25 * math.pi, 0, 0]).unsqueeze(0)

        # Blend of two single-axis rotations should be the midpoint:
        quats1 = quaternion.euler_xyz_to_quaternion(0.3 * math.pi * unit_x)
        quats2 = quaternion.euler_xyz_to_quaternion(0.5 * math.pi * unit_x)
        q_blend = quaternion.blend(torch.cat([quats1, quats2], 0))
        euler_blend = quaternion.euler_xyz_to_quaternion(0.4 * math.pi * unit_x)

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_blend)
                - quaternion.to_rotation_matrix(euler_blend)
            ),
            1e-4,
            "quaternion blending of single-axis Euler rotation should be the midway rotation.",
        )

        weights = torch.tensor([0.75, 0.25])
        q_blend_weighted = quaternion.blend(torch.cat([quats1, quats2], 0), weights)
        euler_blend_weighted = quaternion.euler_xyz_to_quaternion(
            (0.3 * weights[0] + 0.5 * weights[1]) * math.pi * unit_x
        )
        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_blend_weighted)
                - quaternion.to_rotation_matrix(euler_blend_weighted)
            ),
            1e-2,
            "quaternion blending of single-axis Euler rotation should be the midway rotation.",
        )

    def test_from_two_vectors(self) -> None:
        # Test with two random vectors
        n_batch = 5
        v1 = torch.nn.functional.normalize(torch.randn(n_batch, 3), dim=-1)
        v2 = torch.nn.functional.normalize(torch.randn(n_batch, 3), dim=-1)

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(torch.allclose(rotated_v1, v2, rtol=1e-4, atol=1e-6))

    def test_from_two_vectors_opposite(self) -> None:
        # Test with two opposite vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0, 0.0])

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(torch.allclose(rotated_v1, v2))

    def test_from_two_vectors_parallel(self) -> None:
        # Test with two parallel vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([2.0, 0.0, 0.0])

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(torch.allclose(rotated_v1, v1))

    def test_check_and_normalize_weights_squeeze(self) -> None:
        # Test case where weights.dim() == quaternions.dim() (triggers squeeze)
        quaternions = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)
        weights = torch.ones(2, 3, 1)  # Shape: (2, 3, 1) - same dim as quaternions

        # This should trigger the squeeze operation
        normalized_weights = quaternion.check_and_normalize_weights(
            quaternions, weights
        )

        # Verify the weights were properly normalized
        self.assertEqual(normalized_weights.shape, (2, 3))
        self.assertTrue(torch.allclose(normalized_weights.sum(dim=-1), torch.ones(2)))

    def test_check_and_normalize_weights_dim_mismatch(self) -> None:
        # Test case where weights.dim() + 1 != quaternions.dim()
        quaternions = torch.randn(2, 3, 4)  # Shape: (2, 3, 4) - 3 dimensions
        weights = torch.ones(2)  # Shape: (2,) - 1 dimension, but 1 + 1 != 3

        with self.assertRaises(ValueError) as context:
            quaternion.check_and_normalize_weights(quaternions, weights)

        self.assertIn(
            "Expected weights vector to match quaternion vector", str(context.exception)
        )
        self.assertIn("weights=torch.Size([2])", str(context.exception))
        self.assertIn("quaternions=torch.Size([2, 3, 4])", str(context.exception))

    def test_check_and_normalize_weights_size_mismatch(self) -> None:
        # Test case where weights.size(i) != quaternions.size(i) for some dimension i
        quaternions = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)
        weights = torch.ones(2, 5)  # Shape: (2, 5) - first dim matches, second doesn't

        with self.assertRaises(ValueError) as context:
            quaternion.check_and_normalize_weights(quaternions, weights)

        self.assertIn(
            "Expected weights vector to match quaternion vector", str(context.exception)
        )
        self.assertIn("weights=torch.Size([2, 5])", str(context.exception))
        self.assertIn("quaternions=torch.Size([2, 3, 4])", str(context.exception))

    def test_slerp_extremes(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q1 = generateRandomQuats(nMat)
        q2 = generateRandomQuats(nMat)
        # Add another edge case for very close quaternions
        q1 = torch.cat((q1, torch.tensor([[-0.9662, -0.1052, -0.0235, 0.2343]])), dim=0)
        q2 = torch.cat((q2, torch.tensor([[-0.9684, -0.1252, -0.0175, 0.2151]])), dim=0)

        # Test slerp at t=0 should return q1
        q_slerp_0 = quaternion.slerp(q1, q2, torch.tensor([0.0]))
        self.assertTrue(
            torch.allclose(
                quaternion.to_rotation_matrix(q_slerp_0),
                quaternion.to_rotation_matrix(q1),
                rtol=1e-3,
                atol=1e-4,
            )
        )

        # Test slerp at t=1 should return q2
        q_slerp_1 = quaternion.slerp(q1, q2, torch.tensor([1.0]))
        self.assertTrue(
            torch.allclose(
                quaternion.to_rotation_matrix(q_slerp_1),
                quaternion.to_rotation_matrix(q2),
                rtol=1e-3,
                atol=1e-4,
            )
        )

    def test_slerp_midpoint(self) -> None:
        q1 = quaternion.euler_xyz_to_quaternion(torch.tensor([0.0, 0.0, 0.0]))
        q2 = quaternion.euler_xyz_to_quaternion(torch.tensor([math.pi / 2, 0.0, 0.0]))

        # Test slerp at t=0.5 should be in the middle
        q_result = quaternion.slerp(q1, q2, torch.tensor([0.5]))

        q_target = quaternion.euler_xyz_to_quaternion(
            torch.tensor([math.pi / 4, 0.0, 0.0])
        )
        # Check if the slerp result is approximately in the middle
        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_result)
                - quaternion.to_rotation_matrix(q_target)
            ),
            1e-1,
            "Expected slerp result to be approximately in the middle.",
        )

        q3 = quaternion.euler_xyz_to_quaternion(torch.tensor([math.pi, 0.0, 0.0]))
        q_result2 = quaternion.slerp(q1, q3, torch.tensor([0.5]))
        q_target2a = quaternion.euler_xyz_to_quaternion(
            torch.tensor([math.pi / 2, 0.0, 0.0])
        )
        q_target2b = quaternion.euler_xyz_to_quaternion(
            torch.tensor([-math.pi / 2, 0.0, 0.0])
        )
        # Could go either way around the circle, so check both
        diff2a = torch.norm(
            quaternion.to_rotation_matrix(q_result2)
            - quaternion.to_rotation_matrix(q_target2a)
        )
        diff2b = torch.norm(
            quaternion.to_rotation_matrix(q_result2)
            - quaternion.to_rotation_matrix(q_target2b)
        )
        self.assertTrue(diff2a < 1e-5 or diff2b < 1e-5)

    def test_slerp_close_matrices(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        nMat = 5
        q1 = generateRandomQuats(nMat)

        q_result = quaternion.slerp(q1, q1, torch.tensor([0.5]))

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_result)
                - quaternion.to_rotation_matrix(q1)
            ),
            1e-5,
        )

        q_result = quaternion.slerp(q1, -q1, torch.tensor([0.5]))

        self.assertLess(
            torch.norm(
                quaternion.to_rotation_matrix(q_result)
                - quaternion.to_rotation_matrix(q1)
            ),
            1e-5,
        )

    # ===============================================================
    # Custom Backpropagation Tests
    # ======================================================================

    def assert_all_close(
        self, a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5
    ) -> None:
        """Helper method for comparing tensors with a tolerance."""
        self.assertTrue(torch.allclose(a, b, atol=tol, rtol=tol))

    def test_normalize_backprop(self) -> None:
        """Test custom backpropagation for quaternion normalization."""
        torch.manual_seed(0)

        q = torch.randn((1024, 2, 4), dtype=torch.float64)
        g = torch.randn((1024, 2, 4), dtype=torch.float64)

        pq = P(q)
        qn = quaternion.normalize(pq)

        # Backward via autograd
        (dq1,) = torch.autograd.grad(
            outputs=[qn],
            inputs=[pq],
            grad_outputs=[g],
        )
        # Backward via our custom backprop function
        dq2 = quaternion.normalize_backprop(pq, g)
        self.assert_all_close(dq1, dq2)

    def test_rotate_vector_backprop(self) -> None:
        """Test custom backpropagation for quaternion vector rotation."""
        torch.manual_seed(0)

        q = torch.randn((1024, 2, 4), dtype=torch.float64)
        v = torch.randn((1024, 2, 3), dtype=torch.float64)

        pq = P(q)
        pv = P(v)

        # Forward
        qv = quaternion.rotate_vector(pq, pv)
        # Backward via autograd
        dqv = torch.randn_like(qv)
        dq1, dv1 = torch.autograd.grad(
            outputs=[qv],
            inputs=[pq, pv],
            grad_outputs=[dqv],
        )
        # Backward via our custom backprop function
        dq2, dv2 = quaternion.rotate_vector_backprop(pq, pv, dqv)

        # Check vector gradients
        self.assert_all_close(dv1, dv2)
        # Check quaternion gradients (scalar part)
        self.assert_all_close(dq1[..., 3], dq2[..., 3])
        # Check quaternion gradients (vector part)
        self.assert_all_close(dq1[..., :3], dq2[..., :3])

    def test_rotate_vector_backprop_assume_normalized(self) -> None:
        """Test custom backpropagation for quaternion vector rotation with normalized quaternions."""
        torch.manual_seed(0)

        batch_size = 1024
        q = torch.randn((batch_size, 2, 4), dtype=torch.float64)
        q = quaternion.normalize(q)
        v = torch.randn((batch_size, 2, 3), dtype=torch.float64)

        pq = P(q)
        pv = P(v)

        # Forward using the assume_normalized version
        qv = quaternion.rotate_vector(pq, pv)
        # Backward via autograd
        dqv = torch.randn_like(qv)
        dq1, dv1 = torch.autograd.grad(
            outputs=[qv],
            inputs=[pq, pv],
            grad_outputs=[dqv],
        )
        # Backward via our custom backprop function
        dq2, dv2 = quaternion.rotate_vector_backprop_assume_normalized(pq, pv, dqv)

        # Check vector gradients
        self.assert_all_close(dv1, dv2)
        # Check quaternion gradients (scalar part)
        self.assert_all_close(dq1[..., 3], dq2[..., 3])
        # Check quaternion gradients (vector part)
        self.assert_all_close(dq1[..., :3], dq2[..., :3])

    def test_multiply_backprop(self) -> None:
        """Test custom backpropagation for quaternion multiplication."""
        torch.manual_seed(0)

        q1 = torch.randn((1024, 2, 4), dtype=torch.float64)
        q2 = torch.randn((1024, 2, 4), dtype=torch.float64)

        pq1 = P(q1)
        pq2 = P(q2)

        q = quaternion.multiply(pq1, pq2)
        dq = torch.randn_like(q)

        # Backward via autograd
        dq11, dq21 = torch.autograd.grad(
            outputs=[q],
            inputs=[pq1, pq2],
            grad_outputs=[dq],
        )
        # Backward via our custom backprop function
        dq12, dq22 = quaternion.multiply_backprop(pq1, pq2, dq)

        self.assert_all_close(dq11, dq12)
        self.assert_all_close(dq21, dq22)

    def test_multiply_backprop_assume_normalized(self) -> None:
        """Test custom backpropagation for quaternion multiplication with normalized quaternions."""
        torch.manual_seed(0)

        q1 = torch.randn((1024, 2, 4), dtype=torch.float64)
        q2 = torch.randn((1024, 2, 4), dtype=torch.float64)
        q1 = quaternion.normalize(q1)
        q2 = quaternion.normalize(q2)

        pq1 = P(q1)
        pq2 = P(q2)

        q = quaternion.multiply(pq1, pq2)
        dq = torch.randn_like(q)

        # Backward via autograd
        dq11, dq21 = torch.autograd.grad(
            outputs=[q],
            inputs=[pq1, pq2],
            grad_outputs=[dq],
        )
        # Backward via our custom backprop function
        dq12, dq22 = quaternion.multiply_backprop_assume_normalized(pq1, pq2, dq)

        self.assert_all_close(dq11, dq12)
        self.assert_all_close(dq21, dq22)

    def test_backprop_consistency(self) -> None:
        """Test that the normalizing and assume_normalized backprop versions are consistent."""
        torch.manual_seed(0)

        # Test with already normalized quaternions - both versions should give same results
        q1 = torch.randn((100, 4), dtype=torch.float64)
        q2 = torch.randn((100, 4), dtype=torch.float64)
        q1_norm = quaternion.normalize(q1)
        q2_norm = quaternion.normalize(q2)

        pq1_norm = P(q1_norm)
        pq2_norm = P(q2_norm)

        # Test multiply backprop consistency
        q_result = quaternion.multiply(pq1_norm, pq2_norm)
        dq = torch.randn_like(q_result)

        dq1_general, dq2_general = quaternion.multiply_backprop(pq1_norm, pq2_norm, dq)
        dq1_assume, dq2_assume = quaternion.multiply_backprop_assume_normalized(
            pq1_norm, pq2_norm, dq
        )

        # The results should be very close since inputs are already normalized
        self.assert_all_close(dq1_general, dq1_assume, tol=1e-4)
        self.assert_all_close(dq2_general, dq2_assume, tol=1e-4)

        # Test rotate vector backprop consistency
        v = torch.randn((100, 3), dtype=torch.float64)
        pv = P(v)

        qv_result = quaternion.rotate_vector(pq1_norm, pv)
        dqv = torch.randn_like(qv_result)

        dq_general, dv_general = quaternion.rotate_vector_backprop(pq1_norm, pv, dqv)
        dq_assume, dv_assume = quaternion.rotate_vector_backprop_assume_normalized(
            pq1_norm, pv, dqv
        )

        # The results should be very close since input quaternion is already normalized
        self.assert_all_close(dq_general, dq_assume, tol=1e-4)
        self.assert_all_close(dv_general, dv_assume, tol=1e-4)

    def test_from_rotation_matrix_numerical_stability(self) -> None:
        """Test the enhanced matrix-to-quaternion conversion with numerical stability."""
        torch.manual_seed(0)

        # Test with random rotation matrices
        nBatch = 10
        quats_orig = generateRandomQuats(nBatch)
        matrices = quaternion.to_rotation_matrix(quats_orig)

        # Convert back using the enhanced method
        quats_recovered = quaternion.from_rotation_matrix(matrices)
        matrices_recovered = quaternion.to_rotation_matrix(quats_recovered)

        # Check that matrices match (accounting for quaternion sign ambiguity)
        self.assertLess(
            torch.norm(matrices - matrices_recovered),
            1e-4,
            "Enhanced matrix-to-quaternion conversion should be stable",
        )

        # Test with identity matrices (edge case)
        identity_matrices = torch.eye(3).expand(5, 3, 3)
        identity_quats = quaternion.from_rotation_matrix(identity_matrices)
        expected_identity = quaternion.identity().expand(5, 4)

        # Check that identity matrices produce identity quaternions (up to sign)
        diff1 = torch.norm(identity_quats - expected_identity, dim=-1)
        diff2 = torch.norm(identity_quats + expected_identity, dim=-1)
        min_diff = torch.minimum(diff1, diff2)
        self.assertLess(torch.max(min_diff).item(), 1e-4)

    def test_quaternion_inverse_numerical_stability(self) -> None:
        """Test that the improved inverse function handles near-zero quaternions."""
        torch.manual_seed(0)

        # Test with normal quaternions
        q_normal = generateRandomQuats(5)
        q_inv = quaternion.inverse(q_normal)
        q_identity = quaternion.multiply(q_normal, q_inv)
        expected_identity = quaternion.identity().expand_as(q_identity)

        self.assertLess(
            torch.norm(q_identity - expected_identity),
            1e-4,
            "Inverse should work correctly for normal quaternions",
        )

        # Test with very small quaternions (near-zero norm)
        q_small = torch.tensor([[1e-8, 1e-8, 1e-8, 1e-8]], dtype=torch.float64)
        q_inv_small = quaternion.inverse(q_small)

        # Should not produce NaN or Inf values
        self.assertFalse(
            torch.isnan(q_inv_small).any(), "Inverse should not produce NaN"
        )
        self.assertFalse(
            torch.isinf(q_inv_small).any(), "Inverse should not produce Inf"
        )

    def test_quaternion_cross_validation(self) -> None:
        """Cross-validate quaternion operations against rotation matrix operations."""
        torch.manual_seed(0)
        nMat = 10

        # Generate random quaternions and vectors
        q1 = generateRandomQuats(nMat)
        q2 = generateRandomQuats(nMat)
        v = torch.normal(mean=0, std=1, size=(nMat, 3), dtype=torch.float64)

        # Test multiplication consistency
        q_mult = quaternion.multiply(q1, q2)
        m1 = quaternion.to_rotation_matrix(q1)
        m2 = quaternion.to_rotation_matrix(q2)
        m_mult = torch.bmm(m1, m2)
        m_from_q_mult = quaternion.to_rotation_matrix(q_mult)

        self.assertLess(
            torch.norm(m_mult - m_from_q_mult),
            1e-4,
            "Quaternion multiplication should match matrix multiplication",
        )

        # Test vector rotation consistency
        v_rotated_q = quaternion.rotate_vector(q1, v)
        v_rotated_m = torch.bmm(m1, v.unsqueeze(-1)).squeeze(-1)

        self.assertLess(
            torch.norm(v_rotated_q - v_rotated_m),
            1e-4,
            "Quaternion vector rotation should match matrix rotation",
        )

    def test_assume_normalized_variants_consistency(self) -> None:
        """Test that assume_normalized functions produce consistent results."""
        torch.manual_seed(0)

        # Generate already normalized quaternions
        q1 = generateRandomQuats(10)
        q2 = generateRandomQuats(10)
        v = torch.normal(mean=0, std=1, size=(10, 3), dtype=torch.float64)

        # Test multiply variants
        q_mult_normal = quaternion.multiply(q1, q2)
        q_mult_assume = quaternion.multiply_assume_normalized(q1, q2)

        self.assertLess(
            torch.norm(q_mult_normal - q_mult_assume),
            1e-4,
            "multiply and multiply_assume_normalized should give same results for normalized inputs",
        )

        # Test rotate vector variants
        v_rot_normal = quaternion.rotate_vector(q1, v)
        v_rot_assume = quaternion.rotate_vector_assume_normalized(q1, v)

        self.assertLess(
            torch.norm(v_rot_normal - v_rot_assume),
            1e-4,
            "rotate_vector and rotate_vector_assume_normalized should give same results for normalized inputs",
        )

    def test_matrix_conversion_round_trip(self) -> None:
        """Test round-trip conversion between quaternions and matrices."""
        torch.manual_seed(0)

        # Test with various quaternion configurations
        test_cases = [
            generateRandomQuats(10),  # Random quaternions
            quaternion.identity().expand(5, 4),  # Identity quaternions
            quaternion.from_axis_angle(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),  # Basic rotations
        ]

        for quats in test_cases:
            # Forward: quaternion -> matrix -> quaternion
            matrices = quaternion.to_rotation_matrix(quats)
            quats_recovered = quaternion.from_rotation_matrix(matrices)
            matrices_recovered = quaternion.to_rotation_matrix(quats_recovered)

            # Check matrix consistency (matrices should match exactly)
            self.assertLess(
                torch.norm(matrices - matrices_recovered),
                1e-4,
                "Round-trip matrix conversion should be consistent",
            )

            # Check quaternion consistency (up to sign due to quaternion double cover)
            diff_pos = torch.norm(quats - quats_recovered, dim=-1)
            diff_neg = torch.norm(quats + quats_recovered, dim=-1)
            min_diff = torch.minimum(diff_pos, diff_neg)
            self.assertLess(
                torch.max(min_diff).item(),
                1e-4,
                "Round-trip quaternion conversion should be consistent up to sign",
            )

    def test_backprop_gradient_consistency(self) -> None:
        """Test that custom backprop functions match PyTorch autograd gradients."""
        torch.manual_seed(0)

        # Test with different batch sizes and ensure consistency
        for batch_size in [1, 10, 100]:
            q = torch.randn((batch_size, 4), dtype=torch.float64, requires_grad=True)
            v = torch.randn((batch_size, 3), dtype=torch.float64, requires_grad=True)

            # Test normalize backprop
            grad_out = torch.randn_like(q)
            autograd_grad = torch.autograd.grad(
                quaternion.normalize(q), q, grad_out, retain_graph=True
            )[0]
            custom_grad = quaternion.normalize_backprop(q, grad_out)

            self.assertLess(
                torch.norm(autograd_grad - custom_grad),
                1e-5,
                f"Normalize backprop should match autograd (batch_size={batch_size})",
            )

            # Test rotate_vector backprop
            grad_out_v = torch.randn((batch_size, 3), dtype=torch.float64)
            autograd_grad_q, autograd_grad_v = torch.autograd.grad(
                quaternion.rotate_vector(q, v), [q, v], grad_out_v, retain_graph=True
            )
            custom_grad_q, custom_grad_v = quaternion.rotate_vector_backprop(
                q, v, grad_out_v
            )

            self.assertLess(
                torch.norm(autograd_grad_q - custom_grad_q),
                1e-5,
                f"Rotate backprop quaternion grad should match autograd (batch_size={batch_size})",
            )
            self.assertLess(
                torch.norm(autograd_grad_v - custom_grad_v),
                1e-5,
                f"Rotate backprop vector grad should match autograd (batch_size={batch_size})",
            )


if __name__ == "__main__":
    unittest.main()
