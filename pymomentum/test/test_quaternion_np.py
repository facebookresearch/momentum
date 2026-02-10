# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import numpy as np
import pymomentum.quaternion_np as quaternion


def generateRandomQuats(sz: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return quaternion.normalize(rng.normal(loc=0, scale=4, size=(sz, 4)))


class TestQuaternionNumpy(unittest.TestCase):
    def test_euler_conversion(self) -> None:
        rng = np.random.default_rng(0)
        nBatch = 6
        nMat = 5
        euler1 = rng.normal(loc=0, scale=4, size=(nBatch, nMat, 3))

        quats1 = quaternion.euler_xyz_to_quaternion(euler1)
        euler2 = quaternion.quaternion_to_xyz_euler(quats1)
        quats2 = quaternion.euler_xyz_to_quaternion(euler2)

        quaternion.inverse(quats1)

        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(quats1)
                - quaternion.to_rotation_matrix(quats2)
            ),
            1e-4,
            "Expected rotation difference to be small.",
        )

        # check corner case conversion
        q_cornercase = np.array([0.4832, -0.5161, 0.4835, 0.5162], dtype=np.float32)
        euler_cornercase = quaternion.quaternion_to_xyz_euler(q_cornercase)
        self.assertFalse(np.isnan(euler_cornercase).any())

    def test_matrix_to_quaternion(self) -> None:
        nBatch = 6
        quats = generateRandomQuats(nBatch)
        mats = quaternion.to_rotation_matrix(quats)
        quats2 = quaternion.from_rotation_matrix(mats)
        mats2 = quaternion.to_rotation_matrix(quats2)
        diff = np.minimum(
            np.linalg.norm(quats - quats, axis=-1),
            np.linalg.norm(quats + quats2, axis=-1),
        )
        self.assertLess(
            np.linalg.norm(diff), 1e-4, "Expected quaternions to match (up to sign)"
        )
        self.assertLess(
            np.linalg.norm(mats2 - mats),
            1e-4,
            "Expected quaternions to match (up to sign)",
        )

    def test_multiply(self) -> None:
        nMat = 5
        q1 = generateRandomQuats(nMat)
        q2 = generateRandomQuats(nMat)
        q12 = quaternion.multiply(q1, q2)

        m1 = quaternion.to_rotation_matrix(q1)
        m2 = quaternion.to_rotation_matrix(q2)
        m12 = np.einsum("bij,bjk->bik", m1, m2)

        self.assertLess(
            np.linalg.norm(quaternion.to_rotation_matrix(q12) - m12),
            1e-4,
            "Expected rotation difference to be small.",
        )

    def test_identity(self) -> None:
        nMat = 5
        q1 = generateRandomQuats(nMat)
        ident = np.broadcast_to(quaternion.identity(dtype=q1.dtype), q1.shape).copy()

        self.assertLess(
            np.linalg.norm(quaternion.multiply(q1, ident) - q1),
            1e-4,
            "Identity on the right",
        )

        self.assertLess(
            np.linalg.norm(quaternion.multiply(ident, q1) - q1),
            1e-4,
            "Identity on the left",
        )

    def test_rotate_vector(self) -> None:
        rng = np.random.default_rng(0)
        nMat = 5
        q = generateRandomQuats(nMat)

        vec = rng.normal(loc=0, scale=4, size=(nMat, 3))

        rotated1 = quaternion.rotate_vector(q, vec)
        rotated2 = np.einsum("bij,bj->bi", quaternion.to_rotation_matrix(q), vec)

        self.assertLess(
            np.linalg.norm(rotated1 - rotated2),
            1e-4,
            "Matrix rotation should match quaternion rotation",
        )

    def test_blend_same(self) -> None:
        nMat = 5
        q = generateRandomQuats(nMat)
        q_dup = np.broadcast_to(q[:, None, :], (nMat, 5, 4)).copy()
        q_blend = quaternion.blend(q_dup)

        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(q_blend)
                - quaternion.to_rotation_matrix(q)
            ),
            1e-4,
            "quaternion blending of the same quaternion should be identity.",
        )

    def test_blend_euler(self) -> None:
        unit_x = np.array([[0.25 * math.pi, 0, 0]])

        # Blend of two single-axis rotations should be the midpoint:
        quats1 = quaternion.euler_xyz_to_quaternion(0.3 * math.pi * unit_x)
        quats2 = quaternion.euler_xyz_to_quaternion(0.5 * math.pi * unit_x)
        q_blend = quaternion.blend(np.concatenate([quats1, quats2], axis=0))
        euler_blend = quaternion.euler_xyz_to_quaternion(0.4 * math.pi * unit_x)

        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(q_blend)
                - quaternion.to_rotation_matrix(euler_blend)
            ),
            1e-4,
            "quaternion blending of single-axis Euler rotation should be the midway rotation.",
        )

        weights = np.array([0.75, 0.25])
        q_blend_weighted = quaternion.blend(
            np.concatenate([quats1, quats2], axis=0), weights
        )
        euler_blend_weighted = quaternion.euler_xyz_to_quaternion(
            (0.3 * weights[0] + 0.5 * weights[1]) * math.pi * unit_x
        )
        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(q_blend_weighted)
                - quaternion.to_rotation_matrix(euler_blend_weighted)
            ),
            1e-2,
            "quaternion blending of single-axis Euler rotation should be the midway rotation.",
        )

    def test_from_two_vectors(self) -> None:
        # Test with two random vectors
        rng = np.random.default_rng(0)
        n_batch = 5
        v1 = rng.normal(size=(n_batch, 3))
        v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        v2 = rng.normal(size=(n_batch, 3))
        v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(np.allclose(rotated_v1, v2, rtol=1e-4, atol=1e-6))

    def test_from_two_vectors_opposite(self) -> None:
        # Test with two opposite vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(np.allclose(rotated_v1, v2))

    def test_from_two_vectors_parallel(self) -> None:
        # Test with two parallel vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])

        q = quaternion.from_two_vectors(v1, v2)
        rotated_v1 = quaternion.rotate_vector(q, v1)

        self.assertTrue(np.allclose(rotated_v1, v1))

    def test_check_and_normalize_weights_squeeze(self) -> None:
        # Test case where weights.ndim == quaternions.ndim (triggers squeeze)
        rng = np.random.default_rng(0)
        quaternions = rng.normal(size=(2, 3, 4))
        weights = np.ones((2, 3, 1))  # Same ndim as quaternions

        # This should trigger the squeeze operation
        normalized_weights = quaternion.check_and_normalize_weights(
            quaternions, weights
        )

        # Verify the weights were properly normalized
        self.assertEqual(normalized_weights.shape, (2, 3))
        self.assertTrue(np.allclose(normalized_weights.sum(axis=-1), np.ones(2)))

    def test_check_and_normalize_weights_dim_mismatch(self) -> None:
        # Test case where weights.ndim + 1 != quaternions.ndim
        rng = np.random.default_rng(0)
        quaternions = rng.normal(size=(2, 3, 4))  # 3 dimensions
        weights = np.ones(2)  # 1 dimension, but 1 + 1 != 3

        with self.assertRaises(ValueError) as context:
            quaternion.check_and_normalize_weights(quaternions, weights)

        self.assertIn(
            "Expected weights vector to match quaternion vector",
            str(context.exception),
        )

    def test_check_and_normalize_weights_size_mismatch(self) -> None:
        # Test case where weights.shape[i] != quaternions.shape[i] for some dimension i
        rng = np.random.default_rng(0)
        quaternions = rng.normal(size=(2, 3, 4))  # Shape: (2, 3, 4)
        weights = np.ones((2, 5))  # Shape: (2, 5) - first dim matches, second doesn't

        with self.assertRaises(ValueError) as context:
            quaternion.check_and_normalize_weights(quaternions, weights)

        self.assertIn(
            "Expected weights vector to match quaternion vector",
            str(context.exception),
        )

    def test_slerp_extremes(self) -> None:
        rng = np.random.default_rng(0)
        nMat = 5
        q1 = quaternion.normalize(rng.normal(loc=0, scale=4, size=(nMat, 4)))
        q2 = quaternion.normalize(rng.normal(loc=0, scale=4, size=(nMat, 4)))
        # Add another edge case for very close quaternions
        q1 = np.concatenate(
            (q1, np.array([[-0.9662, -0.1052, -0.0235, 0.2343]])), axis=0
        )
        q2 = np.concatenate(
            (q2, np.array([[-0.9684, -0.1252, -0.0175, 0.2151]])), axis=0
        )

        # Test slerp at t=0 should return q1
        q_slerp_0 = quaternion.slerp(q1, q2, np.array([0.0]))
        self.assertTrue(
            np.allclose(
                quaternion.to_rotation_matrix(q_slerp_0),
                quaternion.to_rotation_matrix(q1),
                rtol=1e-3,
                atol=1e-4,
            )
        )

        # Test slerp at t=1 should return q2
        q_slerp_1 = quaternion.slerp(q1, q2, np.array([1.0]))
        self.assertTrue(
            np.allclose(
                quaternion.to_rotation_matrix(q_slerp_1),
                quaternion.to_rotation_matrix(q2),
                rtol=1e-3,
                atol=1e-4,
            )
        )

    def test_slerp_midpoint(self) -> None:
        q1 = quaternion.euler_xyz_to_quaternion(np.array([0.0, 0.0, 0.0]))
        q2 = quaternion.euler_xyz_to_quaternion(np.array([math.pi / 2, 0.0, 0.0]))

        # Test slerp at t=0.5 should be in the middle
        q_result = quaternion.slerp(q1, q2, np.array([0.5]))

        q_target = quaternion.euler_xyz_to_quaternion(np.array([math.pi / 4, 0.0, 0.0]))
        # Check if the slerp result is approximately in the middle
        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(q_result)
                - quaternion.to_rotation_matrix(q_target)
            ),
            1e-1,
            "Expected slerp result to be approximately in the middle.",
        )

        q3 = quaternion.euler_xyz_to_quaternion(np.array([math.pi, 0.0, 0.0]))
        q_result2 = quaternion.slerp(q1, q3, np.array([0.5]))
        q_target2a = quaternion.euler_xyz_to_quaternion(
            np.array([math.pi / 2, 0.0, 0.0])
        )
        q_target2b = quaternion.euler_xyz_to_quaternion(
            np.array([-math.pi / 2, 0.0, 0.0])
        )
        # Could go either way around the circle, so check both
        diff2a = np.linalg.norm(
            quaternion.to_rotation_matrix(q_result2)
            - quaternion.to_rotation_matrix(q_target2a)
        )
        diff2b = np.linalg.norm(
            quaternion.to_rotation_matrix(q_result2)
            - quaternion.to_rotation_matrix(q_target2b)
        )
        self.assertTrue(diff2a < 1e-5 or diff2b < 1e-5)

    def test_slerp_close_matrices(self) -> None:
        nMat = 5
        q1 = generateRandomQuats(nMat)

        q_result = quaternion.slerp(q1, q1, np.array([0.5]))

        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(q_result)
                - quaternion.to_rotation_matrix(q1)
            ),
            1e-5,
        )

        q_result = quaternion.slerp(q1, -q1, np.array([0.5]))

        self.assertLess(
            np.linalg.norm(
                quaternion.to_rotation_matrix(q_result)
                - quaternion.to_rotation_matrix(q1)
            ),
            1e-5,
        )

    def test_from_rotation_matrix_numerical_stability(self) -> None:
        """Test the enhanced matrix-to-quaternion conversion with numerical stability."""
        # Test with random rotation matrices
        nBatch = 10
        quats_orig = generateRandomQuats(nBatch)
        matrices = quaternion.to_rotation_matrix(quats_orig)

        # Convert back using the enhanced method
        quats_recovered = quaternion.from_rotation_matrix(matrices)
        matrices_recovered = quaternion.to_rotation_matrix(quats_recovered)

        # Check that matrices match (accounting for quaternion sign ambiguity)
        self.assertLess(
            np.linalg.norm(matrices - matrices_recovered),
            1e-4,
            "Enhanced matrix-to-quaternion conversion should be stable",
        )

        # Test with identity matrices (edge case)
        identity_matrices = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()
        identity_quats = quaternion.from_rotation_matrix(identity_matrices)
        expected_identity = np.broadcast_to(quaternion.identity(), (5, 4))

        # Check that identity matrices produce identity quaternions (up to sign)
        diff1 = np.linalg.norm(identity_quats - expected_identity, axis=-1)
        diff2 = np.linalg.norm(identity_quats + expected_identity, axis=-1)
        min_diff = np.minimum(diff1, diff2)
        self.assertLess(np.max(min_diff).item(), 1e-4)

    def test_quaternion_inverse_numerical_stability(self) -> None:
        """Test that the improved inverse function handles near-zero quaternions."""
        # Test with normal quaternions
        q_normal = generateRandomQuats(5)
        q_inv = quaternion.inverse(q_normal)
        q_identity = quaternion.multiply(q_normal, q_inv)
        expected_identity = np.broadcast_to(
            quaternion.identity(dtype=q_identity.dtype), q_identity.shape
        )

        self.assertLess(
            np.linalg.norm(q_identity - expected_identity),
            1e-4,
            "Inverse should work correctly for normal quaternions",
        )

        # Test with very small quaternions (near-zero norm)
        q_small = np.array([[1e-8, 1e-8, 1e-8, 1e-8]], dtype=np.float64)
        q_inv_small = quaternion.inverse(q_small)

        # Should not produce NaN or Inf values
        self.assertFalse(np.isnan(q_inv_small).any(), "Inverse should not produce NaN")
        self.assertFalse(np.isinf(q_inv_small).any(), "Inverse should not produce Inf")

    def test_quaternion_cross_validation(self) -> None:
        """Cross-validate quaternion operations against rotation matrix operations."""
        rng = np.random.default_rng(0)
        nMat = 10

        # Generate random quaternions and vectors
        q1 = generateRandomQuats(nMat)
        q2 = generateRandomQuats(nMat)
        v = rng.normal(loc=0, scale=1, size=(nMat, 3))

        # Test multiplication consistency
        q_mult = quaternion.multiply(q1, q2)
        m1 = quaternion.to_rotation_matrix(q1)
        m2 = quaternion.to_rotation_matrix(q2)
        m_mult = np.einsum("bij,bjk->bik", m1, m2)
        m_from_q_mult = quaternion.to_rotation_matrix(q_mult)

        self.assertLess(
            np.linalg.norm(m_mult - m_from_q_mult),
            1e-4,
            "Quaternion multiplication should match matrix multiplication",
        )

        # Test vector rotation consistency
        v_rotated_q = quaternion.rotate_vector(q1, v)
        v_rotated_m = np.einsum("bij,bj->bi", m1, v)

        self.assertLess(
            np.linalg.norm(v_rotated_q - v_rotated_m),
            1e-4,
            "Quaternion vector rotation should match matrix rotation",
        )

    def test_assume_normalized_variants_consistency(self) -> None:
        """Test that assume_normalized functions produce consistent results."""
        rng = np.random.default_rng(0)
        # Generate already normalized quaternions
        q1 = generateRandomQuats(10)
        q2 = generateRandomQuats(10)
        v = rng.normal(loc=0, scale=1, size=(10, 3))

        # Test multiply variants
        q_mult_normal = quaternion.multiply(q1, q2)
        q_mult_assume = quaternion.multiply_assume_normalized(q1, q2)

        self.assertLess(
            np.linalg.norm(q_mult_normal - q_mult_assume),
            1e-4,
            "multiply and multiply_assume_normalized should give same results for normalized inputs",
        )

        # Test rotate vector variants
        v_rot_normal = quaternion.rotate_vector(q1, v)
        v_rot_assume = quaternion.rotate_vector_assume_normalized(q1, v)

        self.assertLess(
            np.linalg.norm(v_rot_normal - v_rot_assume),
            1e-4,
            "rotate_vector and rotate_vector_assume_normalized should give same results for normalized inputs",
        )

    def test_matrix_conversion_round_trip(self) -> None:
        """Test round-trip conversion between quaternions and matrices."""
        # Test with various quaternion configurations
        test_cases = [
            generateRandomQuats(10),  # Random quaternions
            np.broadcast_to(
                quaternion.identity(), (5, 4)
            ).copy(),  # Identity quaternions
            quaternion.from_axis_angle(
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),  # Basic rotations
        ]

        for quats in test_cases:
            # Forward: quaternion -> matrix -> quaternion
            matrices = quaternion.to_rotation_matrix(quats)
            quats_recovered = quaternion.from_rotation_matrix(matrices)
            matrices_recovered = quaternion.to_rotation_matrix(quats_recovered)

            # Check matrix consistency (matrices should match exactly)
            self.assertLess(
                np.linalg.norm(matrices - matrices_recovered),
                1e-4,
                "Round-trip matrix conversion should be consistent",
            )

            # Check quaternion consistency (up to sign due to quaternion double cover)
            diff_pos = np.linalg.norm(quats - quats_recovered, axis=-1)
            diff_neg = np.linalg.norm(quats + quats_recovered, axis=-1)
            min_diff = np.minimum(diff_pos, diff_neg)
            self.assertLess(
                np.max(min_diff).item(),
                1e-4,
                "Round-trip quaternion conversion should be consistent up to sign",
            )

    def test_euler_zyx_to_quaternion(self) -> None:
        """Test the ZYX Euler angle to quaternion conversion function."""
        rng = np.random.default_rng(0)
        nBatch = 6
        nMat = 5
        euler_zyx = rng.normal(loc=0, scale=2, size=(nBatch, nMat, 3))

        # Test that ZYX conversion produces valid quaternions
        quats_zyx = quaternion.euler_zyx_to_quaternion(euler_zyx)

        # Check that they're normalized
        norms = np.linalg.norm(quats_zyx, axis=-1)
        self.assertTrue(np.allclose(norms, np.ones_like(norms), atol=1e-4))

        # Check that rotation matrices are orthogonal
        matrices_zyx = quaternion.to_rotation_matrix(quats_zyx)
        matrices_transpose = np.swapaxes(matrices_zyx, -1, -2)
        identity = np.broadcast_to(np.eye(3), matrices_zyx.shape)
        product = np.einsum("...ij,...jk->...ik", matrices_zyx, matrices_transpose)

        self.assertLess(
            np.linalg.norm(product - identity),
            1e-4,
            "ZYX quaternion should produce orthogonal rotation matrix",
        )


if __name__ == "__main__":
    unittest.main()
