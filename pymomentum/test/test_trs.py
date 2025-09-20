# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.trs as pym_trs
import torch


class TestTRS(unittest.TestCase):
    def test_from_translation(self) -> None:
        """Test creating TRS from translation vector."""
        translation = torch.tensor([1.0, 2.0, 3.0])
        t, r, s = pym_trs.from_translation(translation)

        # Check translation matches
        self.assertTrue(torch.allclose(t, translation))

        # Check rotation is identity matrix
        expected_rotation = torch.eye(3)
        self.assertTrue(torch.allclose(r, expected_rotation))

        # Check scale is 1
        expected_scale = torch.tensor([1.0])
        self.assertTrue(torch.allclose(s, expected_scale))

    def test_from_rotation_matrix(self) -> None:
        """Test creating TRS from rotation matrix."""
        # Create a simple rotation matrix (90 degrees around Z axis)
        rotation_matrix = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        t, r, s = pym_trs.from_rotation_matrix(rotation_matrix)

        # Check rotation matches
        self.assertTrue(torch.allclose(r, rotation_matrix))

        # Check translation is zero
        expected_translation = torch.zeros(3)
        self.assertTrue(torch.allclose(t, expected_translation))

        # Check scale is 1
        expected_scale = torch.tensor([1.0])
        self.assertTrue(torch.allclose(s, expected_scale))

    def test_from_scale(self) -> None:
        """Test creating TRS from scale factor."""
        scale = torch.tensor([2.5])
        t, r, s = pym_trs.from_scale(scale)

        # Check scale matches
        self.assertTrue(torch.allclose(s, scale))

        # Check translation is zero
        expected_translation = torch.zeros(3)
        self.assertTrue(torch.allclose(t, expected_translation))

        # Check rotation is identity matrix
        expected_rotation = torch.eye(3)
        self.assertTrue(torch.allclose(r, expected_rotation))

    def test_identity_no_batch(self) -> None:
        """Test identity transform with no batch dimensions."""
        t, r, s = pym_trs.identity()

        # Check shapes
        self.assertEqual(t.shape, (3,))
        self.assertEqual(r.shape, (3, 3))
        self.assertEqual(s.shape, (1,))

        # Check values
        self.assertTrue(torch.allclose(t, torch.zeros(3)))
        self.assertTrue(torch.allclose(r, torch.eye(3)))
        self.assertTrue(torch.allclose(s, torch.ones(1)))

    def test_identity_with_batch(self) -> None:
        """Test identity transform with batch dimensions."""
        batch_size = [2, 3]
        t, r, s = pym_trs.identity(size=batch_size)

        # Check shapes
        self.assertEqual(t.shape, (2, 3, 3))
        self.assertEqual(r.shape, (2, 3, 3, 3))
        self.assertEqual(s.shape, (2, 3, 1))

        # Check values - all should be identity transforms
        expected_t = torch.zeros(2, 3, 3)
        expected_r = torch.eye(3).expand(2, 3, 3, 3)
        expected_s = torch.ones(2, 3, 1)

        self.assertTrue(torch.allclose(t, expected_t))
        self.assertTrue(torch.allclose(r, expected_r))
        self.assertTrue(torch.allclose(s, expected_s))

    def test_multiply(self) -> None:
        """Test multiplying two TRS transforms."""
        # Create first transform: translation only
        t1 = torch.tensor([1.0, 0.0, 0.0])
        trs1 = pym_trs.from_translation(t1)

        # Create second transform: scale only
        s2 = torch.tensor([2.0])
        trs2 = pym_trs.from_scale(s2)

        # Multiply transforms
        t_result, r_result, s_result = pym_trs.multiply(trs1, trs2)

        # Check that rotation is still identity
        self.assertTrue(torch.allclose(r_result, torch.eye(3)))

        # Check that scale is 2.0
        self.assertTrue(torch.allclose(s_result, torch.tensor([2.0])))

        # For TRS1 * TRS2, the translation should be t1 + R1 * (s1 * t2)
        # t1 = [1,0,0], R1 = I, s1 = 1, t2 = [0,0,0]
        # So result should be [1,0,0] + I * (1 * [0,0,0]) = [1,0,0]
        expected_translation = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(t_result, expected_translation))

    def test_inverse(self) -> None:
        """Test inverting a TRS transform."""
        # Create a transform with translation and scale
        t = torch.tensor([2.0, 4.0, 6.0])
        s = torch.tensor([2.0])
        trs_original = pym_trs.multiply(
            pym_trs.from_translation(t), pym_trs.from_scale(s)
        )

        # Compute inverse
        trs_inv = pym_trs.inverse(trs_original)

        # Multiply original by inverse should give identity
        identity_result = pym_trs.multiply(trs_original, trs_inv)
        t_id, r_id, s_id = identity_result

        # Check that result is close to identity
        self.assertTrue(torch.allclose(t_id, torch.zeros(3), atol=1e-6))
        self.assertTrue(torch.allclose(r_id, torch.eye(3), atol=1e-6))
        self.assertTrue(torch.allclose(s_id, torch.ones(1), atol=1e-6))

    def test_transform_points(self) -> None:
        """Test transforming points with TRS transform."""
        # Create a simple translation transform
        translation = torch.tensor([1.0, 2.0, 3.0])
        trs_trans = pym_trs.from_translation(translation)

        # Transform a point
        points = torch.tensor([[0.0, 0.0, 0.0]])
        transformed = pym_trs.transform_points(trs_trans, points)

        # Should be translated by [1, 2, 3]
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.allclose(transformed, expected))

    def test_transform_points_invalid_shape(self) -> None:
        """Test that transform_points raises error for invalid point shapes."""
        trs_identity = pym_trs.identity()

        # Test points with wrong last dimension
        invalid_points = torch.randn(5, 2)  # Should be (..., 3)

        with self.assertRaises(ValueError) as context:
            pym_trs.transform_points(trs_identity, invalid_points)

        self.assertIn(
            "Points tensor should have last dimension 3", str(context.exception)
        )

    def test_to_matrix(self) -> None:
        """Test converting TRS transform to 4x4 matrix."""
        # Create a TRS with translation and scale
        translation = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor([2.0])
        trs_transform = pym_trs.multiply(
            pym_trs.from_translation(translation), pym_trs.from_scale(scale)
        )

        # Convert to matrix
        matrix = pym_trs.to_matrix(trs_transform)

        # Check shape
        self.assertEqual(matrix.shape, (4, 4))

        # Check that it matches expected structure
        # Top-left 3x3 should be scaled identity matrix
        expected_linear = torch.eye(3) * 2.0
        self.assertTrue(torch.allclose(matrix[:3, :3], expected_linear))

        # Right column should be translation
        self.assertTrue(torch.allclose(matrix[:3, 3], translation))

        # Bottom row should be [0, 0, 0, 1]
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(matrix[3, :], expected_bottom))

    def test_from_matrix(self) -> None:
        """Test converting 4x4 matrix to TRS transform."""
        # Create a 4x4 matrix manually
        matrix = torch.tensor(
            [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 2.0],
                [0.0, 0.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Convert to TRS
        t, r, s = pym_trs.from_matrix(matrix)

        # Check translation
        expected_translation = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(t, expected_translation))

        # Check rotation (should be identity)
        expected_rotation = torch.eye(3)
        self.assertTrue(torch.allclose(r, expected_rotation, atol=1e-6))

        # Check scale
        expected_scale = torch.tensor([2.0])
        self.assertTrue(torch.allclose(s, expected_scale, atol=1e-6))

    def test_matrix_roundtrip(self) -> None:
        """Test that to_matrix and from_matrix are inverse operations."""
        # Create a random TRS transform
        translation = torch.tensor([1.5, -2.3, 0.7])
        scale = torch.tensor([1.8])
        trs_original = pym_trs.multiply(
            pym_trs.from_translation(translation), pym_trs.from_scale(scale)
        )

        # Convert to matrix and back
        matrix = pym_trs.to_matrix(trs_original)
        trs_reconstructed = pym_trs.from_matrix(matrix)

        # Convert both to matrices for comparison (since TRS may not be identical due to rotation extraction)
        matrix_original = pym_trs.to_matrix(trs_original)
        matrix_reconstructed = pym_trs.to_matrix(trs_reconstructed)

        # Matrices should be very close
        self.assertTrue(
            torch.allclose(matrix_original, matrix_reconstructed, atol=1e-6)
        )

    def test_from_matrix_invalid_shape(self) -> None:
        """Test that from_matrix raises error for invalid matrix shapes."""
        # Test matrix with wrong dimensions
        invalid_matrix = torch.randn(3, 3)  # Should be 4x4

        with self.assertRaises(ValueError) as context:
            pym_trs.from_matrix(invalid_matrix)

        self.assertIn("Expected a tensor of 4x4 matrices", str(context.exception))

    def test_from_skeleton_state(self) -> None:
        """Test converting skeleton state to TRS transform."""
        # Create a skeleton state: [tx, ty, tz, rx, ry, rz, rw, s]
        # Using identity quaternion (0, 0, 0, 1) and scale 2.0
        skeleton_state = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0])

        # Convert to TRS
        t, r, s = pym_trs.from_skeleton_state(skeleton_state)

        # Check translation
        expected_translation = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(t, expected_translation))

        # Check rotation (should be identity for identity quaternion)
        expected_rotation = torch.eye(3)
        self.assertTrue(torch.allclose(r, expected_rotation, atol=1e-6))

        # Check scale
        expected_scale = torch.tensor([2.0])
        self.assertTrue(torch.allclose(s, expected_scale))

    def test_to_skeleton_state(self) -> None:
        """Test converting TRS transform to skeleton state."""
        # Create a TRS transform
        translation = torch.tensor([1.5, -2.3, 0.7])
        scale = torch.tensor([1.8])
        trs_transform = pym_trs.multiply(
            pym_trs.from_translation(translation), pym_trs.from_scale(scale)
        )

        # Convert to skeleton state
        skeleton_state = pym_trs.to_skeleton_state(trs_transform)

        # Check shape
        self.assertEqual(skeleton_state.shape, (8,))

        # Check translation part
        self.assertTrue(torch.allclose(skeleton_state[:3], translation))

        # Check scale part
        self.assertTrue(torch.allclose(skeleton_state[7:], scale))

        # Check that quaternion part is identity (since rotation was identity)
        identity_quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(skeleton_state[3:7], identity_quaternion, atol=1e-6)
        )

    def test_skeleton_state_roundtrip(self) -> None:
        """Test that skeleton state conversion is round-trip consistent."""
        # Create a skeleton state with non-trivial quaternion
        import math

        # Quaternion for 45 degree rotation around Z axis
        angle = math.pi / 4
        z_quat = torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
        skeleton_state = torch.tensor(
            [1.0, 2.0, 3.0, z_quat[0], z_quat[1], z_quat[2], z_quat[3], 1.5]
        )

        # Convert to TRS and back
        trs_transform = pym_trs.from_skeleton_state(skeleton_state)
        skeleton_state_reconstructed = pym_trs.to_skeleton_state(trs_transform)

        # Should be very close (allowing for quaternion sign ambiguity)
        diff = torch.abs(skeleton_state - skeleton_state_reconstructed)
        # Alternative quaternion (negated) should also be close
        alt_quat = skeleton_state_reconstructed.clone()
        alt_quat[3:7] *= -1
        diff_alt = torch.abs(skeleton_state - alt_quat)

        # Either the original or the negated quaternion should be close
        is_close = torch.all(diff < 1e-5) or torch.all(diff_alt < 1e-5)
        self.assertTrue(
            is_close, f"Roundtrip failed. Diff: {diff}, Alt diff: {diff_alt}"
        )

    def test_from_skeleton_state_invalid_shape(self) -> None:
        """Test that from_skeleton_state raises error for invalid shapes."""
        # Test skeleton state with wrong dimensions
        invalid_skeleton_state = torch.randn(7)  # Should be 8

        with self.assertRaises(ValueError) as context:
            pym_trs.from_skeleton_state(invalid_skeleton_state)

        self.assertIn(
            "Expected skeleton state to have last dimension 8", str(context.exception)
        )

    def test_slerp_extremes(self) -> None:
        """Test slerp at t=0 and t=1."""
        # Create two different TRS transforms
        trs0 = pym_trs.from_translation(torch.tensor([1.0, 0.0, 0.0]))
        trs1 = pym_trs.from_translation(torch.tensor([0.0, 1.0, 0.0]))

        # Test slerp at t=0 should return trs0
        t_interp_0, r_interp_0, s_interp_0 = pym_trs.slerp(
            trs0, trs1, torch.tensor([0.0])
        )
        expected_t0 = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(t_interp_0, expected_t0, atol=1e-6))

        # Test slerp at t=1 should return trs1
        t_interp_1, r_interp_1, s_interp_1 = pym_trs.slerp(
            trs0, trs1, torch.tensor([1.0])
        )
        expected_t1 = torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(t_interp_1, expected_t1, atol=1e-6))

    def test_slerp_midpoint(self) -> None:
        """Test slerp at t=0.5."""
        # Create two TRS transforms with different translation and scale
        trs0 = pym_trs.multiply(
            pym_trs.from_translation(torch.tensor([0.0, 0.0, 0.0])),
            pym_trs.from_scale(torch.tensor([1.0])),
        )
        trs1 = pym_trs.multiply(
            pym_trs.from_translation(torch.tensor([2.0, 4.0, 6.0])),
            pym_trs.from_scale(torch.tensor([3.0])),
        )

        # Interpolate at midpoint
        t_interp, r_interp, s_interp = pym_trs.slerp(trs0, trs1, torch.tensor([0.5]))

        # Translation should be midpoint
        expected_t = torch.tensor([1.0, 2.0, 3.0])  # (0+2)/2, (0+4)/2, (0+6)/2
        self.assertTrue(torch.allclose(t_interp, expected_t, atol=1e-6))

        # Scale should be midpoint
        expected_s = torch.tensor([2.0])  # (1+3)/2
        self.assertTrue(torch.allclose(s_interp, expected_s, atol=1e-6))

        # Rotation should still be identity (since both inputs have identity rotation)
        self.assertTrue(torch.allclose(r_interp, torch.eye(3), atol=1e-6))

    def test_blend_equal_weights(self) -> None:
        """Test blending with equal weights."""
        # Create three TRS transforms
        trs_list = [
            pym_trs.from_translation(torch.tensor([1.0, 0.0, 0.0])),
            pym_trs.from_translation(torch.tensor([0.0, 1.0, 0.0])),
            pym_trs.from_translation(torch.tensor([0.0, 0.0, 1.0])),
        ]

        # Blend with equal weights (should use default equal weights)
        t_blend, r_blend, s_blend = pym_trs.blend(trs_list)

        # Translation should be average
        expected_t = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self.assertTrue(torch.allclose(t_blend, expected_t, atol=1e-6))

        # Rotation should be identity (all inputs have identity)
        self.assertTrue(torch.allclose(r_blend, torch.eye(3), atol=1e-6))

        # Scale should be 1 (all inputs have scale 1)
        expected_s = torch.tensor([1.0])
        self.assertTrue(torch.allclose(s_blend, expected_s, atol=1e-6))

    def test_blend_custom_weights(self) -> None:
        """Test blending with custom weights."""
        # Create two TRS transforms
        trs_list = [
            pym_trs.from_translation(torch.tensor([0.0, 0.0, 0.0])),
            pym_trs.from_translation(torch.tensor([4.0, 0.0, 0.0])),
        ]

        # Blend with custom weights [0.25, 0.75]
        weights = torch.tensor([0.25, 0.75])
        t_blend, r_blend, s_blend = pym_trs.blend(trs_list, weights)

        # Translation should be weighted average: 0.25*[0,0,0] + 0.75*[4,0,0] = [3,0,0]
        expected_t = torch.tensor([3.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(t_blend, expected_t, atol=1e-6))

    def test_blend_single_transform(self) -> None:
        """Test that blending a single transform returns the transform itself."""
        trs_single = pym_trs.from_translation(torch.tensor([1.0, 2.0, 3.0]))

        result = pym_trs.blend([trs_single])

        # Should return the same transform
        self.assertTrue(torch.allclose(result[0], trs_single[0]))
        self.assertTrue(torch.allclose(result[1], trs_single[1]))
        self.assertTrue(torch.allclose(result[2], trs_single[2]))

    def test_blend_empty_list(self) -> None:
        """Test that blending empty list raises error."""
        with self.assertRaises(ValueError) as context:
            pym_trs.blend([])

        self.assertIn("Cannot blend empty list of transforms", str(context.exception))


if __name__ == "__main__":
    unittest.main()
