# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import pymomentum.trs as pym_trs
import torch


class TestTRS(unittest.TestCase):
    def _rotation_x(self, angle: float) -> torch.Tensor:
        """Create rotation matrix around X-axis."""
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def _rotation_y(self, angle: float) -> torch.Tensor:
        """Create rotation matrix around Y-axis."""
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def _rotation_z(self, angle: float) -> torch.Tensor:
        """Create rotation matrix around Z-axis."""
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _quat_from_axis_angle(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        """Create quaternion from axis-angle representation using pymomentum.quaternion."""
        # Normalize the axis and scale by the angle to create axis-angle vector
        axis_normalized = axis / torch.norm(axis)
        axis_angle = axis_normalized * angle
        return pym_quaternion.from_axis_angle(axis_angle)

    def _random_skel_states(
        self,
        sz: int,
    ) -> torch.Tensor:
        # [sz, 3
        trans: torch.Tensor = torch.normal(
            mean=0,
            std=4,
            size=(sz, 3),
            dtype=torch.float64,
            requires_grad=True,
        )

        rot: torch.Tensor = pym_quaternion.normalize(
            torch.normal(
                mean=0,
                std=4,
                size=(sz, 4),
                dtype=torch.float64,
                requires_grad=True,
            )
        )

        scale: torch.Tensor = torch.rand(
            size=(sz, 1), dtype=torch.float64, requires_grad=True
        )
        return torch.cat([trans, rot, scale], -1)

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
        # Create first transform: 90-degree rotation around Z axis and translation
        rotation_z_90 = self._rotation_z(math.pi / 2)
        t1 = torch.tensor([1.0, 0.0, 0.0])
        trs1 = pym_trs.multiply(
            pym_trs.from_rotation_matrix(rotation_z_90), pym_trs.from_translation(t1)
        )

        # Create second transform: 45-degree rotation around X axis and scale
        rotation_x_45 = self._rotation_x(math.pi / 4)
        s2 = torch.tensor([2.0])
        trs2 = pym_trs.multiply(
            pym_trs.from_rotation_matrix(rotation_x_45), pym_trs.from_scale(s2)
        )

        # Multiply transforms
        _, r_result, s_result = pym_trs.multiply(trs1, trs2)

        # Check that scale is 2.0
        self.assertTrue(torch.allclose(s_result, torch.tensor([2.0])))

        # Check that rotation is composition of both rotations
        expected_rotation = torch.matmul(rotation_z_90, rotation_x_45)
        self.assertTrue(torch.allclose(r_result, expected_rotation, atol=1e-6))

    def test_inverse(self) -> None:
        """Test inverting a TRS transform."""
        # Create a transform with rotation around Y axis, translation, and scale
        rotation_y_60 = self._rotation_y(math.pi / 3)  # 60 degrees
        t = torch.tensor([2.0, 4.0, 6.0])
        s = torch.tensor([2.0])

        trs_original = pym_trs.multiply(
            pym_trs.from_rotation_matrix(rotation_y_60),
            pym_trs.multiply(pym_trs.from_translation(t), pym_trs.from_scale(s)),
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
        # Create a transform with 90-degree rotation around Z-axis, translation, and scale
        # Using direct construction instead of multiplication
        rotation_z_90 = self._rotation_z(math.pi / 2)
        translation = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor([2.0])

        # Create TRS directly with all components
        trs_transform = (translation, rotation_z_90, scale)

        # Transform a point at [1, 0, 0]
        points = torch.tensor([[1.0, 0.0, 0.0]])
        transformed = pym_trs.transform_points(trs_transform, points)

        # Order is Scale->Rotate->Translate:
        # Point [1,0,0] scaled by 2: [2,0,0]
        # [2,0,0] rotated 90° around Z: [0,2,0]
        # [0,2,0] translated by [1,2,3]: [1,4,3]
        expected = torch.tensor([[1.0, 4.0, 3.0]])
        self.assertTrue(torch.allclose(transformed, expected, atol=1e-6))

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
        # Create a TRS directly with 45-degree rotation around X, translation, and scale
        rotation_x_45 = self._rotation_x(math.pi / 4)
        translation = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor([2.0])

        # Create TRS directly with all components
        trs_transform = (translation, rotation_x_45, scale)

        # Convert to matrix
        matrix = pym_trs.to_matrix(trs_transform)

        # Check shape
        self.assertEqual(matrix.shape, (4, 4))

        # Check that it matches expected structure
        # Top-left 3x3 should be scaled rotation matrix
        expected_linear = rotation_x_45 * 2.0
        self.assertTrue(torch.allclose(matrix[:3, :3], expected_linear, atol=1e-6))

        # Right column should be translation
        self.assertTrue(torch.allclose(matrix[:3, 3], translation))

        # Bottom row should be [0, 0, 0, 1]
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(matrix[3, :], expected_bottom))

    def test_from_matrix(self) -> None:
        """Test converting 4x4 matrix to TRS transform."""
        # Create a 4x4 matrix with a 90-degree rotation around Z axis, scale, and translation
        rotation_z_90 = self._rotation_z(math.pi / 2)
        scale = 2.0
        translation = torch.tensor([1.0, 2.0, 3.0])

        # Build matrix manually: R*S for top-left 3x3, translation in top-right column
        matrix = torch.zeros(4, 4)
        matrix[:3, :3] = rotation_z_90 * scale
        matrix[:3, 3] = translation
        matrix[3, 3] = 1.0

        # Convert to TRS
        t, r, s = pym_trs.from_matrix(matrix)

        # Check translation
        self.assertTrue(torch.allclose(t, translation, atol=1e-6))

        # Check rotation (should be 90-degree rotation around Z)
        self.assertTrue(torch.allclose(r, rotation_z_90, atol=1e-6))

        # Check scale
        expected_scale = torch.tensor([scale])
        self.assertTrue(torch.allclose(s, expected_scale, atol=1e-6))

    def test_matrix_roundtrip(self) -> None:
        """Test that to_matrix and from_matrix are inverse operations."""
        # Create a TRS transform with 30-degree rotation around arbitrary axis, translation, and scale
        axis = torch.tensor([1.0, 2.0, 3.0])
        quat = self._quat_from_axis_angle(axis, math.pi / 6)  # 30 degrees
        translation = torch.tensor([1.5, -2.3, 0.7])
        scale = torch.tensor([1.8])

        # Create transform using the skeleton state approach since we have quaternion
        skeleton_state = torch.cat([translation, quat, scale])
        trs_original = pym_trs.from_skeleton_state(skeleton_state)

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
        t_interp_0, _, _ = pym_trs.slerp(trs0, trs1, torch.tensor([0.0]))
        expected_t0 = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(t_interp_0, expected_t0, atol=1e-6))

        # Test slerp at t=1 should return trs1
        t_interp_1, _, _ = pym_trs.slerp(trs0, trs1, torch.tensor([1.0]))
        expected_t1 = torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(t_interp_1, expected_t1, atol=1e-6))

    def test_slerp_midpoint(self) -> None:
        """Test slerp at t=0.5."""
        # Create two TRS transforms with different rotations, translation, and scale
        quat0 = self._quat_from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0]), 0.0
        )  # Identity
        quat1 = self._quat_from_axis_angle(
            torch.tensor([1.0, 0.0, 0.0]), math.pi / 2
        )  # 90° around X

        state0 = torch.cat([torch.tensor([0.0, 0.0, 0.0]), quat0, torch.tensor([1.0])])
        state1 = torch.cat([torch.tensor([2.0, 4.0, 6.0]), quat1, torch.tensor([3.0])])

        trs0 = pym_trs.from_skeleton_state(state0)
        trs1 = pym_trs.from_skeleton_state(state1)

        # Interpolate at midpoint
        t_interp, r_interp, s_interp = pym_trs.slerp(trs0, trs1, torch.tensor([0.5]))

        # Translation should be midpoint
        expected_t = torch.tensor([1.0, 2.0, 3.0])  # (0+2)/2, (0+4)/2, (0+6)/2
        self.assertTrue(torch.allclose(t_interp, expected_t, atol=1e-6))

        # Scale should be midpoint
        expected_s = torch.tensor([2.0])  # (1+3)/2
        self.assertTrue(torch.allclose(s_interp, expected_s, atol=1e-6))

        # Rotation should be interpolated between identity and 90° X rotation
        self.assertFalse(torch.allclose(r_interp, torch.eye(3), atol=1e-2))

    def test_blend_equal_weights(self) -> None:
        """Test blending with equal weights."""
        # Create three TRS transforms with different rotations
        quat1 = self._quat_from_axis_angle(
            torch.tensor([1.0, 0.0, 0.0]), math.pi / 6
        )  # 30° around X
        quat2 = self._quat_from_axis_angle(
            torch.tensor([0.0, 1.0, 0.0]), math.pi / 4
        )  # 45° around Y
        quat3 = self._quat_from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0]), math.pi / 3
        )  # 60° around Z

        state1 = torch.cat([torch.tensor([1.0, 0.0, 0.0]), quat1, torch.tensor([1.0])])
        state2 = torch.cat([torch.tensor([0.0, 1.0, 0.0]), quat2, torch.tensor([1.0])])
        state3 = torch.cat([torch.tensor([0.0, 0.0, 1.0]), quat3, torch.tensor([1.0])])

        trs_list = [
            pym_trs.from_skeleton_state(state1),
            pym_trs.from_skeleton_state(state2),
            pym_trs.from_skeleton_state(state3),
        ]

        # Blend with equal weights (should use default equal weights)
        t_blend, r_blend, s_blend = pym_trs.blend(trs_list)

        # Translation should be average
        expected_t = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self.assertTrue(torch.allclose(t_blend, expected_t, atol=1e-6))

        # Rotation should NOT be identity since we're blending different rotations
        self.assertFalse(torch.allclose(r_blend, torch.eye(3), atol=1e-2))

        # Scale should be 1 (all inputs have scale 1)
        expected_s = torch.tensor([1.0])
        self.assertTrue(torch.allclose(s_blend, expected_s, atol=1e-6))

    def test_blend_custom_weights(self) -> None:
        """Test blending with custom weights."""
        # Create two TRS transforms with different rotations
        quat1 = self._quat_from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0]), math.pi / 2
        )  # 90° around Z
        quat2 = self._quat_from_axis_angle(
            torch.tensor([1.0, 0.0, 0.0]), math.pi / 3
        )  # 60° around X

        state1 = torch.cat([torch.tensor([0.0, 0.0, 0.0]), quat1, torch.tensor([1.0])])
        state2 = torch.cat([torch.tensor([4.0, 0.0, 0.0]), quat2, torch.tensor([1.0])])

        trs_list = [
            pym_trs.from_skeleton_state(state1),
            pym_trs.from_skeleton_state(state2),
        ]

        # Blend with custom weights [0.25, 0.75]
        weights = torch.tensor([0.25, 0.75])
        t_blend, r_blend, _ = pym_trs.blend(trs_list, weights)

        # Translation should be weighted average: 0.25*[0,0,0] + 0.75*[4,0,0] = [3,0,0]
        expected_t = torch.tensor([3.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(t_blend, expected_t, atol=1e-6))

        # Rotation should be a weighted blend of the two different rotations
        self.assertFalse(torch.allclose(r_blend, torch.eye(3), atol=1e-2))

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

    def test_batched_skeleton_state_equivalence(self) -> None:
        """Test that batched operations produce equivalent results through different paths."""
        # Create batched skeleton states with different rotations and transformations
        batch_size = 3

        skel_state1 = self._random_skel_states(sz=(batch_size))
        skel_state2 = self._random_skel_states(sz=(batch_size))
        skel_state_final = pym_skel_state.multiply(skel_state1, skel_state2)
        mat1 = pym_skel_state.to_matrix(skel_state_final)

        trs1 = pym_trs.from_skeleton_state(skel_state1)
        trs2 = pym_trs.from_skeleton_state(skel_state2)
        trs_final = pym_trs.multiply(trs1, trs2)
        mat2 = pym_trs.to_matrix(trs_final)

        self.assertTrue(torch.allclose(mat1, mat2))

    def test_transform_points_skel_state_equivalence(self) -> None:
        """Test that batched operations produce equivalent results through different paths."""
        # Create batched skeleton states with different rotations and transformations
        torch.manual_seed(1002389)

        batch_size = 3

        skel_state = self._random_skel_states(sz=(batch_size))
        trs = pym_trs.from_skeleton_state(skel_state)

        points = torch.randn(batch_size, 3)

        transformed1 = pym_skel_state.transform_points(skel_state, points)
        transformed2 = pym_trs.transform_points(trs, points)

        self.assertTrue(torch.allclose(transformed1, transformed2))

    def test_rotmat_from_euler_xyz(self) -> None:
        """Test rotmat_from_euler_xyz function."""
        # Test with zero angles (should give identity matrix)
        euler_zero = torch.tensor([0.0, 0.0, 0.0])
        rotmat_zero = pym_trs.rotmat_from_euler_xyz(euler_zero)
        expected_identity = torch.eye(3)
        self.assertTrue(torch.allclose(rotmat_zero, expected_identity, atol=1e-6))

        # Test with 90-degree rotation around X-axis
        euler_x_90 = torch.tensor([math.pi / 2, 0.0, 0.0])
        rotmat_x_90 = pym_trs.rotmat_from_euler_xyz(euler_x_90)
        expected_x_90 = self._rotation_x(math.pi / 2)
        self.assertTrue(torch.allclose(rotmat_x_90, expected_x_90, atol=1e-6))

        # Test with 90-degree rotation around Y-axis
        euler_y_90 = torch.tensor([0.0, math.pi / 2, 0.0])
        rotmat_y_90 = pym_trs.rotmat_from_euler_xyz(euler_y_90)
        expected_y_90 = self._rotation_y(math.pi / 2)
        self.assertTrue(torch.allclose(rotmat_y_90, expected_y_90, atol=1e-6))

        # Test with 90-degree rotation around Z-axis
        euler_z_90 = torch.tensor([0.0, 0.0, math.pi / 2])
        rotmat_z_90 = pym_trs.rotmat_from_euler_xyz(euler_z_90)
        expected_z_90 = self._rotation_z(math.pi / 2)
        self.assertTrue(torch.allclose(rotmat_z_90, expected_z_90, atol=1e-6))

        # Test with combined rotations (30, 45, 60 degrees)
        rx, ry, rz = math.pi / 6, math.pi / 4, math.pi / 3
        euler_combined = torch.tensor([rx, ry, rz])
        rotmat_combined = pym_trs.rotmat_from_euler_xyz(euler_combined)

        # Manually compute expected result (ZYX order: Rz * Ry * Rx)
        expected_combined = torch.matmul(
            torch.matmul(self._rotation_z(rz), self._rotation_y(ry)),
            self._rotation_x(rx),
        )
        self.assertTrue(torch.allclose(rotmat_combined, expected_combined, atol=1e-6))

        # Test batched input
        batch_euler = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Identity
                [math.pi / 2, 0.0, 0.0],  # 90° around X
                [0.0, math.pi / 2, 0.0],  # 90° around Y
            ]
        )
        batch_rotmat = pym_trs.rotmat_from_euler_xyz(batch_euler)

        # Check shapes
        self.assertEqual(batch_rotmat.shape, (3, 3, 3))

        # Check individual matrices
        self.assertTrue(torch.allclose(batch_rotmat[0], torch.eye(3), atol=1e-6))
        self.assertTrue(
            torch.allclose(batch_rotmat[1], self._rotation_x(math.pi / 2), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(batch_rotmat[2], self._rotation_y(math.pi / 2), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
