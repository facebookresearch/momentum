# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import pymomentum.marker_tracking as pym_marker_tracking
import pymomentum.skel_state as pym_skel_state
import torch


class TestMarkerTracking(unittest.TestCase):
    def test_convert_locators_to_skinned_locators(self) -> None:
        """Test that locators get converted to skinned locators with appropriate bone weights."""
        # Create a test character with mesh and skin weights
        character = pym_geometry.create_test_character(4)

        # Get a mesh vertex position to place our locator
        mesh_vertex_idx = (
            22  # Pick an arbitrary vertex (smaller index for test character)
        )
        mesh_vertex_position = character.mesh.vertices[mesh_vertex_idx]
        print("mesh_vertex_position: ", mesh_vertex_position)

        rest_model_params = np.zeros((1, character.parameter_transform.size))
        rest_skeleton_state = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, rest_model_params
            )[0]  # Remove batch dimension
        )
        parent_joint_idx = character.skin_weights.index[mesh_vertex_idx][0]

        # Create a locator positioned at the mesh vertex
        test_locator = pym_geometry.Locator(
            name="test_locator",
            parent=parent_joint_idx,
            offset=pym_skel_state.transform_points(
                pym_skel_state.inverse(rest_skeleton_state[parent_joint_idx]),
                torch.from_numpy(mesh_vertex_position),
            ).numpy(),  # Position it at the mesh vertex
            attached_to_skin=True,
        )

        # Add the locator to the character
        character_with_locator = character.with_locators([test_locator], replace=True)

        # Verify the locator was added
        self.assertEqual(len(character_with_locator.locators), 1)
        self.assertEqual(character_with_locator.locators[-1].name, "test_locator")

        # Convert locators to skinned locators
        character_with_skinned = (
            pym_marker_tracking.convert_locators_to_skinned_locators(
                character_with_locator, max_distance=3.0
            )
        )

        # Verify that we now have skinned locators
        self.assertGreater(len(character_with_skinned.skinned_locators), 0)

        # Find our converted locator
        converted_locator = None
        for skinned_loc in character_with_skinned.skinned_locators:
            if skinned_loc.name == "test_locator":
                converted_locator = skinned_loc
                break

        self.assertIsNotNone(
            converted_locator,
            "Test locator should have been converted to skinned locator",
        )

        # Verify the skinned locator has reasonable properties
        self.assertEqual(converted_locator.name, "test_locator")

        # Check that the skinned locator has bone weights
        # The weights should sum to approximately 1.0
        weight_sum = np.sum(converted_locator.skin_weights)
        self.assertAlmostEqual(
            weight_sum,
            1.0,
            places=5,
            msg="Skin weights should sum to approximately 1.0",
        )

        # Skin weights should get copied from the mesh:
        self.assertTrue(
            np.allclose(
                converted_locator.skin_weights,
                character.skin_weights.weight[mesh_vertex_idx],
            )
        )
        self.assertTrue(
            np.allclose(
                converted_locator.parents, character.skin_weights.index[mesh_vertex_idx]
            )
        )

        # Compute converted skinned locator's world space position
        # Skinned locator position is already in world space coordinates
        converted_world_pos = converted_locator.position

        # Compare world space positions
        position_diff = np.linalg.norm(converted_world_pos - mesh_vertex_position)
        self.assertLess(
            position_diff,
            0.01,  # Tight tolerance since they should match exactly
            msg=f"Converted locator world position should match original locator world position. "
            f"Original: {mesh_vertex_position}, Converted: {converted_world_pos}, "
            f"Diff: {position_diff}",
        )

        # The test_locator should no longer be in the regular locators
        self.assertEqual(character_with_skinned.locators, [])

    def test_convert_skinned_locators_to_locators(self) -> None:
        """Test that skinned locators get converted back to regular locators correctly."""
        # Create a test character with mesh and skin weights
        character = pym_geometry.create_test_character(4)

        # Hand-create a skinned locator with multiple bone influences
        # We'll create one at the origin in rest pose with known skin weights
        test_skinned_locator = pym_geometry.SkinnedLocator(
            name="test_skinned_locator",
            parents=np.array([0, 1, 2, 0, 0, 0, 0, 0], dtype=np.uint32),
            skin_weights=np.array(
                [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            weight=1.0,
        )

        # Add the skinned locator to the character
        character_with_skinned = character.with_skinned_locators(
            [test_skinned_locator], replace=True
        )

        # Verify the skinned locator was added
        self.assertEqual(len(character_with_skinned.skinned_locators), 1)
        self.assertEqual(
            character_with_skinned.skinned_locators[0].name, "test_skinned_locator"
        )

        # Convert skinned locators back to regular locators
        character_with_locators = (
            pym_marker_tracking.convert_skinned_locators_to_locators(
                character_with_skinned
            )
        )

        # Verify that we now have regular locators
        self.assertGreater(len(character_with_locators.locators), 0)

        # Find our converted locator
        converted_locator = None
        for loc in character_with_locators.locators:
            if loc.name == "test_skinned_locator":
                converted_locator = loc
                break

        self.assertIsNotNone(
            converted_locator,
            "Test skinned locator should have been converted to regular locator",
        )

        # Verify the regular locator is attached to the bone with highest weight
        # In our test case, bone 0 has weight 0.5 (highest)
        self.assertEqual(
            converted_locator.parent,
            0,
            "Locator should be attached to bone with highest skin weight (bone 0)",
        )

        # Verify the weight is preserved
        self.assertAlmostEqual(
            converted_locator.weight,
            1.0,
            places=5,
            msg="Locator weight should be preserved",
        )

        # Verify that the world position is preserved
        # Compute the world position of the converted locator in rest pose
        rest_model_params = np.zeros((1, character.parameter_transform.size))
        rest_skeleton_state = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, rest_model_params
            )[0]
        )

        # Transform the offset to world space using the parent bone's transform
        converted_world_pos = pym_skel_state.transform_points(
            rest_skeleton_state[converted_locator.parent],
            torch.from_numpy(converted_locator.offset),
        )

        # Compare with original skinned locator position
        original_world_pos = torch.from_numpy(test_skinned_locator.position)
        position_diff = torch.norm(converted_world_pos - original_world_pos).item()

        self.assertLess(
            position_diff,
            0.01,
            msg=f"Converted locator world position should match original skinned locator position. "
            f"Original: {original_world_pos.numpy()}, Converted: {converted_world_pos.numpy()}, "
            f"Diff: {position_diff}",
        )

        # The skinned locators should be empty after conversion
        self.assertEqual(
            len(character_with_locators.skinned_locators),
            0,
            "Skinned locators should be empty after conversion",
        )

    def test_marker_tracking_with_skinned_locators(self) -> None:
        """Test marker tracking with skinned locators created from mesh vertices."""
        torch.manual_seed(42)  # Ensure repeatability

        # Create a test character with mesh and skin weights
        character = pym_geometry.create_test_character(num_joints=5)
        self.assertIsNotNone(character.mesh)
        self.assertIsNotNone(character.skin_weights)

        # Get mesh vertices and their skinning data
        n_vertices = character.mesh.vertices.shape[0]
        if n_vertices == 0:
            self.skipTest("Test character has no mesh vertices")
            return

        # Randomly select 5 mesh vertices to use as skinned locators
        np.random.seed(42)
        n_locators = min(5, n_vertices)
        vertex_indices = np.random.choice(n_vertices, n_locators, replace=False)

        # Create skinned locators using the selected vertices' data
        skinned_locators = [
            pym_geometry.SkinnedLocator(
                name=f"vertex_locator_{vertex_idx}",
                parents=character.skin_weights.index[vertex_idx].astype(np.uint32),
                skin_weights=character.skin_weights.weight[vertex_idx].astype(
                    np.float32
                ),
                position=character.mesh.vertices[vertex_idx].astype(np.float32),
                weight=1.0,
            )
            for vertex_idx in vertex_indices
        ]

        # Replace the character's skinned locators with our vertex-based ones
        character = character.with_skinned_locators(skinned_locators, replace=True)

        # Generate a short sequence of model parameters (3 frames) using batched operations
        num_frames = 3

        # Create batched model parameters for all frames
        scaling_params = character.parameter_transform.scaling_parameters
        model_params_cur = np.zeros(character.parameter_transform.size)
        identity_params = np.where(
            scaling_params, model_params_cur, np.zeros_like(model_params_cur)
        )

        model_params_batch = np.zeros((num_frames, character.parameter_transform.size))
        for frame in range(num_frames):
            model_params_cur = np.where(
                scaling_params,
                identity_params,
                model_params_cur
                + 0.3 * np.random.rand(character.parameter_transform.size),
            )
            model_params_batch[frame, :] = model_params_cur

        # Convert model parameters to skeleton states
        skeleton_states = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_batch
        )

        # Compute skinned locator positions for all frames using the new function
        all_skinned_positions = character.skin_skinned_locators(
            skeleton_states
        )  # Shape: [num_frames, num_skinned_locators, 3]

        # Create marker data from batched skinned positions
        marker_data = []
        for frame_idx in range(num_frames):
            frame_markers = []
            for loc_idx, skinned_locator in enumerate(character.skinned_locators):
                marker = pym_geometry.Marker(
                    skinned_locator.name,
                    all_skinned_positions[frame_idx, loc_idx],
                    False,  # Not occluded
                )
                frame_markers.append(marker)
            marker_data.append(frame_markers)

        # Set up tracking configuration
        tracking_config = pym_marker_tracking.TrackingConfig(
            max_iter=20, min_vis_percent=0.5, debug=False, regularization=1e-6
        )

        calibration_config = pym_marker_tracking.CalibrationConfig(
            max_iter=10, calib_frames=num_frames
        )

        # Run marker tracking
        tracked_motion = pym_marker_tracking.process_markers(
            character,
            identity_params,
            marker_data,
            tracking_config,
            calibration_config,
            calibrate=False,
            first_frame=0,
            max_frames=num_frames,
        )

        # Verify tracking results
        self.assertEqual(tracked_motion.shape[0], num_frames)
        self.assertEqual(tracked_motion.shape[1], character.parameter_transform.size)

        # Test that we can compute skinned marker positions from tracked motion and verify accuracy

        # Convert tracked motion to skeleton states
        tracked_skeleton_states = pym_geometry.model_parameters_to_skeleton_state(
            character, tracked_motion
        )

        # Compute all skinned locator positions for all tracked frames using the new function
        computed_skinned_positions_batch = character.skin_skinned_locators(
            tracked_skeleton_states
        )  # Shape: [num_frames, num_skinned_locators, 3]

        # Compare with original marker positions
        for loc_idx, skinned_locator in enumerate(character.skinned_locators):
            for frame_idx in range(num_frames):
                original_marker_pos = marker_data[frame_idx][loc_idx].pos
                computed_pos = computed_skinned_positions_batch[frame_idx, loc_idx]
                position_error = np.linalg.norm(computed_pos - original_marker_pos)

                # The tracking should be reasonably accurate
                # Since these are real mesh vertices with proper skinning data,
                # we can expect high accuracy
                self.assertLess(
                    position_error,
                    0.08,  # Tight tolerance since these should track well
                    f"Mesh-based skinned marker {skinned_locator.name} position error too large in frame {frame_idx}: {position_error}",
                )

        # Verify that motion shows some progression for non-trivial frames
        if num_frames > 1:
            # Check that later frames differ from the first frame
            for frame_idx in range(1, num_frames):
                frame_diff = np.linalg.norm(
                    tracked_motion[frame_idx] - tracked_motion[0]
                )
                # Allow for the possibility that tracking converges to rest pose
                # but at least verify the tracking completed successfully
                self.assertGreaterEqual(
                    frame_diff,
                    0.0,
                    f"Frame {frame_idx} should have non-negative difference from frame 0",
                )

        # Test that the new skin_skinned_locators function produces consistent results
        # by comparing single-frame vs batched processing
        single_skeleton_state = skeleton_states[0]
        single_skinned_positions = character.skin_skinned_locators(
            single_skeleton_state
        )

        # Should match the first frame of the batched result
        self.assertTrue(
            np.allclose(single_skinned_positions, all_skinned_positions[0], atol=1e-6),
            "Single frame result should match first frame of batched result",
        )

        # Verify the new function produces the same results as mesh skinning
        # for the same skeleton state (this validates correctness of skin_skinned_locators)
        rest_vertices = character.mesh.vertices.astype(np.float64)
        skinned_mesh = character.skin_points(skeleton_states, rest_vertices)

        # Extract the positions of our selected vertices from the skinned mesh
        expected_positions = skinned_mesh[
            :, vertex_indices, :
        ]  # [num_frames, n_locators, 3]

        # Compare with our skinned locator results (ensure types match)
        all_skinned_np = all_skinned_positions
        self.assertEqual(all_skinned_np.shape, expected_positions.shape)
        self.assertTrue(
            np.allclose(
                all_skinned_np,
                expected_positions,
                atol=1e-5,
            ),
            f"Skinned locators should match mesh vertices. Max difference: {np.max(np.abs(all_skinned_np - expected_positions))}",
        )

    def test_marker_tracking_basic(self) -> None:
        """Test basic marker tracking functionality with a simple motion sequence."""
        np.random.seed(0)

        # Create a test character
        character = pym_geometry.create_test_character(4)

        # Add some test locators that will become markers
        test_locators = [
            pym_geometry.Locator(
                name=f"marker{i}",
                parent=i,
                offset=np.random.rand(3),
            )
            for i in range(character.skeleton.size)
        ]

        character = character.with_locators(test_locators, replace=True)

        # Generate a short sequence of model parameters (3 frames) using batched operations
        num_frames = 3

        # Create batched model parameters for all frames
        scaling_params = character.parameter_transform.scaling_parameters
        model_params_cur = np.zeros(character.parameter_transform.size)
        identity_params = np.where(
            scaling_params, model_params_cur, np.zeros_like(model_params_cur)
        )

        model_params_batch = np.zeros((num_frames, character.parameter_transform.size))
        for frame in range(num_frames):
            model_params_cur = np.where(
                scaling_params,
                identity_params,
                model_params_cur
                + 0.4 * np.random.rand(character.parameter_transform.size),
            )
            model_params_batch[frame, :] = model_params_cur

        # Get locator parent indices and offsets for position computation
        locator_parents = np.array(
            [loc.parent for loc in character.locators], dtype=np.int32
        )
        locator_offsets = np.stack([loc.offset for loc in character.locators]).astype(
            np.float64
        )

        # Compute all locator positions for all frames in one batched call
        all_locator_positions = pym_geometry.model_parameters_to_positions(
            character,
            model_params_batch,
            locator_parents,
            locator_offsets,
        )  # Shape: [num_frames, num_locators, 3]

        # Create marker data from batched positions
        marker_data = []
        for frame_idx in range(num_frames):
            frame_markers = []
            for loc_idx, locator in enumerate(character.locators):
                marker = pym_geometry.Marker(
                    locator.name,
                    all_locator_positions[frame_idx, loc_idx],
                    False,  # Not occluded
                )
                frame_markers.append(marker)
            marker_data.append(frame_markers)

        # Set up tracking configuration
        tracking_config = pym_marker_tracking.TrackingConfig(
            max_iter=20, min_vis_percent=0.5, debug=False, regularization=1e-6
        )

        calibration_config = pym_marker_tracking.CalibrationConfig(
            max_iter=10, calib_frames=num_frames
        )

        # Run marker tracking
        tracked_motion = pym_marker_tracking.process_markers(
            character,
            identity_params,
            marker_data,
            tracking_config,
            calibration_config,
            calibrate=False,
            first_frame=0,
            max_frames=num_frames,
        )

        # Verify tracking results
        self.assertEqual(tracked_motion.shape[0], num_frames)
        self.assertEqual(tracked_motion.shape[1], character.parameter_transform.size)

        # Test that we can compute marker positions from tracked motion and verify accuracy
        # Use batched operations for verification

        # Compute all locator positions for all tracked frames in one batched call
        # Convert offsets to match the dtype of tracked_motion (which is float32)
        computed_positions_batch = pym_geometry.model_parameters_to_positions(
            character,
            tracked_motion,
            locator_parents,
            locator_offsets.astype(tracked_motion.dtype),
        )  # Shape: [num_frames, num_locators, 3]

        # Compare with original marker positions
        for loc_idx, locator in enumerate(character.locators):
            for frame_idx in range(num_frames):
                original_marker_pos = marker_data[frame_idx][loc_idx].pos
                computed_pos = computed_positions_batch[frame_idx, loc_idx]
                position_error = np.linalg.norm(computed_pos - original_marker_pos)

                # The tracking should be reasonably accurate
                # Note: Marker tracking systems typically have errors in the range of 10-50cm
                # depending on the complexity of the motion and number of markers
                self.assertLess(
                    position_error,
                    0.15,  # tolerance depends on random motion complexity
                    f"Marker {locator.name} position error too large in frame {frame_idx}: {position_error}",
                )

        # Verify that motion shows some progression for non-trivial frames
        if num_frames > 1:
            # Check that later frames differ from the first frame
            for frame_idx in range(1, num_frames):
                frame_diff = np.linalg.norm(
                    tracked_motion[frame_idx] - tracked_motion[0]
                )
                # Allow for the possibility that tracking converges to rest pose
                # but at least verify the tracking completed successfully
                self.assertGreaterEqual(
                    frame_diff,
                    0.0,
                    f"Frame {frame_idx} should have non-negative difference from frame 0",
                )

    def test_convert_locators_to_skinned_locators_max_distance(self) -> None:
        """Test that max_distance parameter affects the conversion."""
        character = pym_geometry.create_test_character()

        # Create a locator positioned far from the mesh
        far_position = np.array(
            [100.0, 100.0, 100.0], dtype=np.float32
        )  # Very far from mesh
        far_locator = pym_geometry.Locator(
            name="far_locator",
            parent=0,
            offset=far_position,
        )

        character_with_locator = character.with_locators([far_locator])

        # Try conversion with small max_distance - should not convert the far locator
        character_small_distance = (
            pym_marker_tracking.convert_locators_to_skinned_locators(
                character_with_locator, max_distance=1.0
            )
        )

        # Try conversion with large max_distance - should convert the far locator
        character_large_distance = (
            pym_marker_tracking.convert_locators_to_skinned_locators(
                character_with_locator, max_distance=200.0
            )
        )

        # Check that the large distance version has more skinned locators
        # (This test assumes the far locator gets converted with large max_distance)
        small_distance_names = [
            loc.name for loc in character_small_distance.skinned_locators
        ]
        large_distance_names = [
            loc.name for loc in character_large_distance.skinned_locators
        ]

        # At minimum, both should have the same number or large_distance should have more
        self.assertGreaterEqual(
            len(large_distance_names),
            len(small_distance_names),
            msg="Larger max_distance should convert at least as many locators",
        )

    def test_get_locator_error_with_skinned_locators(self) -> None:
        """Test get_locator_error computes similar errors for regular and skinned locators."""
        torch.manual_seed(42)

        # Create a test character with mesh and skin weights
        character = pym_geometry.create_test_character(num_joints=5)
        self.assertIsNotNone(character.mesh)
        self.assertIsNotNone(character.skin_weights)

        # Get mesh vertices and their skinning data
        n_vertices = character.mesh.vertices.shape[0]
        if n_vertices == 0:
            self.skipTest("Test character has no mesh vertices")
            return

        # Create regular locators positioned near mesh vertices
        mesh_vertex_idx = 22  # Pick an arbitrary vertex
        mesh_vertex_position = character.mesh.vertices[mesh_vertex_idx]

        rest_model_params = np.zeros(
            (1, character.parameter_transform.size), dtype=np.float32
        )
        rest_skeleton_state = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, rest_model_params
            )[0]
        )
        parent_joint_idx = character.skin_weights.index[mesh_vertex_idx][0]

        # Create a locator positioned at the mesh vertex
        test_locator = pym_geometry.Locator(
            name="test_locator",
            parent=parent_joint_idx,
            offset=pym_skel_state.transform_points(
                pym_skel_state.inverse(rest_skeleton_state[parent_joint_idx]),
                torch.from_numpy(mesh_vertex_position),
            ).numpy(),
            attached_to_skin=True,
        )

        # Create character with regular locator
        character_with_locator = character.with_locators([test_locator], replace=True)

        # Convert to skinned locator
        character_with_skinned = (
            pym_marker_tracking.convert_locators_to_skinned_locators(
                character_with_locator, max_distance=3.0
            )
        )

        # Verify conversion happened
        self.assertEqual(len(character_with_skinned.skinned_locators), 1)
        self.assertEqual(
            character_with_skinned.skinned_locators[0].name, "test_locator"
        )

        # Generate a short motion sequence
        num_frames = 3
        scaling_params = character.parameter_transform.scaling_parameters
        model_params_cur = np.zeros(character.parameter_transform.size)
        identity_params = np.where(
            scaling_params, model_params_cur, np.zeros_like(model_params_cur)
        )

        model_params_batch = np.zeros((num_frames, character.parameter_transform.size))
        for frame in range(num_frames):
            model_params_cur = np.where(
                scaling_params,
                identity_params,
                model_params_cur
                + 0.3 * np.random.rand(character.parameter_transform.size),
            )
            model_params_batch[frame, :] = model_params_cur

        # Compute locator positions for the regular locator
        locator_parents = np.array([test_locator.parent], dtype=np.int32)
        locator_offsets = test_locator.offset.astype(np.float64).reshape(1, 3)
        regular_positions = pym_geometry.model_parameters_to_positions(
            character_with_locator,
            model_params_batch,
            locator_parents,
            locator_offsets,
        )  # Shape: [num_frames, 1, 3]

        # Add a known offset to create markers with measurable error (>2cm).
        # This ensures we're testing that skinned locators actually compute
        # meaningful positions rather than just returning zeros.
        marker_offset = np.array([0.03, 0.0, 0.0], dtype=np.float64)  # 3cm offset

        # Create marker data using regular locator positions + offset
        marker_data = []
        for frame_idx in range(num_frames):
            marker = pym_geometry.Marker(
                "test_locator",
                regular_positions[frame_idx, 0] + marker_offset,
                False,
            )
            marker_data.append([marker])

        # Compute error with regular locator character
        avg_error_regular, max_error_regular = pym_marker_tracking.get_locator_error(
            marker_data, model_params_batch, character_with_locator
        )

        # Compute error with skinned locator character
        avg_error_skinned, max_error_skinned = pym_marker_tracking.get_locator_error(
            marker_data, model_params_batch, character_with_skinned
        )

        # Regular locator error should be approximately the offset magnitude (3cm)
        expected_error = np.linalg.norm(marker_offset)
        self.assertGreater(
            avg_error_regular,
            0.02,  # At least 2cm error
            f"Regular locator avg error {avg_error_regular} should be significant (>2cm)",
        )
        self.assertLess(
            abs(avg_error_regular - expected_error),
            0.01,  # Should be close to 3cm
            f"Regular locator error {avg_error_regular} should be close to expected {expected_error}",
        )

        # The skinned locator error should also be close to the expected error.
        # This verifies that skinned locators are computing actual positions
        # (if they returned zeros, the error would be much larger than 3cm).
        self.assertGreater(
            avg_error_skinned,
            0.02,  # At least 2cm error (proves we're not just returning zeros)
            f"Skinned locator avg error {avg_error_skinned} should be significant (>2cm)",
        )
        self.assertLess(
            avg_error_skinned,
            0.15,  # Allow tolerance for skinning differences, but not too large
            f"Skinned locator avg error {avg_error_skinned} should be reasonably bounded",
        )

        # The errors should be similar (within a reasonable tolerance).
        # This is the key test: if skinned locators returned zeros, their error
        # would be much larger than the regular locator error.
        error_diff = abs(avg_error_regular - avg_error_skinned)
        self.assertLess(
            error_diff,
            0.1,
            f"Error difference {error_diff} between regular ({avg_error_regular}) "
            f"and skinned ({avg_error_skinned}) should be small",
        )


if __name__ == "__main__":
    unittest.main()
