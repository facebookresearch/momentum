# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import math
import pkgutil
import tempfile
import unittest

import numpy as np
import pymomentum.geometry as pym_geometry  # @manual=:geometry
import pymomentum.geometry_test_utils as pym_test_utils  # @manual=:geometry_test_utils


class TestFBX(unittest.TestCase):
    def setUp(self) -> None:
        self.character = pym_test_utils.create_test_character()

        nBatch = 5
        nParams = self.character.parameter_transform.size
        np.random.seed(0)  # ensure repeatability
        self.model_params = pym_geometry.uniform_random_to_model_parameters(
            self.character, np.random.rand(nBatch, nParams).astype(np.float32)
        ).astype(np.float64)
        self.joint_params = self.character.parameter_transform.apply(self.model_params)
        self.skeleton_state = pym_geometry.model_parameters_to_skeleton_state(
            self.character, self.model_params
        )

    def test_fbx_builder_multiple_characters(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        body = pym_test_utils.create_test_character(num_joints=5)
        prop = pym_test_utils.create_test_character(num_joints=3)
        prop_model_params = np.zeros(
            (2, prop.parameter_transform.size), dtype=np.float32
        )
        prop_joint_params = prop.parameter_transform.apply(prop_model_params)

        builder = pym_geometry.FbxBuilder()
        body_name = builder.add_character(body)
        prop_name = builder.add_rigid_body(prop)
        self.assertEqual(body_name, "test character")
        self.assertEqual(prop_name, "test character_1")
        builder.add_motion_with_joint_params(
            prop,
            fps=30.0,
            joint_params=prop_joint_params,
            character_name=prop_name,
        )

        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            builder.save(temp_file.name)
            loaded = pym_geometry.Character.load_fbx(
                temp_file.name,
                permissive=True,
                strip_namespaces=False,
            )

        self.assertEqual(
            loaded.skeleton.size,
            body.skeleton.size + prop.skeleton.size,
        )
        self.assertEqual(
            sum(parent == -1 for parent in loaded.skeleton.joint_parents),
            2,
        )
        self.assertIn(
            f"{prop_name}:{prop.skeleton.joint_names[0]}",
            loaded.skeleton.joint_names,
        )

    def test_fbx_builder_duplicate_name_mismatch_raises(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        body = pym_test_utils.create_test_character(num_joints=5)
        prop = pym_test_utils.create_test_character(num_joints=3)
        body_joint_params = body.parameter_transform.apply(
            np.zeros((2, body.parameter_transform.size), dtype=np.float32)
        )

        builder = pym_geometry.FbxBuilder()
        body_name = builder.add_character(body)
        prop_name = builder.add_rigid_body(prop)
        self.assertEqual(body_name, "test character")
        self.assertEqual(prop_name, "test character_1")

        # Passing the explicit name of a different-skeleton character (the rigid body, 3 joints)
        # while animating body (5 joints) must raise.
        with self.assertRaisesRegex(Exception, "does not match exported character"):
            builder.add_motion_with_joint_params(
                body,
                fps=30.0,
                joint_params=body_joint_params,
                character_name=prop_name,
            )

        # Passing the explicit correct name works.
        builder.add_motion_with_joint_params(
            body,
            fps=30.0,
            joint_params=body_joint_params,
            character_name=body_name,
        )

    def test_fbx_builder_duplicate_name_sequential_suffixes(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        builder = pym_geometry.FbxBuilder()
        self.assertEqual(
            builder.add_rigid_body(pym_test_utils.create_test_character(num_joints=3)),
            "test character",
        )
        self.assertEqual(
            builder.add_rigid_body(pym_test_utils.create_test_character(num_joints=3)),
            "test character_1",
        )
        self.assertEqual(
            builder.add_rigid_body(pym_test_utils.create_test_character(num_joints=3)),
            "test character_2",
        )

    def test_load_animation(self) -> None:
        """
        Loads a very simple 3-bone chain with animation from FBX and verifies that
        the animation is correct, including non-animated channels.
        """
        fbx_bytes = pkgutil.get_data(
            __package__,
            "resources/animation_test.fbx",
        )
        assert fbx_bytes is not None

        (
            character,
            animations,
            fps,
        ) = pym_geometry.Character.load_fbx_with_motion_from_bytes(fbx_bytes)
        self.assertEqual(fps, 24)
        self.assertEqual(len(animations), 1)
        joint_params = animations[0]

        joint1 = character.skeleton.joint_index("joint1")
        joint2 = character.skeleton.joint_index("joint2")
        joint3 = character.skeleton.joint_index("joint3")
        joint_params = joint_params.reshape(-1, len(character.skeleton.joint_names), 7)
        joint_params_first = joint_params[0]
        joint_params_last = joint_params[-1]

        self.assertTrue(
            np.allclose(
                joint_params_first[joint1][3:6],
                np.asarray([math.pi / 2, 0, 0], dtype=np.float32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                joint_params_first[joint2][3:6],
                np.asarray([0, 0, math.pi / 2], dtype=np.float32),
                atol=1e-5,
            )
        )

        self.assertTrue(
            np.allclose(
                joint_params_last[joint1][3:6],
                np.asarray([math.pi / 2, math.pi / 2, 0], dtype=np.float32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                joint_params_last[joint2][3:6],
                np.asarray([0, 0, math.pi / 2], dtype=np.float32),
                atol=1e-5,
            )
        )

        skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params
        )

        skel_state_first = skel_state[0]
        skel_state_last = skel_state[-1]

        # Start and end values read out from Maya.
        # 3-joint chain basically just rotates by 90 degrees.
        self.assertTrue(
            np.allclose(skel_state_first[joint1][0:3], np.array([0, 0, 4]), atol=1e-5)
        )
        self.assertTrue(
            np.allclose(skel_state_first[joint2][0:3], np.array([4, 0, 4]), atol=1e-5)
        )
        self.assertTrue(
            np.allclose(skel_state_first[joint3][0:3], np.array([4, 0, 8]), atol=1e-5)
        )

        self.assertTrue(
            np.allclose(skel_state_last[joint1][0:3], np.array([0, 0, 4]), atol=1e-5)
        )
        self.assertTrue(
            np.allclose(skel_state_last[joint2][0:3], np.array([0, 0, 0]), atol=1e-5)
        )
        self.assertTrue(
            np.allclose(skel_state_last[joint3][0:3], np.array([4, 0, 0]), atol=1e-5)
        )

    def test_save_motions_with_model_params(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        # Test saving with model parameters
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params,
                offsets=offsets,
                fps=60,
            )
            self._verify_fbx(temp_file.name)

        # Test saving with model parameters
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            options = pym_geometry.FileSaveOptions(
                coord_system_info=pym_geometry.FbxCoordSystemInfo(
                    pym_geometry.FbxUpVector.YAxis,
                    pym_geometry.FbxFrontVector.ParityEven,
                    pym_geometry.FbxCoordSystem.LeftHanded,
                )
            )
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params,
                offsets=offsets,
                fps=60,
                options=options,
            )
            self._verify_fbx(temp_file.name)

    def test_save_motions_with_joint_params(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        # Test saving with joint parameters
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=self.character,
                joint_params=self.joint_params,
                fps=60,
            )
            self._verify_fbx(temp_file.name)

        # Test saving with joint parameters using non-default coord-system
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            options = pym_geometry.FileSaveOptions(
                coord_system_info=pym_geometry.FbxCoordSystemInfo(
                    pym_geometry.FbxUpVector.YAxis,
                    pym_geometry.FbxFrontVector.ParityEven,
                    pym_geometry.FbxCoordSystem.RightHanded,
                )
            )
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=self.character,
                joint_params=self.joint_params,
                fps=60,
                options=options,
            )
            self._verify_fbx(temp_file.name)

    def test_save_motions_with_skeleton_states(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        # Test saving with skeleton states
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            pym_geometry.Character.save_with_skel_states(
                path=temp_file.name,
                character=self.character,
                fps=60,
                skel_states=self.skeleton_state,
            )
            self._verify_fbx(temp_file.name)

    def _verify_fbx(self, file_name: str) -> None:
        # Load FBX file
        _, motion, fps = pym_geometry.Character.load_fbx_with_motion(file_name)
        self.assertEqual(1, len(motion))
        self.assertEqual(motion[0].shape, self.joint_params.shape)
        self.assertEqual(fps, 60)

    def test_save_with_namespace(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        """Test FBX save with namespace parameter."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            # Save with namespace
            options = pym_geometry.FileSaveOptions(fbx_namespace="test_ns")
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params,
                offsets=offsets,
                fps=60,
                options=options,
            )
            # Verify file can be loaded
            self._verify_fbx(temp_file.name)

    def test_save_with_joint_params_and_namespace(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        """Test FBX save with joint params and namespace parameter."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            options = pym_geometry.FileSaveOptions(fbx_namespace="test_ns")
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=self.character,
                joint_params=self.joint_params,
                fps=60,
                options=options,
            )
            # Verify file can be loaded
            self._verify_fbx(temp_file.name)

    def test_unified_save_fbx(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        """Test unified save function with FBX extension."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            # Use unified save function - should auto-detect FBX format
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                fps=60,
                motion=self.model_params,
            )
            # Verify file can be loaded
            self._verify_fbx(temp_file.name)

    def test_marker_sequence_fbx_roundtrip(self) -> None:
        """Test saving and loading marker sequences with FBX.

        Note: FBX format does not preserve occlusion information. Occluded markers
        are not saved, and when loaded, all markers are interpolated for all frames.
        """
        if not pym_geometry.is_fbxsdk_available():
            return

        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            # Create test marker sequence without occlusion
            # (FBX doesn't preserve occlusion state)
            nFrames = 5
            markers_per_frame = []
            for frame in range(nFrames):
                frame_markers = []
                for i in range(3):
                    marker = pym_geometry.Marker(
                        name=f"marker_{i}",
                        pos=np.array(
                            [float(frame + i), float(i), float(frame)], dtype=np.float32
                        ),
                        occluded=False,  # FBX doesn't support occlusion
                    )
                    frame_markers.append(marker)
                markers_per_frame.append(frame_markers)

            # Save with unified function
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                fps=60,
                markers=markers_per_frame,
            )

            # Load markers using load_markers function
            marker_sequences = pym_geometry.load_markers(temp_file.name)
            self.assertEqual(len(marker_sequences), 1)

            loaded_sequence = marker_sequences[0]
            self.assertEqual(len(loaded_sequence.frames), nFrames)
            self.assertEqual(loaded_sequence.fps, 60.0)

            # Verify marker data - all markers should be present in all frames
            for _frame_idx, frame in enumerate(loaded_sequence.frames):
                self.assertEqual(len(frame), 3)  # All 3 markers present
                # Verify marker names
                marker_names = {m.name for m in frame}
                self.assertEqual(marker_names, {"marker_0", "marker_1", "marker_2"})

    def test_marker_sequence_fbx_with_motion(self) -> None:
        """Test saving markers and motion together in FBX."""
        if not pym_geometry.is_fbxsdk_available():
            return

        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            # Create test marker sequence
            nFrames = 3
            markers_per_frame = []
            for frame in range(nFrames):
                frame_markers = []
                for i in range(2):
                    marker = pym_geometry.Marker(
                        name=f"marker_{i}",
                        pos=np.array([float(i), float(frame), 0.0], dtype=np.float32),
                        occluded=False,
                    )
                    frame_markers.append(marker)
                markers_per_frame.append(frame_markers)

            # Save with both motion and markers
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params[:nFrames],
                fps=60,
                markers=markers_per_frame,
            )

            # Load and verify markers
            marker_sequences = pym_geometry.load_markers(temp_file.name)
            self.assertEqual(len(marker_sequences), 1)
            self.assertEqual(len(marker_sequences[0].frames), nFrames)

            # Load and verify motion
            _loaded_char, motion, _loaded_fps = (
                pym_geometry.Character.load_fbx_with_motion(temp_file.name)
            )
            self.assertEqual(len(motion), 1)
            self.assertEqual(motion[0].shape[0], nFrames)

    def test_marker_sequence_sparse_keyframes(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        """Test that marker sequences support sparse keyframes (not all frames have markers)."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            # Create sparse marker sequence - only keyframe at frame 0 and 2
            nFrames = 3
            markers_per_frame = []

            # Frame 0: has markers
            frame_markers_0 = [
                pym_geometry.Marker(
                    name="marker_0",
                    pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    occluded=False,
                )
            ]
            markers_per_frame.append(frame_markers_0)

            # Frame 1: empty - no keyframe
            markers_per_frame.append([])

            # Frame 2: has markers
            frame_markers_2 = [
                pym_geometry.Marker(
                    name="marker_0",
                    pos=np.array([2.0, 2.0, 2.0], dtype=np.float32),
                    occluded=False,
                )
            ]
            markers_per_frame.append(frame_markers_2)

            # Save
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                fps=60,
                markers=markers_per_frame,
            )

            # Load markers
            marker_sequences = pym_geometry.load_markers(temp_file.name)
            self.assertEqual(len(marker_sequences), 1)

            loaded_sequence = marker_sequences[0]
            # Should have 3 frames total (sparse support)
            self.assertEqual(len(loaded_sequence.frames), nFrames)

    def test_load_markers_empty_file(self) -> None:
        if not pym_geometry.is_fbxsdk_available():
            return

        """Test loading markers from a file without markers."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            # Save character without markers
            pym_geometry.Character.save(
                path=temp_file.name, character=self.character, fps=60
            )

            # Load markers - should return empty sequence
            marker_sequences = pym_geometry.load_markers(temp_file.name)
            self.assertEqual(len(marker_sequences), 1)
            self.assertEqual(len(marker_sequences[0].frames), 0)
