# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import tempfile
import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import torch


class TestFBXIO(unittest.TestCase):
    def setUp(self) -> None:
        self.character = pym_geometry.create_test_character()
        torch.manual_seed(0)  # ensure repeatability

        nBatch = 5
        nParams = self.character.parameter_transform.size
        self.model_params = pym_geometry.uniform_random_to_model_parameters(
            self.character, torch.rand(nBatch, nParams)
        ).double()
        self.joint_params = self.character.parameter_transform.apply(self.model_params)

    def test_save_motions_with_model_params(self) -> None:
        # TODO: Disable this test programmatically if momentum is not built with FBX support
        # Test saving with model parameters
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params.numpy(),
                offsets=offsets,
                fps=60,
            )
            self._verify_fbx(temp_file.name)

        # Test saving with model parameters
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params.numpy(),
                offsets=offsets,
                fps=60,
                coord_system_info=pym_geometry.FBXCoordSystemInfo(
                    pym_geometry.FBXUpVector.YAxis,
                    pym_geometry.FBXFrontVector.ParityEven,
                    pym_geometry.FBXCoordSystem.LeftHanded,
                ),
            )
            self._verify_fbx(temp_file.name)

    def test_save_motions_with_joint_params(self) -> None:
        # TODO: Disable this test programmatically if momentum is not built with FBX support
        # Test saving with joint parameters
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=self.character,
                joint_params=self.joint_params.numpy(),
                fps=60,
            )
            self._verify_fbx(temp_file.name)

        # Test saving with joint parameters using non-default coord-system
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=self.character,
                joint_params=self.joint_params.numpy(),
                fps=60,
                coord_system_info=pym_geometry.FBXCoordSystemInfo(
                    pym_geometry.FBXUpVector.YAxis,
                    pym_geometry.FBXFrontVector.ParityEven,
                    pym_geometry.FBXCoordSystem.RightHanded,
                ),
            )
            self._verify_fbx(temp_file.name)

    def _verify_fbx(self, file_name: str) -> None:
        # Load FBX file
        _l_character, motion, fps = pym_geometry.Character.load_fbx_with_motion(
            file_name
        )
        self.assertEqual(1, len(motion))
        self.assertEqual(motion[0].shape, self.joint_params.shape)
        self.assertEqual(fps, 60)

    def test_save_with_namespace(self) -> None:
        """Test FBX save with namespace parameter."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            # Save with namespace
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params.numpy(),
                offsets=offsets,
                fps=60,
                fbx_namespace="test_ns",
            )
            # Verify file can be loaded
            self._verify_fbx(temp_file.name)

    def test_save_with_joint_params_and_namespace(self) -> None:
        """Test FBX save with joint params and namespace parameter."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=self.character,
                joint_params=self.joint_params.numpy(),
                fps=60,
                fbx_namespace="test_ns",
            )
            # Verify file can be loaded
            self._verify_fbx(temp_file.name)

    def test_unified_save_fbx(self) -> None:
        """Test unified save function with FBX extension."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            # Use unified save function - should auto-detect FBX format
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params.numpy(),
                offsets=offsets,
                fps=60,
            )
            # Verify file can be loaded
            self._verify_fbx(temp_file.name)

    def test_unified_save_glb(self) -> None:
        """Test unified save function with GLB extension."""
        with tempfile.NamedTemporaryFile(suffix=".glb") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            # Use unified save function - should auto-detect GLTF format
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params.numpy(),
                offsets=offsets,
                fps=60,
            )
            # Verify file can be loaded
            loaded_char, loaded_motion, _loaded_offsets, loaded_fps = (
                pym_geometry.Character.load_gltf_with_motion(temp_file.name)
            )
            self.assertEqual(loaded_char.skeleton.size, self.character.skeleton.size)
            self.assertEqual(loaded_motion.shape, self.model_params.shape)
            self.assertEqual(loaded_fps, 60)

    def test_unified_save_gltf(self) -> None:
        """Test unified save function with GLTF extension."""
        with tempfile.NamedTemporaryFile(suffix=".gltf") as temp_file:
            offsets = np.zeros(self.joint_params.shape[1])
            # Use unified save function - should auto-detect GLTF format
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params.numpy(),
                offsets=offsets,
                fps=60,
            )
            # Verify file can be loaded
            loaded_char, loaded_motion, _loaded_offsets, loaded_fps = (
                pym_geometry.Character.load_gltf_with_motion(temp_file.name)
            )
            self.assertEqual(loaded_char.skeleton.size, self.character.skeleton.size)
            self.assertEqual(loaded_motion.shape, self.model_params.shape)
            self.assertEqual(loaded_fps, 60)

    def test_marker_sequence_fbx_roundtrip(self) -> None:
        """Test saving and loading marker sequences with FBX."""
        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            # Create test marker sequence
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
                        occluded=(frame % 2 == 0 and i == 2),
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

            # Verify marker data
            for frame_idx, frame in enumerate(loaded_sequence.frames):
                # Check marker count (marker 2 is occluded on even frames)
                if frame_idx % 2 == 0:
                    # marker_2 should be occluded
                    visible_markers = [m for m in frame.markers if not m.occluded]
                    self.assertEqual(len(visible_markers), 2)
                else:
                    # All markers visible
                    visible_markers = [m for m in frame.markers if not m.occluded]
                    self.assertEqual(len(visible_markers), 3)

    def test_marker_sequence_fbx_with_motion(self) -> None:
        """Test saving markers and motion together in FBX."""
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
            offsets = np.zeros(self.joint_params.shape[1])
            pym_geometry.Character.save(
                path=temp_file.name,
                character=self.character,
                motion=self.model_params[:nFrames].numpy(),
                offsets=offsets,
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
