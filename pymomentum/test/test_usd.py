# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import tempfile
import unittest
from typing import Optional

import numpy as np
import pymomentum.geometry as pym_geometry
import pymomentum.geometry_test_utils as pym_test_utils
import pymomentum.io_usd as pym_io_usd
import pymomentum.skel_state_np as pym_skel_state


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: Optional[str],
) -> unittest.TestSuite:
    """
    Custom test loader that excludes USD tests when USD is unavailable.

    This prevents 'Skipping' notifications in CI by not discovering
    the tests at all, rather than discovering and then skipping them.
    Works with uses_legacy_listing = True (static listing handles
    this correctly by finding 0 tests when load_tests returns empty).
    """
    if not pym_io_usd.is_usd_available():
        return unittest.TestSuite()
    return standard_tests


class TestUsd(unittest.TestCase):
    def test_is_usd_available(self) -> None:
        """Test that USD support is available.

        Note: USD is only available in ARVR BUCK builds, not fbcode builds.
        In fbcode builds, USD is disabled to avoid binary size explosion
        (OpenUSD is >4GB). This test only runs when USD is available;
        otherwise load_tests() returns an empty suite.
        """
        result = pym_io_usd.is_usd_available()
        self.assertIsInstance(result, bool)
        # When this test runs, USD should be available
        self.assertTrue(result)

    def test_save_and_load_usd_character(self) -> None:
        """Test saving and reloading a USD character."""
        char = pym_test_utils.create_test_character()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "character.usda")
            pym_io_usd.save_character(temp_path, char)

            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)

            loaded_char = pym_io_usd.load_character(temp_path)

            self.assertEqual(
                len(loaded_char.skeleton.joints),
                len(char.skeleton.joints),
            )
            for i, (loaded_joint, orig_joint) in enumerate(
                zip(loaded_char.skeleton.joints, char.skeleton.joints)
            ):
                self.assertEqual(
                    loaded_joint.name,
                    orig_joint.name,
                    f"Joint name mismatch at index {i}",
                )
            if char.mesh is not None:
                self.assertIsNotNone(loaded_char.mesh)
                self.assertEqual(
                    len(loaded_char.mesh.vertices),
                    len(char.mesh.vertices),
                )

    def test_save_usd_with_file_save_options(self) -> None:
        """Test saving a USD character with FileSaveOptions."""
        char = pym_test_utils.create_test_character()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save with mesh disabled
            path_no_mesh = os.path.join(temp_dir, "no_mesh.usda")
            options = pym_geometry.FileSaveOptions(mesh=False)
            pym_io_usd.save_character(path_no_mesh, char, options=options)

            self.assertTrue(os.path.exists(path_no_mesh))
            loaded = pym_io_usd.load_character(path_no_mesh)
            self.assertEqual(
                len(loaded.skeleton.joints),
                len(char.skeleton.joints),
            )

            # Save with default options (mesh enabled)
            path_with_mesh = os.path.join(temp_dir, "with_mesh.usda")
            pym_io_usd.save_character(path_with_mesh, char)

            self.assertTrue(os.path.exists(path_with_mesh))
            # File with mesh should be larger than without
            self.assertGreater(
                os.path.getsize(path_with_mesh),
                os.path.getsize(path_no_mesh),
            )

    def test_load_usd_from_bytes(self) -> None:
        """Test loading a USD character from bytes after saving."""
        char = pym_test_utils.create_test_character()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "character.usda")
            pym_io_usd.save_character(temp_path, char)

            with open(temp_path, "rb") as f:
                usd_bytes = f.read()

            loaded_char = pym_io_usd.load_character_from_bytes(usd_bytes)

            self.assertGreater(len(loaded_char.skeleton.joints), 0)
            self.assertEqual(
                len(loaded_char.skeleton.joints),
                len(char.skeleton.joints),
            )

    def test_save_load_usd_character_with_motion(self) -> None:
        """Test round-trip save/load with model parameters and offsets."""
        ref = pym_test_utils.create_test_character()
        frames = 3
        np.random.seed(0)
        poses = np.random.rand(frames, len(ref.parameter_transform.names)).astype(
            np.float32
        )
        offsets = np.random.rand(
            len(ref.skeleton.joint_names) * pym_geometry.PARAMETERS_PER_JOINT
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "character_motion.usda")
            pym_io_usd.save_character(
                path=path,
                character=ref,
                fps=120,
                motion=(ref.parameter_transform.names, poses),
                offsets=(ref.skeleton.joint_names, offsets),
            )

            char, ps, offs, fps = pym_io_usd.load_character_with_motion(path)

            self.assertEqual(char.skeleton.size, ref.skeleton.size)
            self.assertTrue(np.allclose(poses, ps, atol=1e-5))
            self.assertTrue(np.allclose(offsets, offs, atol=1e-5))
            self.assertEqual(fps, 120.0)

    def test_save_load_usd_character_with_motion_and_markers(self) -> None:
        """Test round-trip with model parameters and markers."""
        ref = pym_test_utils.create_test_character()
        frames = 3
        np.random.seed(0)
        poses = np.random.rand(frames, len(ref.parameter_transform.names)).astype(
            np.float32
        )
        offsets = np.random.rand(
            len(ref.skeleton.joint_names) * pym_geometry.PARAMETERS_PER_JOINT
        ).astype(np.float32)
        marker = pym_geometry.Marker(
            "m", np.asarray([1.0, 1.0, 1.0], dtype=np.float32), False
        )
        marker_list = [[marker] * 3] * frames

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "character_markers.usda")
            pym_io_usd.save_character(
                path=path,
                character=ref,
                fps=120,
                motion=(ref.parameter_transform.names, poses),
                offsets=(ref.skeleton.joint_names, offsets),
                markers=marker_list,
            )

            char, ps, offs, fps = pym_io_usd.load_character_with_motion(path)

            self.assertEqual(char.skeleton.size, ref.skeleton.size)
            self.assertTrue(np.allclose(poses, ps, atol=1e-5))
            self.assertTrue(np.allclose(offsets, offs, atol=1e-5))

            # Verify markers were preserved.
            # USD deduplicates markers by name, so the 3 identical "m" markers
            # collapse to 1 per frame.
            marker_seq = pym_io_usd.load_marker_sequence(path)
            self.assertEqual(len(marker_seq.frames), frames)
            for frame_markers in marker_seq.frames:
                self.assertEqual(len(frame_markers), 1)
                m = frame_markers[0]
                self.assertEqual(m.name, "m")
                self.assertFalse(m.occluded)
                np.testing.assert_allclose(m.pos, [1.0, 1.0, 1.0], atol=1e-5)

    def test_save_load_usd_character_with_motion_from_bytes(self) -> None:
        """Test loading motion from bytes."""
        ref = pym_test_utils.create_test_character()
        frames = 3
        np.random.seed(0)
        poses = np.random.rand(frames, len(ref.parameter_transform.names)).astype(
            np.float32
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "character_motion.usda")
            pym_io_usd.save_character(
                path=path,
                character=ref,
                fps=60,
                motion=(ref.parameter_transform.names, poses),
            )

            with open(path, "rb") as f:
                usd_bytes = f.read()

            char, ps, offs, fps = pym_io_usd.load_character_with_motion_from_bytes(
                usd_bytes
            )

            self.assertEqual(char.skeleton.joint_names, ref.skeleton.joint_names)
            self.assertTrue(np.allclose(ps, poses, atol=1e-5))

    def test_save_load_usd_character_with_skel_states(self) -> None:
        """Test round-trip with skeleton states."""
        ref = pym_test_utils.create_test_character()
        frames = 3
        np.random.seed(0)
        model_params = np.random.random(
            (frames, len(ref.parameter_transform.names))
        ).astype(np.float32)
        skeleton_states = pym_geometry.model_parameters_to_skeleton_state(
            ref, model_params
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "character_skel.usda")
            pym_io_usd.save_character_from_skel_states(
                path=path,
                character=ref,
                fps=120,
                skel_states=skeleton_states,
            )

            char, skel_states_read, timestamps = (
                pym_io_usd.load_character_with_skel_states(path)
            )

            self.assertEqual(char.skeleton.size, ref.skeleton.size)
            self.assertEqual(skel_states_read.shape, skeleton_states.shape)
            np.testing.assert_allclose(
                pym_skel_state.to_matrix(skeleton_states),
                pym_skel_state.to_matrix(skel_states_read),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_save_load_usd_character_with_skel_states_from_bytes(self) -> None:
        """Test loading skeleton states from bytes."""
        ref = pym_test_utils.create_test_character()
        frames = 3
        np.random.seed(0)
        model_params = np.random.random(
            (frames, len(ref.parameter_transform.names))
        ).astype(np.float32)
        skeleton_states = pym_geometry.model_parameters_to_skeleton_state(
            ref, model_params
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "character_skel.usda")
            pym_io_usd.save_character_from_skel_states(
                path=path,
                character=ref,
                fps=60,
                skel_states=skeleton_states,
            )

            with open(path, "rb") as f:
                usd_bytes = f.read()

            char, skel_states_read, timestamps = (
                pym_io_usd.load_character_with_skel_states_from_bytes(usd_bytes)
            )

            self.assertEqual(char.skeleton.size, ref.skeleton.size)
            self.assertEqual(skel_states_read.shape, skeleton_states.shape)
            np.testing.assert_allclose(
                pym_skel_state.to_matrix(skeleton_states),
                pym_skel_state.to_matrix(skel_states_read),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_load_usd_with_motion_model_parameter_scales(self) -> None:
        """Test loading motion with model parameter scales.

        Verifies that:
        1. We can load motion using load_character_with_motion_model_parameter_scales
        2. Results are consistent with load_character_with_skel_states
        3. The function returns model parameter scales (not joint parameters)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            character = pym_test_utils.create_test_character(num_joints=5)
            n_params = len(character.parameter_transform.names)
            n_frames = 10
            fps = 60.0

            np.random.seed(42)
            motion = np.random.rand(n_frames, n_params).astype(np.float32) * 0.1

            # Set scale parameters to consistent values across frames
            scaling_params = character.parameter_transform.scaling_parameters
            for i in range(n_params):
                if scaling_params[i]:
                    motion[:, i] = motion[0, i]

            path_test = os.path.join(temp_dir, "motion_test.usda")
            pym_io_usd.save_character(
                path=path_test,
                character=character,
                fps=fps,
                motion=(character.parameter_transform.names, motion),
            )

            # Load with model parameter scales
            char1, motion1, identity1, fps1 = (
                pym_io_usd.load_character_with_motion_model_parameter_scales(path_test)
            )

            # Load with regular motion for comparison
            char2, motion2, _, fps2 = pym_io_usd.load_character_with_motion(path_test)

            # Load with skel states for comparison
            char3, skel_states, _ = pym_io_usd.load_character_with_skel_states(
                path_test
            )

            # Verify characters match
            self.assertEqual(char1.skeleton.size, character.skeleton.size)
            self.assertEqual(char2.skeleton.size, character.skeleton.size)
            self.assertEqual(char3.skeleton.size, character.skeleton.size)

            # Verify FPS
            self.assertEqual(fps1, fps)
            self.assertEqual(fps2, fps)

            # Verify motion shapes
            self.assertEqual(motion1.shape, (n_frames, n_params))
            self.assertEqual(motion2.shape, (n_frames, n_params))

            # Verify model parameter scales returns correct identity shape
            self.assertEqual(
                identity1.shape,
                (n_params,),
                f"Expected model param scales shape ({n_params},), got {identity1.shape}",
            )

            # Compare with skeleton states.
            # Use atol/rtol=1e-3 because load_character_with_skel_states reconstructs
            # joint params from USD TRS (quaternion->Euler + log2 scale round-trip),
            # introducing small numerical differences vs the model-parameter path.
            for i in range(n_frames):
                skel_state_from_motion = (
                    pym_geometry.model_parameters_to_skeleton_state(char1, motion1[i])
                )
                self.assertTrue(
                    np.allclose(
                        pym_skel_state.to_matrix(skel_state_from_motion),
                        pym_skel_state.to_matrix(skel_states[i]),
                        atol=1e-3,
                        rtol=1e-3,
                    ),
                    f"Frame {i}: Skeleton states should match between methods",
                )

    def test_load_usd_marker_sequence(self) -> None:
        """Test loading a marker sequence from a USD file."""
        ref = pym_test_utils.create_test_character()
        frames = 4
        np.random.seed(0)
        poses = np.random.rand(frames, len(ref.parameter_transform.names)).astype(
            np.float32
        )

        markers_per_frame = []
        for frame in range(frames):
            frame_markers = []
            for i in range(3):
                marker = pym_geometry.Marker(
                    name=f"marker_{i}",
                    pos=np.array(
                        [float(frame + i), float(i), float(frame)], dtype=np.float64
                    ),
                    occluded=False,
                )
                frame_markers.append(marker)
            markers_per_frame.append(frame_markers)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "markers.usda")
            pym_io_usd.save_character(
                path=path,
                character=ref,
                fps=24,
                motion=(ref.parameter_transform.names, poses),
                markers=markers_per_frame,
            )

            marker_seq = pym_io_usd.load_marker_sequence(path)
            self.assertIsNotNone(marker_seq)
            self.assertEqual(len(marker_seq.frames), frames)
            for frame_idx, frame_markers in enumerate(marker_seq.frames):
                self.assertEqual(len(frame_markers), 3)
                for i, m in enumerate(frame_markers):
                    self.assertEqual(m.name, f"marker_{i}")
                    self.assertFalse(m.occluded)
                    expected_pos = [
                        float(frame_idx + i),
                        float(i),
                        float(frame_idx),
                    ]
                    np.testing.assert_allclose(
                        m.pos,
                        expected_pos,
                        atol=1e-5,
                        err_msg=f"Marker {i} position mismatch at frame {frame_idx}",
                    )

    def test_save_load_usd_character(self) -> None:
        """Test save_character and load_character round-trip."""
        character = pym_test_utils.create_test_character()
        np.random.seed(42)
        model_params = np.random.rand(
            5, len(character.parameter_transform.names)
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_save_load.usda")
            pym_io_usd.save_character(
                path=path,
                character=character,
                fps=60,
                motion=(character.parameter_transform.names, model_params),
            )
            # Verify file was created
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            # Verify we can load it back
            loaded_char = pym_io_usd.load_character(path)
            self.assertEqual(loaded_char.skeleton.size, character.skeleton.size)

    def test_save_usd_character_with_special_char_locators(self) -> None:
        """Test that locators with special characters in names don't crash USD save.

        USD SDF prim names must match [a-zA-Z_][a-zA-Z0-9_]*.
        Names with dots, dashes, colons, or spaces must be sanitized.
        The original names should be preserved via attributes for round-trip.
        """
        char = pym_test_utils.create_test_character()

        # Replace locators with ones that have special characters in names
        special_locators = [
            pym_geometry.Locator(name="nose.tip", parent=0, offset=[0.0, 1.0, 0.0]),
            pym_geometry.Locator(name="left-eye", parent=0, offset=[1.0, 0.0, 0.0]),
            pym_geometry.Locator(name="brow:inner.L", parent=1, offset=[0.0, 0.0, 1.0]),
            pym_geometry.Locator(name="chin tip", parent=1, offset=[0.5, 0.5, 0.0]),
        ]
        char = char.with_locators(special_locators, replace=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "special_locators.usda")
            pym_io_usd.save_character(path, char)

            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            loaded = pym_io_usd.load_character(path)
            self.assertEqual(len(loaded.locators), len(special_locators))

            # Verify original names are preserved via the momentum:name attribute
            for orig, loaded_loc in zip(special_locators, loaded.locators):
                self.assertEqual(
                    loaded_loc.name,
                    orig.name,
                    f"Locator name round-trip failed: expected '{orig.name}', got '{loaded_loc.name}'",
                )

    def test_save_usd_character_with_special_char_blendshapes(self) -> None:
        """Test that blend shapes with special characters in names don't crash USD save."""
        char = pym_test_utils.create_test_character(with_blendshapes=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "blendshapes.usda")
            pym_io_usd.save_character(path, char)

            self.assertTrue(os.path.exists(path))
            loaded = pym_io_usd.load_character(path)
            self.assertIsNotNone(loaded.blend_shape)

    def test_save_usd_with_collision_geometry(self) -> None:
        """Test that characters with collision geometry save/load correctly via USD."""
        char = pym_test_utils.create_test_character()

        # Verify the test character has collision geometry
        self.assertGreater(len(char.collision_geometry), 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "collision.usda")
            pym_io_usd.save_character(path, char)

            self.assertTrue(os.path.exists(path))
            loaded = pym_io_usd.load_character(path)
            self.assertEqual(
                len(loaded.collision_geometry),
                len(char.collision_geometry),
            )

    def test_save_load_usd_character_with_skel_states_roundtrip(self) -> None:
        """Test save_character_from_skel_states and load_character_with_skel_states."""
        ref = pym_test_utils.create_test_character()
        frames = 3
        np.random.seed(0)
        model_params = np.random.random(
            (frames, len(ref.parameter_transform.names))
        ).astype(np.float32)
        skeleton_states = pym_geometry.model_parameters_to_skeleton_state(
            ref, model_params
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_skel_roundtrip.usda")
            pym_io_usd.save_character_from_skel_states(
                path=path,
                character=ref,
                fps=60,
                skel_states=skeleton_states,
            )
            self.assertTrue(os.path.exists(path))

            char, skel_states_read, _ = pym_io_usd.load_character_with_skel_states(path)
            self.assertEqual(char.skeleton.size, ref.skeleton.size)
            self.assertEqual(skel_states_read.shape, skeleton_states.shape)


if __name__ == "__main__":
    unittest.main()
