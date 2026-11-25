# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import numpy as np
import pymomentum.geometry as pym_geometry  # @manual=:geometry
import pymomentum.geometry_test_utils as pym_test_utils  # @manual=:geometry_test_utils

_PPJ: int = pym_geometry.PARAMETERS_PER_JOINT


def _by_joint(
    character: pym_geometry.Character, states: np.ndarray
) -> dict[str, np.ndarray]:
    """Global skeleton states [n_frames x 8] of every joint, by joint name."""
    return {
        name: states[:, joint]
        for joint, name in enumerate(character.skeleton.joint_names)
    }


def _model_states(
    character: pym_geometry.Character, motion: np.ndarray
) -> dict[str, np.ndarray]:
    """The joint states of the model parameters `motion` [n_frames x n_parameters], by joint name."""
    return _by_joint(
        character, pym_geometry.model_parameters_to_skeleton_state(character, motion)
    )


def _joint_states(
    character: pym_geometry.Character, joint_parameters: np.ndarray
) -> dict[str, np.ndarray]:
    """The joint states of `joint_parameters` [n_frames x n_joint_parameters], by joint name."""
    return _by_joint(
        character,
        pym_geometry.joint_parameters_to_skeleton_state(character, joint_parameters),
    )


def _without_scales(
    character: pym_geometry.Character, motion: np.ndarray
) -> np.ndarray:
    """`motion` without the model parameters that drive a joint scale, which BVH cannot store."""
    scale_rows = character.parameter_transform.transform[_PPJ - 1 :: _PPJ]
    result = motion.copy()
    result[:, np.any(scale_rows != 0, axis=0)] = 0.0
    return result


class TestBvh(unittest.TestCase):
    def _expect_same_joint_transforms(
        self,
        expected: dict[str, np.ndarray],
        actual: dict[str, np.ndarray],
    ) -> None:
        self.assertEqual(sorted(actual), sorted(expected))
        for name, expected_states in expected.items():
            actual_states = actual[name]
            np.testing.assert_allclose(
                actual_states[:, :3], expected_states[:, :3], atol=1e-3, err_msg=name
            )
            # q and -q are the same rotation.
            dots = np.abs(
                np.sum(actual_states[:, 3:7] * expected_states[:, 3:7], axis=1)
            )
            np.testing.assert_allclose(dots, 1.0, atol=1e-5, err_msg=name)

    def test_saved_motion_keeps_the_joint_transforms(self) -> None:
        character = pym_test_utils.create_test_character()
        rng = np.random.default_rng(12345)
        motion = _without_scales(
            character,
            rng.uniform(-0.5, 0.5, size=(3, character.parameter_transform.size)).astype(
                np.float32
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "motion.bvh")
            pym_geometry.Character.save_bvh(path, character, 30.0, motion)
            loaded, loaded_motion, fps = pym_geometry.Character.load_bvh_with_motion(
                path
            )
            loaded_without_motion = pym_geometry.Character.load_bvh(path)

        self.assertAlmostEqual(fps, 30.0, places=2)
        self.assertEqual(loaded_motion.shape[0], motion.shape[0])
        self._expect_same_joint_transforms(
            _model_states(character, motion), _model_states(loaded, loaded_motion)
        )
        self.assertEqual(
            loaded_without_motion.skeleton.joint_names, character.skeleton.joint_names
        )

    def test_identity_offsets_are_added_to_every_frame(self) -> None:
        character = pym_test_utils.create_test_character()
        motion = np.zeros((2, character.parameter_transform.size), dtype=np.float32)
        offsets = np.zeros(character.skeleton.size * _PPJ, dtype=np.float32)
        offsets[_PPJ + 1] = 2.0  # The y translation of joint 1.

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "offsets.bvh")
            pym_geometry.Character.save_bvh(path, character, 30.0, motion, offsets)
            loaded, loaded_motion, _ = pym_geometry.Character.load_bvh_with_motion(path)

        self._expect_same_joint_transforms(
            _joint_states(
                character, character.parameter_transform.apply(motion) + offsets
            ),
            _model_states(loaded, loaded_motion),
        )

    def test_saved_joint_parameters_keep_the_joint_transforms(self) -> None:
        character = pym_test_utils.create_test_character()
        rng = np.random.default_rng(12345)
        joint_parameters = rng.uniform(
            -0.5, 0.5, size=(3, character.skeleton.size, _PPJ)
        ).astype(np.float32)
        joint_parameters[:, :, _PPJ - 1] = 0.0  # BVH cannot store joint scales.
        joint_parameters = joint_parameters.reshape(3, -1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "joint_parameters.bvh")
            pym_geometry.Character.save_bvh_with_joint_params(
                path, character, 30.0, joint_parameters
            )
            loaded, loaded_motion, _ = pym_geometry.Character.load_bvh_with_motion(path)

        self._expect_same_joint_transforms(
            _joint_states(character, joint_parameters),
            _model_states(loaded, loaded_motion),
        )

    def test_motion_with_another_parameter_count_raises(self) -> None:
        character = pym_test_utils.create_test_character()
        motion = np.zeros((2, character.parameter_transform.size + 1), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                pym_geometry.Character.save_bvh(
                    os.path.join(temp_dir, "bad.bvh"), character, 30.0, motion
                )

    def test_saved_character_without_motion_has_one_frame_of_its_rest_pose(
        self,
    ) -> None:
        character = pym_test_utils.create_test_character()
        offsets = np.zeros(character.skeleton.size * _PPJ, dtype=np.float32)
        offsets[_PPJ + 1] = 2.0

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "rest.bvh")
            pym_geometry.Character.save_bvh(path, character, 24.0, offsets=offsets)
            loaded, loaded_motion, fps = pym_geometry.Character.load_bvh_with_motion(
                path
            )

        self.assertAlmostEqual(fps, 24.0, places=2)
        self.assertEqual(loaded_motion.shape[0], 1)
        self._expect_same_joint_transforms(
            _joint_states(character, offsets[np.newaxis, :]),
            _model_states(loaded, loaded_motion),
        )
