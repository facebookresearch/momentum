# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import torch
from pymomentum.geometry import (
    Character,
    Joint,
    Locator,
    PARAMETERS_PER_JOINT,
    ParameterTransform,
    Skeleton,
    uniform_random_to_model_parameters,
)
from pymomentum.geometry_test_utils import create_test_character


class TestParameterTransform(unittest.TestCase):
    def test_get_transform(self) -> None:
        character: Character = create_test_character()
        transform = character.parameter_transform.transform
        self.assertEqual(tuple(transform.shape), (3 * 7, 10))

        np.random.seed(42)
        model_params = uniform_random_to_model_parameters(
            character, np.random.rand(1, 10).astype(np.float32)
        ).squeeze()
        joint_params_1 = torch.from_numpy(
            character.parameter_transform.apply(model_params[None, :])
        ).flatten()
        # transform is now a numpy array, so convert to tensor for matmul
        joint_params_2 = torch.matmul(
            torch.from_numpy(transform), torch.from_numpy(model_params)
        )
        self.assertTrue(torch.allclose(joint_params_1, joint_params_2))

    def test_parameter_transform_round_trip(self) -> None:
        """Test that converting parameter transform to dense matrix and back is lossless."""
        # Create a test character with parameter transform
        character: Character = create_test_character()
        original_pt = character.parameter_transform

        # Get the transform as a dense matrix (already a numpy array)
        transform_numpy = original_pt.transform

        # Create a new parameter transform from the dense matrix
        new_pt = ParameterTransform(
            names=original_pt.names,
            skeleton=character.skeleton,
            transform=transform_numpy,
        )

        # Verify the transform produces identical results
        np.random.seed(42)
        model_params = uniform_random_to_model_parameters(
            character, np.random.rand(5, 10).astype(np.float32)
        )

        joint_params_original = torch.from_numpy(original_pt.apply(model_params))
        joint_params_new = torch.from_numpy(new_pt.apply(model_params))

        self.assertTrue(
            torch.allclose(joint_params_original, joint_params_new, atol=1e-6)
        )

        # Verify parameter names are preserved
        self.assertEqual(original_pt.names, new_pt.names)

        # Verify size is preserved
        self.assertEqual(original_pt.size, new_pt.size)

    def test_active_joint_params_initialized_dense(self) -> None:
        """Test that activeJointParams is initialized when constructing
        ParameterTransform from a dense numpy array."""
        character: Character = create_test_character()
        original_pt = character.parameter_transform

        new_pt = ParameterTransform(
            names=original_pt.names,
            skeleton=character.skeleton,
            transform=original_pt.transform,
        )

        # pose_parameters requires activeJointParams to be properly sized;
        # if it were empty, the solver would crash with an Eigen assertion.
        pose_params = new_pt.pose_parameters
        self.assertEqual(len(pose_params), new_pt.size)
        self.assertTrue(pose_params.any())

    def test_active_joint_params_single_joint(self) -> None:
        """Test that a 1-joint character built from Python can be used with
        the marker tracking solver (regression test for activeJointParams
        not being initialized in the Python ParameterTransform constructor)."""
        from pymomentum.marker_tracking import (
            CalibrationConfig,
            process_markers,
            TrackingConfig,
        )

        root_joint = Joint(
            name="root",
            parent=-1,
            pre_rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            translation_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        skeleton = Skeleton(joint_list=[root_joint])
        n_joint_params = PARAMETERS_PER_JOINT * skeleton.size
        transform = np.eye(n_joint_params, dtype=np.float32)
        names = [
            "root_tx",
            "root_ty",
            "root_tz",
            "root_rx",
            "root_ry",
            "root_rz",
            "scale_global",
        ]
        pt = ParameterTransform(names=names, skeleton=skeleton, transform=transform)

        character = Character(
            name="test_rigid",
            skeleton=skeleton,
            parameter_transform=pt,
        )
        locators = [
            Locator(
                name=f"m{i}",
                parent=0,
                offset=np.array(
                    [float(i), 0.0, 0.0],
                    dtype=np.float32,
                ),
                weight=1.0,
                locked=np.array([1, 1, 1], dtype=np.int32),
            )
            for i in range(4)
        ]
        character = character.with_locators(locators)

        # Build synthetic marker data: 4 markers, 5 frames, at known positions
        from pymomentum.geometry import Marker

        marker_data = []
        for _frame in range(5):
            frame_markers = []
            for i in range(4):
                frame_markers.append(
                    Marker(
                        name=f"m{i}",
                        pos=np.array(
                            [float(i) + 10.0, 5.0, 0.0],
                            dtype=np.float32,
                        ),
                        occluded=False,
                    )
                )
            marker_data.append(frame_markers)

        identity = np.zeros(pt.size, dtype=np.float64)
        tracking_config = TrackingConfig(max_iter=5, regularization=0.1)
        calibration_config = CalibrationConfig(
            max_iter=5,
            calib_frames=5,
            locators_only=False,
        )

        # This would crash with "Assertion index >= 0 && index < size()"
        # if activeJointParams was not initialized.
        motion = process_markers(
            character,
            identity,
            marker_data,
            tracking_config,
            calibration_config,
            calibrate=True,
            max_frames=5,
        )
        self.assertEqual(motion.shape[0], 5)
        self.assertEqual(motion.shape[1], pt.size)


if __name__ == "__main__":
    unittest.main()
