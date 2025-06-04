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
        l_character, motion, fps = pym_geometry.Character.load_fbx_with_motion(
            file_name
        )
        self.assertEqual(1, len(motion))
        self.assertEqual(motion[0].shape, self.joint_params.shape)
        self.assertEqual(fps, 60)
