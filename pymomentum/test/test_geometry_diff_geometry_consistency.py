# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Consistency tests between pymomentum.geometry and pymomentum.diff_geometry.

This file contains tests that verify that the numpy-based implementations in
pymomentum.geometry produce the same results as the torch-based implementations
in pymomentum.diff_geometry.
"""

import unittest

import numpy as np
import pymomentum.diff_geometry as pym_diff_geometry
import pymomentum.geometry as pym_geometry
import torch


class TestGeometryDiffGeometryConsistency(unittest.TestCase):
    """Tests verifying consistency between geometry and diff_geometry modules."""

    def test_apply_parameter_transform_matches(self) -> None:
        """Verify that geometry.apply_parameter_transform matches diff_geometry.apply_parameter_transform."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2

        # Create model parameters with numpy
        model_params_np = np.random.rand(
            nBatch, character.parameter_transform.size
        ).astype(np.float32)

        # Call geometry version (numpy)
        joint_params_geometry = pym_geometry.apply_parameter_transform(
            character, model_params_np
        )

        # Call diff_geometry version (torch)
        model_params_torch = torch.from_numpy(model_params_np)
        joint_params_diff_geometry = pym_diff_geometry.apply_parameter_transform(
            character, model_params_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                joint_params_geometry, joint_params_diff_geometry.numpy(), atol=1e-6
            ),
            "geometry.apply_parameter_transform should match diff_geometry.apply_parameter_transform",
        )

    def test_model_parameters_to_skeleton_state_matches(self) -> None:
        """Verify that geometry.model_parameters_to_skeleton_state matches diff_geometry.model_parameters_to_skeleton_state."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2

        # Create model parameters with numpy
        model_params_np = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size), dtype=np.float32
        )

        # Call geometry version (numpy)
        skel_state_geometry = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_np
        )

        # Call diff_geometry version (torch)
        model_params_torch = torch.from_numpy(model_params_np)
        skel_state_diff_geometry = pym_diff_geometry.model_parameters_to_skeleton_state(
            character, model_params_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                skel_state_geometry, skel_state_diff_geometry.numpy(), atol=1e-6
            ),
            "geometry.model_parameters_to_skeleton_state should match diff_geometry.model_parameters_to_skeleton_state",
        )

    def test_model_parameters_to_local_skeleton_state_matches(self) -> None:
        """Verify that geometry.model_parameters_to_local_skeleton_state matches diff_geometry.model_parameters_to_local_skeleton_state."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2

        # Create model parameters with numpy
        model_params_np = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size), dtype=np.float32
        )

        # Call geometry version (numpy)
        local_skel_state_geometry = (
            pym_geometry.model_parameters_to_local_skeleton_state(
                character, model_params_np
            )
        )

        # Call diff_geometry version (torch)
        model_params_torch = torch.from_numpy(model_params_np)
        local_skel_state_diff_geometry = (
            pym_diff_geometry.model_parameters_to_local_skeleton_state(
                character, model_params_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                local_skel_state_geometry,
                local_skel_state_diff_geometry.numpy(),
                atol=1e-6,
            ),
            "geometry.model_parameters_to_local_skeleton_state should match diff_geometry.model_parameters_to_local_skeleton_state",
        )

    def test_joint_parameters_to_skeleton_state_matches(self) -> None:
        """Verify that geometry.joint_parameters_to_skeleton_state matches diff_geometry.joint_parameters_to_skeleton_state."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size

        # Create joint parameters with numpy (flat format)
        joint_params_np = np.random.rand(nBatch, nJoints * 7).astype(np.float32)

        # Call geometry version (numpy)
        skel_state_geometry = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params_np
        )

        # Call diff_geometry version (torch)
        joint_params_torch = torch.from_numpy(joint_params_np)
        skel_state_diff_geometry = pym_diff_geometry.joint_parameters_to_skeleton_state(
            character, joint_params_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                skel_state_geometry, skel_state_diff_geometry.numpy(), atol=1e-6
            ),
            "geometry.joint_parameters_to_skeleton_state should match diff_geometry.joint_parameters_to_skeleton_state",
        )

    def test_joint_parameters_to_local_skeleton_state_matches(self) -> None:
        """Verify that geometry.joint_parameters_to_local_skeleton_state matches diff_geometry.joint_parameters_to_local_skeleton_state."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size

        # Create joint parameters with numpy (flat format)
        joint_params_np = np.random.rand(nBatch, nJoints * 7).astype(np.float32)

        # Call geometry version (numpy)
        local_skel_state_geometry = (
            pym_geometry.joint_parameters_to_local_skeleton_state(
                character, joint_params_np
            )
        )

        # Call diff_geometry version (torch)
        joint_params_torch = torch.from_numpy(joint_params_np)
        local_skel_state_diff_geometry = (
            pym_diff_geometry.joint_parameters_to_local_skeleton_state(
                character, joint_params_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                local_skel_state_geometry,
                local_skel_state_diff_geometry.numpy(),
                atol=1e-6,
            ),
            "geometry.joint_parameters_to_local_skeleton_state should match diff_geometry.joint_parameters_to_local_skeleton_state",
        )

    def test_skeleton_state_to_joint_parameters_matches(self) -> None:
        """Verify that geometry.skeleton_state_to_joint_parameters matches diff_geometry.skeleton_state_to_joint_parameters."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2

        # Create model parameters and convert to skeleton state
        model_params_np = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size), dtype=np.float32
        )
        skel_state_np = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_np
        )

        # Call geometry version (numpy)
        joint_params_geometry = pym_geometry.skeleton_state_to_joint_parameters(
            character, skel_state_np
        )

        # Call diff_geometry version (torch)
        skel_state_torch = torch.from_numpy(skel_state_np)
        joint_params_diff_geometry = (
            pym_diff_geometry.skeleton_state_to_joint_parameters(
                character, skel_state_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                joint_params_geometry, joint_params_diff_geometry.numpy(), atol=1e-6
            ),
            "geometry.skeleton_state_to_joint_parameters should match diff_geometry.skeleton_state_to_joint_parameters",
        )

    def test_local_skeleton_state_to_joint_parameters_matches(self) -> None:
        """Verify that geometry.local_skeleton_state_to_joint_parameters matches diff_geometry.local_skeleton_state_to_joint_parameters."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2

        # Create model parameters and convert to local skeleton state
        model_params_np = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size), dtype=np.float32
        )
        local_skel_state_np = pym_geometry.model_parameters_to_local_skeleton_state(
            character, model_params_np
        )

        # Call geometry version (numpy)
        joint_params_geometry = pym_geometry.local_skeleton_state_to_joint_parameters(
            character, local_skel_state_np
        )

        # Call diff_geometry version (torch)
        local_skel_state_torch = torch.from_numpy(local_skel_state_np)
        joint_params_diff_geometry = (
            pym_diff_geometry.local_skeleton_state_to_joint_parameters(
                character, local_skel_state_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                joint_params_geometry, joint_params_diff_geometry.numpy(), atol=1e-6
            ),
            "geometry.local_skeleton_state_to_joint_parameters should match diff_geometry.local_skeleton_state_to_joint_parameters",
        )


if __name__ == "__main__":
    unittest.main()
