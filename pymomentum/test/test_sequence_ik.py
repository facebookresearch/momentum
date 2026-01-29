# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.diff_geometry as pym_diff_geometry
import pymomentum.geometry as pym_geometry
import pymomentum.solver as pym_solver
import torch
from pymomentum.solver import ErrorFunctionType

AUTOGRAD_ENABLED: bool = pym_geometry.AUTOGRAD_ENABLED


class TestSolver(unittest.TestCase):
    def test_sequence_ik_basic(self) -> None:
        """Test solve_sequence_ik() with multiple poses and one scale to see if we can fit:"""

        # The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
        # and a few dummy faces.
        character = pym_geometry.create_test_character()

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        n_frames = 3

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_target = torch.rand(n_frames, n_params, dtype=torch.float64)

        # Make sure the scaling parameters match up between all the frames:
        scaling_params = torch.from_numpy(
            character.parameter_transform.scaling_parameters
        )
        avg_parameters = model_params_target.mean(0)
        model_params_target[:, scaling_params] = avg_parameters.unsqueeze(0).expand_as(
            model_params_target
        )[:, scaling_params]

        pos_cons_parents = torch.arange(0, n_joints, 1)
        pos_cons_offsets = torch.zeros(n_frames, n_joints, 3)
        pos_cons_targets = pym_diff_geometry.model_parameters_to_positions(
            character, model_params_target, pos_cons_parents, pos_cons_offsets
        ).detach()

        orient_cons_parents = torch.arange(n_joints)
        skel_state_init = pym_diff_geometry.model_parameters_to_skeleton_state(
            character, model_params_target
        )
        orient_cons_targets = skel_state_init.index_select(1, orient_cons_parents)[
            :, :, 3:7
        ]

        def get_positions(
            model_params: torch.Tensor,
            character: pym_geometry.Character = character,
            pos_cons_parents: torch.Tensor = pos_cons_parents,
            pos_cons_offsets: torch.Tensor = pos_cons_offsets,
        ) -> torch.Tensor:
            return pym_diff_geometry.model_parameters_to_positions(
                character, model_params, pos_cons_parents, pos_cons_offsets
            ).detach()

        active_error_functions = [
            ErrorFunctionType.Position,
            ErrorFunctionType.Orientation,
        ]
        error_function_weights = 100 * torch.ones(
            n_frames,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        # start from all zeros:
        model_params_init = torch.zeros(n_frames, n_params)

        # Test whether ik works without proj or dist constraints:
        model_params_final = pym_solver.solve_sequence_ik(
            character=character,
            active_parameters=torch.from_numpy(
                character.parameter_transform.all_parameters
            ),
            shared_parameters=scaling_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_targets=pos_cons_targets,
            orientation_cons_parents=orient_cons_parents,
            orientation_cons_targets=orient_cons_targets,
        ).detach()

        # We can't really verify the joint angles because they're nonunique, but we can check that it reached the target poses:
        self.assertTrue(
            get_positions(model_params_final).allclose(
                get_positions(model_params_target)
            )
        )

        # Also the scale _should_ be unique so verify we solved for it correctly:
        self.assertTrue(
            model_params_final[:, scaling_params].allclose(
                model_params_target[:, scaling_params], atol=1e-5
            )
        )


if __name__ == "__main__":
    unittest.main()
