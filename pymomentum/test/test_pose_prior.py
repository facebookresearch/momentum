# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import unittest

import pymomentum.geometry as pym_geometry
import pymomentum.solver as pym_solver
import torch
from pymomentum.solver import ErrorFunctionType

logger: logging.Logger = logging.getLogger(__name__)

# Flag to check if autograd is enabled (disabled in arvr build modes)
AUTOGRAD_ENABLED: bool = pym_geometry.AUTOGRAD_ENABLED


class TestPosePrior(unittest.TestCase):
    def test_ik_pose_prior(self) -> None:
        """Test solve_ik() with just the pose prior and positions."""

        # The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
        # and a few dummy faces.
        character = pym_geometry.create_test_character()

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        batch_size = 2

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(batch_size, n_params, dtype=torch.float64)

        # =============== Position constraints:
        n_pos_cons = 4 * n_joints
        pos_cons_parents = torch.randint(
            low=0,
            high=character.skeleton.size,
            size=[n_pos_cons],
            dtype=torch.float64,
        )
        pos_cons_offsets = torch.normal(
            mean=0, std=4, size=(n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_offsets.requires_grad = AUTOGRAD_ENABLED
        pos_cons_targets = pym_geometry.model_parameters_to_positions(
            character, model_params_init, pos_cons_parents, pos_cons_offsets
        ).detach() + torch.normal(
            mean=0, std=1, size=(batch_size, n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_targets.requires_grad = AUTOGRAD_ENABLED
        pos_cons_weights = torch.ones(
            batch_size, n_pos_cons, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )

        orient_cons_parents = torch.arange(n_joints)
        orient_cons_weights = torch.ones(
            batch_size, n_joints, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )
        skel_state_init = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_init
        )
        orient_cons_targets = skel_state_init.index_select(1, orient_cons_parents)[
            :, :, 3:7
        ]

        n_modes = 1
        posePrior_pi = torch.ones(n_modes, dtype=torch.float64)
        posePrior_pi.requires_grad = AUTOGRAD_ENABLED
        posePrior_mu = torch.normal(
            mean=0, std=1, size=(n_modes, n_params), dtype=torch.float64
        )
        posePrior_mu.requires_grad = AUTOGRAD_ENABLED
        posePrior_W = torch.normal(
            mean=0, std=1, size=(n_modes, 2, n_params), dtype=torch.float64
        )
        posePrior_W.requires_grad = AUTOGRAD_ENABLED
        posePrior_sigma = torch.ones(n_modes, dtype=torch.float64)
        posePrior_sigma.requires_grad = AUTOGRAD_ENABLED
        posePrior_parameterIndices = torch.arange(0, n_params)

        # =============== End building constraints

        scaling_params = character.parameter_transform.scaling_parameters
        active_params = ~scaling_params

        active_error_functions = [
            ErrorFunctionType.Limit,
            ErrorFunctionType.Position,
            ErrorFunctionType.Orientation,
            ErrorFunctionType.PosePrior,
        ]
        error_function_weights = torch.ones(
            batch_size,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        def solve_ik(
            modelParameters_init: torch.Tensor,
            errf_weights: torch.Tensor,
            pc_parents: torch.Tensor,
            pc_offsets: torch.Tensor,
            pc_weights: torch.Tensor,
            pc_targets: torch.Tensor,
            pp_pi: torch.Tensor,
            pp_mu: torch.Tensor,
            pp_W: torch.Tensor,
            pp_sigma: torch.Tensor,
            character: pym_geometry.Character = character,
            active_params: torch.Tensor = active_params,
            active_error_functions: list[ErrorFunctionType] = active_error_functions,
            orient_cons_parents: torch.Tensor = orient_cons_parents,
            orient_cons_targets: torch.Tensor = orient_cons_targets,
            orient_cons_weights: torch.Tensor = orient_cons_weights,
            pos_cons_parents: torch.Tensor = pos_cons_parents,
            pos_cons_offsets: torch.Tensor = pos_cons_offsets,
            pos_cons_weights: torch.Tensor = pos_cons_weights,
            pos_cons_targets: torch.Tensor = pos_cons_targets,
            posePrior_parameterIndices: torch.Tensor = posePrior_parameterIndices,
            model_params_init: torch.Tensor = model_params_init,
        ) -> torch.Tensor:
            return pym_solver.solve_ik(
                character=character,
                active_parameters=active_params,
                model_parameters_init=model_params_init,
                active_error_functions=active_error_functions,
                error_function_weights=errf_weights,
                position_cons_parents=pos_cons_parents,
                position_cons_offsets=pos_cons_offsets,
                position_cons_weights=pos_cons_weights,
                position_cons_targets=pos_cons_targets,
                orientation_cons_parents=orient_cons_parents,
                orientation_cons_weights=orient_cons_weights,
                orientation_cons_targets=orient_cons_targets,
                pose_prior_model=[
                    pp_pi,
                    pp_mu,
                    pp_W,
                    pp_sigma,
                    posePrior_parameterIndices,
                ],
            )

        # Test whether ik works without proj or dist constraints:
        model_params_final = solve_ik(
            model_params_init,
            error_function_weights,
            pos_cons_parents,
            pos_cons_offsets,
            pos_cons_weights,
            pos_cons_targets,
            posePrior_pi,
            posePrior_mu,
            posePrior_W,
            posePrior_sigma,
        )

        solverOptions = pym_solver.SolverOptions()
        solverOptions.linear_solver = pym_solver.LinearSolverType.QR
        solverOptions.line_search = False

        inputs = [
            model_params_final,
            error_function_weights,
            pos_cons_parents,
            pos_cons_offsets,
            pos_cons_weights,
            pos_cons_targets,
            posePrior_pi,
            posePrior_mu,
            posePrior_W,
            posePrior_sigma,
        ]
        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                solve_ik,
                inputs,
                eps=0.0001,
                atol=1e-1,
                rtol=1e-3,
                raise_exception=True,
            )

            loss_func = torch.nn.MSELoss()
            result = loss_func(model_params_final, model_params_init)
            result.backward()

            (nNonZero, nTotal) = pym_solver.get_gradient_statistics()
            self.assertEqual(nNonZero, 0)
            self.assertGreater(nTotal, 0)


if __name__ == "__main__":
    unittest.main()
