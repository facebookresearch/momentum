# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from multiprocessing.dummy import Pool

import pymomentum.diff_geometry as pym_diff_geometry
import pymomentum.geometry as pym_geometry
import pymomentum.skel_state as pym_skel_state
import pymomentum.solver as pym_solver
import torch
from pymomentum.solver import ErrorFunctionType

# Flag to check if autograd is enabled (disabled in arvr build modes)
AUTOGRAD_ENABLED: bool = pym_solver.AUTOGRAD_ENABLED


def solve_one_ik_problem(index: int) -> torch.Tensor:
    character = pym_geometry.create_test_character()

    n_joints = character.skeleton.size
    n_params = character.parameter_transform.size

    batch_size = 2

    # Ensure repeatability in the rng:
    torch.manual_seed(0)
    model_params_init = torch.zeros(batch_size, n_params, dtype=torch.float64)
    model_params_target = torch.linspace(start=-1, end=1, steps=n_params)

    # =============== Position constraints:
    pos_cons_parents = torch.arange(0, n_joints)
    pos_cons_offsets = torch.zeros(size=(n_joints, 3), dtype=torch.float64)

    pos_cons_targets = pym_geometry.model_parameters_to_positions(
        character, model_params_target, pos_cons_parents, pos_cons_offsets
    ).detach()

    # TODO randomize
    pos_cons_weights = torch.ones(
        batch_size, n_joints, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
    )
    pos_cons_weights.requires_grad = AUTOGRAD_ENABLED

    active_error_functions = [
        ErrorFunctionType.Limit,
        ErrorFunctionType.Position,
    ]
    error_function_weights = torch.ones(
        batch_size,
        len(active_error_functions),
        requires_grad=AUTOGRAD_ENABLED,
        dtype=torch.float64,
    )

    # Test whether ik works without proj or dist constraints:
    return pym_solver.solve_ik(
        character=character,
        active_parameters=character.parameter_transform.all_parameters,
        model_parameters_init=model_params_init,
        active_error_functions=active_error_functions,
        error_function_weights=error_function_weights,
        position_cons_parents=pos_cons_parents,
        position_cons_weights=pos_cons_weights,
        position_cons_targets=pos_cons_targets,
    )


class TestSolver(unittest.TestCase):
    def test_ik_basic(self) -> None:
        """Test solve_ik() with basic constraints:
        - position
        - orientation
        - motion
        """

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

        # TODO randomize
        pos_cons_weights = torch.ones(
            batch_size, n_pos_cons, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )

        # =============== Orientation constraints:
        orient_cons_parents = torch.arange(n_joints)
        orient_cons_weights = torch.ones(
            batch_size, n_joints, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )

        skel_state_init = pym_diff_geometry.model_parameters_to_skeleton_state(
            character, model_params_init
        )
        orient_cons_targets = skel_state_init.index_select(1, orient_cons_parents)[
            :, :, 3:7
        ]

        # =============== Motion constraints:
        motion_targets = model_params_init.detach().clone().double()
        # motion_targets.requires_grad = AUTOGRAD_ENABLED
        motion_weights = torch.ones(
            batch_size, n_params, dtype=torch.float64, requires_grad=AUTOGRAD_ENABLED
        )

        # =============== End building constraints

        scaling_params = character.parameter_transform.scaling_parameters
        active_params = ~scaling_params

        active_error_functions = [
            ErrorFunctionType.Limit,
            ErrorFunctionType.Position,
            ErrorFunctionType.Orientation,
            ErrorFunctionType.Motion,
        ]
        error_function_weights = torch.ones(
            batch_size,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        # Test whether ik works without proj or dist constraints:
        model_params_final = pym_solver.solve_ik(
            character=character,
            active_parameters=active_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_weights=pos_cons_weights,
            position_cons_targets=pos_cons_targets,
            orientation_cons_parents=orient_cons_parents,
            orientation_cons_weights=orient_cons_weights,
            orientation_cons_targets=orient_cons_targets,
            motion_targets=motion_targets,
            motion_weights=motion_weights,
        )

        # print ("Dumping problem to messagepack...")
        # with open("/home/cdtwigg/problem.msgpack", "wb") as f:
        #   iBatch = 0
        #   problem = { "model_params_init": model_params_init.select(0, iBatch).tolist(),
        #               "pos_cons_parents": pos_cons_parents.flatten().tolist(),
        #               "pos_cons_offsets": pos_cons_offsets.flatten().tolist(),
        #               "pos_cons_weights": pos_cons_weights.select(0, iBatch).flatten().tolist(),
        #               "pos_cons_targets": pos_cons_targets.select(0, iBatch).flatten().tolist(),
        #               "orient_cons_parents": orient_cons_parents.flatten().tolist(),
        #               "orient_cons_weights": orient_cons_weights.select(0, iBatch).flatten().tolist(),
        #               "orient_cons_targets": orient_cons_targets.select(0, iBatch).flatten().tolist(),
        #               "error_function_weights": error_function_weights.select(0, iBatch).flatten().tolist(),
        #               "motion_targets": motion_targets.select(0, iBatch).flatten().tolist(),
        #               "motion_weights": motion_weights.select(0, iBatch).flatten().tolist(),
        #               "model_params_final": model_params_final.select(0, iBatch).flatten().tolist()}
        #   msgpack.pack(problem, f)

        solverOptions = pym_solver.SolverOptions()
        solverOptions.linear_solver = pym_solver.LinearSolverType.QR
        solverOptions.line_search = False

        inputs = [
            character,
            active_params,
            model_params_final.detach(),
            active_error_functions,
            error_function_weights,
            solverOptions,
            pos_cons_parents,
            pos_cons_offsets,
            pos_cons_weights,
            pos_cons_targets,
            orient_cons_parents,
            None,  # orientation offsets
            orient_cons_weights,
            orient_cons_targets,
            None,  # pose prior model
            motion_targets,
            motion_weights,
        ]
        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                pym_solver.solve_ik,
                inputs,
                eps=0.001,
                atol=1e-3,
                raise_exception=True,
            )

            loss_func = torch.nn.MSELoss()
            result = loss_func(model_params_final, model_params_init)
            result.backward()

            (nNonZero, nTotal) = pym_solver.get_gradient_statistics()
            self.assertEqual(nNonZero, 0)
            self.assertGreater(nTotal, 0)

    def test_ik_multithreaded(self) -> None:
        n_workers = 3
        n_jobs = 100
        with Pool(n_workers) as pool:
            result: list[torch.Tensor] = pool.map(
                solve_one_ik_problem, range(0, n_jobs)
            )
            self.assertEqual(len(result), n_jobs)
            for r in result:
                self.assertEqual(r.shape, result[0].shape)
                self.assertTrue(r.equal(result[0]))

    def test_ik_nans(self) -> None:
        """Make sure IK handles NaNs correctly."""

        # The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
        # and a few dummy faces.
        character = pym_geometry.create_test_character()

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        batch_size = 1

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(batch_size, n_params, dtype=torch.float64)

        # unsqueeze one dim for n_cam
        pos_cons_parents = torch.arange(n_joints).unsqueeze(0)
        pos_cons_targets = torch.ones(n_joints, 3)
        pos_cons_targets[0, 0] = float("nan")

        result = pym_solver.solve_ik(
            character=character,
            active_parameters=character.parameter_transform.all_parameters,
            model_parameters_init=model_params_init,
            active_error_functions=[ErrorFunctionType.Position],
            error_function_weights=torch.ones(batch_size, 1, dtype=torch.float64),
            position_cons_parents=pos_cons_parents,
            position_cons_targets=pos_cons_targets,
        )

        self.assertTrue(result.allclose(model_params_init, atol=1e-8))

    def test_gradient(self) -> None:
        character = pym_geometry.create_test_character()

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        n_pos_cons = 4 * n_joints
        batch_size = 3

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params = torch.zeros(
            batch_size, n_params, dtype=torch.float64, requires_grad=AUTOGRAD_ENABLED
        )

        # Position constraints:
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
            character, model_params, pos_cons_parents, pos_cons_offsets
        ).detach() + torch.normal(
            mean=0, std=1, size=(batch_size, n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_targets.requires_grad = AUTOGRAD_ENABLED

        # TODO randomize
        pos_cons_weights = torch.ones(
            batch_size, n_pos_cons, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )

        motion_targets = model_params.detach().clone()
        motion_weights = torch.ones(batch_size, n_params, dtype=torch.float64)
        active_error_functions = [
            ErrorFunctionType.Limit,
            ErrorFunctionType.Position,
            ErrorFunctionType.Motion,
        ]
        error_function_weights = torch.ones(
            batch_size,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        pym_solver.gradient(
            character=character,
            model_parameters=model_params,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_weights=pos_cons_weights,
            position_cons_targets=pos_cons_targets,
            motion_targets=motion_targets,
            motion_weights=motion_weights,
        )

        inputs = [
            character,
            model_params,
            active_error_functions,
            error_function_weights,
            pos_cons_parents,
            pos_cons_offsets,
            pos_cons_weights,
            pos_cons_targets,
            None,  # orientation parents
            None,  # orientation offsets
            None,  # orientation weights
            None,  # orientation targets
            None,  # pose prior model
            motion_targets,
            motion_weights,
        ]
        # I believe we have to use a slightly larger atol here than we would otherwise like because
        # the approximate d/dTheta (dErr/dTheta) = J^T * J drops the higher-order terms.
        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                pym_solver.gradient,
                inputs,
                eps=1e-4,
                atol=1e-2,
                raise_exception=True,
            )

    def test_residual(self) -> None:
        character = pym_geometry.create_test_character()

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        n_pos_cons = 4 * n_joints
        batch_size = 3

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params = torch.zeros(
            batch_size, n_params, dtype=torch.float64, requires_grad=AUTOGRAD_ENABLED
        )

        # Position constraints:
        pos_cons_parents = torch.randint(
            low=0,
            high=character.skeleton.size,
            size=[n_pos_cons],
            dtype=torch.float64,
        )
        pos_cons_offsets = torch.normal(
            mean=0, std=4, size=(n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_offsets.requires_grad = False

        pos_cons_targets = pym_geometry.model_parameters_to_positions(
            character, model_params, pos_cons_parents, pos_cons_offsets
        ).detach() + torch.normal(
            mean=0, std=1, size=(batch_size, n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_targets.requires_grad = False

        # We don't currently compute derivatives of the residual wrt the error function
        # weights, although we could consider adding this in the future.
        pos_cons_weights = torch.ones(
            batch_size, n_pos_cons, requires_grad=False, dtype=torch.float64
        )

        motion_targets = model_params.detach().clone()
        motion_weights = torch.ones(batch_size, n_params, dtype=torch.float64)
        active_error_functions = [
            ErrorFunctionType.Limit,
            ErrorFunctionType.Position,
            ErrorFunctionType.Motion,
        ]
        error_function_weights = torch.ones(
            batch_size,
            len(active_error_functions),
            requires_grad=False,
            dtype=torch.float64,
        )

        grad = pym_solver.gradient(
            character=character,
            model_parameters=model_params,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_weights=pos_cons_weights,
            position_cons_targets=pos_cons_targets,
            motion_targets=motion_targets,
            motion_weights=motion_weights,
        )

        (residual, jacobian) = pym_solver.jacobian(
            character=character,
            model_parameters=model_params,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_weights=pos_cons_weights,
            position_cons_targets=pos_cons_targets,
            motion_targets=motion_targets,
            motion_weights=motion_weights,
        )

        grad2 = 2 * torch.bmm(residual.unsqueeze(-2), jacobian).squeeze(-2)
        self.assertTrue(grad2.allclose(grad))

        inputs = [
            character,
            model_params,
            active_error_functions,
            error_function_weights,
            pos_cons_parents,
            pos_cons_offsets,
            pos_cons_weights,
            pos_cons_targets,
            None,  # orientation parents
            None,  # orientation offsets
            None,  # orientation weights
            None,  # orientation targets
            None,  # pose prior model
            motion_targets,
            motion_weights,
        ]

        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                pym_solver.residual,
                inputs,
                raise_exception=True,
            )

    def test_nonzero_gradient(self) -> None:
        """Test to make sure the gradient is nonzero check triggers."""

        pym_solver.reset_gradient_statistics()

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
        pos_cons_targets = torch.normal(
            mean=0, std=5, size=(batch_size, n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_weights = torch.ones(
            batch_size, n_pos_cons, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )
        active_error_functions = [
            ErrorFunctionType.Limit,
            ErrorFunctionType.Position,
        ]
        error_function_weights = torch.ones(
            batch_size,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        # Make absolutely sure it doesn't converge:
        solverOptions = pym_solver.SolverOptions()
        solverOptions.min_iter = 0
        solverOptions.max_iter = 0

        scaling_params = character.parameter_transform.scaling_parameters
        active_params = ~scaling_params

        model_params_final = pym_solver.solve_ik(
            options=solverOptions,
            character=character,
            active_parameters=active_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_weights=5.0 * pos_cons_weights,
            position_cons_targets=pos_cons_targets,
        )

        if AUTOGRAD_ENABLED:
            loss_func = torch.nn.MSELoss()
            result = loss_func(model_params_final, model_params_init)
            result.backward()

            (nNonZero, nTotal) = pym_solver.get_gradient_statistics()
            self.assertGreater(nNonZero, 0)

            # Because of nonzero gradient, all error function gradients should be 0:
            self.assertTrue(torch.norm(pos_cons_weights.grad) == 0)
            self.assertTrue(torch.norm(error_function_weights.grad) == 0)

    def test_solver_stats(self) -> None:
        """Test solver stats."""

        pym_solver.reset_solve_ik_statistics()

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
        pos_cons_targets = torch.normal(
            mean=0, std=5, size=(batch_size, n_pos_cons, 3), dtype=torch.float64
        )
        pos_cons_weights = torch.ones(
            batch_size, n_pos_cons, requires_grad=AUTOGRAD_ENABLED, dtype=torch.float64
        )
        active_error_functions = [
            ErrorFunctionType.Limit,
            ErrorFunctionType.Position,
        ]
        error_function_weights = torch.ones(
            batch_size,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        # Make sure it only has 3 iter
        solverOptions = pym_solver.SolverOptions()
        solverOptions.min_iter = 3
        solverOptions.max_iter = 3
        # prevent solver from thinking it converges
        solverOptions.threshold = 0.0

        scaling_params = character.parameter_transform.scaling_parameters
        active_params = ~scaling_params

        pym_solver.solve_ik(
            options=solverOptions,
            character=character,
            active_parameters=active_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            position_cons_parents=pos_cons_parents,
            position_cons_offsets=pos_cons_offsets,
            position_cons_weights=5.0 * pos_cons_weights,
            position_cons_targets=pos_cons_targets,
        )

        n_solve, n_solve_iter = pym_solver.get_solve_ik_statistics()
        self.assertEqual(n_solve, 2)
        self.assertEqual(n_solve_iter, 6)

    def test_transform_pose(self) -> None:
        character = pym_geometry.create_test_character()
        torch.manual_seed(0)  # ensure repeatability

        nBatch = 5
        nParams = character.parameter_transform.size
        model_params_init = pym_geometry.uniform_random_to_model_parameters(
            character, torch.rand(nBatch, nParams)
        ).double()

        skel_state_init = pym_diff_geometry.model_parameters_to_skeleton_state(
            character, model_params_init
        )

        def remove_scale(skel_state: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                tensors=[skel_state[..., :7], torch.ones_like(skel_state[..., 7:8])],
                dim=-1,
            )

        # Transform the leaf node to the origin:
        transformed_joint = -1
        transform = pym_skel_state.inverse(
            remove_scale(skel_state_init[:, transformed_joint])
        )
        expected_final_transform = pym_skel_state.multiply(
            transform, remove_scale(skel_state_init[:, transformed_joint])
        )
        model_params_final = pym_solver.transform_pose(
            character, model_params_init, transform
        )
        skel_state_final = pym_diff_geometry.model_parameters_to_skeleton_state(
            character, model_params_final
        )

        expected_matrices_final = pym_skel_state.to_matrix(expected_final_transform)
        matrices_final = pym_skel_state.to_matrix(
            remove_scale(skel_state_final[:, transformed_joint])
        )
        self.assertTrue(torch.allclose(matrices_final, expected_matrices_final))

    def test_projection(self) -> None:
        """Test solve_ik() with projection constraints."""

        # The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
        # and a few dummy faces.
        character = pym_geometry.create_test_character()

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros((1, n_params), dtype=torch.float64)

        # =============== constraints

        vert_proj_cons_vertex = torch.LongTensor(
            [
                [
                    20,
                ]
            ]
        )
        vert_proj_cons_weights = torch.FloatTensor(
            [
                [
                    1.0,
                ]
            ]
        )
        vert_proj_cons_weights.requires_grad = AUTOGRAD_ENABLED
        vert_proj_cons_target_positions = torch.FloatTensor([[1.0, 1.0]])
        vert_proj_cons_target_positions.requires_grad = AUTOGRAD_ENABLED
        vert_proj_cons_projection = torch.FloatTensor(
            [
                [
                    [
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        10.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        10.0,
                    ],
                ]
            ]
        )
        vert_proj_cons_projection.requires_grad = AUTOGRAD_ENABLED

        # =============== End building constraints

        scaling_params = character.parameter_transform.scaling_parameters
        active_params = ~scaling_params

        active_error_functions = [
            ErrorFunctionType.VertexProjection,
        ]
        error_function_weights = torch.ones(
            1,
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
            dtype=torch.float64,
        )

        # Test whether ik works with proj constraint:
        model_params_final = pym_solver.solve_ik(
            character=character,
            active_parameters=active_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            vertex_proj_cons_vertices=vert_proj_cons_vertex,
            vertex_proj_cons_weights=vert_proj_cons_weights,
            vertex_proj_cons_target_positions=vert_proj_cons_target_positions,
            vertex_proj_cons_projections=vert_proj_cons_projection,
        )

        residual_optimized = pym_solver.residual(
            character=character,
            model_parameters=model_params_final,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            vertex_proj_cons_vertices=vert_proj_cons_vertex,
            vertex_proj_cons_weights=vert_proj_cons_weights,
            vertex_proj_cons_target_positions=vert_proj_cons_target_positions,
            vertex_proj_cons_projections=vert_proj_cons_projection,
        )

        self.assertTrue(residual_optimized[0, :2].norm() < 1e-6)

        # check if the residual is as expected
        residual = pym_solver.residual(
            character=character,
            model_parameters=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            vertex_proj_cons_vertices=vert_proj_cons_vertex,
            vertex_proj_cons_weights=vert_proj_cons_weights,
            vertex_proj_cons_target_positions=vert_proj_cons_target_positions,
            vertex_proj_cons_projections=vert_proj_cons_projection,
        )

        # we expect the residual to be the difference between the target position and the projected position
        vertex_cam = (
            vert_proj_cons_projection[0, :, :3]
            @ torch.FloatTensor(character.mesh.vertices[20])
            + vert_proj_cons_projection[0, :, 3]
        )
        vertex_screen = vertex_cam[:2] / vertex_cam[2]
        vertex_residual = vertex_screen - vert_proj_cons_target_positions[0]

        self.assertTrue(
            torch.allclose(residual[0, :2], vertex_residual.to(torch.double))
        )


if __name__ == "__main__":
    unittest.main()
