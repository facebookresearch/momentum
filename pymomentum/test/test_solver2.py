# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import numpy.typing as npt
import pymomentum.geometry as pym_geometry
import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import pymomentum.solver2 as pym_solver2
import torch


def _normalize_vec(vec: npt.NDArray) -> npt.NDArray:
    return vec / np.linalg.norm(vec)


class TestSolver(unittest.TestCase):
    def test_ik_basic(self) -> None:
        """Test solve_ik() with just position constraints."""

        # The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
        # and a few dummy faces.
        character = pym_geometry.create_test_character(num_joints=4)

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        model_params_target = torch.rand_like(model_params_init)
        skel_state_target = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_target.numpy()
            )
        )

        pos_error = pym_solver2.PositionErrorFunction(character)

        pos_error.add_constraint(
            parent=0,
            offset=np.array([1.0, 0.0, 0.0]),
            target=np.array([10.0, 0.0, 0.0]),
            weight=1.0,
        )
        self.assertTrue(len(pos_error.constraints) == 1)
        self.assertTrue(
            np.isclose(pos_error.constraints[0].offset, [1.0, 0.0, 0.0]).all()
        )
        self.assertTrue(
            np.isclose(pos_error.constraints[0].target, [10.0, 0.0, 0.0]).all()
        )

        pos_error.clear_constraints()
        self.assertTrue(len(pos_error.constraints) == 0)

        pos_error.add_constraints(
            parent=np.arange(n_joints), target=skel_state_target[:, :3].numpy()
        )
        solver_function = pym_solver2.SkeletonSolverFunction(character, [pos_error])

        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        self.assertTrue(
            torch.allclose(
                skel_state_final[:, :3], skel_state_target[:, :3], rtol=1e-5, atol=1e-5
            )
        )

        self.assertGreater(len(solver.per_iteration_errors), 1)
        self.assertLess(solver.per_iteration_errors[-1], solver.per_iteration_errors[0])

        # make sure it's deterministic:
        per_iter_errors_prev = solver.per_iteration_errors
        model_params_final = solver.solve(model_params_init.numpy())
        assert solver.per_iteration_errors == per_iter_errors_prev

        # delete constraints and ensure they're empty
        pos_error.clear_constraints()
        self.assertTrue(len(pos_error.constraints) == 0)

    def test_incorrect_params(self) -> None:
        """Test solve_ik() with incorrect parameter transform."""

        # The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
        # and a few dummy faces.
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [pym_solver2.LimitErrorFunction(character)]
        )
        solver = pym_solver2.GaussNewtonSolver(
            solver_function, pym_solver2.GaussNewtonSolverOptions()
        )
        self.assertRaises(
            RuntimeError, solver.solve, np.zeros(n_params + 1, dtype=np.float32)
        )

        # make sure if we try to combine error functions from different characters, it fails:
        character2 = pym_geometry.create_test_character(num_joints=4)
        self.assertRaises(
            RuntimeError,
            solver_function.add_error_function,
            pym_solver2.LimitErrorFunction(character2),
        )

    def test_get_gradient_and_jacobian(self) -> None:
        """Test that get_gradient and get_jacobian do something reasonable for error functions."""

        character = pym_geometry.create_test_character(num_joints=4)

        n_joints = character.skeleton.size
        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        random_positions = torch.rand(n_joints, 3, dtype=torch.float32)

        pos_error = pym_solver2.PositionErrorFunction(character)

        for i_joint in range(n_joints):
            pos_error.add_constraint(
                parent=i_joint,
                weight=1.0,
                target=random_positions[i_joint, :3].numpy(),
            )

        error = pos_error.get_error(model_params_init.numpy())
        self.assertTrue(error > 0.0)

        # Test get_gradient
        grad = pos_error.get_gradient(model_params_init.numpy())
        eps = 1e-3
        for i_param in range(n_params):
            mp_plus = np.copy(model_params_init)
            mp_plus[i_param] += eps
            grad_est = (pos_error.get_error(mp_plus) - error) / eps
            self.assertAlmostEqual(
                grad_est, grad[i_param], delta=1e-1 * max(1.0, abs(grad_est))
            )

        # Test get_jacobian
        res, jac = pos_error.get_jacobian(model_params_init.numpy())
        grad_jac = 2.0 * np.matmul(np.transpose(jac), res)
        self.assertTrue(np.allclose(grad_jac, grad, rtol=1e-5, atol=1e-5))

        # Combine two error functions in a SkeletonSolverFunction:
        model_params_error = pym_solver2.ModelParametersErrorFunction(character)
        model_params_error.set_target_parameters(
            torch.rand(n_params, dtype=torch.float32).numpy()
        )
        skel_solver_function = pym_solver2.SkeletonSolverFunction(character)
        skel_solver_function.add_error_function(model_params_error)
        skel_solver_function.add_error_function(pos_error)
        error_combined = skel_solver_function.get_error(model_params_init.numpy())
        self.assertAlmostEqual(
            error_combined,
            model_params_error.get_error(model_params_init.numpy())
            + pos_error.get_error(model_params_init.numpy()),
            delta=1e-4,
        )

        self.assertTrue(
            np.allclose(
                skel_solver_function.get_gradient(model_params_init.numpy()),
                model_params_error.get_gradient(model_params_init.numpy())
                + pos_error.get_gradient(model_params_init.numpy()),
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_model_parameters_error(self) -> None:
        """Test ModelParametersError to ensure solved model parameters match the target."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Set target model parameters
        model_params_target = torch.rand_like(model_params_init)

        # Create ModelParametersErrorFunction
        model_params_error = pym_solver2.ModelParametersErrorFunction(character)

        # Set target parameters in the error function
        model_params_error.set_target_parameters(
            model_params_target.numpy(), np.ones(n_params)
        )

        # Create solver function with the model parameters error
        solver_function = pym_solver2.SkeletonSolverFunction(character)
        solver_function.error_functions = [model_params_error]

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Assert that the solved model parameters are close to the target
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(model_params_final),
                model_params_target,
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_solver_sequence_per_frame_model_parameters_error(self) -> None:
        """Test solve_sequence() with per-frame ModelParametersError to ensure
        that the result matches the target on every frame."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size
        n_frames = 5

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros((n_frames, n_params), dtype=torch.float32)

        # Set target model parameters for each frame
        model_params_target = torch.rand((n_frames, n_params), dtype=torch.float32)

        # Create SequenceSolverFunction
        solver_function = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Add per-frame ModelParametersErrorFunction
        for i_frame in range(n_frames):
            model_params_error = pym_solver2.ModelParametersErrorFunction(character)
            model_params_error.set_target_parameters(
                model_params_target[i_frame].numpy(), np.ones(n_params)
            )
            solver_function.add_error_function(i_frame, model_params_error)

        # Set solver options
        solver_options = pym_solver2.SequenceSolverOptions()
        solver_options.max_iterations = 10
        solver_options.regularization = 1e-5

        # Solve the sequence
        model_params_final = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Assert that the solved model parameters are close to the target for each frame
        for i_frame in range(n_frames):
            self.assertTrue(
                torch.allclose(
                    torch.from_numpy(model_params_final[i_frame]),
                    model_params_target[i_frame],
                    rtol=1e-5,
                    atol=1e-5,
                )
            )

    def test_solver_sequence_smoothness(self) -> None:
        """Test solve_sequence() with a smoothness constraint to ensure
        that the result matches the target on the first frame and is smooth across frames."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size
        n_frames = 5

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros((n_frames, n_params), dtype=torch.float32)

        # Set target model parameters for the first frame
        model_params_target_first_frame = torch.rand(n_params, dtype=torch.float32)

        # Create SequenceSolverFunction
        solver_function = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Add ModelParametersErrorFunction for the first frame
        model_params_error = pym_solver2.ModelParametersErrorFunction(character)
        model_params_error.set_target_parameters(
            model_params_target_first_frame.numpy(), np.ones(n_params)
        )
        solver_function.add_error_function(0, model_params_error)

        for i_frame in range(n_frames - 1):
            solver_function.add_sequence_error_function(
                i_frame, pym_solver2.ModelParametersSequenceErrorFunction(character)
            )

        # Add StateSequenceErrorFunction for smoothness across frames
        smoothness_error = pym_solver2.StateSequenceErrorFunction(character, weight=1.0)
        solver_function.add_sequence_error_function_all_frames(smoothness_error)

        # Set solver options
        solver_options = pym_solver2.SequenceSolverOptions()
        solver_options.max_iterations = 10
        solver_options.regularization = 1e-5

        # Solve the sequence
        model_params_final = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Assert that the solved model parameters for the first frame are close to the target
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(model_params_final[0]),
                model_params_target_first_frame,
                rtol=1e-5,
                atol=1e-5,
            )
        )

        # Assert smoothness across frames by checking small differences between consecutive frames
        for i_frame in range(1, n_frames):
            self.assertTrue(
                torch.allclose(
                    torch.from_numpy(model_params_final[i_frame]),
                    torch.from_numpy(model_params_final[i_frame - 1]),
                    rtol=1e-2,
                    atol=1e-2,
                )
            )

    def test_state_error_function_target_match(self) -> None:
        """Test StateErrorFunction to ensure it can match a given target skeleton state."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Generate a random set of model parameters as the target
        model_params_target = torch.rand_like(model_params_init)

        # Convert target model parameters to a target skeleton state
        skel_state_target = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_target.numpy()
            )
        )

        # Create StateErrorFunction
        state_error_function = pym_solver2.StateErrorFunction(character)

        # Set the target skeleton state in the error function
        state_error_function.set_target_state(skel_state_target.numpy())

        # Create solver function with the state error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [state_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Assert that the solved skeleton state is close to the target
        self.assertTrue(
            torch.allclose(
                pym_skel_state.to_matrix(skel_state_final),
                pym_skel_state.to_matrix(skel_state_target),
                rtol=1e-3,
                atol=1e-3,
            )
        )

    def test_set_active_parameters(self) -> None:
        """Test set_active_parameters() to ensure only active parameters change."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Set target model parameters
        model_params_target = torch.rand_like(model_params_init)

        # Create ModelParametersErrorFunction
        model_params_error = pym_solver2.ModelParametersErrorFunction(character)

        # Set target parameters in the error function
        model_params_error.set_target_parameters(
            model_params_target.numpy(), np.ones(n_params)
        )

        # Create solver function with the model parameters error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [model_params_error]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)

        # Define active parameters (e.g., only the first half are active)
        active_parameters = np.zeros(n_params, dtype=bool)
        active_parameters[: n_params // 2] = True

        # Set active parameters in the solver
        solver.set_enabled_parameters(active_parameters)

        # Solve with active parameters
        model_params_final = solver.solve(model_params_init.numpy())

        # Verify that only active parameters have changed
        self.assertTrue(
            np.allclose(
                model_params_final[: n_params // 2],
                model_params_target[: n_params // 2].numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                model_params_final[n_params // 2 :],
                model_params_init[n_params // 2 :].numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_point_triangle_error_function(self) -> None:
        """Test PointTriangleVertexErrorFunction to ensure a point is close to the target triangle."""
        torch.manual_seed(0)

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        # Create a PointTriangleVertexErrorFunction
        ptv_error_function = pym_solver2.PointTriangleVertexErrorFunction(character)

        # Define a triangle and a point
        triangle_indices = character.mesh.faces[0]
        point_index = character.mesh.vertices.shape[0] - 1
        triangle_bary_coords = [0.3, 0.3, 0.4]
        depth = 0.0
        weight = 1.0

        # Add a constraint
        ptv_error_function.add_constraints(
            np.asarray([point_index], dtype=np.int32),
            np.asarray([triangle_indices], dtype=np.int32),
            np.asarray([triangle_bary_coords], dtype=np.float32),
            np.asarray([depth], dtype=np.float32),
            np.asarray([weight], dtype=np.float32),
        )

        # Create solver function with the point-triangle vertex error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [ptv_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverQROptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 0.001
        solver_options.do_line_search = True

        # Create and run the solver
        model_params_init = torch.randn(
            character.parameter_transform.size, dtype=torch.float32
        )
        solver = pym_solver2.GaussNewtonSolverQR(solver_function, solver_options)
        enabled_params = ~(
            character.parameter_transform.scaling_parameters
            | character.parameter_transform.rigid_parameters
        )
        solver.set_enabled_parameters(enabled_params)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position of the point
        final_mesh = character.skin_points(skel_state_final.numpy())
        final_point_position = torch.from_numpy(final_mesh[point_index, :3])

        # Compute the target position of the point on the triangle
        triangle_vertices = torch.from_numpy(final_mesh[triangle_indices, :3])
        final_target_position = (
            triangle_bary_coords[0] * triangle_vertices[0]
            + triangle_bary_coords[1] * triangle_vertices[1]
            + triangle_bary_coords[2] * triangle_vertices[2]
        )

        # Assert that the final point position is close to the target position
        self.assertTrue(
            torch.allclose(
                final_point_position, final_target_position, rtol=1e-1, atol=1e-1
            )
        )

        # delete constraints and ensure they're empty
        ptv_error_function.clear_constraints()
        # self.assertTrue(len(ptv_error_function.constraints) == 0)

    def test_vertex_error_function(self) -> None:
        """Test VertexErrorFunction to ensure a vertex is targeted to a specific location."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        # Create a VertexErrorFunction
        vertex_error_function = pym_solver2.VertexErrorFunction(
            character, pym_solver2.VertexConstraintType.Position
        )
        vertex_error_function.weight = 2.0
        self.assertAlmostEqual(vertex_error_function.weight, 2.0)

        # Define a vertex and its target position
        vertex_index = 0
        target_position = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        target_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        weight = 1.0

        # Add a constraint to the vertex error function
        vertex_error_function.add_constraint(
            vertex_index, weight, target_position, target_normal
        )
        self.assertEqual(len(vertex_error_function.constraints), 1)

        # Create solver function with the vertex error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [vertex_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        model_params_init = torch.zeros(
            character.parameter_transform.size, dtype=torch.float32
        )
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        solver.set_enabled_parameters(character.parameter_transform.rigid_parameters)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position of the vertex
        final_mesh = character.skin_points(skel_state_final.numpy())
        final_vertex_position = torch.from_numpy(final_mesh[vertex_index, :3])

        # Assert that the final vertex position is close to the target position
        self.assertTrue(
            torch.allclose(
                final_vertex_position,
                torch.from_numpy(target_position),
                rtol=1e-3,
                atol=1e-3,
            )
        )

        # delete constraints and ensure they're empty
        vertex_error_function.clear_constraints()
        self.assertTrue(len(vertex_error_function.constraints) == 0)

    def test_pose_prior_error_function(self) -> None:
        """Test PosePriorErrorFunction to ensure it can converge to multiple modes."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)

        n_modes = 2

        # Generate a random set of model parameters as the target
        model_params_target = torch.rand(n_modes, n_params, dtype=torch.float32)

        n_pca = 2
        pose_prior_model = pym_geometry.Mppca(
            pi=torch.ones(n_modes).numpy(),
            mu=model_params_target.numpy(),
            W=torch.rand(n_modes, n_pca, n_params).numpy(),
            sigma=torch.ones(n_modes).numpy(),
            names=character.parameter_transform.names,
        )

        # Create PosePriorErrorFunction
        pose_prior_error_function = pym_solver2.PosePriorErrorFunction(
            character, pose_prior_model
        )

        # Create solver function with the pose prior error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [pose_prior_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5
        solver_options.verbose = True
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        # Should converge to the closest mode:
        for i_mode in range(n_modes):
            model_params_init = model_params_target[i_mode] + 0.1 * torch.randn_like(
                model_params_target[i_mode]
            )
            model_params_final = solver.solve(model_params_init.numpy())
            self.assertTrue(
                torch.allclose(
                    torch.from_numpy(model_params_final),
                    model_params_target[i_mode],
                    rtol=1e-5,
                    atol=1e-5,
                )
            )

    def test_aim_dir_constraint(self) -> None:
        """Test AimDirErrorFunction to ensure a local ray aims at a global target."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        def _normalize_vec(vec: npt.NDArray) -> npt.NDArray:
            return vec / np.linalg.norm(vec)

        # Define local ray origin and direction
        local_point = np.random.randn(3).astype(np.float32)
        local_dir = _normalize_vec(np.random.randn(3).astype(np.float32))

        # Define global target
        global_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Create AimDirErrorFunction
        aim_dir_error_function = pym_solver2.AimDirErrorFunction(character)

        # Add aim constraint
        parent_idx: int = character.skeleton.size - 1
        aim_dir_error_function.add_constraint(
            local_point, local_dir, global_target, parent=parent_idx, weight=1.0
        )

        # Create solver function with the aim direction error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [aim_dir_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position and direction of the local ray
        final_point = pym_skel_state.transform_points(
            skel_state_final[parent_idx],
            torch.from_numpy(local_point),
        )
        final_dir = pym_quaternion.rotate_vector(
            skel_state_final[parent_idx, 3:7], torch.from_numpy(local_dir)
        )

        # Compute the direction to the global target
        target_dir = _normalize_vec(global_target - final_point.numpy())

        # Assert that the final direction is close to the target direction
        self.assertTrue(np.allclose(final_dir, target_dir, rtol=1e-3, atol=1e-3))

        # delete constraints and ensure they're empty
        aim_dir_error_function.clear_constraints()
        self.assertTrue(len(aim_dir_error_function.constraints) == 0)

    def test_fixed_axis_constraint(self) -> None:
        """Test FixedAxisErrorFunction to ensure a local axis aligns with a global axis."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define local and global axes
        local_axis = _normalize_vec(np.random.randn(3).astype(np.float32))
        target_global_axis = _normalize_vec(np.random.randn(3).astype(np.float32))

        # Create FixedAxisErrorFunction
        fixed_axis_error_function = pym_solver2.FixedAxisDiffErrorFunction(character)
        fixed_axis_error_function2 = pym_solver2.FixedAxisCosErrorFunction(character)
        fixed_axis_error_function3 = pym_solver2.FixedAxisAngleErrorFunction(character)

        # Add fixed axis constraint
        parent_idx: int = character.skeleton.size - 1
        fixed_axis_error_function.add_constraint(
            local_axis, target_global_axis, parent=parent_idx, weight=1.0
        )

        # ensure all three variations work
        fixed_axis_error_function2.add_constraint(
            local_axis, target_global_axis, parent=parent_idx, weight=1.0
        )
        fixed_axis_error_function3.add_constraint(
            local_axis, target_global_axis, parent=parent_idx, weight=1.0
        )

        # Create solver function with the fixed axis error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [fixed_axis_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final local axis in global space
        final_global_axis = pym_quaternion.rotate_vector(
            skel_state_final[parent_idx, 3:7], torch.from_numpy(local_axis)
        )

        # Assert that the final local axis is close to the global axis
        self.assertTrue(
            np.allclose(final_global_axis, target_global_axis, rtol=1e-3, atol=1e-3)
        )

        # delete constraints and ensure they're empty
        fixed_axis_error_function.clear_constraints()
        self.assertTrue(len(fixed_axis_error_function.constraints) == 0)

    def test_normal_constraint(self) -> None:
        """Test NormalErrorFunction to ensure a point-to-plane distance is minimized."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define local point, local normal, and global target point
        local_point = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        local_normal = _normalize_vec(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        global_point = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        # Create NormalErrorFunction
        normal_error_function = pym_solver2.NormalErrorFunction(character)

        # Add normal constraint
        parent_idx: int = character.skeleton.size - 1
        normal_error_function.add_constraint(
            local_normal,
            global_point,
            parent=parent_idx,
            local_point=local_point,
            weight=1.0,
        )

        # Create solver function with the normal error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [normal_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position and normal in global space
        final_point = pym_skel_state.transform_points(
            skel_state_final[parent_idx],
            torch.from_numpy(local_point),
        )
        final_normal = pym_quaternion.rotate_vector(
            skel_state_final[parent_idx, 3:7], torch.from_numpy(local_normal)
        )

        # Calculate the signed distance from the global point to the plane
        # defined by the final point and normal
        global_point_tensor = torch.from_numpy(global_point)
        point_to_plane_vector = global_point_tensor - final_point
        signed_distance = torch.dot(point_to_plane_vector, final_normal)

        # Assert that the signed distance is close to zero
        self.assertAlmostEqual(signed_distance.item(), 0.0, delta=1e-3)

        # delete constraints and ensure they're empty
        normal_error_function.clear_constraints()
        self.assertTrue(len(normal_error_function.constraints) == 0)

    def test_distance_constraint(self) -> None:
        """Test DistanceErrorFunction to ensure a point maintains a target distance from an origin."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define origin point, target distance, and offset
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Origin in world space
        target_distance = 2.0  # Target distance from origin
        offset = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # Offset from parent joint

        # Create DistanceErrorFunction
        distance_error_function = pym_solver2.DistanceErrorFunction(character)

        # Add distance constraint
        parent_idx: int = character.skeleton.size - 1
        distance_error_function.add_constraint(
            origin=origin,
            target=target_distance,
            parent=parent_idx,
            offset=offset,
            weight=1.0,
        )

        # Create solver function with the distance error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [distance_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position of the point in global space
        final_point = pym_skel_state.transform_points(
            skel_state_final[parent_idx],
            torch.from_numpy(offset),
        )

        # Calculate the distance from the origin to the final point
        origin_tensor = torch.from_numpy(origin)
        actual_distance = torch.norm(final_point - origin_tensor).item()

        # Assert that the actual distance is close to the target distance
        self.assertAlmostEqual(actual_distance, target_distance, delta=1e-3)

        # Test with multiple constraints
        distance_error_function.clear_constraints()

        # Define multiple constraints
        origins = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        target_distances = np.array([2.0, 3.0], dtype=np.float32)
        parents = np.array([parent_idx, parent_idx], dtype=np.int32)
        offsets = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
        weights = np.array([1.0, 1.0], dtype=np.float32)

        # Add multiple constraints
        distance_error_function.add_constraints(
            origin=origins,
            target=target_distances,
            parent=parents,
            offset=offsets,
            weight=weights,
        )

        # Verify the number of constraints
        self.assertEqual(distance_error_function.num_constraints(), 2)

    def test_orientation_constraint(self) -> None:
        """Test OrientationErrorFunction to ensure a joint's orientation matches a target."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define offset and target quaternions
        axis1 = _normalize_vec(np.random.randn(3).astype(np.float32))
        angle1 = np.random.uniform(0, np.pi)
        offset_quat = pym_quaternion.from_axis_angle(torch.from_numpy(axis1 * angle1))

        axis2 = _normalize_vec(np.random.randn(3).astype(np.float32))
        angle2 = np.random.uniform(0, np.pi)
        target_quat = pym_quaternion.from_axis_angle(torch.from_numpy(axis2 * angle2))

        # Create OrientationErrorFunction
        orientation_error_function = pym_solver2.OrientationErrorFunction(character)

        # Add orientation constraint
        parent_idx: int = character.skeleton.size - 1
        orientation_error_function.add_constraint(
            offset=offset_quat.numpy(),
            target=target_quat.numpy(),
            parent=parent_idx,
            weight=1.0,
        )

        # Create solver function with the orientation error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [orientation_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final orientation
        final_orientation = skel_state_final[parent_idx, 3:7]  # quaternion part

        # The expected orientation is target * offset^-1
        expected_orientation = pym_quaternion.multiply(
            target_quat,
            pym_quaternion.inverse(offset_quat),
        )

        # Assert that the final orientation is close to the expected orientation
        # We need to check both q and -q since they represent the same rotation
        self.assertTrue(
            torch.allclose(
                pym_quaternion.to_rotation_matrix(final_orientation),
                pym_quaternion.to_rotation_matrix(expected_orientation),
                rtol=1e-3,
                atol=1e-3,
            )
        )

        # delete constraints and ensure they're empty
        orientation_error_function.clear_constraints()
        self.assertTrue(len(orientation_error_function.constraints) == 0)

    def test_plane_constraint(self) -> None:
        """Test PlaneErrorFunction to ensure a point stays on or above a plane."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define plane parameters
        offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Point in local space
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Up direction
        d = 1.0  # Plane equation: y = 1.0

        # Create PlaneErrorFunction for equality constraint (on the plane)
        plane_error_function = pym_solver2.PlaneErrorFunction(character, above=False)

        # Add plane constraint
        parent_idx: int = character.skeleton.size - 1
        plane_error_function.add_constraint(
            offset=offset,
            normal=normal,
            d=d,
            parent=parent_idx,
            weight=1.0,
        )

        # Create solver function with the plane error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [plane_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position of the point in global space
        final_point = pym_skel_state.transform_points(
            skel_state_final[parent_idx],
            torch.from_numpy(offset),
        )

        # Calculate the signed distance to the plane
        distance = final_point[1].item() - d  # y - d

        # Assert that the point is on the plane (distance close to zero)
        self.assertAlmostEqual(distance, 0.0, delta=1e-3)

        # Now test the inequality constraint (above the plane)
        plane_error_function = pym_solver2.PlaneErrorFunction(character, above=True)

        # Add plane constraint
        plane_error_function.add_constraint(
            offset=offset,
            normal=normal,
            d=d,
            parent=parent_idx,
            weight=1.0,
        )

        # Create solver function with the plane error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [plane_error_function]
        )

        # Create and run the solver with initial parameters that put the point below the plane
        model_params_below = model_params_init.clone()
        model_params_below[0] = -2.0  # Move the point below the plane

        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_below.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position of the point in global space
        final_point = pym_skel_state.transform_points(
            skel_state_final[parent_idx],
            torch.from_numpy(offset),
        )

        # Calculate the signed distance to the plane
        distance = final_point[1].item() - d  # y - d

        # Assert that the point is above or on the plane (distance >= 0)
        self.assertGreaterEqual(distance, -1e-3)

        # delete constraints and ensure they're empty
        plane_error_function.clear_constraints()
        self.assertTrue(len(plane_error_function.constraints) == 0)

    def test_projection_constraint(self) -> None:
        """Test ProjectionErrorFunction to ensure a 3D point projects to a target 2D point."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define projection parameters
        # Simple perspective projection matrix (3x4)
        projection = np.array(
            [
                [1.5, 0.2, 0.4, 5.0],  # x = X/Z
                [0.1, 1.0, 0.2, 0.0],  # y = Y/Z
                [0.1, 0.2, 1.4, 0.0],  # z = Z
            ],
            dtype=np.float32,
        )

        # Target 2D point (in normalized device coordinates)
        target_2d = np.array([0.5, 0.5], dtype=np.float32)

        # Local offset from the parent joint
        offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Create ProjectionErrorFunction
        projection_error_function = pym_solver2.ProjectionErrorFunction(
            character, near_clip=0.1, weight=1.0
        )

        # Add projection constraint
        parent_idx: int = character.skeleton.size - 1
        projection_error_function.add_constraint(
            projection=projection,
            target=target_2d,
            parent=parent_idx,
            offset=offset,
            weight=1.0,
        )

        # Create solver function with the projection error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [projection_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final position of the point in global space
        final_point = pym_skel_state.transform_points(
            skel_state_final[parent_idx],
            torch.from_numpy(offset),
        ).numpy()

        # Apply the projection matrix to get the projected 2D point
        # First create homogeneous coordinates by adding 1 as the 4th component
        point_homogeneous = np.append(final_point, 1.0)

        # Apply projection matrix
        projected = np.dot(projection, point_homogeneous)

        self.assertGreater(
            projected[2], 0.1
        )  # Check that point is in front of the camera

        # Convert to normalized device coordinates (divide by Z)
        projected_2d = projected[:2] / projected[2]

        # Assert that the projected point is close to the target 2D point
        self.assertTrue(
            np.allclose(projected_2d, target_2d, rtol=1e-3, atol=1e-3),
            f"Projected point {projected_2d} is not close to target {target_2d}",
        )

        # delete constraints and ensure they're empty
        projection_error_function.clear_constraints()
        self.assertTrue(len(projection_error_function.constraints) == 0)

    def test_camera_projection_constraint(self) -> None:
        """Test CameraProjectionErrorFunction moves a joint to match a target 2D pixel."""
        import pymomentum.camera as pym_camera

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)
        n_params = character.parameter_transform.size

        # Set up a pinhole camera with known intrinsics
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        intrinsics = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        # Place the camera looking down the +Z axis with a translation offset
        # so the skeleton (near origin) is in front of the camera.
        eye_from_world = np.eye(4, dtype=np.float32)
        eye_from_world[2, 3] = 5.0  # translate points +5 in Z in eye space

        # Create the error function with a static camera (camera_parent=None)
        cam_proj = pym_solver2.CameraProjectionErrorFunction(
            character,
            intrinsics,
            camera_offset=eye_from_world,
            weight=1.0,
        )

        # Target pixel: slightly off-center
        target_2d = np.array([400.0, 300.0], dtype=np.float32)

        # Add constraint on the last joint
        parent_idx: int = character.skeleton.size - 1
        cam_proj.add_constraint(
            parent=parent_idx,
            target=target_2d,
            weight=1.0,
        )

        self.assertEqual(cam_proj.num_constraints(), 1)

        # Create solver and run
        model_params_init = np.zeros(n_params, dtype=np.float32)
        solver_function = pym_solver2.SkeletonSolverFunction(character, [cam_proj])
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init)

        # Get the final world-space position of the constrained joint
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_final
        )
        final_pos = skel_state_final[parent_idx, :3]

        # Project the final position through the camera to get the pixel coordinates
        # Transform to eye space: p_eye = eye_from_world @ [p_world; 1]
        p_eye = eye_from_world[:3, :3] @ final_pos + eye_from_world[:3, 3]

        # Pinhole projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
        projected_u = fx * p_eye[0] / p_eye[2] + cx
        projected_v = fy * p_eye[1] / p_eye[2] + cy
        projected_2d = np.array([projected_u, projected_v])

        np.testing.assert_allclose(
            projected_2d,
            target_2d,
            atol=0.5,
            err_msg=f"Projected pixel {projected_2d} does not match target {target_2d}",
        )

        # Verify constraints property and clear
        constraints = cam_proj.constraints
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].parent, parent_idx)
        np.testing.assert_allclose(constraints[0].target, target_2d)

        cam_proj.clear_constraints()
        self.assertEqual(cam_proj.num_constraints(), 0)

        # Test the Camera-based constructor: should produce the same result
        camera = pym_camera.Camera(intrinsics, eye_from_world)
        cam_proj2 = pym_solver2.CameraProjectionErrorFunction(
            character,
            camera,
            weight=1.0,
        )
        cam_proj2.add_constraint(
            parent=parent_idx,
            target=target_2d,
            weight=1.0,
        )
        solver_function2 = pym_solver2.SkeletonSolverFunction(character, [cam_proj2])
        solver2 = pym_solver2.GaussNewtonSolver(solver_function2, solver_options)
        model_params_final2 = solver2.solve(model_params_init)

        np.testing.assert_allclose(
            model_params_final2,
            model_params_final,
            atol=1e-5,
            err_msg="Camera constructor should produce the same result as intrinsics+offset",
        )

    def test_vertex_projection_constraint(self) -> None:
        """Test VertexProjectionErrorFunction to ensure a 3D vertex projects to a target 2D point."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define projection parameters
        # Simple perspective projection matrix (3x4)
        projection = np.array(
            [
                [1.5, 0.2, 0.4, 5.0],  # x = X/Z
                [0.1, 1.0, 0.2, 0.0],  # y = Y/Z
                [0.1, 0.2, 1.4, 10.0],  # z = Z
            ],
            dtype=np.float32,
        )

        # Target 2D point (in normalized device coordinates)
        target_2d = np.array([0.5, 0.5], dtype=np.float32)

        # Choose a vertex to project
        vertex_index = 0  # Use the first vertex of the mesh

        # Create VertexProjectionErrorFunction
        vertex_projection_error_function = pym_solver2.VertexProjectionErrorFunction(
            character, max_threads=0
        )

        # Add vertex projection constraint
        vertex_projection_error_function.add_constraint(
            vertex_index=vertex_index,
            weight=1.0,
            target_position=target_2d,
            projection=projection,
        )
        self.assertEqual(len(vertex_projection_error_function.constraints), 1)

        # Create solver function with the vertex projection error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [vertex_projection_error_function]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert the solved model parameters to a skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute the final mesh
        final_mesh = character.skin_points(skel_state_final.numpy())

        # Get the final position of the vertex
        final_vertex_position = final_mesh[vertex_index, :3]

        # Apply the projection matrix to get the projected 2D point
        # First create homogeneous coordinates by adding 1 as the 4th component
        vertex_homogeneous = np.append(final_vertex_position, 1.0)

        # Apply projection matrix
        projected = np.dot(projection, vertex_homogeneous)

        self.assertGreater(
            projected[2], 0.1
        )  # Check that vertex is in front of the camera

        # Convert to normalized device coordinates (divide by Z)
        projected_2d = projected[:2] / projected[2]

        # Assert that the projected vertex is close to the target 2D point
        self.assertTrue(
            np.allclose(projected_2d, target_2d, rtol=1e-3, atol=1e-3),
            f"Projected vertex {projected_2d} is not close to target {target_2d}",
        )

        # delete constraints and ensure they're empty
        vertex_projection_error_function.clear_constraints()
        self.assertTrue(len(vertex_projection_error_function.constraints) == 0)

    def test_vertex_sequence_error_function(self) -> None:
        """Test VertexSequenceErrorFunction to ensure vertex velocities match target velocities."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size
        n_frames = 2  # VertexSequenceErrorFunction works with 2 frames

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialize model parameters for both frames
        model_params_init = torch.zeros((n_frames, n_params), dtype=torch.float32)

        # Set up different poses for the two frames to create motion
        model_params_frame0 = torch.zeros(n_params, dtype=torch.float32)
        model_params_frame1 = torch.zeros(n_params, dtype=torch.float32)

        # Create some motion by changing translation parameters
        model_params_frame1[0] = 1.0  # Move in x direction
        model_params_frame1[1] = 0.5  # Move in y direction

        model_params_init[0] = model_params_frame0
        model_params_init[1] = model_params_frame1

        # Create VertexSequenceErrorFunction
        vertex_sequence_error = pym_solver2.VertexSequenceErrorFunction(character)
        self.assertEqual(vertex_sequence_error.num_constraints, 0)

        # Define target velocities for specific vertices
        vertex_indices = [0, 1, 2]  # Test with first 3 vertices
        target_velocities = np.array(
            [
                [1.0, 0.5, 0.0],  # Vertex 0: move in x and y
                [0.5, 1.0, 0.0],  # Vertex 1: move in y primarily
                [0.0, 0.0, 0.5],  # Vertex 2: move in z
            ],
            dtype=np.float32,
        )
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Add constraints using the batch method
        vertex_sequence_error.add_constraints(
            vertex_index=np.array(vertex_indices, dtype=np.int32),
            weight=weights,
            target_velocity=target_velocities,
        )
        self.assertEqual(vertex_sequence_error.num_constraints, 3)

        # Test individual constraint addition as well
        vertex_sequence_error.add_constraint(
            vertex_index=3,
            weight=1.0,
            target_velocity=np.array([0.2, 0.3, 0.1], dtype=np.float32),
        )
        self.assertEqual(vertex_sequence_error.num_constraints, 4)

        # Verify constraints were added correctly
        constraints = vertex_sequence_error.constraints
        self.assertEqual(len(constraints), 4)
        self.assertEqual(constraints[0].vertex_index, 0)
        self.assertTrue(np.allclose(constraints[0].target_velocity, [1.0, 0.5, 0.0]))
        self.assertAlmostEqual(constraints[0].weight, 1.0)

        # Create SequenceSolverFunction
        solver_function = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Add the vertex sequence error function
        solver_function.add_sequence_error_function(0, vertex_sequence_error)

        # Set solver options
        solver_options = pym_solver2.SequenceSolverOptions()
        solver_options.max_iterations = 50
        solver_options.regularization = 1e-5

        # Solve the sequence
        model_params_final = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Convert final model parameters to skeleton states
        skel_state_frame0 = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final[0]
            )
        )
        skel_state_frame1 = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final[1]
            )
        )

        # Compute final meshes for both frames
        mesh_frame0 = character.skin_points(skel_state_frame0.numpy())
        mesh_frame1 = character.skin_points(skel_state_frame1.numpy())

        # Verify that vertex velocities match target velocities
        for i, vertex_idx in enumerate(vertex_indices):
            # Compute actual velocity (difference between frames)
            actual_velocity = torch.from_numpy(
                mesh_frame1[vertex_idx, :3] - mesh_frame0[vertex_idx, :3]
            )
            expected_velocity = target_velocities[i]

            # Assert that actual velocity is close to target velocity
            self.assertTrue(
                torch.allclose(
                    actual_velocity,
                    torch.from_numpy(expected_velocity),
                    rtol=1e-3,  # Allow some tolerance due to optimization
                    atol=1e-3,
                ),
                f"Vertex {vertex_idx}: actual velocity {actual_velocity.numpy()} "
                f"does not match target {expected_velocity}",
            )

        # Test with zero target velocities (stationary constraint)
        vertex_sequence_error.clear_constraints()
        self.assertEqual(vertex_sequence_error.num_constraints, 0)

        # Add zero velocity constraints
        zero_velocities = np.zeros((2, 3), dtype=np.float32)
        vertex_sequence_error.add_constraints(
            vertex_index=np.array([0, 1], dtype=np.int32),
            weight=np.array([2.0, 2.0], dtype=np.float32),
            target_velocity=zero_velocities,
        )

        # Solve again with zero velocity constraints
        model_params_final_zero = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Convert to skeleton states
        skel_state_frame0_zero = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final_zero[0]
            )
        )
        skel_state_frame1_zero = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final_zero[1]
            )
        )

        # Compute meshes
        mesh_frame0_zero = character.skin_points(skel_state_frame0_zero.numpy())
        mesh_frame1_zero = character.skin_points(skel_state_frame1_zero.numpy())

        # Verify that constrained vertices have minimal motion
        for vertex_idx in [0, 1]:
            actual_velocity = (
                mesh_frame1_zero[vertex_idx, :3] - mesh_frame0_zero[vertex_idx, :3]
            )
            velocity_magnitude = np.linalg.norm(actual_velocity)

            # Assert that velocity is close to zero
            self.assertLess(
                velocity_magnitude,
                1e-4,  # Small tolerance for numerical precision
                f"Vertex {vertex_idx} should be stationary but has velocity magnitude {velocity_magnitude}",
            )

        # Test error function properties
        self.assertEqual(vertex_sequence_error.character, character)
        self.assertGreater(vertex_sequence_error.weight, 0.0)

        # Test string representation
        repr_str = repr(vertex_sequence_error)
        self.assertIn("VertexSequenceErrorFunction", repr_str)
        self.assertIn("num_constraints=2", repr_str)

    def test_acceleration_sequence_error_function(self) -> None:
        """Test AccelerationSequenceErrorFunction to ensure joint accelerations match targets."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size
        n_frames = 3  # AccelerationSequenceErrorFunction requires 3 frames

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialize model parameters for three frames
        model_params_init = torch.zeros((n_frames, n_params), dtype=torch.float32)

        # Create AccelerationSequenceErrorFunction
        # Test basic construction
        accel_error = pym_solver2.AccelerationSequenceErrorFunction(character)

        # Test construction with parameters
        accel_error_with_params = pym_solver2.AccelerationSequenceErrorFunction(
            character,
            weight=2.0,
            joint_weights=np.ones(character.skeleton.size, dtype=np.float32),
            target_acceleration=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        # Verify it was constructed (silence unused variable warning)
        self.assertIsNotNone(accel_error_with_params)

        # Test that set methods work
        accel_error.set_target_acceleration(
            np.array([0.0, -9.8, 0.0], dtype=np.float32)
        )

        # Create per-joint accelerations
        per_joint_accels = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(character.skeleton.size)
        ]
        accel_error.set_target_accelerations(per_joint_accels)

        # Test set_target_weights
        joint_weights = np.ones(character.skeleton.size, dtype=np.float32)
        joint_weights[0] = 2.0  # Give first joint more weight
        accel_error.set_target_weights(joint_weights)

        # Test reset
        accel_error.reset()

        # ========================================
        # Test 1: Zero target acceleration (smoothness)
        # Set up constant velocity motion: the solution should have zero acceleration
        # ========================================
        model_params_init[0, 0] = 0.0
        model_params_init[1, 0] = 1.0
        model_params_init[2, 0] = 2.0

        solver_function = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Anchor first and last frames
        first_frame_pos_error = pym_solver2.ModelParametersErrorFunction(character)
        first_frame_pos_error.set_target_parameters(
            model_params_init[0].numpy(), np.ones(n_params)
        )
        solver_function.add_error_function(0, first_frame_pos_error)

        last_frame_pos_error = pym_solver2.ModelParametersErrorFunction(character)
        last_frame_pos_error.set_target_parameters(
            model_params_init[2].numpy(), np.ones(n_params)
        )
        solver_function.add_error_function(2, last_frame_pos_error)

        # Add acceleration error with zero target (smoothness constraint)
        accel_error_zero = pym_solver2.AccelerationSequenceErrorFunction(
            character,
            weight=1.0,
            target_acceleration=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        solver_function.add_sequence_error_function(0, accel_error_zero)

        solver_options = pym_solver2.SequenceSolverOptions()
        solver_options.max_iterations = 50
        solver_options.regularization = 1e-5

        model_params_final = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Convert to skeleton states and verify accelerations match zero target
        skel_states = [
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final[i]
            )
            for i in range(n_frames)
        ]

        for joint_idx in range(character.skeleton.size):
            pos0 = skel_states[0][joint_idx, :3]
            pos1 = skel_states[1][joint_idx, :3]
            pos2 = skel_states[2][joint_idx, :3]

            # Acceleration = pos[t+1] - 2*pos[t] + pos[t-1]
            acceleration = pos2 - 2 * pos1 + pos0
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # Assert that measured acceleration matches target
            self.assertTrue(
                np.allclose(
                    acceleration,
                    target,
                    rtol=0.1,
                    atol=0.1,
                ),
                f"Joint {joint_idx}: acceleration {acceleration} "
                f"does not match target {target}",
            )

        # ========================================
        # Test 2: Non-zero target acceleration (ballistic motion)
        # For ballistic motion with constant acceleration a, the analytical solution is:
        #   p(t) = p0 + v0*t + 0.5*a*t^2
        # For three frames at t=0, 1, 2 with dt=1:
        #   p0 = p0
        #   p1 = p0 + v0 + 0.5*a
        #   p2 = p0 + 2*v0 + 2*a
        # The finite difference acceleration is:
        #   accel_fd = p2 - 2*p1 + p0 = a (for dt=1)
        # ========================================
        target_accel = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # Gravity-like

        # Compute analytical positions for p0=(0,0,0), v0=(0,1,0), a=(0,-1,0)
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # p(t) = p0 + v0*t + 0.5*a*t^2
        p1_analytical = p0 + v0 * 1.0 + 0.5 * target_accel * 1.0**2
        p2_analytical = p0 + v0 * 2.0 + 0.5 * target_accel * 2.0**2

        # Set up initial model parameters with these positions
        model_params_init_ballistic = torch.zeros(
            (n_frames, n_params), dtype=torch.float32
        )
        model_params_init_ballistic[0, :3] = torch.from_numpy(p0)
        model_params_init_ballistic[1, :3] = torch.from_numpy(p1_analytical)
        model_params_init_ballistic[2, :3] = torch.from_numpy(p2_analytical)

        # Create a new solver
        solver_function_ballistic = pym_solver2.SequenceSolverFunction(
            character, n_frames
        )

        # Anchor first frame strongly
        first_frame_error = pym_solver2.ModelParametersErrorFunction(character)
        first_frame_error.set_target_parameters(
            model_params_init_ballistic[0].numpy(), np.ones(n_params) * 10.0
        )
        solver_function_ballistic.add_error_function(0, first_frame_error)

        # Add acceleration constraint with gravity target
        accel_error_gravity = pym_solver2.AccelerationSequenceErrorFunction(
            character,
            weight=10.0,
            target_acceleration=target_accel,
        )
        solver_function_ballistic.add_sequence_error_function(0, accel_error_gravity)

        # Start from a perturbed initial guess
        model_params_init_perturbed = model_params_init_ballistic.clone()
        model_params_init_perturbed[1, 1] += 0.3  # Perturb y position of frame 1
        model_params_init_perturbed[2, 1] += 0.5  # Perturb y position of frame 2

        model_params_final_ballistic = pym_solver2.solve_sequence(
            solver_function_ballistic,
            model_params_init_perturbed.numpy(),
            solver_options,
        )

        # Convert to skeleton states and verify accelerations match target
        skel_states_ballistic = [
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final_ballistic[i]
            )
            for i in range(n_frames)
        ]

        # Check that acceleration matches target for root joint
        pos0 = skel_states_ballistic[0][0, :3]
        pos1 = skel_states_ballistic[1][0, :3]
        pos2 = skel_states_ballistic[2][0, :3]
        measured_accel = pos2 - 2 * pos1 + pos0

        self.assertTrue(
            np.allclose(
                measured_accel,
                target_accel,
                rtol=0.1,
                atol=0.1,
            ),
            f"Ballistic test: measured acceleration {measured_accel} "
            f"does not match target {target_accel}",
        )

    def test_jerk_sequence_error_function(self) -> None:
        """Test JerkSequenceErrorFunction to ensure joint jerks match targets."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size
        n_frames = 4  # JerkSequenceErrorFunction requires 4 frames

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialize model parameters for four frames
        model_params_init = torch.zeros((n_frames, n_params), dtype=torch.float32)

        # Create JerkSequenceErrorFunction
        # Test basic construction
        jerk_error = pym_solver2.JerkSequenceErrorFunction(character)

        # Test construction with parameters
        jerk_error_with_params = pym_solver2.JerkSequenceErrorFunction(
            character,
            weight=2.0,
            joint_weights=np.ones(character.skeleton.size, dtype=np.float32),
            target_jerk=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        # Verify it was constructed (silence unused variable warning)
        self.assertIsNotNone(jerk_error_with_params)

        # Test that set methods work
        jerk_error.set_target_jerk(np.array([0.0, -0.1, 0.0], dtype=np.float32))

        # Create per-joint jerks
        per_joint_jerks = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(character.skeleton.size)
        ]
        jerk_error.set_target_jerks(per_joint_jerks)

        # Test set_target_weights
        joint_weights = np.ones(character.skeleton.size, dtype=np.float32)
        joint_weights[0] = 2.0  # Give first joint more weight
        jerk_error.set_target_weights(joint_weights)

        # Test reset
        jerk_error.reset()

        # ========================================
        # Test 1: Zero target jerk (smoothness in acceleration)
        # Set up constant acceleration motion: the solution should have zero jerk
        # For constant acceleration a: p(t) = p0 + v0*t + 0.5*a*t^2
        # ========================================
        # Initial conditions: p0=0, v0=0, a=1
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        accel = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Compute analytical positions for t=0, 1, 2, 3
        for t in range(n_frames):
            pos = p0 + v0 * float(t) + 0.5 * accel * float(t) ** 2
            model_params_init[t, :3] = torch.from_numpy(pos)

        solver_function = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Anchor first and last frames
        first_frame_pos_error = pym_solver2.ModelParametersErrorFunction(character)
        first_frame_pos_error.set_target_parameters(
            model_params_init[0].numpy(), np.ones(n_params)
        )
        solver_function.add_error_function(0, first_frame_pos_error)

        last_frame_pos_error = pym_solver2.ModelParametersErrorFunction(character)
        last_frame_pos_error.set_target_parameters(
            model_params_init[3].numpy(), np.ones(n_params)
        )
        solver_function.add_error_function(3, last_frame_pos_error)

        # Add jerk error with zero target (smoothness constraint)
        jerk_error_zero = pym_solver2.JerkSequenceErrorFunction(
            character,
            weight=1.0,
            target_jerk=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        solver_function.add_sequence_error_function(0, jerk_error_zero)

        solver_options = pym_solver2.SequenceSolverOptions()
        solver_options.max_iterations = 50
        solver_options.regularization = 1e-5

        model_params_final = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Convert to skeleton states and verify jerk matches zero target
        skel_states = [
            torch.from_numpy(
                pym_geometry.model_parameters_to_skeleton_state(
                    character, model_params_final[i]
                )
            )
            for i in range(n_frames)
        ]

        for joint_idx in range(character.skeleton.size):
            pos0 = skel_states[0][joint_idx, :3]
            pos1 = skel_states[1][joint_idx, :3]
            pos2 = skel_states[2][joint_idx, :3]
            pos3 = skel_states[3][joint_idx, :3]

            # Jerk = pos[t-1] - 3*pos[t] + 3*pos[t+1] - pos[t+2]
            jerk = pos0 - 3 * pos1 + 3 * pos2 - pos3
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # Assert that measured jerk matches target
            self.assertTrue(
                torch.allclose(
                    jerk,
                    torch.from_numpy(target),
                    rtol=0.1,
                    atol=0.1,
                ),
                f"Joint {joint_idx}: jerk {jerk.numpy()} does not match target {target}",
            )

    def test_vertex_vertex_distance_constraint(self) -> None:
        """Test VertexVertexDistanceErrorFunction to ensure vertices are pulled to target distance."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Choose two vertices to constrain - use vertices that are initially far apart
        vertex_index1 = 0
        vertex_index2 = character.mesh.vertices.shape[0] - 1  # Last vertex
        target_distance = 0.5  # Target distance between the two vertices
        weight = 1.0

        # Get initial positions of the vertices
        skel_state_init = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_init.numpy()
            )
        )
        initial_mesh = character.skin_points(skel_state_init.numpy())
        initial_pos1 = initial_mesh[vertex_index1, :3]
        initial_pos2 = initial_mesh[vertex_index2, :3]
        initial_distance = np.linalg.norm(initial_pos2 - initial_pos1)

        # Create VertexVertexDistanceErrorFunction
        vertex_distance_error = pym_solver2.VertexVertexDistanceErrorFunction(character)

        # Test basic properties
        self.assertEqual(vertex_distance_error.num_constraints(), 0)
        self.assertEqual(len(vertex_distance_error.constraints), 0)

        # Add a single constraint
        vertex_distance_error.add_constraint(
            vertex_index1=vertex_index1,
            vertex_index2=vertex_index2,
            weight=weight,
            target_distance=target_distance,
        )

        # Verify constraint was added
        self.assertEqual(vertex_distance_error.num_constraints(), 1)
        self.assertEqual(len(vertex_distance_error.constraints), 1)

        constraint = vertex_distance_error.constraints[0]
        self.assertEqual(constraint.vertex_index1, vertex_index1)
        self.assertEqual(constraint.vertex_index2, vertex_index2)
        self.assertAlmostEqual(constraint.weight, weight)
        self.assertAlmostEqual(constraint.target_distance, target_distance)

        # Create solver function with the vertex distance error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [vertex_distance_error]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute final mesh and vertex positions
        final_mesh = character.skin_points(skel_state_final.numpy())
        final_pos1 = final_mesh[vertex_index1, :3]
        final_pos2 = final_mesh[vertex_index2, :3]
        final_distance = np.linalg.norm(final_pos2 - final_pos1)

        # Assert that the final distance is close to the target distance
        self.assertAlmostEqual(
            final_distance,
            target_distance,
            delta=1e-3,
            msg=f"Final distance {final_distance} does not match target {target_distance}",
        )

        # Verify that the distance actually changed from the initial distance
        self.assertNotAlmostEqual(
            initial_distance,
            final_distance,
            delta=1e-1,
            msg=f"Distance did not change significantly from initial {initial_distance} to final {final_distance}",
        )

        # Test multiple constraints using add_constraints
        vertex_distance_error.clear_constraints()
        self.assertEqual(vertex_distance_error.num_constraints(), 0)

        # Add multiple constraints
        vertex_indices1 = np.array([0, 1], dtype=np.int32)
        vertex_indices2 = np.array([2, 3], dtype=np.int32)
        weights = np.array([1.0, 2.0], dtype=np.float32)
        target_distances = np.array([0.3, 0.7], dtype=np.float32)

        vertex_distance_error.add_constraints(
            vertex_index1=vertex_indices1,
            vertex_index2=vertex_indices2,
            weight=weights,
            target_distance=target_distances,
        )

        # Verify multiple constraints were added
        self.assertEqual(vertex_distance_error.num_constraints(), 2)
        constraints = vertex_distance_error.constraints
        self.assertEqual(len(constraints), 2)

        # Check first constraint
        self.assertEqual(constraints[0].vertex_index1, 0)
        self.assertEqual(constraints[0].vertex_index2, 2)
        self.assertAlmostEqual(constraints[0].weight, 1.0)
        self.assertAlmostEqual(constraints[0].target_distance, 0.3)

        # Check second constraint
        self.assertEqual(constraints[1].vertex_index1, 1)
        self.assertEqual(constraints[1].vertex_index2, 3)
        self.assertAlmostEqual(constraints[1].weight, 2.0)
        self.assertAlmostEqual(constraints[1].target_distance, 0.7)

        # Test string representation
        repr_str = repr(vertex_distance_error)
        self.assertIn("VertexVertexDistanceErrorFunction", repr_str)
        self.assertIn("num_constraints=2", repr_str)

        # Test constraint string representation
        constraint_repr = repr(constraints[0])
        self.assertIn("VertexVertexDistanceConstraint", constraint_repr)
        self.assertIn("vertex_index1=0", constraint_repr)
        self.assertIn("vertex_index2=2", constraint_repr)

    def test_joint_to_joint_distance_constraint(self) -> None:
        """Test JointToJointDistanceErrorFunction to ensure joints are pulled to target distance."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Choose two joints to constrain - use joints that are initially far apart
        joint_index1 = 0
        joint_index2 = character.skeleton.size - 1  # Last joint
        offset1 = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        offset2 = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        target_distance = 1.5  # Target distance between the two points
        weight = 1.0

        # Get initial positions of the points
        skel_state_init = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_init.numpy()
            )
        )
        initial_point1 = pym_skel_state.transform_points(
            skel_state_init[joint_index1],
            torch.from_numpy(offset1),
        )
        initial_point2 = pym_skel_state.transform_points(
            skel_state_init[joint_index2],
            torch.from_numpy(offset2),
        )
        initial_distance = torch.norm(initial_point2 - initial_point1).item()

        # Create JointToJointDistanceErrorFunction
        joint_distance_error = pym_solver2.JointToJointDistanceErrorFunction(character)

        # Test basic properties
        self.assertEqual(len(joint_distance_error.constraints), 0)

        # Add a single constraint
        joint_distance_error.add_constraint(
            joint1=joint_index1,
            offset1=offset1,
            joint2=joint_index2,
            offset2=offset2,
            target_distance=target_distance,
            weight=weight,
        )

        # Verify constraint was added
        self.assertEqual(len(joint_distance_error.constraints), 1)

        constraint = joint_distance_error.constraints[0]
        self.assertEqual(constraint.joint1, joint_index1)
        self.assertEqual(constraint.joint2, joint_index2)
        self.assertTrue(np.allclose(constraint.offset1, offset1))
        self.assertTrue(np.allclose(constraint.offset2, offset2))
        self.assertAlmostEqual(constraint.weight, weight)
        self.assertAlmostEqual(constraint.target_distance, target_distance)

        # Create solver function with the joint distance error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [joint_distance_error]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute final positions of the points
        final_point1 = pym_skel_state.transform_points(
            skel_state_final[joint_index1],
            torch.from_numpy(offset1),
        )
        final_point2 = pym_skel_state.transform_points(
            skel_state_final[joint_index2],
            torch.from_numpy(offset2),
        )
        final_distance = torch.norm(final_point2 - final_point1).item()

        # Assert that the final distance is close to the target distance
        self.assertAlmostEqual(
            final_distance,
            target_distance,
            delta=1e-3,
            msg=f"Final distance {final_distance} does not match target {target_distance}",
        )

        # Verify that the distance actually changed from the initial distance
        self.assertNotAlmostEqual(
            initial_distance,
            final_distance,
            delta=1e-1,
            msg=f"Distance did not change significantly from initial {initial_distance} to final {final_distance}",
        )

        # Test multiple constraints using add_constraints
        joint_distance_error.clear_constraints()
        self.assertEqual(len(joint_distance_error.constraints), 0)

        # Add multiple constraints
        joint_indices1 = np.array([0, 1], dtype=np.int32)
        offsets1 = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
        joint_indices2 = np.array([2, 3], dtype=np.int32)
        offsets2 = np.array([[0.0, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=np.float32)
        target_distances = np.array([0.8, 1.2], dtype=np.float32)
        weights = np.array([1.0, 2.0], dtype=np.float32)

        joint_distance_error.add_constraints(
            joint1=joint_indices1,
            offset1=offsets1,
            joint2=joint_indices2,
            offset2=offsets2,
            target_distance=target_distances,
            weight=weights,
        )

        # Verify multiple constraints were added
        self.assertEqual(len(joint_distance_error.constraints), 2)
        constraints = joint_distance_error.constraints

        # Check first constraint
        self.assertEqual(constraints[0].joint1, 0)
        self.assertEqual(constraints[0].joint2, 2)
        self.assertTrue(np.allclose(constraints[0].offset1, [0.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(constraints[0].offset2, [0.0, 0.0, 0.5]))
        self.assertAlmostEqual(constraints[0].weight, 1.0)
        self.assertAlmostEqual(constraints[0].target_distance, 0.8)

        # Check second constraint
        self.assertEqual(constraints[1].joint1, 1)
        self.assertEqual(constraints[1].joint2, 3)
        self.assertTrue(np.allclose(constraints[1].offset1, [0.0, 0.5, 0.0]))
        self.assertTrue(np.allclose(constraints[1].offset2, [0.5, 0.5, 0.0]))
        self.assertAlmostEqual(constraints[1].weight, 2.0)
        self.assertAlmostEqual(constraints[1].target_distance, 1.2)

        # Test string representation
        repr_str = repr(joint_distance_error)
        self.assertIn("JointToJointDistanceErrorFunction", repr_str)
        self.assertIn("num_constraints=2", repr_str)

        # Test constraint string representation
        constraint_repr = repr(constraints[0])
        self.assertIn("JointToJointDistanceConstraint", constraint_repr)
        self.assertIn("joint1=0", constraint_repr)
        self.assertIn("joint2=2", constraint_repr)

    def test_weight_validation(self) -> None:
        """Test that error functions throw ValueError when negative weights are passed."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        # Test scalar weight validation in constructor
        with self.assertRaises(ValueError) as context:
            pym_solver2.PositionErrorFunction(character, weight=-1.0)
        self.assertIn("weight must be non-negative", str(context.exception))

        # Test scalar weight validation in property setter
        pos_error = pym_solver2.PositionErrorFunction(character, weight=1.0)
        with self.assertRaises(ValueError) as context:
            pos_error.weight = -0.5
        self.assertIn("weight must be non-negative", str(context.exception))

        # Test scalar weight validation in add_constraint
        with self.assertRaises(ValueError) as context:
            pos_error.add_constraint(
                parent=0,
                target=np.array([1.0, 0.0, 0.0]),
                weight=-2.0,
            )
        self.assertIn("weight must be non-negative", str(context.exception))

        # Test array weight validation in add_constraints
        with self.assertRaises(ValueError) as context:
            pos_error.add_constraints(
                parent=np.array([0, 1], dtype=np.int32),
                target=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                weight=np.array(
                    [1.0, -1.5], dtype=np.float32
                ),  # Second weight is negative
            )
        self.assertIn("all weights must be non-negative", str(context.exception))
        self.assertIn("index 1", str(context.exception))

        # Test array weight validation in ModelParametersErrorFunction
        with self.assertRaises(ValueError) as context:
            pym_solver2.ModelParametersErrorFunction(
                character,
                weights=np.array([1.0, -0.1, 2.0]),  # Second weight is negative
            )
        self.assertIn("all weights must be non-negative", str(context.exception))
        self.assertIn("index 1", str(context.exception))

        # Test that valid weights work correctly
        pos_error_valid = pym_solver2.PositionErrorFunction(character, weight=2.0)
        self.assertEqual(pos_error_valid.weight, 2.0)

        pos_error_valid.add_constraint(
            parent=0,
            target=np.array([1.0, 0.0, 0.0]),
            weight=1.5,
        )
        self.assertEqual(len(pos_error_valid.constraints), 1)

        pos_error_valid.add_constraints(
            parent=np.array([1, 2], dtype=np.int32),
            target=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            weight=np.array([0.5, 2.5], dtype=np.float32),
        )
        self.assertEqual(len(pos_error_valid.constraints), 3)

    def test_joint_to_joint_position_constraint(self) -> None:
        """Test JointToJointPositionErrorFunction to ensure a source point matches a target position in reference frame."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Choose two joints to constrain
        source_joint = character.skeleton.size - 1  # Last joint
        reference_joint = 0  # Root joint
        source_offset = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        reference_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target_position = np.array(
            [1.0, 0.5, 0.0], dtype=np.float32
        )  # Target in reference frame
        weight = 1.0

        # Create JointToJointPositionErrorFunction
        joint_position_error = pym_solver2.JointToJointPositionErrorFunction(character)

        # Test basic properties
        self.assertEqual(joint_position_error.num_constraints(), 0)
        self.assertEqual(len(joint_position_error.constraints), 0)

        # Add a single constraint
        joint_position_error.add_constraint(
            source_joint=source_joint,
            source_offset=source_offset,
            reference_joint=reference_joint,
            reference_offset=reference_offset,
            target=target_position,
            weight=weight,
        )

        # Verify constraint was added
        self.assertEqual(joint_position_error.num_constraints(), 1)
        self.assertEqual(len(joint_position_error.constraints), 1)

        constraint = joint_position_error.constraints[0]
        self.assertEqual(constraint.source_joint, source_joint)
        self.assertEqual(constraint.reference_joint, reference_joint)
        self.assertTrue(np.allclose(constraint.source_offset, source_offset))
        self.assertTrue(np.allclose(constraint.reference_offset, reference_offset))
        self.assertTrue(np.allclose(constraint.target, target_position))
        self.assertAlmostEqual(constraint.weight, weight)

        # Create solver function with the joint position error
        solver_function = pym_solver2.SkeletonSolverFunction(
            character, [joint_position_error]
        )

        # Set solver options
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5

        # Create and run the solver
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )

        # Compute final positions of the points
        source_point_world = pym_skel_state.transform_points(
            skel_state_final[source_joint],
            torch.from_numpy(source_offset),
        )
        reference_point_world = pym_skel_state.transform_points(
            skel_state_final[reference_joint],
            torch.from_numpy(reference_offset),
        )

        # Transform source point into reference frame
        # Get inverse rotation of reference joint
        ref_quat = skel_state_final[reference_joint, 3:7]
        ref_quat_inv = pym_quaternion.inverse(ref_quat)

        # Compute source position in reference frame
        source_in_ref_frame = pym_quaternion.rotate_vector(
            ref_quat_inv, source_point_world - reference_point_world
        )

        # Assert that the source position in reference frame is close to the target
        self.assertTrue(
            torch.allclose(
                source_in_ref_frame,
                torch.from_numpy(target_position),
                rtol=1e-2,
                atol=1e-2,
            ),
            msg=f"Source position in ref frame {source_in_ref_frame.numpy()} does not match target {target_position}",
        )

        # Test multiple constraints using add_constraints
        joint_position_error.clear_constraints()
        self.assertEqual(joint_position_error.num_constraints(), 0)

        # Add multiple constraints
        source_joints = np.array([3, 2], dtype=np.int32)
        source_offsets = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
        reference_joints = np.array([0, 1], dtype=np.int32)
        reference_offsets = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float32
        )
        target_positions = np.array(
            [[0.8, 0.3, 0.0], [0.5, 0.6, 0.2]], dtype=np.float32
        )
        weights = np.array([1.0, 2.0], dtype=np.float32)

        joint_position_error.add_constraints(
            source_joint=source_joints,
            source_offset=source_offsets,
            reference_joint=reference_joints,
            reference_offset=reference_offsets,
            target=target_positions,
            weight=weights,
        )

        # Verify multiple constraints were added
        self.assertEqual(joint_position_error.num_constraints(), 2)
        constraints = joint_position_error.constraints

        # Check first constraint
        self.assertEqual(constraints[0].source_joint, 3)
        self.assertEqual(constraints[0].reference_joint, 0)
        self.assertTrue(np.allclose(constraints[0].source_offset, [0.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(constraints[0].reference_offset, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(constraints[0].target, [0.8, 0.3, 0.0]))
        self.assertAlmostEqual(constraints[0].weight, 1.0)

        # Check second constraint
        self.assertEqual(constraints[1].source_joint, 2)
        self.assertEqual(constraints[1].reference_joint, 1)
        self.assertTrue(np.allclose(constraints[1].source_offset, [0.0, 0.5, 0.0]))
        self.assertTrue(np.allclose(constraints[1].reference_offset, [0.0, 0.0, 0.5]))
        self.assertTrue(np.allclose(constraints[1].target, [0.5, 0.6, 0.2]))
        self.assertAlmostEqual(constraints[1].weight, 2.0)

        # Test string representation
        repr_str = repr(joint_position_error)
        self.assertIn("JointToJointPositionErrorFunction", repr_str)
        self.assertIn("num_constraints=2", repr_str)

        # Test constraint string representation
        constraint_repr = repr(constraints[0])
        self.assertIn("JointToJointPositionData", constraint_repr)
        self.assertIn("source_joint=3", constraint_repr)
        self.assertIn("reference_joint=0", constraint_repr)

    def test_height_error_solver_convergence(self) -> None:
        """Test HeightErrorFunction can solve for the correct height using pose+scale parameters."""

        # Create test character with mesh
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Set target height different from rest pose
        target_height = 4.5

        # Create height error function with required target_height parameter
        height_error = pym_solver2.HeightErrorFunction(
            character,
            target_height=target_height,
            up_direction=np.array([0.0, 1.0, 0.0]),
            k=1,
            weight=1.0,
        )

        # Create solver function
        solver_function = pym_solver2.SkeletonSolverFunction(character, [height_error])

        # Initialize parameters to zero (rest pose)
        model_params_init = np.zeros(n_params, dtype=np.float32)

        # Create solver
        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 50
        solver_options.min_iterations = 10
        solver_options.regularization = 1e-7
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)

        # Solve for target height
        model_params_final = solver.solve(model_params_init)

        # Compute actual height from optimized parameters
        skel_state = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final
            )
        )
        mesh_vertices = character.skin_points(skel_state.numpy())

        # Compute height by projecting all vertices onto the up direction
        up_direction = np.array([0.0, 1.0, 0.0])
        projections = mesh_vertices @ up_direction
        actual_height = projections.max() - projections.min()

        # Check that the actual height is close to the target height
        self.assertTrue(
            np.isclose(actual_height, target_height, rtol=1e-2, atol=1e-2),
            f"Solved height {actual_height} does not match target height {target_height}",
        )

        # Verify that the solver converged to low error
        self.assertGreater(len(solver.per_iteration_errors), 1)
        self.assertLess(
            solver.per_iteration_errors[-1],
            1e-4,
            f"Solver did not converge to low error. Final error: {solver.per_iteration_errors[-1]}",
        )

        # Test with a different target height to ensure solver works consistently
        target_height_2 = 5.5
        height_error.target_height = target_height_2

        # Reset parameters
        model_params_init = np.zeros(n_params, dtype=np.float32)
        model_params_final_2 = solver.solve(model_params_init)

        # Compute actual height from optimized parameters
        skel_state_2 = torch.from_numpy(
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final_2
            )
        )
        mesh_vertices_2 = character.skin_points(skel_state_2.numpy())

        # Compute height
        projections_2 = mesh_vertices_2 @ up_direction
        actual_height_2 = projections_2.max() - projections_2.min()

        # Check that the actual height is close to the second target height
        self.assertTrue(
            np.isclose(actual_height_2, target_height_2, rtol=1e-2, atol=1e-2),
            f"Solved height {actual_height_2} does not match second target height {target_height_2}",
        )

    def test_velocity_magnitude_sequence_error_function(self) -> None:
        """Test VelocityMagnitudeSequenceErrorFunction to ensure joint velocity magnitudes match targets."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size
        n_joints = character.skeleton.size
        n_frames = 2  # VelocityMagnitudeSequenceErrorFunction requires 2 frames

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialize model parameters for two frames
        model_params_init = torch.zeros((n_frames, n_params), dtype=torch.float32)

        # ========================================
        # Test basic construction
        # ========================================
        vel_mag_error = pym_solver2.VelocityMagnitudeSequenceErrorFunction(character)
        self.assertIsNotNone(vel_mag_error)

        # Test construction with parameters
        vel_mag_error_with_params = pym_solver2.VelocityMagnitudeSequenceErrorFunction(
            character,
            weight=2.0,
            joint_weights=np.ones(n_joints, dtype=np.float32),
            target_speed=0.5,
        )
        self.assertIsNotNone(vel_mag_error_with_params)

        # ========================================
        # Test setter methods
        # ========================================
        vel_mag_error.set_target_speed(1.0)

        # Test set_target_speeds (per-joint)
        per_joint_speeds = np.linspace(0.0, 1.0, n_joints).astype(np.float32)
        vel_mag_error.set_target_speeds(per_joint_speeds)

        # Test set_target_weights
        joint_weights = np.ones(n_joints, dtype=np.float32)
        joint_weights[0] = 2.0  # Give first joint more weight
        vel_mag_error.set_target_weights(joint_weights)

        # Test property accessors
        retrieved_speeds = vel_mag_error.target_speeds
        self.assertEqual(len(retrieved_speeds), n_joints)

        retrieved_weights = vel_mag_error.target_weights
        self.assertEqual(len(retrieved_weights), n_joints)

        # Test reset
        vel_mag_error.reset()

        # ========================================
        # Test 1: Zero target speed (stationary constraint)
        # Set up motion where velocity should be penalized
        # ========================================
        # Set up frames with some initial movement
        model_params_init[0, 0] = 0.0  # First frame at x=0
        model_params_init[1, 0] = 1.0  # Second frame at x=1 (moving)

        solver_function = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Anchor first frame
        first_frame_error = pym_solver2.ModelParametersErrorFunction(character)
        first_frame_error.set_target_parameters(
            model_params_init[0].numpy(), np.ones(n_params)
        )
        solver_function.add_error_function(0, first_frame_error)

        # Add velocity magnitude error with zero target (stationary constraint)
        vel_mag_error_zero = pym_solver2.VelocityMagnitudeSequenceErrorFunction(
            character,
            weight=1.0,
            target_speed=0.0,
        )
        solver_function.add_sequence_error_function(0, vel_mag_error_zero)

        solver_options = pym_solver2.SequenceSolverOptions()
        solver_options.max_iterations = 50
        solver_options.regularization = 1e-5

        model_params_final = pym_solver2.solve_sequence(
            solver_function, model_params_init.numpy(), solver_options
        )

        # Convert to skeleton states and verify velocity magnitudes are close to zero
        skel_states = [
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final[i]
            )
            for i in range(n_frames)
        ]

        speeds = []
        for joint_idx in range(n_joints):
            pos0 = skel_states[0][joint_idx, :3]
            pos1 = skel_states[1][joint_idx, :3]

            # Velocity = pos[t+1] - pos[t]
            velocity = pos1 - pos0
            speed = float(np.linalg.norm(velocity))

            speeds.append(speed)
            # Assert that measured speed is close to zero target
            self.assertLess(
                speed,
                1e-3,
                f"Joint {joint_idx}: speed {speed} should be close to 0",
            )

        # ========================================
        # Test 2: Non-zero target speed
        # Set up motion where velocity magnitude should match a target
        # ========================================
        target_speed = 0.5

        # Reset solver
        solver_function2 = pym_solver2.SequenceSolverFunction(character, n_frames)

        # Anchor first frame
        first_frame_error2 = pym_solver2.ModelParametersErrorFunction(character)
        first_frame_error2.set_target_parameters(
            np.zeros(n_params, dtype=np.float32), np.ones(n_params)
        )
        solver_function2.add_error_function(0, first_frame_error2)

        # Add velocity magnitude error with non-zero target
        vel_mag_error_target = pym_solver2.VelocityMagnitudeSequenceErrorFunction(
            character,
            weight=1.0,
            target_speed=target_speed,
        )
        solver_function2.add_sequence_error_function(0, vel_mag_error_target)

        # Start from a configuration with some movement
        model_params_init2 = torch.zeros((n_frames, n_params), dtype=torch.float32)
        model_params_init2[1, 0] = 1.0  # Initial guess with movement in x

        model_params_final2 = pym_solver2.solve_sequence(
            solver_function2, model_params_init2.numpy(), solver_options
        )

        # Convert to skeleton states and verify velocity magnitudes match target
        skel_states2 = [
            pym_geometry.model_parameters_to_skeleton_state(
                character, model_params_final2[i]
            )
            for i in range(n_frames)
        ]

        # Check root joint velocity magnitude
        pos0 = skel_states2[0][0, :3]
        pos1 = skel_states2[1][0, :3]
        measured_speed = float(np.linalg.norm(pos1 - pos0))

        self.assertTrue(
            np.isclose(measured_speed, target_speed, rtol=1e-3, atol=1e-3),
            f"Root joint: measured speed {measured_speed} does not match target {target_speed}",
        )

    def test_sequence_error_function_get_error(self) -> None:
        """Test SequenceErrorFunction.get_error() for sequence error functions."""

        character = pym_geometry.create_test_character(num_joints=4)
        n_params = character.parameter_transform.size

        # Test with ModelParametersSequenceErrorFunction (2-frame window)
        seq_error = pym_solver2.ModelParametersSequenceErrorFunction(
            character, weight=1.0
        )

        # Same parameters on both frames should give zero error
        model_params_same = np.zeros((2, n_params), dtype=np.float32)
        error = seq_error.get_error(model_params_same)
        self.assertAlmostEqual(error, 0.0, places=6)

        # Different parameters on consecutive frames should give non-zero error
        np.random.seed(0)
        model_params_diff = np.random.rand(2, n_params).astype(np.float32)
        error = seq_error.get_error(model_params_diff)
        self.assertGreater(error, 0.0)

        # Manually compute expected error for ModelParametersSequenceErrorFunction:
        # error = sum((params[1][i] - params[0][i])^2) * weight * kMotionWeight
        # where targetWeights = all 1s, weight = 1.0, kMotionWeight = 1e-1
        diff = model_params_diff[1] - model_params_diff[0]
        expected_error = float(np.sum(diff**2)) * 0.1
        self.assertAlmostEqual(error, expected_error, places=4)

        # Test with StateSequenceErrorFunction (2-frame window)
        state_seq_error = pym_solver2.StateSequenceErrorFunction(character, weight=1.0)

        # Same parameters should give zero error
        error = state_seq_error.get_error(model_params_same)
        self.assertAlmostEqual(error, 0.0, places=6)

        # Different parameters should give non-zero error
        error = state_seq_error.get_error(model_params_diff)
        self.assertGreater(error, 0.0)

        # Test error handling: wrong number of dimensions
        with self.assertRaises(RuntimeError):
            seq_error.get_error(np.zeros(n_params, dtype=np.float32))

        # Test error handling: wrong number of frames
        with self.assertRaises(RuntimeError):
            seq_error.get_error(np.zeros((3, n_params), dtype=np.float32))

        # Test error handling: wrong number of parameters
        with self.assertRaises(RuntimeError):
            seq_error.get_error(np.zeros((2, n_params + 1), dtype=np.float32))

    def test_add_camera_intrinsics_parameters(self) -> None:
        """Test adding camera intrinsics parameters for multiple cameras."""
        import pymomentum.camera as pym_camera

        character = pym_geometry.create_test_character(num_joints=4)
        orig_n_params = character.parameter_transform.size

        # Create two cameras
        cam0 = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
        )
        cam0.name = "cam0"

        cam1 = pym_camera.OpenCVFisheyeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
        )
        cam1.name = "cam1"

        # Add cam0 (pinhole: 4 params)
        character2 = pym_solver2.add_camera_intrinsics_parameters(character, cam0)
        self.assertEqual(character2.parameter_transform.size, orig_n_params + 4)

        # Add cam1 (fisheye: 8 params)
        character3 = pym_solver2.add_camera_intrinsics_parameters(character2, cam1)
        self.assertEqual(character3.parameter_transform.size, orig_n_params + 4 + 8)

        # Verify parameter names
        param_names = character3.parameter_transform.names
        cam0_names = [n for n in param_names if n.startswith("intrinsics_cam0_")]
        cam1_names = [n for n in param_names if n.startswith("intrinsics_cam1_")]

        self.assertEqual(len(cam0_names), 4)
        self.assertEqual(len(cam1_names), 8)
        self.assertIn("intrinsics_cam0_fx", cam0_names)
        self.assertIn("intrinsics_cam0_cy", cam0_names)
        self.assertIn("intrinsics_cam1_k1", cam1_names)
        self.assertIn("intrinsics_cam1_k4", cam1_names)

        # Verify idempotency: adding cam0 again should not increase the count
        character4 = pym_solver2.add_camera_intrinsics_parameters(character3, cam0)
        self.assertEqual(
            character4.parameter_transform.size, character3.parameter_transform.size
        )

    def test_extract_and_set_camera_intrinsics(self) -> None:
        """Test round-tripping intrinsics values through model parameters."""
        import pymomentum.camera as pym_camera

        character = pym_geometry.create_test_character(num_joints=4)

        cam = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        cam.name = "cam0"

        character2 = pym_solver2.add_camera_intrinsics_parameters(character, cam)
        n_params = character2.parameter_transform.size

        # set_camera_intrinsics: write model values into model params (returns new array)
        model_params = pym_solver2.set_camera_intrinsics(
            character2, cam, np.zeros(n_params, dtype=np.float32)
        )

        # Extract back and verify round-trip
        extracted = pym_solver2.extract_camera_intrinsics(character2, model_params, cam)
        np.testing.assert_allclose(extracted, [500.0, 500.0, 320.0, 240.0], atol=1e-5)

        # Modify a value in model_params and verify extract picks it up
        param_names = list(character2.parameter_transform.names)
        fx_idx = param_names.index("intrinsics_cam0_fx")
        model_params[fx_idx] = 600.0
        extracted2 = pym_solver2.extract_camera_intrinsics(
            character2, model_params, cam
        )
        self.assertAlmostEqual(float(extracted2[0]), 600.0, places=5)
        self.assertAlmostEqual(float(extracted2[1]), 500.0, places=5)  # fy unchanged

    def test_batched_camera_intrinsics(self) -> None:
        """Test extract/set camera intrinsics with batched (2D) model parameters."""
        import pymomentum.camera as pym_camera

        character = pym_geometry.create_test_character(num_joints=4)

        cam = pym_camera.PinholeIntrinsicsModel(
            image_width=640,
            image_height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )
        cam.name = "cam0"

        character2 = pym_solver2.add_camera_intrinsics_parameters(character, cam)
        n_params = character2.parameter_transform.size

        batch_size = 3
        # Create batched zeros and set intrinsics
        model_params_batch = np.zeros((batch_size, n_params), dtype=np.float32)
        result_batch = pym_solver2.set_camera_intrinsics(
            character2, cam, model_params_batch
        )
        self.assertEqual(result_batch.shape, (batch_size, n_params))

        # Extract intrinsics from the batched result
        extracted_batch = pym_solver2.extract_camera_intrinsics(
            character2, result_batch, cam
        )
        self.assertEqual(extracted_batch.shape, (batch_size, 4))

        # Each batch element should have the same intrinsics
        for i in range(batch_size):
            np.testing.assert_allclose(
                extracted_batch[i], [500.0, 500.0, 320.0, 240.0], atol=1e-5
            )

        # Modify fx in the second batch element and verify
        param_names = list(character2.parameter_transform.names)
        fx_idx = param_names.index("intrinsics_cam0_fx")
        result_batch[1, fx_idx] = 700.0

        extracted_batch2 = pym_solver2.extract_camera_intrinsics(
            character2, result_batch, cam
        )
        self.assertAlmostEqual(float(extracted_batch2[0, 0]), 500.0, places=5)
        self.assertAlmostEqual(float(extracted_batch2[1, 0]), 700.0, places=5)
        self.assertAlmostEqual(float(extracted_batch2[2, 0]), 500.0, places=5)
