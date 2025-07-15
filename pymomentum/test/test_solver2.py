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
        skel_state_target = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_target
        )

        pos_error = pym_solver2.PositionErrorFunction(character)
        pos_error.add_constraints(
            parent=np.arange(n_joints), target=skel_state_target[:, :3].numpy()
        )

        solver_function = pym_solver2.SkeletonSolverFunction(character, [pos_error])

        solver_options = pym_solver2.GaussNewtonSolverOptions()
        solver_options.max_iterations = 100
        solver_options.regularization = 1e-5
        solver = pym_solver2.GaussNewtonSolver(solver_function, solver_options)
        model_params_final = solver.solve(model_params_init.numpy())
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        solver_function.add_sequence_error_function(0, smoothness_error)

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
        skel_state_target = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_target
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        enabled_params = torch.logical_not(
            torch.logical_or(
                character.parameter_transform.scaling_parameters,
                character.parameter_transform.rigid_parameters,
            )
        )
        solver.set_enabled_parameters(enabled_params.numpy())
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
        )

        # Compute the final position of the point
        final_mesh = character.skin_points(skel_state_final)
        final_point_position = final_mesh[point_index, :3]

        # Compute the target position of the point on the triangle
        triangle_vertices = final_mesh[triangle_indices, :3]
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
        solver.set_enabled_parameters(
            character.parameter_transform.rigid_parameters.numpy()
        )
        model_params_final = solver.solve(model_params_init.numpy())

        # Convert final model parameters to skeleton state
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
        )

        # Compute the final position of the vertex
        final_mesh = character.skin_points(skel_state_final)
        final_vertex_position = final_mesh[vertex_index, :3]

        # Assert that the final vertex position is close to the target position
        self.assertTrue(
            torch.allclose(
                final_vertex_position,
                torch.from_numpy(target_position),
                rtol=1e-3,
                atol=1e-3,
            )
        )

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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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

    def test_fixed_axis_constraint(self) -> None:
        """Test FixedAxisErrorFunction to ensure a local axis aligns with a global axis."""

        # Create a test character
        character = pym_geometry.create_test_character(num_joints=4)

        n_params = character.parameter_transform.size

        # Ensure repeatability in the rng:
        torch.manual_seed(0)
        model_params_init = torch.zeros(n_params, dtype=torch.float32)

        # Define local and global axes
        local_axis = _normalize_vec(np.random.randn(3).astype(np.float32))
        target_global_axis = _normalize_vec(np.random.randn(3).astype(np.float32))

        # Create FixedAxisErrorFunction
        fixed_axis_error_function = pym_solver2.FixedAxisDiffErrorFunction(character)

        # Add fixed axis constraint
        parent_idx: int = character.skeleton.size - 1
        fixed_axis_error_function.add_constraint(
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
        )

        # Compute the final local axis in global space
        final_global_axis = pym_quaternion.rotate_vector(
            skel_state_final[parent_idx, 3:7], torch.from_numpy(local_axis)
        )

        # Assert that the final local axis is close to the global axis
        self.assertTrue(
            np.allclose(final_global_axis, target_global_axis, rtol=1e-3, atol=1e-3)
        )

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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
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
        skel_state_final = pym_geometry.model_parameters_to_skeleton_state(
            character, torch.from_numpy(model_params_final)
        )

        # Compute the final mesh
        final_mesh = character.skin_points(skel_state_final)

        # Get the final position of the vertex
        final_vertex_position = final_mesh[vertex_index, :3].numpy()

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
