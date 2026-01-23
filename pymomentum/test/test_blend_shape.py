# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import pymomentum.skel_state as pym_skel_state
import pymomentum.solver as pym_solver
import torch
from pymomentum.solver import ErrorFunctionType

# Flag to check if autograd is enabled (disabled in arvr build modes)
AUTOGRAD_ENABLED: bool = pym_geometry.AUTOGRAD_ENABLED


def _build_shape_vectors(
    c: pym_geometry.Character,
) -> np.ndarray:
    np.random.seed(0)
    n_pts = c.mesh.n_vertices
    n_blend = 4
    shape_vectors = np.random.rand(n_blend, n_pts, 3)
    return shape_vectors


def _build_blend_shape_basis(
    c: pym_geometry.Character,
) -> pym_geometry.BlendShape:
    np.random.seed(0)
    shape_vectors = _build_shape_vectors(c)
    n_pts = shape_vectors.shape[1]
    base_shape = np.random.rand(n_pts, 3)
    blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)
    return blend_shape


def _apply_blend_coeffs(
    blend_shape: pym_geometry.BlendShapeBase,
    base_shape: np.ndarray | None,
    shape_vectors: np.ndarray,
) -> [np.ndarray, np.ndarray, torch.Tensor]:
    n_blend = shape_vectors.shape[0]
    n_pts = shape_vectors.shape[1]

    nBatch = 2
    n_coeffs = min(blend_shape.n_shapes, 10)
    coeffs = torch.rand(
        nBatch, n_coeffs, dtype=torch.float64, requires_grad=AUTOGRAD_ENABLED
    )

    shape1 = blend_shape.compute_shape(coeffs).select(0, 0)
    c1 = coeffs.select(0, 0).detach().numpy()

    # Compute the shape another way:
    shape2 = np.dot(shape_vectors.reshape(n_blend, n_pts * 3).transpose(), c1).reshape(
        n_pts, 3
    )
    if base_shape is not None:
        shape2 += base_shape
    return shape1, torch.from_numpy(shape2).float(), coeffs


class TestBlendShapeBase(unittest.TestCase):
    def test_diff_apply_blend_coeffs(self) -> None:
        np.random.seed(0)

        n_pts = 10
        n_blend = 4

        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(shape_vectors)

        shape1, shape2, coeffs = _apply_blend_coeffs(blend_shape, None, shape_vectors)

        self.assertTrue(shape1.allclose(shape2))
        self.assertTrue(len(blend_shape.shape_names) == n_blend)

        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                blend_shape.compute_shape,
                [coeffs],
                eps=1e-3,
                atol=1e-4,
                raise_exception=True,
            )

    def test_save_and_load(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

        c = pym_geometry.create_test_character()
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(_build_shape_vectors(c))

        bs_bytes = blend_shape.to_bytes()
        blend_shape2 = pym_geometry.BlendShapeBase.from_bytes(bs_bytes)

        self.assertTrue(
            np.allclose(blend_shape.shape_vectors, blend_shape2.shape_vectors)
        )

    def test_blend_shape_character(self) -> None:
        np.random.seed(0)

        c = pym_geometry.create_test_character()

        # Build a set of shape vectors and instantiate as blend shape base
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(_build_shape_vectors(c))
        self.assertTrue(len(blend_shape.shape_names) == 4)
        self.assertTrue(blend_shape.shape_names[0] == "shape_0")

        c2 = c.with_face_expression_blend_shape(blend_shape)
        # Check the right parameters are retrieved
        params = np.random.rand(c2.parameter_transform.size).astype(np.float32)
        bp1 = params[c2.parameter_transform.face_expression_parameters.numpy()]
        bp2 = pym_geometry.model_parameters_to_face_expression_coefficients(c2, params)
        self.assertTrue(np.allclose(bp1, bp2))

        # Check the shape vectors have been passed on correctly
        blend_shape_2 = c2.face_expression_blend_shape
        self.assertTrue(blend_shape_2 is not None)
        self.assertTrue(
            np.allclose(blend_shape_2.shape_vectors, blend_shape.shape_vectors)
        )

        # Check shape vectors are not initialized when not passed
        c3 = c.with_face_expression_blend_shape(None)
        self.assertTrue(c3.face_expression_blend_shape is None)
        self.assertTrue(
            np.sum(c3.parameter_transform.face_expression_parameters.numpy()) == 0
        )

    def test_solve_face_expression_parameters(self) -> None:
        c = pym_geometry.create_test_character()
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(_build_shape_vectors(c))
        c = c.with_face_expression_blend_shape(blend_shape)
        pt = c.parameter_transform

        gt_model_params = torch.rand(c.parameter_transform.size).masked_fill(
            pt.pose_parameters | pt.scaling_parameters | pt.blend_shape_parameters,
            0,
        )
        gt_joint_params = pym_geometry.apply_parameter_transform(
            c, gt_model_params.numpy()
        )
        gt_blend_coeffs = pym_geometry.model_parameters_to_face_expression_coefficients(
            c, gt_model_params.numpy()
        )
        gt_blend_coeffs = torch.from_numpy(gt_blend_coeffs)
        rest_shape = torch.from_numpy(c.mesh.vertices)
        gt_shape = rest_shape + blend_shape.compute_shape(gt_blend_coeffs)
        gt_skel_state = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(c, gt_joint_params)
        )
        gt_posed_shape = c.skin_points(gt_skel_state, gt_shape)

        active_params = c.parameter_transform.face_expression_parameters
        active_error_functions = [ErrorFunctionType.Limit, ErrorFunctionType.Vertex]
        error_function_weights = torch.ones(
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
        )
        model_params_init = torch.zeros(c.parameter_transform.size)

        # Test whether ik works without proj or dist constraints:
        test_model_params = pym_solver.solve_ik(
            character=c,
            active_parameters=active_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            vertex_cons_vertices=torch.arange(0, c.mesh.n_vertices),
            vertex_cons_target_positions=gt_posed_shape,
        )
        test_joint_params = pym_geometry.apply_parameter_transform(
            c, test_model_params.detach().numpy()
        )
        test_blend_coeffs = (
            pym_geometry.model_parameters_to_face_expression_coefficients(
                c, test_model_params.detach().numpy()
            )
        )
        test_blend_coeffs = torch.from_numpy(test_blend_coeffs)
        test_shape = rest_shape + blend_shape.compute_shape(test_blend_coeffs)
        test_skel_state = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(c, test_joint_params)
        )
        test_posed_shape = c.skin_points(test_skel_state, test_shape)

        # Debug output for ASAN mode tolerance issues
        max_abs_diff = torch.max(torch.abs(test_posed_shape - gt_posed_shape)).item()
        max_rel_diff = torch.max(
            torch.abs((test_posed_shape - gt_posed_shape) / (gt_posed_shape + 1e-8))
        ).item()
        print(f"Max absolute difference: {max_abs_diff}")
        print(f"Max relative difference: {max_rel_diff}")

        # Relaxed tolerances for ASAN mode - ASAN has different numerical behavior
        self.assertTrue(test_posed_shape.allclose(gt_posed_shape, rtol=5e-3, atol=5e-2))


class TestBlendShape(unittest.TestCase):
    def test_diff_apply_blend_coeffs(self) -> None:
        np.random.seed(0)

        n_pts = 10
        n_blend = 4

        base_shape = np.random.rand(n_pts, 3)
        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)

        shape1, shape2, coeffs = _apply_blend_coeffs(
            blend_shape, base_shape, shape_vectors
        )

        self.assertTrue(shape1.allclose(shape2))

        if AUTOGRAD_ENABLED:
            torch.autograd.gradcheck(
                blend_shape.compute_shape,
                [coeffs],
                eps=1e-3,
                atol=1e-4,
                raise_exception=True,
            )

    def test_blend_shape_character(self) -> None:
        np.random.seed(0)  # ensure repeatability

        c = pym_geometry.create_test_character()

        # Build a blend shape basis:
        blend_shape = _build_blend_shape_basis(c)

        c2 = c.with_blend_shape(blend_shape)
        params = np.random.rand(c2.parameter_transform.size).astype(np.float32)
        bp1 = params[c2.parameter_transform.blend_shape_parameters.numpy()]
        bp2 = pym_geometry.model_parameters_to_blend_shape_coefficients(c2, params)
        self.assertTrue(np.allclose(bp1, bp2))

        blend_shape_2 = c2.blend_shape
        self.assertTrue(blend_shape_2 is not None)
        self.assertTrue(
            np.allclose(blend_shape_2.shape_vectors, blend_shape.shape_vectors)
        )
        self.assertTrue(np.allclose(blend_shape_2.base_shape, blend_shape.base_shape))

        c3 = c.with_blend_shape(None)
        self.assertTrue(c3.blend_shape is None)
        self.assertTrue(
            np.sum(c3.parameter_transform.blend_shape_parameters.numpy()) == 0
        )

    def test_save_and_load(self) -> None:
        np.random.seed(0)  # ensure repeatability

        c = pym_geometry.create_test_character()

        # Build a blend shape basis:
        blend_shape = _build_blend_shape_basis(c)

        bs_bytes = blend_shape.to_bytes()
        blend_shape2 = pym_geometry.BlendShape.from_bytes(bs_bytes)

        self.assertTrue(np.allclose(blend_shape.base_shape, blend_shape2.base_shape))
        self.assertTrue(
            np.allclose(blend_shape.shape_vectors, blend_shape2.shape_vectors)
        )

    def test_skinning_compare_momentum(self) -> None:
        """Compare the pymomentum skinning against the native momentum skinning."""

        c = pym_geometry.create_test_character()
        torch.manual_seed(0)  # ensure repeatability
        n_model_params = c.parameter_transform.size

        model_params = np.random.rand(n_model_params).astype(np.float32) * 5.0 - 2.5
        joint_params = pym_geometry.apply_parameter_transform(c, model_params)
        joint_params_tensor = torch.from_numpy(joint_params)
        skel_state = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(c, joint_params)
        )

        m1 = torch.from_numpy(c.pose_mesh(joint_params_tensor).vertices)
        m2 = c.skin_points(skel_state)
        self.assertTrue(m1.allclose(m2, rtol=1e-5, atol=1e-6))

        rest_points = torch.from_numpy(c.mesh.vertices)
        m3 = c.skin_points(pym_skel_state.to_matrix(skel_state), rest_points)
        self.assertTrue(m1.allclose(m3, rtol=1e-5, atol=1e-6))

    # NOTE: Skinning derivative tests are now in test_diff_geometry.py
    # (test_skinning_check_derivatives) since pymomentum.geometry uses numpy
    # arrays without gradient support. Use pymomentum.diff_geometry for
    # differentiable operations with PyTorch autograd.

    def test_solve_blend_shape(self) -> None:
        c = pym_geometry.create_test_character()
        blend_shape = _build_blend_shape_basis(c)
        c = c.with_blend_shape(blend_shape)
        pt = c.parameter_transform

        gt_model_params = torch.rand(c.parameter_transform.size).masked_fill(
            pt.pose_parameters | pt.scaling_parameters,
            0,
        )
        gt_joint_params = pym_geometry.apply_parameter_transform(
            c, gt_model_params.numpy()
        )
        gt_blend_coeffs = pym_geometry.model_parameters_to_blend_shape_coefficients(
            c, gt_model_params.numpy()
        )
        gt_blend_coeffs = torch.from_numpy(gt_blend_coeffs)
        gt_rest_shape = blend_shape.compute_shape(gt_blend_coeffs)
        gt_skel_state = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(c, gt_joint_params)
        )
        gt_posed_shape = c.skin_points(gt_skel_state, gt_rest_shape)

        active_params = c.parameter_transform.blend_shape_parameters
        active_error_functions = [ErrorFunctionType.Limit, ErrorFunctionType.Vertex]
        error_function_weights = torch.ones(
            len(active_error_functions),
            requires_grad=AUTOGRAD_ENABLED,
        )
        model_params_init = torch.zeros(c.parameter_transform.size)

        # Test whether ik works without proj or dist constraints:
        test_model_params = pym_solver.solve_ik(
            character=c,
            active_parameters=active_params,
            model_parameters_init=model_params_init,
            active_error_functions=active_error_functions,
            error_function_weights=error_function_weights,
            vertex_cons_vertices=torch.arange(0, c.mesh.n_vertices),
            vertex_cons_target_positions=gt_posed_shape,
        )
        test_joint_params = pym_geometry.apply_parameter_transform(
            c, test_model_params.detach().numpy()
        )
        test_blend_coeffs = pym_geometry.model_parameters_to_blend_shape_coefficients(
            c, test_model_params.detach().numpy()
        )
        test_blend_coeffs = torch.from_numpy(test_blend_coeffs)
        test_rest_shape = blend_shape.compute_shape(test_blend_coeffs)
        test_skel_state = torch.from_numpy(
            pym_geometry.joint_parameters_to_skeleton_state(c, test_joint_params)
        )
        test_posed_shape = c.skin_points(test_skel_state, test_rest_shape)

        # Debug output for ASAN mode tolerance issues
        max_abs_diff = torch.max(torch.abs(test_posed_shape - gt_posed_shape)).item()
        max_rel_diff = torch.max(
            torch.abs((test_posed_shape - gt_posed_shape) / (gt_posed_shape + 1e-8))
        ).item()
        print(f"Max absolute difference: {max_abs_diff}")
        print(f"Max relative difference: {max_rel_diff}")

        # Relaxed tolerances for ASAN mode - ASAN has different numerical behavior
        self.assertTrue(test_posed_shape.allclose(gt_posed_shape, rtol=5e-3, atol=5e-2))

    def test_bake_blend_shape(self) -> None:
        """Test the bake_blend_shape method with numpy arrays."""
        np.random.seed(0)  # ensure repeatability

        # Create test character with blend shapes
        c = pym_geometry.create_test_character()
        blend_shape = _build_blend_shape_basis(c)
        c_with_blend = c.with_blend_shape(blend_shape)

        # Create test blend weights as numpy array
        n_blend_shapes = blend_shape.n_shapes
        blend_weights = np.random.rand(n_blend_shapes).astype(np.float32)

        # Test bake_blend_shape method
        c_baked = c_with_blend.bake_blend_shape(blend_weights)

        # Verify the character structure is preserved
        self.assertEqual(c_baked.skeleton.size, c_with_blend.skeleton.size)
        self.assertEqual(c_baked.name, c_with_blend.name)
        self.assertTrue(c_baked.mesh is not None)
        self.assertTrue(c_baked.skin_weights is not None)

        # Verify blend shape parameters are removed from parameter transform
        # Original character should have blend shape parameters
        self.assertGreater(
            np.sum(c_with_blend.parameter_transform.blend_shape_parameters.numpy()), 0
        )
        # Baked character should have no blend shape parameters
        self.assertEqual(
            np.sum(c_baked.parameter_transform.blend_shape_parameters.numpy()), 0
        )

        # The parameter transform should be smaller (no blend shape parameters)
        self.assertLess(
            c_baked.parameter_transform.size, c_with_blend.parameter_transform.size
        )

        # Verify the baked mesh matches the expected blend shape result
        expected_mesh_vertices = (
            blend_shape.compute_shape(torch.from_numpy(blend_weights).unsqueeze(0))
            .squeeze(0)
            .numpy()
        )
        baked_mesh_vertices = c_baked.mesh.vertices
        self.assertTrue(
            np.allclose(
                baked_mesh_vertices, expected_mesh_vertices, rtol=1e-5, atol=1e-6
            )
        )

        # Test with different array types and shapes
        # Test float64 array
        blend_weights_double = blend_weights.astype(np.float64)
        c_baked_double = c_with_blend.bake_blend_shape(blend_weights_double)
        baked_vertices_double = c_baked_double.mesh.vertices
        self.assertTrue(
            np.allclose(
                baked_vertices_double, expected_mesh_vertices, rtol=1e-5, atol=1e-6
            )
        )

        # Test error handling - should raise exception for non-1D array
        with self.assertRaises(RuntimeError):
            c_with_blend.bake_blend_shape(np.random.rand(2, n_blend_shapes))

        # Test zero blend weights (should result in base shape)
        zero_weights = np.zeros(n_blend_shapes, dtype=np.float32)
        c_baked_zero = c_with_blend.bake_blend_shape(zero_weights)
        expected_base_vertices = blend_shape.base_shape
        baked_base_vertices = c_baked_zero.mesh.vertices
        self.assertTrue(
            np.allclose(
                baked_base_vertices, expected_base_vertices, rtol=1e-5, atol=1e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
