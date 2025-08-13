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


def _build_blend_shape_basis(
    c: pym_geometry.Character,
) -> pym_geometry.BlendShape:
    base_shape = c.mesh.vertices
    n_pts = base_shape.shape[0]
    n_blend = 4
    np.random.seed(0)
    shape_vectors = np.random.rand(n_blend, n_pts, 3)
    blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)
    return blend_shape


class TestBlendShape(unittest.TestCase):
    def test_diff_apply_blend_coeffs(self) -> None:
        np.random.seed(0)

        n_pts = 10
        n_blend = 4

        base_shape = np.random.rand(n_pts, 3)
        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)

        nBatch = 2
        n_coeffs = min(blend_shape.n_shapes, 10)
        coeffs = torch.rand(nBatch, n_coeffs, dtype=torch.float64, requires_grad=True)

        shape1 = blend_shape.compute_shape(coeffs).select(0, 0)
        c1 = coeffs.select(0, 0).detach().numpy()

        # Compute the shape another way:
        shape2 = base_shape + np.dot(
            shape_vectors.reshape(n_blend, n_pts * 3).transpose(), c1
        ).reshape(n_pts, 3)
        self.assertTrue(shape1.allclose(torch.from_numpy(shape2).float()))

        torch.autograd.gradcheck(
            blend_shape.compute_shape,
            [coeffs],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

    def test_blend_shape_character(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        np.random.seed(0)

        c = pym_geometry.create_test_character()

        # Build a blend shape basis:
        blend_shape = _build_blend_shape_basis(c)

        c2 = c.with_blend_shape(blend_shape)
        params = torch.rand(c2.parameter_transform.size)
        bp1 = params[c2.parameter_transform.blend_shape_parameters]
        bp2 = pym_geometry.model_parameters_to_blend_shape_coefficients(c2, params)
        self.assertTrue(bp1.allclose(bp2))

    def test_save_and_load(self) -> None:
        torch.manual_seed(0)  # ensure repeatability
        np.random.seed(0)

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

        model_params = torch.rand(n_model_params) * 5.0 - 2.5
        joint_params = pym_geometry.apply_parameter_transform(c, model_params)
        skel_state = pym_geometry.joint_parameters_to_skeleton_state(c, joint_params)

        m1 = torch.from_numpy(c.pose_mesh(joint_params.numpy()).vertices)
        m2 = c.skin_points(skel_state)
        self.assertTrue(m1.allclose(m2))

        rest_points = torch.from_numpy(c.mesh.vertices)
        m3 = c.skin_points(pym_skel_state.to_matrix(skel_state), rest_points)
        self.assertTrue(m1.allclose(m3))

    def test_skinning_check_derivatives(self) -> None:
        """Check the skinning derivatives."""

        torch.set_printoptions(profile="full")

        c = pym_geometry.create_test_character()
        torch.manual_seed(0)  # ensure repeatability
        n_model_params = c.parameter_transform.size

        n_batch = 2
        model_params = (
            torch.rand(n_batch, n_model_params, requires_grad=True, dtype=torch.float64)
            * 5.0
            - 2.5
        )
        joint_params = pym_geometry.apply_parameter_transform(c, model_params)
        skel_state = pym_geometry.joint_parameters_to_skeleton_state(c, joint_params)
        transforms = pym_skel_state.to_matrix(skel_state)

        # Derivatives with default rest points:
        torch.autograd.gradcheck(
            c.skin_points,
            [transforms],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

        # Derivatives with specified rest points:
        rest_points = torch.from_numpy(c.mesh.vertices).double()
        rest_points.requires_grad = True
        torch.autograd.gradcheck(
            c.skin_points,
            [transforms, rest_points],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

    def test_solve_blend_shape(self) -> None:
        c = pym_geometry.create_test_character()
        blend_shape = _build_blend_shape_basis(c)
        c = c.with_blend_shape(blend_shape)
        pt = c.parameter_transform

        gt_model_params = torch.rand(c.parameter_transform.size).masked_fill(
            pt.pose_parameters | pt.scaling_parameters,
            0,
        )
        gt_joint_params = pym_geometry.apply_parameter_transform(c, gt_model_params)
        gt_blend_coeffs = pym_geometry.model_parameters_to_blend_shape_coefficients(
            c, gt_model_params
        )
        gt_rest_shape = blend_shape.compute_shape(gt_blend_coeffs)
        gt_skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            c, gt_joint_params
        )
        gt_posed_shape = c.skin_points(gt_skel_state, gt_rest_shape)

        active_params = c.parameter_transform.blend_shape_parameters
        active_error_functions = [ErrorFunctionType.Limit, ErrorFunctionType.Vertex]
        error_function_weights = torch.ones(
            len(active_error_functions),
            requires_grad=True,
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
        test_joint_params = pym_geometry.apply_parameter_transform(c, test_model_params)
        test_blend_coeffs = pym_geometry.model_parameters_to_blend_shape_coefficients(
            c, test_model_params
        )
        test_rest_shape = blend_shape.compute_shape(test_blend_coeffs)
        test_skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            c, test_joint_params
        )
        test_posed_shape = c.skin_points(test_skel_state, test_rest_shape)
        self.assertTrue(test_posed_shape.allclose(gt_posed_shape, rtol=1e-3, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
