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
import pymomentum.skel_state as pym_skel_state
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

    def test_apply_inverse_parameter_transform_matches(self) -> None:
        """Verify that ParameterTransform.inverse().apply matches diff_geometry.apply_inverse_parameter_transform."""
        np.random.seed(42)

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size

        # Create joint parameters with numpy (flat format)
        joint_params_np = np.random.rand(nBatch, nJoints * 7).astype(np.float32)

        # Call geometry version (numpy) using inverse parameter transform
        inv_param_transform = character.parameter_transform.inverse()
        model_params_geometry = inv_param_transform.apply(joint_params_np)

        # Call diff_geometry version (torch)
        joint_params_torch = torch.from_numpy(joint_params_np)
        model_params_diff_geometry = (
            pym_diff_geometry.apply_inverse_parameter_transform(
                inv_param_transform, joint_params_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                model_params_geometry, model_params_diff_geometry.numpy(), atol=1e-6
            ),
            "ParameterTransform.inverse().apply should match diff_geometry.apply_inverse_parameter_transform",
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

    def test_blend_shape_base_compute_shape_matches(self) -> None:
        """Verify that BlendShapeBase.compute_shape matches diff_geometry.compute_blend_shape."""
        np.random.seed(42)

        n_pts = 10
        n_blend = 4
        nBatch = 2

        # Create BlendShapeBase
        shape_vectors = np.random.rand(n_blend, n_pts, 3).astype(np.float32)
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(shape_vectors)

        # Create coefficients
        n_coeffs = blend_shape.n_shapes
        coeffs_np = np.random.rand(nBatch, n_coeffs).astype(np.float32)

        # Call geometry version (numpy)
        result_geometry = blend_shape.compute_shape(coeffs_np)

        # Call diff_geometry version (torch)
        coeffs_torch = torch.from_numpy(coeffs_np)
        result_diff_geometry = pym_diff_geometry.compute_blend_shape(
            blend_shape, coeffs_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(result_geometry, result_diff_geometry.numpy(), atol=1e-6),
            "BlendShapeBase.compute_shape should match diff_geometry.compute_blend_shape",
        )

    def test_blend_shape_compute_shape_matches(self) -> None:
        """Verify that BlendShape.compute_shape matches diff_geometry.compute_blend_shape."""
        np.random.seed(42)

        n_pts = 10
        n_blend = 4
        nBatch = 2

        # Create BlendShape with base shape
        base_shape = np.random.rand(n_pts, 3).astype(np.float32)
        shape_vectors = np.random.rand(n_blend, n_pts, 3).astype(np.float32)
        blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)

        # Create coefficients
        n_coeffs = blend_shape.n_shapes
        coeffs_np = np.random.rand(nBatch, n_coeffs).astype(np.float32)

        # Call geometry version (numpy)
        result_geometry = blend_shape.compute_shape(coeffs_np)

        # Call diff_geometry version (torch)
        coeffs_torch = torch.from_numpy(coeffs_np)
        result_diff_geometry = pym_diff_geometry.compute_blend_shape(
            blend_shape, coeffs_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(result_geometry, result_diff_geometry.numpy(), atol=1e-6),
            "BlendShape.compute_shape should match diff_geometry.compute_blend_shape",
        )

    def test_blend_shape_with_character_matches(self) -> None:
        """Verify that BlendShape.compute_shape matches for character-based blend shapes."""
        np.random.seed(42)

        # Create a test character with blend shape
        c = pym_geometry.create_test_character()
        n_pts = c.mesh.n_vertices
        n_blend = 4
        nBatch = 2

        # Create BlendShape
        base_shape = np.random.rand(n_pts, 3).astype(np.float32)
        shape_vectors = np.random.rand(n_blend, n_pts, 3).astype(np.float32)
        blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)

        # Create coefficients
        n_coeffs = blend_shape.n_shapes
        coeffs_np = np.random.rand(nBatch, n_coeffs).astype(np.float32)

        # Call geometry version (numpy)
        result_geometry = blend_shape.compute_shape(coeffs_np)

        # Call diff_geometry version (torch)
        coeffs_torch = torch.from_numpy(coeffs_np)
        result_diff_geometry = pym_diff_geometry.compute_blend_shape(
            blend_shape, coeffs_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(result_geometry, result_diff_geometry.numpy(), atol=1e-6),
            "BlendShape.compute_shape with character should match diff_geometry.compute_blend_shape",
        )

    def test_model_parameters_to_blend_shape_coefficients_matches(self) -> None:
        """Verify that geometry.model_parameters_to_blend_shape_coefficients matches diff_geometry version."""
        np.random.seed(42)

        # Create a test character with blend shapes
        character = pym_geometry.create_test_character(with_blendshapes=True)
        nBatch = 2

        # Create model parameters with numpy
        model_params_np = np.random.rand(
            nBatch, character.parameter_transform.size
        ).astype(np.float32)

        # Call geometry version (numpy)
        coeffs_geometry = pym_geometry.model_parameters_to_blend_shape_coefficients(
            character, model_params_np
        )

        # Call diff_geometry version (torch)
        model_params_torch = torch.from_numpy(model_params_np)
        coeffs_diff_geometry = (
            pym_diff_geometry.model_parameters_to_blend_shape_coefficients(
                character, model_params_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(coeffs_geometry, coeffs_diff_geometry.numpy(), atol=1e-6),
            "geometry.model_parameters_to_blend_shape_coefficients should match diff_geometry version",
        )

    def test_model_parameters_to_face_expression_coefficients_matches(self) -> None:
        """Verify that geometry.model_parameters_to_face_expression_coefficients matches diff_geometry version."""
        np.random.seed(42)

        # Create a test character with blend shapes
        character = pym_geometry.create_test_character(with_blendshapes=True)
        nBatch = 2

        # Create model parameters with numpy
        model_params_np = np.random.rand(
            nBatch, character.parameter_transform.size
        ).astype(np.float32)

        # Call geometry version (numpy)
        coeffs_geometry = pym_geometry.model_parameters_to_face_expression_coefficients(
            character, model_params_np
        )

        # Call diff_geometry version (torch)
        model_params_torch = torch.from_numpy(model_params_np)
        coeffs_diff_geometry = (
            pym_diff_geometry.model_parameters_to_face_expression_coefficients(
                character, model_params_torch
            )
        )

        # Compare results
        self.assertTrue(
            np.allclose(coeffs_geometry, coeffs_diff_geometry.numpy(), atol=1e-6),
            "geometry.model_parameters_to_face_expression_coefficients should match diff_geometry version",
        )

    def test_skin_points_matches(self) -> None:
        """Verify that Character.skin_points matches diff_geometry.skin_points."""
        np.random.seed(42)

        # Create a test character with mesh
        character = pym_geometry.create_test_character()
        nBatch = 2

        # Debugging checks for skin weights and skeleton consistency
        nJoints = character.skeleton.size

        # Check that skin weight indices are valid
        if character.skin_weights is not None:
            max_skin_index = np.max(character.skin_weights.index)
            self.assertLess(
                max_skin_index,
                nJoints,
                f"Skin weight indices should be < nJoints. "
                f"Found max index {max_skin_index} but nJoints={nJoints}",
            )

        # Create model parameters and convert to skeleton state
        model_params_np = np.random.rand(
            nBatch, character.parameter_transform.size
        ).astype(np.float32)
        skel_state_np = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_np
        )
        skel_state_torch = torch.from_numpy(skel_state_np)

        # Verify skeleton state shape
        self.assertEqual(
            skel_state_np.shape,
            (nBatch, nJoints, 8),
            f"Skeleton state shape should be ({nBatch}, {nJoints}, 8)",
        )

        # Test 1: With explicit rest vertices
        # Note: rest_vertices should have the same batch dimensions as skel_state
        rest_vertices_base = character.mesh.vertices.astype(np.float32)
        # Expand to have batch dimension: (nVertices, 3) -> (nBatch, nVertices, 3)
        rest_vertices_np = np.tile(
            np.expand_dims(rest_vertices_base, 0), (nBatch, 1, 1)
        )
        rest_vertices_torch = torch.from_numpy(rest_vertices_np)

        skinned_geometry = character.skin_points(skel_state_np, rest_vertices_np)
        skinned_diff_geometry = pym_diff_geometry.skin_points(
            character, skel_state_torch, rest_vertices_torch
        )

        self.assertTrue(
            np.allclose(skinned_geometry, skinned_diff_geometry.numpy(), atol=1e-5),
            f"Character.skin_points should match diff_geometry.skin_points with explicit rest_vertices. "
            f"Max difference: {np.max(np.abs(skinned_geometry - skinned_diff_geometry.numpy()))}",
        )

        # Test 2: Without explicit rest vertices (using default from character.mesh)
        skinned_geometry_default = character.skin_points(skel_state_np)
        skinned_diff_geometry_default = pym_diff_geometry.skin_points(
            character, skel_state_torch
        )

        self.assertTrue(
            np.allclose(
                skinned_geometry_default,
                skinned_diff_geometry_default.numpy(),
                atol=1e-5,
            ),
            f"Character.skin_points should match diff_geometry.skin_points without rest_vertices. "
            f"Max difference: {np.max(np.abs(skinned_geometry_default - skinned_diff_geometry_default.numpy()))}",
        )

        # Test 3: Both approaches should give the same result
        self.assertTrue(
            np.allclose(skinned_geometry, skinned_geometry_default, atol=1e-7),
            "skin_points with explicit rest_vertices should match default rest_vertices from character.mesh",
        )

    def test_compute_vertex_normals_matches(self) -> None:
        """Verify that diff_geometry.compute_vertex_normals matches geometry.compute_vertex_normals."""
        np.random.seed(42)

        # Test with a simple triangle (no batch dimension)
        triangles_np = np.array([[0, 1, 2]], dtype=np.int32)
        vertices_np = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=np.float32)

        # Call geometry version (numpy)
        normals_geometry = pym_geometry.compute_vertex_normals(
            vertices_np, triangles_np
        )

        # Call diff_geometry version (torch)
        normals_diff_geometry = pym_diff_geometry.compute_vertex_normals(
            torch.from_numpy(vertices_np), torch.from_numpy(triangles_np)
        )

        # Compare results
        self.assertTrue(
            np.allclose(normals_geometry, normals_diff_geometry.numpy()),
            "diff_geometry.compute_vertex_normals should match geometry.compute_vertex_normals",
        )

        # Test with batch dimension
        character = pym_geometry.create_test_character(num_joints=5)
        vertex_positions_np = character.mesh.vertices.astype(np.float32)
        triangles_np = character.mesh.faces

        # Create batched input
        vertex_positions_batch_np = np.expand_dims(vertex_positions_np, 0)
        vertex_positions_batch_np = np.tile(vertex_positions_batch_np, (2, 1, 1))

        # Call geometry version (numpy)
        normals_geometry_batch = pym_geometry.compute_vertex_normals(
            vertex_positions_batch_np, triangles_np
        )

        # Call diff_geometry version (torch)
        normals_diff_geometry_batch = pym_diff_geometry.compute_vertex_normals(
            torch.from_numpy(vertex_positions_batch_np), torch.from_numpy(triangles_np)
        )

        # Compare results
        self.assertTrue(
            np.allclose(normals_geometry_batch, normals_diff_geometry_batch.numpy()),
            "Batched diff_geometry.compute_vertex_normals should match geometry.compute_vertex_normals",
        )

    def test_map_model_parameters_matches(self) -> None:
        """Verify that diff_geometry.map_model_parameters matches geometry.map_model_parameters."""
        c = pym_geometry.create_test_character()
        active = np.zeros(c.parameter_transform.size, dtype=bool)
        active[0] = True
        active[3] = True
        active[5] = True

        c2 = pym_geometry.reduce_to_selected_model_parameters(
            c, torch.from_numpy(active)
        )

        np.random.seed(0)
        nBatch = 5
        mp_np = pym_geometry.uniform_random_to_model_parameters(
            c2, np.random.rand(nBatch, c2.parameter_transform.size).astype(np.float32)
        )

        # Call geometry version (numpy)
        mp2_geometry = pym_geometry.map_model_parameters(mp_np, c2, c)
        mp3_geometry = pym_geometry.map_model_parameters(mp2_geometry, c, c2)

        # Call diff_geometry version (torch)
        mp_torch = torch.from_numpy(mp_np)
        mp2_diff_geometry = pym_diff_geometry.map_model_parameters(mp_torch, c2, c)
        mp3_diff_geometry = pym_diff_geometry.map_model_parameters(
            mp2_diff_geometry, c, c2
        )

        # Compare results
        self.assertTrue(
            np.allclose(mp2_geometry, mp2_diff_geometry.numpy()),
            "diff_geometry.map_model_parameters should match geometry.map_model_parameters",
        )
        self.assertTrue(
            np.allclose(mp3_geometry, mp3_diff_geometry.numpy()),
            "Round-trip diff_geometry.map_model_parameters should match geometry.map_model_parameters",
        )

    def test_map_model_parameters_with_names_matches(self) -> None:
        """Verify that diff_geometry.map_model_parameters (with names) matches geometry version."""
        c = pym_geometry.create_test_character()
        c2 = pym_geometry.create_test_character()

        np.random.seed(0)
        nBatch = 5
        mp_np = pym_geometry.uniform_random_to_model_parameters(
            c2, np.random.rand(nBatch, c2.parameter_transform.size).astype(np.float32)
        )

        # Get source parameter names from c2
        source_parameter_names = c2.parameter_transform.names

        # Call geometry version (numpy) using names overload
        mp_mapped_geometry = pym_geometry.map_model_parameters(
            mp_np, source_parameter_names, c, verbose=False
        )

        # Call diff_geometry version (torch)
        mp_torch = torch.from_numpy(mp_np)
        mp_mapped_diff_geometry = pym_diff_geometry.map_model_parameters(
            mp_torch, source_parameter_names, c, verbose=False
        )

        # Compare results
        self.assertTrue(
            np.allclose(mp_mapped_geometry, mp_mapped_diff_geometry.numpy()),
            "diff_geometry.map_model_parameters (with names) should match geometry version",
        )

    def test_map_joint_parameters_matches(self) -> None:
        """Verify that diff_geometry.map_joint_parameters matches geometry.map_joint_parameters."""
        c = pym_geometry.create_test_character()
        c2 = pym_geometry.create_test_character()

        np.random.seed(0)
        nBatch = 5
        mp_np = pym_geometry.uniform_random_to_model_parameters(
            c2, np.random.rand(nBatch, c2.parameter_transform.size).astype(np.float32)
        )
        jp_np = c.parameter_transform.apply(mp_np)

        # Call geometry version (numpy)
        jp2_geometry = pym_geometry.map_joint_parameters(jp_np, c2, c)
        jp3_geometry = pym_geometry.map_joint_parameters(jp2_geometry, c, c2)

        # Call diff_geometry version (torch)
        jp_torch = torch.from_numpy(jp_np)
        jp2_diff_geometry = pym_diff_geometry.map_joint_parameters(jp_torch, c2, c)
        jp3_diff_geometry = pym_diff_geometry.map_joint_parameters(
            jp2_diff_geometry, c, c2
        )

        # Compare results
        self.assertTrue(
            np.allclose(jp2_geometry, jp2_diff_geometry.numpy()),
            "diff_geometry.map_joint_parameters should match geometry.map_joint_parameters",
        )
        self.assertTrue(
            np.allclose(jp3_geometry, jp3_diff_geometry.numpy()),
            "Round-trip diff_geometry.map_joint_parameters should match geometry.map_joint_parameters",
        )

    def test_model_parameters_to_positions_matches(self) -> None:
        """Verify that diff_geometry.model_parameters_to_positions matches geometry.model_parameters_to_positions."""
        torch.manual_seed(0)
        np.random.seed(0)

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size
        nConstraints = 4 * nJoints

        modelParams_np = np.zeros(
            (nBatch, character.parameter_transform.size), dtype=np.float32
        )
        posConstraint_parents_np = np.random.randint(
            low=0, high=character.skeleton.size, size=[nConstraints], dtype=np.int32
        )
        posConstraints_offsets_np = np.random.normal(
            loc=0, scale=4, size=(nConstraints, 3)
        ).astype(np.float32)

        # Call geometry version (numpy)
        positions_geometry = pym_geometry.model_parameters_to_positions(
            character,
            modelParams_np,
            posConstraint_parents_np,
            posConstraints_offsets_np,
        )

        # Call diff_geometry version (torch)
        modelParams_torch = torch.from_numpy(modelParams_np)
        posConstraint_parents_torch = torch.from_numpy(posConstraint_parents_np)
        posConstraints_offsets_torch = torch.from_numpy(posConstraints_offsets_np)

        positions_diff_geometry = pym_diff_geometry.model_parameters_to_positions(
            character,
            modelParams_torch,
            posConstraint_parents_torch,
            posConstraints_offsets_torch,
        )

        # Compare results
        self.assertTrue(
            np.allclose(positions_geometry, positions_diff_geometry.numpy(), atol=1e-5),
            "diff_geometry.model_parameters_to_positions should match geometry.model_parameters_to_positions",
        )

    def test_joint_parameters_to_positions_matches(self) -> None:
        """Verify that diff_geometry.joint_parameters_to_positions matches geometry.joint_parameters_to_positions."""
        torch.manual_seed(0)
        np.random.seed(0)

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size
        nConstraints = 4 * nJoints

        modelParams_np = 0.2 * np.ones(
            (nBatch, character.parameter_transform.size),
            dtype=np.float32,
        )
        jointParams_np = character.parameter_transform.apply(modelParams_np)

        posConstraint_parents_np = np.random.randint(
            low=0, high=character.skeleton.size, size=[nConstraints], dtype=np.int32
        )
        posConstraints_offsets_np = np.random.normal(
            loc=0, scale=4, size=(nConstraints, 3)
        ).astype(np.float32)

        # Call geometry version (numpy)
        positions_geometry = pym_geometry.joint_parameters_to_positions(
            character,
            jointParams_np,
            posConstraint_parents_np,
            posConstraints_offsets_np,
        )

        # Call diff_geometry version (torch)
        jointParams_torch = torch.from_numpy(jointParams_np)
        posConstraint_parents_torch = torch.from_numpy(posConstraint_parents_np)
        posConstraints_offsets_torch = torch.from_numpy(posConstraints_offsets_np)

        positions_diff_geometry = pym_diff_geometry.joint_parameters_to_positions(
            character,
            jointParams_torch,
            posConstraint_parents_torch,
            posConstraints_offsets_torch,
        )

        # Compare results
        self.assertTrue(
            np.allclose(positions_geometry, positions_diff_geometry.numpy(), atol=1e-5),
            "diff_geometry.joint_parameters_to_positions should match geometry.joint_parameters_to_positions",
        )

    def test_skin_points_skeleton_state_vs_matrix(self) -> None:
        """Verify that Character.skin_points produces same results with skeleton state and matrix formats."""
        np.random.seed(42)

        # Create a test character with mesh
        character = pym_geometry.create_test_character()

        nJoints = character.skeleton.size
        nBatch = 2

        # Create model parameters and convert to skeleton state
        model_params_np = np.random.rand(
            nBatch, character.parameter_transform.size
        ).astype(np.float32)
        skel_state_np = pym_geometry.model_parameters_to_skeleton_state(
            character, model_params_np
        )

        # Convert skeleton state to transform matrices using pym_skel_state
        skel_state_torch = torch.from_numpy(skel_state_np)
        transform_matrices_torch = pym_skel_state.to_matrix(skel_state_torch)
        transform_matrices_np = transform_matrices_torch.numpy()

        # Verify shapes
        self.assertEqual(
            skel_state_np.shape,
            (nBatch, nJoints, 8),
            f"Skeleton state shape should be ({nBatch}, {nJoints}, 8)",
        )
        self.assertEqual(
            transform_matrices_np.shape,
            (nBatch, nJoints, 4, 4),
            f"Transform matrices shape should be ({nBatch}, {nJoints}, 4, 4)",
        )

        # Call skin_points with skeleton state format
        skinned_from_skel_state = character.skin_points(skel_state_np)

        # Call skin_points with transform matrix format
        skinned_from_matrices = character.skin_points(transform_matrices_np)

        # Compare results - they should be identical
        max_diff = np.max(np.abs(skinned_from_skel_state - skinned_from_matrices))
        self.assertTrue(
            np.allclose(skinned_from_skel_state, skinned_from_matrices, atol=1e-5),
            f"skin_points with skeleton state should match skin_points with transform matrices. "
            f"Max difference: {max_diff}",
        )

        # Also test with explicit rest vertices
        rest_vertices_np = np.tile(
            np.expand_dims(character.mesh.vertices.astype(np.float32), 0),
            (nBatch, 1, 1),
        )

        skinned_from_skel_state_explicit = character.skin_points(
            skel_state_np, rest_vertices_np
        )
        skinned_from_matrices_explicit = character.skin_points(
            transform_matrices_np, rest_vertices_np
        )

        max_diff_explicit = np.max(
            np.abs(skinned_from_skel_state_explicit - skinned_from_matrices_explicit)
        )
        self.assertTrue(
            np.allclose(
                skinned_from_skel_state_explicit,
                skinned_from_matrices_explicit,
                atol=1e-5,
            ),
            f"skin_points with explicit rest_vertices should match between formats. "
            f"Max difference: {max_diff_explicit}",
        )

    def test_apply_model_param_limits_matches(self) -> None:
        """Verify that Character.apply_model_param_limits matches diff_geometry.apply_model_param_limits."""
        # The test character has only one parameter limit: min-max type [-0.1, 0.1] for root (joint index 0).
        character = pym_geometry.create_test_character()
        n_model_params = character.parameter_transform.size
        np.random.seed(0)
        n_batch = 2

        # Create uniform distribution of [-2.5, 2.5)
        model_params_np = (
            np.random.rand(n_batch, n_model_params).astype(np.float32) * 5.0 - 2.5
        )

        # Call geometry version (numpy)
        clamped_params_geometry = character.apply_model_param_limits(model_params_np)

        # Call diff_geometry version (torch)
        model_params_torch = torch.from_numpy(model_params_np)
        clamped_params_diff_geometry = pym_diff_geometry.apply_model_param_limits(
            character, model_params_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                clamped_params_geometry, clamped_params_diff_geometry.numpy(), atol=1e-6
            ),
            "Character.apply_model_param_limits should match diff_geometry.apply_model_param_limits",
        )

        # Verify limits were applied correctly
        self.assertTrue((clamped_params_geometry[:, 0] <= 0.1).all())
        self.assertTrue((clamped_params_geometry[:, 0] >= -0.1).all())
        # Check no-limit model params are the same
        self.assertTrue(
            np.allclose(clamped_params_geometry[:, 1:], model_params_np[:, 1:])
        )

    def test_find_closest_points_matches(self) -> None:
        """Verify that geometry.find_closest_points matches diff_geometry.find_closest_points."""
        np.random.seed(0)

        n_src_pts = 5
        n_tgt_pts = 20
        n_batch = 2

        # Test both 2D and 3D
        for dim in [2, 3]:
            src_pts_np = np.random.rand(n_batch, n_src_pts, dim).astype(np.float32)
            tgt_pts_np = np.random.rand(n_batch, n_tgt_pts, dim).astype(np.float32)

            # Call geometry version (numpy)
            closest_pts_geometry, indices_geometry, valid_geometry = (
                pym_geometry.find_closest_points(src_pts_np, tgt_pts_np)
            )

            # Call diff_geometry version (torch)
            src_pts_torch = torch.from_numpy(src_pts_np)
            tgt_pts_torch = torch.from_numpy(tgt_pts_np)

            closest_pts_diff_geometry, indices_diff_geometry, valid_diff_geometry = (
                pym_diff_geometry.find_closest_points(src_pts_torch, tgt_pts_torch)
            )

            # Compare results
            self.assertTrue(
                np.allclose(
                    closest_pts_geometry, closest_pts_diff_geometry.numpy(), atol=1e-5
                ),
                f"geometry.find_closest_points (dim={dim}) closest points should match diff_geometry",
            )
            self.assertTrue(
                np.array_equal(indices_geometry, indices_diff_geometry.numpy()),
                f"geometry.find_closest_points (dim={dim}) indices should match diff_geometry",
            )
            self.assertTrue(
                np.array_equal(valid_geometry, valid_diff_geometry.numpy()),
                f"geometry.find_closest_points (dim={dim}) valid should match diff_geometry",
            )

    def test_find_closest_points_with_normals_matches(self) -> None:
        """Verify that geometry.find_closest_points (with normals) matches diff_geometry version."""
        np.random.seed(0)

        n_src_pts = 5
        n_tgt_pts = 20
        n_batch = 2

        # Generate random points and normals
        src_pts_np = np.random.rand(n_batch, n_src_pts, 3).astype(np.float32)
        tgt_pts_np = np.random.rand(n_batch, n_tgt_pts, 3).astype(np.float32)

        # Generate random normalized normals
        src_normals_np = np.random.rand(n_batch, n_src_pts, 3).astype(np.float32)
        src_normals_np = src_normals_np / np.linalg.norm(
            src_normals_np, axis=2, keepdims=True
        )

        tgt_normals_np = np.random.rand(n_batch, n_tgt_pts, 3).astype(np.float32)
        tgt_normals_np = tgt_normals_np / np.linalg.norm(
            tgt_normals_np, axis=2, keepdims=True
        )

        # Call geometry version (numpy)
        (
            closest_pts_geometry,
            closest_normals_geometry,
            indices_geometry,
            valid_geometry,
        ) = pym_geometry.find_closest_points(
            src_pts_np, src_normals_np, tgt_pts_np, tgt_normals_np
        )

        # Call diff_geometry version (torch)
        src_pts_torch = torch.from_numpy(src_pts_np)
        src_normals_torch = torch.from_numpy(src_normals_np)
        tgt_pts_torch = torch.from_numpy(tgt_pts_np)
        tgt_normals_torch = torch.from_numpy(tgt_normals_np)

        (
            closest_pts_diff_geometry,
            closest_normals_diff_geometry,
            indices_diff_geometry,
            valid_diff_geometry,
        ) = pym_diff_geometry.find_closest_points(
            src_pts_torch, src_normals_torch, tgt_pts_torch, tgt_normals_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(
                closest_pts_geometry, closest_pts_diff_geometry.numpy(), atol=1e-5
            ),
            "geometry.find_closest_points (with normals) closest points should match diff_geometry",
        )
        self.assertTrue(
            np.allclose(
                closest_normals_geometry,
                closest_normals_diff_geometry.numpy(),
                atol=1e-5,
            ),
            "geometry.find_closest_points (with normals) closest normals should match diff_geometry",
        )
        self.assertTrue(
            np.array_equal(indices_geometry, indices_diff_geometry.numpy()),
            "geometry.find_closest_points (with normals) indices should match diff_geometry",
        )
        self.assertTrue(
            np.array_equal(valid_geometry, valid_diff_geometry.numpy()),
            "geometry.find_closest_points (with normals) valid should match diff_geometry",
        )

    def test_find_closest_points_on_mesh_matches(self) -> None:
        """Verify that geometry.find_closest_points_on_mesh matches diff_geometry version."""
        # Create a simple mesh
        vertices_np = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        triangles_np = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Create query points
        query_points_np = np.array(
            [[0.5, 0.5, 0.5], [0.25, 0.25, -0.5]], dtype=np.float32
        )

        # Call geometry version (numpy)
        valid_geometry, closest_pts_geometry, face_indices_geometry, bary_geometry = (
            pym_geometry.find_closest_points_on_mesh(
                query_points_np, vertices_np, triangles_np
            )
        )

        # Call diff_geometry version (torch)
        query_points_torch = torch.from_numpy(query_points_np)
        vertices_torch = torch.from_numpy(vertices_np)
        triangles_torch = torch.from_numpy(triangles_np)

        (
            valid_diff_geometry,
            closest_pts_diff_geometry,
            face_indices_diff_geometry,
            bary_diff_geometry,
        ) = pym_diff_geometry.find_closest_points_on_mesh(
            query_points_torch, vertices_torch, triangles_torch
        )

        # Compare results
        self.assertTrue(
            np.array_equal(valid_geometry, valid_diff_geometry.numpy()),
            "geometry.find_closest_points_on_mesh valid should match diff_geometry",
        )
        self.assertTrue(
            np.allclose(
                closest_pts_geometry, closest_pts_diff_geometry.numpy(), atol=1e-5
            ),
            "geometry.find_closest_points_on_mesh closest points should match diff_geometry",
        )
        self.assertTrue(
            np.array_equal(face_indices_geometry, face_indices_diff_geometry.numpy()),
            "geometry.find_closest_points_on_mesh face indices should match diff_geometry",
        )
        self.assertTrue(
            np.allclose(bary_geometry, bary_diff_geometry.numpy(), atol=1e-5),
            "geometry.find_closest_points_on_mesh barycentric coords should match diff_geometry",
        )

    def test_skin_skinned_locators_matches(self) -> None:
        """Verify that Character.skin_skinned_locators matches diff_geometry.skin_skinned_locators."""
        character = pym_geometry.create_test_character()

        # Skip test if character has no skinned locators
        if not character.skinned_locators:
            self.skipTest("Test character has no skinned locators")
            return

        nBatch = 2

        # Create random model parameters
        np.random.seed(0)
        modelParams_np = (
            np.random.rand(nBatch, character.parameter_transform.size).astype(
                np.float32
            )
            * 0.1
        )

        # Convert to skeleton state
        skelState_np = pym_geometry.model_parameters_to_skeleton_state(
            character, modelParams_np
        )

        # Call geometry version (numpy)
        locators_geometry = character.skin_skinned_locators(skelState_np)

        # Call diff_geometry version (torch)
        skelState_torch = torch.from_numpy(skelState_np)
        locators_diff_geometry = pym_diff_geometry.skin_skinned_locators(
            character, skelState_torch
        )

        # Compare results
        self.assertTrue(
            np.allclose(locators_geometry, locators_diff_geometry.numpy(), atol=1e-5),
            "Character.skin_skinned_locators should match diff_geometry.skin_skinned_locators",
        )


if __name__ == "__main__":
    unittest.main()
