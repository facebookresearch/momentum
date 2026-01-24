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


if __name__ == "__main__":
    unittest.main()
