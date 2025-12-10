# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Tests for pymomentum.diff_geometry module.

This file contains all torch-dependent tests migrated from test_geometry.py and test_blend_shape.py.
These tests verify that differentiable operations work correctly with PyTorch autograd.
"""

import math
import unittest

import numpy as np
import pymomentum.diff_geometry as pym_diff_geometry
import pymomentum.geometry as pym_geometry
import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import torch


def _brute_force_closest_points(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    tgt_normals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Brute force closest point computation for testing."""
    n_batch = src_pts.size(0)
    n_src_pts = src_pts.size(1)

    closest_pts = torch.zeros(src_pts.shape)
    closest_normals = torch.zeros(src_pts.shape)
    closest_index = torch.zeros(n_batch, n_src_pts, dtype=torch.int)

    for i_batch in range(n_batch):
        for j_pt in range(n_src_pts):
            gt_diff = (
                src_pts[i_batch, j_pt, :].expand_as(tgt_pts[i_batch, ...])
                - tgt_pts[i_batch, ...]
            )
            gt_dist = torch.linalg.norm(gt_diff, dim=-1, keepdim=False)
            gt_closest_dist, gt_closest_pt_idx = torch.min(gt_dist, dim=0)
            closest_index[i_batch, j_pt] = gt_closest_pt_idx
            closest_pts[i_batch, j_pt, :] = tgt_pts[i_batch, gt_closest_pt_idx, :]
            if tgt_normals is not None:
                closest_normals[i_batch, j_pt, :] = tgt_normals[
                    i_batch, gt_closest_pt_idx, :
                ]

    return closest_pts, closest_normals, closest_index, closest_index >= 0


class TestDiffGeometry(unittest.TestCase):
    """Tests for differentiable geometry operations in diff_geometry module."""

    def test_diffParamTransform(self) -> None:
        character = pym_geometry.create_test_character()
        torch.manual_seed(0)  # ensure repeatability

        nBatch = 5
        nParams = character.parameter_transform.size
        modelParams = pym_diff_geometry.uniform_random_to_model_parameters(
            character, torch.rand(nBatch, nParams)
        ).double()
        modelParams.requires_grad = True

        jp = character.parameter_transform.apply(modelParams)

        for i in range(nBatch):
            jp1 = character.parameter_transform.apply(modelParams.select(0, i))
            self.assertTrue(jp1.allclose(jp.select(0, i)))

        inputs = [character, modelParams]
        torch.autograd.gradcheck(
            pym_diff_geometry.apply_parameter_transform,
            inputs,
            raise_exception=True,
        )

    def test_diffInverseParamTransform(self) -> None:
        character = pym_geometry.create_test_character()
        invTransform = character.parameter_transform.inverse()

        nBatch = 2
        # Some nonzero vector as 0 always maps to 0 in momentum:
        modelParams = 0.3 * torch.ones(
            nBatch, character.parameter_transform.size, requires_grad=True
        )
        jointParams = character.parameter_transform.apply(modelParams)

        # Validate that inversion works correctly:
        modelParams2 = invTransform.apply(jointParams)
        diff = torch.norm(modelParams2 - modelParams)
        self.assertTrue(diff < 0.001)

        # Validate the gradients.
        inputs = [jointParams]
        torch.autograd.gradcheck(
            invTransform.apply, inputs, eps=1e-2, atol=1e-3, raise_exception=True
        )

    def test_diffmodel_parameters_to_positions(self) -> None:
        torch.manual_seed(0)  # ensure repeatability

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size
        modelParams = torch.zeros(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )
        nConstraints = 4 * nJoints
        posConstraint_parents = torch.randint(
            low=0,
            high=character.skeleton.size,
            size=[nConstraints],
        )
        posConstraints_offsets = torch.normal(mean=0, std=4, size=(nConstraints, 3))
        inputs = [character, modelParams, posConstraint_parents, posConstraints_offsets]
        torch.autograd.gradcheck(
            pym_diff_geometry.model_parameters_to_positions,
            inputs,
            raise_exception=True,
        )

    def test_multiCharacter(self) -> None:
        # Test basic multi-character support.
        torch.manual_seed(0)  # ensure repeatability

        nBatch = 3

        # Generate a list of characters with different sizes:
        characters = [
            pym_geometry.create_test_character().scaled(i + 1) for i in range(nBatch)
        ]

        # Apply some random parameters and verify that uniform_random_to_model_parameters of the
        # whole batch produces the same result as applying to each character separately.
        nJoints = characters[0].skeleton.size
        nParams = characters[0].parameter_transform.size
        modelParams = pym_diff_geometry.uniform_random_to_model_parameters(
            characters[0], torch.rand(nBatch, nParams)
        )
        posConstraint_parents = torch.Tensor(list(range(nJoints)))
        posConstraint_offsets = torch.zeros(nJoints, 3, requires_grad=True)

        positions_full = pym_diff_geometry.model_parameters_to_positions(
            characters, modelParams, posConstraint_parents, posConstraint_offsets
        )

        for i in range(nBatch):
            p_cur = pym_diff_geometry.model_parameters_to_positions(
                characters[i],
                modelParams.select(0, i),
                posConstraint_parents,
                posConstraint_offsets,
            )
            self.assertLess(
                torch.linalg.norm(p_cur - positions_full.select(0, i)), 1e-3
            )

    def test_model_parameters_to_positions_inputs(self) -> None:
        torch.manual_seed(0)  # ensure repeatability

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size
        modelParams = torch.zeros(nBatch, character.parameter_transform.size)

        nConstraints = 4 * nJoints
        posConstraint_parents = torch.randint(
            low=0,
            high=character.skeleton.size,
            size=[nConstraints],
        )
        posConstraint_offsets = torch.zeros(nJoints, 3, requires_grad=True)

        def bad_joints(
            character: pym_geometry.Character = character,
            modelParams: torch.Tensor = modelParams,
            posConstraint_parents: torch.Tensor = posConstraint_parents,
            posConstraint_offsets: torch.Tensor = posConstraint_offsets,
        ) -> None:
            pym_diff_geometry.model_parameters_to_positions(
                character,
                modelParams,
                posConstraint_parents + 100,
                posConstraint_offsets,
            )

        self.assertRaises(RuntimeError, bad_joints)

    def test_diffjoint_parameters_to_positions(self) -> None:
        torch.manual_seed(0)  # ensure repeatability

        character = pym_geometry.create_test_character()
        nBatch = 2
        nJoints = character.skeleton.size
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )
        jointParams = character.parameter_transform.apply(modelParams).type(
            torch.float64
        )

        nConstraints = 4 * nJoints
        posConstraint_parents = torch.randint(
            low=0,
            high=character.skeleton.size,
            size=[nConstraints],
        )
        posConstraints_offsets = torch.normal(
            mean=0, std=4, size=(nConstraints, 3), dtype=torch.float64
        )

        positions_1 = pym_diff_geometry.model_parameters_to_positions(
            character,
            modelParams,
            posConstraint_parents,
            posConstraints_offsets,
        ).type(torch.float64)
        positions_2 = pym_diff_geometry.joint_parameters_to_positions(
            character, jointParams, posConstraint_parents, posConstraints_offsets
        )
        self.assertTrue(positions_1.allclose(positions_2))

        inputs = [character, jointParams, posConstraint_parents, posConstraints_offsets]
        torch.autograd.gradcheck(
            pym_diff_geometry.joint_parameters_to_positions,
            inputs,
            raise_exception=True,
        )

    def test_diffmodel_parameters_to_skeleton_state(self) -> None:
        character = pym_geometry.create_test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )
        inputs = [character, modelParams]
        torch.autograd.gradcheck(
            pym_diff_geometry.model_parameters_to_skeleton_state,
            inputs,
            eps=1e-2,
            atol=1e-3,
            raise_exception=True,
        )

    def test_diffmodel_parameters_to_local_skeleton_state(self) -> None:
        character = pym_geometry.create_test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )

        local_skel_state = pym_diff_geometry.model_parameters_to_local_skeleton_state(
            character,
            modelParams,
        )
        global_skel_state = pym_diff_geometry.model_parameters_to_skeleton_state(
            character,
            modelParams,
        )
        # Check root joint states are the same in both local and global.
        self.assertTrue(
            torch.allclose(
                local_skel_state.select(-2, 0),
                global_skel_state.select(-2, 0),
            )
        )

        local_mat = pym_skel_state.to_matrix(local_skel_state)
        global_mat = pym_skel_state.to_matrix(global_skel_state)

        accum = torch.zeros_like(local_mat)
        for i in range(character.skeleton.size):
            parent = character.skeleton.get_parent(i)
            if parent >= 0:
                accum[:, i, :, :] = torch.bmm(
                    accum[:, parent, :, :], local_mat[:, i, :, :]
                )
            else:
                accum[:, i, :, :] = local_mat[:, i, :, :]

        self.assertTrue(torch.allclose(accum, global_mat))

        inputs = [character, modelParams]
        torch.autograd.gradcheck(
            pym_diff_geometry.model_parameters_to_local_skeleton_state,
            inputs,
            eps=1e-2,
            atol=1e-3,
            raise_exception=True,
        )

    def test_local_skeleton_state_to_joint_parameters(self) -> None:
        character = pym_geometry.create_test_character()
        n_batch = 2
        model_params = 0.2 * torch.ones(
            n_batch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )

        joint_params1 = character.parameter_transform.apply(model_params).reshape(
            n_batch, -1, 7
        )

        local_skel_state1 = pym_diff_geometry.joint_parameters_to_local_skeleton_state(
            character,
            joint_params1,
        )
        t1, _r1, s1 = pym_skel_state.split(local_skel_state1)
        joint_params2 = pym_diff_geometry.local_skeleton_state_to_joint_parameters(
            character, local_skel_state1
        )

        # verify the translations match
        self.assertTrue(
            torch.allclose(joint_params2[..., 0:3], joint_params1[..., 0:3])
        )

        local_skel_state2 = pym_diff_geometry.joint_parameters_to_local_skeleton_state(
            character,
            joint_params2,
        )
        t2, _r2, s2 = pym_skel_state.split(local_skel_state2)

        self.assertTrue(torch.allclose(t1, t2))
        self.assertTrue(torch.allclose(s1, s2))

        # Nonuniqueness bites us in two ways: joint parameters are non-unique due to Euler angle
        # weirdness and skel_states are nonunique due to quaternion q=-q.  So the only thing that
        # is safe to compare here are matrices:
        self.assertTrue(
            torch.allclose(
                pym_skel_state.to_matrix(local_skel_state1),
                pym_skel_state.to_matrix(local_skel_state2),
            )
        )

    def test_skeleton_state_to_joint_parameters(self) -> None:
        character = pym_geometry.create_test_character()
        n_batch = 2
        model_params = 0.2 * torch.ones(
            n_batch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )

        joint_params1 = character.parameter_transform.apply(model_params).reshape(
            n_batch, -1, 7
        )

        skel_state1 = pym_diff_geometry.joint_parameters_to_skeleton_state(
            character,
            joint_params1,
        )
        t1, _r1, s1 = pym_skel_state.split(skel_state1)
        joint_params2 = pym_diff_geometry.skeleton_state_to_joint_parameters(
            character, skel_state1
        )

        # verify the translations match
        self.assertTrue(
            torch.allclose(joint_params2[..., 0:3], joint_params1[..., 0:3])
        )

        skel_state2 = pym_diff_geometry.joint_parameters_to_skeleton_state(
            character,
            joint_params2,
        )
        t2, _r2, s2 = pym_skel_state.split(skel_state2)

        self.assertTrue(torch.allclose(t1, t2))
        self.assertTrue(torch.allclose(s1, s2))

        # Nonuniqueness bites us in two ways: joint parameters are non-unique due to Euler angle
        # weirdness and skel_states are nonunique due to quaternion q=-q.  So the only thing that
        # is safe to compare here are matrices:
        self.assertTrue(
            torch.allclose(
                pym_skel_state.to_matrix(skel_state1),
                pym_skel_state.to_matrix(skel_state2),
            )
        )

    def test_typePromotion(self) -> None:
        # Verify that float/double promotion is working.
        character = pym_geometry.create_test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )

        mp_float = pym_diff_geometry.model_parameters_to_skeleton_state(
            character,
            modelParams.float(),
        )
        mp_double = pym_diff_geometry.model_parameters_to_skeleton_state(
            character,
            modelParams.double(),
        )

        self.assertEqual(mp_float.dtype, torch.float32)
        self.assertEqual(mp_double.dtype, torch.float64)

    def test_diffjoint_parameters_to_skeleton_state(self) -> None:
        character = pym_geometry.create_test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )
        jointParams = character.parameter_transform.apply(modelParams)
        inputs = [character, jointParams]
        torch.autograd.gradcheck(
            pym_diff_geometry.joint_parameters_to_skeleton_state,
            inputs,
            eps=1e-2,
            atol=1e-3,
            raise_exception=True,
        )

    def test_map_model_parameters(self) -> None:
        # map onto a "reduced" character and back:
        c = pym_geometry.create_test_character()
        active = torch.zeros(c.parameter_transform.size, dtype=torch.bool)
        active[0] = True
        active[3] = True
        active[5] = True

        c2 = pym_diff_geometry.reduce_to_selected_model_parameters(c, active)

        torch.manual_seed(0)  # ensure repeatability

        nBatch = 5
        mp = pym_diff_geometry.uniform_random_to_model_parameters(
            c2, torch.rand(nBatch, c2.parameter_transform.size)
        )

        mp2 = pym_diff_geometry.map_model_parameters(mp, c2, c)
        mp3 = pym_diff_geometry.map_model_parameters(mp2, c, c2)

        self.assertTrue(mp.allclose(mp2.index_select(-1, torch.LongTensor([0, 3, 5]))))
        self.assertTrue(mp3.allclose(mp))

    def test_map_joint_parameters(self) -> None:
        # just check that it's idempotent for now:
        c = pym_geometry.create_test_character()
        c2 = pym_geometry.create_test_character()

        torch.manual_seed(0)  # ensure repeatability
        nBatch = 5
        jp = c.parameter_transform.apply(
            pym_diff_geometry.uniform_random_to_model_parameters(
                c2, torch.rand(nBatch, c2.parameter_transform.size)
            )
        )

        jp2 = pym_diff_geometry.map_joint_parameters(jp, c2, c)
        jp3 = pym_diff_geometry.map_joint_parameters(jp2, c, c2)

        self.assertTrue(jp.allclose(jp2))
        self.assertTrue(jp2.allclose(jp3))

    def test_apply_limits(self) -> None:
        # The test character has only one parameter limit: min-max type [-0.1, 0.1] for root (joint index 0).
        c = pym_geometry.create_test_character()
        n_model_params = c.parameter_transform.size
        torch.manual_seed(0)  # ensure repeatability
        n_batch = 2
        # Create uniform distribution of [-2.5, 2.5)
        model_params = (
            torch.rand(n_batch, n_model_params, requires_grad=True) * 5.0 - 2.5
        )
        clamped_params = pym_diff_geometry.apply_model_param_limits(c, model_params)

        self.assertTrue(clamped_params.shape == model_params.shape)
        # Check differentiability:
        clamped_params.backward(gradient=torch.ones(clamped_params.shape))
        # Check no-limit model params are the same:
        self.assertTrue((clamped_params[:, 1:] == model_params[:, 1:]).all())
        self.assertTrue((clamped_params[:, 0] <= 0.1).all())
        self.assertTrue((clamped_params[:, 0] >= -0.1).all())

    def test_compute_vertex_normals(self) -> None:
        # Test with a single triangle (no batch dimension)
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        vertices = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=torch.float32)
        vertex_normals = pym_diff_geometry.compute_vertex_normals(vertices, triangles)
        expected = (
            torch.tensor([1, 0, 0], dtype=torch.float32).unsqueeze(0).expand([3, 3])
        )
        self.assertTrue(torch.allclose(expected, vertex_normals))

        # Test with multiple triangles sharing vertices (no batch dimension)
        # Create a simple square made of two triangles
        vertices_multi = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float32
        )
        triangles_multi = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
        vertex_normals_multi = pym_diff_geometry.compute_vertex_normals(
            vertices_multi, triangles_multi
        )
        # All normals should point in the z direction for this planar mesh
        expected_multi = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(vertex_normals_multi, expected_multi))
        self.assertEqual(vertex_normals_multi.shape, (4, 3))

        # Test with batch dimension
        vertices_batch = vertices_multi.unsqueeze(0).expand(2, -1, -1)
        vertex_normals_batch = pym_diff_geometry.compute_vertex_normals(
            vertices_batch, triangles_multi
        )
        self.assertEqual(vertex_normals_batch.shape, (2, 4, 3))
        # Verify both batches have the same normals
        self.assertTrue(torch.allclose(vertex_normals_batch[0], expected_multi))
        self.assertTrue(torch.allclose(vertex_normals_batch[1], expected_multi))

    def test_scale_character_skinning(self) -> None:
        c = pym_geometry.create_test_character()
        c_scaled = c.scaled(10.0)
        self.assertEqual(c.mesh.vertices.shape, c_scaled.mesh.vertices.shape)
        self.assertTrue(np.allclose(10.0 * c.mesh.vertices, c_scaled.mesh.vertices))

        model_params = pym_diff_geometry.uniform_random_to_model_parameters(
            c, torch.rand(1, c.parameter_transform.size)
        )
        # Don't apply root translations because they are affected by scale:
        model_params[0:3] = 0

        skel_state = pym_diff_geometry.model_parameters_to_skeleton_state(
            c, model_params
        )[0]
        skel_state_scaled = pym_diff_geometry.model_parameters_to_skeleton_state(
            c_scaled, model_params
        )[0]

        posed_mesh = pym_diff_geometry.skin_points(c, skel_state)
        posed_mesh_scaled = pym_diff_geometry.skin_points(c_scaled, skel_state_scaled)

        self.assertEqual(posed_mesh.shape, posed_mesh_scaled.shape)
        self.assertTrue((10.0 * posed_mesh).allclose(posed_mesh_scaled))

    def test_transform_character_skinning(self) -> None:
        c = pym_geometry.create_test_character()
        xf = pym_skel_state.from_quaternion(
            pym_quaternion.euler_xyz_to_quaternion(
                torch.FloatTensor([[0, math.pi / 4.0, math.pi]])
            )
        ).squeeze()
        xf[0] = 10.0

        c_transformed: pym_geometry.Character = c.transformed(
            pym_skel_state.to_matrix(xf).numpy()
        )

        self.assertEqual(c.mesh.vertices.shape, c_transformed.mesh.vertices.shape)
        self.assertTrue(
            torch.allclose(
                pym_skel_state.transform_points(xf, torch.from_numpy(c.mesh.vertices)),
                torch.from_numpy(c_transformed.mesh.vertices),
            )
        )

        model_params = pym_diff_geometry.uniform_random_to_model_parameters(
            c, torch.rand(1, c.parameter_transform.size)
        )
        # Don't apply root translations because they are affected by scale:
        model_params[0:3] = 0

        skel_state = pym_diff_geometry.model_parameters_to_skeleton_state(
            c, model_params
        )[0]
        skel_state_transformed = pym_diff_geometry.model_parameters_to_skeleton_state(
            c_transformed, model_params
        )[0]

        posed_mesh = pym_diff_geometry.skin_points(c, skel_state)
        posed_mesh_transformed = pym_diff_geometry.skin_points(
            c_transformed, skel_state_transformed
        )

        self.assertEqual(posed_mesh.shape, posed_mesh_transformed.shape)
        self.assertTrue(
            torch.allclose(
                pym_skel_state.transform_points(xf, posed_mesh),
                posed_mesh_transformed,
                atol=1e-4,
            )
        )

    def test_skin_skinned_locators(self) -> None:
        """Test the skin_skinned_locators function using mesh vertices as ground truth."""
        torch.manual_seed(42)  # ensure repeatability

        # Create a test character that has mesh and skin weights
        c = pym_geometry.create_test_character(num_joints=5)
        self.assertIsNotNone(c.mesh)
        self.assertIsNotNone(c.skin_weights)

        # Get mesh vertices and their skinning data
        n_vertices = c.mesh.vertices.shape[0]
        if n_vertices == 0:
            self.skipTest("Test character has no mesh vertices")
            return

        # Randomly select 5 mesh vertices to use as skinned locators
        np.random.seed(42)
        n_locators = min(5, n_vertices)
        vertex_indices = np.random.choice(n_vertices, n_locators, replace=False)

        # Create skinned locators using the selected vertices' data
        skinned_locators = [
            pym_geometry.SkinnedLocator(
                name=f"vertex_locator_{vertex_idx}",
                parents=c.skin_weights.index[vertex_idx].astype(np.uint32),
                skin_weights=c.skin_weights.weight[vertex_idx].astype(np.float32),
                position=c.mesh.vertices[vertex_idx].astype(np.float32),
                weight=1.0,
            )
            for vertex_idx in vertex_indices
        ]

        # Replace the character's skinned locators with our vertex-based ones
        c = c.with_skinned_locators(skinned_locators, replace=True)

        # Test with different skeleton states
        n_batch = 1
        n_model_params = c.parameter_transform.size
        model_params = torch.rand(
            n_batch, n_model_params, dtype=torch.float64, requires_grad=True
        )

        skel_state = pym_diff_geometry.model_parameters_to_skeleton_state(
            c, model_params
        )

        # Skin the mesh vertices using skin_points (ground truth)
        rest_vertices = torch.from_numpy(c.mesh.vertices).to(torch.float64)
        skinned_mesh = pym_diff_geometry.skin_points(c, skel_state, rest_vertices)

        # Extract the positions of our selected vertices from the skinned mesh
        expected_positions = skinned_mesh[
            :, vertex_indices, :
        ]  # [n_batch, n_locators, 3]

        # Skin the locators using skin_skinned_locators
        locator_positions = pym_diff_geometry.skin_skinned_locators(c, skel_state)

        # Compare results - they should be identical
        self.assertEqual(locator_positions.shape, expected_positions.shape)
        self.assertTrue(
            torch.allclose(locator_positions, expected_positions, atol=1e-5),
            f"Max difference: {torch.max(torch.abs(locator_positions - expected_positions)).item()}",
        )

        # Test with custom rest positions (override vertex positions)
        custom_rest_positions = (
            torch.randn(n_batch, n_locators, 3, dtype=torch.float64) * 0.5
        )

        # Skin using custom rest positions
        locator_positions_custom = pym_diff_geometry.skin_skinned_locators(
            c, skel_state, custom_rest_positions
        )

        # Skin the same custom positions using skin_points for verification
        # We need to create a full mesh with our custom positions at the selected indices
        full_custom_mesh = rest_vertices.unsqueeze(0).expand(n_batch, -1, -1).clone()
        full_custom_mesh[:, vertex_indices, :] = custom_rest_positions

        skinned_custom_mesh = pym_diff_geometry.skin_points(
            c, skel_state, full_custom_mesh
        )
        expected_custom_positions = skinned_custom_mesh[:, vertex_indices, :]

        # Compare results
        self.assertEqual(
            locator_positions_custom.shape, expected_custom_positions.shape
        )
        self.assertTrue(
            torch.allclose(
                locator_positions_custom, expected_custom_positions, atol=1e-5
            ),
            f"Max difference with custom positions: {torch.max(torch.abs(locator_positions_custom - expected_custom_positions)).item()}",
        )

        # Test gradients
        torch.autograd.gradcheck(
            lambda skel: pym_diff_geometry.skin_skinned_locators(c, skel),
            (skel_state,),
            eps=1e-4,
            atol=1e-3,
            raise_exception=True,
        )

        # Test single batch case
        skel_state_single = skel_state[0]  # Remove batch dimension
        locator_positions_single = pym_diff_geometry.skin_skinned_locators(
            c, skel_state_single
        )
        expected_positions_single = expected_positions[0]  # Remove batch dimension

        self.assertEqual(
            locator_positions_single.shape, expected_positions_single.shape
        )
        self.assertTrue(
            torch.allclose(
                locator_positions_single, expected_positions_single, atol=1e-5
            )
        )

    def test_closest_point_on_mesh(self) -> None:
        # x ->
        # 1___2
        # |  /|
        # | / | /|\
        # |/__|  y
        # 0   3
        vertices = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float32
        )
        triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

        query_points = torch.tensor([[1, 0, 0.5], [0, 1, 0.5]], dtype=torch.float32)
        valid, closest_points, triangle_indices, _ = (
            pym_diff_geometry.find_closest_points_on_mesh(
                points_source=query_points,
                vertices_target=vertices,
                faces_target=triangles,
            )
        )

        self.assertEqual(valid.tolist(), [True, True])
        self.assertEqual(triangle_indices.tolist(), [0, 1])
        self.assertTrue(torch.allclose(closest_points[:, 0], query_points[:, 0]))
        self.assertTrue(torch.allclose(closest_points[:, 1], query_points[:, 1]))
        self.assertTrue(
            torch.allclose(closest_points[:, 2], torch.zeros_like(query_points[:, 2]))
        )

    def test_skinning_check_derivatives(self) -> None:
        """Check the skinning derivatives using diff_geometry.skin_points."""

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
        joint_params = pym_diff_geometry.apply_parameter_transform(c, model_params)
        skel_state = pym_diff_geometry.joint_parameters_to_skeleton_state(
            c, joint_params
        )
        transforms = pym_skel_state.to_matrix(skel_state)

        # Derivatives with default rest points:
        torch.autograd.gradcheck(
            lambda xf: pym_diff_geometry.skin_points(c, xf),
            [transforms],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

        # Derivatives with specified rest points:
        rest_points = torch.from_numpy(c.mesh.vertices).double()
        rest_points.requires_grad = True
        torch.autograd.gradcheck(
            lambda xf, rp: pym_diff_geometry.skin_points(c, xf, rp),
            [transforms, rest_points],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

    def test_diff_apply_blend_coeffs_base(self) -> None:
        """Test BlendShapeBase.compute_shape with gradcheck."""
        np.random.seed(0)

        n_pts = 10
        n_blend = 4

        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(shape_vectors)

        nBatch = 2
        n_coeffs = min(blend_shape.n_shapes, 10)
        coeffs = torch.rand(nBatch, n_coeffs, dtype=torch.float64, requires_grad=True)

        shape1 = blend_shape.compute_shape(coeffs).select(0, 0)
        c1 = coeffs.select(0, 0).detach().numpy()

        # Compute the shape another way:
        shape2 = np.dot(
            shape_vectors.reshape(n_blend, n_pts * 3).transpose(), c1
        ).reshape(n_pts, 3)

        self.assertTrue(shape1.allclose(torch.from_numpy(shape2).float()))
        self.assertTrue(len(blend_shape.shape_names) == n_blend)

        torch.autograd.gradcheck(
            blend_shape.compute_shape,
            [coeffs],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

    def test_diff_apply_blend_coeffs(self) -> None:
        """Test BlendShape.compute_shape with gradcheck."""
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
        shape2 = (
            np.dot(shape_vectors.reshape(n_blend, n_pts * 3).transpose(), c1).reshape(
                n_pts, 3
            )
            + base_shape
        )

        self.assertTrue(shape1.allclose(torch.from_numpy(shape2).float()))

        torch.autograd.gradcheck(
            blend_shape.compute_shape,
            [coeffs],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

    def test_compute_blend_shape(self) -> None:
        """Test diff_geometry.compute_blend_shape wrapper function."""
        np.random.seed(0)

        n_pts = 10
        n_blend = 4

        base_shape = np.random.rand(n_pts, 3)
        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)

        nBatch = 2
        n_coeffs = blend_shape.n_shapes
        coeffs = torch.rand(nBatch, n_coeffs, dtype=torch.float64, requires_grad=True)

        # Test the module-level function
        result1 = pym_diff_geometry.compute_blend_shape(blend_shape, coeffs)
        result2 = blend_shape.compute_shape(coeffs)

        self.assertTrue(torch.allclose(result1, result2))

        # Test gradients
        torch.autograd.gradcheck(
            lambda c: pym_diff_geometry.compute_blend_shape(blend_shape, c),
            [coeffs],
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

    def test_find_closest_points(self) -> None:
        """Test find_closest_points (without normals)."""
        torch.manual_seed(0)  # ensure repeatability

        n_src_pts = 3
        n_tgt_pts = 20
        n_batch = 2
        dim = 3
        src_pts = torch.rand(n_batch, n_src_pts, dim)
        tgt_pts = torch.rand(n_batch, n_tgt_pts, dim)

        closest_pts, closest_idx, closest_valid = pym_diff_geometry.find_closest_points(
            src_pts, tgt_pts
        )
        closest_dist = torch.norm(closest_pts - src_pts, dim=-1, keepdim=False)
        self.assertTrue(torch.all(closest_valid))

        gt_closest_pts, _, gt_closest_idx, _ = _brute_force_closest_points(
            src_pts, tgt_pts
        )
        self.assertTrue(torch.allclose(gt_closest_pts, closest_pts))
        self.assertTrue(torch.allclose(gt_closest_idx, closest_idx))

        # Verify that if we pass in a small enough max_dist we don't return any points:
        min_all_dist, _ = torch.min(torch.flatten(closest_dist), dim=0)
        closest_pts_2, closest_idx_2, closest_valid_2 = (
            pym_diff_geometry.find_closest_points(
                src_pts, tgt_pts, 0.5 * min_all_dist.item()
            )
        )
        self.assertFalse(torch.any(closest_valid_2))
        self.assertTrue(
            torch.allclose(
                closest_idx_2, -1 * torch.ones(n_batch, n_src_pts, dtype=torch.int)
            )
        )

    def test_find_closest_points_with_normal(self) -> None:
        """Test find_closest_points (with normals)."""
        torch.manual_seed(0)  # ensure repeatability

        n_src_pts = 6
        n_tgt_pts = 22
        n_batch = 3
        dim = 3
        src_pts = torch.rand(n_batch, n_src_pts, dim)
        tgt_pts = torch.rand(n_batch, n_tgt_pts, dim)

        def normalized(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(t, dim=-1)

        src_normals = normalized(torch.abs(torch.rand(n_batch, n_src_pts, dim)))
        tgt_normals = normalized(torch.abs(torch.rand(n_batch, n_tgt_pts, dim)))

        (
            closest_pts,
            closest_normals,
            closest_idx,
            closest_valid,
        ) = pym_diff_geometry.find_closest_points(
            src_pts, src_normals, tgt_pts, tgt_normals
        )
        self.assertTrue(torch.all(closest_valid))

        # Normals are all in the positive quadrant so no points should be rejected.
        (
            gt_closest_pts,
            gt_closest_normals,
            gt_closest_idx,
            _,
        ) = _brute_force_closest_points(src_pts, tgt_pts, tgt_normals)

        self.assertTrue(torch.allclose(gt_closest_pts, closest_pts))
        self.assertTrue(torch.allclose(gt_closest_normals, closest_normals))
        self.assertTrue(torch.allclose(gt_closest_idx, closest_idx))

        # Now try with the opposite normals, all points should be rejected:
        _, _, closest_idx_2, closest_valid_2 = pym_diff_geometry.find_closest_points(
            src_pts, -src_normals, tgt_pts, tgt_normals
        )
        self.assertFalse(torch.any(closest_valid_2))
        self.assertFalse(torch.any(closest_idx_2 >= 0))

    def test_model_parameters_to_blend_shape_coefficients(self) -> None:
        """Test extraction of blend shape coefficients from model parameters."""
        np.random.seed(0)
        torch.manual_seed(0)

        c = pym_geometry.create_test_character()

        # Build a blend shape basis
        n_pts = c.mesh.n_vertices if c.mesh else 10
        n_blend = 4
        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        base_shape = np.random.rand(n_pts, 3)
        blend_shape = pym_geometry.BlendShape.from_tensors(base_shape, shape_vectors)

        c2 = c.with_blend_shape(blend_shape)
        params = torch.rand(c2.parameter_transform.size)

        # Extract coefficients using diff_geometry
        bp1 = params[c2.parameter_transform.blend_shape_parameters]
        bp2 = pym_diff_geometry.model_parameters_to_blend_shape_coefficients(c2, params)

        # Should extract the same coefficients
        self.assertTrue(bp1.allclose(bp2))
        self.assertEqual(bp2.shape[0], n_blend)

    def test_model_parameters_to_face_expression_coefficients(self) -> None:
        """Test extraction of face expression coefficients from model parameters."""
        np.random.seed(0)
        torch.manual_seed(0)

        c = pym_geometry.create_test_character()

        # Build a face expression blend shape basis
        n_pts = c.mesh.n_vertices if c.mesh else 10
        n_blend = 4
        shape_vectors = np.random.rand(n_blend, n_pts, 3)
        blend_shape = pym_geometry.BlendShapeBase.from_tensors(shape_vectors)

        c2 = c.with_face_expression_blend_shape(blend_shape)
        params = torch.rand(c2.parameter_transform.size)

        # Extract coefficients using diff_geometry
        bp1 = params[c2.parameter_transform.face_expression_parameters]
        bp2 = pym_diff_geometry.model_parameters_to_face_expression_coefficients(
            c2, params
        )

        # Should extract the same coefficients
        self.assertTrue(bp1.allclose(bp2))
        self.assertEqual(bp2.shape[0], n_blend)


if __name__ == "__main__":
    unittest.main()
