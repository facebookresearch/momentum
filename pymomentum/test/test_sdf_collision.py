# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for pure PyTorch SDF collision geometry (sdf_collision.py)."""

import unittest

import numpy as np
import pymomentum.axel as axel
import pymomentum.geometry as pym_geometry
import pymomentum.geometry_test_utils as pym_test_utils
import pymomentum.quaternion as pym_quaternion
import pymomentum.torch.character as character
import torch
from pymomentum.torch.sdf_collision import (
    evaluate_sdf,
    SDFCollider,
    SDFCollisionGeometry,
)


class TestSDFCollision(unittest.TestCase):
    """Tests for evaluate_sdf, SDFCollider, and SDFCollisionGeometry."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_linear_sdf(
        self, nx: int = 3, ny: int = 5, nz: int = 7
    ) -> axel.SignedDistanceField:
        """Non-cubic SDF with linear values f(i,j,k) = i + 2j + 3k.

        Trilinear interpolation is exact for linear functions, so this
        lets us verify correctness at arbitrary interior positions.
        """
        bounds = axel.BoundingBox(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([float(nx), float(ny), float(nz)], dtype=np.float32),
        )
        sdf = axel.SignedDistanceField(bounds, np.array([nx, ny, nz], dtype=np.int32))
        sdf_view = np.asarray(sdf)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    sdf_view[i, j, k] = float(i) + 2.0 * j + 3.0 * k
        return sdf

    def _create_sphere_sdf(
        self, nx: int = 5, ny: int = 7, nz: int = 3
    ) -> axel.SignedDistanceField:
        """Non-cubic sphere SDF centered at origin with radius 1."""
        radius = 1.0
        padding = 0.5
        bounds = axel.BoundingBox(
            np.array([-(radius + padding)] * 3, dtype=np.float32),
            np.array([radius + padding] * 3, dtype=np.float32),
        )
        sdf = axel.SignedDistanceField(bounds, np.array([nx, ny, nz], dtype=np.int32))
        sdf_view = np.asarray(sdf)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    world = sdf.grid_to_world(
                        np.array([float(i), float(j), float(k)], dtype=np.float32)
                    )
                    sdf_view[i, j, k] = float(np.linalg.norm(world)) - radius
        return sdf

    def _sdf_to_tensors(
        self, sdf: axel.SignedDistanceField
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a pymomentum SDF to (sdf_data, bounds_min, voxel_size).

        Returns sdf_data in (nz, ny, nx) DHW order matching evaluate_sdf's
        expected layout.
        """
        res = sdf.resolution
        nx, ny, nz = int(res[0]), int(res[1]), int(res[2])
        sdf_data = torch.tensor(
            np.asarray(sdf).ravel(order="F"), dtype=torch.float32
        ).reshape(nz, ny, nx)
        bounds_min = torch.tensor(np.asarray(sdf.bounds.min), dtype=torch.float32)
        voxel_size = torch.tensor(np.asarray(sdf.voxel_size), dtype=torch.float32)
        return sdf_data, bounds_min, voxel_size

    # ------------------------------------------------------------------
    # Trilinear interpolation correctness
    # ------------------------------------------------------------------

    def test_trilinear_at_grid_points_non_cubic(self) -> None:
        """Values at exact grid points match what was stored, on a non-cubic grid."""
        sdf = self._create_linear_sdf(3, 5, 7)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)
        nx, ny, nz = 3, 5, 7

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    world = sdf.grid_to_world(
                        np.array([float(i), float(j), float(k)], dtype=np.float32)
                    )
                    point = torch.tensor(world).unsqueeze(0)
                    result = evaluate_sdf(sdf_data, bounds_min, voxel_size, point)
                    expected = float(i) + 2.0 * j + 3.0 * k
                    self.assertAlmostEqual(
                        result.item(),
                        expected,
                        places=4,
                        msg=f"Mismatch at grid ({i},{j},{k})",
                    )

    def test_trilinear_exact_for_linear_function(self) -> None:
        """Trilinear interpolation is exact for linear functions at arbitrary positions."""
        sdf = self._create_linear_sdf(3, 5, 7)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        torch.manual_seed(42)
        grid_coords = torch.rand(50, 3) * torch.tensor([2.0, 4.0, 6.0])
        world_points = grid_coords * voxel_size + bounds_min
        results = evaluate_sdf(sdf_data, bounds_min, voxel_size, world_points)

        expected = grid_coords[:, 0] + 2.0 * grid_coords[:, 1] + 3.0 * grid_coords[:, 2]
        torch.testing.assert_close(results, expected, atol=1e-5, rtol=1e-5)

    def test_evaluate_sdf_vs_cpp_non_cubic(self) -> None:
        """evaluate_sdf matches C++ sdf.sample() on a non-cubic grid."""
        sdf = self._create_sphere_sdf(5, 7, 3)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        np.random.seed(42)
        points_np = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 4.0
        cpp_values = sdf.sample(points_np)

        torch_values = evaluate_sdf(
            sdf_data, bounds_min, voxel_size, torch.tensor(points_np)
        )
        np.testing.assert_allclose(
            torch_values.detach().numpy(), cpp_values, atol=1e-5, rtol=1e-5
        )

    def test_evaluate_sdf_out_of_bounds_vs_cpp(self) -> None:
        """Out-of-bounds queries match C++ sample() (boundary value + offset distance)."""
        sdf = self._create_sphere_sdf(5, 7, 3)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        far_points = np.array(
            [[10.0, 10.0, 10.0], [-5.0, 0.0, 0.0], [0.0, 0.0, 8.0]],
            dtype=np.float32,
        )
        cpp_values = sdf.sample(far_points)
        torch_values = evaluate_sdf(
            sdf_data, bounds_min, voxel_size, torch.tensor(far_points)
        )
        np.testing.assert_allclose(
            torch_values.detach().numpy(), cpp_values, atol=1e-5, rtol=1e-5
        )

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def test_batch_shapes(self) -> None:
        """Output shape matches input batch dimensions."""
        sdf = self._create_sphere_sdf(5, 7, 3)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        for shape in [(1, 3), (10, 3), (2, 5, 3), (2, 3, 4, 3)]:
            points = torch.randn(shape)
            result = evaluate_sdf(sdf_data, bounds_min, voxel_size, points)
            self.assertEqual(result.shape, shape[:-1])

    def test_batch_consistency(self) -> None:
        """Batched evaluation produces the same results as unbatched loop."""
        sdf = self._create_sphere_sdf(5, 7, 3)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        torch.manual_seed(0)
        points = torch.randn(2, 3, 4, 3)
        batched = evaluate_sdf(sdf_data, bounds_min, voxel_size, points)

        for b1 in range(2):
            for b2 in range(3):
                single = evaluate_sdf(sdf_data, bounds_min, voxel_size, points[b1, b2])
                torch.testing.assert_close(batched[b1, b2], single)

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def test_gradient_vs_cpp_interior(self) -> None:
        """Autograd gradient matches C++ sdf.gradient() for interior points."""
        sdf = self._create_sphere_sdf(5, 7, 3)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        # Points strictly inside the grid (one voxel margin from boundary)
        np.random.seed(42)
        b_min = np.asarray(sdf.bounds.min)
        margin = np.asarray(sdf.voxel_size) * 1.5
        inner_min = b_min + margin
        inner_max = np.asarray(sdf.bounds.max) - margin
        points_np = (
            np.random.rand(20, 3).astype(np.float32) * (inner_max - inner_min)
            + inner_min
        )

        cpp_grad = sdf.gradient(points_np)

        points = torch.tensor(points_np, requires_grad=True)
        values = evaluate_sdf(sdf_data, bounds_min, voxel_size, points)
        torch_grad = torch.autograd.grad(values.sum(), points)[0]

        np.testing.assert_allclose(
            torch_grad.detach().numpy(), cpp_grad, atol=1e-4, rtol=1e-4
        )

    def test_gradient_of_gradient(self) -> None:
        """Second-order gradients are computable (cross-derivatives of trilinear blend)."""
        sdf = self._create_sphere_sdf(5, 7, 3)
        sdf_data, bounds_min, voxel_size = self._sdf_to_tensors(sdf)

        points = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        values = evaluate_sdf(sdf_data, bounds_min, voxel_size, points)

        grad1 = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1.sum(), points)[0]

        self.assertTrue(torch.isfinite(grad2).all())
        # Cross-derivatives of trilinear blend are non-zero in general
        self.assertTrue((grad2.abs() > 0).any())

    # ------------------------------------------------------------------
    # SDFCollider
    # ------------------------------------------------------------------

    def test_from_pymomentum_preserves_data(self) -> None:
        """from_pymomentum produces correct SDF shape and metadata."""
        sdf_pym = self._create_sphere_sdf(5, 7, 3)
        collider_pym = pym_geometry.SDFCollider(
            translation=np.array([0.5, -0.3, 0.1], dtype=np.float32),
            parent=2,
            sdf=sdf_pym,
        )
        collider = SDFCollider.from_pymomentum(collider_pym)

        self.assertEqual(collider.sdf_data.shape, (3, 7, 5))  # (nz, ny, nx)
        self.assertEqual(collider.parent_joint, 2)
        np.testing.assert_allclose(
            collider.bounds_min.numpy(), np.asarray(sdf_pym.bounds.min), atol=1e-6
        )
        np.testing.assert_allclose(
            collider.voxel_size.numpy(), np.asarray(sdf_pym.voxel_size), atol=1e-6
        )
        np.testing.assert_allclose(
            collider.local_translation.numpy(), [0.5, -0.3, 0.1], atol=1e-6
        )

    def test_collider_world_space(self) -> None:
        """World-space SDFCollider (parent=-1) applies local transform correctly."""
        sdf_pym = self._create_sphere_sdf(5, 7, 3)
        collider_pym = pym_geometry.SDFCollider(
            translation=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            parent=-1,
            sdf=sdf_pym,
        )
        collider = SDFCollider.from_pymomentum(collider_pym)

        # World point [1,0,0] → SDF local point [0,0,0]
        dummy_skel = torch.zeros(1, 1, 8)
        points = torch.tensor([[1.0, 0.0, 0.0]])
        result = collider.evaluate(dummy_skel, points)
        expected = float(sdf_pym.sample(np.array([0.0, 0.0, 0.0])))
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_collider_joint_space(self) -> None:
        """Joint-space SDFCollider composes joint and local transforms."""
        sdf_pym = self._create_sphere_sdf(5, 7, 3)
        char_pym = pym_test_utils.create_test_character()
        skel = character.Skeleton(char_pym)

        collider_pym = pym_geometry.SDFCollider(parent=0, sdf=sdf_pym)
        collider = SDFCollider.from_pymomentum(collider_pym)

        # Rest pose
        n_joints = len(char_pym.skeleton.joint_names)
        joint_params = torch.zeros(1, n_joints * 7)
        skel_state = skel.forward(joint_params)

        # Manually transform a world point using the joint transform
        joint_t = skel_state[0, 0, 0:3]
        joint_q = skel_state[0, 0, 3:7]
        joint_R = pym_quaternion.to_rotation_matrix(joint_q.unsqueeze(0)).squeeze(0)

        world_point = torch.tensor([[0.5, 0.3, -0.2]])
        p_local = (world_point - joint_t) @ joint_R

        expected = evaluate_sdf(
            collider.sdf_data, collider.bounds_min, collider.voxel_size, p_local
        )
        # evaluate() broadcasts skel_state batch dim (1,) with points
        actual = collider.evaluate(skel_state, world_point)
        torch.testing.assert_close(actual.squeeze(0), expected)

    # ------------------------------------------------------------------
    # SDFCollisionGeometry
    # ------------------------------------------------------------------

    def test_evaluate_min_picks_smallest(self) -> None:
        """evaluate_min returns the minimum distance across colliders."""
        sdf_pym = self._create_sphere_sdf(5, 7, 3)
        c1 = pym_geometry.SDFCollider(
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            parent=-1,
            sdf=sdf_pym,
        )
        c2 = pym_geometry.SDFCollider(
            translation=np.array([2.0, 0.0, 0.0], dtype=np.float32),
            parent=-1,
            sdf=sdf_pym,
        )
        geo = SDFCollisionGeometry.from_pymomentum([c1, c2])

        dummy_skel = torch.zeros(1, 1, 8)
        points = torch.tensor([[0.0, 0.0, 0.0]])

        per_collider = geo.evaluate(dummy_skel, points)
        self.assertEqual(len(per_collider), 2)

        min_dist = geo.evaluate_min(dummy_skel, points)
        expected_min = min(float(d.item()) for d in per_collider)
        self.assertAlmostEqual(float(min_dist.item()), expected_min, places=5)

    # ------------------------------------------------------------------
    # Coverage gap: gradient through joint-space coordinate transform
    # ------------------------------------------------------------------

    def test_gradient_through_joint_transform(self) -> None:
        """Gradients flow through the joint-space coordinate transform in evaluate()."""
        sdf_pym = self._create_sphere_sdf(5, 7, 3)
        char_pym = pym_test_utils.create_test_character()
        skel = character.Skeleton(char_pym)

        collider_pym = pym_geometry.SDFCollider(parent=0, sdf=sdf_pym)
        collider = SDFCollider.from_pymomentum(collider_pym)

        n_joints = len(char_pym.skeleton.joint_names)
        joint_params = torch.zeros(1, n_joints * 7, requires_grad=True)
        skel_state = skel.forward(joint_params)

        points = torch.tensor([[0.3, 0.2, -0.1]])
        dist = collider.evaluate(skel_state, points)
        grad = torch.autograd.grad(dist.sum(), joint_params)[0]

        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(
            (grad.abs() > 0).any(),
            "Gradient should be non-zero — joint params affect SDF distance",
        )

    # ------------------------------------------------------------------
    # Coverage gap: non-identity local rotation
    # ------------------------------------------------------------------

    def test_from_pymomentum_with_rotation(self) -> None:
        """from_pymomentum correctly converts a non-identity rotation quaternion."""
        sdf_pym = self._create_linear_sdf(3, 5, 7)

        # 90-degree rotation around Z axis: quat (x,y,z,w) = (0,0,sin45,cos45)
        s = float(np.sin(np.pi / 4))
        c = float(np.cos(np.pi / 4))
        rot_quat = np.array([0.0, 0.0, s, c], dtype=np.float32)

        collider_pym = pym_geometry.SDFCollider(
            translation=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            rotation=rot_quat,  # pyre-ignore[6]: ndarray is valid for Optional[ndarray]
            parent=-1,
            sdf=sdf_pym,
        )
        collider = SDFCollider.from_pymomentum(collider_pym)
        R = collider.local_rotation

        # Rotation matrix should be orthogonal and non-identity
        torch.testing.assert_close(R @ R.T, torch.eye(3), atol=1e-5, rtol=0.0)
        self.assertFalse(torch.allclose(R, torch.eye(3), atol=1e-3))

        # Point at translation maps to SDF origin
        dummy_skel = torch.zeros(1, 1, 8)
        dist = collider.evaluate(dummy_skel, torch.tensor([[1.0, 2.0, 3.0]]))
        expected = float(sdf_pym.sample(np.zeros(3, dtype=np.float32)))
        self.assertAlmostEqual(float(dist.item()), expected, places=4)

    # ------------------------------------------------------------------
    # Coverage gap: batched SDFCollider.evaluate()
    # ------------------------------------------------------------------

    def test_collider_batched_skel_state(self) -> None:
        """SDFCollider.evaluate() handles batched skel_state and multiple points."""
        sdf_pym = self._create_sphere_sdf(5, 7, 3)
        char_pym = pym_test_utils.create_test_character()
        skel = character.Skeleton(char_pym)

        collider_pym = pym_geometry.SDFCollider(parent=0, sdf=sdf_pym)
        collider = SDFCollider.from_pymomentum(collider_pym)

        n_joints = len(char_pym.skeleton.joint_names)
        batch_size = 3
        n_points = 4

        torch.manual_seed(123)
        joint_params = torch.randn(batch_size, n_joints * 7) * 0.1
        skel_state = skel.forward(joint_params)

        points = torch.randn(n_points, 3) * 0.3

        batched = collider.evaluate(skel_state, points)
        self.assertEqual(batched.shape, (batch_size, n_points))

        for b in range(batch_size):
            single = collider.evaluate(skel_state[b : b + 1], points)
            torch.testing.assert_close(batched[b], single.squeeze(0))
