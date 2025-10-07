# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np

import pymomentum.renderer as pym_renderer
import torch
from numpy.typing import NDArray


class TestRendering(unittest.TestCase):
    def test_lighting(self) -> None:
        """Test that directional lighting works."""

        image_width, image_height = 30, 30
        camera = pym_renderer.Camera(
            pym_renderer.PinholeIntrinsicsModel(
                image_width=image_width,
                image_height=image_height,
            )
        )

        def rasterize_spheres_with_lights(
            lights: list[pym_renderer.Light] | None,
            camera_param: pym_renderer.Camera = camera,
        ) -> torch.Tensor:
            z_buffer = pym_renderer.create_z_buffer(camera_param)
            rgb_buffer = pym_renderer.create_rgb_buffer(camera_param)

            pym_renderer.rasterize_spheres(
                center=torch.as_tensor([[0, 0, 20]], dtype=torch.float32),
                lights=lights,
                radius=torch.as_tensor([5], dtype=torch.float32),
                camera=camera_param,
                rgb_buffer=rgb_buffer,
                z_buffer=z_buffer,
            )
            return rgb_buffer

        def mean_absolute_difference(t1: torch.Tensor, t2: torch.Tensor) -> float:
            return torch.mean(torch.abs(t1 - t2)).item()

        default_lighting = rasterize_spheres_with_lights(None)
        lights_at_camera = rasterize_spheres_with_lights(
            [pym_renderer.Light.create_point_light(camera.center_of_projection)]
        )
        # pyre-fixme[6]: For 2nd argument expected `SupportsDunderLT[_T]` but got
        #  `float`.
        self.assertGreater(torch.mean(default_lighting), 0.15)
        front_directional_lighting = rasterize_spheres_with_lights(
            [
                pym_renderer.Light.create_directional_light(
                    camera.world_space_principle_axis
                )
            ]
        )

        # One light at the camera should be identical to the default lighting.
        self.assertLess(
            mean_absolute_difference(default_lighting, lights_at_camera), 1e-4
        )
        # Directional lighting is a bit different from a point light at the camera but
        # should be pretty close.
        self.assertLess(
            mean_absolute_difference(default_lighting, front_directional_lighting), 0.05
        )
        self.assertGreater(
            torch.mean(front_directional_lighting),
            # pyre-fixme[6]: For 2nd argument expected `SupportsDunderLT[_T]` but
            #  got `Tensor`.
            torch.mean(default_lighting),
        )

        back_lighting = rasterize_spheres_with_lights(
            [
                pym_renderer.Light.create_directional_light(
                    -camera.world_space_principle_axis
                )
            ]
        )

        # We should see very little light coming from the back:
        self.assertLess(
            mean_absolute_difference(back_lighting, torch.zeros_like(default_lighting)),
            1e-4,
        )

        # All lighting should be from the left side:
        left_lighting = rasterize_spheres_with_lights(
            [
                pym_renderer.Light.create_directional_light(
                    np.asarray([1, 0, 0], dtype=np.float32)
                )
            ]
        )
        # pyre-fixme[6]: For 2nd argument expected `SupportsDunderLT[_T]` but got
        #  `float`.
        self.assertGreater(torch.mean(left_lighting[:, : image_width // 2]), 0.1)
        # pyre-fixme[6]: For 2nd argument expected `SupportsDunderGT[_T]` but got
        #  `float`.
        self.assertLess(torch.mean(left_lighting[:, image_width // 2 :]), 1e-5)

        # All lighting should be from the right side:
        right_lighting = rasterize_spheres_with_lights(
            [
                pym_renderer.Light.create_directional_light(
                    np.asarray([-1, 0, 0], dtype=np.float32)
                )
            ]
        )
        # pyre-fixme[6]: For 2nd argument expected `SupportsDunderGT[_T]` but got
        #  `float`.
        self.assertLess(torch.mean(right_lighting[:, : image_width // 2]), 1e-5)
        # pyre-fixme[6]: For 2nd argument expected `SupportsDunderLT[_T]` but got
        #  `float`.
        self.assertGreater(torch.mean(right_lighting[:, image_width // 2 :]), 0.1)

    def test_subdivide_uneven(self) -> None:
        """Check that we can do uneven subdivision where only some edges are split."""

        def _compute_area(
            mesh_positions: NDArray[np.float32], triangles: NDArray[np.int32]
        ) -> float:
            v1 = mesh_positions[triangles[:, 1]] - mesh_positions[triangles[:, 0]]
            v2 = mesh_positions[triangles[:, 2]] - mesh_positions[triangles[:, 0]]
            cross_prod = np.cross(v1, v2)
            return 0.5 * np.sum(np.linalg.norm(cross_prod, axis=-1))

        def _compute_triangle_normals(
            mesh_positions: NDArray[np.float32], triangles: NDArray[np.int32]
        ) -> NDArray[np.float32]:
            v1 = mesh_positions[triangles[:, 1]] - mesh_positions[triangles[:, 0]]
            v2 = mesh_positions[triangles[:, 2]] - mesh_positions[triangles[:, 0]]
            cross_prod = np.cross(v1, v2)
            return cross_prod / np.linalg.norm(cross_prod, axis=-1, keepdims=True)

        mesh_positions_orig = np.asarray([[0, 0, 0], [0.25, 0, 0], [0, 1, 0]])

        mesh_normals_orig = np.asarray([[0, 0, 1]] * 3)
        mesh_triangles_orig = np.asarray([[0, 1, 2]], dtype=np.int32)

        self.assertAlmostEqual(
            _compute_area(mesh_positions_orig, mesh_triangles_orig), 0.125
        )

        # Check basic invariants:
        #   total area is unchanged
        #   all triangles have the correct winding
        def _check_invariants(
            max_edge_length: float,
            expected_triangles: int,
            positions: NDArray[np.float32] = mesh_positions_orig,
            normals: NDArray[np.float32] = mesh_normals_orig,
            triangles: NDArray[np.int32] = mesh_triangles_orig,
        ) -> None:
            (
                mesh_positions_subd,
                mesh_normals_subd,
                mesh_triangles_subd,
                texture_coords_subd,
                texture_triangles_subd,
            ) = pym_renderer.subdivide_mesh(
                positions,
                normals,
                triangles,
                levels=1,
                max_edge_length=max_edge_length,
            )

            self.assertEqual(mesh_triangles_subd.shape[0], expected_triangles)

            # Since we didn't provide texture coords, should get just all-zeros texture coords:
            self.assertTrue(np.array_equal(mesh_triangles_subd, texture_triangles_subd))
            self.assertFalse(np.any(texture_coords_subd))

            area_new = _compute_area(mesh_positions_subd, mesh_triangles_subd)
            self.assertAlmostEqual(area_new, 0.125)

            normals_new = _compute_triangle_normals(
                mesh_positions_subd, mesh_triangles_subd
            )
            self.assertEqual(normals_new.shape[0], expected_triangles)
            for i in range(normals_new.shape[0]):
                self.assertLess(np.linalg.norm(normals_new[i] - normals[0]), 1e-3)

        _check_invariants(10, 1)
        _check_invariants(1.01, 2)
        _check_invariants(0.99, 3)
        _check_invariants(0.001, 4)

    def test_shadow_projection_matrix_point_light(self) -> None:
        """Check that the projection matrix actually projects points."""

        points = np.asarray(
            [[0, 0, 0], [0, 10, 0], [0, 0, 10], [0, 10, 10], [10, 20, 20]],
            dtype=np.float32,
        )

        pos_light = pym_renderer.Light.create_point_light(
            np.asarray([0, 50, 0], dtype=np.float32)
        )
        pos_proj_mat = pym_renderer.create_shadow_projection_matrix(pos_light)
        plane_normal = np.asarray([0, 1, 0], dtype=np.float32)

        for i in range(points.shape[0]):
            p = np.concatenate([points[i, :], np.ones([1], dtype=np.float32)])
            p_proj_unnormalized = np.matmul(pos_proj_mat, p)
            p_proj_normalized = p_proj_unnormalized[0:3] / p_proj_unnormalized[3]

            # vector pointing from p toward light:
            light_vec = pos_light.position - p[0:3]
            light_vec = light_vec / np.linalg.norm(light_vec)

            # intersect with plane:
            # (p + t * light_vec) . n = 0
            # p . n = -(light_vec . n) * t
            # Solve for t:
            t = -np.dot(p[0:3], plane_normal) / np.dot(light_vec, plane_normal)
            p_proj = p[0:3] + t * light_vec
            self.assertTrue(np.allclose(p_proj, p_proj_normalized))

    def test_shadow_projection_matrix_dir_light(self) -> None:
        points = np.asarray(
            [[0, 0, 0], [0, 10, 0], [0, 0, 10], [0, 10, 10], [10, 20, 20]],
            dtype=np.float32,
        )

        dir_light = pym_renderer.Light.create_directional_light(
            np.asarray([-1, -1, -2], dtype=np.float32)
        )
        plane_normal = np.asarray([0, 1, 0], dtype=np.float32)
        plane_origin = np.asarray([0, -1, 0], dtype=np.float32)
        plane_offset = np.dot(plane_normal, plane_origin)
        dir_proj_mat = pym_renderer.create_shadow_projection_matrix(
            dir_light, plane_normal=plane_normal, plane_origin=plane_origin
        )

        for i in range(points.shape[0]):
            p = np.concatenate([points[i, :], np.ones([1], dtype=np.float32)])
            p_proj_unnormalized = np.matmul(dir_proj_mat, p)
            p_proj_normalized = p_proj_unnormalized[0:3] / p_proj_unnormalized[3]

            # vector pointing from p toward light:
            light_vec = dir_light.position
            light_vec = light_vec / np.linalg.norm(light_vec)

            # intersect with plane:
            # (p + t * light_vec) . n = plane_offset
            # t * (light_vec . n) = plane_offset - (p . n)
            # Solve for t:
            t = (plane_offset - np.dot(p[0:3], plane_normal)) / np.dot(
                light_vec, plane_normal
            )
            p_proj = p[0:3] + t * light_vec
            self.assertTrue(np.allclose(p_proj, p_proj_normalized))
            self.assertAlmostEqual(p_proj[1], -1)
