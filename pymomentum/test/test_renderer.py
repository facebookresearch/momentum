# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest

import numpy as np
import torch  # isort: skip  # Must be imported before pymomentum C++ extensions
import pymomentum.camera as pym_camera  # @manual=:camera
import pymomentum.renderer as pym_renderer  # @manual=:renderer
from numpy.typing import NDArray


class TestRendering(unittest.TestCase):
    def test_lighting(self) -> None:
        """Test that directional lighting works."""

        image_width, image_height = 30, 30
        camera = pym_camera.Camera(
            pym_camera.PinholeIntrinsicsModel(
                image_width=image_width,
                image_height=image_height,
            )
        )

        def rasterize_spheres_with_lights(
            lights: list[pym_renderer.Light] | None,
            camera_param: pym_camera.Camera = camera,
        ) -> NDArray[np.float32]:
            z_buffer = pym_renderer.create_z_buffer(camera_param)
            rgb_buffer = pym_renderer.create_rgb_buffer(camera_param)

            pym_renderer.rasterize_spheres(
                center=np.array([[0, 0, 20]], dtype=np.float32),
                lights=lights,
                radius=np.array([5], dtype=np.float32),
                camera=camera_param,
                rgb_buffer=rgb_buffer,
                z_buffer=z_buffer,
            )
            return rgb_buffer

        def mean_absolute_difference(
            t1: NDArray[np.float32], t2: NDArray[np.float32]
        ) -> float:
            return float(np.mean(np.abs(t1 - t2)))

        default_lighting = rasterize_spheres_with_lights(None)
        lights_at_camera = rasterize_spheres_with_lights(
            [pym_renderer.Light.create_point_light(camera.center_of_projection)]
        )
        self.assertGreater(np.mean(default_lighting), 0.15)
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
            np.mean(front_directional_lighting),
            np.mean(default_lighting),
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
            mean_absolute_difference(back_lighting, np.zeros_like(default_lighting)),
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
        self.assertGreater(np.mean(left_lighting[:, : image_width // 2]), 0.1)
        self.assertLess(np.mean(left_lighting[:, image_width // 2 :]), 1e-5)

        # All lighting should be from the right side:
        right_lighting = rasterize_spheres_with_lights(
            [
                pym_renderer.Light.create_directional_light(
                    np.asarray([-1, 0, 0], dtype=np.float32)
                )
            ]
        )
        self.assertLess(np.mean(right_lighting[:, : image_width // 2]), 1e-5)
        self.assertGreater(np.mean(right_lighting[:, image_width // 2 :]), 0.1)

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

    def _get_rendered_mask(self, z_buffer: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Return a boolean mask of pixels that were actually rendered.

        The z-buffer is initialized to far_clip (FLT_MAX by default).
        Rendered pixels have depth values significantly smaller than FLT_MAX.
        """
        far_clip = np.float32(3.4028234663852886e38)
        return z_buffer < far_clip

    def _rasterize_quad(
        self,
        camera: pym_camera.Camera,
        texture_coordinates: torch.Tensor | None = None,
        texture_triangles: torch.Tensor | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Rasterize a large front-facing quad and return (z_buffer, rgb_buffer).

        The quad spans [-5, -5, 5] to [5, 5, 5] to fill the viewport.
        Back-face culling is disabled so winding order does not matter.
        """
        positions = np.array(
            [[-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]], dtype=np.float32
        )
        normals = np.array(
            [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype=np.float32
        )
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Convert optional torch texture args to numpy if provided
        tex_coords_np = (
            texture_coordinates.numpy() if texture_coordinates is not None else None
        )
        tex_tris_np = (
            texture_triangles.numpy() if texture_triangles is not None else None
        )

        z_buffer = pym_renderer.create_z_buffer(camera)
        rgb_buffer = pym_renderer.create_rgb_buffer(camera)
        pym_renderer.rasterize_mesh(
            positions,
            normals,
            triangles,
            camera,
            z_buffer,
            rgb_buffer,
            texture_coordinates=tex_coords_np,
            texture_triangles=tex_tris_np,
            back_face_culling=False,
        )
        return z_buffer, rgb_buffer

    def test_rasterize_mesh_texcoords_match_vertices(self) -> None:
        """Test rasterize_mesh when texture coordinate count matches vertex count.

        Verifies that providing texture coordinates (same count as vertices)
        does not alter the rendered geometry: the z-buffer should match a
        render without texture coordinates, depth values should be correct,
        and the rgb_buffer should contain visible (lit) pixels.
        """
        # The SIMD rasterizer uses large stack frames (drjit SIMD types +
        # deep inlining) that overflow the default 1 MB Windows thread stack.
        if sys.platform == "win32":
            return

        camera = pym_camera.Camera(
            pym_camera.PinholeIntrinsicsModel(image_width=64, image_height=64)
        )

        # 4 texture coordinates — same count as 4 quad vertices.
        tex_coords = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)

        # Render WITHOUT texture coordinates as a baseline.
        z_no_tex, rgb_no_tex = self._rasterize_quad(camera)

        # Render WITH texture coordinates.
        z_tex, rgb_tex = self._rasterize_quad(camera, texture_coordinates=tex_coords)

        # 1. Verify geometry was rendered (pixels with depth < far_clip).
        rendered_mask = self._get_rendered_mask(z_tex)
        num_rendered = rendered_mask.sum().item()
        self.assertGreater(num_rendered, 0, "No pixels were rendered")

        # 2. Verify depth values are correct: the quad is at z=5, so all
        #    rendered pixels should have depth close to 5.
        rendered_depths = z_tex[rendered_mask]
        self.assertTrue(
            np.all(rendered_depths > 4.0).item(),
            f"Min depth {rendered_depths.min().item()} is too small (expected > 4.0)",
        )
        self.assertTrue(
            np.all(rendered_depths < 6.0).item(),
            f"Max depth {rendered_depths.max().item()} is too large (expected < 6.0)",
        )

        # 3. Verify texture coordinates don't change the geometry: depth
        #    buffers should be identical with and without texture coordinates.
        np.testing.assert_allclose(z_tex, z_no_tex)

        # 4. Verify the rgb_buffer has visible (non-zero) rendered pixels,
        #    confirming that lighting was applied to the rendered geometry.
        rendered_rgb = rgb_tex[rendered_mask]
        self.assertTrue(
            np.any(rendered_rgb > 0).item(),
            "No visible pixels in rgb_buffer after rendering",
        )

    def test_rasterize_mesh_texcoords_fewer_than_vertices(self) -> None:
        """Test rasterize_mesh when there are fewer texture coordinates than vertices.

        Verifies that separate texture_triangles with fewer texture
        coordinates than vertices produces correct geometry: the z-buffer
        depth values should match the expected geometry depth, the rendered
        pixel count should match a baseline render without texture
        coordinates, and the geometry is unaffected by the texture mapping.
        """
        # The SIMD rasterizer uses large stack frames (drjit SIMD types +
        # deep inlining) that overflow the default 1 MB Windows thread stack.
        if sys.platform == "win32":
            return

        camera = pym_camera.Camera(
            pym_camera.PinholeIntrinsicsModel(image_width=64, image_height=64)
        )

        # 3 texture coordinates — fewer than 4 quad vertices. Use
        # texture_triangles to index into these coordinates independently.
        tex_coords = torch.tensor([[0, 0], [1, 0], [0.5, 1]], dtype=torch.float32)
        tex_triangles = torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.int32)

        # Render WITHOUT texture coordinates as a baseline.
        z_no_tex, rgb_no_tex = self._rasterize_quad(camera)

        # Render WITH fewer texture coordinates and separate texture_triangles.
        z_tex, rgb_tex = self._rasterize_quad(
            camera,
            texture_coordinates=tex_coords,
            texture_triangles=tex_triangles,
        )

        # 1. Verify geometry was rendered.
        rendered_mask = self._get_rendered_mask(z_tex)
        num_rendered = rendered_mask.sum().item()
        self.assertGreater(num_rendered, 0, "No pixels were rendered")

        # 2. Verify depth values are correct (quad at z=5).
        rendered_depths = z_tex[rendered_mask]
        self.assertTrue(
            np.all(rendered_depths > 4.0).item(),
            f"Min depth {rendered_depths.min().item()} is too small (expected > 4.0)",
        )
        self.assertTrue(
            np.all(rendered_depths < 6.0).item(),
            f"Max depth {rendered_depths.max().item()} is too large (expected < 6.0)",
        )

        # 3. Verify texture coordinates don't change the geometry: the depth
        #    buffers should be identical regardless of texture coordinates.
        np.testing.assert_allclose(z_tex, z_no_tex)

        # 4. Verify the same number of pixels were rendered in both cases,
        #    confirming that the separate texture_triangles don't affect
        #    which pixels are covered.
        baseline_count = self._get_rendered_mask(z_no_tex).sum().item()
        self.assertEqual(baseline_count, num_rendered)

        # 5. Verify the rgb_buffer has visible (lit) pixels.
        rendered_rgb = rgb_tex[rendered_mask]
        self.assertTrue(
            np.any(rendered_rgb > 0).item(),
            "No visible pixels in rgb_buffer after rendering",
        )

    def test_rasterize_mesh_texcoords_more_than_vertices(self) -> None:
        """Test rasterize_mesh when there are more texture coordinates than vertices.

        Simulates a UV seam scenario where the number of texture coordinates
        exceeds the number of vertices. Verifies that:
        - The extra texture coordinates are accepted without error.
        - Different texture_triangles referencing different subsets of the
          extra texture coordinates both render correctly.
        - The z-buffer depth values are correct.
        - The geometry (z-buffer) is unaffected by the choice of texture
          triangle indices.
        """
        # The SIMD rasterizer uses large stack frames (drjit SIMD types +
        # deep inlining) that overflow the default 1 MB Windows thread stack.
        if sys.platform == "win32":
            return

        camera = pym_camera.Camera(
            pym_camera.PinholeIntrinsicsModel(image_width=64, image_height=64)
        )

        # A large triangle with 3 vertices that fills the viewport.
        positions = np.array([[-5, -5, 5], [5, -5, 5], [0, 5, 5]], dtype=np.float32)
        normals = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)

        # 6 texture coordinates — more than 3 vertices. This is common in real
        # meshes where UV seams require duplicated texture coordinates.
        tex_coords = np.array(
            [[0, 0], [1, 0], [0.5, 1], [0.5, 0], [1, 0.5], [0, 0.5]],
            dtype=np.float32,
        )

        def rasterize_triangle(
            tex_tri: NDArray[np.int32],
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
            z_buf = pym_renderer.create_z_buffer(camera)
            rgb_buf = pym_renderer.create_rgb_buffer(camera)
            pym_renderer.rasterize_mesh(
                positions,
                normals,
                triangles,
                camera,
                z_buf,
                rgb_buf,
                texture_coordinates=tex_coords,
                texture_triangles=tex_tri,
                back_face_culling=False,
            )
            return z_buf, rgb_buf

        # Render with texture_triangles referencing the first three tex coords.
        tex_triangles_a = np.array([[0, 1, 2]], dtype=np.int32)
        z_buffer_a, rgb_buffer_a = rasterize_triangle(tex_triangles_a)

        # Render with texture_triangles referencing the last three tex coords
        # (indices 3, 4, 5) — exercises the extra texture coordinates.
        tex_triangles_b = np.array([[3, 4, 5]], dtype=np.int32)
        z_buffer_b, rgb_buffer_b = rasterize_triangle(tex_triangles_b)

        # 1. Verify geometry was rendered in both cases.
        rendered_mask_a = self._get_rendered_mask(z_buffer_a)
        rendered_mask_b = self._get_rendered_mask(z_buffer_b)
        self.assertGreater(
            rendered_mask_a.sum().item(), 0, "No pixels rendered with tex_triangles_a"
        )
        self.assertGreater(
            rendered_mask_b.sum().item(), 0, "No pixels rendered with tex_triangles_b"
        )

        # 2. Verify depth values are correct (triangle at z=5).
        depths_a = z_buffer_a[rendered_mask_a]
        self.assertTrue(
            np.all(depths_a > 4.0).item(),
            f"Min depth {depths_a.min().item()} is too small (expected > 4.0)",
        )
        self.assertTrue(
            np.all(depths_a < 6.0).item(),
            f"Max depth {depths_a.max().item()} is too large (expected < 6.0)",
        )

        # 3. Verify geometry is identical regardless of which texture
        #    coordinates are referenced — the z-buffers and rendered pixel
        #    masks should match exactly.
        np.testing.assert_allclose(z_buffer_a, z_buffer_b)
        self.assertTrue(np.array_equal(rendered_mask_a, rendered_mask_b))

        # 4. Verify the rgb_buffer has visible (lit) pixels.
        rendered_rgb_a = rgb_buffer_a[rendered_mask_a]
        self.assertTrue(
            np.any(rendered_rgb_a > 0).item(),
            "No visible pixels in rgb_buffer after rendering",
        )
