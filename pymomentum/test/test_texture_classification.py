# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.geometry as pym  # @manual=:geometry


def _make_quad_mesh() -> pym.Mesh:
    """Create a unit quad (two triangles) with UVs spanning [0,1] x [0,1]."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    texcoords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return pym.Mesh(
        vertices=vertices,
        faces=faces,
        texcoords=texcoords,
        texcoord_faces=faces.copy(),
    )


class TestTextureClassification(unittest.TestCase):
    """Tests for classify_triangles_by_texture function."""

    def test_uniform_texture_classifies_all_triangles(self) -> None:
        """All triangles on a uniform-color texture should match that region."""
        mesh = _make_quad_mesh()
        texture = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        region_colors = np.array([[255, 0, 0]], dtype=np.uint8)

        result = pym.classify_triangles_by_texture(
            mesh=mesh, texture=texture, region_colors=region_colors
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0, 1])

    def test_two_region_spatial_separation(self) -> None:
        """Triangles in distinct texture regions should be classified separately.

        Two non-overlapping triangles map to opposite ends of the texture:
        triangle 0 to the left quarter (red) and triangle 1 to the right
        quarter (blue), with a clear gap in between.
        """
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [2, 0, 0], [3, 0, 0], [2.5, 1, 0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        # Triangle 0 UVs in [0, 0.25] (left quarter); triangle 1 in [0.75, 1.0]
        texcoords = np.array(
            [
                [0.0, 0.25],
                [0.25, 0.25],
                [0.125, 0.75],
                [0.75, 0.25],
                [1.0, 0.25],
                [0.875, 0.75],
            ],
            dtype=np.float32,
        )
        mesh = pym.Mesh(
            vertices=vertices,
            faces=faces,
            texcoords=texcoords,
            texcoord_faces=faces.copy(),
        )

        # Left half red, right half blue
        texture = np.zeros((10, 10, 3), dtype=np.uint8)
        texture[:, :5, 0] = 255
        texture[:, 5:, 2] = 255
        region_colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        result = pym.classify_triangles_by_texture(
            mesh=mesh, texture=texture, region_colors=region_colors
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0])
        self.assertEqual(result[1], [1])

    def test_threshold_excludes_partial_matches(self) -> None:
        """With threshold=1.0, triangles straddling two regions should be excluded.

        Both triangles in the quad span the full UV range and the texture is
        split 50/50 red/blue, so no triangle can have all samples match a
        single region.
        """
        mesh = _make_quad_mesh()

        # Half red, half blue — each triangle straddles the boundary
        texture = np.zeros((10, 10, 3), dtype=np.uint8)
        texture[:, :5, 0] = 255  # left half red
        texture[:, 5:, 2] = 255  # right half blue
        region_colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        # threshold=0 (default): any single matching sample is enough
        result_low = pym.classify_triangles_by_texture(
            mesh=mesh,
            texture=texture,
            region_colors=region_colors,
            threshold=0.0,
            num_samples=10,
        )
        # Both triangles span full UV, so both should match both regions
        self.assertEqual(result_low[0], [0, 1])
        self.assertEqual(result_low[1], [0, 1])

        # threshold=1.0: ALL samples must match a single region
        result_high = pym.classify_triangles_by_texture(
            mesh=mesh,
            texture=texture,
            region_colors=region_colors,
            threshold=1.0,
            num_samples=10,
        )
        # No triangle can have all samples in one color, so both regions empty
        self.assertEqual(result_high[0], [])
        self.assertEqual(result_high[1], [])


class TestReduceMeshByFaces(unittest.TestCase):
    """Tests for Mesh-level reduce_mesh_by_faces and reduce_mesh_by_vertices."""

    def test_reduce_mesh_by_faces_keeps_selected(self) -> None:
        """Selecting one face should keep only the vertices used by that face."""
        mesh = _make_quad_mesh()
        active_faces = np.array([True, False], dtype=np.bool_)

        result = pym.reduce_mesh_by_faces(mesh=mesh, active_faces=active_faces)

        # Face 0 uses vertices 0,1,2 — so 3 vertices, 1 face
        self.assertEqual(len(result.faces), 1)
        self.assertEqual(len(result.vertices), 3)
        # Texcoord faces should also be reduced
        self.assertEqual(len(result.texcoord_faces), 1)
        self.assertEqual(len(result.texcoords), 3)
        # All face indices should be valid
        for idx in result.faces[0]:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(result.vertices))

    def test_reduce_mesh_by_vertices_keeps_selected(self) -> None:
        """Selecting 3 of 4 vertices should keep only faces using those vertices."""
        mesh = _make_quad_mesh()
        # Keep vertices 0,1,2 — only face 0 (0,1,2) uses only these vertices
        active_vertices = np.array([True, True, True, False], dtype=np.bool_)

        result = pym.reduce_mesh_by_vertices(mesh=mesh, active_vertices=active_vertices)

        self.assertEqual(len(result.faces), 1)
        self.assertEqual(len(result.vertices), 3)

    def test_reduce_mesh_preserves_all_when_all_active(self) -> None:
        """Marking all faces active should return a mesh identical in size."""
        mesh = _make_quad_mesh()
        active_faces = np.array([True, True], dtype=np.bool_)

        result = pym.reduce_mesh_by_faces(mesh=mesh, active_faces=active_faces)

        self.assertEqual(len(result.faces), len(mesh.faces))
        self.assertEqual(len(result.vertices), len(mesh.vertices))
        self.assertEqual(len(result.texcoords), len(mesh.texcoords))


class TestSplitMeshByTextureRegion(unittest.TestCase):
    """Tests for split_mesh_by_texture_region function."""

    def _check_mesh_valid(
        self,
        mesh: pym.Mesh,
        expected_normal_direction: np.ndarray | None = None,
    ) -> None:
        """Assert structural validity of a non-empty mesh.

        Checks face/texcoord index validity, no degenerate triangles,
        consistent triangle winding, and edge-manifold topology.

        Args:
            mesh: The mesh to validate.
            expected_normal_direction: If provided, verify all triangle normals
                point in the same direction (dot product > 0). This is critical
                for SDF rasterization where winding determines inside/outside.
        """
        n_verts = len(mesh.vertices)
        n_texcoords = len(mesh.texcoords)
        verts = mesh.vertices

        # Face indices must be valid
        for fi, face in enumerate(mesh.faces):
            for idx in face:
                self.assertGreaterEqual(idx, 0, f"face {fi} has negative index {idx}")
                self.assertLess(
                    idx, n_verts, f"face {fi} index {idx} >= vertex count {n_verts}"
                )

        # texcoord_faces count must match faces
        self.assertEqual(
            len(mesh.texcoord_faces),
            len(mesh.faces),
            "texcoord_faces count != faces count",
        )

        # Texcoord face indices must be valid
        for fi, tf in enumerate(mesh.texcoord_faces):
            for idx in tf:
                self.assertGreaterEqual(
                    idx, 0, f"texcoord_face {fi} has negative index {idx}"
                )
                self.assertLess(
                    idx,
                    n_texcoords,
                    f"texcoord_face {fi} index {idx} >= texcoord count {n_texcoords}",
                )

        # No degenerate (zero-area) triangles + collect normals for winding check
        normals = []
        for fi, face in enumerate(mesh.faces):
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * float(np.linalg.norm(cross))
            self.assertGreater(area, 1e-10, f"face {fi} has zero area")
            normals.append(cross / np.linalg.norm(cross))

        # Consistent winding: all triangle normals should point the same way.
        # For split meshes derived from a planar source this is exact; for
        # general meshes we check against the expected direction.
        if expected_normal_direction is not None:
            ref = expected_normal_direction / np.linalg.norm(expected_normal_direction)
            for fi, n in enumerate(normals):
                dot = float(np.dot(n, ref))
                self.assertGreater(
                    dot,
                    0.0,
                    f"face {fi} normal {n} opposes expected direction {ref} "
                    f"(dot={dot:.4f}) — winding is flipped",
                )

        # Edge-manifold check: every interior edge should be shared by exactly
        # 2 faces; boundary edges by exactly 1. No edge should appear 3+ times,
        # which would indicate a non-manifold mesh.
        edge_counts: dict[tuple[int, int], int] = {}
        for face in mesh.faces:
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                edge = (min(a, b), max(a, b))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        for edge, count in edge_counts.items():
            self.assertLessEqual(
                count,
                2,
                f"edge {edge} shared by {count} faces — mesh is non-manifold",
            )

    @staticmethod
    def _mesh_area(mesh: pym.Mesh) -> float:
        """Compute total surface area of a mesh."""
        verts = mesh.vertices
        total = 0.0
        for face in mesh.faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            total += 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
        return total

    def test_uniform_inside_preserves_mesh(self) -> None:
        """All-inside uniform texture should preserve face/vertex counts and area."""
        mesh = _make_quad_mesh()
        texture = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        region_colors = np.array([[255, 0, 0]], dtype=np.uint8)

        result = pym.split_mesh_by_texture_region(
            mesh=mesh, texture=texture, region_colors=region_colors
        )

        self.assertEqual(len(result.faces), 2)
        self.assertEqual(len(result.vertices), 4)
        self._check_mesh_valid(result, expected_normal_direction=np.array([0, 0, 1]))
        self.assertAlmostEqual(self._mesh_area(result), 1.0, places=5)

    def test_uniform_outside_produces_empty_mesh(self) -> None:
        """No matching colors should produce an empty mesh."""
        mesh = _make_quad_mesh()
        texture = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        region_colors = np.array([[0, 255, 0]], dtype=np.uint8)

        result = pym.split_mesh_by_texture_region(
            mesh=mesh, texture=texture, region_colors=region_colors
        )

        self.assertEqual(len(result.faces), 0)
        self.assertEqual(len(result.vertices), 0)

    def test_half_texture_splits_boundary(self) -> None:
        """Left-half red / right-half blue: verify splitting geometry and area.

        The quad mesh has vertices at (0,0), (1,0), (1,1), (0,1) with UVs
        matching positions. The texture boundary at u=0.5 maps to x=0.5.
        The red (inside) half should have area ≈ 0.5 and new boundary vertices
        should lie near x=0.5 within binary search tolerance.
        """
        mesh = _make_quad_mesh()
        texture = np.zeros((100, 100, 3), dtype=np.uint8)
        texture[:, :50, 0] = 255  # left half red
        texture[:, 50:, 2] = 255  # right half blue
        region_colors = np.array([[255, 0, 0]], dtype=np.uint8)

        result = pym.split_mesh_by_texture_region(
            mesh=mesh, texture=texture, region_colors=region_colors
        )

        self._check_mesh_valid(result, expected_normal_direction=np.array([0, 0, 1]))

        # Boundary triangles should be split, producing more faces
        self.assertGreater(len(result.faces), 2)

        # Area of the inside (left) half should be ≈ 0.5
        # Binary search with 8 steps gives 1/256 ≈ 0.004 precision per edge,
        # so area tolerance of ~0.02 is conservative
        area = self._mesh_area(result)
        self.assertAlmostEqual(area, 0.5, delta=0.02)

        # New split vertices should lie near x=0.5 boundary
        # Original vertices are at x=0 and x=1; any vertex near 0.5 must be new
        x_coords = result.vertices[:, 0]
        boundary_verts = x_coords[(x_coords > 0.1) & (x_coords < 0.9)]
        self.assertGreater(
            len(boundary_verts), 0, "Expected new vertices near boundary"
        )
        # Those boundary vertices should be close to x=0.5 (within binary search
        # tolerance on a 100-pixel texture: 1/256 * 1.0 ≈ 0.004)
        for x in boundary_verts:
            self.assertAlmostEqual(float(x), 0.5, delta=0.02)

    def test_multiple_region_colors_covers_expected_area(self) -> None:
        """Union of two colors covering the top-left and top-right quadrants.

        With a 4-quadrant texture (red TL, green TR, blue BL, white BR) and
        region_colors = [red, green], the inside region is the entire top half.
        The resulting mesh area should be ≈ 0.5.
        """
        mesh = _make_quad_mesh()
        texture = np.zeros((100, 100, 3), dtype=np.uint8)
        texture[:50, :50] = [255, 0, 0]  # red (top-left)
        texture[:50, 50:] = [0, 255, 0]  # green (top-right)
        texture[50:, :50] = [0, 0, 255]  # blue (bottom-left)
        texture[50:, 50:] = [255, 255, 255]  # white (bottom-right)

        region_colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        result = pym.split_mesh_by_texture_region(
            mesh=mesh, texture=texture, region_colors=region_colors
        )

        self._check_mesh_valid(result, expected_normal_direction=np.array([0, 0, 1]))
        area = self._mesh_area(result)
        self.assertAlmostEqual(area, 0.5, delta=0.02)


if __name__ == "__main__":
    unittest.main()
