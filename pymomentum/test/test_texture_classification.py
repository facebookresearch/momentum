# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pymomentum.geometry as pym


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


if __name__ == "__main__":
    unittest.main()
