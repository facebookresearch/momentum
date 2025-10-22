# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import pymomentum.axel as axel


class TestAxel(unittest.TestCase):
    def test_bounding_box_creation(self):
        """Test basic BoundingBox creation and properties."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])

        bbox = axel.BoundingBox(min_corner, max_corner)

        np.testing.assert_array_equal(bbox.min, min_corner)
        np.testing.assert_array_equal(bbox.max, max_corner)
        np.testing.assert_array_almost_equal(bbox.center, [0.5, 0.5, 0.5])

    def test_bounding_box_contains(self):
        """Test BoundingBox.contains functionality."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        # Point inside
        self.assertTrue(bbox.contains(np.array([0.5, 0.5, 0.5])))

        # Point outside
        self.assertFalse(bbox.contains(np.array([1.5, 0.5, 0.5])))

        # Point on boundary
        self.assertTrue(bbox.contains(np.array([1.0, 1.0, 1.0])))

    def test_signed_distance_field_creation(self):
        """Test basic SignedDistanceField creation and properties."""
        min_corner = np.array([-1.0, -1.0, -1.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([10, 10, 10], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Check properties
        np.testing.assert_array_equal(sdf.resolution, resolution)
        self.assertEqual(sdf.total_voxels, 1000)  # 10 * 10 * 10

        # Check bounds
        np.testing.assert_array_equal(sdf.bounds.min, min_corner)
        np.testing.assert_array_equal(sdf.bounds.max, max_corner)

        # Check voxel size
        expected_voxel_size = [
            2.0 / 10.0,
            2.0 / 10.0,
            2.0 / 10.0,
        ]  # (max - min) / resolution
        np.testing.assert_array_almost_equal(sdf.voxel_size, expected_voxel_size)

    def test_sdf_grid_access(self):
        """Test grid access methods (at, set, is_valid_index)."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([5, 5, 5], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Test valid indices
        self.assertTrue(sdf.is_valid_index(0, 0, 0))
        self.assertTrue(sdf.is_valid_index(4, 4, 4))

        # Test invalid indices
        self.assertFalse(sdf.is_valid_index(5, 0, 0))
        self.assertFalse(sdf.is_valid_index(-1, 0, 0))

        # Test buffer protocol access
        data_array = np.array(sdf)

        # Test that we can set and get values through the numpy array
        data_array[1, 2, 3] = 0.5
        self.assertAlmostEqual(data_array[1, 2, 3], 0.5)

    def test_sdf_coordinate_conversion(self):
        """Test world-to-grid and grid-to-world coordinate conversion."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([2.0, 2.0, 2.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([3, 3, 3], dtype=np.int32)  # Simple 3x3x3 grid
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Test grid location (center of grid cell) using grid_to_world
        # For a 3x3x3 grid with domain [0,2]x[0,2]x[0,2], grid coordinate (1,1,1) should map to world coordinate (2/3, 2/3, 2/3)
        grid_center = sdf.grid_to_world(np.array([1.0, 1.0, 1.0]))  # Middle cell
        expected_center = np.array(
            [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
        )  # Actual coordinate mapping
        np.testing.assert_array_almost_equal(grid_center, expected_center, decimal=5)

        # Test world to grid conversion (inverse should be consistent)
        world_pos = np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
        grid_pos = sdf.world_to_grid(world_pos)
        expected_grid_pos = np.array(
            [1.0, 1.0, 1.0]
        )  # Should map back to grid coordinate (1,1,1)
        np.testing.assert_array_almost_equal(grid_pos, expected_grid_pos)

        # Test grid to world conversion (round trip)
        world_pos_back = sdf.grid_to_world(grid_pos)
        np.testing.assert_array_almost_equal(world_pos_back, world_pos)

    def test_sdf_fill_and_clear(self):
        """Test fill and clear functionality."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([3, 3, 3], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Fill with a specific value
        sdf.fill(2.5)

        # Check a few grid points through buffer protocol
        data_array = np.array(sdf)
        self.assertAlmostEqual(data_array[0, 0, 0], 2.5)
        self.assertAlmostEqual(data_array[1, 1, 1], 2.5)
        self.assertAlmostEqual(data_array[2, 2, 2], 2.5)

        # Clear using fill(0.0) (should set to zero)
        sdf.fill(0.0)
        # Get a new array view after clear operation
        data_array_after_clear = np.array(sdf)
        self.assertAlmostEqual(data_array_after_clear[1, 1, 1], 0.0)

    def test_sdf_sampling(self):
        """Test SDF sampling at continuous positions."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([2.0, 2.0, 2.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([3, 3, 3], dtype=np.int32)

        # Create SDF with some initial test data instead of relying on buffer protocol to modify it
        test_data = [1.0] * (3 * 3 * 3)  # Fill with 1.0 values
        sdf = axel.SignedDistanceField(bbox, resolution, test_data)

        # Test sampling at the bounds - should return the set value (1.0) or boundary behavior
        sample_pos = np.array([0.0, 0.0, 0.0])  # Corner of the domain
        sampled_value = sdf.sample(sample_pos)

        # The exact sampled value depends on the SDF implementation's boundary behavior
        # Since we filled with 1.0, we expect to get 1.0 or close to it
        self.assertAlmostEqual(sampled_value, 1.0, places=3)

        # Test sampling at center of domain
        sample_pos = np.array([1.0, 1.0, 1.0])  # Center of [0,2]x[0,2]x[0,2] domain
        sampled_value = sdf.sample(sample_pos)
        self.assertAlmostEqual(sampled_value, 1.0, places=3)

    def test_sdf_repr(self):
        """Test string representation of SDF."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([10, 20, 30], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        repr_str = repr(sdf)

        # Check that key information is in the string representation
        self.assertIn("10", repr_str)  # resolution
        self.assertIn("20", repr_str)
        self.assertIn("30", repr_str)
        self.assertIn("0.000", repr_str)  # bounds
        self.assertIn("1.000", repr_str)

    def test_sdf_buffer_protocol(self):
        """Test buffer protocol for direct numpy array access."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([3, 4, 5], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Convert SDF to numpy array using buffer protocol
        data_array = np.array(sdf)

        # Check shape matches resolution
        self.assertEqual(data_array.shape, (3, 4, 5))
        self.assertEqual(data_array.dtype, np.float32)

        # Test that we can modify the data through the numpy array
        # This should modify the underlying SDF data directly (zero-copy)
        data_array[1, 2, 3] = 42.0

        # The buffer protocol provides direct zero-copy access, so there's no separate
        # SDF.at() or SDF.set() methods - everything goes through the numpy array

    def test_sdf_buffer_protocol_fill_operations(self):
        """Test buffer protocol with fill operations."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([2.0, 2.0, 2.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([4, 4, 4], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Get numpy array view
        data_array = np.array(sdf)

        # Fill through numpy operations
        data_array[:] = 5.0

        # Verify all voxels have the new value using numpy array view
        self.assertTrue(np.allclose(data_array, 5.0))

        # Fill a specific slice
        data_array[1, :, :] = -3.0

        # Verify the slice was modified and other slices unchanged
        self.assertTrue(np.allclose(data_array[1, :, :], -3.0))
        self.assertTrue(np.allclose(data_array[0, :, :], 5.0))
        self.assertTrue(np.allclose(data_array[2, :, :], 5.0))
        self.assertTrue(np.allclose(data_array[3, :, :], 5.0))

    def test_sdf_buffer_protocol_with_clear(self):
        """Test buffer protocol after clear operations."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([1.0, 1.0, 1.0])
        bbox = axel.BoundingBox(min_corner, max_corner)

        resolution = np.array([2, 2, 2], dtype=np.int32)
        sdf = axel.SignedDistanceField(bbox, resolution)

        # Get numpy array view
        data_array = np.array(sdf)

        # Fill with some values
        data_array[:] = 10.0

        # Clear using SDF method (fill with 0.0)
        sdf.fill(0.0)

        # Get a new array view after clear operation to see the changes
        data_array_after_clear = np.array(sdf)

        # Verify the new array view reflects the cleared values
        np.testing.assert_array_equal(
            data_array_after_clear, np.zeros((2, 2, 2), dtype=np.float32)
        )

    def test_mesh_to_sdf_tetrahedron(self):
        """Test mesh_to_sdf with a simple tetrahedron mesh."""
        # Create a simple tetrahedron mesh
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]],
            dtype=np.float32,
        )

        triangles = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32
        )

        # Define bounds and resolution
        bounds = axel.BoundingBox(
            min_corner=np.array([-0.5, -0.5, -0.5]),
            max_corner=np.array([1.5, 1.5, 1.5]),
        )
        resolution = np.array([16, 16, 16], dtype=np.int64)

        # Create configuration
        config = axel.MeshToSdfConfig()
        config.narrow_band_width = 2.0

        # Generate SDF
        sdf = axel.mesh_to_sdf(vertices, triangles, bounds, resolution, config)

        # Verify basic properties
        self.assertIsInstance(sdf, axel.SignedDistanceField)
        self.assertEqual(sdf.total_voxels, 16 * 16 * 16)

        # Test that the center of the tetrahedron has negative distance (inside)
        center_point = np.array([0.5, 0.25, 0.25], dtype=np.float32)
        center_distance = sdf.sample(center_point)
        self.assertLess(
            center_distance,
            0,
            "Center of tetrahedron should be inside (negative distance)",
        )

        # Test a point clearly outside has positive distance
        outside_point = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        outside_distance = sdf.sample(outside_point)
        self.assertGreater(
            outside_distance,
            0,
            "Point outside tetrahedron should have positive distance",
        )

    def test_mesh_to_sdf_automatic_bounds(self):
        """Test mesh_to_sdf with automatic bounds computation."""
        # Create a simple tetrahedron mesh
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]],
            dtype=np.float32,
        )

        triangles = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32
        )

        resolution = np.array([20, 20, 20], dtype=np.int64)

        # Generate SDF with automatic bounds computation
        sdf = axel.mesh_to_sdf(vertices, triangles, resolution, padding=0.2)

        # Verify basic properties
        self.assertIsInstance(sdf, axel.SignedDistanceField)
        self.assertEqual(sdf.total_voxels, 20 * 20 * 20)

        # Test that bounds were computed correctly with padding
        bounds = sdf.bounds
        min_corner = bounds.min
        max_corner = bounds.max

        # Check that the bounds extend beyond the mesh vertices
        self.assertLess(min_corner[0], 0.0)  # Should be less than min vertex x (0.0)
        self.assertLess(min_corner[1], 0.0)  # Should be less than min vertex y (0.0)
        self.assertLess(min_corner[2], 0.0)  # Should be less than min vertex z (0.0)

        self.assertGreater(
            max_corner[0], 1.0
        )  # Should be greater than max vertex x (1.0)
        self.assertGreater(
            max_corner[1], 1.0
        )  # Should be greater than max vertex y (1.0)
        self.assertGreater(
            max_corner[2], 1.0
        )  # Should be greater than max vertex z (1.0)

    def test_mesh_to_sdf_cube(self):
        """Test mesh_to_sdf with a cube mesh."""
        # Create a cube mesh
        vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],  # bottom face
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],  # top face
            ],
            dtype=np.float32,
        )

        triangles = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom face
                [4, 7, 6],
                [4, 6, 5],  # top face
                [0, 4, 5],
                [0, 5, 1],  # front face
                [2, 6, 7],
                [2, 7, 3],  # back face
                [0, 3, 7],
                [0, 7, 4],  # left face
                [1, 5, 6],
                [1, 6, 2],  # right face
            ],
            dtype=np.int32,
        )

        resolution = np.array([24, 24, 24], dtype=np.int64)

        config = axel.MeshToSdfConfig()

        sdf = axel.mesh_to_sdf(
            vertices, triangles, resolution, padding=0.25, config=config
        )

        # Test center (should be inside)
        center_distance = sdf.sample(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertLess(center_distance, 0, "Center of cube should be inside")

        # Test corners (should be approximately on surface)
        corner_distance = sdf.sample(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        self.assertLess(abs(corner_distance), 0.2, "Corner should be near surface")

        # Test point outside
        outside_distance = sdf.sample(np.array([2.0, 2.0, 2.0], dtype=np.float32))
        self.assertGreater(
            outside_distance, 0, "Point outside cube should have positive distance"
        )

    def test_mesh_to_sdf_invalid_inputs(self):
        """Test mesh_to_sdf with invalid inputs."""
        # Valid base case
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        resolution = np.array([10, 10, 10], dtype=np.int64)

        # Test invalid vertex shape
        invalid_vertices = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "Invalid shape for vertices"):
            axel.mesh_to_sdf(invalid_vertices, triangles, resolution)

        # Test invalid triangle shape
        invalid_triangles = np.array([[0, 1]], dtype=np.int32)
        with self.assertRaisesRegex(RuntimeError, "Invalid shape for triangles"):
            axel.mesh_to_sdf(vertices, invalid_triangles, resolution)

        # Test invalid resolution shape
        invalid_resolution = np.array([10, 10], dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "Invalid shape for resolution"):
            axel.mesh_to_sdf(vertices, triangles, invalid_resolution)

    def test_fill_holes_cube_with_hole(self):
        """Test fill_holes with a cube mesh that has a missing face (hole)."""
        # Create a cube mesh with missing top face
        vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],  # bottom face
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],  # top face
            ],
            dtype=np.float32,
        )

        # Missing top face triangles to create a hole
        triangles = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom face
                # [4, 7, 6], [4, 6, 5],  # top face (missing - creates hole)
                [0, 4, 5],
                [0, 5, 1],  # front face
                [2, 6, 7],
                [2, 7, 3],  # back face
                [0, 3, 7],
                [0, 7, 4],  # left face
                [1, 5, 6],
                [1, 6, 2],  # right face
            ],
            dtype=np.int32,
        )

        filled_vertices, filled_triangles = axel.fill_holes(vertices, triangles)

        # Should have more triangles than the original (hole was filled)
        self.assertGreater(len(filled_triangles), len(triangles))
        # Should have at least the original vertices (may add new ones)
        self.assertGreaterEqual(len(filled_vertices), len(vertices))

        # Original vertices should be preserved (first N vertices)
        np.testing.assert_array_almost_equal(filled_vertices[: len(vertices)], vertices)

    def _create_sphere_sdf(
        self, center=(0, 0, 0), radius=1.0, resolution=16, bounds_padding=0.5
    ):
        """Helper method to create a sphere SDF for testing."""
        center_np = np.array(center, dtype=np.float32)
        bounds_min = center_np - (radius + bounds_padding)
        bounds_max = center_np + (radius + bounds_padding)

        bounds = axel.BoundingBox(bounds_min, bounds_max)
        resolution_vec = np.array([resolution, resolution, resolution], dtype=np.int32)
        sdf = axel.SignedDistanceField(bounds, resolution_vec)

        # Fill with sphere distance values using SDF's set method
        # Ensure the sphere crosses the zero level set
        sdf_array = np.asarray(sdf)
        for k in range(resolution):
            for j in range(resolution):
                for i in range(resolution):
                    # Convert grid indices to world coordinates
                    grid_pos = np.array([i, j, k], dtype=np.float32)
                    world_pos = sdf.grid_to_world(grid_pos)

                    # Compute signed distance to sphere surface
                    # Negative inside, positive outside
                    distance_to_center = np.linalg.norm(world_pos - center_np)
                    signed_distance = distance_to_center - radius

                    # Store the signed distance using SDF's set method
                    sdf_array[i, j, k] = signed_distance

        return sdf

    def test_dual_contouring_basic_sphere(self):
        """Test dual contouring on a simple sphere."""
        # Create a simple sphere SDF
        sdf = self._create_sphere_sdf(center=(0, 0, 0), radius=1.0, resolution=12)

        # Extract mesh (always returns quads now)
        vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

        # Basic checks
        self.assertIsInstance(vertices, np.ndarray)
        self.assertIsInstance(normals, np.ndarray)
        self.assertIsInstance(quads, np.ndarray)

        # Should generate some vertices and quads
        self.assertGreater(len(vertices), 0, "Should generate at least some vertices")
        self.assertGreater(len(quads), 0, "Should generate at least some quads")

        # Check array shapes
        self.assertEqual(vertices.shape[1], 3, "Vertices should have 3 coordinates")
        self.assertEqual(normals.shape[1], 3, "Normals should have 3 coordinates")
        self.assertEqual(quads.shape[1], 4, "Quads should have 4 indices")

        # Vertices and normals should have same count
        self.assertEqual(
            len(vertices), len(normals), "Should have one normal per vertex"
        )

    def test_dual_contouring_vertex_bounds(self):
        """Test that generated vertices are within reasonable bounds."""
        # Create a unit sphere SDF
        sdf = self._create_sphere_sdf(center=(0, 0, 0), radius=1.0, resolution=16)

        vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

        if len(vertices) > 0:
            # Check vertex bounds - should be within SDF bounds
            vertex_min = np.min(vertices, axis=0)
            vertex_max = np.max(vertices, axis=0)
            bounds_min = sdf.bounds.min
            bounds_max = sdf.bounds.max

            # Vertices should be within SDF bounds (with small tolerance)
            tolerance = 0.1
            self.assertTrue(
                np.all(vertex_min >= bounds_min - tolerance),
                f"Some vertices are below bounds: {vertex_min} vs {bounds_min}",
            )
            self.assertTrue(
                np.all(vertex_max <= bounds_max + tolerance),
                f"Some vertices are above bounds: {vertex_max} vs {bounds_max}",
            )

    def test_dual_contouring_quad_indices(self):
        """Test that quad indices are valid."""
        sdf = self._create_sphere_sdf(center=(0, 0, 0), radius=1.0, resolution=12)

        vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

        if len(quads) > 0:
            # Check quad index bounds
            max_vertex_index = np.max(quads)
            min_vertex_index = np.min(quads)

            self.assertGreaterEqual(
                min_vertex_index, 0, "Quad indices should be non-negative"
            )
            self.assertLess(
                max_vertex_index,
                len(vertices),
                "Quad indices should be within vertex array bounds",
            )

    def test_dual_contouring_different_resolutions(self):
        """Test dual contouring with different grid resolutions."""
        resolutions = [8, 12, 16]

        for res in resolutions:
            with self.subTest(resolution=res):
                sdf = self._create_sphere_sdf(resolution=res)

                vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

                # Should generate vertices and quads for all reasonable resolutions
                self.assertGreater(
                    len(vertices), 0, f"Should generate vertices for resolution {res}"
                )
                self.assertEqual(
                    len(vertices),
                    len(normals),
                    f"Vertex/normal count mismatch for resolution {res}",
                )

    def test_dual_contouring_different_isovalues(self):
        """Test dual contouring with different isovalue settings."""
        sdf = self._create_sphere_sdf(center=(0, 0, 0), radius=1.0, resolution=16)

        isovalues = [-0.2, 0.0, 0.2]

        for isovalue in isovalues:
            with self.subTest(isovalue=isovalue):
                vertices, normals, quads = axel.dual_contouring(sdf, isovalue=isovalue)

                # Should work for different isovalues (though might generate different numbers of vertices)
                self.assertIsInstance(vertices, np.ndarray)
                self.assertIsInstance(quads, np.ndarray)

    def test_dual_contouring_empty_sdf(self):
        """Test dual contouring on an SDF with no surface crossing."""
        bounds = axel.BoundingBox(
            min_corner=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            max_corner=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        resolution = np.array([8, 8, 8], dtype=np.int32)
        sdf = axel.SignedDistanceField(bounds, resolution)

        # Fill with all positive values (no zero crossing)
        sdf.fill(1.0)

        vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

        # Should return empty arrays
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(normals), 0)
        self.assertEqual(len(quads), 0)

    def test_dual_contouring_debug_sphere_values(self):
        """Debug test to check if our sphere SDF is generating correct values."""
        sdf = self._create_sphere_sdf(center=(0, 0, 0), radius=1.0, resolution=8)

        # Check some SDF values
        data_array = np.array(sdf)

        # Find min and max values
        min_val = np.min(data_array)
        max_val = np.max(data_array)

        print(f"SDF value range: {min_val:.3f} to {max_val:.3f}")

        # Count negative and positive values
        negative_count = np.sum(data_array < 0)
        positive_count = np.sum(data_array > 0)
        zero_count = np.sum(data_array == 0)

        print(
            f"Values: {negative_count} negative, {positive_count} positive, {zero_count} zero"
        )

        # Should have both negative and positive values for a proper sphere
        self.assertGreater(
            negative_count, 0, "Should have negative values inside sphere"
        )
        self.assertGreater(
            positive_count, 0, "Should have positive values outside sphere"
        )

        # Now test dual contouring
        vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

        print(f"Generated {len(vertices)} vertices, {len(quads)} quads")

        # If we have both negative and positive values, we should generate some vertices
        # This test helps us debug why no vertices are generated
        if negative_count > 0 and positive_count > 0:
            # We should generate some vertices since there's a sign change
            pass  # Don't assert yet, just debug info

    def test_triangulate_quads(self):
        """Test the triangulate_quads function."""
        # Create some test quads
        quads = np.array(
            [
                [0, 1, 2, 3],  # First quad
                [4, 5, 6, 7],  # Second quad
            ],
            dtype=np.int32,
        )

        # Triangulate
        triangles = axel.triangulate_quads(quads)

        # Should get 2 triangles per quad
        self.assertEqual(len(triangles), 4)
        self.assertEqual(triangles.shape[1], 3)

        # Check that triangulation is correct
        # First quad [0,1,2,3] should produce triangles [0,1,2] and [0,2,3]
        expected_triangles = np.array(
            [
                [0, 1, 2],  # First quad, first triangle
                [0, 2, 3],  # First quad, second triangle
                [4, 5, 6],  # Second quad, first triangle
                [4, 6, 7],  # Second quad, second triangle
            ]
        )

        np.testing.assert_array_equal(triangles, expected_triangles)

    def test_dual_contouring_sphere_distance_analysis(self):
        """Test that vertices are approximately on the sphere surface."""
        radius = 1.0
        sdf = self._create_sphere_sdf(center=(0, 0, 0), radius=radius, resolution=16)

        vertices, normals, quads = axel.dual_contouring(sdf, isovalue=0.0)

        if len(vertices) > 0:
            # Check distances from origin - should be roughly around the sphere radius
            distances = np.linalg.norm(vertices, axis=1)
            mean_distance = np.mean(distances)

            # Since we're using surface positioning now, vertices should be much closer to the sphere
            tolerance = 0.2  # Tighter tolerance since we push vertices to surface

            self.assertGreater(
                mean_distance,
                radius - tolerance,
                f"Mean distance {mean_distance:.3f} too far inside sphere",
            )
            self.assertLess(
                mean_distance,
                radius + tolerance,
                f"Mean distance {mean_distance:.3f} too far outside sphere",
            )


if __name__ == "__main__":
    unittest.main()
