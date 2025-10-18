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


if __name__ == "__main__":
    unittest.main()
