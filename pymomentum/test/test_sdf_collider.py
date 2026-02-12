#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for SDFCollider Python bindings."""

import unittest

import numpy as np


class TestSDFCollider(unittest.TestCase):
    """Tests for SDFCollider bindings."""

    def test_import(self) -> None:
        """Test that SDFCollider can be imported."""
        from pymomentum import geometry

        # Should be able to access the class
        self.assertTrue(hasattr(geometry, "SDFCollider"))

    def test_create_empty_collider(self) -> None:
        """Test creating an empty SDFCollider."""
        from pymomentum import geometry

        collider = geometry.SDFCollider()
        self.assertEqual(collider.parent, -1)  # World-space by default
        self.assertFalse(collider.is_valid())  # No SDF pointer
        np.testing.assert_array_almost_equal(
            collider.translation, np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )

    def test_create_collider_with_parent(self) -> None:
        """Test creating an SDFCollider with a parent joint."""
        from pymomentum import geometry

        collider = geometry.SDFCollider(
            translation=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            parent=5,
        )
        self.assertEqual(collider.parent, 5)
        self.assertFalse(collider.is_valid())  # Still no SDF
        np.testing.assert_array_almost_equal(
            collider.translation, np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    def test_create_collider_with_sdf(self) -> None:
        """Test creating an SDFCollider with an SDF."""
        from pymomentum import axel, geometry

        # Create a simple sphere SDF
        bounds = axel.BoundingBox(
            np.array([-2.0, -2.0, -2.0], dtype=np.float32),
            np.array([2.0, 2.0, 2.0], dtype=np.float32),
        )
        sdf = axel.SignedDistanceField(bounds, np.array([10, 10, 10]))

        collider = geometry.SDFCollider(
            translation=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            parent=3,
            sdf=sdf,
        )
        self.assertEqual(collider.parent, 3)
        self.assertTrue(collider.is_valid())
        np.testing.assert_array_almost_equal(
            collider.translation, np.array([0.5, 0.5, 0.5], dtype=np.float32)
        )

    def test_repr(self) -> None:
        """Test SDFCollider __repr__."""
        from pymomentum import geometry

        collider = geometry.SDFCollider(
            translation=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            parent=2,
        )
        repr_str = repr(collider)
        self.assertIn("SDFCollider", repr_str)
        self.assertIn("parent=2", repr_str)
        self.assertIn("valid=False", repr_str)


if __name__ == "__main__":
    unittest.main()
