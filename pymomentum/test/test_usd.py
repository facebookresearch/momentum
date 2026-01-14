# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import tempfile
import unittest

import pymomentum.geometry as pym_geometry

# Check USD availability once at module load for skipIf decorator
_USD_AVAILABLE = pym_geometry.is_usd_available()


class TestUsd(unittest.TestCase):
    @unittest.skipIf(not _USD_AVAILABLE, "USD support not available")
    def test_is_usd_available(self) -> None:
        """Test that USD support is available.

        In BUCK builds, USD is always available (preprocessor_flags includes
        -DMOMENTUM_WITH_USD). In OSS/CMake builds, availability depends on
        MOMENTUM_BUILD_IO_USD option.
        """
        result = pym_geometry.is_usd_available()
        self.assertIsInstance(result, bool)
        # When this test runs, USD should be available
        self.assertTrue(result)

    @unittest.skipIf(not _USD_AVAILABLE, "USD support not available")
    def test_save_and_load_usd_character(self) -> None:
        """Test saving and reloading a USD character."""

        # Create a test character
        char = pym_geometry.create_test_character()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
            temp_path = f.name

        try:
            pym_geometry.Character.save_usd(temp_path, char)  # type: ignore[attr-defined]

            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)

            # Reload and verify
            loaded_char = pym_geometry.Character.load_usd(temp_path)  # type: ignore[attr-defined]

            # Verify skeleton joints match
            self.assertEqual(
                len(loaded_char.skeleton.joints),
                len(char.skeleton.joints),
            )

            # Verify joint names match
            for i, (loaded_joint, orig_joint) in enumerate(
                zip(loaded_char.skeleton.joints, char.skeleton.joints)
            ):
                self.assertEqual(
                    loaded_joint.name,
                    orig_joint.name,
                    f"Joint name mismatch at index {i}",
                )

            # Verify mesh vertices match
            if char.mesh is not None:
                self.assertIsNotNone(loaded_char.mesh)
                self.assertEqual(
                    len(loaded_char.mesh.vertices),
                    len(char.mesh.vertices),
                )
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @unittest.skipIf(not _USD_AVAILABLE, "USD support not available")
    def test_load_usd_from_bytes(self) -> None:
        """Test loading a USD character from bytes after saving."""
        # Create a test character and save it
        char = pym_geometry.create_test_character()

        with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
            temp_path = f.name

        try:
            pym_geometry.Character.save_usd(temp_path, char)  # type: ignore[attr-defined]

            # Read file contents
            with open(temp_path, "rb") as f:
                usd_bytes = f.read()

            # Load from bytes
            loaded_char = pym_geometry.Character.load_usd_from_bytes(usd_bytes)  # type: ignore[attr-defined]

            # Verify skeleton was loaded
            self.assertGreater(len(loaded_char.skeleton.joints), 0)

            # Verify skeleton joints match
            self.assertEqual(
                len(loaded_char.skeleton.joints),
                len(char.skeleton.joints),
            )
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
