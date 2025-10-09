# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import tempfile
import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import torch


class TestLegacyJsonIO(unittest.TestCase):
    def setUp(self) -> None:
        self.character = pym_geometry.create_test_character()
        torch.manual_seed(0)  # ensure repeatability

    def test_round_trip_conversion(self) -> None:
        """Test that Character -> Legacy JSON -> Character preserves data."""
        # Convert character to legacy JSON string
        json_string = pym_geometry.Character.to_legacy_json_string(self.character)

        # Convert back to character
        round_trip_character = pym_geometry.Character.load_legacy_json_from_string(
            json_string
        )

        # Verify skeleton structure is preserved
        self.assertEqual(
            round_trip_character.skeleton.size, self.character.skeleton.size
        )
        self.assertEqual(
            round_trip_character.skeleton.joint_names,
            self.character.skeleton.joint_names,
        )
        self.assertEqual(
            round_trip_character.skeleton.joint_parents,
            self.character.skeleton.joint_parents,
        )

        # Compare joint offsets and pre-rotations
        np.testing.assert_allclose(
            round_trip_character.skeleton.offsets,
            self.character.skeleton.offsets,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            round_trip_character.skeleton.pre_rotations,
            self.character.skeleton.pre_rotations,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_file_operations(self) -> None:
        """Test saving to and loading from files."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            # Save character to file
            pym_geometry.Character.save_legacy_json(self.character, temp_file.name)

            # Load character from file
            loaded_character = pym_geometry.Character.load_legacy_json(temp_file.name)

            # Verify loaded character matches original
            self.assertEqual(
                loaded_character.skeleton.size, self.character.skeleton.size
            )
            self.assertEqual(
                loaded_character.skeleton.joint_names,
                self.character.skeleton.joint_names,
            )

    def test_bytes_operations(self) -> None:
        """Test loading from bytes."""
        # Convert to string first
        json_string = pym_geometry.Character.to_legacy_json_string(self.character)

        # Convert to bytes and load
        json_bytes = json_string.encode("utf-8")
        loaded_character = pym_geometry.Character.load_legacy_json_from_bytes(
            json_bytes
        )

        # Verify loaded character matches original
        self.assertEqual(loaded_character.skeleton.size, self.character.skeleton.size)
        self.assertEqual(
            loaded_character.skeleton.joint_names, self.character.skeleton.joint_names
        )


if __name__ == "__main__":
    unittest.main()
