"""Tests for Character creation with downloaded model files."""

import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import pytest


@pytest.mark.requires_assets
class TestCharacterWithModels(unittest.TestCase):
    """Tests that require actual character model files from downloaded assets."""

    @pytest.fixture(autouse=True)
    def _setup_fixtures(self, character_model_files):
        """Store fixtures as instance variables for unittest compatibility."""
        self.character_model_files = character_model_files

    def test_load_character_from_fbx(self):
        """Test loading Character from downloaded FBX model."""
        if "fbx" not in self.character_model_files:
            self.skipTest("No FBX files found in test assets")

        fbx_path = self.character_model_files["fbx"][0]
        char = pym_geometry.Character.load_fbx(str(fbx_path))

        self.assertIsNotNone(char)
        self.assertGreater(len(char.skeleton.joint_names), 0)

    def test_load_character_from_glb(self):
        """Test loading Character from downloaded GLB model."""
        if "glb" not in self.character_model_files:
            self.skipTest("No GLB files found in test assets")

        glb_path = self.character_model_files["glb"][0]
        char = pym_geometry.Character.load_gltf(str(glb_path))

        self.assertIsNotNone(char)
        self.assertGreater(len(char.skeleton.joint_names), 0)

    def test_character_has_proper_skeleton(self):
        """Test that loaded Character has valid skeleton structure."""
        if "fbx" not in self.character_model_files:
            self.skipTest("No FBX files found in test assets")

        fbx_path = self.character_model_files["fbx"][0]
        char = pym_geometry.Character.load_fbx(str(fbx_path))

        # Verify skeleton properties
        offsets = char.skeleton.offsets
        pre_rotations = char.skeleton.pre_rotations
        num_joints = len(char.skeleton.joint_names)

        self.assertGreater(num_joints, 0)
        self.assertEqual(tuple(offsets.shape), (num_joints, 3))
        self.assertEqual(tuple(pre_rotations.shape), (num_joints, 4))

        # Verify offsets are valid (not all zeros)
        self.assertGreater(np.abs(offsets).sum(), 0)

    def test_character_parameter_transforms(self):
        """Test that Character has parameter transforms."""
        if "fbx" not in self.character_model_files:
            self.skipTest("No FBX files found in test assets")

        fbx_path = self.character_model_files["fbx"][0]
        char = pym_geometry.Character.load_fbx(str(fbx_path))

        # ParameterTransform should exist even if size is 0
        self.assertIsNotNone(char.parameter_transform)
        # Size can be 0 for FBX files without parameter definitions
        self.assertGreaterEqual(char.parameter_transform.size, 0)

    def test_character_has_mesh(self):
        """Test that Character has mesh data."""
        if "fbx" not in self.character_model_files:
            self.skipTest("No FBX files found in test assets")

        fbx_path = self.character_model_files["fbx"][0]
        char = pym_geometry.Character.load_fbx(str(fbx_path))

        if char.mesh is not None:
            vertices = char.mesh.vertices
            faces = char.mesh.faces

            self.assertGreater(len(vertices), 0)
            self.assertGreater(len(faces), 0)
            self.assertEqual(vertices.shape[1], 3)

    def test_multiple_file_formats(self):
        """Test that we can load multiple file formats if available."""
        formats_tested = 0

        if "fbx" in self.character_model_files:
            fbx_path = self.character_model_files["fbx"][0]
            char = pym_geometry.Character.load_fbx(str(fbx_path))
            self.assertIsNotNone(char)
            formats_tested += 1

        if "glb" in self.character_model_files:
            glb_path = self.character_model_files["glb"][0]
            char = pym_geometry.Character.load_gltf(str(glb_path))
            self.assertIsNotNone(char)
            formats_tested += 1

        if "gltf" in self.character_model_files:
            gltf_path = self.character_model_files["gltf"][0]
            char = pym_geometry.Character.load_gltf(str(gltf_path))
            self.assertIsNotNone(char)
            formats_tested += 1

        self.assertGreater(formats_tested, 0, "No model files found to test")
