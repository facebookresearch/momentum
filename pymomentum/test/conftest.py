"""Pytest configuration for pymomentum tests."""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_test_assets import ASSETS_DIR, download_assets


@pytest.fixture(scope="session", autouse=True)
def test_assets():
    """Download test assets before running any tests that need them.

    This fixture runs once per test session and ensures that test assets
    are downloaded before any tests execute. It's not autouse for all tests,
    but provides the assets directory when requested.
    """
    return ASSETS_DIR


@pytest.fixture
def character_model_files(test_assets):
    """Provide paths to character model files from downloaded assets.

    Returns:
        A dictionary with paths to various model files, or None if not found.
    """
    # First, ensure assets are downloaded
    if not download_assets():
        pytest.skip("Failed to download test assets")

    # Scan the assets directory to find available model files
    model_files = {}

    # Look for FBX files
    fbx_files = list(test_assets.glob("**/*.fbx"))
    if fbx_files:
        model_files["fbx"] = fbx_files

    # Look for GLB files
    glb_files = list(test_assets.glob("**/*.glb"))
    if glb_files:
        model_files["glb"] = glb_files

    # Look for GLTF files
    gltf_files = list(test_assets.glob("**/*.gltf"))
    if gltf_files:
        model_files["gltf"] = gltf_files

    if not model_files:
        pytest.skip(f"No model files found in {test_assets}")

    return model_files


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_assets: mark test as requiring downloaded model assets"
    )
