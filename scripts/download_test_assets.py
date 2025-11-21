#!/usr/bin/env python3
"""Download official Momentum Human Rig (MHR) assets.

These assets are the foundational data required to create the official
Momentum Human Rig, including character models, blend shapes, and model
configurations released as part of the MHR project.

Source: https://github.com/facebookresearch/MHR/releases/tag/v1.0.0
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ASSETS_URL = (
    "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"
)
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSET_MARKER = ASSETS_DIR / ".downloaded"


def download_assets(force: bool = False) -> bool:
    """Download and extract official MHR assets.

    Downloads the Momentum Human Rig (MHR) assets including:
    - Character models at various LODs (FBX format)
    - Corrective blend shapes
    - Model configuration files
    - PyTorch model weights

    Args:
        force: Force re-download even if assets exist

    Returns:
        True if assets are ready, False on failure
    """

    # Check if already downloaded
    if ASSET_MARKER.exists() and not force:
        print(f"MHR assets already present at {ASSETS_DIR}")
        return True

    print(f"Downloading MHR assets from {ASSETS_URL}...")
    print("These are the official Momentum Human Rig assets (~100MB)")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ASSETS_DIR / "assets.zip"

    try:
        # Download
        print("Downloading...")
        urlretrieve(ASSETS_URL, zip_path)
        print(f"Downloaded to {zip_path}")

        # Extract
        print("Extracting assets...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ASSETS_DIR)
        print(f"Extracted to {ASSETS_DIR}")

        # Clean up zip
        zip_path.unlink()

        # Create marker file
        ASSET_MARKER.write_text(f"Downloaded from {ASSETS_URL}")

        print("MHR assets ready!")
        return True

    except Exception as e:
        print(f"Failed to download assets: {e}", file=sys.stderr)
        if zip_path.exists():
            zip_path.unlink()
        return False


def clean_assets():
    """Remove downloaded MHR assets."""
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)
        print(f"Removed {ASSETS_DIR}")
    else:
        print("No assets to clean")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage official Momentum Human Rig (MHR) assets"
    )
    parser.add_argument("--clean", action="store_true", help="Remove downloaded assets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    if args.clean:
        clean_assets()
    else:
        success = download_assets(force=args.force)
        sys.exit(0 if success else 1)
