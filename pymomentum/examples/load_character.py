"""Example: Loading and inspecting MHR Characters from model files.

This example demonstrates how to:
1. Download official Momentum Human Rig (MHR) assets
2. Load MHR Characters from FBX files at different LODs
3. Inspect skeleton structure, parameters, and mesh data
4. Understand the multi-resolution character models

The MHR provides professional-quality character models with full skeletal rigs,
multi-resolution meshes (LOD 0-6), and corrective blend shapes.

Run with:
    pixi run download_assets
    pixi run -e py312 python pymomentum/examples/load_character.py
"""

import sys
from pathlib import Path

# Add scripts to path for asset download
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pymomentum.geometry as pym_geometry
from download_test_assets import ASSETS_DIR, download_assets


def find_model_files(assets_dir):
    """Find available model files in the assets directory."""
    model_files = {
        "fbx": list(assets_dir.glob("**/*.fbx")),
        "glb": list(assets_dir.glob("**/*.glb")),
        "gltf": list(assets_dir.glob("**/*.gltf")),
    }
    return {k: v for k, v in model_files.items() if v}


def inspect_character(char, name="Character"):
    """Print detailed information about a Character."""
    print(f"\n{'=' * 60}")
    print(f"{name} Information")
    print(f"{'=' * 60}")

    # Skeleton info
    num_joints = len(char.skeleton.joint_names)
    print(f"\nSkeleton:")
    print(f"  Total joints: {num_joints}")
    print(f"  Joint names (first 10):")
    for i, name in enumerate(char.skeleton.joint_names[:10]):
        print(f"    [{i}] {name}")
    if num_joints > 10:
        print(f"    ... and {num_joints - 10} more")

    # Parameter info
    num_params = char.parameter_transform.size
    print(f"\nParameters:")
    print(f"  Total model parameters: {num_params}")

    # Mesh info
    if char.mesh is not None:
        vertices = char.mesh.vertices
        faces = char.mesh.faces
        print(f"\nMesh:")
        print(f"  Vertices: {vertices.shape[0]}")
        print(f"  Faces: {faces.shape[0]}")
    else:
        print(f"\nMesh: None")

    # Skeleton hierarchy (first few joints)
    print(f"\nSkeleton offsets (first 5 joints):")
    offsets = char.skeleton.offsets[:5]
    for i, offset in enumerate(offsets):
        print(f"  Joint {i}: [{offset[0]:7.3f}, {offset[1]:7.3f}, {offset[2]:7.3f}]")


def main():
    print("PyMomentum Character Loading Example")
    print("=" * 60)

    # Ensure assets are downloaded
    print("\nChecking for test assets...")
    if not download_assets():
        print("Failed to download assets")
        return 1

    print(f"Assets directory: {ASSETS_DIR}")

    # Find available model files
    model_files = find_model_files(ASSETS_DIR)

    if not model_files:
        print(f"\nNo model files found in {ASSETS_DIR}")
        print("Available files:")
        for f in ASSETS_DIR.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(ASSETS_DIR)}")
        return 1

    print(f"\nFound model files:")
    for format_type, files in model_files.items():
        print(f"  {format_type.upper()}: {len(files)} file(s)")
        for f in files[:3]:
            print(f"    - {f.relative_to(ASSETS_DIR)}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")

    # Load and inspect FBX files
    if "fbx" in model_files:
        for i, fbx_path in enumerate(model_files["fbx"][:2]):
            print(f"\n\nLoading FBX file: {fbx_path.name}")
            try:
                char = pym_geometry.Character.load_fbx(str(fbx_path))
                inspect_character(char, f"FBX Character {i+1}")
            except Exception as e:
                print(f"Failed to load {fbx_path.name}: {e}")

    # Load and inspect GLB files
    if "glb" in model_files:
        glb_path = model_files["glb"][0]
        print(f"\n\nLoading GLB file: {glb_path.name}")
        try:
            char = pym_geometry.Character.load_gltf(str(glb_path))
            inspect_character(char, "GLB Character")
        except Exception as e:
            print(f"Failed to load {glb_path.name}: {e}")

    print(f"\n{'=' * 60}")
    print("Example completed successfully!")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
