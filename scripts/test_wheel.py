# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test wheel imports using uv for fast, reliable testing across platforms.

Supports:
- Linux: Tests in manylinux_2_28 container via docker or podman (auto-detected)
- macOS/Windows: Tests locally using uv virtual environments
- Multiple Python versions via uv's Python management
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_torch_index(wheel_type: str) -> str:
    """Get the appropriate PyTorch index URL for the wheel type."""
    if wheel_type == "gpu":
        return "https://download.pytorch.org/whl/cu128"
    return "https://download.pytorch.org/whl/cpu"


def run_import_tests(python_exe: str) -> int:
    """Run import and parallel operation tests using the specified Python."""
    test_script = """
import sys
import os

# Force UTF-8 output on Windows to avoid 'charmap' codec errors with Unicode characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

print("Testing imports...")
try:
    # IMPORTANT: Import torch first to set up library paths (LD_LIBRARY_PATH)
    # This ensures libtorch.so is findable when loading pymomentum native extensions.
    # Note: pymomentum's __init__.py should also do this, but we import torch
    # explicitly here as a backup measure.
    import torch
    import pymomentum.geometry as geom
    import pymomentum.solver as solver
    import pymomentum.solver2 as solver2
    import pymomentum.marker_tracking as marker_tracking
    import pymomentum.axel as axel
    print("[PASS] All imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}", file=sys.stderr)
    sys.exit(1)

print("Testing parallel operations (MHR issue #44 regression test)...")
try:
    import numpy as np
    import pymomentum.geometry as geom

    # Create a simple triangle mesh for testing
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    # compute_vertex_normals uses dispenso::parallel_for internally;
    # this would segfault if libgomp is not bundled correctly
    print("  Computing vertex normals...")
    normals = geom.compute_vertex_normals(vertices, triangles)
    assert normals.shape[-1] == 3, f"Expected 3D normals, got shape {normals.shape}"
    assert np.all(np.isfinite(normals)), "Normals contain non-finite values"
    print(f"  Normals shape: {normals.shape}")

    # find_closest_points also uses dispenso::parallel_for;
    # test with batched points to stress the parallel codepath
    source = np.random.rand(100, 3).astype(np.float32)
    target = np.random.rand(50, 3).astype(np.float32)
    print("  Finding closest points...")
    result = geom.find_closest_points(source, target)
    print("  Closest points computation completed")

    print("[PASS] Parallel operations test passed")
except Exception as e:
    print(f"[FAIL] Parallel operations test failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[PASS] All tests passed!")
"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [python_exe, "-c", test_script], capture_output=False, env=env
    )
    return result.returncode


def find_container_runtime() -> str | None:
    """Find an available container runtime, preferring docker over podman."""
    for runtime in ("docker", "podman"):
        if shutil.which(runtime):
            return runtime
    return None


def test_in_container(wheel_file: Path, wheel_type: str, py_ver: str) -> int:
    """Test wheel in a manylinux_2_28 container using docker or podman."""
    container_runtime = find_container_runtime()
    if container_runtime is None:
        print(
            "[FAIL] No container runtime found (tried docker, podman). "
            "Falling back to local uv-based testing.",
            file=sys.stderr,
        )
        return test_locally_with_uv(wheel_file, wheel_type, py_ver)

    print(
        f"Testing wheel in manylinux_2_28 container ({container_runtime}): "
        f"{wheel_file.name}"
    )

    torch_index = get_torch_index(wheel_type)
    py_tag = py_ver[2:]  # e.g., "312" or "313"

    container_script = f"""
set -e
PYTHON=/opt/python/cp{py_tag}-cp{py_tag}/bin/python

echo "Installing uv..."
$PYTHON -m pip install --quiet --root-user-action=ignore uv

echo "Creating virtual environment with Python {py_tag}..."
$PYTHON -m uv venv --python $PYTHON /tmp/test_venv

echo "Installing torch from {torch_index}..."
$PYTHON -m uv pip install --python /tmp/test_venv/bin/python "torch>=2.8.0,<2.9" --index-url {torch_index}

echo "Installing wheel..."
$PYTHON -m uv pip install --python /tmp/test_venv/bin/python /wheel/{wheel_file.name}

echo "Testing imports..."
/tmp/test_venv/bin/python -c "
import sys
try:
    # IMPORTANT: Import torch first to set up library paths (LD_LIBRARY_PATH)
    # This ensures libtorch.so is findable when loading pymomentum native extensions.
    # Note: pymomentum's __init__.py should also do this, but we import torch
    # explicitly here as a backup measure.
    import torch
    import pymomentum.geometry as geom
    import pymomentum.solver as solver
    import pymomentum.solver2 as solver2
    import pymomentum.marker_tracking as marker_tracking
    import pymomentum.axel as axel
    print('[PASS] All imports successful')
except Exception as e:
    print(f'[FAIL] Import failed: {{e}}', file=sys.stderr)
    sys.exit(1)
"

echo "Testing parallel operations (MHR issue #44 regression test)..."
/tmp/test_venv/bin/python -c "
import sys
import numpy as np
try:
    # IMPORTANT: Import torch first to set up library paths (LD_LIBRARY_PATH)
    import torch
    import pymomentum.geometry as geom

    # Create a simple triangle mesh for testing
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    # compute_vertex_normals uses dispenso::parallel_for internally;
    # this would segfault if libgomp is not bundled correctly
    print('  Computing vertex normals...')
    normals = geom.compute_vertex_normals(vertices, triangles)
    assert normals.shape[-1] == 3, f'Expected 3D normals, got shape {{normals.shape}}'
    assert np.all(np.isfinite(normals)), 'Normals contain non-finite values'
    print(f'  Normals shape: {{normals.shape}}')

    # find_closest_points also uses dispenso::parallel_for;
    # test with batched points to stress the parallel codepath
    source = np.random.rand(100, 3).astype(np.float32)
    target = np.random.rand(50, 3).astype(np.float32)
    print('  Finding closest points...')
    result = geom.find_closest_points(source, target)
    print('  Closest points computation completed')

    print('[PASS] Parallel operations test passed')
except Exception as e:
    print(f'[FAIL] Parallel operations test failed: {{e}}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
"""

    result = subprocess.run(
        [
            container_runtime,
            "run",
            "--rm",
            "-v",
            f"{wheel_file.parent.absolute()}:/wheel:ro",
            "quay.io/pypa/manylinux_2_28_x86_64:latest",
            "/bin/bash",
            "-c",
            container_script,
        ],
        capture_output=False,
    )

    return result.returncode


def test_locally_with_uv(wheel_file: Path, wheel_type: str, py_ver: str) -> int:
    """Test wheel locally using uv for fast virtual environment creation."""
    print(f"Testing wheel locally with uv: {wheel_file.name}")

    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        print("Installing uv...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True
        )
        uv_path = shutil.which("uv") or "uv"

    torch_index = get_torch_index(wheel_type)
    py_version = f"{py_ver[2]}.{py_ver[3:]}"  # "cp312" -> "3.12"

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "test_venv"

        # Determine python executable path based on platform
        if sys.platform == "win32":
            python_exe = str(venv_path / "Scripts" / "python.exe")
        else:
            python_exe = str(venv_path / "bin" / "python")

        print(f"Creating virtual environment with Python {py_version}...")
        result = subprocess.run(
            [uv_path, "venv", "--python", py_version, str(venv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to create venv: {result.stderr}", file=sys.stderr)
            return result.returncode

        print(f"Installing torch from {torch_index}...")
        result = subprocess.run(
            [
                uv_path,
                "pip",
                "install",
                "--python",
                python_exe,
                "torch>=2.8.0,<2.9",
                "--index-url",
                torch_index,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to install torch: {result.stderr}", file=sys.stderr)
            return result.returncode

        print(f"Installing wheel {wheel_file.name}...")
        result = subprocess.run(
            [uv_path, "pip", "install", "--python", python_exe, str(wheel_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to install wheel: {result.stderr}", file=sys.stderr)
            return result.returncode

        return run_import_tests(python_exe)


def main():
    # Allow override via environment variables for CI flexibility
    py_ver = os.environ.get("WHEEL_TEST_PYTHON_VERSION")
    if not py_ver:
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME", "")
    wheel_type = os.environ.get("WHEEL_TEST_TYPE")
    if not wheel_type:
        wheel_type = "gpu" if "cuda" in env_name or "gpu" in env_name else "cpu"

    print(f"Testing {wheel_type} wheel for Python {py_ver}")

    # Find the wheel file
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("[FAIL] dist/ directory not found", file=sys.stderr)
        sys.exit(1)

    wheel_files = [
        f
        for f in dist_dir.iterdir()
        if f.name.startswith(f"pymomentum_{wheel_type}")
        and py_ver in f.name
        and f.name.endswith(".whl")
    ]

    if not wheel_files:
        # Try finding any wheel for this Python version (for platforms that don't include type in name)
        wheel_files = [
            f
            for f in dist_dir.iterdir()
            if py_ver in f.name and f.name.endswith(".whl")
        ]

    if not wheel_files:
        print(f"[FAIL] No wheel found for {wheel_type} {py_ver}", file=sys.stderr)
        print(f"  Available wheels: {list(dist_dir.glob('*.whl'))}", file=sys.stderr)
        sys.exit(1)

    # Pick the most recently modified wheel to avoid stale wheels from prior steps
    wheel_file = max(wheel_files, key=lambda f: f.stat().st_mtime)
    print(f"Found wheel: {wheel_file.name}")

    # Choose test method based on platform and wheel type
    if sys.platform == "linux" and "manylinux" in wheel_file.name:
        # Test in container for manylinux wheels to ensure compatibility
        returncode = test_in_container(wheel_file, wheel_type, py_ver)
    else:
        # For macOS/Windows or non-manylinux wheels, test locally with uv
        returncode = test_locally_with_uv(wheel_file, wheel_type, py_ver)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
