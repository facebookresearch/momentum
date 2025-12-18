# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test wheel imports in a clean virtual environment using uv."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    # Determine Python version and wheel type
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME", "")
    wheel_type = "gpu" if "cuda" in env_name else "cpu"

    # Find the wheel file
    dist_dir = Path("dist")
    wheel_files = [
        f
        for f in dist_dir.iterdir()
        if f.name.startswith(f"pymomentum_{wheel_type}")
        and py_ver in f.name
        and f.name.endswith(".whl")
    ]

    if not wheel_files:
        print(f"✗ No wheel found for {wheel_type} {py_ver}", file=sys.stderr)
        sys.exit(1)

    wheel_file = wheel_files[0]
    print(f"Testing wheel: {wheel_file.name}")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        venv_path = tmpdir_path / ".venv"

        # Prepare environment
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)

        # Create virtual environment
        subprocess.run(
            ["uv", "venv", "--python", sys.executable],
            cwd=tmpdir,
            env=env,
            check=True,
        )

        # For GPU wheels, install torch from the CUDA index first
        # The wheel metadata specifies torch>=2.8.0,<2.9 but without index URL
        # so we need to pre-install CUDA-enabled torch before the wheel
        if wheel_type == "gpu":
            print("Installing CUDA-enabled PyTorch for GPU wheel testing...")
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "torch>=2.8.0,<2.9",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
                cwd=tmpdir,
                env=env,
                check=True,
            )

        # Install the wheel with dependencies
        # For GPU wheels, torch is already installed from CUDA index above
        # For CPU wheels, this will install torch from PyPI (CPU version)
        subprocess.run(
            ["uv", "pip", "install", str(wheel_file.absolute())],
            cwd=tmpdir,
            env=env,
            check=True,
        )

        # Test imports
        test_script = """
import sys
try:
    import pymomentum.geometry as geom
    import pymomentum.solver as solver
    import pymomentum.solver2 as solver2
    import pymomentum.marker_tracking as marker_tracking
    import pymomentum.axel as axel
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}", file=sys.stderr)
    sys.exit(1)
"""

        result = subprocess.run(
            ["uv", "run", "--no-project", "python", "-c", test_script],
            cwd=tmpdir,
            env=env,
        )

        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
