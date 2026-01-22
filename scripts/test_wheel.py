# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test wheel imports in a clean virtual environment.

For Linux manylinux wheels: tests in a manylinux_2_28 container using podman.
For macOS/Windows wheels: tests locally using uv to create a clean venv.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def test_locally(wheel_file: Path, wheel_type: str, py_ver: str) -> int:
    """Test wheel locally in a clean virtual environment using uv."""
    print(f"Testing wheel locally: {wheel_file.name}")

    # Torch index URL based on wheel type
    if wheel_type == "gpu":
        torch_index = "https://download.pytorch.org/whl/cu128"
    else:
        torch_index = "https://download.pytorch.org/whl/cpu"

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory(prefix="pymomentum_test_") as tmpdir:
        venv_path = Path(tmpdir) / "test_venv"

        try:
            # Check if uv is available
            uv_cmd = shutil.which("uv")
            if not uv_cmd:
                print("✗ uv not found. Installing uv...", file=sys.stderr)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "uv"],
                    check=True,
                    capture_output=True,
                )
                uv_cmd = "uv"

            print(f"Creating virtual environment at {venv_path}...")
            subprocess.run(
                [uv_cmd, "venv", str(venv_path), "--python", sys.executable],
                check=True,
                capture_output=True,
            )

            # Determine the python executable in the venv
            if sys.platform == "win32":
                venv_python = venv_path / "Scripts" / "python.exe"
            else:
                venv_python = venv_path / "bin" / "python"

            print(f"Installing torch from {torch_index}...")
            subprocess.run(
                [
                    uv_cmd,
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    "torch>=2.8.0,<2.9",
                    "--index-url",
                    torch_index,
                ],
                check=True,
                capture_output=True,
            )

            print(f"Installing wheel: {wheel_file.name}...")
            subprocess.run(
                [
                    uv_cmd,
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    str(wheel_file.absolute()),
                ],
                check=True,
                capture_output=True,
            )

            print("Testing imports...")
            # Dynamically discover and test all pymomentum submodules
            test_script = """
import sys
import pkgutil
import importlib

try:
    import pymomentum
    print(f'pymomentum version: {getattr(pymomentum, "__version__", "unknown")}')

    # Dynamically discover all submodules
    failed = []
    succeeded = []
    for importer, modname, ispkg in pkgutil.iter_modules(pymomentum.__path__):
        full_name = f'pymomentum.{modname}'
        try:
            importlib.import_module(full_name)
            succeeded.append(modname)
        except Exception as e:
            failed.append((modname, str(e)))

    print(f'Successfully imported: {", ".join(succeeded)}')

    if failed:
        print(f'Failed imports:', file=sys.stderr)
        for modname, err in failed:
            print(f'  - {modname}: {err}', file=sys.stderr)
        sys.exit(1)

    print(f'All {len(succeeded)} submodules imported successfully')
except Exception as e:
    print(f'Import failed: {e}', file=sys.stderr)
    sys.exit(1)
"""
            result = subprocess.run(
                [str(venv_python), "-c", test_script],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"✗ Import test failed: {result.stderr}", file=sys.stderr)
                return 1

            print("✓ " + result.stdout.strip())
            return 0

        except subprocess.CalledProcessError as e:
            print(f"✗ Test failed: {e}", file=sys.stderr)
            if e.stderr:
                print(
                    f"  stderr: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}",
                    file=sys.stderr,
                )
            return 1


def test_in_container(wheel_file: Path, wheel_type: str, py_ver: str) -> int:
    """Test wheel in a manylinux_2_28 container using podman."""
    print(f"Testing wheel in manylinux_2_28 container: {wheel_file.name}")

    # Torch index URL based on wheel type
    if wheel_type == "gpu":
        torch_index = "https://download.pytorch.org/whl/cu128"
    else:
        torch_index = "https://download.pytorch.org/whl/cpu"

    # Container command to test the wheel
    # py_ver is like "cp312" or "cp313", so py_ver[2:] gives "312" or "313"
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
import pkgutil
import importlib

try:
    import pymomentum
    print(f'pymomentum version: {{getattr(pymomentum, \"__version__\", \"unknown\")}}')

    # Dynamically discover all submodules
    failed = []
    succeeded = []
    for importer, modname, ispkg in pkgutil.iter_modules(pymomentum.__path__):
        full_name = f'pymomentum.{{modname}}'
        try:
            importlib.import_module(full_name)
            succeeded.append(modname)
        except Exception as e:
            failed.append((modname, str(e)))

    print(f'Successfully imported: {{\", \".join(succeeded)}}')

    if failed:
        print(f'Failed imports:', file=sys.stderr)
        for modname, err in failed:
            print(f'  - {{modname}}: {{err}}', file=sys.stderr)
        sys.exit(1)

    print(f'All {{len(succeeded)}} submodules imported successfully')
except Exception as e:
    print(f'Import failed: {{e}}', file=sys.stderr)
    sys.exit(1)
"
"""

    # Run the test in a manylinux_2_28 container
    result = subprocess.run(
        [
            "podman",
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

    # Test in container for Linux manylinux wheels
    if sys.platform == "linux" and "manylinux" in wheel_file.name:
        returncode = test_in_container(wheel_file, wheel_type, py_ver)
    else:
        # For non-manylinux wheels (macOS, Windows), test locally with uv
        returncode = test_locally(wheel_file, wheel_type, py_ver)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
