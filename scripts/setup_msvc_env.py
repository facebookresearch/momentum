#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Set up MSVC environment variables for Ninja + CUDA builds on Windows.

When using the Ninja CMake generator with CUDA on Windows inside a pixi/conda
environment, the PATH may not include MSVC tools (cl.exe, link.exe, etc.).
This script finds vcvarsall.bat, runs it, captures the environment changes,
and applies them to the current process before executing the given command.

Usage:
    python scripts/setup_msvc_env.py <command> [args...]

Example:
    python scripts/setup_msvc_env.py pip wheel . --no-deps --wheel-dir=dist
"""

import os
import subprocess
import sys
from pathlib import Path


def find_vcvarsall() -> Path:
    """Find vcvarsall.bat using vswhere.exe."""
    vswhere = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not vswhere.exists():
        raise FileNotFoundError(f"vswhere.exe not found at {vswhere}")

    result = subprocess.run(
        [str(vswhere), "-latest", "-property", "installationPath"],
        capture_output=True,
        text=True,
        check=True,
    )
    vs_path = Path(result.stdout.strip())
    vcvarsall = vs_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.exists():
        raise FileNotFoundError(f"vcvarsall.bat not found at {vcvarsall}")
    return vcvarsall


def get_msvc_env(vcvarsall: Path, arch: str = "amd64") -> dict:
    """Run vcvarsall.bat and capture the resulting environment variables."""
    # Run vcvarsall.bat and print the environment
    cmd = f'"{vcvarsall}" {arch} && set'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return env


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <command> [args...]", file=sys.stderr)
        sys.exit(1)

    if sys.platform != "win32":
        # On non-Windows, just exec the command directly
        os.execvp(sys.argv[1], sys.argv[1:])

    print("=== Setting up MSVC environment for Ninja + CUDA build ===")

    # Find vcvarsall.bat
    vcvarsall = find_vcvarsall()
    print(f"Found vcvarsall.bat: {vcvarsall}")

    # Get MSVC environment
    msvc_env = get_msvc_env(vcvarsall)

    # Merge MSVC environment into current environment
    # Key variables to merge: PATH, INCLUDE, LIB, LIBPATH, etc.
    merge_vars = [
        "PATH",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "WindowsSdkDir",
        "WindowsSdkVersion",
        "VCToolsInstallDir",
        "VCToolsVersion",
        "VSCMD_ARG_TGT_ARCH",
    ]

    for var in merge_vars:
        if var in msvc_env:
            if var == "PATH":
                # Prepend MSVC paths to existing PATH (preserve conda paths too)
                current_path = os.environ.get("PATH", "")
                msvc_path = msvc_env["PATH"]
                # Extract paths from MSVC env that aren't already in current PATH
                current_paths = set(current_path.lower().split(os.pathsep))
                new_paths = [
                    p
                    for p in msvc_path.split(os.pathsep)
                    if p.lower() not in current_paths
                ]
                if new_paths:
                    os.environ["PATH"] = (
                        os.pathsep.join(new_paths) + os.pathsep + current_path
                    )
                    print(f"Added {len(new_paths)} MSVC paths to PATH")
            else:
                os.environ[var] = msvc_env[var]
                print(f"Set {var}")

    # Verify cl.exe is accessible
    try:
        result = subprocess.run(
            ["where", "cl.exe"], capture_output=True, text=True, check=True
        )
        print(f"cl.exe found: {result.stdout.strip().splitlines()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: cl.exe not found on PATH after MSVC setup", file=sys.stderr)

    print("=== MSVC environment ready ===")
    print(f"Executing: {' '.join(sys.argv[1:])}")

    # Execute the requested command
    result = subprocess.run(sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
