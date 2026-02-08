#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Set up MSVC environment variables for Ninja + CUDA builds on Windows.

When using the Ninja CMake generator with CUDA on Windows inside a pixi/conda
environment, the PATH may not include MSVC tools (cl.exe, link.exe, etc.) and
the INCLUDE/LIB environment variables may not include Windows SDK paths.

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
    print(f"Visual Studio installation: {vs_path}")

    vcvarsall = vs_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.exists():
        raise FileNotFoundError(f"vcvarsall.bat not found at {vcvarsall}")
    return vcvarsall


def get_msvc_env(vcvarsall: Path, arch: str = "amd64") -> dict:
    """Run vcvarsall.bat and capture the resulting environment variables."""
    # Run vcvarsall.bat and print the environment
    # Use a unique separator to avoid confusion with normal output
    separator = "===MSVC_ENV_START==="
    cmd = f'"{vcvarsall}" {arch} >nul 2>&1 && echo {separator} && set'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"vcvarsall.bat stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"vcvarsall.bat failed with exit code {result.returncode}")

    # Parse environment variables after the separator
    env = {}
    found_separator = False
    for line in result.stdout.splitlines():
        if separator in line:
            found_separator = True
            continue
        if found_separator and "=" in line:
            key, _, value = line.partition("=")
            env[key] = value

    if not found_separator:
        # Fallback: try parsing all lines as env vars
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
    print(f"Python: {sys.executable}")
    print(f"Platform: {sys.platform}")

    # Find vcvarsall.bat
    try:
        vcvarsall = find_vcvarsall()
        print(f"Found vcvarsall.bat: {vcvarsall}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Falling back to running command without MSVC setup",
            file=sys.stderr,
        )
        result = subprocess.run(sys.argv[1:])
        sys.exit(result.returncode)

    # Get MSVC environment
    try:
        msvc_env = get_msvc_env(vcvarsall)
        print(f"Captured {len(msvc_env)} environment variables from vcvarsall.bat")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Falling back to running command without MSVC setup",
            file=sys.stderr,
        )
        result = subprocess.run(sys.argv[1:])
        sys.exit(result.returncode)

    # Merge MSVC environment into current environment
    # Key variables that Ninja + CUDA need from MSVC:
    # - PATH: for cl.exe, link.exe, lib.exe, rc.exe
    # - INCLUDE: for Windows SDK headers and MSVC headers
    # - LIB: for Windows SDK libs and MSVC libs
    # - LIBPATH: for .NET framework libraries
    merge_vars = {
        "PATH": "prepend",  # Prepend MSVC paths to existing PATH
        "INCLUDE": "replace",  # Replace with MSVC INCLUDE (conda doesn't set this)
        "LIB": "replace",  # Replace with MSVC LIB
        "LIBPATH": "replace",  # Replace with MSVC LIBPATH
        "WindowsSdkDir": "replace",
        "WindowsSdkVersion": "replace",
        "VCToolsInstallDir": "replace",
        "VCToolsVersion": "replace",
        "VSCMD_ARG_TGT_ARCH": "replace",
        "UCRTVersion": "replace",
        "UniversalCRTSdkDir": "replace",
    }

    for var, mode in merge_vars.items():
        if var not in msvc_env:
            continue

        if mode == "prepend" and var == "PATH":
            # Prepend MSVC paths to existing PATH (preserve conda paths too)
            current_path = os.environ.get("PATH", "")
            msvc_path = msvc_env["PATH"]
            # Extract paths from MSVC env that aren't already in current PATH
            current_paths_lower = {
                p.lower().rstrip("\\") for p in current_path.split(os.pathsep) if p
            }
            new_paths = [
                p
                for p in msvc_path.split(os.pathsep)
                if p and p.lower().rstrip("\\") not in current_paths_lower
            ]
            if new_paths:
                os.environ["PATH"] = (
                    os.pathsep.join(new_paths) + os.pathsep + current_path
                )
                print(f"Prepended {len(new_paths)} MSVC paths to PATH")
        else:
            os.environ[var] = msvc_env[var]
            # Print truncated values for readability
            val = msvc_env[var]
            if len(val) > 100:
                val = val[:100] + "..."
            print(f"Set {var}={val}")

    # Verify critical tools are accessible
    for tool in ["cl.exe", "link.exe", "rc.exe"]:
        try:
            result = subprocess.run(
                ["where", tool], capture_output=True, text=True, check=True
            )
            print(f"{tool} found: {result.stdout.strip().splitlines()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"WARNING: {tool} not found on PATH", file=sys.stderr)

    # Check if nvcc is available
    try:
        result = subprocess.run(
            ["where", "nvcc"], capture_output=True, text=True, check=True
        )
        print(f"nvcc found: {result.stdout.strip().splitlines()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: nvcc not found on PATH", file=sys.stderr)

    # Check if ninja is available
    try:
        result = subprocess.run(
            ["where", "ninja"], capture_output=True, text=True, check=True
        )
        print(f"ninja found: {result.stdout.strip().splitlines()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: ninja not found on PATH", file=sys.stderr)

    # Print CMAKE_CUDA_HOST_COMPILER if set
    cuda_host = os.environ.get("CMAKE_CUDA_HOST_COMPILER", "")
    if cuda_host:
        print(f"CMAKE_CUDA_HOST_COMPILER={cuda_host}")

    print("=== MSVC environment ready ===")
    print(f"Executing: {' '.join(sys.argv[1:])}")
    sys.stdout.flush()

    # Execute the requested command
    result = subprocess.run(sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
