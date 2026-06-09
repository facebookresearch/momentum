#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build a variant-specific PyPI wheel."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


VARIANT_ALIASES = {
    "base": "core",
    "torch_cpu": "cpu",
    "torch_gpu": "gpu",
}
TORCH_INDEX_BY_VARIANT = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "gpu": "https://download.pytorch.org/whl/cu129",
}
TORCH_REQUIREMENT_BY_PY_VERSION = {
    "312": "torch>=2.8.0,<2.9",
    "313": "torch>=2.8.0,<2.9",
}


def normalize_variant(value: str) -> str:
    return VARIANT_ALIASES.get(value, value)


def run(args: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.check_call(args, env=env)


def torch_cmake_prefix_path() -> str:
    return subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.utils.cmake_prefix_path)",
        ],
        text=True,
    ).strip()


def main() -> None:
    variant = normalize_variant(os.environ.get("PYMOMENTUM_VARIANT", "core"))
    if variant not in {"core", "cpu", "gpu"}:
        raise SystemExit(f"Unknown PYMOMENTUM_VARIANT: {variant}")

    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    source = Path(f"pyproject-pypi-{variant}-py{py_ver}.toml")
    if not source.exists():
        raise SystemExit(f"{source} not found; run generate_pyproject first")

    backup_dir = Path(".tmp")
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / "pyproject-backup.toml"
    shutil.copy("pyproject.toml", backup)

    env = os.environ.copy()
    try:
        print(f"Using {source} for {variant} build", flush=True)
        shutil.copy(source, "pyproject.toml")

        if variant != "core":
            torch_requirement = TORCH_REQUIREMENT_BY_PY_VERSION.get(py_ver)
            if torch_requirement is None:
                raise SystemExit(
                    f"Unsupported Python version for torch wheel: {py_ver}"
                )

            torch_index = TORCH_INDEX_BY_VARIANT[variant]
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    torch_requirement,
                    "--index-url",
                    torch_index,
                ]
            )

            torch_prefix = torch_cmake_prefix_path()
            existing_prefix = env.get("CMAKE_PREFIX_PATH")
            env["CMAKE_PREFIX_PATH"] = (
                torch_prefix
                if not existing_prefix
                else os.pathsep.join([torch_prefix, existing_prefix])
            )
            print(
                f"Prepended PyPI torch CMake prefix: {torch_prefix}",
                flush=True,
            )

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir=dist",
            ],
            env=env,
        )
    finally:
        if backup.exists():
            shutil.move(backup, "pyproject.toml")


if __name__ == "__main__":
    main()
