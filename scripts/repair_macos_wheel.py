#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Repair macOS PyPI wheels with delocate."""

import glob
import os
import shutil
import subprocess
from pathlib import Path


VARIANT_ALIASES = {
    "base": "core",
    "torch_cpu": "cpu",
    "torch_gpu": "gpu",
}
WHEEL_PATTERNS = {
    "core": "dist/pymomentum_core-*.whl",
    "cpu": "dist/pymomentum_cpu-*.whl",
    "gpu": "dist/pymomentum_gpu-*.whl",
}
PYTORCH_DYLIB_EXCLUDES = [
    "libtorch",
    "libc10",
    "libshm",
    "libomp",
]


def normalize_variant(value: str) -> str:
    return VARIANT_ALIASES.get(value, value)


def main() -> None:
    variant = normalize_variant(os.environ.get("PYMOMENTUM_VARIANT", "core"))
    if variant not in WHEEL_PATTERNS:
        raise SystemExit(f"Unknown PYMOMENTUM_VARIANT: {variant}")

    wheels = glob.glob(WHEEL_PATTERNS[variant])
    print(f"Repairing macOS {variant} wheels with delocate: {wheels}", flush=True)
    if not wheels:
        raise SystemExit(f"No {variant} wheels found")

    out_dir = Path("dist/repaired")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_path = str(Path(conda_prefix) / "lib")
        for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
            env[key] = lib_path + (os.pathsep + env[key] if env.get(key) else "")

    command = [
        "delocate-wheel",
        "--ignore-missing-dependencies",
        "-w",
        str(out_dir),
        "-v",
    ]
    if variant != "core":
        for excluded in PYTORCH_DYLIB_EXCLUDES:
            command.extend(["--exclude", excluded])
    command.extend(wheels)
    print("+ " + " ".join(command), flush=True)
    subprocess.check_call(command, env=env)

    for wheel in wheels:
        os.remove(wheel)
    for repaired in out_dir.glob("*.whl"):
        shutil.move(str(repaired), Path("dist") / repaired.name)
    shutil.rmtree(out_dir)


if __name__ == "__main__":
    main()
