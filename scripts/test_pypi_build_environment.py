#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke-test PyPI Torch against native packages from the Pixi environment."""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import build_pypi_wheel


def run(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.check_call(args)


def venv_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def import_script(first_import: str, second_import: str) -> str:
    return f"""
{first_import}
{second_import}
import sys
from pathlib import Path

venv_root = Path(sys.prefix).resolve()
torch_path = Path(torch.__file__).resolve()
numpy_path = Path(np.__file__).resolve()
assert torch_path.is_relative_to(venv_root), (torch_path, venv_root)
assert not numpy_path.is_relative_to(venv_root), (numpy_path, venv_root)
print(f"NumPy {{np.__version__}}: {{numpy_path}}")
print(f"PyTorch {{torch.__version__}}: {{torch_path}}")
print(f"PyTorch CMake prefix: {{torch.utils.cmake_prefix_path}}")
"""


def main() -> None:
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    torch_requirement = build_pypi_wheel.TORCH_REQUIREMENT_BY_PY_VERSION.get(py_version)
    if torch_requirement is None:
        raise SystemExit(f"Unsupported Python version: {py_version}")

    with tempfile.TemporaryDirectory(prefix="pymomentum-torch-smoke-") as temp:
        environment = Path(temp)
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = str(venv_python(environment))

        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                torch_requirement,
                "--index-url",
                build_pypi_wheel.TORCH_INDEX_BY_VARIANT["cpu"],
            ]
        )

        for first_import, second_import in (
            ("import numpy as np", "import torch"),
            ("import torch", "import numpy as np"),
        ):
            run([python, "-c", import_script(first_import, second_import)])


if __name__ == "__main__":
    main()
