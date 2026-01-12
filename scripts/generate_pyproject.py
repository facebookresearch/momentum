#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate pyproject-pypi-{cpu,gpu}.toml from template.

PyTorch version defaults can be customized via command-line arguments.
For Python 3.14+, add new arguments following the pattern:
  --torch-min-py314 X.X.X --torch-max-py314 X.X
  --torch-min-py314-macos X.X.X --torch-max-py314-macos X.X
"""

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyPI pyproject.toml files from template",
        epilog="NOTE: For future Python versions (3.14+), add new arguments following the pattern above.",
    )
    # Linux/Windows PyTorch versions (support latest with CUDA 12.9)
    parser.add_argument(
        "--torch-min-py312",
        default="2.8.0",
        help="Minimum PyTorch version for Python 3.12 (Linux/Windows): >=X.X.X",
    )
    parser.add_argument(
        "--torch-max-py312",
        default="2.9",
        help="Maximum PyTorch version for Python 3.12 (Linux/Windows): <X.X (exclusive)",
    )
    parser.add_argument(
        "--torch-min-py313",
        default="2.8.0",
        help="Minimum PyTorch version for Python 3.13 (Linux/Windows): >=X.X.X",
    )
    parser.add_argument(
        "--torch-max-py313",
        default="2.9",
        help="Maximum PyTorch version for Python 3.13 (Linux/Windows): <X.X (exclusive)",
    )
    # macOS PyTorch versions (now matches Linux/Windows as PyTorch 2.8+ is available on PyPI for macOS ARM)
    parser.add_argument(
        "--torch-min-py312-macos",
        default="2.8.0",
        help="Minimum PyTorch version for Python 3.12 (macOS): >=X.X.X",
    )
    parser.add_argument(
        "--torch-max-py312-macos",
        default="2.9",
        help="Maximum PyTorch version for Python 3.12 (macOS): <X.X (exclusive)",
    )
    parser.add_argument(
        "--torch-min-py313-macos",
        default="2.8.0",
        help="Minimum PyTorch version for Python 3.13 (macOS): >=X.X.X",
    )
    parser.add_argument(
        "--torch-max-py313-macos",
        default="2.9",
        help="Maximum PyTorch version for Python 3.13 (macOS): <X.X (exclusive)",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    # Setup Jinja2
    template_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
    )
    template = env.get_template("pyproject-pypi.toml.j2")

    # Generate CPU configs for each Python version
    for py_ver in ["312", "313"]:
        py_ver_min = f"3.{py_ver[1:]}"
        py_ver_max = f"3.{int(py_ver[1:]) + 1}"
        cpu_config = template.render(
            variant="cpu",
            description_suffix="CPU-only version for Linux, macOS Intel, and macOS ARM",
            python_version_min=py_ver_min,
            python_version_max=py_ver_max,
            torch_min_py312=args.torch_min_py312,
            torch_max_py312=args.torch_max_py312,
            torch_min_py313=args.torch_min_py313,
            torch_max_py313=args.torch_max_py313,
            torch_min_py312_macos=args.torch_min_py312_macos,
            torch_max_py312_macos=args.torch_max_py312_macos,
            torch_min_py313_macos=args.torch_min_py313_macos,
            torch_max_py313_macos=args.torch_max_py313_macos,
        )
        (output_dir / f"pyproject-pypi-cpu-py{py_ver}.toml").write_text(cpu_config)
        print(
            f"Generated pyproject-pypi-cpu-py{py_ver}.toml (requires-python: >={py_ver_min},<{py_ver_max})"
        )

    # Generate GPU config (supports Python 3.12+)
    gpu_config = template.render(
        variant="gpu",
        description_suffix="GPU-accelerated version with CUDA support",
        python_version_min="3.12",
        python_version_max="3.14",
        torch_min_py312=args.torch_min_py312,
        torch_max_py312=args.torch_max_py312,
        torch_min_py313=args.torch_min_py313,
        torch_max_py313=args.torch_max_py313,
    )
    (output_dir / "pyproject-pypi-gpu.toml").write_text(gpu_config)
    print(
        f"Generated pyproject-pypi-gpu.toml (requires-python: >=3.12,<3.14; Py3.12: >={args.torch_min_py312},<{args.torch_max_py312}; Py3.13: >={args.torch_min_py313},<{args.torch_max_py313})"
    )


if __name__ == "__main__":
    main()
