#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate PyPI pyproject.toml files from the shared template.

PyTorch version defaults can be customized via command-line arguments.
For Python 3.14+, add new arguments following the pattern:
  --torch-min-py314 X.X.X --torch-max-py314 X.X
  --torch-min-py314-macos X.X.X --torch-max-py314-macos X.X
Then update SUPPORTED_PYTHON_TAGS and PYTHON_REQUIRES_MAX below.
"""

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


SUPPORTED_PYTHON_TAGS = ["312", "313"]
PYTHON_REQUIRES_MIN = "3.12"
PYTHON_REQUIRES_MAX = "3.14"


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
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=select_autoescape(),
    )
    template = env.get_template("pyproject-pypi.toml.j2")

    # Common template variables
    common_vars = {
        "torch_min_py312": args.torch_min_py312,
        "torch_max_py312": args.torch_max_py312,
        "torch_min_py313": args.torch_min_py313,
        "torch_max_py313": args.torch_max_py313,
        "torch_min_py312_macos": args.torch_min_py312_macos,
        "torch_max_py312_macos": args.torch_max_py312_macos,
        "torch_min_py313_macos": args.torch_min_py313_macos,
        "torch_max_py313_macos": args.torch_max_py313_macos,
    }

    variants = [
        {
            "tag": "core",
            "distribution_name": "pymomentum-core",
            "description_suffix": "core package without Torch C++ extensions",
            "build_torch_extensions": False,
            "wheel_libs_dir": "pymomentum_core.libs",
        },
        {
            "tag": "cpu",
            "distribution_name": "pymomentum-cpu",
            "description_suffix": "Torch C++ extension package linked against CPU PyTorch",
            "build_torch_extensions": True,
            "wheel_libs_dir": "pymomentum_cpu.libs",
        },
        {
            "tag": "gpu",
            "distribution_name": "pymomentum-gpu",
            "description_suffix": "Torch C++ extension package linked against CUDA PyTorch",
            "build_torch_extensions": True,
            "wheel_libs_dir": "pymomentum_gpu.libs",
        },
    ]

    for variant in variants:
        for py_ver in SUPPORTED_PYTHON_TAGS:
            config = template.render(
                python_requires_min=PYTHON_REQUIRES_MIN,
                python_requires_max=PYTHON_REQUIRES_MAX,
                **variant,
                **common_vars,
            )
            output_path = (
                output_dir / f"pyproject-pypi-{variant['tag']}-py{py_ver}.toml"
            )
            output_path.write_text(config)
            print(
                f"Generated {output_path.name} "
                f"(requires-python: >={PYTHON_REQUIRES_MIN},<{PYTHON_REQUIRES_MAX})"
            )


if __name__ == "__main__":
    main()
